"""Tests for ``turboedge state pack/unpack`` and ``turboedge db compact``.

No test here touches the network. Config/state dir fixtures mirror
``tests/test_cli.py`` (copied rather than imported -- this repo has no
``tests/__init__.py``, so cross-module fixture imports aren't available;
see the note at the top of ``tests/test_cli.py``).
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import pytest
from typer.testing import CliRunner

from turboedge.cli import app
from turboedge.config import CONFIG_FILES
from turboedge.reporting.redaction import redact_console_enabled

runner = CliRunner()

_PASSPHRASE = secrets.token_urlsafe(32)


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    project_root = Path(__file__).parent.parent
    for config_file in CONFIG_FILES:
        src = project_root / "configs" / config_file
        if src.exists():
            (config_dir / config_file).write_text(src.read_text())
    return config_dir


@pytest.fixture
def tmp_state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


def _base_args(config_dir: Path, state_dir: Path) -> list[str]:
    return ["--config-dir", str(config_dir), "--state-dir", str(state_dir)]


@pytest.fixture(autouse=True)
def _no_leftover_state_key_env():  # type: ignore[no-untyped-def]
    """Guard against TURBOEDGE_STATE_KEY leaking in from the outer shell
    environment into a test that expects it unset."""
    original = os.environ.pop("TURBOEDGE_STATE_KEY", None)
    yield
    if original is not None:
        os.environ["TURBOEDGE_STATE_KEY"] = original
    else:
        os.environ.pop("TURBOEDGE_STATE_KEY", None)


# -- state pack -------------------------------------------------------------


def test_state_pack_missing_key_exits_2(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    result = runner.invoke(
        app,
        [*_base_args(tmp_config_dir, tmp_state_dir), "state", "pack", "--out", "/tmp/x.tar.enc"],
    )
    assert result.exit_code == 2
    assert "TURBOEDGE_STATE_KEY" in result.stderr


def test_state_pack_success(tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "archive.tar.enc"
    result = runner.invoke(
        app,
        [*_base_args(tmp_config_dir, tmp_state_dir), "state", "pack", "--out", str(out_path)],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert result.exit_code == 0, result.output
    assert out_path.is_file()
    assert "Packed" in result.stdout


# -- state unpack -------------------------------------------------------------


def test_state_unpack_missing_key_exits_2(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    result = runner.invoke(
        app,
        [*_base_args(tmp_config_dir, tmp_state_dir), "state", "unpack", "--in", "/tmp/x.tar.enc"],
    )
    assert result.exit_code == 2
    assert "TURBOEDGE_STATE_KEY" in result.stderr


def test_state_unpack_missing_archive_without_allow_missing_exits_2(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "does_not_exist.tar.enc"
    result = runner.invoke(
        app,
        [*_base_args(tmp_config_dir, tmp_state_dir), "state", "unpack", "--in", str(missing)],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert result.exit_code == 2


def test_state_unpack_missing_archive_with_allow_missing_exits_0(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "does_not_exist.tar.enc"
    result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "state",
            "unpack",
            "--in",
            str(missing),
            "--allow-missing",
        ],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert result.exit_code == 0, result.output
    # Rich wraps long lines, so match tolerantly rather than on one exact
    # phrase spanning a possible line break.
    assert "no state archive found" in result.stdout.lower()
    assert "fresh" in result.stdout.lower()


def test_state_pack_unpack_roundtrip_via_cli(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path
) -> None:
    # Create some state first (db info creates+inits the DuckDB file).
    init_result = runner.invoke(app, [*_base_args(tmp_config_dir, tmp_state_dir), "db", "info"])
    assert init_result.exit_code == 0, init_result.output

    archive_path = tmp_path / "archive.tar.enc"
    pack_result = runner.invoke(
        app,
        [*_base_args(tmp_config_dir, tmp_state_dir), "state", "pack", "--out", str(archive_path)],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert pack_result.exit_code == 0, pack_result.output
    assert archive_path.is_file()

    restore_dir = tmp_path / "restored_state"
    unpack_result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, restore_dir),
            "state",
            "unpack",
            "--in",
            str(archive_path),
        ],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert unpack_result.exit_code == 0, unpack_result.output
    assert (restore_dir / "turboedge.duckdb").is_file()


def test_state_unpack_wrong_key_exits_2(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path
) -> None:
    runner.invoke(app, [*_base_args(tmp_config_dir, tmp_state_dir), "db", "info"])
    archive_path = tmp_path / "archive.tar.enc"
    pack_result = runner.invoke(
        app,
        [*_base_args(tmp_config_dir, tmp_state_dir), "state", "pack", "--out", str(archive_path)],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert pack_result.exit_code == 0, pack_result.output

    restore_dir = tmp_path / "restored_state_wrong_key"
    unpack_result = runner.invoke(
        app,
        [*_base_args(tmp_config_dir, restore_dir), "state", "unpack", "--in", str(archive_path)],
        env={"TURBOEDGE_STATE_KEY": secrets.token_urlsafe(32)},  # different key
    )
    assert unpack_result.exit_code == 2
    assert "wrong" in unpack_result.stderr.lower() or "failed" in unpack_result.stderr.lower()


# -- db compact -------------------------------------------------------------


def test_db_compact_runs_on_fresh_db(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    result = runner.invoke(
        app, [*_base_args(tmp_config_dir, tmp_state_dir), "db", "compact", "--keep-days", "45"]
    )
    assert result.exit_code == 0, result.output
    assert "product_snapshots rows before" in result.stdout
    assert "product_snapshots rows after" in result.stdout


def test_db_compact_default_keep_days(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    result = runner.invoke(app, [*_base_args(tmp_config_dir, tmp_state_dir), "db", "compact"])
    assert result.exit_code == 0, result.output
    assert "keep_days" in result.stdout


def test_db_info_still_works_alongside_db_compact(
    tmp_config_dir: Path, tmp_state_dir: Path
) -> None:
    """Registering `db compact` must not disturb the pre-existing `db info`
    command."""
    result = runner.invoke(app, [*_base_args(tmp_config_dir, tmp_state_dir), "db", "info"])
    assert result.exit_code == 0
    assert "product_snapshots" in result.stdout


# -- reporting.redaction -----------------------------------------------------


def test_redact_console_enabled_delegates_to_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TURBOEDGE_PUBLIC_LOGS", "1")
    assert redact_console_enabled() is True

    monkeypatch.setenv("TURBOEDGE_PUBLIC_LOGS", "0")
    assert redact_console_enabled() is False

    monkeypatch.delenv("TURBOEDGE_PUBLIC_LOGS", raising=False)
    assert redact_console_enabled() is False
