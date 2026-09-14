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


def test_state_pack_default_excludes_snapshots(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path
) -> None:
    """`state pack`'s default -- no --include-snapshots -- must not pack
    state/snapshots/ (the immutable Parquet archive): this is the lean
    package every scan/eod/monthly pipeline run uploads several times a
    day (see state/archive.py LEAN_INCLUDE_DIRS)."""
    snap_dir = tmp_state_dir / "snapshots" / "product_snapshots" / "date=2026-09-10"
    snap_dir.mkdir(parents=True)
    (snap_dir / "run1.parquet").write_bytes(b"fake-parquet-bytes")

    out_path = tmp_path / "lean.tar.enc"
    result = runner.invoke(
        app,
        [*_base_args(tmp_config_dir, tmp_state_dir), "state", "pack", "--out", str(out_path)],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert result.exit_code == 0, result.output
    assert "lean" in result.stdout.lower()

    restore_dir = tmp_path / "restored_lean"
    unpack_result = runner.invoke(
        app,
        [*_base_args(tmp_config_dir, restore_dir), "state", "unpack", "--in", str(out_path)],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert unpack_result.exit_code == 0, unpack_result.output
    assert not (restore_dir / "snapshots").exists()


def test_state_pack_include_snapshots_flag_packs_the_parquet_archive(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path
) -> None:
    snap_dir = tmp_state_dir / "snapshots" / "product_snapshots" / "date=2026-09-10"
    snap_dir.mkdir(parents=True)
    (snap_dir / "run1.parquet").write_bytes(b"fake-parquet-bytes")

    out_path = tmp_path / "full.tar.enc"
    result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "state",
            "pack",
            "--out",
            str(out_path),
            "--include-snapshots",
        ],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert result.exit_code == 0, result.output
    assert "full" in result.stdout.lower()

    restore_dir = tmp_path / "restored_full"
    unpack_result = runner.invoke(
        app,
        [*_base_args(tmp_config_dir, restore_dir), "state", "unpack", "--in", str(out_path)],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert unpack_result.exit_code == 0, unpack_result.output
    assert (
        restore_dir / "snapshots" / "product_snapshots" / "date=2026-09-10" / "run1.parquet"
    ).read_bytes() == b"fake-parquet-bytes"


# -- state pack-snapshots (incremental per-scan Parquet artifact) -----------


def test_state_pack_snapshots_missing_key_exits_2(
    tmp_config_dir: Path, tmp_state_dir: Path
) -> None:
    result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "state",
            "pack-snapshots",
            "--run-id",
            "run1",
            "--out",
            "/tmp/x.tar.enc",
        ],
    )
    assert result.exit_code == 2
    assert "TURBOEDGE_STATE_KEY" in result.stderr


def test_state_pack_snapshots_packs_only_the_requested_run_ids(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path
) -> None:
    """Simulates the scan job's real scenario: older Parquet history from a
    prior run already sits under state/snapshots/, plus two new files this
    "run" just wrote (one per underlying). `pack-snapshots` with only the
    two new run_ids must produce an archive containing exactly those two
    files -- never the older one, and never anything from state/registry or
    state/ledger."""
    old_dir = tmp_state_dir / "snapshots" / "product_snapshots" / "date=2026-09-09"
    old_dir.mkdir(parents=True)
    (old_dir / "run-old.parquet").write_bytes(b"old-bytes")

    new_dir = tmp_state_dir / "snapshots" / "product_snapshots" / "date=2026-09-10"
    new_dir.mkdir(parents=True)
    (new_dir / "run-dax.parquet").write_bytes(b"dax-bytes")
    (new_dir / "run-ndx.parquet").write_bytes(b"ndx-bytes")

    (tmp_state_dir / "registry").mkdir()
    (tmp_state_dir / "registry" / "models.json").write_text('{"champion": "tsmom"}')

    out_path = tmp_path / "snapshot-batch.tar.enc"
    result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "state",
            "pack-snapshots",
            "--run-id",
            "run-dax",
            "--run-id",
            "run-ndx",
            "--out",
            str(out_path),
        ],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert result.exit_code == 0, result.output
    assert "Packed 2 snapshot file(s)" in result.stdout
    assert out_path.is_file()

    restore_dir = tmp_path / "restored"
    unpack_result = runner.invoke(
        app,
        [*_base_args(tmp_config_dir, restore_dir), "state", "unpack", "--in", str(out_path)],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert unpack_result.exit_code == 0, unpack_result.output
    assert (
        restore_dir / "snapshots" / "product_snapshots" / "date=2026-09-10" / "run-dax.parquet"
    ).read_bytes() == b"dax-bytes"
    assert (
        restore_dir / "snapshots" / "product_snapshots" / "date=2026-09-10" / "run-ndx.parquet"
    ).read_bytes() == b"ndx-bytes"
    # Neither the older run's file nor anything outside snapshots/ was pulled in.
    assert not (
        restore_dir / "snapshots" / "product_snapshots" / "date=2026-09-09" / "run-old.parquet"
    ).exists()
    assert not (restore_dir / "registry").exists()


def test_state_pack_snapshots_no_matching_run_ids_writes_nothing(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path
) -> None:
    """The scan job's workflow step relies on --out NOT being created when
    there is nothing to pack (it gates the artifact upload on the file's
    existence) -- this must exit 0 without writing --out."""
    snap_dir = tmp_state_dir / "snapshots" / "product_snapshots" / "date=2026-09-10"
    snap_dir.mkdir(parents=True)
    (snap_dir / "run-unrelated.parquet").write_bytes(b"bytes")

    out_path = tmp_path / "snapshot-batch.tar.enc"
    result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "state",
            "pack-snapshots",
            "--run-id",
            "run-never-written",
            "--out",
            str(out_path),
        ],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert result.exit_code == 0, result.output
    assert "nothing to pack" in result.stdout.lower()
    assert not out_path.exists()


def test_state_pack_snapshots_no_run_ids_given_writes_nothing(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path
) -> None:
    out_path = tmp_path / "snapshot-batch.tar.enc"
    result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "state",
            "pack-snapshots",
            "--out",
            str(out_path),
        ],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert result.exit_code == 0, result.output
    assert not out_path.exists()


def test_state_pack_snapshots_rejects_unsafe_run_id(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path
) -> None:
    out_path = tmp_path / "snapshot-batch.tar.enc"
    result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "state",
            "pack-snapshots",
            "--run-id",
            "../../etc/passwd",
            "--out",
            str(out_path),
        ],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert result.exit_code == 2
    assert "unsafe run_id" in result.stderr


# -- state restore-snapshots -------------------------------------------------


def test_state_restore_snapshots_merges_onto_lean_restore(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path
) -> None:
    snap_dir = tmp_state_dir / "snapshots" / "product_snapshots" / "date=2026-09-10"
    snap_dir.mkdir(parents=True)
    (snap_dir / "run1.parquet").write_bytes(b"fake-parquet-bytes")
    (tmp_state_dir / "registry").mkdir()
    (tmp_state_dir / "registry" / "models.json").write_text('{"champion": "tsmom"}')

    full_path = tmp_path / "full.tar.enc"
    pack_result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "state",
            "pack",
            "--out",
            str(full_path),
            "--include-snapshots",
        ],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert pack_result.exit_code == 0, pack_result.output

    lean_path = tmp_path / "lean.tar.enc"
    lean_pack_result = runner.invoke(
        app,
        [*_base_args(tmp_config_dir, tmp_state_dir), "state", "pack", "--out", str(lean_path)],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert lean_pack_result.exit_code == 0, lean_pack_result.output

    restore_dir = tmp_path / "restored"
    unpack_result = runner.invoke(
        app,
        [*_base_args(tmp_config_dir, restore_dir), "state", "unpack", "--in", str(lean_path)],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert unpack_result.exit_code == 0, unpack_result.output
    assert not (restore_dir / "snapshots").exists()

    restore_snapshots_result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, restore_dir),
            "state",
            "restore-snapshots",
            "--in",
            str(full_path),
        ],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert restore_snapshots_result.exit_code == 0, restore_snapshots_result.output
    assert (
        restore_dir / "snapshots" / "product_snapshots" / "date=2026-09-10" / "run1.parquet"
    ).read_bytes() == b"fake-parquet-bytes"
    # Untouched by the additive merge.
    assert (restore_dir / "registry" / "models.json").read_text() == '{"champion": "tsmom"}'


def test_state_restore_snapshots_missing_key_exits_2(
    tmp_config_dir: Path, tmp_state_dir: Path
) -> None:
    result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "state",
            "restore-snapshots",
            "--in",
            "/tmp/x.tar.enc",
        ],
    )
    assert result.exit_code == 2
    assert "TURBOEDGE_STATE_KEY" in result.stderr


def test_state_restore_snapshots_missing_archive_exits_2(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "does_not_exist.tar.enc"
    result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "state",
            "restore-snapshots",
            "--in",
            str(missing),
        ],
        env={"TURBOEDGE_STATE_KEY": _PASSPHRASE},
    )
    assert result.exit_code == 2


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
    assert "hard_delete_after_days" in result.stdout


def test_db_compact_hard_delete_after_days_option(
    tmp_config_dir: Path, tmp_state_dir: Path
) -> None:
    result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "db",
            "compact",
            "--keep-days",
            "45",
            "--hard-delete-after-days",
            "100",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "rows hard-deleted" in result.stdout.lower()


def test_db_compact_hard_delete_after_days_must_exceed_keep_days(
    tmp_config_dir: Path, tmp_state_dir: Path
) -> None:
    result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "db",
            "compact",
            "--keep-days",
            "45",
            "--hard-delete-after-days",
            "10",
        ],
    )
    assert result.exit_code != 0


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
