"""Tests for turboedge CLI commands.

No test in this module touches the network: ``universe``/``scan`` tests
monkeypatch ``turboedge.cli.build_product_adapters`` (and, for ``scan``,
``turboedge.cli.YFinancePriceAdapter``/``turboedge.cli.EcbEstrAdapter``) with
small in-memory fakes. These fakes are deliberately self-contained here
rather than imported from ``tests/pipeline/conftest.py``: with no
``tests/__init__.py`` in this repo, pytest's default (rootdir-relative,
non-package) import mode does not put ``tests`` on ``sys.path`` as an
importable package, so ``from tests.pipeline.conftest import ...`` fails at
collection time (verified) even though a plain ``python -c`` import of the
same dotted path succeeds. The fakes below mirror the structural protocols
``ProductSourceAdapter``/``PriceSource``/``EstrSource`` expect.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import turboedge.cli as cli_module
from turboedge.adapters.base import AdapterError, AdapterMetadata, HealthCheckResult
from turboedge.cli import app
from turboedge.storage.schemas import (
    Direction,
    HealthStatus,
    ProductSnapshot,
    ProductType,
    UnderlyingBar,
)

runner = CliRunner()

_DAX_SPOT = 24000.0


class _FakeProductAdapter:
    """Minimal in-memory ``ProductSourceAdapter`` for CLI tests (no network)."""

    def __init__(
        self,
        name: str,
        *,
        products: Sequence[ProductSnapshot] | None = None,
        exception: Exception | None = None,
    ) -> None:
        self._name = name
        self._products = list(products) if products is not None else []
        self._exception = exception

    @property
    def name(self) -> str:
        return self._name

    def fetch_products(self, underlying_ids: Sequence[str]) -> list[ProductSnapshot]:
        if self._exception is not None:
            raise self._exception
        return list(self._products)

    def healthcheck(self) -> HealthCheckResult:
        return HealthCheckResult(
            source=self._name,
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=1.0,
            checked_at=datetime.now(UTC),
            message="fake product adapter",
        )

    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(name=self._name, kind="product", version="1")


def _make_dax_product(
    *,
    isin: str,
    quote_timestamp: datetime,
    financing_level: float = 20000.0,
    ratio: float = 0.01,
    direction: Direction = Direction.LONG,
) -> ProductSnapshot:
    """A realistic, fully-priced DAX open-end turbo (bid/ask ~ intrinsic + a small premium)."""
    intrinsic = max(_DAX_SPOT - financing_level, 0.0) * ratio
    ask = round(intrinsic + 0.10, 2)
    bid = round(ask - 0.10, 2)
    return ProductSnapshot(
        isin=isin,
        wkn=isin[-6:],
        issuer="TestBank",
        venue="stuttgart",
        underlying_raw="DAX",
        underlying_id="DAX",
        direction=direction,
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=financing_level,
        knockout_barrier=financing_level,
        ratio=ratio,
        currency="EUR",
        underlying_currency="EUR",
        quanto=False,
        open_end=True,
        maturity=None,
        first_trading_day=None,
        bid=bid,
        ask=ask,
        bid_size=5000.0,
        ask_size=5000.0,
        quote_timestamp=quote_timestamp,
        quote_presence=True,
        bid_only=False,
        knocked_out=False,
        trading_hours="09:00-22:00",
        product_age_days=100,
        underlying_price_ref=None,
        raw_hash=hashlib.sha256(f"{isin}-{quote_timestamp.isoformat()}".encode()).hexdigest(),
        observation_time=quote_timestamp,
        available_at=quote_timestamp,
        retrieved_at=quote_timestamp,
        source_timestamp=quote_timestamp,
        source="fake_source",
        parser_version="1",
        is_stale=False,
        quality_score=0.9,
    )


def _make_dax_bars(*, count: int = 60, start: date = date(2025, 1, 1)) -> list[UnderlyingBar]:
    """Small deterministic run of daily DAX bars (flat-ish, Mon-Fri only)."""
    bars: list[UnderlyingBar] = []
    close = _DAX_SPOT
    d = start
    step = 0
    while len(bars) < count:
        if d.weekday() < 5:
            new_close = close * (1.0 + (0.0005 if step % 2 == 0 else -0.0003))
            ts = datetime(d.year, d.month, d.day, tzinfo=UTC)
            available_at = datetime(d.year, d.month, d.day, 22, 0, tzinfo=UTC)
            bars.append(
                UnderlyingBar(
                    underlying_id="DAX",
                    ts=ts,
                    interval="1d",
                    open=close,
                    high=max(close, new_close) * 1.001,
                    low=min(close, new_close) * 0.999,
                    close=new_close,
                    volume=1_000_000.0,
                    observation_time=available_at,
                    available_at=available_at,
                    retrieved_at=available_at,
                    source_timestamp=available_at,
                    source="fake_price",
                    parser_version="1",
                    is_stale=False,
                    quality_score=0.9,
                )
            )
            close = new_close
            step += 1
        d += timedelta(days=1)
    return bars


class _FakePriceAdapter:
    """Minimal in-memory ``PriceSource`` for CLI tests."""

    def __init__(self, bars_by_underlying: dict[str, list[UnderlyingBar]]) -> None:
        self._bars = bars_by_underlying

    def __call__(self, *_args: Any, **_kwargs: Any) -> _FakePriceAdapter:
        # Allows monkeypatching `turboedge.cli.YFinancePriceAdapter` to this
        # already-constructed instance: `cli.py` calls it as a no-arg
        # constructor (`YFinancePriceAdapter()`), so the instance itself
        # must be callable and return something usable -- itself.
        return self

    def fetch_daily_bars(
        self, underlying_id: str, *, lookback_days: int | None = None
    ) -> list[UnderlyingBar]:
        if underlying_id in self._bars:
            return list(self._bars[underlying_id])
        raise AdapterError(f"fake price fetch failure for {underlying_id!r}")

    def healthcheck(self) -> HealthCheckResult:
        return HealthCheckResult(
            source="fake_price",
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=1.0,
            checked_at=datetime.now(UTC),
            message="fake price adapter",
        )


class _FakeEstrAdapter:
    """Minimal in-memory ``EstrSource`` for CLI tests."""

    def __init__(self, rate: float = 0.03) -> None:
        self.rate = rate

    def __call__(self, *_args: Any, **_kwargs: Any) -> _FakeEstrAdapter:
        return self

    def get_estr(self) -> float:
        return self.rate

    def healthcheck(self) -> HealthCheckResult:
        return HealthCheckResult(
            source="fake_estr",
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=1.0,
            checked_at=datetime.now(UTC),
            message="fake estr adapter",
        )


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory with valid config files."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Copy real config files from the project
    project_root = Path(__file__).parent.parent
    for config_file in [
        "default.yaml",
        "sources.yaml",
        "risk.yaml",
        "universe.yaml",
        "models.yaml",
        "gmail.yaml",
        "governance.yaml",
    ]:
        src = project_root / "configs" / config_file
        if src.exists():
            dst = config_dir / config_file
            dst.write_text(src.read_text())

    return config_dir


@pytest.fixture
def tmp_state_dir(tmp_path: Path) -> Path:
    """Create a temporary state directory."""
    return tmp_path / "state"


def test_cli_help() -> None:
    """Test that --help works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "TurboEdge-DE" in result.stdout


def test_db_info(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    """Test 'db info' command."""
    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "db",
            "info",
        ],
    )
    assert result.exit_code == 0
    assert "runs" in result.stdout
    assert "positions_manual" in result.stdout


def test_position_add_list_close(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    """Test position add, list, and close roundtrip."""
    # Add a position
    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "position",
            "add",
            "--wkn",
            "ABC123",
            "--qty",
            "100",
            "--price",
            "4.86",
            "--date",
            "2026-09-10",
        ],
    )
    assert result.exit_code == 0, f"add failed: {result.stdout}"
    assert "Position added" in result.stdout

    # List positions
    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "position",
            "list",
        ],
    )
    assert result.exit_code == 0
    assert "ABC123" in result.stdout
    assert "open" in result.stdout

    # Close the position
    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "position",
            "close",
            "--wkn",
            "ABC123",
            "--price",
            "5.42",
            "--date",
            "2026-09-15",
        ],
    )
    assert result.exit_code == 0
    assert "Position closed" in result.stdout

    # List again and verify closed
    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "position",
            "list",
            "--status",
            "closed",
        ],
    )
    assert result.exit_code == 0
    assert "ABC123" in result.stdout
    assert "closed" in result.stdout


def test_position_add_validation(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    """Test position add with invalid input."""
    # Invalid WKN (too short)
    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "position",
            "add",
            "--wkn",
            "ABC",
            "--qty",
            "100",
            "--price",
            "4.86",
            "--date",
            "2026-09-10",
        ],
    )
    assert result.exit_code == 2
    assert "invalid WKN" in result.stdout or "invalid WKN" in result.stderr

    # Invalid qty (negative)
    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "position",
            "add",
            "--wkn",
            "ABC123",
            "--qty",
            "-10",
            "--price",
            "4.86",
            "--date",
            "2026-09-10",
        ],
    )
    assert result.exit_code == 2


def test_notify_test_dry_run(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    """Test 'notify test' in dry-run mode (no env vars)."""
    # Clear gmail env vars to force dry-run
    env = os.environ.copy()
    env.pop("GMAIL_USER", None)
    env.pop("GMAIL_APP_PASSWORD", None)
    env.pop("TURBOEDGE_EMAIL_TO", None)

    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "notify",
            "test",
        ],
        env=env,
    )
    assert result.exit_code == 0
    assert "notification" in result.stdout or "dry-run" in result.stdout


def test_sources_health_no_adapters(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    """Test 'sources health' when no product adapters are registered."""
    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "sources",
            "health",
        ],
    )
    # Should warn about no adapters but still succeed
    assert result.exit_code == 0


def test_sources_health_json_output(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    """Test 'sources health' with JSON output."""
    json_out = tmp_state_dir / "health.json"
    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "sources",
            "health",
            "--json-out",
            str(json_out),
        ],
    )
    assert result.exit_code == 0
    # JSON file should be created (even if empty)
    assert json_out.exists()
    try:
        data = json.loads(json_out.read_text())
        assert isinstance(data, list)
    except json.JSONDecodeError:
        pass  # Empty or malformed is OK in this test


def test_config_error(tmp_state_dir: Path) -> None:
    """Test CLI with invalid config directory."""
    result = runner.invoke(
        app,
        [
            "--config-dir",
            "/nonexistent/configs",
            "--state-dir",
            str(tmp_state_dir),
            "db",
            "info",
        ],
    )
    assert result.exit_code == 2
    assert "Configuration error" in result.stdout or "Configuration error" in result.stderr


def test_log_format_option(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    """Test that --log-format option is accepted."""
    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "--log-format",
            "json",
            "db",
            "info",
        ],
    )
    assert result.exit_code == 0


# --------------------------------------------------------------------------
# universe
# --------------------------------------------------------------------------


def test_universe_ok(
    tmp_config_dir: Path, tmp_state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`turboedge universe --underlying DAX` succeeds with a fake product source."""
    product = _make_dax_product(isin="DE000UNIV001", quote_timestamp=datetime.now(UTC))
    adapter = _FakeProductAdapter("source_a", products=[product])
    monkeypatch.setattr(cli_module, "build_product_adapters", lambda cfg, only=None: [adapter])

    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "universe",
            "--underlying",
            "DAX",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Universe run" in result.stdout
    assert "DE000UNIV001" not in result.stdout  # summary table, not per-product


def test_universe_all_sources_fail_exit_3(
    tmp_config_dir: Path, tmp_state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`turboedge universe` exits 3 when every product source raises."""
    adapter = _FakeProductAdapter("source_a", exception=AdapterError("upstream down"))
    monkeypatch.setattr(cli_module, "build_product_adapters", lambda cfg, only=None: [adapter])

    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "universe",
            "--underlying",
            "DAX",
        ],
    )
    assert result.exit_code == 3
    assert "upstream down" in result.stdout or "upstream down" in result.stderr


# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------


def _patch_scan_adapters(
    monkeypatch: pytest.MonkeyPatch,
    *,
    product_adapters: list[Any],
    price_adapter: _FakePriceAdapter,
    estr_adapter: _FakeEstrAdapter,
) -> None:
    monkeypatch.setattr(
        cli_module, "build_product_adapters", lambda cfg, only=None: product_adapters
    )
    monkeypatch.setattr(cli_module, "YFinancePriceAdapter", price_adapter)
    monkeypatch.setattr(cli_module, "EcbEstrAdapter", estr_adapter)
    monkeypatch.setattr(cli_module, "build_reference_healthchecks", lambda cfg: [])


def test_scan_writes_json_and_report_without_actionable(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`scan --json-out --report-out` writes both files; never ACTIONABLE; footer present."""
    product = _make_dax_product(isin="DE000SCAN001", quote_timestamp=datetime.now(UTC))
    adapter = _FakeProductAdapter("source_a", products=[product])
    price_adapter = _FakePriceAdapter({"DAX": _make_dax_bars()})
    estr_adapter = _FakeEstrAdapter(rate=0.03)
    _patch_scan_adapters(
        monkeypatch,
        product_adapters=[adapter],
        price_adapter=price_adapter,
        estr_adapter=estr_adapter,
    )

    json_out = tmp_path / "scan.json"
    report_out = tmp_path / "scan.txt"

    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "scan",
            "--underlying",
            "DAX",
            "--json-out",
            str(json_out),
            "--report-out",
            str(report_out),
        ],
    )
    assert result.exit_code == 0, result.stdout

    assert json_out.exists()
    data = json.loads(json_out.read_text())
    assert data["candidates"], "expected at least one candidate"
    assert all(c["category"] != "ACTIONABLE" for c in data["candidates"])

    assert report_out.exists()
    report_text = report_out.read_text()
    assert "Research system" in report_text
    assert "ACTIONABLE" not in [c["category"] for c in data["candidates"]]


def test_scan_invalid_horizon_exit_2(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "scan",
            "--underlying",
            "DAX",
            "--horizon",
            "9d",
        ],
    )
    assert result.exit_code == 2


def test_scan_unknown_underlying_exit_2(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "scan",
            "--underlying",
            "NOT_A_REAL_UNDERLYING",
        ],
    )
    assert result.exit_code == 2


def test_scan_invalid_direction_exit_2(tmp_config_dir: Path, tmp_state_dir: Path) -> None:
    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "scan",
            "--underlying",
            "DAX",
            "--direction",
            "sideways",
        ],
    )
    assert result.exit_code == 2


# --------------------------------------------------------------------------
# sources health: no product adapter -> WARN, not a silent PASS
# --------------------------------------------------------------------------


def test_sources_health_no_product_adapters_warns(
    tmp_config_dir: Path, tmp_state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_module, "build_product_adapters", lambda cfg, only=None: [])
    monkeypatch.setattr(cli_module, "build_reference_healthchecks", lambda cfg: [])

    result = runner.invoke(
        app,
        [
            "--config-dir",
            str(tmp_config_dir),
            "--state-dir",
            str(tmp_state_dir),
            "sources",
            "health",
        ],
    )
    assert result.exit_code == 0
    assert "WARN" in result.stdout
