"""Tests for the Contract v3 integration-wave CLI commands
(``turboedge.cli_learn``): ``scan-all``, ``label``, ``learn``, ``position
reevaluate``, ``report monthly``, ``research tournament``, ``forecast``,
``backtest``. No network: every adapter is monkeypatched to an in-memory
fake, mirroring ``tests/test_cli.py``'s own pattern (this repo has no
``tests/__init__.py``, so fixtures are duplicated rather than imported --
see the note at the top of ``tests/test_cli.py``).

Every test ``monkeypatch.chdir(tmp_path)`` before invoking the CLI so the
``reports/summary.json`` convention never writes into the real repository.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import turboedge.cli_learn as cli_learn_module
from turboedge.adapters.base import AdapterMetadata, HealthCheckResult, ProductFetchContext
from turboedge.cli import app
from turboedge.config import CONFIG_FILES
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Direction,
    HealthStatus,
    ProductSnapshot,
    ProductType,
    UnderlyingBar,
)

runner = CliRunner()
_EVAL_TIME = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)


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


class _FakeProductAdapter:
    def __init__(self, name: str, products: Sequence[ProductSnapshot]) -> None:
        self._name = name
        self._products = list(products)

    @property
    def name(self) -> str:
        return self._name

    def fetch_products(
        self,
        underlying_ids: Sequence[str],
        *,
        context: ProductFetchContext | None = None,
    ) -> list[ProductSnapshot]:
        del context
        return list(self._products)

    def healthcheck(self) -> HealthCheckResult:
        return HealthCheckResult(
            source=self._name,
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=1.0,
            checked_at=datetime.now(UTC),
            message="fake",
        )

    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(name=self._name, kind="product", version="1")


class _FakePriceAdapter:
    def __init__(self, bars_by_underlying: dict[str, list[UnderlyingBar]]) -> None:
        self._bars = bars_by_underlying

    def __call__(self) -> _FakePriceAdapter:  # patched in as a class replacement
        return self

    def fetch_daily_bars(
        self, underlying_id: str, *, lookback_days: int | None = None
    ) -> list[UnderlyingBar]:
        return list(self._bars.get(underlying_id, []))

    def healthcheck(self) -> HealthCheckResult:
        return HealthCheckResult(
            source="fake_price",
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=1.0,
            checked_at=datetime.now(UTC),
            message="fake",
        )


class _FakeEstrAdapter:
    def __init__(self, http: Any = None, fallback_rate: float = 0.03) -> None:
        self.rate = fallback_rate

    def get_estr(self) -> float:
        return self.rate

    def healthcheck(self) -> HealthCheckResult:
        return HealthCheckResult(
            source="fake_estr",
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=1.0,
            checked_at=datetime.now(UTC),
            message="fake",
        )


def _make_bars(count: int = 60) -> list[UnderlyingBar]:
    bars: list[UnderlyingBar] = []
    close = 24000.0
    d = date(2025, 1, 1)
    rng = np.random.default_rng(5)
    while len(bars) < count:
        if d.weekday() < 5:
            new_close = close * (1.0 + float(rng.normal(0.0005, 0.005)))
            ts = datetime(d.year, d.month, d.day, tzinfo=UTC)
            avail = datetime(d.year, d.month, d.day, 22, 0, tzinfo=UTC)
            bars.append(
                UnderlyingBar(
                    underlying_id="DAX",
                    ts=ts,
                    open=close,
                    high=max(close, new_close) * 1.001,
                    low=min(close, new_close) * 0.999,
                    close=new_close,
                    volume=1.0,
                    observation_time=avail,
                    available_at=avail,
                    retrieved_at=avail,
                    source="fake",
                    parser_version="1",
                    is_stale=False,
                    quality_score=0.9,
                )
            )
            close = new_close
        d += timedelta(days=1)
    return bars


def _make_product(isin: str = "DE000CLILRN1") -> ProductSnapshot:
    return ProductSnapshot(
        isin=isin,
        wkn=isin[-6:],
        issuer="BankA",
        venue="stuttgart",
        underlying_raw="DAX",
        underlying_id="DAX",
        direction=Direction.LONG,
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=18000.0,
        knockout_barrier=18000.0,
        ratio=0.01,
        currency="EUR",
        underlying_currency="EUR",
        quanto=False,
        open_end=True,
        bid=59.9,
        ask=60.1,
        bid_size=100.0,
        ask_size=100.0,
        quote_timestamp=_EVAL_TIME,
        quote_presence=True,
        bid_only=False,
        knocked_out=False,
        raw_hash=hashlib.sha256(isin.encode()).hexdigest(),
        observation_time=_EVAL_TIME,
        available_at=_EVAL_TIME,
        retrieved_at=_EVAL_TIME,
        source="fake",
        parser_version="1",
        is_stale=False,
        quality_score=0.9,
    )


def _patch_scan_all_adapters(
    monkeypatch: pytest.MonkeyPatch, bars: dict[str, list[UnderlyingBar]]
) -> None:
    products = [_make_product()]
    monkeypatch.setattr(
        cli_learn_module,
        "build_product_adapters",
        lambda cfg: [_FakeProductAdapter("source_a", products)],
    )
    monkeypatch.setattr(cli_learn_module, "YFinancePriceAdapter", lambda: _FakePriceAdapter(bars))
    monkeypatch.setattr(cli_learn_module, "EcbEstrAdapter", _FakeEstrAdapter)
    monkeypatch.setattr(cli_learn_module, "build_reference_healthchecks", lambda cfg: [])


def test_scan_all_writes_summary_json(
    tmp_config_dir: Path,
    tmp_state_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_scan_all_adapters(monkeypatch, {"DAX": _make_bars(), "EURUSD": _make_bars()})

    result = runner.invoke(
        app, [*_base_args(tmp_config_dir, tmp_state_dir), "scan-all", "--underlying", "DAX"]
    )
    assert result.exit_code == 0, result.stdout

    summary_path = tmp_path / "reports" / "summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["mode"] == "scan"
    assert "counts" in payload
    assert sum(payload["counts"].values()) >= 1


def test_scan_all_summary_json_run_ids_match_written_snapshot_files(
    tmp_config_dir: Path,
    tmp_state_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``reports/summary.json``'s ``run_ids`` (added so pipeline.yml's scan
    job can build the incremental Parquet snapshot artifact via ``state
    pack-snapshots --run-id ...``) must be exactly the run_ids `scan-all`
    used -- and each one must have a real Parquet file under
    state/snapshots/ to show for it, otherwise the workflow step reading
    this field would pack nothing."""
    monkeypatch.chdir(tmp_path)
    _patch_scan_all_adapters(monkeypatch, {"DAX": _make_bars(), "EURUSD": _make_bars()})

    result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "scan-all",
            "--underlying",
            "DAX",
            "--underlying",
            "EURUSD",
        ],
    )
    assert result.exit_code == 0, result.stdout

    payload = json.loads((tmp_path / "reports" / "summary.json").read_text())
    run_ids = payload["run_ids"]
    assert set(run_ids) == {"DAX", "EURUSD"}
    assert len(set(run_ids.values())) == 2  # every underlying gets its own run_id

    from turboedge.storage.snapshots import snapshot_paths_for_run_ids

    found = snapshot_paths_for_run_ids(tmp_state_dir, list(run_ids.values()))
    assert len(found) == 2


def test_label_and_learn_no_due_entries_write_summary(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, [*_base_args(tmp_config_dir, tmp_state_dir), "label"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert payload["mode"] == "label"
    assert payload["counts"]["labeled"] == 0

    result2 = runner.invoke(app, [*_base_args(tmp_config_dir, tmp_state_dir), "learn"])
    assert result2.exit_code == 0, result2.stdout
    payload2 = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert payload2["mode"] == "learn"
    assert payload2["counts"]["posteriors_updated"] == 0


def test_report_monthly_without_trades_is_status_only(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, [*_base_args(tmp_config_dir, tmp_state_dir), "report", "monthly", "--month", "2026-06"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert payload["mode"] == "report_monthly"
    assert payload["counts"]["status_only"] == 1
    assert (tmp_path / "reports" / "monthly" / "2026-06.json").exists()


def test_research_tournament_runs_with_empty_ledger(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, [*_base_args(tmp_config_dir, tmp_state_dir), "research", "tournament"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert payload["mode"] == "research_tournament"


def test_position_reevaluate_invalidates_position_without_isin(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    add_result = runner.invoke(
        app,
        [
            *_base_args(tmp_config_dir, tmp_state_dir),
            "position",
            "add",
            "--wkn",
            "ABC123",
            "--qty",
            "10",
            "--price",
            "5.0",
            "--date",
            "2026-01-01",
        ],
    )
    assert add_result.exit_code == 0, add_result.stdout

    result = runner.invoke(
        app, [*_base_args(tmp_config_dir, tmp_state_dir), "position", "reevaluate"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert payload["mode"] == "position_reevaluate"
    assert payload["counts"].get("INVALIDATED") == 1

    with Store(tmp_state_dir / "turboedge.duckdb") as store:
        store.init_schema()
        assert store.table_counts()["position_evaluations"] == 1


def test_forecast_diagnostic_no_network(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_learn_module,
        "YFinancePriceAdapter",
        lambda: _FakePriceAdapter({"DAX": _make_bars(300)}),
    )
    result = runner.invoke(
        app, [*_base_args(tmp_config_dir, tmp_state_dir), "forecast", "--underlying", "DAX"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert payload["mode"] == "forecast"
    assert payload["counts"]["models_fit"] >= 1


def test_backtest_persists_walkforward_results(
    tmp_config_dir: Path, tmp_state_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_learn_module,
        "YFinancePriceAdapter",
        lambda: _FakePriceAdapter({"DAX": _make_bars(400)}),
    )
    result = runner.invoke(
        app, [*_base_args(tmp_config_dir, tmp_state_dir), "backtest", "--underlying", "DAX"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert payload["mode"] == "backtest"

    with Store(tmp_state_dir / "turboedge.duckdb") as store:
        store.init_schema()
        rows = store.list_walkforward_results(underlying_id="DAX")
        assert len(rows) >= 1
        assert all(r.brier_null is not None for r in rows)
