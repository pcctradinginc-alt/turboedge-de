from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Category,
    Direction,
    LedgerEntry,
    LedgerEntryStatus,
    UnderlyingBar,
)


@pytest.fixture
def store(tmp_path: Path):  # type: ignore[no-untyped-def]
    with Store(tmp_path / "turboedge.duckdb") as s:
        s.init_schema()
        yield s


def _utc(year: int, month: int, day: int, hour: int = 12, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


@pytest.fixture
def make_ledger_entry() -> Callable[..., LedgerEntry]:
    """Factory for a valid, fully-populated LedgerEntry with sane overrides."""

    def _make(**overrides: object) -> LedgerEntry:
        defaults: dict[str, object] = dict(
            run_id="run-1",
            candidate_id="cand-1",
            signal_id="tsmom_horizon_norm_v1",
            signal_version_hash="sig-hash-1",
            trial_id="TR-2026Q3-abc123",
            prediction_time=_utc(2026, 9, 1, 8, 0),
            underlying="DAX",
            direction=Direction.LONG,
            horizon_days=7,
            regime_bucket=None,
            cluster_id=None,
            feature_hash="feat-hash-1",
            model_hash="model-hash-1",
            config_hash="cfg-hash-1",
            git_commit=None,
            category=Category.WATCH,
            selected_wkn="ABC123",
            selected_isin="DE000ABC1234",
            issuer="TestBank",
            entry_bid=4.80,
            entry_ask=4.86,
            entry_spread=(4.86 - 4.80) / 4.86,
            entry_quote_timestamp=_utc(2026, 9, 1, 8, 0),
            entry_underlying_timestamp=_utc(2026, 9, 1, 8, 0),
            financing_level_entry=18000.0,
            barrier_entry=18000.0,
            ratio=0.01,
            fx=1.0,
            predicted_return=0.01,
            p_profit=0.55,
            p_ko=0.05,
            expected_shortfall=-0.1,
            lcb_ev=0.001,
            uncertainty=0.02,
            shrinkage_intensity=0.5,
            is_shadow=False,
            shadow_stratum=None,
            suggested_position_fraction=None,
            exit_due=date(2026, 9, 10),
            alternatives=[],
            feature_snapshot={},
            status=LedgerEntryStatus.OPEN,
        )
        defaults.update(overrides)
        return LedgerEntry(**defaults)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def make_underlying_bar() -> Callable[..., UnderlyingBar]:
    """Factory for a valid, fully-populated UnderlyingBar with sane overrides."""

    def _make(**overrides: object) -> UnderlyingBar:
        defaults: dict[str, object] = dict(
            underlying_id="DAX",
            ts=_utc(2026, 9, 1),
            interval="1d",
            open=18000.0,
            high=18100.0,
            low=17950.0,
            close=18050.0,
            volume=1000.0,
            observation_time=_utc(2026, 9, 1),
            available_at=_utc(2026, 9, 1, 22),
            retrieved_at=_utc(2026, 9, 1, 22, 5),
            source="yfinance",
            parser_version="1",
            quality_score=0.9,
        )
        defaults.update(overrides)
        return UnderlyingBar(**defaults)  # type: ignore[arg-type]

    return _make
