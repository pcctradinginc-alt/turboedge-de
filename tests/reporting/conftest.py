from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Category,
    Direction,
    ExitReason,
    LedgerEntry,
    LedgerEntryStatus,
    LedgerLabel,
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
    """Factory for a valid, fully-populated LedgerEntry with sane overrides
    (mirrors tests/learning/conftest.py's fixture of the same name)."""

    def _make(**overrides: object) -> LedgerEntry:
        defaults: dict[str, object] = dict(
            run_id="run-1",
            candidate_id="cand-1",
            signal_id="tsmom_horizon_norm_v1",
            signal_version_hash="sig-hash-1",
            trial_id="TR-2026Q3-abc123",
            prediction_time=_utc(2026, 8, 1, 8, 0),
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
            entry_quote_timestamp=_utc(2026, 8, 1, 8, 0),
            entry_underlying_timestamp=_utc(2026, 8, 1, 8, 0),
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
            exit_due=date(2026, 8, 10),
            alternatives=[],
            feature_snapshot={},
            status=LedgerEntryStatus.OPEN,
        )
        defaults.update(overrides)
        return LedgerEntry(**defaults)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def make_ledger_label() -> Callable[..., LedgerLabel]:
    """Factory for a valid, fully-populated LedgerLabel with sane overrides."""

    def _make(**overrides: object) -> LedgerLabel:
        defaults: dict[str, object] = dict(
            entry_id="placeholder",
            labeled_at=_utc(2026, 8, 10, 15, 30),
            exit_bid=5.10,
            exit_quote_timestamp=_utc(2026, 8, 10, 15, 30),
            financing_level_exit=18020.0,
            exit_reason=ExitReason.HORIZON,
            realized_selected_pnl=5.10 / 4.86 - 1,
            underlying_pnl=0.01,
            median_turbo_pnl=None,
            best_turbo_pnl=None,
            ideal_turbo_pnl=None,
            mfe=0.06,
            mae=-0.01,
            ko_hit=False,
            time_to_ko_days=None,
            ambiguous_path=False,
        )
        defaults.update(overrides)
        return LedgerLabel(**defaults)  # type: ignore[arg-type]

    return _make


def _record_labeled(
    store: Store,
    make_ledger_entry: Callable[..., LedgerEntry],
    make_ledger_label: Callable[..., LedgerLabel],
    *,
    entry_overrides: dict[str, object] | None = None,
    label_overrides: dict[str, object] | None = None,
) -> tuple[LedgerEntry, LedgerLabel]:
    """Record one LedgerEntry + attach a matching LedgerLabel in one call,
    returning both (as actually persisted)."""
    from turboedge.learning.ledger import ForwardLedger

    entry = make_ledger_entry(**(entry_overrides or {}))
    ledger = ForwardLedger(store)
    ledger.record([entry])
    label_kwargs = dict(label_overrides or {})
    label_kwargs.setdefault("entry_id", entry.entry_id)
    label = make_ledger_label(**label_kwargs)
    ledger.attach_label(label)
    return entry, label


@pytest.fixture
def record_labeled(
    store: Store,
    make_ledger_entry: Callable[..., LedgerEntry],
    make_ledger_label: Callable[..., LedgerLabel],
) -> Callable[..., tuple[LedgerEntry, LedgerLabel]]:
    """Fixture wrapping :func:`_record_labeled`, bound to this test's
    ``store``/factories -- avoids a same-package ``tests.reporting.conftest``
    import (there is no ``tests/__init__.py`` in this repo, so pytest test
    modules share helpers via fixtures, not module imports)."""

    def _make(
        *,
        entry_overrides: dict[str, object] | None = None,
        label_overrides: dict[str, object] | None = None,
    ) -> tuple[LedgerEntry, LedgerLabel]:
        return _record_labeled(
            store,
            make_ledger_entry,
            make_ledger_label,
            entry_overrides=entry_overrides,
            label_overrides=label_overrides,
        )

    return _make
