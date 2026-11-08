"""Research capture scheduling, actual market-session evidence and E2E safety."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from turboedge.research.capture_monitor import (
    record_capture_attempt,
    verify_restored_tape,
)
from turboedge.research.data_factory import (
    TapeCapture,
    capture_daily_research_tape,
    load_research_tape,
)
from turboedge.research.market_session import (
    is_conservative_market_day,
    is_research_market_session,
)
from turboedge.storage.schemas import FieldReliability, ProductSnapshot

_MONDAY = datetime(2026, 11, 9, 14, 30, tzinfo=UTC)
_MISSING = TapeCapture(0, 0, 0, 0, False, False)


def _products(make: Callable[..., ProductSnapshot], at: datetime) -> list[ProductSnapshot]:
    products = []
    for i, issuer in enumerate(("AlphaBank", "BetaBank")):
        products.append(
            make(
                isin=f"DE000MKT000{i}",
                issuer=issuer,
                underlying_price_ref=20000.0,
                knockout_barrier=19000.0,
                financing_level=19000.0,
                ratio_reliability=FieldReliability.SOURCE_REPORTED,
                bid=9.90 - 0.05 * i,
                ask=10.00,
                quote_timestamp=at - timedelta(seconds=5),
                underlying_price_ref_timestamp=at,
                observation_time=at,
                available_at=at,
                retrieved_at=at,
            )
        )
    return products


def test_market_session_handles_german_summer_and_winter_time() -> None:
    assert is_research_market_session(datetime(2026, 1, 12, 14, 30, tzinfo=UTC))
    assert is_research_market_session(datetime(2026, 7, 13, 14, 30, tzinfo=UTC))
    assert not is_research_market_session(datetime(2026, 1, 12, 22, 40, tzinfo=UTC))
    assert not is_research_market_session(datetime(2026, 7, 13, 22, 40, tzinfo=UTC))
    with pytest.raises(ValueError, match="timezone"):
        is_research_market_session(datetime(2026, 11, 9, 14, 30))


def test_weekends_known_holidays_and_friday_are_conservative() -> None:
    # 2026 Easter: Good Friday 3 April, Easter Monday 6 April.
    assert not is_conservative_market_day(date(2026, 4, 3))
    assert not is_conservative_market_day(date(2026, 4, 6))
    assert not is_conservative_market_day(date(2026, 5, 1))
    assert not is_conservative_market_day(date(2026, 12, 24))
    assert not is_conservative_market_day(date(2026, 12, 31))
    assert not is_research_market_session(datetime(2026, 11, 8, 14, 30, tzinfo=UTC))
    friday = datetime(2026, 11, 13, 14, 30, tzinfo=UTC)
    assert is_research_market_session(friday)
    assert not is_research_market_session(friday, entry=True)


def test_no_weekend_entry_or_faked_weekend_exit(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    state = tmp_path / "state"
    thursday = _MONDAY + timedelta(days=3)
    result = capture_daily_research_tape(
        state, products=_products(make_product_snapshot, thursday), now=thursday
    )
    assert result.entry_pairs == 1
    saturday = thursday + timedelta(days=2)
    weekend = capture_daily_research_tape(
        state, products=_products(make_product_snapshot, saturday), now=saturday
    )
    assert weekend.entry_pairs == 0
    assert weekend.followup_quotes == 0
    assert len(load_research_tape(state, as_of=saturday)) == 2


def test_friday_followup_allowed_but_new_friday_cohort_forbidden(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    state = tmp_path / "state"
    thursday = _MONDAY + timedelta(days=3)
    friday = thursday + timedelta(days=1)
    capture_daily_research_tape(
        state, products=_products(make_product_snapshot, thursday), now=thursday
    )
    result = capture_daily_research_tape(
        state, products=_products(make_product_snapshot, friday), now=friday
    )
    assert result.followup_quotes == 2
    assert result.entry_pairs == 0
    assert result.eligible_pair_found is False


def test_capture_monitor_alerts_after_three_distinct_missing_days(tmp_path: Path) -> None:
    state = tmp_path / "state"
    seen = [
        record_capture_attempt(state, now=_MONDAY + timedelta(days=i), result=_MISSING)
        for i in range(3)
    ]
    assert [x.consecutive_no_pair for x in seen] == [1, 2, 3]
    assert [x.alert for x in seen] == [False, False, True]
    assert seen[-1].reason == "NO_FRESH_CROSS_ISSUER_PAIRS"


def test_same_day_retry_and_friday_are_not_new_failure_days(tmp_path: Path) -> None:
    state = tmp_path / "state"
    monday = record_capture_attempt(state, now=_MONDAY, result=_MISSING)
    retry = record_capture_attempt(state, now=_MONDAY + timedelta(minutes=15), result=_MISSING)
    assert monday.consecutive_no_pair == retry.consecutive_no_pair == 1
    friday = record_capture_attempt(state, now=_MONDAY + timedelta(days=4), result=_MISSING)
    assert friday.consecutive_no_pair == 1
    assert not friday.alert


def test_observed_pair_without_new_cohort_resets_capture_failures(tmp_path: Path) -> None:
    state = tmp_path / "state"
    record_capture_attempt(state, now=_MONDAY, result=_MISSING)
    good = record_capture_attempt(
        state,
        now=_MONDAY + timedelta(days=1),
        result=TapeCapture(0, 0, 0, 0, False, True),
    )
    assert good.consecutive_no_pair == 0
    assert not good.alert


def test_restored_tape_missing_is_hard_error_not_fake_zero(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    state = tmp_path / "state"
    capture = capture_daily_research_tape(
        state, products=_products(make_product_snapshot, _MONDAY), now=_MONDAY
    )
    assert capture.entry_pairs == 1
    record_capture_attempt(state, now=_MONDAY, result=capture)
    tape_dir = state / "snapshots" / "research_product_quotes"
    assert tape_dir.is_dir()
    for file in tape_dir.rglob("*.parquet"):
        file.unlink()
    with pytest.raises(ValueError, match="missing, rolled back"):
        verify_restored_tape(state, now=_MONDAY + timedelta(days=1))


def test_pipeline_separates_capture_eod_and_live_scan() -> None:
    workflow = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "pipeline.yml"
    ).read_text(encoding="utf-8")
    assert '- cron: "30 14 * * 1-5" # research_capture' in workflow
    assert '"30 14 * * 1-5") mode="research_capture"' in workflow
    capture, eod = workflow.split("  research_capture:", 1)[1].split("  eod:", 1)
    assert "archive products --require-market-hours --monitor" in capture
    assert "turboedge-state-pack" in capture
    assert "archive products" not in eod.split("  external:", 1)[0]
    assert "position reevaluate --email" in eod
