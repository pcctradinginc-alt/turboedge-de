"""Autonomous Alpha Discovery Phase 1: synthetic, dated, reproducible cases.

Synthetic outcomes are deliberately chosen; success here establishes safe
automation and test wiring, emphatically not measured market alpha.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from turboedge.meta.research_opportunity import ResearchStatus
from turboedge.research.alpha_discovery import (
    HARNESS,
    MIN_VALIDATION_DAYS,
    Feature,
    Rule,
    _candidate_sessions,
    _split,
    evaluate,
    generate,
    run_discovery_step,
)
from turboedge.research.controller import EXPLORATORY_SCREENS_PER_QUARTER
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Direction,
    FieldReliability,
    ProductSnapshot,
    ProductType,
    ResearchScreen,
    ScreenStatus,
)

_NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
_FIRST = datetime(2025, 9, 1, 12, 0, 30, tzinfo=UTC)


def _quote(
    isin: str,
    issuer: str,
    at: datetime,
    *,
    bid: float,
    ask: float = 10.0,
    age_seconds: int = 0,
    size: float = 100.0,
    knocked_out: bool = False,
) -> ProductSnapshot:
    return ProductSnapshot(
        isin=isin,
        wkn=isin[-6:],
        issuer=issuer,
        venue="fake",
        underlying_raw="DAX",
        underlying_id="DAX",
        direction=Direction.LONG,
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=19000.0,
        knockout_barrier=19000.0,
        ratio=0.01,
        currency="EUR",
        underlying_currency="EUR",
        quanto=False,
        open_end=True,
        bid=bid,
        ask=ask,
        bid_size=size,
        ask_size=100.0,
        quote_timestamp=at - timedelta(seconds=age_seconds),
        quote_presence=True,
        bid_only=False,
        knocked_out=knocked_out,
        underlying_price_ref=20000.0,
        underlying_price_ref_timestamp=at,
        ratio_reliability=FieldReliability.SOURCE_REPORTED,
        raw_hash=hashlib.sha256((isin + str(at)).encode()).hexdigest(),
        observation_time=at,
        available_at=at,
        retrieved_at=at,
        source="fake",
        parser_version="1",
        is_stale=False,
        quality_score=0.99,
    )


def _history(days: int = 120, *, missing_b_exit: bool = False, negative: bool = False):
    rows: list[ProductSnapshot] = []
    for i in range(days):
        at = _FIRST + timedelta(days=3 * i)
        isin_a = f"DE000GENA{i:04d}"
        isin_b = f"DE000GENB{i:04d}"
        rows.extend(
            [
                _quote(isin_a, "BankA", at, bid=9.50, size=100.0),
                _quote(isin_b, "BankB", at, bid=9.95, size=200.0, age_seconds=20),
            ]
        )
        rows.append(
            _quote(isin_a, "BankA", at + timedelta(days=1), bid=9.2 if negative else 10.8)
        )
        if not missing_b_exit or i < days // 2:
            rows.append(
                _quote(
                    isin_b,
                    "BankB",
                    at + timedelta(days=1, minutes=3),
                    bid=9.0 if negative else 11.5,
                )
            )
    return rows


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "research.duckdb") as db:
        db.init_schema()
        yield db


def test_dsl_rejects_unknown_code_and_unknown_feature() -> None:
    with pytest.raises(ValidationError):
        Rule.model_validate(
            {"primary": "relative_spread", "secondary": "quote_age", "code": "import os"}
        )
    with pytest.raises(ValidationError):
        Rule(primary=Feature.QUOTE_AGE, secondary=Feature.QUOTE_AGE)
    with pytest.raises(ValidationError):
        Rule.model_validate({"primary": "expr(x)", "secondary": "bid_depth"})


def test_generated_hypotheses_use_observations_not_prelisted_names() -> None:
    session = _candidate_sessions(_history(days=40), now=_NOW)
    development, validation = _split(session)
    candidates = generate(development, history=[])
    assert candidates
    assert len(validation) == 20
    assert all(c.rule.hypothesis_id.startswith("RO-GEN-") for c in candidates)
    assert len({c.rule.hypothesis_id for c in candidates}) == len(candidates)
    assert candidates == generate(development, history=[])
    assert all(c.priority > 0 for c in candidates)


def test_full_cycle_proposes_and_tests_without_approving(
    store: Store,
) -> None:
    result = run_discovery_step(store, now=_NOW, snapshots=_history())
    assert result.status == "SCREENED_EXPLORATORY"
    assert result.selected_hypothesis_id is not None
    assert result.screen_id is not None
    assert result.verdict == "CANDIDATE_REQUIRES_INDEPENDENT_CONFIRMATION"
    assert result.validation_days >= MIN_VALIDATION_DAYS
    assert result.measurement is not None
    assert result.measurement.lcb_net is not None
    assert result.measurement.lcb_excess_naive is not None
    assert result.measurement.lcb_net > 0.001
    assert result.measurement.lcb_excess_naive > 0.0
    screen = store.list_research_screens()[0]
    assert screen.harness == HARNESS
    assert screen.status is ScreenStatus.COMPLETED
    assert not screen.produced_support
    assert screen.metrics["confirmation_allowed"] is False
    assert screen.trial_id == result.screen_id
    assert store.get_research_trial(screen.trial_id) is not None
    assert all(
        row.opportunity.status is ResearchStatus.PROPOSED
        for row in store.list_research_opportunities()
    )
    assert store.list_alpha_sources() == []


def test_idempotent_does_not_rescreen_an_existing_hypothesis(store: Store) -> None:
    snapshots = _history()
    first = run_discovery_step(store, now=_NOW, snapshots=snapshots)
    second = run_discovery_step(store, now=_NOW, snapshots=snapshots)
    assert first.screen_id != second.screen_id
    third = run_discovery_step(store, now=_NOW, snapshots=snapshots)
    assert third.screen_id != first.screen_id
    assert len(store.list_research_screens()) == 3
    # The six finite, reviewed DSL permutations eventually exhaust; the
    # engine may try different predeclared rules but not repeat a failed one.
    for _ in range(5):
        run_discovery_step(store, now=_NOW, snapshots=snapshots)
    ids = [s.hypothesis_id for s in store.list_research_screens()]
    assert len(ids) == len(set(ids))
    assert len(ids) <= 6


def test_missing_exit_refuses_positive_evidence(store: Store) -> None:
    result = run_discovery_step(
        store, now=_NOW, snapshots=_history(missing_b_exit=True)
    )
    assert result.status == "SCREENED_EXPLORATORY"
    assert result.verdict == "INCOMPLETE_EVIDENCE"
    assert result.measurement is not None
    assert result.measurement.missing_exit_days > 0


def test_negative_true_bid_ask_returns_reject_alpha(store: Store) -> None:
    result = run_discovery_step(store, now=_NOW, snapshots=_history(negative=True))
    assert result.verdict == "NO_DEMONSTRATED_NET_ALPHA"
    assert result.measurement is not None
    assert result.measurement.mean_net is not None
    assert result.measurement.mean_net < 0


def test_too_few_distinct_days_does_not_consume_trial(store: Store) -> None:
    result = run_discovery_step(store, now=_NOW, snapshots=_history(days=24))
    assert result.status == "AWAIT_DATA"
    assert not store.list_research_trials()
    assert not store.list_research_screens()


def test_dry_run_does_not_modify_any_store_table(store: Store) -> None:
    result = run_discovery_step(store, now=_NOW, snapshots=_history(), dry_run=True)
    assert result.status == "DRY_RUN"
    assert store.list_research_opportunities() == []
    assert store.list_research_screens() == []
    assert store.list_research_trials() == []


def test_shared_quarterly_budget_is_respected(store: Store) -> None:
    for i in range(EXPLORATORY_SCREENS_PER_QUARTER):
        store.upsert_research_screen(
            ResearchScreen(
                screen_id=f"2026Q4_SCREEN_filler_{i}",
                trial_id=f"2026Q4_SCREEN_filler_{i}",
                hypothesis_id=f"FILLER-{i}",
                family="product_selection",
                harness="historical",
                harness_version="1",
                spec={},
                data_version="old",
                quarter="2026Q4",
                status=ScreenStatus.COMPLETED,
                started_at=_NOW - timedelta(days=1),
                finished_at=_NOW - timedelta(days=1),
            )
        )
    result = run_discovery_step(store, now=_NOW, snapshots=_history())
    assert result.status == "BUDGET_EXHAUSTED"
    assert len(store.list_research_screens()) == EXPLORATORY_SCREENS_PER_QUARTER


def test_old_or_late_quote_does_not_get_favored_by_generator() -> None:
    rows = _history(days=6)
    # All BankB entry quotes are stale; valid cross-issuer pairs disappear.
    rows = [
        r.model_copy(update={"quote_timestamp": r.retrieved_at - timedelta(minutes=20)})
        if r.issuer == "BankB" and r.bid == 9.95
        else r
        for r in rows
    ]
    assert _candidate_sessions(rows, now=_NOW) == []


def test_forward_exit_not_known_at_cutoff_cannot_create_positive_ev() -> None:
    rows = _history(days=120)
    sessions = _candidate_sessions(rows, now=_NOW)
    _, validation = _split(sessions)
    rule = Rule(primary=Feature.BID_DEPTH, secondary=Feature.QUOTE_AGE)
    # At entry time none of the exit prices is yet known.
    first = validation[0]
    measurement = evaluate(
        rule,
        validation[:MIN_VALIDATION_DAYS],
        snapshots=rows,
        now=first.left.retrieved_at,
    )
    assert measurement.verdict == "INCOMPLETE_EVIDENCE"


def test_rejected_history_penalises_similar_new_hypotheses() -> None:
    sessions = _candidate_sessions(_history(days=40), now=_NOW)
    development, _ = _split(sessions)
    initial = generate(development, history=[])
    from turboedge.storage.schemas import ResearchScreen

    past = ResearchScreen(
        screen_id="2026Q3_AUTO_DSL_failed",
        trial_id="2026Q3_AUTO_DSL_failed",
        hypothesis_id="RO-GEN-previous",
        family="product_selection",
        harness=HARNESS,
        harness_version="1",
        spec=initial[0].rule.model_dump(mode="json"),
        data_version="past",
        quarter="2026Q3",
        status=ScreenStatus.COMPLETED,
        verdict="NO_DEMONSTRATED_NET_ALPHA",
        started_at=_NOW,
        finished_at=_NOW,
    )
    updated = generate(development, history=[past])
    old = {c.rule.hypothesis_id: c.priority for c in initial}
    assert all(c.priority <= old[c.rule.hypothesis_id] for c in updated)
    assert any(c.overlap_with_failures for c in updated)


def test_duplicate_days_never_count_as_independent_trials() -> None:
    sessions = _candidate_sessions(_history(days=40), now=_NOW)
    _, validation = _split(sessions)
    measurement = evaluate(
        Rule(primary=Feature.BID_DEPTH, secondary=Feature.QUOTE_AGE),
        [validation[0]] * MIN_VALIDATION_DAYS,
        snapshots=_history(days=40),
        now=_NOW,
    )
    assert measurement.verdict == "INVALID_SAMPLE"
    assert measurement.lcb_net is None
