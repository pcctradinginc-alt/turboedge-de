"""What each product source can and cannot deliver, measured not assumed.

Measured in the production run of 2026-10-25 (GitHub Actions run
40186945583, the `eod` job's `archive products` step), which fetched
20,020 snapshots with **zero application warnings** apart from the Citi
ones below:

    citi_partial_universe      DAX covered=25 total=33
    citi_underlying_unresolved NDX, EURUSD, XAU
    citi_quotes_summary        count_total=25
                               count_ask_missing=25
                               count_bid_missing=25

So the Citi feed delivered master data for 25 DAX products and **not one
tradable quote** -- no ask on any row, no bid on any row -- and could not
resolve three of the four enabled underlyings at all.

gettex, by contrast, answered 197 paginated requests with HTTP 200 and
produced no warning of any kind, and BNP answered 26 with no
`bnp_partial_universe` and no `bnp_pagination_offset_error`. The
"zero archived gettex products" and "BNP pagination offset error" reported
against earlier runs did not reproduce.

These tests pin the consequence rather than the measurement: a source that
cannot quote must not be able to produce a trade proposal, whatever else
changes around it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from turboedge.pricing.integrity import IntegrityReport
from turboedge.ranking.gates import GateInput, GateThresholds, evaluate_gates
from turboedge.storage.schemas import Category, FieldReliability

_NOW = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
_THRESHOLDS = GateThresholds(
    max_spread_pct=0.03,
    max_source_quote_age_s=600.0,
    max_quote_age_at_decision_s=300.0,
    min_leverage=2.0,
    max_leverage=50.0,
    min_distance_to_barrier_sigma=1.0,
)


def _gate_input(**over: object) -> GateInput:
    defaults: dict[str, object] = dict(
        integrity=IntegrityReport(passed=True, failures=[], warnings=[]),
        bid_only=False,
        knocked_out=False,
        quote_age_s=10.0,
        spread_pct=0.01,
        leverage=5.0,
        distance_to_barrier_sigma=2.0,
        data_health_pass=True,
        lcb_ev=0.05,
        p_ko=0.1,
        cluster_risk_pass=True,
        ratio_reliability=FieldReliability.SOURCE_REPORTED,
        confirmed_alpha_pass=True,
        portfolio_risk_pass=True,
        suggested_position_fraction=0.01,
    )
    defaults.update(over)
    return GateInput(**defaults)  # type: ignore[arg-type]


def test_a_row_with_no_ask_cannot_be_actionable() -> None:
    # Every one of the 25 Citi rows measured on 2026-10-25 is in this state.
    # Entry is at Ask (CLAUDE.md rule 11), so a row without one cannot be
    # entered at any price -- it is master data, not a tradable quote.
    category, reasons = evaluate_gates(_gate_input(has_ask=False), _THRESHOLDS)

    assert category is Category.REJECT
    assert "no_ask_quote" in reasons


def test_a_row_with_no_live_quote_at_all_cannot_be_actionable() -> None:
    category, reasons = evaluate_gates(_gate_input(no_live_quote=True), _THRESHOLDS)

    assert category is Category.REJECT
    assert "no_live_quote" in reasons


def test_a_bid_only_row_cannot_be_actionable() -> None:
    # The same situation as a missing ask, reported by the source itself.
    category, reasons = evaluate_gates(_gate_input(bid_only=True), _THRESHOLDS)

    assert category is Category.REJECT
    assert "bid_only" in reasons


def test_an_unquotable_row_is_rejected_before_any_economic_gate() -> None:
    # It must not matter how good the EV looks: a row that cannot be bought
    # is rejected on tradability, and the economic preconditions are never
    # reached. Otherwise a closing-price-only feed could carry a proposal on
    # the strength of a model.
    category, reasons = evaluate_gates(
        _gate_input(
            has_ask=False,
            lcb_ev=10.0,
            confirmed_alpha_pass=True,
            portfolio_risk_pass=True,
        ),
        _THRESHOLDS,
    )

    assert category is Category.REJECT
    assert "no_ask_quote" in reasons
    assert "lcb_ev_not_positive" not in reasons


def test_a_fully_quoted_row_is_still_reachable() -> None:
    # The gates must stay satisfiable for a source that does quote, or this
    # would be a disabled feature rather than a classification.
    assert evaluate_gates(_gate_input(), _THRESHOLDS)[0] is Category.ACTIONABLE
