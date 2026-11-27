"""Observed ledger policy diagnostics, not a replay of unavailable candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from math import isfinite
from statistics import mean

from turboedge.research.execution_costs import COST_SCENARIOS, NOTIONALS_EUR
from turboedge.storage.schemas import Category, ExitReason, LedgerEntry, LedgerLabel


def evaluate_filters(
    rows: Sequence[tuple[LedgerEntry, LedgerLabel | None]], *, as_of: datetime
) -> dict[str, object]:
    """Compare fixed policies on the same archived opportunities, never trade.

    Only non-shadow actionable/watch entries are used. WATCH is not an
    isolated production gate, so this is a policy diagnostic, not causal
    evidence that removing any specific gate improves live results.
    """
    cohorts: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    missing = 0
    eligible = 0
    for entry, label in rows:
        if entry.is_shadow or entry.category not in (Category.ACTIONABLE, Category.WATCH):
            continue
        if entry.exit_due >= as_of.date():
            continue
        eligible += 1
        if (
            label is None
            or label.labeled_at > as_of
            or label.ambiguous_path
            or label.exit_reason not in (ExitReason.HORIZON, ExitReason.KO)
            or label.realized_selected_pnl is None
            or not isfinite(label.realized_selected_pnl)
        ):
            missing += 1
            continue
        key = (
            entry.prediction_time.date().isoformat(),
            entry.underlying,
            entry.horizon_days,
            entry.category.value,
        )
        cohorts[key].append(label.realized_selected_pnl)
    effects: list[dict[str, object]] = []
    for horizon in sorted({key[2] for key in cohorts}):
        matched = []
        for day, uid, h, category in sorted(cohorts):
            if h != horizon or category != Category.ACTIONABLE.value:
                continue
            watch = cohorts.get((day, uid, h, Category.WATCH.value))
            if watch:
                active = cohorts[(day, uid, h, category)]
                matched.append((sum(active) / len(active), sum(watch) / len(watch)))
        effects.append(
            {
                "horizon": horizon,
                "matched_day_underlyings": len(matched),
                "watch_minus_actionable": (
                    sum(b - a for a, b in matched) / len(matched) if matched else None
                ),
            }
        )
    if not eligible:
        status = "INSUFFICIENT_DATA"
    elif missing:
        status = "INCOMPLETE_OUTCOMES"
    elif not any(effect["watch_minus_actionable"] is not None for effect in effects):
        status = "NO_MATCHED_COHORTS"
    else:
        status = "DESCRIPTIVE_ONLY"
    return {
        "status": status,
        "eligible_rows": eligible,
        "missing_or_unverified_rows": missing,
        "observed_day_category_cohorts": len(cohorts),
        "effects": effects,
        "cost_basis": "observed_bid_ask_proxy_before_broker_costs",
        "causal_filter_ablation": False,
    }


def evaluate_lcb(
    rows: Sequence[tuple[LedgerEntry, LedgerLabel | None]], *, as_of: datetime
) -> dict[str, object]:
    """Descriptive ranking diagnostics, not confidence-interval coverage."""
    groups: dict[tuple[int, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    missing = eligible = 0
    for entry, label in rows:
        if entry.exit_due >= as_of.date() or entry.prediction_time > as_of:
            continue
        eligible += 1
        if (
            label is None
            or label.labeled_at > as_of
            or label.ambiguous_path
            or label.exit_reason not in (ExitReason.HORIZON, ExitReason.KO)
            or label.realized_selected_pnl is None
            or not isfinite(label.realized_selected_pnl)
            or not isfinite(entry.lcb_ev)
        ):
            missing += 1
            continue
        # Separate model/config versions and horizons; aggregate by day first.
        group = groups[(entry.horizon_days, entry.model_hash, entry.config_hash)]
        bucket = "positive" if entry.lcb_ev > 0 else "nonpositive"
        key = f"{bucket}:{entry.prediction_time.date().isoformat()}"
        group[key].append(label.realized_selected_pnl)
    effects = []
    for (horizon, model, config), days in sorted(groups.items()):
        positive = [mean(v) for k, v in days.items() if k.startswith("positive:")]
        negative = [mean(v) for k, v in days.items() if k.startswith("nonpositive:")]
        effects.append(
            {
                "horizon": horizon,
                "model_hash": model,
                "config_hash": config,
                "positive_days": len(positive),
                "nonpositive_days": len(negative),
                "positive_mean": mean(positive) if positive else None,
                "nonpositive_mean": mean(negative) if negative else None,
            }
        )
    status = (
        "INCOMPLETE_OUTCOMES"
        if missing
        else ("DESCRIPTIVE_ONLY" if effects else "INSUFFICIENT_DATA")
    )
    return {
        "status": status,
        "eligible_rows": eligible,
        "missing_rows": missing,
        "effects": effects,
        "formal_coverage_claim": False,
    }


# These are research contrasts, not permission to bypass a live gate.
# REJECT reasons are short-circuited before WATCH checks, so their outcomes
# can describe a technical filter but cannot establish actionable status.
_REPLAY_GATES = (
    "lcb_ev_not_positive",
    "spread_too_high",
    "leverage_out_of_range",
    "barrier_distance_too_small",
    "no_confirmed_alpha_source",
    "suggested_size_zero",
)
_NON_REPLAY_GATES = (
    "data_quality",
    "bid_only",
    "knocked_out",
    "quote_timestamp_missing",
    "source_quote_stale",
    "quote_age_at_decision",
    "no_live_quote",
    "no_ask_quote",
    "lcb_ev_not_evaluated",
    "p_ko_not_evaluated",
    "cluster_risk_not_confirmed",
    "ratio_unverified",
    "portfolio_risk_not_confirmed",
)


def _gate_effect(
    gate: str, rows: Sequence[tuple[LedgerEntry, LedgerLabel | None]], *, as_of: datetime
) -> dict[str, object]:
    cohorts: dict[tuple[int, str, str, str, str, str], list[float]] = defaultdict(list)
    affected = missing = shadow = 0
    for entry, label in rows:
        flags = entry.feature_snapshot
        if (
            entry.category not in (Category.WATCH, Category.REJECT)
            or flags.get(f"gate_reason:{gate}") != 1.0
            or flags.get("gate_failure_count") != 1.0
        ):
            continue
        affected += 1
        shadow += int(entry.is_shadow)
        if (
            label is None
            or label.labeled_at > as_of
            or label.ambiguous_path
            or label.exit_reason not in (ExitReason.HORIZON, ExitReason.KO)
            or label.realized_selected_pnl is None
            or not isfinite(label.realized_selected_pnl)
            or label.realized_selected_pnl < -1
        ):
            missing += 1
            continue
        key = (
            entry.horizon_days,
            entry.model_hash,
            entry.config_hash,
            entry.underlying,
            entry.direction.value,
            entry.prediction_time.date().isoformat(),
        )
        cohorts[key].append(label.realized_selected_pnl)

    if not rows:
        status = "NO_ARCHIVED_GATE_INPUTS"
    elif missing:
        status = "INCOMPLETE_OUTCOMES"
    elif not affected:
        status = "NO_AFFECTED_ROWS"
    else:
        status = "DESCRIPTIVE_ONLY"
    cohort_means = [mean(v) for v in cohorts.values()]
    observed = mean(cohort_means) if cohort_means and not missing else None
    scenarios = [
        {
            "scenario": name,
            "notional_eur": notional,
            "net_return_vs_no_trade": (
                costs.net_return(observed, notional) if observed is not None else None
            ),
        }
        for name, costs in COST_SCENARIOS.items()
        for notional in NOTIONALS_EUR
    ]
    return {
        "gate": gate,
        "status": status,
        "affected_rows": affected,
        "shadow_rows": shadow,
        "missing_outcomes": missing,
        "observed_cohorts": len(cohorts),
        "candidate_return_vs_no_trade": observed,
        "cost_scenarios": scenarios,
        "later_gates_may_be_unobserved": gate in (
            "spread_too_high",
            "leverage_out_of_range",
            "barrier_distance_too_small",
        ),
    }


def evaluate_gate_replay(
    rows: Sequence[tuple[LedgerEntry, LedgerLabel | None]], *, as_of: datetime
) -> dict[str, object]:
    """Single-reason candidate contrasts; no inference of policy causality."""
    eligible = [
        (entry, label)
        for entry, label in rows
        if entry.exit_due < as_of.date() and entry.prediction_time <= as_of
    ]
    archived = [
        (entry, label)
        for entry, label in eligible
        if entry.feature_snapshot.get("gate_archive_v1") == 1.0
    ]
    effects = [_gate_effect(gate, archived, as_of=as_of) for gate in _REPLAY_GATES]
    if not archived:
        status = "NO_ARCHIVED_GATE_INPUTS"
    elif any(effect["status"] == "INCOMPLETE_OUTCOMES" for effect in effects):
        status = "INCOMPLETE_OUTCOMES"
    elif all(effect["status"] == "NO_AFFECTED_ROWS" for effect in effects):
        status = "NO_AFFECTED_ROWS"
    else:
        status = "DESCRIPTIVE_ONLY"
    return {
        "status": status,
        "archived_rows": len(archived),
        "legacy_rows": len(eligible) - len(archived),
        "effects": effects,
        "non_replayed_safety_or_missing_input_gates": list(_NON_REPLAY_GATES),
        "promotion_allowed": False,
        "cost_basis": "hypothetical_execution_costs_on_observed_bid_ask",
        "scope": "archived_candidates_only_not_full_policy_replay",
        "statistical_significance_claimed": False,
    }
