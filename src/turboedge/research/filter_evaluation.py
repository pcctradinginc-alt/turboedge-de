"""Observed ledger policy diagnostics, not a replay of unavailable candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from math import isfinite
from statistics import mean

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


def evaluate_gate_replay(
    rows: Sequence[tuple[LedgerEntry, LedgerLabel | None]], *, as_of: datetime
) -> dict[str, object]:
    """Candidate-level LCB gate counterfactual, not a portfolio replay.

    Only archived final WATCH decisions with a sole LCB failure qualify.
    Earlier reject branches can hide later failures and are not replayed.
    """
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
    day_returns: dict[tuple[int, str, str, str], list[float]] = defaultdict(list)
    affected = missing = 0
    for entry, label in archived:
        flags = entry.feature_snapshot
        if (
            entry.category != Category.WATCH
            or flags.get("gate_reason:lcb_ev_not_positive") != 1.0
            or flags.get("gate_failure_count") != 1.0
        ):
            continue
        affected += 1
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
            entry.horizon_days,
            entry.model_hash,
            entry.config_hash,
            entry.prediction_time.date().isoformat(),
        )
        day_returns[key].append(label.realized_selected_pnl)
    if not archived:
        status = "NO_ARCHIVED_GATE_INPUTS"
    elif missing:
        status = "INCOMPLETE_OUTCOMES"
    elif not affected:
        status = "NO_AFFECTED_ROWS"
    else:
        status = "DESCRIPTIVE_ONLY"
    return {
        "status": status,
        "gate": "lcb_ev_not_positive",
        "archived_rows": len(archived),
        "legacy_rows": len(eligible) - len(archived),
        "affected_rows": affected,
        "missing_outcomes": missing,
        "observed_cohorts": len(day_returns),
        "candidate_return_vs_no_trade": (
            mean(mean(values) for values in day_returns.values())
            if day_returns and not missing
            else None
        ),
        "promotion_allowed": False,
        "cost_basis": "bid_ask_proxy_before_broker_costs",
        "scope": "archived_candidates_only_not_full_policy_replay",
    }
