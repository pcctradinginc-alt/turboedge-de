"""Observed ledger policy diagnostics, not a replay of unavailable candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from math import isfinite

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
        effects.append({
            "horizon": horizon,
            "matched_day_underlyings": len(matched),
            "watch_minus_actionable": (
                sum(b - a for a, b in matched) / len(matched) if matched else None
            ),
        })
    return {
        "status": "INCOMPLETE_OUTCOMES" if missing else "DESCRIPTIVE_ONLY",
        "eligible_rows": eligible,
        "missing_or_unverified_rows": missing,
        "observed_day_category_cohorts": len(cohorts),
        "effects": effects,
        "cost_basis": "observed_bid_ask_proxy_before_broker_costs",
        "causal_filter_ablation": False,
    }
