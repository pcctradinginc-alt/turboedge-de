"""Counterfactual learning: evaluate a ledger entry's discarded alternatives
under the exact same exit rules as the selected product (Master Spec §24).

Not only the chosen product's outcome matters -- for the same underlying
signal, at the same horizon, a handful of similar-but-different products
(same direction, similar leverage/barrier, a different issuer) are also
evaluated after the fact, so it stays visible whether an edge came from
direction/timing (the signal), from wrapper selection (issuer/spread/
financing), or from luck (Master Spec §24).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from turboedge.learning.labeler import LabelerConfig, resolve_exit_with_fallback
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import LedgerEntry, ProductType, UnderlyingBar


@dataclass(frozen=True)
class CounterfactualResult:
    """Master Spec §24 metrics for one ledger entry.

    ``n_evaluated`` counts only alternatives for which *some* realized P&L
    could be resolved (an alternative with genuinely no data at all --
    ``EXPIRED_NO_DATA`` -- is excluded, never forced to a fabricated value);
    it can be 0 if `entry.alternatives` is empty or none of them had data.
    """

    median_turbo_pnl: float | None
    best_turbo_pnl: float | None
    n_evaluated: int
    product_selection_edge: float | None  # selected - median
    regret: float | None  # best - selected


def _alternative_entry_terms(
    store: Store, isin: str, prediction_time: datetime
) -> tuple[float, float | None, float, ProductType | None] | None:
    """Reconstruct one alternative ISIN's own entry terms as of
    ``prediction_time`` (never after -- CLAUDE.md rule 5, no look-ahead):
    ``(entry_ask, barrier, spread, product_type)``, or ``None`` if no
    snapshot at/before that time exists for it at all."""
    snapshot = store.latest_product_snapshot_at_or_before(isin, prediction_time)
    if snapshot is None or snapshot.ask is None:
        return None
    spread = (
        (snapshot.ask - snapshot.bid) / snapshot.ask
        if snapshot.bid is not None and snapshot.ask
        else 0.0
    )
    instrument = store.get_instrument(isin)
    product_type = instrument.product_type if instrument is not None else None
    return snapshot.ask, snapshot.knockout_barrier, spread, product_type


def evaluate_counterfactual(
    store: Store,
    entry: LedgerEntry,
    *,
    underlying_bars: Sequence[UnderlyingBar],
    selected_realized_pnl: float | None,
    config: LabelerConfig | None = None,
) -> CounterfactualResult:
    """Evaluate `entry.alternatives` under the same exit rules as the
    selected product (`learning/labeler.py`'s `resolve_exit_with_fallback`),
    and compute Master Spec §24's Product Selection Edge / Regret.

    Args:
        store: Open ``Store`` (queried for each alternative's own snapshot
            history, same as for the selected product).
        entry: The ledger entry whose ``alternatives`` (ISINs) to evaluate.
        underlying_bars: Bars for `entry.underlying`, fetched once by the
            caller (shared with the selected product's own resolution,
            since alternatives share the same underlying by construction).
        selected_realized_pnl: The selected product's own realized net
            return (`LedgerLabel.realized_selected_pnl`), used to compute
            ``product_selection_edge``/``regret``. ``None`` propagates to
            ``None`` results (no meaningful edge/regret without it).
        config: Exit-resolution config; defaults to `LabelerConfig()`.

    Returns:
        The computed `CounterfactualResult`.
    """
    realized_returns: list[float] = []
    for isin in entry.alternatives:
        terms = _alternative_entry_terms(store, isin, entry.prediction_time)
        if terms is None:
            continue
        alt_entry_ask, alt_barrier, alt_spread, alt_product_type = terms
        resolution = resolve_exit_with_fallback(
            store,
            isin=isin,
            direction=entry.direction,
            entry_ask=alt_entry_ask,
            entry_bid=None,
            entry_spread=alt_spread,
            entry_quote_timestamp=entry.entry_quote_timestamp,
            entry_underlying_date=entry.entry_underlying_timestamp.date(),
            exit_due=entry.exit_due,
            barrier=alt_barrier,
            underlying_bars=underlying_bars,
            product_type=alt_product_type,
            config=config,
        )
        if resolution.realized_pnl is not None:
            realized_returns.append(resolution.realized_pnl)

    if not realized_returns:
        return CounterfactualResult(
            median_turbo_pnl=None,
            best_turbo_pnl=None,
            n_evaluated=0,
            product_selection_edge=None,
            regret=None,
        )

    median_pnl = statistics.median(realized_returns)
    best_pnl = max(realized_returns)
    edge = selected_realized_pnl - median_pnl if selected_realized_pnl is not None else None
    regret = best_pnl - selected_realized_pnl if selected_realized_pnl is not None else None

    return CounterfactualResult(
        median_turbo_pnl=median_pnl,
        best_turbo_pnl=best_pnl,
        n_evaluated=len(realized_returns),
        product_selection_edge=edge,
        regret=regret,
    )


__all__ = ["CounterfactualResult", "evaluate_counterfactual"]
