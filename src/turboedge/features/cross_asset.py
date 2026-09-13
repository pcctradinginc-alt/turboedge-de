"""Cross-asset lead/lag alignment: auxiliary-series features with no look-ahead.

Formula reference: Master Spec §8.6 ("Cross-Asset Lead/Lag"); Build Contract
v2 W9 item. A cross-asset feature (e.g. "yesterday's S&P 500 close" as a
predictor for today's DAX forecast) must only ever use an auxiliary bar whose
own ``available_at`` is at or before the primary underlying's prediction time
(CLAUDE.md rule 5). Because different exchanges close at different wall-clock
times, "yesterday" vs. "today" is not a fixed offset -- it falls straight out
of comparing each bar's own ``available_at`` timestamp, which
``turboedge.adapters.fallback_prices._normalize_history`` already computes
per-underlying (equity session close in the underlying's own timezone, or a
conservative "next calendar day" rule for 24h markets). This module never
computes or overrides ``available_at`` itself; it only aligns already-dated
series onto a primary bar axis using each side's own timestamps.

Every function here is a pure, causal transform: ``out[t]`` (for primary bar
``t``) only ever reads auxiliary entries with ``available_at <=
primary_bars[t].available_at``, so truncating either the primary or the
auxiliary sequence to an earlier prefix never changes an already-computed
value (mirrors ``features/returns.py``'s causality contract, and is the
property that lets ``models/challengers.py`` compute one aligned series over
full history and safely slice it to any as-of prefix rather than realigning
per walk-forward fold -- see ``tests/features/test_cross_asset.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from turboedge.storage.schemas import UnderlyingBar


def _available_at_timestamps(bars: Sequence[UnderlyingBar]) -> npt.NDArray[np.float64]:
    return np.array([b.available_at.timestamp() for b in bars], dtype=np.float64)


def align_auxiliary_series(
    primary_bars: Sequence[UnderlyingBar],
    auxiliary_bars: Sequence[UnderlyingBar],
    aux_values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """For each primary bar, the most recent ``aux_values`` entry available in time.

    ``aux_values`` must be one value per ``auxiliary_bars`` entry (same
    order, not necessarily the same length as ``primary_bars``) -- e.g. the
    auxiliary series' own 1-day log return or a level. For primary bar ``t``,
    returns ``aux_values[j]`` for the largest ``j`` with
    ``auxiliary_bars[j].available_at <= primary_bars[t].available_at``, or
    ``NaN`` if no such ``j`` exists (auxiliary history starts later than
    ``t``). Ties in ``available_at`` are included (``<=``, not ``<``).

    Neither ``primary_bars`` nor ``auxiliary_bars`` need be pre-sorted by
    ``available_at``; both are sorted internally by this function. This is
    the single no-look-ahead primitive every cross-asset feature in
    ``models/challengers.py`` is built from.
    """
    if len(aux_values) != len(auxiliary_bars):
        raise ValueError(
            f"aux_values must have one entry per auxiliary_bars, got "
            f"{len(aux_values)} values for {len(auxiliary_bars)} bars"
        )
    n_primary = len(primary_bars)
    if n_primary == 0:
        return np.zeros(0, dtype=np.float64)
    if len(auxiliary_bars) == 0:
        return np.full(n_primary, np.nan, dtype=np.float64)

    aux_avail = _available_at_timestamps(auxiliary_bars)
    aux_vals = np.asarray(aux_values, dtype=np.float64)
    order = np.argsort(aux_avail, kind="stable")
    aux_avail_sorted = aux_avail[order]
    aux_vals_sorted = aux_vals[order]

    primary_avail = _available_at_timestamps(primary_bars)
    # Rightmost insertion point among aux entries <= target -> index of the
    # last aux entry with available_at <= primary_avail[t], or -1 if none.
    idx = np.searchsorted(aux_avail_sorted, primary_avail, side="right") - 1

    out = np.full(n_primary, np.nan, dtype=np.float64)
    valid = idx >= 0
    out[valid] = aux_vals_sorted[idx[valid]]
    return out


def own_log_return_1d(bars: Sequence[UnderlyingBar]) -> npt.NDArray[np.float64]:
    """``out[i] = ln(close[i] / close[i-1])`` for an auxiliary series, aligned to ``bars``.

    ``out[0]`` is ``NaN`` (no prior bar). This is the per-entry ``aux_values``
    array typically passed into :func:`align_auxiliary_series` for a
    "yesterday's return" style feature.
    """
    closes = np.array([b.close for b in bars], dtype=np.float64)
    out = np.full(closes.shape[0], np.nan, dtype=np.float64)
    if closes.shape[0] > 1:
        out[1:] = np.log(closes[1:] / closes[:-1])
    return out


def own_level_diff_1d(bars: Sequence[UnderlyingBar]) -> npt.NDArray[np.float64]:
    """``out[i] = close[i] - close[i-1]`` for a level series (e.g. a yield index like ``^TNX``).

    Used instead of :func:`own_log_return_1d` for series where a simple log
    return is not the economically meaningful unit (an interest-rate level
    can be near zero, and what matters for a rates-move feature is the
    absolute daily change, not its ratio).
    """
    closes = np.array([b.close for b in bars], dtype=np.float64)
    out = np.full(closes.shape[0], np.nan, dtype=np.float64)
    if closes.shape[0] > 1:
        out[1:] = closes[1:] - closes[:-1]
    return out


def own_level(bars: Sequence[UnderlyingBar]) -> npt.NDArray[np.float64]:
    """The auxiliary series' own close level, one entry per bar (no transform)."""
    return np.array([b.close for b in bars], dtype=np.float64)


__all__ = [
    "align_auxiliary_series",
    "own_level",
    "own_level_diff_1d",
    "own_log_return_1d",
]
