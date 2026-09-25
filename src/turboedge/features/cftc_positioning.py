"""CFTC positioning features (Workstream W12-D).

Six causal features per (market, trader group), computed only from
observations whose ``available_at <= prediction_time`` -- the same guard
every W12 family routes through (`features/availability.py`).

Two design points worth stating, because both are places a positioning
feature usually goes wrong:

*Normalise by open interest, not raw contracts.* A net position of -366,841
contracts means nothing on its own: the S&P 500 e-mini book has grown by
orders of magnitude since 2010, so a raw net series is dominated by market
growth rather than by positioning. Every level feature here is a share of
open interest.

*Weekly data, daily predictions.* The report changes once a week, so between
two as-of dates every feature is constant by construction. That is correct
(it is genuinely the latest known positioning) but it means these are regime
features, not timing features -- and the effective sample is the number of
weeks, not the number of days. Stated here so nobody reads the daily row
count as independent evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from turboedge.features.availability import select_available
from turboedge.storage.schemas import ExternalObservation

GROUPS: tuple[str, ...] = ("dealer", "asset_mgr", "lev_money")

FEATURE_SUFFIXES: tuple[str, ...] = (
    "share_of_oi",
    "zscore_3y",
    "percentile_5y",
    "change_1w",
    "change_4w",
    "velocity",
    "acceleration",
)

_WEEKS_PER_YEAR = 52
_ZSCORE_WINDOW = 3 * _WEEKS_PER_YEAR
_PERCENTILE_WINDOW = 5 * _WEEKS_PER_YEAR


def _history(observations: Sequence[ExternalObservation], series_id: str) -> list[float]:
    return [o.value for o in observations if o.series_id == series_id]


def _zscore(history: Sequence[float], window: int) -> float | None:
    """``None`` rather than 0.0 on a degenerate window -- 0.0 reads as
    'exactly average', which is a real and different statement."""
    if len(history) < window:
        return None
    tail = np.asarray(history[-window:], dtype=np.float64)
    sd = float(tail.std(ddof=1))
    if not (sd > 0.0) or not np.isfinite(sd):
        return None
    return float((tail[-1] - tail.mean()) / sd)


def _percentile(history: Sequence[float], window: int) -> float | None:
    if len(history) < window:
        return None
    tail = np.asarray(history[-window:], dtype=np.float64)
    return float((tail <= tail[-1]).mean())


def _change(history: Sequence[float], lag: int) -> float | None:
    if len(history) <= lag:
        return None
    return float(history[-1] - history[-1 - lag])


def build_cftc_features(
    observations: Sequence[ExternalObservation],
    prediction_time: datetime,
    *,
    market: str,
) -> dict[str, float | None]:
    """Positioning features for one market as of ``prediction_time``.

    Returns ``None`` for any feature whose inputs are unavailable -- never a
    substituted value (CLAUDE.md rule 29). ``crowding`` is expressed as the
    5-year percentile of the leveraged-money share: an extreme reading means
    that group is more one-sided than in 95% of the last five years, which is
    what 'crowded' is supposed to mean.
    """
    available = select_available(observations, prediction_time)
    prefix = market.upper()
    oi = _history(available, f"{prefix}.OPEN_INTEREST")
    out: dict[str, float | None] = {}

    for group in GROUPS:
        net = _history(available, f"{prefix}.{group.upper()}_NET")
        # Share of open interest, aligned pairwise: both series come from the
        # same rows, so a length mismatch means one leg was skipped for a
        # week and the pairing would silently shift.
        n = min(len(net), len(oi))
        share = [
            net[len(net) - n + i] / oi[len(oi) - n + i] for i in range(n) if oi[len(oi) - n + i] > 0
        ]
        base = f"cftc_{prefix.lower()}_{group}"
        out[f"{base}_share_of_oi"] = float(share[-1]) if share else None
        out[f"{base}_zscore_3y"] = _zscore(share, _ZSCORE_WINDOW)
        out[f"{base}_percentile_5y"] = _percentile(share, _PERCENTILE_WINDOW)
        out[f"{base}_change_1w"] = _change(share, 1)
        out[f"{base}_change_4w"] = _change(share, 4)
        # Velocity is the 1-week change; acceleration is its own change, i.e.
        # whether the move is speeding up or fading. Both on the normalised
        # share, so neither tracks book growth.
        velocity = _change(share, 1)
        prev_velocity = _change(share[:-1], 1) if len(share) > 2 else None
        out[f"{base}_velocity"] = velocity
        out[f"{base}_acceleration"] = (
            velocity - prev_velocity if velocity is not None and prev_velocity is not None else None
        )

    return out


def feature_names_for(market: str) -> list[str]:
    prefix = market.lower()
    return [f"cftc_{prefix}_{g}_{s}" for g in GROUPS for s in FEATURE_SUFFIXES]


__all__ = [
    "FEATURE_SUFFIXES",
    "GROUPS",
    "build_cftc_features",
    "feature_names_for",
]
