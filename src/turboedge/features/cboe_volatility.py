"""Cboe volatility-state features (Workstream W12-A).

Ten causal features over the six Cboe series, all computed strictly from
observations whose ``available_at <= prediction_time``. Nothing here reads a
raw series directly: every input goes through
``features.availability.select_available`` first, so a future close cannot
enter a feature even if the caller hands one in.

The ratios are deliberately expressed as ``x/y - 1`` rather than ``x/y``.
Both carry identical information, but centring on zero means a regularized
linear model's shrinkage-toward-zero is shrinkage toward "no term-structure
signal" rather than toward "the short end is worth nothing", and it makes
the sign directly readable: positive = short-dated volatility richer than
long-dated, the classic stress configuration.

Relation to W9: `models/challengers.py::VixTermStructure` already used two
of these (`vix9d_over_vix_minus_1`, `vix_over_vix3m_minus_1`) via yfinance
and was measured `dormant` -- best edge 1.85 bp against a 10 bp bar and a
50-150 bp cost band (docs/measured_results.md §2, trial W9-2026Q3-004).
This module is broader (VVIX/OVX/GVZ, levels, changes, z-scores, regimes)
and sourced officially, but that prior negative result is the honest prior
for what to expect here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

import numpy as np

from turboedge.features.availability import select_available
from turboedge.storage.schemas import ExternalObservation

#: Series the builder needs. A missing one yields `None` for the features
#: that depend on it -- never a substituted value (CLAUDE.md rule 29).
REQUIRED_SERIES: tuple[str, ...] = (
    "VIX.CLOSE",
    "VIX9D.CLOSE",
    "VIX3M.CLOSE",
    "VVIX.CLOSE",
    "OVX.CLOSE",
    "GVZ.CLOSE",
)

FEATURE_NAMES: tuple[str, ...] = (
    "vix_level",
    "vix_change_5d",
    "vix_zscore_20d",
    "vix9d_over_vix_minus_1",
    "vix_over_vix3m_minus_1",
    "vvix_over_vix_minus_1",
    "term_structure_slope",
    "term_structure_curvature",
    "volatility_regime",
    "vol_of_vol_regime",
)

_ZSCORE_WINDOW = 20
_CHANGE_WINDOW = 5
_REGIME_WINDOW = 252


def _series_history(observations: Sequence[ExternalObservation], series_id: str) -> list[float]:
    """Chronological values of one series, latest last.

    Caller must already have filtered by availability; this only groups.
    """
    return [o.value for o in observations if o.series_id == series_id]


def _latest(history: Sequence[float]) -> float | None:
    return float(history[-1]) if history else None


def _ratio_minus_one(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator - 1.0


def _zscore(history: Sequence[float], window: int) -> float | None:
    """Z-score of the latest value within its own trailing window.

    Returns ``None`` rather than 0.0 when the window is degenerate (constant
    series, too few points): 0.0 would read as "exactly average", a real and
    quite different statement from "not computable".
    """
    if len(history) < window:
        return None
    tail = np.asarray(history[-window:], dtype=np.float64)
    sd = float(tail.std(ddof=1))
    if not (sd > 0.0) or not math.isfinite(sd):
        return None
    return float((tail[-1] - tail.mean()) / sd)


def _percentile_rank(history: Sequence[float], window: int) -> float | None:
    """Rank of the latest value within its trailing window, in [0, 1].

    Used for the two regime features. A percentile is preferred over a
    threshold because volatility levels are not comparable across years --
    a VIX of 20 meant something different in 2017 than in 2022 -- while its
    rank within a trailing year is.
    """
    if len(history) < window:
        return None
    tail = np.asarray(history[-window:], dtype=np.float64)
    latest = tail[-1]
    return float((tail <= latest).mean())


def build_cboe_features(
    observations: Sequence[ExternalObservation],
    prediction_time: datetime,
) -> dict[str, float | None]:
    """The ten W12-A features as of ``prediction_time``.

    Every value is ``None`` when its inputs are unavailable -- a missing
    volatility print is missing information, never zero and never
    forward-filled (CLAUDE.md rule 29). Downstream models must handle
    ``None`` explicitly rather than receiving a silently imputed number.
    """
    available = select_available(observations, prediction_time)

    vix = _series_history(available, "VIX.CLOSE")
    vix9d = _series_history(available, "VIX9D.CLOSE")
    vix3m = _series_history(available, "VIX3M.CLOSE")
    vvix = _series_history(available, "VVIX.CLOSE")

    vix_now = _latest(vix)
    vix9d_now = _latest(vix9d)
    vix3m_now = _latest(vix3m)
    vvix_now = _latest(vvix)

    short_leg = _ratio_minus_one(vix9d_now, vix_now)
    long_leg = _ratio_minus_one(vix_now, vix3m_now)

    change_5d: float | None = None
    if len(vix) > _CHANGE_WINDOW and vix[-1 - _CHANGE_WINDOW] > 0.0:
        change_5d = vix[-1] / vix[-1 - _CHANGE_WINDOW] - 1.0

    # Slope: 9-day vs 3-month, the full term structure in one number
    # (positive = inverted = stressed). Curvature: how much the 30-day point
    # deviates from the straight line between the two ends -- a hump the two
    # individual legs cannot express, and the one quantity here that W9's
    # two-feature version could not see at all.
    slope = _ratio_minus_one(vix9d_now, vix3m_now)
    curvature: float | None = None
    if short_leg is not None and long_leg is not None:
        curvature = short_leg - long_leg

    return {
        "vix_level": vix_now,
        "vix_change_5d": change_5d,
        "vix_zscore_20d": _zscore(vix, _ZSCORE_WINDOW),
        "vix9d_over_vix_minus_1": short_leg,
        "vix_over_vix3m_minus_1": long_leg,
        "vvix_over_vix_minus_1": _ratio_minus_one(vvix_now, vix_now),
        "term_structure_slope": slope,
        "term_structure_curvature": curvature,
        "volatility_regime": _percentile_rank(vix, _REGIME_WINDOW),
        "vol_of_vol_regime": _percentile_rank(vvix, _REGIME_WINDOW),
    }


__all__ = [
    "FEATURE_NAMES",
    "REQUIRED_SERIES",
    "build_cboe_features",
]
