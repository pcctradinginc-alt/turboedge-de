"""Model disagreement as information (Phase 1, M3).

Measures how far apart the ensemble's members are, on the axes that matter
for a knock-out product: direction, central location, and the tails.

One caution is built into this module's design and should stay there:
**agreement is not evidence of correctness.** Five models built on related
features can agree precisely because they share a blind spot, and in W4/W9
this system's models agreed with each other while all losing to the null
model. So these numbers are reported as uncertainty inputs, and whether low
dispersion actually predicts better outcomes is an empirical question this
layer records data for rather than assumes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from turboedge.models.forecast import HorizonForecast

_QUANTILE_KEYS = ("q05", "q25", "q50", "q75", "q95")
_TAIL_KEYS = ("q05", "q95")


def _normalising_scale(forecasts: Sequence[HorizonForecast]) -> float:
    """Median predictive sigma, used to express dispersion in sigma units.

    Raw dispersion is not comparable across horizons or underlyings -- a
    0.01 spread is large at a 3-day horizon and small at 14. Falls back to
    1.0 when every sigma is degenerate, which keeps the output finite and
    lets the caller see an undiluted raw spread rather than a division blow-up.
    """
    sigmas = [f.sigma for f in forecasts if f.sigma > 0]
    return float(np.median(sigmas)) if sigmas else 1.0


def forecast_dispersion(forecasts: Sequence[HorizonForecast]) -> float:
    """Spread of the point forecasts, in units of predictive sigma."""
    if len(forecasts) < 2:
        return 0.0
    means = np.asarray([f.mean for f in forecasts], dtype=np.float64)
    return float(means.std(ddof=1) / _normalising_scale(forecasts))


def direction_disagreement(forecasts: Sequence[HorizonForecast]) -> float:
    """Fraction of models on the minority side of the up/down call.

    0.0 = unanimous, 0.5 = an even split. Uses ``p_up`` rather than the sign
    of the mean: a model that says 51% up is expressing a far weaker view
    than one saying 80%, and the direction call is the one this maps.
    """
    if len(forecasts) < 2:
        return 0.0
    up = sum(1 for f in forecasts if f.p_up > 0.5)
    minority = min(up, len(forecasts) - up)
    return float(minority / len(forecasts))


def quantile_disagreement(forecasts: Sequence[HorizonForecast]) -> float:
    """Mean across-model spread of the five quantiles, in sigma units."""
    usable = [f for f in forecasts if all(k in f.quantiles for k in _QUANTILE_KEYS)]
    if len(usable) < 2:
        return 0.0
    scale = _normalising_scale(usable)
    spreads = [
        float(np.asarray([f.quantiles[k] for f in usable], dtype=np.float64).std(ddof=1))
        for k in _QUANTILE_KEYS
    ]
    return float(np.mean(spreads) / scale)


def tail_risk_disagreement(forecasts: Sequence[HorizonForecast]) -> float:
    """Disagreement in the tails only (q05/q95), in sigma units.

    Kept separate from `quantile_disagreement` because for a knock-out
    product the tails are the product: an error in the 5th percentile
    decides whether the barrier is hit, while the same error at the median
    barely moves the payoff.
    """
    usable = [f for f in forecasts if all(k in f.quantiles for k in _TAIL_KEYS)]
    if len(usable) < 2:
        return 0.0
    scale = _normalising_scale(usable)
    spreads = [
        float(np.asarray([f.quantiles[k] for f in usable], dtype=np.float64).std(ddof=1))
        for k in _TAIL_KEYS
    ]
    return float(np.mean(spreads) / scale)


def aggregate_disagreement(forecasts: Sequence[HorizonForecast]) -> float:
    """One bounded [0, 1] summary for the uncertainty vector.

    Direction disagreement is doubled because its natural maximum is 0.5;
    the dispersion measures are squashed by ``x/(1+x)`` so that an extreme
    outlier cannot dominate the aggregate. Tail disagreement carries the
    heaviest weight for the reason given above.
    """
    if len(forecasts) < 2:
        return 0.0
    direction = direction_disagreement(forecasts) * 2.0
    dispersion = forecast_dispersion(forecasts)
    quantile = quantile_disagreement(forecasts)
    tail = tail_risk_disagreement(forecasts)
    squashed = [x / (1.0 + x) for x in (dispersion, quantile, tail)]
    combined = 0.25 * direction + 0.20 * squashed[0] + 0.20 * squashed[1] + 0.35 * squashed[2]
    return float(min(max(combined, 0.0), 1.0))


__all__ = [
    "aggregate_disagreement",
    "direction_disagreement",
    "forecast_dispersion",
    "quantile_disagreement",
    "tail_risk_disagreement",
]
