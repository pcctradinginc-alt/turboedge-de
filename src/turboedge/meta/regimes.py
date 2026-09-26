"""Regime classification for the meta layer (Phase 1).

Deliberately coarse -- three volatility buckets and three trend buckets --
for a reason that matters more than the choice of thresholds: the meta layer
asks "how often have we seen a situation like this before?", and that
question is only answerable if the buckets are wide enough to accumulate
observations. Nine cells over ~4,000 daily bars average ~440 each; ninety
cells would average ~44, and every regime would look unfamiliar forever.

Both classifiers are causal: they read only bars at or before
``prediction_time``. The volatility bucket reuses the same
quantile-against-own-history idea as
``simulation/ko_calibration._regime_bucket`` rather than inventing a second
convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from turboedge.storage.schemas import UnderlyingBar

VOLATILITY_REGIMES: tuple[str, ...] = ("low_vol", "mid_vol", "high_vol")
TREND_REGIMES: tuple[str, ...] = ("downtrend", "range", "uptrend")
UNKNOWN_REGIME = "unknown"

_MIN_HISTORY = 126
_TREND_LOOKBACK = 63
_VOL_LOOKBACK = 21
#: Trend is scored in units of its own volatility, so the cut is comparable
#: across underlyings and across calm/turbulent periods alike.
_TREND_CUT_SIGMA = 0.5


def _causal_closes(bars: Sequence[UnderlyingBar], prediction_time: datetime) -> np.ndarray:
    """Closes at or before ``prediction_time``, chronological.

    The filter is on the bar's own ``available_at`` where present, falling
    back to its timestamp -- the same rule the rest of the system uses, so a
    bar that had not been published yet cannot classify the regime it
    belongs to.
    """
    usable = [
        b
        for b in bars
        if (b.available_at if b.available_at is not None else b.ts) <= prediction_time
    ]
    usable.sort(key=lambda b: b.ts)
    return np.asarray([b.close for b in usable], dtype=np.float64)


def classify_volatility_regime(bars: Sequence[UnderlyingBar], prediction_time: datetime) -> str:
    """Terciles of realised vol against the underlying's own history.

    Against its OWN history, not a fixed level: a VIX of 20 meant something
    different in 2017 than in 2022, and the same is true of realised
    volatility. Returns ``unknown`` rather than guessing when history is too
    short -- a regime nobody can classify is exactly the situation the
    abstention path is for.
    """
    closes = _causal_closes(bars, prediction_time)
    if len(closes) < _MIN_HISTORY:
        return UNKNOWN_REGIME
    rets = np.diff(np.log(closes))
    if len(rets) < _VOL_LOOKBACK * 2:
        return UNKNOWN_REGIME
    rolling = np.asarray(
        [rets[i - _VOL_LOOKBACK : i].std(ddof=1) for i in range(_VOL_LOOKBACK, len(rets) + 1)]
    )
    rolling = rolling[np.isfinite(rolling)]
    if len(rolling) < 3 or not (rolling.std() > 0):
        return UNKNOWN_REGIME
    current = rolling[-1]
    low, high = np.quantile(rolling, [1 / 3, 2 / 3])
    if current <= low:
        return "low_vol"
    if current >= high:
        return "high_vol"
    return "mid_vol"


def classify_trend_regime(bars: Sequence[UnderlyingBar], prediction_time: datetime) -> str:
    """Volatility-normalised 63-day drift, cut at +/- 0.5 sigma."""
    closes = _causal_closes(bars, prediction_time)
    if len(closes) < _MIN_HISTORY:
        return UNKNOWN_REGIME
    logp = np.log(closes)
    rets = np.diff(logp)
    sigma = float(rets[-_TREND_LOOKBACK:].std(ddof=1))
    if not (sigma > 0) or not np.isfinite(sigma):
        return UNKNOWN_REGIME
    drift = float(logp[-1] - logp[-1 - _TREND_LOOKBACK])
    score = drift / (sigma * np.sqrt(_TREND_LOOKBACK))
    if score <= -_TREND_CUT_SIGMA:
        return "downtrend"
    if score >= _TREND_CUT_SIGMA:
        return "uptrend"
    return "range"


def count_similar_history(
    bars: Sequence[UnderlyingBar],
    prediction_time: datetime,
    volatility_regime: str,
    trend_regime: str,
    *,
    step: int = 5,
) -> int:
    """How many past days fell in this same (vol, trend) cell.

    This is the number that makes "unfamiliar regime" checkable instead of
    rhetorical. Sampled every ``step`` bars -- consecutive days are nearly
    the same situation, so counting all of them would inflate familiarity
    without adding evidence.

    Returns 0 for an unknown regime: unclassifiable is not the same as rare,
    but both warrant the same caution here.
    """
    if volatility_regime == UNKNOWN_REGIME or trend_regime == UNKNOWN_REGIME:
        return 0
    usable = [
        b
        for b in bars
        if (b.available_at if b.available_at is not None else b.ts) <= prediction_time
    ]
    usable.sort(key=lambda b: b.ts)
    count = 0
    for end in range(_MIN_HISTORY, len(usable), step):
        window = usable[:end]
        as_of = window[-1].ts
        if classify_volatility_regime(window, as_of) == volatility_regime and (
            classify_trend_regime(window, as_of) == trend_regime
        ):
            count += 1
    return count


__all__ = [
    "TREND_REGIMES",
    "UNKNOWN_REGIME",
    "VOLATILITY_REGIMES",
    "classify_trend_regime",
    "classify_volatility_regime",
    "count_similar_history",
]
