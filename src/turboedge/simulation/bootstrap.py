"""Block, weekend-conditioned and regime-conditioned resampling of daily components.

Formula reference: Master Spec §11.1.A ("Empirical block bootstrap") and
§11.1.B ("Regime-conditioned bootstrap"), §11.2 ("Overnight und Weekend
getrennt").

Block choice (documented, CLAUDE.md rule 33 -- no silent reoptimization):
this module implements a **fixed-length, circular block bootstrap** (each
block has exactly ``block_size`` consecutive historical trading days; blocks
wrap around the end of the historical sample so any ``block_size``/history
length combination is coverable), not the stationary bootstrap's
geometric-random block length. ``block_size=1`` degenerates to ordinary
i.i.d. bootstrap, which :mod:`turboedge.simulation.paths` relies on for its
Monte Carlo method's overnight-gap and intraday-range sampling.

Weekend handling: the historical daily series is split into a *weekday*
subsequence (``weekend_before == False``) and a *weekend* subsequence
(``weekend_before == True``, i.e. the day after a >1-calendar-day gap). The
simulation calendar's weekend positions (``paths.py``'s own
``weekend_before``, computed from the real Mon-Fri calendar) are always
filled by drawing *from the weekend subsequence*, independently of
``block_size`` and always i.i.d. (single-day draws): weekend transitions are
far too sparse in any realistic lookback window to form meaningful blocks of
consecutive weekend days, so blocking them would concentrate the whole
simulated tail risk on a handful of historical weekends. Weekday positions
use the ordinary block bootstrap over the weekday subsequence.

Regime conditioning: the "current regime" is determined by EWMA volatility of
the *weekday* total daily log return (``gap + intraday``) as of the most
recent historical day, bucketed into ``n_vol_buckets`` empirical-quantile
terciles (2 or 3, clamped). Weekday sampling is then restricted to days whose
own EWMA volatility falls in the same bucket, falling back to the full
weekday sample when that bucket holds fewer than ``min_bucket_days`` days
(too few for a stable resample). Weekend sampling is always drawn from the
*full* weekend subsequence regardless of regime, for the same sparsity reason
as above.

``min_bucket_days`` default (Build Contract v2 W5 follow-up 2, measured
2026-09-12 on 558 ^GDAXI start dates, ``scratchpad/w5_simulation_validation.md``):
at the default ``lookback_days=750`` (``paths.py``), a 3-way tercile split of
the weekday subsequence produces buckets of ~195-199 days on *every* date
tested -- a direct consequence of ``bucket_size ~= lookback_days *
weekday_fraction / n_buckets`` with ``weekday_fraction ~= 0.8``, essentially
constant regardless of which regime is "current". The previous default
(``min_bucket_days=250``) was therefore never satisfiable at
``lookback_days=750`` and silently fell back to the unconditioned sample on
100% of dates tested -- a dead code path, not a feature. Fixed here by
lowering the default to ``150`` (comfortably below the measured ~195-199
floor) rather than raising ``lookback_days``: that parameter is shared with
``block_bootstrap``/``monte_carlo`` in ``simulate_paths``, and enlarging it
would feed *more* regime-mixed history into precisely the methods whose bias
this validation attributes to unconditioned regime mixing in the first
place -- fixing regime_bootstrap that way would have come at the cost of
making the other methods' calibration worse, and would break the
apples-to-apples comparison between methods that all share one
``lookback_days``.

Vol-standardized bootstrap (Master Spec §11.1, Build Contract v2 W5 follow-up
2): :func:`vol_scaled_bootstrap_daily` conditions on the prevailing
volatility regime *without any bucket-size threshold at all* -- each
historical day's ``(gap, intraday, hi, lo)`` is first standardized by that
day's own *trailing* (lagged by one day, i.e. computed from strictly earlier
returns only -- CLAUDE.md rule 4/5, no look-ahead) EWMA volatility, turning
every historical day into a unit-scale "shock" ``z``; the same block/weekend
resampling as :func:`block_bootstrap_daily` is applied to these standardized
shocks, and the result is rescaled by the single current EWMA volatility (as
of the last historical day) before being returned. This uses the *entire*
historical sample for the shock *shape* (skew, kurtosis, block/weekend
structure) while anchoring the *scale* to today's regime -- sidestepping the
whole bucket-size/activation problem that regime_bootstrap has.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from turboedge.features.product import ewma_volatility
from turboedge.simulation.overnight import DailyComponents

_DEFAULT_BLOCK_SIZE = 5
_DEFAULT_N_VOL_BUCKETS = 3
_DEFAULT_MIN_BUCKET_DAYS = 150
_DEFAULT_EWMA_LAMBDA = 0.94
# Floor for the trailing standardization volatility in vol_scaled_bootstrap_daily,
# as a fraction of the sample's own median trailing vol -- guards against a
# near-zero EWMA estimate (e.g. a run of literally flat historical bars)
# producing an exploding standardized shock.
_MIN_TRAILING_VOL_FRACTION = 1e-3


@dataclass(frozen=True, slots=True)
class RegimeBootstrapConfig:
    """Parameters of the regime-conditioned bootstrap (§11.1.B)."""

    n_vol_buckets: int = _DEFAULT_N_VOL_BUCKETS
    min_bucket_days: int = _DEFAULT_MIN_BUCKET_DAYS
    ewma_lambda: float = _DEFAULT_EWMA_LAMBDA


@dataclass(frozen=True, slots=True)
class DailyDraws:
    """Bootstrapped daily components, shape ``(n_paths, horizon_days)`` each."""

    gap: npt.NDArray[np.float64]
    intraday: npt.NDArray[np.float64]
    hi: npt.NDArray[np.float64]
    lo: npt.NDArray[np.float64]


def _select(components: DailyComponents, mask: npt.NDArray[np.bool_]) -> DailyComponents:
    return DailyComponents(
        dates=[d for d, keep in zip(components.dates, mask, strict=True) if keep],
        gap=components.gap[mask],
        weekend_before=components.weekend_before[mask],
        intraday=components.intraday[mask],
        hi=components.hi[mask],
        lo=components.lo[mask],
        n_discarded=components.n_discarded,
    )


def _block_indices(
    n_hist: int, n_slots: int, block_size: int, n_paths: int, rng: np.random.Generator
) -> npt.NDArray[np.intp]:
    """Fixed-length circular block-bootstrap indices into an array of length ``n_hist``.

    Returns an ``(n_paths, n_slots)`` array of indices in ``[0, n_hist)``.
    ``block_size`` is clamped to ``[1, n_hist]``; ``block_size=1`` is i.i.d.
    bootstrap.
    """
    if n_hist <= 0:
        raise ValueError("n_hist must be > 0 to bootstrap from")
    if n_slots == 0:
        return np.empty((n_paths, 0), dtype=np.intp)
    bs = max(1, min(block_size, n_hist))
    n_blocks = -(-n_slots // bs)  # ceil division
    starts = rng.integers(0, n_hist, size=(n_paths, n_blocks))
    offsets = np.arange(bs)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n_hist
    return idx.reshape(n_paths, n_blocks * bs)[:, :n_slots].astype(np.intp)


def _draws_from_components(
    components: DailyComponents,
    weekend_before: npt.NDArray[np.bool_],
    n_paths: int,
    rng: np.random.Generator,
    block_size: int,
) -> DailyDraws:
    horizon_days = weekend_before.size
    weekend_hist_mask = components.weekend_before
    weekday_idx_hist = np.flatnonzero(~weekend_hist_mask)
    weekend_idx_hist = np.flatnonzero(weekend_hist_mask)

    n_weekend = int(weekend_before.sum())
    n_weekday = horizon_days - n_weekend

    out_gap = np.empty((n_paths, horizon_days), dtype=np.float64)
    out_intraday = np.empty((n_paths, horizon_days), dtype=np.float64)
    out_hi = np.empty((n_paths, horizon_days), dtype=np.float64)
    out_lo = np.empty((n_paths, horizon_days), dtype=np.float64)

    if n_weekday > 0:
        if weekday_idx_hist.size == 0:
            raise ValueError("no weekday daily samples available to bootstrap from")
        picks = _block_indices(weekday_idx_hist.size, n_weekday, block_size, n_paths, rng)
        src = weekday_idx_hist[picks]
        weekday_cols = ~weekend_before
        out_gap[:, weekday_cols] = components.gap[src]
        out_intraday[:, weekday_cols] = components.intraday[src]
        out_hi[:, weekday_cols] = components.hi[src]
        out_lo[:, weekday_cols] = components.lo[src]

    if n_weekend > 0:
        if weekend_idx_hist.size == 0:
            raise ValueError("no weekend daily samples available to bootstrap weekend gaps from")
        # Always i.i.d. (block_size=1): weekend transitions are too sparse
        # to form meaningful blocks (see module docstring).
        picks = _block_indices(weekend_idx_hist.size, n_weekend, 1, n_paths, rng)
        src = weekend_idx_hist[picks]
        out_gap[:, weekend_before] = components.gap[src]
        out_intraday[:, weekend_before] = components.intraday[src]
        out_hi[:, weekend_before] = components.hi[src]
        out_lo[:, weekend_before] = components.lo[src]

    return DailyDraws(gap=out_gap, intraday=out_intraday, hi=out_hi, lo=out_lo)


def block_bootstrap_daily(
    components: DailyComponents,
    weekend_before: npt.NDArray[np.bool_],
    n_paths: int,
    rng: np.random.Generator,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> DailyDraws:
    """Empirical block bootstrap (Master Spec §11.1.A) over ``components``.

    ``weekend_before`` is the *simulation calendar*'s weekend-gap flag,
    length ``horizon_days`` (see ``paths.py``): positions where it is True
    are always filled from ``components``' weekend subsequence (i.i.d.);
    the rest from a fixed-length circular block bootstrap of ``block_size``
    over the weekday subsequence.

    Raises:
        ValueError: if a required weekday or weekend historical subsequence
            is empty (never silently substituted, CLAUDE.md rule 29).
    """
    return _draws_from_components(components, weekend_before, n_paths, rng, block_size)


def _regime_bucket_mask(vol: npt.NDArray[np.float64], n_vol_buckets: int) -> npt.NDArray[np.bool_]:
    n_buckets = min(max(n_vol_buckets, 2), 3)
    current_vol = float(vol[-1])
    edges = np.quantile(vol, np.linspace(0.0, 1.0, n_buckets + 1))
    raw_idx = np.searchsorted(edges, current_vol, side="right") - 1
    bucket_idx = int(np.clip(raw_idx, 0, n_buckets - 1))
    lo_edge, hi_edge = edges[bucket_idx], edges[bucket_idx + 1]
    if bucket_idx == n_buckets - 1:
        return (vol >= lo_edge) & (vol <= hi_edge)
    return (vol >= lo_edge) & (vol < hi_edge)


def regime_bootstrap_daily(
    components: DailyComponents,
    weekend_before: npt.NDArray[np.bool_],
    n_paths: int,
    rng: np.random.Generator,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    regime_config: RegimeBootstrapConfig | None = None,
) -> DailyDraws:
    """Regime-conditioned bootstrap (Master Spec §11.1.B): as :func:`block_bootstrap_daily`,
    but the weekday subsequence is first restricted to the EWMA-volatility
    bucket matching the *current* (most recent historical day's) regime,
    falling back to the unrestricted weekday sample when that bucket has
    fewer than ``regime_config.min_bucket_days`` days. Weekend sampling is
    always drawn from the full (unrestricted) weekend subsequence -- see
    module docstring.
    """
    cfg = regime_config or RegimeBootstrapConfig()

    weekday_mask_hist = ~components.weekend_before
    weekday_components = _select(components, weekday_mask_hist)
    weekend_components = _select(components, components.weekend_before)

    r = weekday_components.gap + weekday_components.intraday
    if r.size >= 2:
        vol = ewma_volatility(r, cfg.ewma_lambda)
        bucket_mask = _regime_bucket_mask(vol, cfg.n_vol_buckets)
        if int(bucket_mask.sum()) < cfg.min_bucket_days:
            bucket_mask = np.ones_like(bucket_mask)
    else:
        bucket_mask = np.ones(r.size, dtype=np.bool_)

    filtered_weekday = _select(weekday_components, bucket_mask)

    combined = DailyComponents(
        dates=filtered_weekday.dates + weekend_components.dates,
        gap=np.concatenate([filtered_weekday.gap, weekend_components.gap]),
        weekend_before=np.concatenate(
            [filtered_weekday.weekend_before, weekend_components.weekend_before]
        ),
        intraday=np.concatenate([filtered_weekday.intraday, weekend_components.intraday]),
        hi=np.concatenate([filtered_weekday.hi, weekend_components.hi]),
        lo=np.concatenate([filtered_weekday.lo, weekend_components.lo]),
        n_discarded=components.n_discarded,
    )
    return _draws_from_components(combined, weekend_before, n_paths, rng, block_size)


def _standardized_components(
    components: DailyComponents, ewma_lambda: float
) -> tuple[DailyComponents, float]:
    """Standardize each day's ``(gap, intraday, hi, lo)`` by that day's own
    *trailing* (lagged, no-look-ahead) EWMA volatility of the total daily log
    return, and separately return the *current* (contemporaneous, as of the
    last historical day) EWMA volatility used to rescale simulated draws.

    ``ewma_volatility`` is contemporaneous (``vol[t]`` already incorporates
    ``r[t]`` itself); to standardize day ``t`` by information available
    *before* that day's return is known, we lag it by one day
    (``trailing[t] = vol[t-1]`` for ``t >= 1``; ``trailing[0] = vol[0]``, the
    only day with no prior history to lag from -- documented, not silently
    imputed). ``gap``/``intraday``/``hi``/``lo`` are all divided by the same
    single trailing total-return volatility (rather than four separate
    per-component vols): they are sub-components of one day's variability,
    so standardizing them jointly by the day's overall scale preserves their
    within-day *shape* (e.g. gap vs. intraday split, range width) while
    removing the *day-to-day* scale difference between calm and turbulent
    historical periods -- which is exactly the regime information this
    method conditions on.

    Raises:
        ValueError: if ``components`` has fewer than 2 days (not enough to
            estimate a volatility series at all).
    """
    r = components.gap + components.intraday
    n = r.size
    if n < 2:
        raise ValueError(f"need at least 2 daily components to standardize by volatility, got {n}")
    vol = ewma_volatility(r, ewma_lambda)
    sigma_current = float(vol[-1])

    trailing = np.empty(n, dtype=np.float64)
    trailing[0] = vol[0]
    trailing[1:] = vol[:-1]
    floor = float(np.median(trailing)) * _MIN_TRAILING_VOL_FRACTION
    trailing_safe = np.maximum(trailing, floor if floor > 0 else 1e-8)

    standardized = DailyComponents(
        dates=components.dates,
        gap=components.gap / trailing_safe,
        weekend_before=components.weekend_before,
        intraday=components.intraday / trailing_safe,
        hi=components.hi / trailing_safe,
        lo=components.lo / trailing_safe,
        n_discarded=components.n_discarded,
    )
    return standardized, sigma_current


def vol_scaled_bootstrap_daily(
    components: DailyComponents,
    weekend_before: npt.NDArray[np.bool_],
    n_paths: int,
    rng: np.random.Generator,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    ewma_lambda: float = _DEFAULT_EWMA_LAMBDA,
) -> DailyDraws:
    """Vol-standardized block bootstrap (Build Contract v2 W5 follow-up 2).

    Each historical day is first standardized by its own trailing EWMA
    volatility (:func:`_standardized_components`, no look-ahead), the
    resulting unit-scale shocks are resampled with the identical
    block/weekend logic as :func:`block_bootstrap_daily`, and the draws are
    then rescaled by the single *current* EWMA volatility (as of the most
    recent historical day). This conditions every simulated path on the
    prevailing volatility regime without any bucket-size threshold -- see
    module docstring for why this was chosen over trying to make
    ``regime_bootstrap_daily`` activate more often.

    Raises:
        ValueError: if ``components`` has fewer than 2 days, or a required
            weekday/weekend historical subsequence is empty.
    """
    standardized, sigma_current = _standardized_components(components, ewma_lambda)
    z_draws = _draws_from_components(standardized, weekend_before, n_paths, rng, block_size)
    return DailyDraws(
        gap=z_draws.gap * sigma_current,
        intraday=z_draws.intraday * sigma_current,
        hi=z_draws.hi * sigma_current,
        lo=z_draws.lo * sigma_current,
    )
