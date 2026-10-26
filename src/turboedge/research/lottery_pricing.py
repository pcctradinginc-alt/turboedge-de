"""Are more lottery-like turbos sold at systematically worse executable costs?

PRE-REGISTRATION (written before the first run; result in
`docs/experiment_lottery_pricing_2026-10-26.md`)
---------------------------------------------------------------------
**Motivation.** Driessen, Ebert and Koeter, "Pi-CAPM: The Classical CAPM
with Probability Weighting and Skewed Assets", *Review of Financial
Studies* 38(12), 3497-3541 (verified 2026-10-26, DOI
10.1093/rfs/hhaf045), derive an equilibrium in which skewness carries a
**positive price effect**, amplified by volatility: investors overweight
small probabilities of large gains, so right-skewed payoffs are priced
high and earn less.

A turbo is more lottery-like the closer its knock-out barrier sits to the
spot. The probability of losing everything rises, and the conditional
upside multiple rises with it. If issuers price into that preference, then
**the more lottery-like the product, the worse its executable cost should
be.**

Pi-CAPM is a statement about *expected returns* in equilibrium, not about
dealer margins. It motivates this hypothesis; it does not predict it. That
distinction is the reason this module tests a cost, measures it on quotes
that were actually standing, and claims nothing about returns.

**Primary endpoint.** The slope of the relative round-trip spread,
``(ask - bid) / ask``, against barrier distance, estimated *within* cells
that hold one issuer, one underlying, one direction and one retrieval
minute fixed. Averaged across cells, with a bootstrap over cells.

Negative slope = spread widens as the barrier nears = lottery-like
products cost more. (Barrier distance falls as lottery-ness rises, hence
the sign.)

**Why the relative spread.** It is what a round trip actually costs at
prices that were actually quoted -- entry at Ask, exit at Bid (CLAUDE.md
rule 11). The obvious alternative, premium over intrinsic value divided by
intrinsic, puts the barrier distance in its own denominator, so the
regressor would sit on both sides of the equation. A premium measured as a
fraction of *spot* avoids that and is reported as a secondary endpoint --
but time value and gap risk genuinely do grow as a barrier nears, so a
slope there is not by itself evidence of a preference-driven markup.

**The endogeneity the brief names.** ``pricing/intrinsic.leverage`` is
computed *from the ask*. Regressing a cost on it would put the price on
both sides. The regressor here is therefore barrier distance,
``(spot - barrier) / spot``, which depends only on the product's terms and
the underlying's price -- never on the product's own price. This is not a
hypothetical risk: on 2026-10-25 bucketing on an ask-derived leverage
collapsed a planted rank correlation from about 0.9 to 0.14
(`research/data_audit.py`).

**Fixed effects, by construction rather than by estimation.** A cell is
(issuer, underlying, direction, retrieval minute). Estimating within it
differences out the issuer's pricing style *and* the market moment
together, which is what makes a slope attributable to barrier distance
rather than to who quoted it or when.

**Unit of independence.** One cell is one issuer at one moment, so one
cell is **one** observation, not one per product. The interval resamples
cells. Pooling products as independent would overstate the sample by
roughly the products per cell.

**Falsification, declared in advance.** The hypothesis FAILS if any of:

1. fewer than ``MIN_CELLS`` usable cells;
2. the bootstrap interval for the mean within-cell slope includes zero;
3. the economic size -- the fitted spread difference between the 10th and
   90th percentile of barrier distance -- is below
   ``MIN_ECONOMIC_SIZE_BPS`` basis points, which would make a clean slope
   a measurement without a consequence.

**What a PASS would and would not mean.** It would be a **cost** finding,
not alpha. It says nothing about direction and cannot make a trade
profitable. It could only ever justify preferring a cheaper product for a
trade already decided on other grounds, and the repository's own North
Star is explicit that a cost edge does not substitute for a missing
directional edge.

This module reads no database and writes nothing. It is pure functions
over ``ProductSnapshot`` so it can be tested without a DuckDB fixture.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from turboedge.storage.schemas import Direction, FieldReliability, ProductSnapshot

#: Minimum (issuer, underlying, direction, minute) cells before the result
#: is reportable. Below this the mean of a handful of slopes is noise.
MIN_CELLS = 30

#: Minimum products in a cell. A slope needs spread in the regressor, and
#: two points always fit a line exactly.
MIN_PRODUCTS_PER_CELL = 4

#: Smallest spread difference across the barrier-distance range that counts
#: as economically relevant, in basis points of the ask. One basis point of
#: round-trip spread is below the resolution anyone trades at.
MIN_ECONOMIC_SIZE_BPS = 5.0

#: Oldest quote age, in seconds, still treated as standing. A stale quote is
#: not a price anyone could have traded, and `research/data_audit.py`
#: measured that cheapness and quote age move together, so admitting stale
#: quotes here would import that association into this one.
MAX_QUOTE_AGE_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class CellSlope:
    """One issuer, one underlying, one direction, one minute."""

    issuer: str
    underlying_id: str
    direction: Direction
    minute: datetime
    n_products: int
    #: d(relative spread) / d(barrier distance). Negative means the spread
    #: widens as the barrier nears.
    spread_slope: float
    #: Same, for premium as a fraction of spot. Secondary.
    premium_slope: float
    barrier_distance_p10: float
    barrier_distance_p90: float

    @property
    def economic_size_bps(self) -> float:
        """Fitted spread difference across this cell's barrier range, in bps.

        Signed so that a positive number means the lottery-like end is the
        expensive one, which is the direction the hypothesis predicts.
        """
        span = self.barrier_distance_p90 - self.barrier_distance_p10
        return -self.spread_slope * span * 10_000.0


@dataclass(frozen=True, slots=True)
class LotteryPricingResult:
    """The family's verdict, in the harness vocabulary."""

    cells: tuple[CellSlope, ...]
    mean_spread_slope: float
    spread_ci_low: float
    spread_ci_high: float
    mean_premium_slope: float
    premium_ci_low: float
    premium_ci_high: float
    mean_economic_size_bps: float
    median_economic_size_bps: float
    products_used: int
    dropped: Mapping[str, int]

    @property
    def cells_used(self) -> int:
        return len(self.cells)

    @property
    def interval_excludes_zero(self) -> bool:
        if math.isnan(self.spread_ci_low) or math.isnan(self.spread_ci_high):
            return False
        return (self.spread_ci_low > 0.0) or (self.spread_ci_high < 0.0)

    @property
    def verdict(self) -> str:
        """`evaluate_excursion`'s vocabulary, so the loop reads one language.

        The three falsification criteria are applied in the order they were
        declared: sample, then interval, then economic size. A CANDIDATE
        additionally requires the slope to point the way Pi-CAPM predicts;
        a clean slope in the *opposite* direction is a real result and a
        refutation, not a candidate.
        """
        if self.cells_used < MIN_CELLS:
            return "INSUFFICIENT_SAMPLE"
        if not self.interval_excludes_zero:
            return "NO_IMPROVEMENT"
        if self.mean_spread_slope > 0.0:
            # Spread *narrows* as the barrier nears: the opposite of the
            # hypothesis, and worth recording as such.
            return "NO_IMPROVEMENT"
        # The MEDIAN cell, not the mean one. Measured on the real archive
        # (run 40389103879) the two differ by a factor of seven -- mean
        # 137.1 bps against median 19.6 -- because a handful of cells carry
        # an enormous slope. The pre-registration said "the fitted spread
        # difference across the barrier range" without fixing which
        # average, and the median is both the representative cell and the
        # stricter test, so switching to it after seeing the data can only
        # make the criterion harder to pass. The mean is still reported,
        # because a sevenfold gap between them is itself a finding.
        if self.median_economic_size_bps < MIN_ECONOMIC_SIZE_BPS:
            return "IMPROVEMENT_NOT_SIGNIFICANT"
        return "CANDIDATE"


def _minute(moment: datetime) -> datetime:
    epoch = int(moment.astimezone(UTC).timestamp())
    return datetime.fromtimestamp(epoch - epoch % 60, tz=UTC)


def _usable(snap: ProductSnapshot, dropped: Counter[str]) -> bool:
    """Whether this quote is one a round-trip cost can be measured on.

    Every rejection is counted. A silent filter is how a sample becomes
    unrepresentative without anyone noticing, and `data_audit.py` already
    measured what selective quote availability does here.
    """
    if snap.underlying_id is None:
        dropped["no_underlying_id"] += 1
        return False
    if snap.knocked_out:
        dropped["knocked_out"] += 1
        return False
    if snap.bid_only or snap.ask is None or not snap.ask > 0:
        dropped["no_ask"] += 1
        return False
    if snap.bid is None or not snap.bid > 0:
        dropped["no_bid"] += 1
        return False
    if snap.bid > snap.ask:
        dropped["crossed_quote"] += 1
        return False
    if snap.knockout_barrier is None:
        dropped["no_barrier"] += 1
        return False
    if snap.underlying_price_ref is None or not snap.underlying_price_ref > 0:
        dropped["no_underlying_reference"] += 1
        return False
    if not snap.ratio > 0:
        dropped["no_ratio"] += 1
        return False
    if snap.ratio_reliability is FieldReliability.UNVERIFIED:
        # Every per-unit figure divides by the ratio (CLAUDE.md rule 29).
        dropped["ratio_unverified"] += 1
        return False
    if snap.quote_timestamp is None:
        dropped["no_quote_timestamp"] += 1
        return False
    age = (snap.retrieved_at - snap.quote_timestamp).total_seconds()
    if age > MAX_QUOTE_AGE_SECONDS:
        dropped["quote_too_stale"] += 1
        return False
    return True


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope, or NaN when the regressor does not vary.

    No intercept shortcut: the intercept is the cell's own level, which is
    exactly what the within-cell design is differencing out.
    """
    x_centred = x - x.mean()
    denominator = float(np.dot(x_centred, x_centred))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(x_centred, y - y.mean()) / denominator)


def evaluate(
    snapshots: Sequence[ProductSnapshot],
    *,
    reps: int = 4000,
    alpha: float = 0.05,
    seed: int = 20261026,
) -> LotteryPricingResult:
    """Estimate the within-cell barrier-distance slope of executable cost.

    Raises:
        ValueError: if ``reps < 100`` -- a bootstrap interval from fewer
            draws is coarser than the thresholds it is compared against --
            or ``alpha`` outside ``(0, 1)``.
    """
    if reps < 100:
        raise ValueError(f"reps must be >= 100, got {reps!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    dropped: Counter[str] = Counter()
    cells: dict[tuple[str, str, Direction, datetime], list[tuple[float, float, float]]] = {}

    for snap in snapshots:
        if not _usable(snap, dropped):
            continue
        spot = float(snap.underlying_price_ref or 0.0)
        barrier = float(snap.knockout_barrier or 0.0)
        ask = float(snap.ask or 0.0)
        bid = float(snap.bid or 0.0)
        absolute = spot - barrier if snap.direction is Direction.LONG else barrier - spot
        barrier_distance = absolute / spot
        if not barrier_distance > 0.0:
            dropped["barrier_touched"] += 1
            continue

        relative_spread = (ask - bid) / ask
        # Premium over intrinsic, as a fraction of spot -- deliberately not
        # divided by intrinsic, which is itself the barrier distance.
        intrinsic_per_unit = absolute
        premium_over_spot = (ask / float(snap.ratio) - intrinsic_per_unit) / spot

        assert snap.underlying_id is not None  # `_usable` rejected None
        key = (snap.issuer, snap.underlying_id, snap.direction, _minute(snap.retrieved_at))
        cells.setdefault(key, []).append((barrier_distance, relative_spread, premium_over_spot))

    slopes: list[CellSlope] = []
    products_used = 0
    for (issuer, underlying_id, direction, minute), rows in cells.items():
        if len(rows) < MIN_PRODUCTS_PER_CELL:
            dropped["cell_too_small"] += 1
            continue
        distances = np.asarray([r[0] for r in rows], dtype=np.float64)
        spreads = np.asarray([r[1] for r in rows], dtype=np.float64)
        premiums = np.asarray([r[2] for r in rows], dtype=np.float64)
        spread_slope = _ols_slope(distances, spreads)
        if math.isnan(spread_slope):
            dropped["no_barrier_variation_in_cell"] += 1
            continue
        products_used += len(rows)
        slopes.append(
            CellSlope(
                issuer=issuer,
                underlying_id=underlying_id,
                direction=direction,
                minute=minute,
                n_products=len(rows),
                spread_slope=spread_slope,
                premium_slope=_ols_slope(distances, premiums),
                barrier_distance_p10=float(np.quantile(distances, 0.1)),
                barrier_distance_p90=float(np.quantile(distances, 0.9)),
            )
        )

    if not slopes:
        return LotteryPricingResult(
            cells=(),
            mean_spread_slope=float("nan"),
            spread_ci_low=float("nan"),
            spread_ci_high=float("nan"),
            mean_premium_slope=float("nan"),
            premium_ci_low=float("nan"),
            premium_ci_high=float("nan"),
            mean_economic_size_bps=float("nan"),
            median_economic_size_bps=float("nan"),
            products_used=0,
            dropped=dict(dropped),
        )

    spread_arr = np.asarray([c.spread_slope for c in slopes], dtype=np.float64)
    premium_arr = np.asarray([c.premium_slope for c in slopes], dtype=np.float64)
    size_arr = np.asarray([c.economic_size_bps for c in slopes], dtype=np.float64)

    rng = np.random.default_rng(seed)
    if spread_arr.size >= 3:
        draws = rng.integers(0, spread_arr.size, size=(reps, spread_arr.size))
        spread_means = spread_arr[draws].mean(axis=1)
        premium_means = np.nan_to_num(premium_arr, nan=0.0)[draws].mean(axis=1)
        lo, hi = (
            float(np.quantile(spread_means, alpha / 2)),
            float(np.quantile(spread_means, 1 - alpha / 2)),
        )
        p_lo, p_hi = (
            float(np.quantile(premium_means, alpha / 2)),
            float(np.quantile(premium_means, 1 - alpha / 2)),
        )
    else:
        lo = hi = p_lo = p_hi = float("nan")

    return LotteryPricingResult(
        cells=tuple(slopes),
        mean_spread_slope=float(np.mean(spread_arr)),
        spread_ci_low=lo,
        spread_ci_high=hi,
        mean_premium_slope=float(np.nanmean(premium_arr)) if premium_arr.size else float("nan"),
        premium_ci_low=p_lo,
        premium_ci_high=p_hi,
        mean_economic_size_bps=float(np.mean(size_arr)),
        median_economic_size_bps=float(np.median(size_arr)),
        products_used=products_used,
        dropped=dict(dropped),
    )


__all__ = [
    "MAX_QUOTE_AGE_SECONDS",
    "MIN_CELLS",
    "MIN_ECONOMIC_SIZE_BPS",
    "MIN_PRODUCTS_PER_CELL",
    "CellSlope",
    "LotteryPricingResult",
    "evaluate",
]
