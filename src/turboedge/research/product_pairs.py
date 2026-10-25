"""How many *comparable* turbo products does the archive actually hold?

This is the measurement `docs/research_plan_product_selection_cost_edge.md`
§4 names as the precondition for pre-registering that question. The plan
asks whether systematically selecting between comparable products of
different issuers buys a real, executable cost advantage. Before any of
that can be tested, one count has to exist: **how often does the archive
contain two products that are genuinely comparable and come from different
issuers?**

If that count is small, the honest answer is that the question is not
measurable yet, and the reason would be issuer coverage rather than history
length -- `docs/measured_results.md` §4 records one issuer dominating the
universe, and a direct ISIN-level cross-check finding zero overlap between
that issuer's own feed and the same issuer's products as attributed by a
multi-issuer venue.

This module only counts. It computes no cost, tests no hypothesis, opens no
trial and consumes no research budget.

WHAT "COMPARABLE" MEANS HERE
----------------------------
Same underlying, same direction, quoted in the same short window, with
barrier distance and leverage each within a stated tolerance. Barrier
distance uses the signed percent definition from
`features/product.distance_to_barrier` (positive means not yet touched) and
leverage the one from `pricing/intrinsic.leverage`, so the figures mean the
same thing here as everywhere else in the repository.

A DELIBERATE UNDER-COUNT
------------------------
Matching is done by bucketing barrier distance and leverage at the
tolerance width rather than by comparing every pair. Two products that sit
just either side of a bucket edge are within tolerance of each other and
are not counted as a match, so **every count here is a lower bound**.

That is the right direction for the decision this serves: if the lower
bound is already sufficient, the question is measurable regardless of the
edges. Enumerating all pairs would be exact and quadratic in the number of
products per underlying per window, which runs into the thousands.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from math import log, log1p

from turboedge.pricing.intrinsic import leverage as effective_leverage
from turboedge.storage.schemas import Direction, FieldReliability, ProductSnapshot


@dataclass(frozen=True, slots=True)
class ComparabilityCriteria:
    """What counts as "the same product from a different issuer".

    The tolerances are the plan's (§4): barrier distance within +/-10 %,
    leverage within +/-15 %. They are parameters rather than constants
    because the census is meant to be re-run at other tolerances to see how
    sensitive the count is -- a count that only appears at a loose tolerance
    is not evidence that comparable products exist.
    """

    barrier_pct_tolerance: float = 0.10
    leverage_tolerance: float = 0.15
    window_seconds: int = 60
    #: Exclude products whose Bezugsverhaeltnis was never verified. Every
    #: per-unit cost figure divides by the ratio, so an unverified one makes
    #: the comparison meaningless (CLAUDE.md rule 29).
    require_verified_ratio: bool = True


@dataclass(frozen=True, slots=True)
class MatchedPair:
    """Two comparable products from different issuers, at one moment."""

    underlying_id: str
    direction: Direction
    window_start: datetime
    issuer_a: str
    issuer_b: str
    isin_a: str
    isin_b: str

    @property
    def issuer_pair(self) -> tuple[str, str]:
        return tuple(sorted((self.issuer_a, self.issuer_b)))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PairCensus:
    """The count, and everything needed to judge whether it is enough."""

    pairs: tuple[MatchedPair, ...] = ()
    snapshots_considered: int = 0
    dropped: Mapping[str, int] = field(default_factory=dict)

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    @property
    def pairs_by_underlying(self) -> Mapping[str, int]:
        return Counter(p.underlying_id for p in self.pairs)

    @property
    def pairs_by_issuer_pair(self) -> Mapping[tuple[str, str], int]:
        return Counter(p.issuer_pair for p in self.pairs)

    @property
    def distinct_dates_by_underlying(self) -> Mapping[str, int]:
        """Independent observations, which is what a sample size means here.

        Rows from one day are one observation repeated. Counting pairs
        without this would make a single well-covered afternoon look like a
        sample.
        """
        dates: dict[str, set[object]] = {}
        for pair in self.pairs:
            dates.setdefault(pair.underlying_id, set()).add(pair.window_start.date())
        return {underlying: len(days) for underlying, days in dates.items()}

    @property
    def distinct_issuer_pairs(self) -> int:
        return len(self.pairs_by_issuer_pair)

    def summary(self) -> str:
        if not self.pairs:
            return (
                f"no comparable cross-issuer pair found in {self.snapshots_considered} "
                "usable snapshot(s)"
            )
        return (
            f"{self.pair_count} comparable cross-issuer pair(s) across "
            f"{len(self.pairs_by_underlying)} underlying(s) and "
            f"{self.distinct_issuer_pairs} issuer pair(s)"
        )


def _usable(
    snapshot: ProductSnapshot, criteria: ComparabilityCriteria, dropped: Counter[str]
) -> bool:
    """Whether this snapshot can take part in a cost comparison at all."""
    if snapshot.underlying_id is None:
        dropped["no_underlying_id"] += 1
        return False
    if snapshot.knocked_out:
        dropped["knocked_out"] += 1
        return False
    if snapshot.bid_only or snapshot.ask is None or not snapshot.ask > 0:
        # Entry is at Ask. Without one there is no entry cost to compare.
        dropped["no_ask"] += 1
        return False
    if snapshot.bid is None:
        # Exit is at Bid. A round-trip cost needs both sides.
        dropped["no_bid"] += 1
        return False
    if snapshot.knockout_barrier is None:
        dropped["no_barrier"] += 1
        return False
    if snapshot.underlying_price_ref is None or not snapshot.underlying_price_ref > 0:
        dropped["no_underlying_reference"] += 1
        return False
    if criteria.require_verified_ratio and (
        snapshot.ratio_reliability is FieldReliability.UNVERIFIED
    ):
        dropped["ratio_unverified"] += 1
        return False
    return True


def _barrier_pct(snapshot: ProductSnapshot) -> float:
    """Signed percent distance to the barrier; positive = not yet touched.

    The same definition as `features/product.distance_to_barrier`, without
    the sigma normalisation, which would need a volatility estimate this
    census does not require.
    """
    spot = float(snapshot.underlying_price_ref or 0.0)
    barrier = float(snapshot.knockout_barrier or 0.0)
    absolute = spot - barrier if snapshot.direction is Direction.LONG else barrier - spot
    return absolute / spot


def _relative_bucket(value: float, tolerance: float) -> int:
    """Bucket `value` so that one bucket spans a *relative* tolerance.

    Both tolerances in `ComparabilityCriteria` are relative (+/-10 % of the
    barrier distance, +/-15 % of the leverage), so the bucket width has to
    grow with the value. `floor(log(value) / log(1 + tolerance))` does that;
    a fixed-width bucket would be far too wide at leverage 30 and far too
    narrow at leverage 2.

    An earlier version of this computed `value // (value * tolerance)`,
    which is `1 / tolerance` for every input -- a single bucket, so the
    dimension was silently ignored and every pair in the window counted as
    comparable.
    """
    return int(log(value) // log1p(tolerance))


def _window_start(snapshot: ProductSnapshot, window_seconds: int) -> datetime:
    effective = snapshot.quote_timestamp or snapshot.observation_time
    epoch = int(effective.astimezone(UTC).timestamp())
    return datetime.fromtimestamp(epoch - epoch % window_seconds, tz=UTC)


def census(
    snapshots: Sequence[ProductSnapshot],
    *,
    criteria: ComparabilityCriteria | None = None,
) -> PairCensus:
    """Count comparable cross-issuer product pairs in an archive slice.

    Pairs, not sets: the plan's hypothesis compares a selection rule against
    a naive one, which needs at least two candidates. Reporting pairs also
    keeps the issuer dimension visible, which is the one that decides
    whether the question is answerable at all.
    """
    crit = criteria or ComparabilityCriteria()
    dropped: Counter[str] = Counter()

    # (underlying, direction, window, barrier bucket, leverage bucket) ->
    # one representative product per issuer. One per issuer is enough: a
    # second product from an issuer already present adds no cross-issuer
    # pair that the first does not.
    _BucketKey = tuple[str, Direction, datetime, int, int]
    buckets: dict[_BucketKey, dict[str, str]] = {}
    considered = 0

    for snapshot in snapshots:
        if not _usable(snapshot, crit, dropped):
            continue
        underlying_id = snapshot.underlying_id
        if underlying_id is None:  # pragma: no cover - `_usable` already dropped it
            continue
        try:
            lev = effective_leverage(
                float(snapshot.underlying_price_ref or 0.0),
                float(snapshot.ask or 0.0),
                float(snapshot.ratio),
            )
        except ValueError:
            dropped["leverage_not_computable"] += 1
            continue
        if not lev > 0:
            dropped["leverage_not_positive"] += 1
            continue
        considered += 1

        barrier_pct = _barrier_pct(snapshot)
        if not barrier_pct > 0.0:
            # The barrier is at or past the spot. Whether or not the issuer
            # has flagged `knocked_out` yet, there is no round trip to cost.
            dropped["barrier_touched"] += 1
            considered -= 1
            continue
        key: _BucketKey = (
            underlying_id,
            snapshot.direction,
            _window_start(snapshot, crit.window_seconds),
            # Bucketed at the relative tolerance -- see the module docstring
            # on why this makes every count a lower bound.
            _relative_bucket(barrier_pct, crit.barrier_pct_tolerance),
            _relative_bucket(lev, crit.leverage_tolerance),
        )
        buckets.setdefault(key, {}).setdefault(snapshot.issuer, snapshot.isin)

    pairs: list[MatchedPair] = []
    for key, by_issuer in buckets.items():
        if len(by_issuer) < 2:
            continue
        bucket_underlying, direction, window_start = key[0], key[1], key[2]
        for (issuer_a, isin_a), (issuer_b, isin_b) in combinations(sorted(by_issuer.items()), 2):
            pairs.append(
                MatchedPair(
                    underlying_id=bucket_underlying,
                    direction=direction,
                    window_start=window_start,
                    issuer_a=issuer_a,
                    issuer_b=issuer_b,
                    isin_a=isin_a,
                    isin_b=isin_b,
                )
            )

    return PairCensus(pairs=tuple(pairs), snapshots_considered=considered, dropped=dict(dropped))


__all__ = ["ComparabilityCriteria", "MatchedPair", "PairCensus", "census"]
