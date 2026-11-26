"""Read-only, explicitly exploratory, point-in-time cross-issuer net-EV probe.

One pre-declared rule on DAX LONG: at the first eligible comparable issuer pair
per calendar day, buy the product with the smaller observed relative bid/ask
spread; compare realised next-day Bid/entry Ask against NO_TRADE (zero) and
a fixed issuer-alphabetical baseline. Fees/slippage are conservatively reserved
as an additional 10 bps. A missing exit is never assumed profitable, and a
positive verdict is impossible if any selected observation has missing exits.

This is NOT confirmation or permission to trade. Historical data may already
have influenced the hypothesis; real confirmation needs a fresh untouched
forward partition and a separately approved, pre-registered trial.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from math import isfinite

import numpy as np

from turboedge.research.product_pairs import ComparabilityCriteria, census
from turboedge.storage.schemas import Direction, ProductSnapshot

MIN_INDEPENDENT_DAYS = 30
EXECUTION_BUFFER_BPS = 10.0
COST_SCENARIOS_BPS = (0, 10, 25, 50)
MIN_NET_EDGE = 0.001  # 10 bp, existing promotion ladder; not a promotion
MAX_QUOTE_AGE_S = 300
MIN_QUALITY = 0.8
EXIT_AFTER = timedelta(days=1)
EXIT_UNTIL = timedelta(days=2)
BOOTSTRAP_REPS = 3000
BOOTSTRAP_SEED = 20261108


@dataclass(frozen=True, slots=True)
class NetEVProbe:
    candidate_days: int
    complete_days: int
    missing_exit_days: int
    mean_net_return: float | None
    lower_ci_net_return: float | None
    mean_excess_over_naive: float | None
    lower_ci_excess_over_naive: float | None
    cost_sensitivity: dict[int, float | None]
    cost_sensitivity_lower_bound: dict[int, float | None]

    @property
    def verdict(self) -> str:
        if self.missing_exit_days:
            return "INCOMPLETE_EXITS"
        if self.complete_days < MIN_INDEPENDENT_DAYS:
            return "INSUFFICIENT_SAMPLE"
        if self.lower_ci_net_return is None or self.lower_ci_excess_over_naive is None:
            return "INSUFFICIENT_SAMPLE"
        if self.lower_ci_net_return > MIN_NET_EDGE and self.lower_ci_excess_over_naive > 0.0:
            return "EXPLORATORY_CANDIDATE_NOT_CONFIRMED"
        return "NO_DEMONSTRATED_NET_ALPHA"


def _valid_quote(s: ProductSnapshot) -> bool:
    if (
        s.quote_timestamp is None
        or s.is_stale
        or s.quality_score < MIN_QUALITY
        or s.retrieved_at < s.quote_timestamp
        or s.available_at > s.retrieved_at
        or (s.retrieved_at - s.quote_timestamp).total_seconds() > MAX_QUOTE_AGE_S
    ):
        return False
    # Do not match on a reference spot that was reported after entry or whose
    # independent timestamp is unknown: missing PIT evidence is not a pass.
    ref = s.underlying_price_ref_timestamp
    if ref is None or ref > s.retrieved_at:
        return False
    if (s.retrieved_at - ref).total_seconds() > MAX_QUOTE_AGE_S:
        return False
    return s.ask is not None and s.ask > 0 and s.bid is not None and 0 <= s.bid <= s.ask


def _fresh_exit(s: ProductSnapshot) -> bool:
    return (
        s.quote_timestamp is not None
        and s.bid is not None
        and s.bid >= 0
        and s.retrieved_at >= s.quote_timestamp
        and s.available_at <= s.retrieved_at
        and (s.retrieved_at - s.quote_timestamp).total_seconds() <= MAX_QUOTE_AGE_S
        and not s.is_stale
    )


def _lower_bound(values: list[float]) -> float | None:
    if len(values) < MIN_INDEPENDENT_DAYS or not all(isfinite(x) for x in values):
        return None
    a = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    # Sensitivity across contiguous cohort blocks; preserve local dependence.
    bounds = []
    for block in (3, 5, 10):
        count = (len(a) + block - 1) // block
        starts = rng.integers(0, len(a) - block + 1, size=(BOOTSTRAP_REPS, count))
        indices = (starts[:, :, None] + np.arange(block)).reshape(BOOTSTRAP_REPS, -1)
        means = np.mean(a[indices[:, : len(a)]], axis=1)
        bounds.append(float(np.quantile(means, 0.05)))
    return min(bounds)


def probe(snapshots: list[ProductSnapshot]) -> NetEVProbe:
    """One preselected DAX-LONG matched pair/day; never selects by future P&L."""
    matching = ComparabilityCriteria(window_seconds=60, require_verified_ratio=True)
    pairs = census(snapshots, criteria=matching).pairs
    by_isin: dict[str, list[ProductSnapshot]] = defaultdict(list)
    by_minute: dict[tuple[str, datetime], list[ProductSnapshot]] = defaultdict(list)
    for s in snapshots:
        by_isin[s.isin].append(s)
        if s.quote_timestamp is not None:
            t = s.quote_timestamp.astimezone(UTC)
            minute = datetime.fromtimestamp(int(t.timestamp()) // 60 * 60, tz=UTC)
            by_minute[(s.isin, minute)].append(s)
    for history in by_isin.values():
        history.sort(key=lambda s: (s.retrieved_at, s.observation_time))

    for options in by_minute.values():
        options.sort(key=lambda s: (s.retrieved_at, s.observation_time, s.raw_hash))

    # Select the FIRST eligible matched pair per day, irrespective of whether
    # either product later survives. This rules out ex-post survivor selection.
    entries: dict[date, tuple[ProductSnapshot, ProductSnapshot]] = {}
    ordered_pairs = sorted(
        pairs, key=lambda x: (x.window_start, x.issuer_a, x.isin_a, x.issuer_b, x.isin_b)
    )
    for pair in ordered_pairs:
        if pair.underlying_id != "DAX" or pair.direction is not Direction.LONG:
            continue
        day = pair.window_start.date()
        if day in entries:
            continue
        left_options = by_minute.get((pair.isin_a, pair.window_start), [])
        right_options = by_minute.get((pair.isin_b, pair.window_start), [])
        left = next((x for x in left_options if _valid_quote(x)), None)
        right = next((x for x in right_options if _valid_quote(x)), None)
        if left is None or right is None:
            continue
        # If the two instruments were not actually available within one
        # minute of each other, their quoted spreads are not comparable.
        if abs((left.retrieved_at - right.retrieved_at).total_seconds()) > 60:
            continue
        entries[day] = (left, right)

    def observed_return(entry: ProductSnapshot) -> float | None:
        assert entry.ask is not None
        open_at = entry.retrieved_at
        later = [s for s in by_isin[entry.isin] if open_at < s.retrieved_at <= open_at + EXIT_UNTIL]
        # Stop at the first terminal event; later observations cannot
        # change a position already exited at a valid bid.
        for s in later:
            if s.knocked_out and s.available_at <= s.retrieved_at:
                return -1.0 - EXECUTION_BUFFER_BPS / 10_000.0
            if s.retrieved_at >= open_at + EXIT_AFTER and _fresh_exit(s):
                bid = s.bid
                assert bid is not None
                return float(bid / entry.ask - 1 - EXECUTION_BUFFER_BPS / 10_000.0)
        return None

    values: list[float] = []
    differences: list[float] = []
    missing = 0
    for day in sorted(entries):
        left, right = entries[day]
        naive = min((left, right), key=lambda s: (s.issuer, s.isin))
        selected = min(
            (left, right),
            key=lambda s: (
                (float(s.ask or 0.0) - float(s.bid or 0.0)) / float(s.ask or 1.0),
                s.issuer,
                s.isin,
            ),
        )
        left_return = observed_return(left)
        right_return = observed_return(right)
        if left_return is None or right_return is None:
            missing += 1
            continue
        selected_return = left_return if selected is left else right_return
        naive_return = left_return if naive is left else right_return
        values.append(selected_return)
        differences.append(selected_return - naive_return)
    mean = float(np.mean(values)) if values else None
    lower = _lower_bound(values)
    return NetEVProbe(
        candidate_days=len(entries),
        complete_days=len(values),
        missing_exit_days=missing,
        mean_net_return=mean,
        lower_ci_net_return=lower,
        mean_excess_over_naive=float(np.mean(differences)) if differences else None,
        lower_ci_excess_over_naive=_lower_bound(differences),
        cost_sensitivity={
            bps: None if mean is None else mean + (EXECUTION_BUFFER_BPS - bps) / 10_000
            for bps in COST_SCENARIOS_BPS
        },
        cost_sensitivity_lower_bound={
            bps: None if lower is None else lower + (EXECUTION_BUFFER_BPS - bps) / 10_000
            for bps in COST_SCENARIOS_BPS
        },
    )


__all__ = ["NetEVProbe", "probe"]
