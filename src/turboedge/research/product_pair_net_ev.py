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
    return s.ask is not None and s.bid is not None and 0 <= s.bid <= s.ask


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
    means = np.mean(a[rng.integers(0, len(a), size=(BOOTSTRAP_REPS, len(a)))], axis=1)
    return float(np.quantile(means, 0.05))


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
        # A documented KO takes precedence over any later apparent quote.
        if any(s.knocked_out for s in later):
            return -1.0 - EXECUTION_BUFFER_BPS / 10_000.0
        exits = [s for s in later if s.retrieved_at >= open_at + EXIT_AFTER and _fresh_exit(s)]
        if not exits:
            return None
        bid = exits[0].bid
        assert bid is not None
        return float(bid / entry.ask - 1 - EXECUTION_BUFFER_BPS / 10_000.0)

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
        selected_return = observed_return(selected)
        naive_return = observed_return(naive)
        if selected_return is None or naive_return is None:
            missing += 1
            continue
        values.append(selected_return)
        differences.append(selected_return - naive_return)
    return NetEVProbe(
        candidate_days=len(entries),
        complete_days=len(values),
        missing_exit_days=missing,
        mean_net_return=float(np.mean(values)) if values else None,
        lower_ci_net_return=_lower_bound(values),
        mean_excess_over_naive=float(np.mean(differences)) if differences else None,
        lower_ci_excess_over_naive=_lower_bound(differences),
    )


__all__ = ["NetEVProbe", "probe"]
