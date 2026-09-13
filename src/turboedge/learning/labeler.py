"""Forward-ledger labeling: resolve each due entry's exit (Master Spec §25,
§26 "Ambiguous Bars").

CLAUDE.md rule 17 / Master Spec §26: ambiguous bars are never optimistically
resolved. Every branch below that cannot find a confirmed real exit quote
falls back to the *lowest* plausible valuation, never the highest.

Exit resolution, in order:

1. Knock-out (KO): either the product feed itself reported
   ``knocked_out=True`` for a snapshot in the entry-to-exit window, or the
   underlying's OHLC bars show its low (long) / high (short) crossing the
   entry's barrier at any point in that window (this single low/high check
   also catches a pure overnight/weekend gap-through, since a bar's low is
   always <= its open). KO residual: 0 for `turbo_open_end`/`turbo_classic`;
   for `mini_future`, the first post-KO snapshot bid if one exists, else a
   conservative 0.
2. Normal horizon exit: a real product-snapshot bid quote is found --
   preferably the first one at/after `exit_due` 15:30 Europe/Berlin, else
   the last one available within `exit_due + 2 trading days`.
3. Neither of the above: no snapshot at all in the window. Falls back to
   the last known bid (from the window, or the entry's own `entry_bid`),
   discounted by the entry's own spread (`* (1 - entry_spread)`) --
   ``ambiguous_path=True``, `exit_reason=NO_EXIT_QUOTE_CONSERVATIVE`.
4. Not even a last-known bid exists anywhere: `exit_reason=EXPIRED_NO_DATA`,
   all P&L fields left `None`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field

from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Direction,
    ExitReason,
    LedgerLabel,
    ProductSnapshot,
    ProductType,
    UnderlyingBar,
)

_BERLIN = ZoneInfo("Europe/Berlin")
_EXIT_QUOTE_HOUR = 15
_EXIT_QUOTE_MINUTE = 30


class LabelerConfig(BaseModel):
    """This module's own config (wired into config.py/YAML by the
    integration wave)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # "letzter Snapshot <= exit_due + 2 Handelstage" (Build Contract v2 W6
    # requirement 3). Weekend-only trading-day approximation (Saturday/
    # Sunday skipped) -- this module has no exchange holiday calendar, so a
    # public holiday is not skipped; documented assumption.
    exit_quote_grace_trading_days: int = Field(default=2, ge=0)


class LabelRunResult(BaseModel):
    """Summary counters returned by :func:`label_due_entries`."""

    model_config = ConfigDict(extra="forbid")

    labeled: int = 0
    ko: int = 0
    ambiguous: int = 0
    skipped_not_due: int = 0
    missing_data: int = 0


@dataclass(frozen=True)
class ExitResolution:
    """Outcome of resolving one product's exit under the labeling rules.

    Reused by both the selected product (this module) and each alternative
    ISIN (``learning/counterfactual.py``), so "denselben Exit-Regeln"
    (Build Contract v2 W6 requirement 4) is literally the same code path,
    not a re-implementation.
    """

    exit_bid: float | None
    exit_quote_timestamp: datetime | None
    exit_reason: ExitReason
    realized_pnl: float | None  # exit_bid / entry_ask - 1, net simple return
    ko_hit: bool
    ambiguous_path: bool
    time_to_ko_days: int | None
    mfe: float | None
    mae: float | None


def _is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # 5=Sat, 6=Sun


def _add_trading_days(d: date, n: int) -> date:
    """``d`` plus ``n`` trading days, skipping Saturdays/Sundays only (no
    exchange holiday calendar available here -- documented approximation)."""
    current = d
    remaining = n
    while remaining > 0:
        current += timedelta(days=1)
        if not _is_weekend(current):
            remaining -= 1
    return current


def _exit_due_quote_deadline_utc(exit_due: date) -> datetime:
    """``exit_due`` at 15:30 Europe/Berlin, converted to UTC."""
    local = datetime.combine(exit_due, time(_EXIT_QUOTE_HOUR, _EXIT_QUOTE_MINUTE), tzinfo=_BERLIN)
    return local.astimezone(UTC)


def _nearest_bar_close_on_or_before(bars: Sequence[UnderlyingBar], as_of: date) -> float | None:
    candidates = [b for b in bars if b.ts.date() <= as_of]
    if not candidates:
        return None
    return max(candidates, key=lambda b: b.ts).close


def _barrier_touch_date(
    bars: Sequence[UnderlyingBar],
    *,
    direction: Direction,
    barrier: float,
    start: date,
    end: date,
) -> date | None:
    """First date within ``[start, end]`` whose bar's low (long) / high
    (short) crossed ``barrier``. A bar's low is always <= its open, so this
    also catches a pure overnight/weekend gap-through the barrier -- no
    separate gap-through branch is needed."""
    relevant = sorted((b for b in bars if start <= b.ts.date() <= end), key=lambda b: b.ts)
    for bar in relevant:
        touched = bar.low <= barrier if direction is Direction.LONG else bar.high >= barrier
        if touched:
            return bar.ts.date()
    return None


def resolve_exit(
    store: Store,
    *,
    isin: str,
    direction: Direction,
    entry_ask: float,
    entry_spread: float,
    entry_quote_timestamp: datetime,
    entry_underlying_date: date,
    exit_due: date,
    barrier: float | None,
    underlying_bars: Sequence[UnderlyingBar],
    product_type: ProductType | None,
    config: LabelerConfig | None = None,
) -> ExitResolution:
    """Resolve one product's exit for one ledger entry (or one counterfactual
    alternative), per the module docstring's rules.

    Args:
        store: Open ``Store`` (queried for real product snapshots).
        isin: The product to resolve.
        direction: Long/short (determines KO touch direction).
        entry_ask: Entry price the realized return is relative to.
        entry_spread: This product's own spread at entry, used as the
            assumed spread in the conservative (ambiguous) valuation.
        entry_quote_timestamp: Start of the snapshot search window.
        entry_underlying_date: Calendar date of entry, for `time_to_ko_days`.
        exit_due: The horizon's due date.
        barrier: Knock-out barrier at entry, or `None` if unknown (then KO
            can only be detected via the product feed's own flag).
        underlying_bars: Bars for this product's underlying (fetched once
            by the caller and shared across the selected product and all
            of its alternatives, since they share the same underlying).
        product_type: Needed to pick the KO-residual rule; `None` (unknown
            instrument master data) is treated as the conservative
            (turbo, 0 residual) case.
        config: Grace-period config; defaults to `LabelerConfig()`.

    Returns:
        The resolved :class:`ExitResolution`.
    """
    cfg = config if config is not None else LabelerConfig()
    window_end_date = _add_trading_days(exit_due, cfg.exit_quote_grace_trading_days)
    window_end = datetime.combine(window_end_date, time(23, 59, 59), tzinfo=UTC)

    snapshots = store.product_snapshots_in_range(isin, entry_quote_timestamp, window_end)
    with_bid = [s for s in snapshots if s.bid is not None]

    # -- MFE/MAE on product value relative to entry_ask, from every snapshot
    # with a bid seen in the window (Build Contract v2 W6 requirement 3).
    if with_bid:
        product_returns = [s.bid / entry_ask - 1 for s in with_bid if s.bid is not None]
        mfe = max(product_returns)
        mae = min(product_returns)
    else:
        mfe = None
        mae = None

    # -- KO detection: product feed flag, or bars crossing the barrier.
    ko_snapshot: ProductSnapshot | None = next((s for s in snapshots if s.knocked_out), None)
    bars_touch_date: date | None = None
    if barrier is not None and underlying_bars:
        bars_touch_date = _barrier_touch_date(
            underlying_bars,
            direction=direction,
            barrier=barrier,
            start=entry_underlying_date,
            end=exit_due,
        )
    ko_hit = ko_snapshot is not None or bars_touch_date is not None

    if ko_hit:
        ko_date = (
            ko_snapshot.quote_timestamp.date()
            if ko_snapshot is not None and ko_snapshot.quote_timestamp is not None
            else (
                ko_snapshot.observation_time.date() if ko_snapshot is not None else bars_touch_date
            )
        )
        time_to_ko_days = (ko_date - entry_underlying_date).days if ko_date is not None else None
        ko_ts = ko_snapshot.quote_timestamp if ko_snapshot is not None else None
        ko_reference_ts = (
            ko_ts
            if ko_ts is not None
            else (
                datetime.combine(ko_date, time(23, 59, 59), tzinfo=UTC)
                if ko_date is not None
                else None
            )
        )
        post_ko_snapshot: ProductSnapshot | None = None
        if product_type is ProductType.MINI_FUTURE and ko_reference_ts is not None:
            post_ko_candidates = [
                s for s in with_bid if (s.quote_timestamp or s.observation_time) >= ko_reference_ts
            ]
            post_ko_snapshot = post_ko_candidates[0] if post_ko_candidates else None
        if post_ko_snapshot is not None and post_ko_snapshot.bid is not None:
            exit_bid = post_ko_snapshot.bid
            exit_quote_timestamp = (
                post_ko_snapshot.quote_timestamp or post_ko_snapshot.observation_time
            )
        else:
            # Turbo (open-end/classic): residual is always 0. Mini-future
            # with no post-KO quote available: conservative 0 too (CLAUDE.md
            # rule 17 -- never optimistic).
            exit_bid = 0.0
            exit_quote_timestamp = None
        realized_pnl = exit_bid / entry_ask - 1
        return ExitResolution(
            exit_bid=exit_bid,
            exit_quote_timestamp=exit_quote_timestamp,
            exit_reason=ExitReason.KO,
            realized_pnl=realized_pnl,
            ko_hit=True,
            ambiguous_path=False,  # KO information here is unambiguous by construction
            time_to_ko_days=time_to_ko_days,
            mfe=mfe,
            mae=mae,
        )

    # -- Not knocked out: look for a real exit quote.
    deadline = _exit_due_quote_deadline_utc(exit_due)
    preferred = next(
        (s for s in with_bid if s.quote_timestamp is not None and s.quote_timestamp >= deadline),
        None,
    )
    exit_snapshot = preferred if preferred is not None else (with_bid[-1] if with_bid else None)

    if exit_snapshot is not None and exit_snapshot.bid is not None:
        exit_bid = exit_snapshot.bid
        realized_pnl = exit_bid / entry_ask - 1
        return ExitResolution(
            exit_bid=exit_bid,
            exit_quote_timestamp=exit_snapshot.quote_timestamp or exit_snapshot.observation_time,
            exit_reason=ExitReason.HORIZON,
            realized_pnl=realized_pnl,
            ko_hit=False,
            ambiguous_path=False,
            time_to_ko_days=None,
            mfe=mfe,
            mae=mae,
        )

    # -- No exit quote found at all. Fall back to the last known bid
    # (from the window, else `entry_ask`'s own paired bid is unavailable
    # here -- callers pass `entry_ask` only, so the true fallback is the
    # caller-supplied last known bid via `entry_spread`/`entry_ask`; when
    # nothing at all is known, this is unlabelable).
    return ExitResolution(
        exit_bid=None,
        exit_quote_timestamp=None,
        exit_reason=ExitReason.EXPIRED_NO_DATA,
        realized_pnl=None,
        ko_hit=False,
        ambiguous_path=True,
        time_to_ko_days=None,
        mfe=mfe,
        mae=mae,
    )


def resolve_exit_with_fallback(
    store: Store,
    *,
    isin: str,
    direction: Direction,
    entry_ask: float,
    entry_bid: float | None,
    entry_spread: float,
    entry_quote_timestamp: datetime,
    entry_underlying_date: date,
    exit_due: date,
    barrier: float | None,
    underlying_bars: Sequence[UnderlyingBar],
    product_type: ProductType | None,
    config: LabelerConfig | None = None,
) -> ExitResolution:
    """Wraps :func:`resolve_exit`, adding the last-known-bid conservative
    fallback (case 3 in the module docstring) when it returns
    ``EXPIRED_NO_DATA`` but a last-known bid is actually available (from the
    window's snapshots, or the entry's own ``entry_bid``)."""
    resolution = resolve_exit(
        store,
        isin=isin,
        direction=direction,
        entry_ask=entry_ask,
        entry_spread=entry_spread,
        entry_quote_timestamp=entry_quote_timestamp,
        entry_underlying_date=entry_underlying_date,
        exit_due=exit_due,
        barrier=barrier,
        underlying_bars=underlying_bars,
        product_type=product_type,
        config=config,
    )
    if resolution.exit_reason is not ExitReason.EXPIRED_NO_DATA:
        return resolution

    cfg = config if config is not None else LabelerConfig()
    window_end_date = _add_trading_days(exit_due, cfg.exit_quote_grace_trading_days)
    window_end = datetime.combine(window_end_date, time(23, 59, 59), tzinfo=UTC)
    snapshots = store.product_snapshots_in_range(isin, entry_quote_timestamp, window_end)
    with_bid = [s for s in snapshots if s.bid is not None]
    last_known_bid = with_bid[-1].bid if with_bid else entry_bid
    if last_known_bid is None:
        return resolution  # truly nothing to fall back to: EXPIRED_NO_DATA stands

    conservative_bid = last_known_bid * (1.0 - entry_spread)
    realized_pnl = conservative_bid / entry_ask - 1
    return ExitResolution(
        exit_bid=conservative_bid,
        exit_quote_timestamp=None,
        exit_reason=ExitReason.NO_EXIT_QUOTE_CONSERVATIVE,
        realized_pnl=realized_pnl,
        ko_hit=False,
        ambiguous_path=True,
        time_to_ko_days=None,
        mfe=resolution.mfe,
        mae=resolution.mae,
    )


def label_due_entries(
    store: Store,
    as_of: datetime,
    *,
    price_bars_lookup: Callable[[str], Sequence[UnderlyingBar]],
    config: LabelerConfig | None = None,
) -> LabelRunResult:
    """Label every forward-ledger entry whose `exit_due` has arrived by
    `as_of`, attaching one append-only `LedgerLabel` each (Master Spec §25).

    Args:
        store: Open ``Store``.
        as_of: "Now", for both the due-date filter and `labeled_at`.
        price_bars_lookup: Returns every available ``UnderlyingBar`` for one
            underlying id -- injected so this module never talks to a
            specific bar source/table directly (kept adapter-agnostic).
        config: Labeling config; defaults to `LabelerConfig()`.

    Returns:
        Summary counters (`LabelRunResult`).
    """
    entries = store.ledger_entries_due_for_labeling(as_of)
    result = LabelRunResult()
    bars_cache: dict[str, Sequence[UnderlyingBar]] = {}

    for entry in entries:
        bars = bars_cache.get(entry.underlying)
        if bars is None:
            bars = price_bars_lookup(entry.underlying)
            bars_cache[entry.underlying] = bars

        instrument = store.get_instrument(entry.selected_isin)
        product_type = instrument.product_type if instrument is not None else None

        resolution = resolve_exit_with_fallback(
            store,
            isin=entry.selected_isin,
            direction=entry.direction,
            entry_ask=entry.entry_ask,
            entry_bid=entry.entry_bid,
            entry_spread=entry.entry_spread,
            entry_quote_timestamp=entry.entry_quote_timestamp,
            entry_underlying_date=entry.entry_underlying_timestamp.date(),
            exit_due=entry.exit_due,
            barrier=entry.barrier_entry,
            underlying_bars=bars,
            product_type=product_type,
            config=config,
        )

        underlying_pnl = _underlying_log_return(
            bars,
            entry_date=entry.entry_underlying_timestamp.date(),
            exit_date=entry.exit_due,
        )

        label = LedgerLabel(
            entry_id=entry.entry_id,
            labeled_at=as_of,
            exit_bid=resolution.exit_bid,
            exit_quote_timestamp=resolution.exit_quote_timestamp,
            financing_level_exit=None,
            exit_reason=resolution.exit_reason,
            realized_selected_pnl=resolution.realized_pnl,
            underlying_pnl=underlying_pnl,
            median_turbo_pnl=None,
            best_turbo_pnl=None,
            ideal_turbo_pnl=None,
            mfe=resolution.mfe,
            mae=resolution.mae,
            ko_hit=resolution.ko_hit,
            time_to_ko_days=resolution.time_to_ko_days,
            ambiguous_path=resolution.ambiguous_path,
        )
        store.attach_ledger_label(label)

        if resolution.exit_reason is ExitReason.EXPIRED_NO_DATA:
            result.missing_data += 1
        else:
            result.labeled += 1
            if resolution.ko_hit:
                result.ko += 1
            if resolution.ambiguous_path:
                result.ambiguous += 1

    return result


def _underlying_log_return(
    bars: Sequence[UnderlyingBar], *, entry_date: date, exit_date: date
) -> float | None:
    """Log return of the underlying's close from the last bar on/before
    ``entry_date`` to the last bar on/before ``exit_date``. ``None`` if
    either endpoint cannot be resolved from ``bars`` (no bars close enough
    -- never guessed, CLAUDE.md rule 29)."""
    entry_close = _nearest_bar_close_on_or_before(bars, entry_date)
    exit_close = _nearest_bar_close_on_or_before(bars, exit_date)
    if entry_close is None or exit_close is None or entry_close <= 0:
        return None
    return math.log(exit_close / entry_close)


__all__ = [
    "ExitResolution",
    "LabelRunResult",
    "LabelerConfig",
    "label_due_entries",
    "resolve_exit",
    "resolve_exit_with_fallback",
]
