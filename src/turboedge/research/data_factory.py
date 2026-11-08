"""Lossless, bounded PIT research quote tape in the EXISTING Parquet archive.

A daily deep product fetch keeps at most one new DAX LONG issuer pair every
three calendar days, then tracks both ISINs over the following 48 hours.
Entries and follow-up quotes share one typed Parquet table. The tiny table is
included in the already-encrypted lean state archive, unlike the unbounded
full scan archive. No new DuckDB table, no invented quotes, and no research
writes to any pricing, ledger, model, portfolio or order path.

All reads go through ParquetResearchArchive.read_as_of(), enforcing BOTH the
actual archive file's write-time and the record's availability-time cutoff.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

from turboedge.research.archive import ParquetResearchArchive, PointInTimeUnavailable, UnknownTable
from turboedge.research.product_pair_net_ev import _fresh_exit, _valid_quote
from turboedge.research.product_pairs import ComparabilityCriteria, census
from turboedge.storage.schemas import Direction, ProductSnapshot
from turboedge.storage.snapshots import write_snapshot_parquet

RESEARCH_TAPE_TABLE = "research_product_quotes"
RESEARCH_TAPE_VERSION = "1"
COHORT_SPACING_DAYS = 3
FOLLOWUP_WINDOW = timedelta(hours=48)
EXIT_AFTER = timedelta(hours=24)
MAX_ENTRY_QUOTE_DIFFERENCE_SECONDS = 60


class ResearchQuote(ProductSnapshot):
    """Original quote, plus immutable research-only entry-vs-exit provenance."""

    quote_role: Literal["entry", "followup"]
    cohort_day: date
    research_tape_version: Literal["1"] = "1"


@dataclass(frozen=True, slots=True)
class TapeCapture:
    quotes_written: int
    entry_pairs: int
    followup_quotes: int
    tracked_isins: int
    tape_file_written: bool


@dataclass(frozen=True, slots=True)
class TapeHealth:
    archive_files: int
    quote_rows: int
    source_days: int
    distinct_issuers: int
    entry_days: int
    independent_entry_days: int
    complete_exit_days: int
    missing_exit_days: int
    pending_exit_days: int
    invalid_entry_quotes: int
    invalid_temporal_rows: int


def _assert_cutoff(as_of: datetime) -> datetime:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("research archive cutoff must be timezone-aware")
    return as_of.astimezone(UTC)


def load_research_tape(state_dir: Path, *, as_of: datetime) -> list[ResearchQuote]:
    """Read preserved quote observations, or [] for an uninitialised archive.

    No fallback to the mutable, five-day-compacted DuckDB product snapshots.
    A structurally invalid archive is NOT treated as an empty or good archive.
    """
    cutoff = _assert_cutoff(as_of)
    archive = ParquetResearchArchive(state_dir)
    try:
        data = archive.read_as_of(RESEARCH_TAPE_TABLE, cutoff)
    except UnknownTable:
        return []
    if data.is_empty():
        return []
    records: list[ResearchQuote] = []
    keys: set[tuple[str, datetime, str, date, str]] = set()
    for row in data.to_dicts():
        quote = ResearchQuote.model_validate(row)
        # The archive reader protects file write-time + available_at. The
        # source's own retrieval and availability chronology must also hold.
        if (
            quote.retrieved_at > cutoff
            or quote.available_at > quote.retrieved_at
            or quote.observation_time > quote.retrieved_at
        ):
            raise PointInTimeUnavailable(
                "research quote has future/unreconstructable observation chronology"
            )
        key = (
            quote.isin,
            quote.retrieved_at,
            quote.quote_role,
            quote.cohort_day,
            quote.raw_hash,
        )
        if key not in keys:
            records.append(quote)
            keys.add(key)
    return sorted(records, key=lambda q: (q.retrieved_at, q.isin, q.quote_role))


def _first_verified_pair(
    quotes: list[ProductSnapshot], *, as_of: datetime
) -> tuple[ProductSnapshot, ProductSnapshot] | None:
    """Select from the first PIT-valid comparable pair; no future P&L input."""
    valid = [
        q
        for q in quotes
        if q.underlying_id == "DAX"
        and q.direction is Direction.LONG
        and q.retrieved_at <= as_of
        and q.available_at <= q.retrieved_at
        and _valid_quote(q)
    ]
    by_isin = {q.isin: q for q in valid}
    matches = census(
        valid, criteria=ComparabilityCriteria(window_seconds=60, require_verified_ratio=True)
    ).pairs
    for pair in sorted(
        matches,
        key=lambda p: (p.window_start, p.issuer_a, p.isin_a, p.issuer_b, p.isin_b),
    ):
        left = by_isin.get(pair.isin_a)
        right = by_isin.get(pair.isin_b)
        if left is None or right is None:
            continue
        if abs((left.retrieved_at - right.retrieved_at).total_seconds()) > (
            MAX_ENTRY_QUOTE_DIFFERENCE_SECONDS
        ):
            continue
        return left, right
    return None


def capture_daily_research_tape(
    state_dir: Path,
    *,
    products: list[ProductSnapshot],
    now: datetime,
) -> TapeCapture:
    """Append only a bounded new issuer pair + still-open tracked ISINs.

    Called from the separate deep-archive fetch, not from the live scan.
    Existing cohorts are recovered from the immutable tape itself on every
    run. Restarts never require a second registry or a manual pin list.
    """
    cutoff = _assert_cutoff(now)
    old = load_research_tape(state_dir, as_of=cutoff)
    old_keys = {(q.isin, q.retrieved_at, q.quote_role, q.cohort_day, q.raw_hash) for q in old}
    entries = [q for q in old if q.quote_role == "entry"]
    latest_entry = max((q.cohort_day for q in entries), default=None)

    # Only entry rows own a cohort. A follow-up quote may not be mistaken
    # for a new trade entry when the discovery engine builds test sessions.
    tracked = {
        q.isin: q for q in entries if timedelta(0) < cutoff - q.retrieved_at <= FOLLOWUP_WINDOW
    }
    new_quotes: list[ResearchQuote] = []
    for product in products:
        earlier = tracked.get(product.isin)
        if earlier is None:
            continue
        if not (
            earlier.retrieved_at < product.retrieved_at <= earlier.retrieved_at + FOLLOWUP_WINDOW
            and product.retrieved_at <= cutoff
            and product.available_at <= product.retrieved_at
        ):
            continue
        new_quotes.append(
            ResearchQuote.model_validate(
                {
                    **product.model_dump(),
                    "quote_role": "followup",
                    "cohort_day": earlier.cohort_day,
                }
            )
        )

    pair = _first_verified_pair(products, as_of=cutoff)
    if pair is not None:
        cohort_day = min(p.retrieved_at.astimezone(UTC).date() for p in pair)
        if latest_entry is None or (cohort_day - latest_entry).days >= COHORT_SPACING_DAYS:
            for product in pair:
                new_quotes.append(
                    ResearchQuote.model_validate(
                        {
                            **product.model_dump(),
                            "quote_role": "entry",
                            "cohort_day": cohort_day,
                        }
                    )
                )

    unique: list[ResearchQuote] = []
    seen = set(old_keys)
    for quote in new_quotes:
        key = (
            quote.isin,
            quote.retrieved_at,
            quote.quote_role,
            quote.cohort_day,
            quote.raw_hash,
        )
        if key not in seen:
            seen.add(key)
            unique.append(quote)
    if not unique:
        return TapeCapture(0, 0, 0, len(tracked), False)

    # Never forge an earlier write time. In production this is the actual
    # current UTC time immediately after fetch; injected 'now' exists for
    # deterministic tests.
    run_id = f"{cutoff.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:12]}"
    write_snapshot_parquet(RESEARCH_TAPE_TABLE, unique, run_id, state_dir, snapshot_date=cutoff)
    return TapeCapture(
        quotes_written=len(unique),
        entry_pairs=sum(1 for q in unique if q.quote_role == "entry") // 2,
        followup_quotes=sum(q.quote_role == "followup" for q in unique),
        tracked_isins=len(tracked),
        tape_file_written=True,
    )


def _has_observed_exit(entry: ResearchQuote, history: list[ResearchQuote]) -> bool:
    valid = [
        s
        for s in history
        if s.isin == entry.isin
        and entry.retrieved_at < s.retrieved_at <= entry.retrieved_at + FOLLOWUP_WINDOW
        and s.quote_role == "followup"
    ]
    # A documented KO observed before exit counts as an observed (negative)
    # outcome, not as a falsely "missing" winning Bid.
    if any(s.knocked_out for s in valid):
        return True
    return any(
        s.retrieved_at >= entry.retrieved_at + EXIT_AFTER
        and _fresh_exit(s)
        and s.ask is not None
        and s.bid is not None
        and s.bid <= s.ask
        for s in valid
    )


def audit_research_tape(
    state_dir: Path, *, as_of: datetime, quotes: list[ResearchQuote] | None = None
) -> TapeHealth:
    """Machine-readable availability and exit completeness: counts, not alpha."""
    cutoff = _assert_cutoff(as_of)
    archive = ParquetResearchArchive(state_dir)
    try:
        archive_files = len(archive.files(RESEARCH_TAPE_TABLE))
    except UnknownTable:
        archive_files = 0
    data = quotes if quotes is not None else load_research_tape(state_dir, as_of=cutoff)
    entries_by_day: dict[date, list[ResearchQuote]] = defaultdict(list)
    for q in data:
        if q.quote_role == "entry":
            entries_by_day[q.cohort_day].append(q)
    days = sorted(entries_by_day)
    independent = 0
    last: date | None = None
    for day in days:
        if last is None or (day - last).days >= COHORT_SPACING_DAYS:
            independent += 1
            last = day

    complete = missing = pending = 0
    for cohort in entries_by_day.values():
        if len(cohort) != 2:
            missing += 1
        elif any(q.retrieved_at + FOLLOWUP_WINDOW > cutoff for q in cohort):
            pending += 1
        elif all(_has_observed_exit(q, data) for q in cohort):
            complete += 1
        else:
            missing += 1
    invalid_temporal = sum(
        q.retrieved_at > cutoff
        or q.available_at > q.retrieved_at
        or q.observation_time > q.retrieved_at
        for q in data
    )
    return TapeHealth(
        archive_files=archive_files,
        quote_rows=len(data),
        source_days=len({q.retrieved_at.astimezone(UTC).date() for q in data}),
        distinct_issuers=len({q.issuer for q in data if q.quote_role == "entry"}),
        entry_days=len(days),
        independent_entry_days=independent,
        complete_exit_days=complete,
        missing_exit_days=missing,
        pending_exit_days=pending,
        invalid_entry_quotes=sum(not _valid_quote(q) for q in data if q.quote_role == "entry"),
        invalid_temporal_rows=invalid_temporal,
    )


__all__ = [
    "RESEARCH_TAPE_TABLE",
    "ResearchQuote",
    "TapeCapture",
    "TapeHealth",
    "audit_research_tape",
    "capture_daily_research_tape",
    "load_research_tape",
]
