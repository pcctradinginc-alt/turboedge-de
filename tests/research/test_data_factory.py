"""Phase 2: genuine PIT Parquet reads, bounded daily capture, durable lean backup."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from turboedge.research.alpha_discovery import run_discovery_step
from turboedge.research.archive import PointInTimeUnavailable
from turboedge.research.data_factory import (
    RESEARCH_TAPE_TABLE,
    ResearchQuote,
    audit_research_tape,
    capture_daily_research_tape,
    load_research_tape,
)
from turboedge.state.archive import (
    LEAN_INCLUDE_DIRS,
    ArchiveConfig,
    pack_state,
    unpack_state,
)
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import FieldReliability, ProductSnapshot
from turboedge.storage.snapshots import write_snapshot_parquet

_START = datetime(2026, 4, 1, 12, 0, 30, tzinfo=UTC)
_PASSWORD = "a-very-long-encryption-test-password"


def _pair(
    make: Callable[..., ProductSnapshot],
    at: datetime,
    *,
    exit_bid: bool = False,
) -> list[ProductSnapshot]:
    products = []
    for issuer, suffix, bid, age in [
        ("BankA", "A", 10.8 if exit_bid else 9.50, 0),
        ("BankB", "B", 11.5 if exit_bid else 9.95, 20),
    ]:
        isin = f"DE000DATA{suffix}01"
        timestamp = at - timedelta(seconds=age)
        products.append(
            make(
                isin=isin,
                issuer=issuer,
                raw_hash=hashlib.sha256(f"{isin}/{at.isoformat()}".encode()).hexdigest(),
                underlying_price_ref=20000.0,
                knockout_barrier=19000.0,
                financing_level=19000.0,
                ratio=0.01,
                ratio_reliability=FieldReliability.SOURCE_REPORTED,
                bid=bid,
                ask=11.6 if exit_bid else 10.0,
                bid_size=100.0 if issuer == "BankA" else 200.0,
                quote_timestamp=timestamp,
                underlying_price_ref_timestamp=at,
                observation_time=at,
                available_at=at,
                retrieved_at=at,
                source_timestamp=timestamp,
            )
        )
    return products


def test_daily_deep_archive_keeps_entry_and_48h_followups(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    root = tmp_path / "state"
    day0 = _pair(make_product_snapshot, _START)
    a = capture_daily_research_tape(root, products=day0, now=_START + timedelta(minutes=2))
    assert a.entry_pairs == 1
    assert a.quotes_written == 2
    assert a.followup_quotes == 0
    assert len(load_research_tape(root, as_of=_START + timedelta(minutes=2))) == 2

    # The next day's quotes complete yesterday's issuer comparison. They
    # are NOT allowed to become a second "new entry" on that day.
    next_time = _START + timedelta(days=1)
    b = capture_daily_research_tape(
        root,
        products=_pair(make_product_snapshot, next_time, exit_bid=True),
        now=next_time + timedelta(minutes=2),
    )
    assert b.entry_pairs == 0
    assert b.followup_quotes == 2
    health = audit_research_tape(root, as_of=_START + timedelta(days=3))
    assert health.entry_days == 1
    assert health.complete_exit_days == 1
    assert health.missing_exit_days == 0
    assert health.distinct_issuers == 2
    assert health.archive_files == 2


def test_repeated_capture_does_not_write_duplicate_snapshots(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    root = tmp_path / "state"
    data = _pair(make_product_snapshot, _START)
    now = _START + timedelta(minutes=2)
    first = capture_daily_research_tape(root, products=data, now=now)
    replay = capture_daily_research_tape(root, products=data, now=now)
    assert first.quotes_written == 2
    assert replay.quotes_written == 0
    assert audit_research_tape(root, as_of=now).archive_files == 1


def test_a_fresh_pair_after_three_days_is_an_independent_cohort(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    root = tmp_path / "state"
    capture_daily_research_tape(
        root,
        products=_pair(make_product_snapshot, _START),
        now=_START + timedelta(minutes=2),
    )
    third = _START + timedelta(days=3)
    added = capture_daily_research_tape(
        root,
        products=_pair(make_product_snapshot, third),
        now=third + timedelta(minutes=2),
    )
    assert added.entry_pairs == 1
    assert audit_research_tape(root, as_of=third + timedelta(minutes=2)).independent_entry_days == 2


def test_pending_exits_not_mistaken_for_mature_complete_outcomes(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    root = tmp_path / "state"
    capture_daily_research_tape(
        root,
        products=_pair(make_product_snapshot, _START),
        now=_START + timedelta(minutes=2),
    )
    before = audit_research_tape(root, as_of=_START + timedelta(days=1))
    assert before.pending_exit_days == 1
    assert before.complete_exit_days == 0

    after = audit_research_tape(root, as_of=_START + timedelta(days=3))
    assert after.pending_exit_days == 0
    assert after.missing_exit_days == 1
    assert after.complete_exit_days == 0


def test_file_write_time_prevents_future_backfill_from_entering_past(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    root = tmp_path / "state"
    original = _pair(make_product_snapshot, _START)[0]
    record = ResearchQuote.model_validate(
        {**original.model_dump(), "quote_role": "entry", "cohort_day": _START.date()}
    )
    future = _START + timedelta(days=5)
    write_snapshot_parquet(
        RESEARCH_TAPE_TABLE,
        [record],
        f"{future.strftime('%Y%m%dT%H%M%SZ')}-aaaa1234bbbb",
        root,
        snapshot_date=future,
    )
    assert load_research_tape(root, as_of=_START + timedelta(days=4)) == []
    assert len(load_research_tape(root, as_of=future + timedelta(minutes=1))) == 1


def test_corrupt_chronology_fails_closed(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    root = tmp_path / "state"
    quote = _pair(make_product_snapshot, _START)[0]
    quote = quote.model_copy(update={"retrieved_at": _START + timedelta(days=3)})
    record = ResearchQuote.model_validate(
        {**quote.model_dump(), "quote_role": "entry", "cohort_day": _START.date()}
    )
    tomorrow = _START + timedelta(days=1)
    write_snapshot_parquet(
        RESEARCH_TAPE_TABLE,
        [record],
        f"{tomorrow.strftime('%Y%m%dT%H%M%SZ')}-bbbb1234cccc",
        root,
        snapshot_date=tomorrow,
    )
    with pytest.raises(PointInTimeUnavailable):
        load_research_tape(root, as_of=tomorrow + timedelta(minutes=1))


def test_discovery_never_falls_back_to_compacted_duckdb(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    root = tmp_path / "state"
    with Store(root / "turboedge.duckdb") as store:
        store.init_schema()
        store.append_product_snapshots(_pair(make_product_snapshot, _START))
        result = run_discovery_step(store, now=_START + timedelta(days=4), state_dir=root)
        assert result.status == "AWAIT_DATA"
        assert result.eligible_days == 0
        assert store.list_research_trials() == []
        with pytest.raises(ValueError, match="PIT research tape"):
            run_discovery_step(store, now=_START + timedelta(days=4))


def test_lean_state_carries_only_curated_research_parquet_and_restores_it(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    root = tmp_path / "state"
    root.mkdir()
    with Store(root / "turboedge.duckdb") as store:
        store.init_schema()
    capture_daily_research_tape(
        root,
        products=_pair(make_product_snapshot, _START),
        now=_START + timedelta(minutes=2),
    )
    # The full scan tape is intentionally NOT part of the rolling lean pack.
    unrelated = root / "snapshots" / "product_snapshots" / "date=2026-04-01"
    unrelated.mkdir(parents=True)
    (unrelated / "fake.parquet").write_bytes(b"not archived in lean")

    encrypted = tmp_path / "lean.enc"
    pack_state(
        root,
        encrypted,
        _PASSWORD,
        config=ArchiveConfig(include_dirs=LEAN_INCLUDE_DIRS),
    )
    restored = tmp_path / "restored"
    unpack_state(encrypted, restored, _PASSWORD)
    assert len(load_research_tape(restored, as_of=_START + timedelta(minutes=3))) == 2
    assert not (restored / "snapshots" / "product_snapshots").exists()
    assert (restored / "turboedge.duckdb").is_file()


def test_followup_cannot_be_promoted_to_new_entry_by_discovery(
    tmp_path: Path, make_product_snapshot: Callable[..., ProductSnapshot]
) -> None:
    from turboedge.research.alpha_discovery import _candidate_sessions

    root = tmp_path / "state"
    capture_daily_research_tape(
        root,
        products=_pair(make_product_snapshot, _START),
        now=_START + timedelta(minutes=2),
    )
    day1 = _START + timedelta(days=1)
    capture_daily_research_tape(
        root,
        products=_pair(make_product_snapshot, day1, exit_bid=True),
        now=day1 + timedelta(minutes=2),
    )
    quotes = load_research_tape(root, as_of=_START + timedelta(days=3))
    sessions = _candidate_sessions(quotes, now=_START + timedelta(days=3))
    assert len(sessions) == 1
    assert sessions[0].day == _START.date()
