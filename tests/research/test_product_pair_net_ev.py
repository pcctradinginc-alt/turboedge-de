"""Read-only product selection net EV: PIT entry, real exits, honest abstention."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from turboedge.research.product_pair_net_ev import probe
from turboedge.storage.schemas import (
    Direction,
    FieldReliability,
    ProductSnapshot,
    ProductType,
)

_START = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def quote(
    isin: str,
    issuer: str,
    at: datetime,
    *,
    bid: float,
    ask: float = 10.0,
    knocked_out: bool = False,
) -> ProductSnapshot:
    return ProductSnapshot(
        isin=isin,
        wkn=isin[-6:],
        issuer=issuer,
        venue="fake",
        underlying_raw="DAX",
        underlying_id="DAX",
        direction=Direction.LONG,
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=19000.0,
        knockout_barrier=19000.0,
        ratio=0.01,
        currency="EUR",
        underlying_currency="EUR",
        quanto=False,
        open_end=True,
        bid=bid,
        ask=ask,
        bid_size=100.0,
        ask_size=100.0,
        quote_timestamp=at,
        quote_presence=True,
        bid_only=False,
        knocked_out=knocked_out,
        underlying_price_ref=20000.0,
        underlying_price_ref_timestamp=at,
        ratio_reliability=FieldReliability.SOURCE_REPORTED,
        raw_hash=hashlib.sha256(isin.encode()).hexdigest(),
        observation_time=at,
        available_at=at,
        retrieved_at=at,
        source="fake",
        parser_version="1",
        is_stale=False,
        quality_score=0.99,
    )

def sample(days: int = 31, *, omit_exit: bool = False) -> list[ProductSnapshot]:
    rows = []
    for i in range(days):
        t = _START + timedelta(days=i * 3)
        a = f"DE000PRA{i:04d}"
        b = f"DE000PRB{i:04d}"
        rows.extend([quote(a, "BankA", t, bid=9.90), quote(b, "BankB", t, bid=9.99)])
        rows.append(quote(a, "BankA", t + timedelta(days=1), bid=10.8))
        if not omit_exit or i > 0:
            rows.append(quote(b, "BankB", t + timedelta(days=1, minutes=3), bid=11.5))
    return rows


def test_synthetic_independent_positive_pair_stays_exploratory() -> None:
    r = probe(sample())
    assert r.candidate_days == 31
    assert r.complete_days == 31
    assert r.missing_exit_days == 0
    assert r.mean_net_return is not None and r.mean_net_return > 0
    assert r.verdict == "EXPLORATORY_CANDIDATE_NOT_CONFIRMED"


def test_missing_future_bid_cannot_be_reported_as_alpha() -> None:
    r = probe(sample(omit_exit=True))
    assert r.missing_exit_days == 1
    assert r.verdict == "INCOMPLETE_EXITS"


def test_one_day_with_many_products_is_not_an_independent_sample() -> None:
    r = probe(sample(days=1))
    assert r.complete_days == 1
    assert r.verdict == "INSUFFICIENT_SAMPLE"


def test_future_quote_cannot_change_selection_when_entry_quote_was_stale() -> None:
    rows = sample(days=1)
    rows[1] = rows[1].model_copy(
        update={
            "quote_timestamp": rows[1].retrieved_at - timedelta(minutes=20),
        }
    )
    assert probe(rows).candidate_days == 0
