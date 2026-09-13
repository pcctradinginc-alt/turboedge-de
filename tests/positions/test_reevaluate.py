"""Tests for ``positions/reevaluate.py`` (Contract v3 Abschnitt D). No network."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from turboedge.config import load_config
from turboedge.positions.ledger import PositionLedger
from turboedge.positions.reevaluate import reevaluate_open_positions, reevaluate_position
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Direction,
    PositionEvaluationStatus,
    ProductSnapshot,
    ProductType,
    UnderlyingBar,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
_AS_OF = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)


@pytest.fixture
def cfg() -> Any:
    return load_config(CONFIG_DIR)


@pytest.fixture
def store(tmp_path: Path) -> Any:
    with Store(tmp_path / "state" / "turboedge.duckdb") as s:
        s.init_schema()
        yield s


class _FakePriceAdapter:
    def __init__(self, bars: list[UnderlyingBar]) -> None:
        self._bars = bars

    def fetch_daily_bars(
        self, underlying_id: str, *, lookback_days: int | None = None
    ) -> list[UnderlyingBar]:
        return list(self._bars)


def _bars(count: int = 200, drift: float = 0.0006, noise: float = 0.006) -> list[UnderlyingBar]:
    rng = np.random.default_rng(11)
    bars: list[UnderlyingBar] = []
    close = 24000.0
    d = date(2025, 1, 1)
    from datetime import timedelta

    while len(bars) < count:
        if d.weekday() < 5:
            new_close = close * (1.0 + drift + float(rng.normal(0.0, noise)))
            ts = datetime(d.year, d.month, d.day, tzinfo=UTC)
            avail = datetime(d.year, d.month, d.day, 22, 0, tzinfo=UTC)
            bars.append(
                UnderlyingBar(
                    underlying_id="DAX",
                    ts=ts,
                    open=close,
                    high=max(close, new_close) * 1.001,
                    low=min(close, new_close) * 0.999,
                    close=new_close,
                    volume=1.0,
                    observation_time=avail,
                    available_at=avail,
                    retrieved_at=avail,
                    source="fake",
                    parser_version="1",
                    is_stale=False,
                    quality_score=0.9,
                )
            )
            close = new_close
        d += timedelta(days=1)
    return bars


def _snapshot(
    isin: str, *, bid: float, ask: float, barrier: float | None, knocked_out: bool = False
) -> ProductSnapshot:
    return ProductSnapshot(
        isin=isin,
        wkn=isin[-6:],
        issuer="BankA",
        venue="stuttgart",
        underlying_raw="DAX",
        underlying_id="DAX",
        direction=Direction.LONG,
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=18000.0,
        knockout_barrier=barrier,
        ratio=0.01,
        currency="EUR",
        underlying_currency="EUR",
        quanto=False,
        open_end=True,
        bid=bid,
        ask=ask,
        bid_size=100.0,
        ask_size=100.0,
        quote_timestamp=_AS_OF,
        quote_presence=True,
        bid_only=False,
        knocked_out=knocked_out,
        raw_hash=hashlib.sha256(isin.encode()).hexdigest(),
        observation_time=_AS_OF,
        available_at=_AS_OF,
        retrieved_at=_AS_OF,
        source="fake",
        parser_version="1",
        is_stale=False,
        quality_score=0.9,
    )


def test_reevaluate_no_isin_is_invalidated(cfg: Any, store: Store) -> None:
    ledger = PositionLedger(store, clock=lambda: _AS_OF)
    pos = ledger.add("ABC123", qty=10, price=5.0, entry_date=date(2026, 1, 1))
    result = reevaluate_position(
        cfg=cfg, store=store, position=pos, price_adapter=_FakePriceAdapter([]), as_of=_AS_OF
    )
    assert result.evaluation.status == PositionEvaluationStatus.INVALIDATED
    assert result.evaluation.data_quality_ok is False
    assert "no_isin_on_file" in result.evaluation.reasons


def test_reevaluate_knocked_out_is_exit(cfg: Any, store: Store) -> None:
    isin = "DE000TESTKO1"
    store.append_product_snapshots(
        [_snapshot(isin, bid=0.0, ask=0.01, barrier=18000.0, knocked_out=True)]
    )
    store.upsert_instruments(
        [_snapshot(isin, bid=0.0, ask=0.01, barrier=18000.0, knocked_out=True)]
    )
    ledger = PositionLedger(store, clock=lambda: _AS_OF)
    pos = ledger.add("TESTKO", qty=10, price=60.0, entry_date=date(2026, 1, 1), isin=isin)
    result = reevaluate_position(
        cfg=cfg, store=store, position=pos, price_adapter=_FakePriceAdapter(_bars()), as_of=_AS_OF
    )
    assert result.evaluation.status == PositionEvaluationStatus.EXIT
    assert "knocked_out" in result.evaluation.reasons


def test_reevaluate_hold_with_good_signal_persists_and_is_idempotent_dedup(
    cfg: Any, store: Store
) -> None:
    isin = "DE000TESTHO1"
    snap = _snapshot(isin, bid=60.0, ask=60.2, barrier=18000.0)
    store.append_product_snapshots([snap])
    store.upsert_instruments([snap])
    ledger = PositionLedger(store, clock=lambda: _AS_OF)
    pos = ledger.add("TESTHO", qty=10, price=55.0, entry_date=date(2026, 1, 1), isin=isin)

    results = reevaluate_open_positions(
        cfg=cfg, store=store, price_adapter=_FakePriceAdapter(_bars()), as_of=_AS_OF
    )
    assert len(results) == 1
    evaluation = results[0].evaluation
    assert evaluation.status in (
        PositionEvaluationStatus.HOLD,
        PositionEvaluationStatus.REDUCE,
        PositionEvaluationStatus.EXIT,
    )
    assert evaluation.data_quality_ok is True
    assert evaluation.unrealized_return == pytest.approx(60.0 / 55.0 - 1.0)
    assert store.table_counts()["position_evaluations"] == 1

    # A second run on the same day, unchanged inputs, should not flag a
    # material change (status stable, numbers within threshold).
    results2 = reevaluate_open_positions(
        cfg=cfg, store=store, price_adapter=_FakePriceAdapter(_bars()), as_of=_AS_OF
    )
    assert results2[0].status_changed is False
    del pos
