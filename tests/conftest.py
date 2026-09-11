from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from turboedge.storage.schemas import (
    CandidateEvaluation,
    Category,
    CostDecomposition,
    Direction,
    ProductSnapshot,
    ProductType,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


@pytest.fixture
def config_dir() -> Path:
    """The repo's real configs/ directory (used to test-load real config files)."""
    return CONFIG_DIR


def _utc(year: int, month: int, day: int, hour: int = 12, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


@pytest.fixture
def make_product_snapshot() -> Callable[..., ProductSnapshot]:
    """Factory for a valid, fully-populated ProductSnapshot with sane overrides."""

    def _make(**overrides: object) -> ProductSnapshot:
        defaults: dict[str, object] = dict(
            isin="DE000ABC1234",
            wkn="ABC123",
            issuer="TestBank",
            venue="stuttgart",
            underlying_raw="DAX",
            underlying_id="DAX",
            direction=Direction.LONG,
            product_type=ProductType.TURBO_OPEN_END,
            financing_level=18000.0,
            knockout_barrier=18000.0,
            ratio=0.01,
            currency="EUR",
            underlying_currency="EUR",
            quanto=False,
            open_end=True,
            maturity=None,
            first_trading_day=None,
            bid=4.80,
            ask=4.86,
            bid_size=1000.0,
            ask_size=1000.0,
            quote_timestamp=_utc(2026, 9, 10, 15, 30),
            quote_presence=True,
            bid_only=False,
            knocked_out=False,
            trading_hours="09:00-22:00",
            product_age_days=120,
            underlying_price_ref=18500.0,
            raw_hash=hashlib.sha256(b"raw-record").hexdigest(),
            observation_time=_utc(2026, 9, 10, 15, 30),
            available_at=_utc(2026, 9, 10, 15, 30),
            retrieved_at=_utc(2026, 9, 10, 15, 31),
            source_timestamp=_utc(2026, 9, 10, 15, 30),
            source="test_source",
            parser_version="1",
            is_stale=False,
            quality_score=0.95,
        )
        defaults.update(overrides)
        return ProductSnapshot(**defaults)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def make_large_product_snapshot_batch(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> Callable[[int], list[ProductSnapshot]]:
    """Factory for a batch of ``ProductSnapshot`` that reproduces the
    real-world Parquet schema-inference bug (polars.exceptions.ComputeError):
    the first 200 records have several optional fields (``quanto``,
    ``quote_presence``, ``ask``, ``quote_timestamp``, ``maturity``) set to
    ``None``, and only records from index 200 onward populate them with real
    values - so a naive schema guess from the first ~100 rows sees the wrong
    (or no) type for those columns.
    """

    def _make(n: int = 5000) -> list[ProductSnapshot]:
        records = []
        for i in range(n):
            none_block = i < 200
            records.append(
                make_product_snapshot(
                    isin=f"DE000{i:07d}",
                    quanto=None if none_block else (i % 2 == 0),
                    quote_presence=None if none_block else (i % 3 != 0),
                    ask=None if none_block else round(4.80 + (i % 50) * 0.01, 2),
                    quote_timestamp=None if none_block else _utc(2026, 9, 10, 15, 30),
                    maturity=None if none_block else date(2026, 12, 18),
                    product_type=(
                        ProductType.TURBO_CLASSIC if i % 7 == 0 else ProductType.TURBO_OPEN_END
                    ),
                )
            )
        return records

    return _make


@pytest.fixture
def make_large_candidate_batch() -> Callable[[int], list[CandidateEvaluation]]:
    """Factory for a batch of ``CandidateEvaluation`` with the same
    None-then-populated pattern as :func:`make_large_product_snapshot_batch`,
    additionally covering a nested ``BaseModel`` field (``costs``) and a
    ``dict`` field (``financing_cost_horizon_pct``) going from absent/empty
    to populated partway through the batch.
    """

    def _make(n: int = 5000) -> list[CandidateEvaluation]:
        records = []
        for i in range(n):
            none_block = i < 200
            costs = (
                None
                if none_block
                else CostDecomposition(
                    ask=4.86,
                    bid=4.80,
                    mid=4.83,
                    intrinsic=4.70,
                    trading_spread_component=0.03,
                    fair_gap_premium=0.02,
                    financing_drag=0.01,
                    issuer_margin=0.10,
                    spread_pct=0.0123,
                    gap_premium_pct=0.0041,
                    financing_drag_pct=0.0021,
                    issuer_margin_pct=0.0206,
                )
            )
            records.append(
                CandidateEvaluation(
                    run_id="run-large",
                    candidate_id=f"cand-{i}",
                    isin=f"DE000{i:07d}",
                    wkn=f"WKN{i:04d}",
                    issuer="TestBank",
                    underlying_id="DAX",
                    direction=Direction.LONG if i % 2 == 0 else Direction.SHORT,
                    category=Category.WATCH,
                    reasons=[] if none_block else ["ok"],
                    leverage=None if none_block else 5.0 + (i % 10),
                    leverage_bucket=None if none_block else "5-10",
                    distance_to_barrier_pct=None if none_block else 0.03,
                    distance_to_barrier_sigma=None if none_block else 1.5,
                    costs=costs,
                    realized_financing_spread=None if none_block else 0.02,
                    financing_cost_horizon_pct=({} if none_block else {"3d": 0.001, "7d": 0.003}),
                    cross_issuer_residual_zscore=None if none_block else 0.5,
                    issuer_markup_score=None if none_block else 0.1,
                    quote_dislocation_score=None if none_block else 0.05,
                    wrapper_edge=None,
                    liquidity_factor=None if none_block else 0.8,
                    integrity_passed=i % 2 == 0,
                    lcb_ev=None,
                    cost_rank_score=None if none_block else 0.12,
                )
            )
        return records

    return _make
