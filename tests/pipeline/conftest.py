"""Shared fixtures for pipeline tests: fake adapters, synthetic DAX products/bars.

No test in this package touches the network: every external dependency
(product adapters, price source, reference-rate source, notifier) is a small
in-memory fake implementing exactly the structural protocol the pipeline
needs, so ``pipeline.universe.run_universe`` / ``pipeline.scan.run_scan`` can
be exercised deterministically.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from turboedge.adapters.base import AdapterError, AdapterMetadata, HealthCheckResult
from turboedge.config import TurboEdgeConfig, load_config
from turboedge.notifications.gmail import EmailMessageSpec, NotificationError, SendResult
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Direction,
    HealthStatus,
    ProductSnapshot,
    ProductType,
    UnderlyingBar,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

DAX_SPOT = 24000.0


# --------------------------------------------------------------------------
# config / store
# --------------------------------------------------------------------------


@pytest.fixture
def cfg() -> TurboEdgeConfig:
    """The repo's real config, loaded fresh per test."""
    return load_config(CONFIG_DIR)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    db_path = tmp_path / "state" / "turboedge.duckdb"
    with Store(db_path) as s:
        s.init_schema()
        yield s


# --------------------------------------------------------------------------
# fake product adapter
# --------------------------------------------------------------------------


class FakeProductAdapter:
    """In-memory ``ProductSourceAdapter``.

    ``products`` is returned verbatim (a fresh list each call). Pass
    ``exception`` to make every ``fetch_products`` call raise it instead.
    ``on_fetch`` is an optional zero-arg callable invoked *before* the
    products are returned/the exception is raised -- used to assert
    something (e.g. "the signal row already exists") at the exact moment a
    product source is queried.
    """

    def __init__(
        self,
        name: str,
        *,
        products: Sequence[ProductSnapshot] | None = None,
        exception: Exception | None = None,
        health_status: HealthStatus = HealthStatus.PASS,
        on_fetch: Callable[[], None] | None = None,
    ) -> None:
        self._name = name
        self._products = list(products) if products is not None else []
        self._exception = exception
        self._health_status = health_status
        self._on_fetch = on_fetch
        self.fetch_calls = 0

    @property
    def name(self) -> str:
        return self._name

    def fetch_products(self, underlying_ids: Sequence[str]) -> list[ProductSnapshot]:
        self.fetch_calls += 1
        if self._on_fetch is not None:
            self._on_fetch()
        if self._exception is not None:
            raise self._exception
        return list(self._products)

    def healthcheck(self) -> HealthCheckResult:
        return HealthCheckResult(
            source=self._name,
            status=self._health_status,
            ok=self._health_status != HealthStatus.FAIL,
            latency_ms=1.0,
            checked_at=datetime.now(UTC),
            message="fake product adapter",
        )

    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(name=self._name, kind="product", version="1")


@pytest.fixture
def make_product_adapter() -> Callable[..., FakeProductAdapter]:
    def _make(name: str, **kwargs: Any) -> FakeProductAdapter:
        return FakeProductAdapter(name, **kwargs)

    return _make


# --------------------------------------------------------------------------
# fake price / reference-rate adapters
# --------------------------------------------------------------------------


class FakePriceAdapter:
    """In-memory ``PriceSource`` (``fetch_daily_bars``-compatible)."""

    def __init__(
        self,
        bars_by_underlying: dict[str, list[UnderlyingBar]] | None = None,
        *,
        raise_for: set[str] | None = None,
    ) -> None:
        self._bars = bars_by_underlying or {}
        self._raise_for = raise_for or set()
        self.calls: list[str] = []

    def set_bars(self, underlying_id: str, bars: list[UnderlyingBar]) -> None:
        self._bars[underlying_id] = bars

    def fetch_daily_bars(
        self, underlying_id: str, *, lookback_days: int | None = None
    ) -> list[UnderlyingBar]:
        self.calls.append(underlying_id)
        if underlying_id in self._raise_for:
            raise AdapterError(f"fake price fetch failure for {underlying_id!r}")
        return list(self._bars.get(underlying_id, []))

    def healthcheck(self) -> HealthCheckResult:
        return HealthCheckResult(
            source="fake_price",
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=1.0,
            checked_at=datetime.now(UTC),
            message="fake price adapter",
        )


@pytest.fixture
def make_price_adapter() -> Callable[..., FakePriceAdapter]:
    def _make(**kwargs: Any) -> FakePriceAdapter:
        return FakePriceAdapter(**kwargs)

    return _make


class FakeEstrAdapter:
    """In-memory ``EstrSource`` (``get_estr``-compatible)."""

    def __init__(self, rate: float = 0.03) -> None:
        self.rate = rate

    def get_estr(self) -> float:
        return self.rate

    def healthcheck(self) -> HealthCheckResult:
        return HealthCheckResult(
            source="fake_estr",
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=1.0,
            checked_at=datetime.now(UTC),
            message="fake estr adapter",
        )


@pytest.fixture
def make_estr_adapter() -> Callable[..., FakeEstrAdapter]:
    def _make(rate: float = 0.03) -> FakeEstrAdapter:
        return FakeEstrAdapter(rate)

    return _make


# --------------------------------------------------------------------------
# fake notifier
# --------------------------------------------------------------------------


@dataclass
class _FakeCredentials:
    recipients: list[str] = field(default_factory=lambda: ["research@example.com"])


class FakeNotifier:
    """Duck-typed stand-in for ``GmailNotifier`` (``.credentials`` + ``.send``)."""

    def __init__(self, *, dry_run: bool = False, fail: bool = False) -> None:
        self.credentials: _FakeCredentials | None = None if dry_run else _FakeCredentials()
        self.fail = fail
        self.send_calls = 0
        self.sent_specs: list[EmailMessageSpec] = []

    def send(self, spec: EmailMessageSpec) -> SendResult:
        self.send_calls += 1
        self.sent_specs.append(spec)
        if self.fail:
            raise NotificationError("fake smtp failure")
        if self.credentials is None:
            return SendResult(sent=False, dry_run=True, recipients=spec.to, message="dry-run")
        return SendResult(sent=True, dry_run=False, recipients=spec.to, message="sent")


@pytest.fixture
def make_notifier() -> Callable[..., FakeNotifier]:
    def _make(**kwargs: Any) -> FakeNotifier:
        return FakeNotifier(**kwargs)

    return _make


# --------------------------------------------------------------------------
# synthetic underlying bars
# --------------------------------------------------------------------------


def make_underlying_bars(
    underlying_id: str,
    *,
    count: int = 200,
    start: date = date(2025, 1, 1),
    base_close: float = DAX_SPOT,
    daily_drift: float = 0.0004,
    noise_std: float = 0.006,
    seed: int = 42,
    source: str = "fake_price",
) -> list[UnderlyingBar]:
    """Deterministic (seeded) daily OHLC bars on trading days (Mon-Fri only).

    A small positive ``daily_drift`` yields a clean, reproducible upward
    TSMOM signal (verified: score ~0.83, direction_hint=LONG for the
    defaults) -- useful for exercising "counter_baseline_signal" on SHORT
    candidates. ``available_at`` is set to 22:00 UTC on the bar's own
    trading day, matching a conservative same-day-close convention.
    """
    rng = np.random.default_rng(seed)
    bars: list[UnderlyingBar] = []
    close = base_close
    d = start
    while len(bars) < count:
        if d.weekday() < 5:  # Monday-Friday
            drift = daily_drift + float(rng.normal(0.0, noise_std))
            new_close = close * (1.0 + drift)
            open_ = close * (1.0 + float(rng.normal(0.0, noise_std / 4)))
            high = max(open_, new_close) * 1.002
            low = min(open_, new_close) * 0.998
            ts = datetime(d.year, d.month, d.day, tzinfo=UTC)
            available_at = datetime(d.year, d.month, d.day, 22, 0, tzinfo=UTC)
            bars.append(
                UnderlyingBar(
                    underlying_id=underlying_id,
                    ts=ts,
                    interval="1d",
                    open=open_,
                    high=high,
                    low=low,
                    close=new_close,
                    volume=1_000_000.0,
                    observation_time=available_at,
                    available_at=available_at,
                    retrieved_at=available_at,
                    source_timestamp=available_at,
                    source=source,
                    parser_version="1",
                    is_stale=False,
                    quality_score=0.9,
                )
            )
            close = new_close
        d += timedelta(days=1)
    return bars


@pytest.fixture
def dax_bars() -> list[UnderlyingBar]:
    """~200 trading days of synthetic DAX bars ending 2025-10-something (seed=42)."""
    return make_underlying_bars("DAX", count=200)


@pytest.fixture
def bars_factory() -> Callable[..., list[UnderlyingBar]]:
    return make_underlying_bars


# --------------------------------------------------------------------------
# synthetic DAX turbo products
# --------------------------------------------------------------------------


def make_dax_product(
    *,
    isin: str,
    issuer: str,
    direction: Direction,
    financing_level: float,
    knockout_barrier: float | None = None,
    ratio: float = 0.01,
    spot: float = DAX_SPOT,
    bid: float | None = None,
    ask: float | None = None,
    ask_size: float = 5000.0,
    bid_size: float = 5000.0,
    quote_timestamp: datetime,
    observation_time: datetime | None = None,
    bid_only: bool = False,
    knocked_out: bool = False,
    quote_presence: bool | None = True,
    underlying_price_ref: float | None = None,
    product_type: ProductType = ProductType.TURBO_OPEN_END,
    wkn: str | None = None,
    venue: str = "stuttgart",
    quality_score: float = 0.9,
) -> ProductSnapshot:
    """A realistic DAX open-end turbo. ``bid``/``ask`` default to intrinsic + a small premium."""
    barrier = knockout_barrier if knockout_barrier is not None else financing_level
    intrinsic = (
        max(spot - financing_level, 0.0) * ratio
        if direction == Direction.LONG
        else max(financing_level - spot, 0.0) * ratio
    )
    resolved_ask = ask if ask is not None else round(intrinsic + 0.10, 2)
    resolved_bid = bid if bid is not None else round(resolved_ask - 0.10, 2)
    obs_time = observation_time if observation_time is not None else quote_timestamp

    return ProductSnapshot(
        isin=isin,
        wkn=wkn or isin[-6:],
        issuer=issuer,
        venue=venue,
        underlying_raw="DAX",
        underlying_id="DAX",
        direction=direction,
        product_type=product_type,
        financing_level=financing_level,
        knockout_barrier=barrier,
        ratio=ratio,
        currency="EUR",
        underlying_currency="EUR",
        quanto=False,
        open_end=True,
        maturity=None,
        first_trading_day=None,
        bid=resolved_bid,
        ask=resolved_ask,
        bid_size=bid_size,
        ask_size=ask_size,
        quote_timestamp=quote_timestamp,
        quote_presence=quote_presence,
        bid_only=bid_only,
        knocked_out=knocked_out,
        trading_hours="09:00-22:00",
        product_age_days=100,
        underlying_price_ref=underlying_price_ref,
        raw_hash=hashlib.sha256(f"{isin}-{quote_timestamp.isoformat()}".encode()).hexdigest(),
        observation_time=obs_time,
        available_at=obs_time,
        retrieved_at=obs_time,
        source_timestamp=obs_time,
        source="fake_source",
        parser_version="1",
        is_stale=False,
        quality_score=quality_score,
    )


@pytest.fixture
def dax_product_factory() -> Callable[..., ProductSnapshot]:
    return make_dax_product


__all__ = [
    "DAX_SPOT",
    "FakeEstrAdapter",
    "FakeNotifier",
    "FakePriceAdapter",
    "FakeProductAdapter",
    "make_dax_product",
    "make_underlying_bars",
]
