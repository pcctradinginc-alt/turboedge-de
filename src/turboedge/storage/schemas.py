"""Pydantic v2 data models for every persisted record in TurboEdge-DE.

These models are the single source of truth for what a "product", a
"signal", a "candidate evaluation" etc. look like. ``storage/duckdb.py`` and
``storage/snapshots.py`` both serialize/deserialize against these classes so
the DuckDB tables, the Parquet archive and in-memory pipeline objects can
never silently drift apart.

Validation intentionally stays structural (format, sign, timezone-awareness)
here. Business-level integrity checks (bid <= ask, ratio factor errors,
barrier plausibility, ...) belong to ``pricing/integrity.py`` and operate on
already-valid instances of these models.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated
from uuid import uuid4

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.1.0"

_ISIN_RE = re.compile(r"^[A-Z0-9]{12}$")


def _check_tz_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"datetime must be timezone-aware, got naive value {value!r}")
    return value


def _check_isin(value: str) -> str:
    candidate = value.strip().upper()
    if not _ISIN_RE.match(candidate):
        raise ValueError(f"invalid ISIN {value!r}: expected 12 alphanumeric characters")
    return candidate


def _check_positive(value: float) -> float:
    if not (value > 0):
        raise ValueError(f"value must be > 0, got {value!r}")
    return value


def _check_unit_interval(value: float) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"value must be within [0, 1], got {value!r}")
    return value


# Reusable annotated types -----------------------------------------------

TzAwareDatetime = Annotated[datetime, AfterValidator(_check_tz_aware)]
OptionalTzAwareDatetime = Annotated[
    datetime | None,
    AfterValidator(lambda v: _check_tz_aware(v) if v is not None else v),
]
IsinStr = Annotated[str, AfterValidator(_check_isin)]
PositiveFloat = Annotated[float, AfterValidator(_check_positive)]
UnitFloat = Annotated[float, AfterValidator(_check_unit_interval)]


# Enums --------------------------------------------------------------------


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"


class ProductType(StrEnum):
    TURBO_OPEN_END = "turbo_open_end"  # barrier == financing level, open-ended
    TURBO_CLASSIC = "turbo_classic"  # barrier == strike, fixed maturity
    MINI_FUTURE = "mini_future"  # barrier != financing level (stop-loss buffer)
    UNKNOWN = "unknown"


class Category(StrEnum):
    ACTIONABLE = "ACTIONABLE"
    WATCH = "WATCH"
    REJECT = "REJECT"
    DATA_QUALITY = "DATA_QUALITY"


class HealthStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class PositionStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


# Core models ----------------------------------------------------------------


class Provenance(BaseModel):
    """Common lineage/freshness metadata carried by every observed record.

    ``available_at`` is the point in time at which this observation could
    legitimately have been used by a model (enforced elsewhere as
    ``available_at <= prediction_time``, CLAUDE.md rule 5). ``observation_time``
    is when the underlying fact became true (e.g. a quote's own timestamp
    when known, else same as ``available_at``); ``retrieved_at`` is purely
    when *this process* fetched it.
    """

    model_config = ConfigDict(extra="forbid")

    observation_time: TzAwareDatetime
    available_at: TzAwareDatetime
    retrieved_at: TzAwareDatetime
    source_timestamp: OptionalTzAwareDatetime = None
    source: str
    schema_version: str = SCHEMA_VERSION
    parser_version: str
    is_stale: bool = False
    quality_score: UnitFloat


class ProductSnapshot(Provenance):
    """A single point-in-time quote/state observation of one tradeable certificate."""

    isin: IsinStr
    wkn: str | None = None
    issuer: str
    venue: str
    underlying_raw: str
    underlying_id: str | None = None
    direction: Direction
    product_type: ProductType
    financing_level: float | None = None
    knockout_barrier: float | None = None
    ratio: PositiveFloat  # Bezugsverhaeltnis as a decimal, e.g. 0.01 for 100:1
    currency: str = "EUR"
    underlying_currency: str | None = None
    quanto: bool | None = None
    open_end: bool
    maturity: date | None = None
    first_trading_day: date | None = None
    bid: float | None = None
    ask: float | None = None
    bid_size: float | None = None
    ask_size: float | None = None
    quote_timestamp: OptionalTzAwareDatetime = None
    quote_presence: bool | None = None
    bid_only: bool = False
    knocked_out: bool = False
    trading_hours: str | None = None
    product_age_days: int | None = None
    underlying_price_ref: float | None = None
    # This reference price's OWN observation timestamp, when the source
    # exposes one (e.g. BNP's `first.priceDate`) -- distinct from
    # `quote_timestamp` (the product's own bid/ask timestamp). The two are
    # NOT interchangeable: BNP batches/throttles `first.price` updates
    # independently of (and less frequently than) individual product
    # bid/ask ticks (Build Contract BEFUND 1 measurement:
    # `pipeline/scan.py._resolve_spot` previously used `quote_timestamp` as
    # a freshness proxy for `underlying_price_ref`, which is wrong whenever
    # the two update at different cadences). `None` when the source gives no
    # independent timestamp for its reference price (e.g. Citi, which never
    # populates `underlying_price_ref` at all) -- never guessed.
    underlying_price_ref_timestamp: OptionalTzAwareDatetime = None
    raw_hash: str  # sha256 of the raw source record, for reproducibility


class UnderlyingBar(Provenance):
    """A single OHLC(V) bar for a canonical underlying_id."""

    underlying_id: str
    ts: TzAwareDatetime
    interval: str = "1d"
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class SourceHealthRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    checked_at: TzAwareDatetime
    availability: UnitFloat
    freshness: UnitFloat
    missingness: UnitFloat
    schema_consistency: UnitFloat
    cross_source_agreement: UnitFloat | None = None
    score: UnitFloat
    status: HealthStatus
    message: str


class SignalSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_id: str
    signal_version_hash: str
    underlying_id: str
    prediction_time: TzAwareDatetime
    frozen_at: TzAwareDatetime
    score: float
    components: dict[str, float]
    direction_hint: Direction | None = None
    threshold: float
    config_hash: str
    git_commit: str | None = None
    data_snapshot_hash: str


class CostDecomposition(BaseModel):
    """Ask-price decomposition into intrinsic value plus its cost components.

    All absolute fields are in product currency, per unit (one certificate).
    The ``*_pct`` fields are the same quantities expressed relative to ``ask``.
    """

    model_config = ConfigDict(extra="forbid")

    ask: float
    bid: float
    mid: float
    intrinsic: float
    trading_spread_component: float  # ask - mid
    fair_gap_premium: float
    financing_drag: float
    issuer_margin: float
    spread_pct: float
    gap_premium_pct: float
    financing_drag_pct: float
    issuer_margin_pct: float


class CandidateEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    candidate_id: str
    isin: IsinStr
    wkn: str | None = None
    issuer: str
    underlying_id: str
    direction: Direction
    category: Category
    reasons: list[str]
    leverage: float | None = None
    leverage_bucket: str | None = None
    distance_to_barrier_pct: float | None = None
    distance_to_barrier_sigma: float | None = None
    costs: CostDecomposition | None = None
    realized_financing_spread: float | None = None
    # {"3d": .., "5d": .., "7d": .., "10d": .., "14d": ..}
    financing_cost_horizon_pct: dict[str, float]
    cross_issuer_residual_zscore: float | None = None
    issuer_markup_score: float | None = None
    quote_dislocation_score: float | None = None
    wrapper_edge: float | None = None
    liquidity_factor: float | None = None
    integrity_passed: bool
    # never set in this milestone; the ACTIONABLE gate requires this to be not-None
    lcb_ev: float | None = None
    # "Cost per exposure (h)": total round-trip cost over the scan horizon
    # (spread + gap premium + financing + max(issuer margin, 0)), as a %
    # of ask, divided by leverage -- i.e. re-expressed as a % of
    # UNDERLYING exposure rather than capital employed, so it does not
    # mechanically favor low-leverage products (Build Contract BEFUND 2:
    # "kein pauschales Hebelziel" -- see pipeline/scan.py for the formula
    # and rationale). Low = cheap per unit of underlying exposure; a cost
    # ranking, NOT a return forecast. `None` for a product with no ask
    # (leverage cannot be computed).
    cost_rank_score: float | None = None


# Master data & operational models (not in the spec excerpt, needed by the
# storage layer for `instruments`, `positions_manual` and
# `notifications_sent`) ------------------------------------------------------


class Instrument(BaseModel):
    """Slowly-changing master data for one ISIN, derived from ProductSnapshots.

    Unlike ``ProductSnapshot`` (one row per observation), this is one row per
    ISIN, upserted as new snapshots arrive, tracking the fields that rarely or
    never change plus the observation window in which we have seen it.
    """

    model_config = ConfigDict(extra="forbid")

    isin: IsinStr
    wkn: str | None = None
    issuer: str
    underlying_id: str | None = None
    underlying_raw: str
    direction: Direction
    product_type: ProductType
    ratio: PositiveFloat
    currency: str = "EUR"
    underlying_currency: str | None = None
    quanto: bool | None = None
    open_end: bool
    maturity: date | None = None
    first_trading_day: date | None = None
    venue: str
    first_seen_at: TzAwareDatetime
    last_seen_at: TzAwareDatetime

    @classmethod
    def from_snapshot(
        cls, snapshot: ProductSnapshot, *, first_seen_at: datetime | None = None
    ) -> Instrument:
        """Build (or refresh) an Instrument record from a ProductSnapshot.

        ``first_seen_at`` should be preserved from any existing row on
        upsert; when omitted it defaults to the snapshot's own
        ``observation_time`` (i.e. "first time we saw it is now").
        """
        seen_at = first_seen_at if first_seen_at is not None else snapshot.observation_time
        return cls(
            isin=snapshot.isin,
            wkn=snapshot.wkn,
            issuer=snapshot.issuer,
            underlying_id=snapshot.underlying_id,
            underlying_raw=snapshot.underlying_raw,
            direction=snapshot.direction,
            product_type=snapshot.product_type,
            ratio=snapshot.ratio,
            currency=snapshot.currency,
            underlying_currency=snapshot.underlying_currency,
            quanto=snapshot.quanto,
            open_end=snapshot.open_end,
            maturity=snapshot.maturity,
            first_trading_day=snapshot.first_trading_day,
            venue=snapshot.venue,
            first_seen_at=seen_at,
            last_seen_at=snapshot.observation_time,
        )


class ManualPosition(BaseModel):
    """A manually entered, manually closed position (`turboedge position ...`).

    This milestone never re-evaluates or auto-manages positions; it only
    stores what the user tells it via the CLI.
    """

    model_config = ConfigDict(extra="forbid")

    position_id: str = Field(default_factory=lambda: uuid4().hex)
    wkn: str
    isin: IsinStr | None = None
    qty: PositiveFloat
    entry_price: PositiveFloat
    entry_date: date
    exit_price: float | None = None
    exit_date: date | None = None
    status: PositionStatus = PositionStatus.OPEN
    created_at: TzAwareDatetime
    updated_at: TzAwareDatetime


class NotificationRecord(BaseModel):
    """One row of the notifications_sent dedup table (notifications/dedup.py)."""

    model_config = ConfigDict(extra="forbid")

    notification_hash: str
    candidate_id: str | None = None
    category: str
    sent_at: TzAwareDatetime
    subject: str
