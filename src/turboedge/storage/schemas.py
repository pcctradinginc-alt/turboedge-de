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

import hashlib
import re
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import uuid4

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

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


# -- W6 learning enums (Master Spec §20-27, §46; Build Contract v2 W6) -------


class LedgerEntryStatus(StrEnum):
    """Lifecycle of one `forward_ledger` row (Master Spec §25)."""

    OPEN = "open"
    LABELED = "labeled"
    EXPIRED_NO_DATA = "expired_no_data"


class ExitReason(StrEnum):
    """How a forward-ledger entry's exit was determined (labeler.py)."""

    HORIZON = "horizon"  # normal exit at horizon, real bid quote found
    KO = "ko"  # knocked out before/at horizon
    NO_EXIT_QUOTE_CONSERVATIVE = "no_exit_quote_conservative"  # ambiguous, conservative value used
    EXPIRED_NO_DATA = "expired_no_data"  # neither quote nor bars available at all


class ModelStatus(StrEnum):
    """`model_registry.status` (Master Spec §20-21)."""

    PROTECTED = "protected"  # never demoted/deleted (e.g. TSMOM baseline)
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    DORMANT = "dormant"


class TrialStatus(StrEnum):
    """`research_trials.status` (Master Spec §27.1, GOVERNANCE.md §1.3)."""

    EXPERIMENTAL = "experimental"
    PROMOTED = "promoted"
    DORMANT = "dormant"
    REJECTED = "rejected"


class ShadowPortfolioKind(StrEnum):
    """`shadow_portfolio.portfolio` values (Master Spec §46)."""

    TOP1 = "top1"
    TOP3 = "top3"
    TOP5 = "top5"
    RANDOM_VALID_TURBO = "random_valid_turbo"
    LOWEST_SPREAD = "lowest_spread"
    LOWEST_FINANCING_COST = "lowest_financing_cost"
    HIGHEST_LEVERAGE = "highest_leverage"
    LOWEST_LEVERAGE = "lowest_leverage"
    MEDIAN_PRODUCT = "median_product"


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
    # Issuer-reported annualized financing rate, decimal (e.g. 0.0622 for
    # 6.22% p.a.) -- Citi's `items[].fundingRate` (Contract v3 "Finanzierungs-
    # spread-Prioritaet" (b)): the issuer's own funding spread is then
    # `financing_rate - ref_rate`. Additive, optional: `None` for sources
    # that do not expose this field (e.g. BNP, gettex) -- never guessed.
    financing_rate: float | None = None
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
    # Set by the EV pipeline when the path model is available; required by the ACTIONABLE gate
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


# -- W6: Forward Ledger, Learning & Governance models -------------------------
#
# Master Spec §20-27 ("Self-Learning Architektur" through "Research
# Governance") and §46 ("Shadow Portfolio"). These are the durable,
# append-only records that make the system's learning reproducible and
# auditable (CLAUDE.md rules 31-33). Storage lives in `storage/duckdb.py`
# (`forward_ledger`, `ledger_labels`, `strategy_posteriors`, `model_registry`,
# `model_weight_history`, `research_trials`, `drift_events`,
# `shadow_portfolio`); business logic lives in `turboedge/learning/*`.


def compute_entry_id(run_id: str, candidate_id: str, horizon_days: int) -> str:
    """Forward-ledger primary key: sha256(run_id|candidate_id|horizon)[:20].

    Deterministic so the same (run_id, candidate_id, horizon_days) triple
    always produces the same `entry_id`, which is what makes
    `ForwardLedger.record` idempotent (Build Contract v2, W6 requirement 1/2).
    """
    payload = f"{run_id}|{candidate_id}|{horizon_days}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


class LedgerEntry(BaseModel):
    """One append-only `forward_ledger` row: everything knowable at prediction
    time about one (candidate, horizon) pair -- ACTIONABLE, WATCH, REJECT or a
    stratified shadow sample of otherwise-discarded candidates (Master Spec
    §25: "Nicht nur ACTIONABLE-Kandidaten speichern... stratified Shadow
    Sample verworfener Kandidaten ist Pflicht").

    Never mutated after `ForwardLedger.record()` inserts it, except for the
    `status` transition `open -> labeled` performed by
    `ForwardLedger.attach_label()` (the substantive prediction fields below
    are immutable; the exit-side facts live in the separate, also-append-only
    `LedgerLabel`).
    """

    model_config = ConfigDict(extra="forbid")

    # -- identity / provenance (Spec §25 + §48 reproducibility) --------------
    entry_id: str = ""  # filled by the validator below if left empty
    run_id: str
    candidate_id: str
    signal_id: str
    signal_version_hash: str
    trial_id: str
    prediction_time: TzAwareDatetime
    underlying: str
    direction: Direction
    horizon_days: int = Field(gt=0)
    regime_bucket: str | None = None
    cluster_id: str | None = None
    feature_hash: str
    model_hash: str
    config_hash: str
    git_commit: str | None = None

    # -- selected product / entry terms --------------------------------------
    selected_wkn: str | None = None
    selected_isin: IsinStr
    issuer: str
    entry_bid: float | None = None
    entry_ask: PositiveFloat
    entry_spread: float = Field(ge=0)
    entry_quote_timestamp: TzAwareDatetime
    entry_underlying_timestamp: TzAwareDatetime
    financing_level_entry: float | None = None
    barrier_entry: float | None = None
    ratio: PositiveFloat
    fx: PositiveFloat

    # -- prediction / gating (Spec §19 gates, §15 winner's curse) ------------
    predicted_return: float
    p_profit: UnitFloat
    p_ko: UnitFloat
    expected_shortfall: float
    lcb_ev: float
    uncertainty: float = Field(ge=0)
    shrinkage_intensity: UnitFloat

    # -- Build Contract v2 W6 additions --------------------------------------
    category: Category
    is_shadow: bool
    shadow_stratum: str | None = None
    suggested_position_fraction: float | None = Field(default=None, ge=0)
    exit_due: date
    # Counterfactual set (Master Spec §24): ISINs of same-underlying,
    # same-direction alternatives with similar leverage/barrier, different
    # issuers -- evaluated post-hoc by `learning/counterfactual.py`.
    alternatives: list[IsinStr] = Field(default_factory=list)
    # Feature vector frozen at `prediction_time`, for positive/negative
    # memory (Spec §22/§23) and full reproducibility (Spec §48). Stored as a
    # JSON column by storage/duckdb.py.
    feature_snapshot: dict[str, float] = Field(default_factory=dict)
    status: LedgerEntryStatus = LedgerEntryStatus.OPEN

    @model_validator(mode="after")
    def _fill_entry_id(self) -> LedgerEntry:
        if not self.entry_id:
            self.entry_id = compute_entry_id(self.run_id, self.candidate_id, self.horizon_days)
        return self


class LedgerLabel(BaseModel):
    """Append-only exit-side counterpart of one `LedgerEntry`, attached once
    the entry's `exit_due` date has passed (`learning/labeler.py`).

    CLAUDE.md rule 17: ambiguous bars are never optimistically resolved --
    `ambiguous_path=True` always pairs with a conservative (never-optimistic)
    `realized_selected_pnl`.
    """

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    labeled_at: TzAwareDatetime
    exit_bid: float | None = None
    exit_quote_timestamp: OptionalTzAwareDatetime = None
    financing_level_exit: float | None = None
    exit_reason: ExitReason
    realized_selected_pnl: float | None = None  # net return: exit_bid/entry_ask - 1
    underlying_pnl: float | None = None
    median_turbo_pnl: float | None = None
    best_turbo_pnl: float | None = None
    ideal_turbo_pnl: float | None = None
    mfe: float | None = None
    mae: float | None = None
    ko_hit: bool
    time_to_ko_days: int | None = None
    ambiguous_path: bool


class ModelRegistryEntry(BaseModel):
    """One row of `model_registry` (Master Spec §20-21, `learning/registry.py`).

    `status=PROTECTED` (the TSMOM baseline) is never deleted or set to
    another status by the registry -- CLAUDE.md rule 10.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    model_hash: str
    signal_family: str
    status: ModelStatus
    weight: float = Field(ge=0)
    params: dict[str, Any] = Field(default_factory=dict)
    trial_id: str | None = None
    created_at: TzAwareDatetime
    updated_at: TzAwareDatetime


class ResearchTrial(BaseModel):
    """One row of `research_trials` (Master Spec §27.1, `learning/trials.py`)."""

    model_config = ConfigDict(extra="forbid")

    trial_id: str
    kind: str
    description: str
    created_at: TzAwareDatetime
    quarter: str  # e.g. "2026Q3"
    status: TrialStatus = TrialStatus.EXPERIMENTAL


class ShadowPosition(BaseModel):
    """One row of `shadow_portfolio` (Master Spec §46): one baseline
    portfolio's paper position for one scan run, so a monthly report can
    compare "intelligent ranking" against trivial baselines."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    portfolio: ShadowPortfolioKind
    isin: IsinStr
    horizon_days: int = Field(gt=0)
    entry_ask: PositiveFloat
    exit_due: date
    realized_net_return: float | None = None
    created_at: TzAwareDatetime


class DriftEvent(BaseModel):
    """One row of `drift_events` (Master Spec §30, `learning/drift.py`).

    CLAUDE.md rule 32: drift reduces weights, it never auto-deletes a model
    -- `action` records the *recommendation* (e.g. "weight_reduction"), the
    actual weight change is applied by `learning/ensemble_weights.py`."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    detected_at: TzAwareDatetime
    stream_id: str  # e.g. "calibration_error:tsmom:7d"
    signal_family: str | None = None
    metric: str  # e.g. "calibration_error", "return"
    ph_statistic: float
    threshold: float
    action: str
    details: dict[str, Any] = Field(default_factory=dict)


# -- Integration wave (Contract v3): forecasts, position reevaluation, ---------
# walk-forward persistence. Additive tables/models only (storage/duckdb.py).


class ForecastRecord(BaseModel):
    """One persisted `models.forecast.HorizonForecast` (ensemble or single
    model), frozen at scan time (Contract v3 Abschnitt B step 1: "Ergebnis +
    model_hash ... in ... neuer Tabelle forecasts persistieren").

    Reproducibility fields (`config_hash`/`git_commit`) mirror
    `SignalSnapshot` so a forecast is as fully traceable as the TSMOM signal
    it accompanies (CLAUDE.md rule 33).
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    underlying_id: str
    horizon_days: int = Field(gt=0)
    prediction_time: TzAwareDatetime
    frozen_at: TzAwareDatetime
    p_up: UnitFloat
    mean: float
    sigma: float = Field(ge=0)
    quantiles: dict[str, float]
    expected_shortfall_05: float
    uncertainty: float = Field(ge=0)
    model_id: str
    model_hash: str
    signal_family: str
    n_train: int = Field(ge=0)
    n_effective: float = Field(ge=0)
    component_weights: dict[str, float] = Field(default_factory=dict)
    config_hash: str
    git_commit: str | None = None


class PositionEvaluationStatus(StrEnum):
    """`position_evaluations.status` (Contract v3 Abschnitt D)."""

    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    INVALIDATED = "INVALIDATED"


class PositionEvaluation(BaseModel):
    """One daily re-evaluation of an open manual position (Contract v3
    Abschnitt D, `positions/reevaluate.py`). Append-only: a new row per
    ``(position_id, as_of)``."""

    model_config = ConfigDict(extra="forbid")

    position_id: str
    as_of: TzAwareDatetime
    wkn: str
    isin: IsinStr | None = None
    underlying_id: str | None = None
    status: PositionEvaluationStatus
    reasons: list[str]
    current_bid: float | None = None
    quote_timestamp: OptionalTzAwareDatetime = None
    remaining_horizon_days: int | None = None
    remaining_lcb_ev: float | None = None
    remaining_p_ko: float | None = None
    remaining_p_profit: float | None = None
    unrealized_return: float | None = None
    data_quality_ok: bool
    config_hash: str
    git_commit: str | None = None


class WalkforwardResultRecord(BaseModel):
    """One persisted `backtest.walkforward.WalkForwardResult` row (Contract
    v3 coordinator addition): makes W4's walk-forward evaluation readable by
    `reporting/weekly.run_research_tournament` once that module is updated to
    query it (see Kurzbericht)."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    model_hash: str | None = None
    signal_family: str
    underlying_id: str
    horizon_days: int = Field(gt=0)
    evaluated_at: TzAwareDatetime
    n_folds: int = Field(ge=0)
    brier: float
    brier_null: float | None = None
    log_loss: float
    ece: float
    hit_rate: float
    mean_oos_return: float
    psr: float
    n_effective: float = Field(ge=0)
    config_hash: str
    git_commit: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
