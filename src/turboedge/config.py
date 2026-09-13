"""Typed configuration loading for the ``configs/*.yaml`` files.

``load_config`` reads and validates every YAML file into a single frozen
:class:`TurboEdgeConfig`. Validation errors are collected per-file and raised
as one :class:`ConfigError` with a human-readable summary rather than a raw
pydantic traceback, so a broken config fails fast and legibly whether it is
loaded from a CLI, a test, or a scheduled job.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from turboedge.learning.posterior import PosteriorConfig
from turboedge.learning.trials import TrialsConfig
from turboedge.models.forecast import HORIZONS
from turboedge.provenance import sha256_json
from turboedge.ranking.cluster import ClusterConfig
from turboedge.ranking.ev import EvConfig
from turboedge.ranking.lcb import LcbConfig
from turboedge.ranking.shrinkage import ShrinkageConfig
from turboedge.ranking.sizing import SizingConfig
from turboedge.ranking.utility import UtilityConfig
from turboedge.state.retention import RetentionConfig

CONFIG_FILES: tuple[str, ...] = (
    "default.yaml",
    "sources.yaml",
    "risk.yaml",
    "universe.yaml",
    "models.yaml",
    "gmail.yaml",
    "governance.yaml",
    "forecast.yaml",
    "simulation.yaml",
    "ranking.yaml",
    "learning.yaml",
    "reporting.yaml",
    "state.yaml",
)


class ConfigError(Exception):
    """Raised when a config directory is missing files or fails validation."""


def default_config_dir() -> Path:
    """``$TURBOEDGE_CONFIG_DIR`` if set, else ``./configs``."""
    return Path(os.environ.get("TURBOEDGE_CONFIG_DIR", "./configs"))


def default_state_dir() -> Path:
    """``$TURBOEDGE_STATE_DIR`` if set, else ``./state``."""
    return Path(os.environ.get("TURBOEDGE_STATE_DIR", "./state"))


# -- default.yaml -----------------------------------------------------------


class DefaultConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    log_format: Literal["console", "json"] = "console"
    default_top: int = Field(gt=0, default=20)
    default_horizon: str = "7d"
    timezone: str = "UTC"
    http_default_timeout_s: float = Field(gt=0, default=10.0)
    http_default_user_agent: str = "TurboEdge-DE-Research/0.1"


# -- sources.yaml -------------------------------------------------------------


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    base_url: str
    timeout_s: float = Field(gt=0)
    min_interval_s: float = Field(ge=0)
    max_pages: int = Field(gt=0)
    user_agent: str


# sources.yaml is a flat mapping of source name -> SourceConfig at the YAML
# top level; represented as a plain dict rather than a wrapper model so
# `cfg.sources["ecb"]` works directly.
SourcesConfig = dict[str, SourceConfig]


# -- risk.yaml -----------------------------------------------------------------


class IntegrityTolerances(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ratio_factor_tolerance_pct: float = Field(gt=0)
    bid_ask_epsilon: float = Field(ge=0)
    max_financing_level_jump_pct: float = Field(gt=0)
    # Threshold (fraction of ask) above which pricing/integrity.check_product's
    # coarse (mid - intrinsic) / ask "raw premium" ratio is flagged
    # "premium_pct_high". Consumed by the scan pipeline as `margin_warn_pct`.
    margin_warn_pct: float = Field(gt=0)


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_spread_pct: float = Field(gt=0)
    max_quote_age_s: float = Field(gt=0)
    min_leverage: float = Field(gt=0)
    max_leverage: float = Field(gt=0)
    min_distance_to_barrier_sigma: float = Field(ge=0)
    integrity_tolerances: IntegrityTolerances
    # Plausibility band on the ANNUALIZED implied financing spread returned by
    # pricing.financing.implied_financing_spread (the day-over-day financing
    # -level change scaled by 360/dt) -- NOT a raw day-over-day price-jump
    # threshold. See pricing/financing.py: financing_spread_history().
    financing_adjustment_jump_threshold_pct: float = Field(gt=0)
    default_financing_spread: float
    reference_rate_fallback: float
    # Reference trade notional (EUR) used by ranking/liquidity.py's
    # quote_size_coverage to score how much of a "normal" trade the displayed
    # ask size could fill.
    required_notional_eur: float = Field(gt=0)
    # Build Contract BEFUND 1: maximum relative deviation of a source-reported
    # `underlying_price_ref` from the cross-issuer consensus spot before
    # pipeline.scan._resolve_spot rejects it (falls back to consensus,
    # warning "spot_ref_rejected") even though its own timestamp
    # (`underlying_price_ref_timestamp`) is fresh. Guards against a reference
    # price that updates on its own (slower) cadence and has silently
    # drifted from the market the product's own bid/ask is actually quoting.
    spot_ref_max_deviation_pct: float = Field(gt=0)


# -- universe.yaml --------------------------------------------------------------


class UnderlyingEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    enabled: bool
    asset_class: str
    cluster: str
    currency: str
    yfinance_ticker: str


class UniverseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    underlyings: list[UnderlyingEntry]

    def enabled_ids(self) -> list[str]:
        return [u.id for u in self.underlyings if u.enabled]


# -- models.yaml -----------------------------------------------------------------


class TsmomConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    signal_id: str = "tsmom_horizon_norm_v1"
    lookbacks: list[int]
    ewma_lambda: float = Field(gt=0, lt=1)
    clip: float = Field(gt=0)
    threshold: float
    version: str


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tsmom_horizon_norm_v1: TsmomConfig


# -- gmail.yaml ------------------------------------------------------------------


class GmailConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool
    smtp_host: str
    smtp_port: int = Field(gt=0, le=65535)
    send_on: list[str]
    subject_prefix: str
    daily_digest: bool


# -- governance.yaml -------------------------------------------------------------


class GovernanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    research_adjustment_budget_per_quarter: int = Field(ge=0)
    ladder_min_improvement: float = Field(ge=0)
    fdr_alpha: float = Field(gt=0, lt=1)


# -- forecast.yaml (Contract v3 Abschnitt A) -----------------------------------


class ForecastConfig(BaseModel):
    """Walk-forward / ensemble-fitting parameters shared by the scan pipeline
    (``pipeline/scan.py``), ``turboedge forecast`` and ``turboedge backtest``.

    Individual models (``models/directional.py``'s ``TsmomForecastConfig``/
    ``LogisticDirectionModelConfig``/``NullModelConfig``) keep their own
    hyperparameters; this section only holds the parameters that are
    external to any single model: the horizon ladder, and the walk-forward
    split geometry (``min_train``/``step``/``embargo``, Master Spec §28).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    horizons: list[int] = Field(default_factory=lambda: list(HORIZONS))
    min_train: int = Field(gt=0, default=250)
    step: int = Field(gt=0, default=10)
    embargo: int = Field(ge=0, default=max(HORIZONS))
    # Weight assigned to a model newly registered in `model_registry` (no
    # prior `model_weight_history`) before the learning loop has any
    # evidence to reweight it -- equal-weight until performance data exists.
    default_new_model_weight: float = Field(gt=0, default=1.0)


# -- simulation.yaml (Contract v3 Abschnitt A) ---------------------------------


class SimulationConfig(BaseModel):
    """``simulation/paths.py`` parameters shared by every EV evaluation this
    scan run performs (Contract v3: "n_paths, method, block_size,
    lookback_days, seed")."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_paths: int = Field(gt=0, default=2000)
    method: str = "vol_scaled_bootstrap"
    block_size: int = Field(gt=0, default=5)
    lookback_days: int = Field(gt=0, default=750)
    seed: int = 20260101


# -- ranking.yaml (Contract v3: ev/utility/sizing/cluster) ---------------------


class ScanCandidateFilterConfig(BaseModel):
    """Pre-filter applied *before* the (expensive) EV simulation (Contract
    v3 Abschnitt B "Performance"): only candidates that already pass every
    hard, simulation-free gate (no bid_only/knocked_out, fresh quote, spread
    within limit, leverage within the configured band, barrier distance >=
    the minimum sigma, liquidity above threshold) are simulated at all, and
    of those only the cheapest ``max_candidates_per_bucket`` per (direction,
    leverage_bucket) group. Documented, configurable, and never applied to
    the shadow sample (``pipeline/scan.py`` draws the shadow sample from the
    *full* pre-filter candidate pool, Spec §25 selection-bias protection).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_candidates_per_bucket: int = Field(gt=0, default=25)
    min_liquidity_factor: float = Field(ge=0.0, le=1.0, default=0.05)


class RankingConfig(BaseModel):
    """Consolidates ``ranking/{shrinkage,lcb,utility,sizing,cluster}.py``'s
    own configs plus the handful of ``ranking/ev.py``-only fields
    (``z_pessimistic``/``z_optimistic``/``include_optimistic``/
    ``default_liquidity_factor``) that are not owned by ``simulation.yaml``
    (path-generation parameters) nor by any single nested config. Built into
    a full ``ranking.ev.EvConfig`` by :func:`build_ev_config` below.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    z_pessimistic: float = Field(gt=0.0, default=1.645)
    z_optimistic: float = Field(gt=0.0, default=1.645)
    include_optimistic: bool = False
    default_liquidity_factor: float = Field(gt=0.0, le=1.0, default=1e-6)
    shrinkage: ShrinkageConfig = Field(default_factory=ShrinkageConfig)
    lcb: LcbConfig = Field(default_factory=LcbConfig)
    utility: UtilityConfig = Field(default_factory=UtilityConfig)
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    scan_filter: ScanCandidateFilterConfig = Field(default_factory=ScanCandidateFilterConfig)


def build_ev_config(cfg: TurboEdgeConfig) -> EvConfig:
    """Build a ``ranking.ev.EvConfig`` from ``cfg.simulation`` + ``cfg.ranking``.

    ``ranking/ev.py`` is a finished module (Contract v3: used, not rebuilt)
    that owns its own ``EvConfig`` nesting ``shrinkage``/``lcb``/``utility``/
    ``sizing``; this helper is the integration wave's single place that
    assembles it from the YAML-backed sections so every caller (scan
    pipeline, ``forecast``/``backtest`` CLI diagnostics) constructs it
    identically.
    """
    return EvConfig(
        n_paths=cfg.simulation.n_paths,
        z_pessimistic=cfg.ranking.z_pessimistic,
        z_optimistic=cfg.ranking.z_optimistic,
        include_optimistic=cfg.ranking.include_optimistic,
        path_method=cfg.simulation.method,
        block_size=cfg.simulation.block_size,
        lookback_days=cfg.simulation.lookback_days,
        default_liquidity_factor=cfg.ranking.default_liquidity_factor,
        shrinkage=cfg.ranking.shrinkage,
        lcb=cfg.ranking.lcb,
        utility=cfg.ranking.utility,
        sizing=cfg.ranking.sizing,
    )


# -- learning.yaml (Contract v3 Abschnitt A) -----------------------------------


class LearningConfig(BaseModel):
    """``learning/*``'s own configs (``posterior``, ``trials``), plus the
    exponential-reweighting hyperparameters (``eta``/``w_min``, Master Spec
    §21) consumed by ``learning.registry.ModelRegistry.update_weights``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    eta: float = Field(gt=0.0, default=0.5)
    w_min: float = Field(gt=0.0, default=0.01)
    # Max candidates drawn per (category, direction, leverage_bucket) stratum
    # for the forward-ledger's shadow sample (learning.ledger.select_shadow_sample,
    # Master Spec §25) -- an integration-level default (learning/ledger.py's
    # function takes this as a plain argument, it owns no XxxConfig of its own).
    shadow_sample_per_stratum: int = Field(gt=0, default=3)
    posterior: PosteriorConfig = Field(default_factory=PosteriorConfig)
    trials: TrialsConfig = Field(default_factory=TrialsConfig)


# -- reporting.yaml (Contract v3 Abschnitt A) ----------------------------------


class ReportingConfig(BaseModel):
    """YAML surface for ``reporting/{monthly,weekly}.py``'s own configs.

    Held as plain dicts here (rather than the real ``MonthlyReportConfig``/
    ``WeeklyTournamentConfig`` pydantic types) deliberately: importing
    ``turboedge.reporting`` at module scope would import that package's
    ``__init__`` (pulling in ``pipeline.scan`` -> ``adapters.registry`` ->
    ``turboedge.config`` again) before this module has finished defining
    ``TurboEdgeConfig`` -- a genuine circular import. :meth:`monthly_config`/
    :meth:`weekly_config` build the real, validated config objects lazily
    (imported only when actually called, by which point every module is
    fully loaded), so every field is still pydantic-validated -- just on
    first use rather than at config-load time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    monthly: dict[str, Any] = Field(default_factory=dict)
    weekly: dict[str, Any] = Field(default_factory=dict)

    def monthly_config(self) -> Any:
        from turboedge.reporting.monthly import MonthlyReportConfig

        return MonthlyReportConfig(**self.monthly)

    def weekly_config(self) -> Any:
        from turboedge.reporting.weekly import WeeklyTournamentConfig

        return WeeklyTournamentConfig(**self.weekly)


# -- state.yaml (Contract v3 Abschnitt A) --------------------------------------


class StateManagementConfig(BaseModel):
    """``state/retention.py``'s own config (``keep_days``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retention: RetentionConfig = Field(default_factory=RetentionConfig)


# -- aggregate ---------------------------------------------------------------


class TurboEdgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    default: DefaultConfig
    sources: SourcesConfig
    risk: RiskConfig
    universe: UniverseConfig
    models: ModelsConfig
    gmail: GmailConfig
    governance: GovernanceConfig
    forecast: ForecastConfig
    simulation: SimulationConfig
    ranking: RankingConfig
    learning: LearningConfig
    reporting: ReportingConfig
    state: StateManagementConfig


def _read_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except OSError as exc:
        raise ConfigError(f"could not read config file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        data = {}
    return data


def load_config(config_dir: str | Path) -> TurboEdgeConfig:
    """Load and validate every ``configs/*.yaml`` file under ``config_dir``.

    Raises:
        ConfigError: if the directory or any required file is missing, if a
            file is not valid YAML, or if the combined document fails
            pydantic validation. The message lists every problem found across
            all files, not just the first one.
    """
    directory = Path(config_dir)
    if not directory.is_dir():
        raise ConfigError(f"config directory does not exist: {directory}")

    missing = [name for name in CONFIG_FILES if not (directory / name).is_file()]
    if missing:
        raise ConfigError(f"missing config file(s) in {directory}: {', '.join(missing)}")

    raw: dict[str, Any] = {
        name.removesuffix(".yaml"): _read_yaml(directory / name) for name in CONFIG_FILES
    }

    document = {
        "default": raw["default"],
        "sources": raw["sources"],
        "risk": raw["risk"],
        "universe": raw["universe"],
        "models": raw["models"],
        "gmail": raw["gmail"],
        "governance": raw["governance"],
        "forecast": raw["forecast"],
        "simulation": raw["simulation"],
        "ranking": raw["ranking"],
        "learning": raw["learning"],
        "reporting": raw["reporting"],
        "state": raw["state"],
    }

    try:
        return TurboEdgeConfig.model_validate(document)
    except ValidationError as exc:
        details = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"invalid configuration in {directory}:\n{details}") from exc


def config_hash(cfg: TurboEdgeConfig) -> str:
    """SHA-256 hash over the canonical JSON encoding of the whole config.

    Stable across processes/machines for identical config content; used to
    stamp every ``runs`` row and ``SignalSnapshot`` for reproducibility.
    """
    return sha256_json(cfg)
