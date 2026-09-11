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

from turboedge.provenance import sha256_json

CONFIG_FILES: tuple[str, ...] = (
    "default.yaml",
    "sources.yaml",
    "risk.yaml",
    "universe.yaml",
    "models.yaml",
    "gmail.yaml",
    "governance.yaml",
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
