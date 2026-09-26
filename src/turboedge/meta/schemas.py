"""Meta-layer decision schemas (Phase 1, M1).

The meta layer answers a question the existing pipeline never asks: *does
the system know enough in this situation to be worth listening to?* It runs
strictly in shadow mode -- it records what it would have decided, changes no
model weight, no gate and no threshold.

Every field here is a number the controller actually computed, never a
narrative. `reasons` is rendered from those numbers, not written freehand:
an explanation that cannot be traced back to a value is exactly the kind of
confident-sounding artefact this layer exists to prevent.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from turboedge.storage.schemas import SCHEMA_VERSION, TzAwareDatetime, UnitFloat


class MetaDecisionKind(StrEnum):
    """What the controller would do, if it were allowed to act.

    `WATCH_ONLY` is deliberately distinct from `ABSTAIN`: the first says the
    situation is readable but not worth acting on, the second says the
    system cannot read it. Collapsing them would lose the single most
    interesting signal this layer produces -- how often it does not know.
    """

    PROCEED = "PROCEED"
    WATCH_ONLY = "WATCH_ONLY"
    ABSTAIN = "ABSTAIN"


class ModelTrust(BaseModel):
    """Per-model trust, with every factor kept separately.

    The factors are stored rather than only their product because a single
    score cannot be argued with. "Trust 0.31" is unfalsifiable; "0.31 because
    this regime has 4 historical observations and the model is in drift" can
    be checked, and disagreed with.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    signal_family: str
    historical_oos_quality: float | None = None
    calibration_quality: float | None = None
    regime_similarity: float | None = None
    data_quality: UnitFloat
    drift_penalty: UnitFloat
    uncertainty_penalty: UnitFloat
    trust_score: UnitFloat
    #: Factors that could not be computed at all (no history, no labels).
    #: Non-empty here is itself information: it means trust rests on fewer
    #: legs than the formula suggests.
    missing_factors: list[str] = Field(default_factory=list)


class DecisionConfidence(BaseModel):
    """Uncertainty decomposed, because the parts behave differently.

    A tail-risk error matters far more for a knock-out product than an equal
    error in the expected return, so collapsing these into one "confidence:
    72%" would discard the distinction that matters most here.
    """

    model_config = ConfigDict(extra="forbid")

    epistemic_uncertainty: UnitFloat
    data_uncertainty: UnitFloat
    regime_uncertainty: UnitFloat
    model_disagreement: UnitFloat
    calibration_uncertainty: UnitFloat
    product_data_uncertainty: UnitFloat
    abstain_score: UnitFloat


class MetaDecision(BaseModel):
    """One shadow decision, persisted per (run, underlying, horizon).

    Carries `config_hash`/`git_commit` for the same reason every other
    persisted record does (CLAUDE.md rule 33): a decision that cannot be
    reproduced cannot be audited later, and the whole point of shadow mode
    is being able to ask afterwards whether this layer was right.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    underlying_id: str
    horizon_days: int = Field(gt=0)
    prediction_time: TzAwareDatetime
    decided_at: TzAwareDatetime

    volatility_regime: str
    trend_regime: str
    regime_observation_count: int = Field(ge=0)

    available_models: list[str]
    selected_models: list[str]
    model_weights: dict[str, float]
    model_trust: list[ModelTrust]

    confidence: DecisionConfidence
    final_confidence: UnitFloat
    decision: MetaDecisionKind
    reasons: list[str]

    #: Always true in Phase 1. Persisted rather than assumed so a later
    #: query can separate shadow rows from live ones without guessing.
    shadow_mode: bool = True

    config_hash: str
    git_commit: str | None = None
    schema_version: str = SCHEMA_VERSION


__all__ = [
    "DecisionConfidence",
    "MetaDecision",
    "MetaDecisionKind",
    "ModelTrust",
]
