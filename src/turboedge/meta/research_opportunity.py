"""Research opportunities and their provenance (Phase 2, §6).

Phase 1 scored *models* against tables the repository actually holds. This
module scores *research questions*, and that is a materially harder problem
for one reason: almost nothing about an unrun experiment can be measured.
"Expected information gain" of a study nobody has performed is a judgement,
not an observation.

The failure mode is therefore obvious and worth naming: a priority score
assembled from six invented numbers looks exactly as authoritative as one
assembled from six measured ones, and would let the system launder guesses
into a ranked queue.

So every quantity here carries an `Estimate` with an explicit `basis`:

* ``MEASURED``  -- computed from repository state (failed hypotheses,
  research trials, stored observations, adapter health). Reproducible.
* ``DECLARED``  -- a human judgement recorded in the catalog, with the
  reasoning in `note`. Arguable, and meant to be argued with.
* ``UNKNOWN``   -- not available. Scored as a penalty, never silently
  defaulted to a neutral value.

This mirrors `trust.missing_factors` from Phase 1 deliberately. A score
whose inputs cannot be traced to one of these three states is not
explainable, whatever its reasons field says.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from turboedge.storage.schemas import SCHEMA_VERSION, TzAwareDatetime, UnitFloat


class EstimateBasis(StrEnum):
    """Where a number came from. See the module docstring."""

    MEASURED = "MEASURED"
    DECLARED = "DECLARED"
    UNKNOWN = "UNKNOWN"


class Estimate(BaseModel):
    """One quantity, with its origin attached.

    `note` is required for `DECLARED` and `UNKNOWN`: a judgement without a
    stated reason cannot be reviewed, and an unknown without a stated cause
    cannot be resolved.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float | None = None
    basis: EstimateBasis
    note: str = ""

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.basis is EstimateBasis.UNKNOWN:
            if self.value is not None:
                raise ValueError("an UNKNOWN estimate must not carry a value")
            if not self.note.strip():
                raise ValueError("an UNKNOWN estimate must say why it is unknown")
        else:
            if self.value is None:
                raise ValueError(f"a {self.basis} estimate must carry a value")
            if self.basis is EstimateBasis.DECLARED and not self.note.strip():
                raise ValueError("a DECLARED estimate must state its reasoning")
        return self

    @property
    def is_known(self) -> bool:
        return self.basis is not EstimateBasis.UNKNOWN

    @classmethod
    def measured(cls, value: float, note: str = "") -> Estimate:
        return cls(value=value, basis=EstimateBasis.MEASURED, note=note)

    @classmethod
    def declared(cls, value: float, note: str) -> Estimate:
        return cls(value=value, basis=EstimateBasis.DECLARED, note=note)

    @classmethod
    def unknown(cls, note: str) -> Estimate:
        return cls(value=None, basis=EstimateBasis.UNKNOWN, note=note)


class InformationFamily(StrEnum):
    """Coarse grouping used for redundancy detection (§7, §9).

    Two opportunities in the same family are assumed to compete for the
    same information; this is what makes "redundant feature family" a
    computable penalty rather than an opinion.
    """

    VOLATILITY_SURFACE = "volatility_surface"
    POSITIONING = "positioning"
    SENTIMENT = "sentiment"
    RATES_CREDIT = "rates_credit"
    BREADTH_DISPERSION = "breadth_dispersion"
    EVENT_RISK = "event_risk"
    PATH_STATISTICS = "path_statistics"
    CROSS_ASSET = "cross_asset"
    MICROSTRUCTURE = "microstructure"
    PRODUCT_SELECTION = "product_selection"


class ResearchStatus(StrEnum):
    """Lifecycle of one research question (§8).

    The system may compute and re-order priorities freely, but it may only
    ever place an entry in `PROPOSED`. Every later state requires a human
    act -- see `research_queue.approve`.
    """

    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    RUNNING = "RUNNING"
    MEASURED = "MEASURED"
    PROMOTED = "PROMOTED"
    DORMANT = "DORMANT"
    REJECTED = "REJECTED"


#: States the system itself is allowed to assign. Anything else is a human
#: decision, enforced in `research_queue`.
SYSTEM_ASSIGNABLE_STATES = frozenset({ResearchStatus.PROPOSED})


class ResearchOpportunity(BaseModel):
    """One catalog question, with everything needed to rank it (§6).

    Ranking inputs are `Estimate`s rather than bare floats so that a reader
    can see at a glance which parts of a priority score rest on measurement
    and which rest on judgement. In the shipped catalog that split is
    roughly: cost, complexity and leakage risk are declared; sample size,
    data availability and overlap are measured from repository state.
    """

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    description: str
    information_family: InformationFamily
    affected_underlyings: list[str] = Field(default_factory=list)
    affected_horizons: list[str] = Field(default_factory=list)

    # --- ranking inputs (§6) ---
    expected_information_gain: Estimate
    expected_economic_value: Estimate
    probability_of_resolving_uncertainty: Estimate
    implementation_cost: Estimate
    implementation_complexity: Estimate
    estimated_sample_size: Estimate
    current_uncertainty: Estimate
    data_availability: Estimate
    leakage_risk: Estimate
    overlap_with_existing_research: Estimate

    # --- lifecycle (§8) ---
    status: ResearchStatus = ResearchStatus.PROPOSED
    trial_id: str | None = None
    approved_by: str | None = None
    approved_at: TzAwareDatetime | None = None
    status_note: str = ""

    # --- provenance ---
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _check_approval_recorded(self) -> Self:
        """An entry past PROPOSED must say who moved it and why.

        Without this the human-approval requirement of §8 would be a
        convention rather than a constraint.
        """
        if self.status is not ResearchStatus.PROPOSED and not self.approved_by:
            raise ValueError(
                f"status {self.status} requires approved_by "
                "(§8: human approval required to leave PROPOSED)"
            )
        return self


class PriorityFactor(BaseModel):
    """One named contribution to a priority score, kept for explainability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    value: float
    basis: EstimateBasis
    note: str = ""


class ResearchPriority(BaseModel):
    """The scored result for one opportunity (§7).

    `score` is never returned on its own: `factors`, `penalties` and
    `unknown_inputs` are what make it reviewable, and `unknown_inputs` in
    particular records how much of the score rests on nothing.
    """

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    score: float = Field(ge=0.0)
    base_score: float = Field(ge=0.0)
    factors: list[PriorityFactor] = Field(default_factory=list)
    penalties: list[PriorityFactor] = Field(default_factory=list)
    unknown_inputs: list[str] = Field(default_factory=list)
    evidence_completeness: UnitFloat = 1.0
    reasons: list[str] = Field(default_factory=list)


class SuccessfulResearchPattern(BaseModel):
    """One thing that actually worked, recorded structurally (§10).

    Not "feature X was good" -- that phrasing is what makes a past winner
    un-auditable. An effect is tied to the cell it was measured in
    (underlying, horizon, regime), the sample it rested on, and how it has
    held up since, so that a later priority boost can be checked against the
    conditions the original result required.

    `decay_since_discovery` exists so old winners lose weight on evidence
    rather than on a hunch. It is None until a re-measurement exists;
    scoring must treat that as unknown, not as "no decay".
    """

    model_config = ConfigDict(extra="forbid")

    pattern_id: str
    information_family: InformationFamily
    feature: str
    underlying_id: str
    horizon: str
    volatility_regime: str
    trend_regime: str
    oos_effect: float
    effective_sample: int = Field(ge=0)
    stability: UnitFloat
    economic_value: float
    discovered_at: TzAwareDatetime
    last_confirmed_at: TzAwareDatetime | None = None
    decay_since_discovery: float | None = None
    trial_id: str | None = None
    note: str = ""
    schema_version: str = SCHEMA_VERSION


class StoredOpportunity(BaseModel):
    """An opportunity as persisted, with its last computed priority.

    The priority is kept beside the opportunity rather than inside it
    because re-scoring must never silently rewrite the question itself.
    """

    model_config = ConfigDict(extra="forbid")

    opportunity: ResearchOpportunity
    priority: ResearchPriority | None = None
    scored_at: TzAwareDatetime | None = None


__all__ = [
    "SYSTEM_ASSIGNABLE_STATES",
    "Estimate",
    "EstimateBasis",
    "InformationFamily",
    "PriorityFactor",
    "ResearchOpportunity",
    "ResearchPriority",
    "ResearchStatus",
    "StoredOpportunity",
    "SuccessfulResearchPattern",
]
