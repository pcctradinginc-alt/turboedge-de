"""Alpha-centric research schemas (Alpha Factory, Phase A).

Why this package exists, in one measurement: `regime_conditional` improved
CRPS in 18 of 20 cells at p ~ 0.0135 -- a real forecasting result -- and was
then measured **economically worse than doing nothing**, mean dLCB_NetEV
-0.0205 (`docs/measured_results.md` §6.15). A better model is not an edge.

So the unit of research moves from the *model* to the *alpha source*: a
measurable economic effect, whose unit is net EV after costs rather than
accuracy. `ModelRegistry` keeps tracking models; this keeps track of effects.

Three things stay distinct and must not be conflated (`docs/alpha_factory.md`
§2): a **model** is a predictor, an **alpha source** is an economic effect
that may arise from a model or from a regime interaction, product selection,
issuer behaviour, pricing, execution or path modelling, and a **strategy** is
a rule combining alpha sources and allocating risk.

Nothing here is called from `pipeline/scan.py`. Phase A is infrastructure: it
changes no forecast, no gate and no trading decision.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from turboedge.meta.research_opportunity import Estimate, InformationFamily
from turboedge.storage.schemas import SCHEMA_VERSION, TzAwareDatetime


class AlphaStatus(StrEnum):
    """Lifecycle of an alpha source.

    Deliberately distinct from `ResearchStatus` (which tracks a research
    *question*: proposed, approved, running, measured) and from `ModelStatus`
    (champion / challenger / dormant). Conflating them is how "the model is
    good" becomes "we have an edge". A question can be answered "no" -- that
    is a successful research outcome and a failed alpha.

    The production ladder is staged on purpose: an alpha that clears its gates
    starts at CANARY with a small allocation and earns its way up, so a gate
    that was wrong costs little.
    """

    IDEA = "IDEA"
    EXPLORATORY = "EXPLORATORY"
    VALIDATED = "VALIDATED"
    CONFIRMATORY = "CONFIRMATORY"
    FORWARD_SHADOW = "FORWARD_SHADOW"
    CANARY_PRODUCTION = "CANARY_PRODUCTION"
    LIMITED_PRODUCTION = "LIMITED_PRODUCTION"
    NORMAL_PRODUCTION = "NORMAL_PRODUCTION"
    HEALTHY = "HEALTHY"
    WEAKENING = "WEAKENING"
    DECAYING = "DECAYING"
    DORMANT = "DORMANT"
    REJECTED = "REJECTED"
    DISABLED = "DISABLED"


#: States in which an alpha contributes to a production proposal. Note that
#: "production" here never means an order is placed -- manual execution
#: remains mandatory (Master Spec §49 rules 1-3).
PRODUCTION_STATES: frozenset[AlphaStatus] = frozenset(
    {
        AlphaStatus.CANARY_PRODUCTION,
        AlphaStatus.LIMITED_PRODUCTION,
        AlphaStatus.NORMAL_PRODUCTION,
    }
)

#: Terminal: nothing follows. REJECTED is kept retrievable forever (§7) --
#: failure memory is the point, not a cleanup target.
TERMINAL_STATES: frozenset[AlphaStatus] = frozenset({AlphaStatus.REJECTED})

#: Promotion ladder plus the health states a promoted alpha moves through,
#: demotion, and rejection. Every edge is deliberate:
#:
#: * The research chain is strictly ordered -- an alpha cannot skip
#:   confirmation, because untouched confirmation evidence is the one thing
#:   exploratory work cannot manufacture.
#: * Demotion runs NORMAL -> LIMITED -> CANARY -> FORWARD_SHADOW -> DORMANT,
#:   one rung at a time, so deterioration is graduated rather than binary.
#: * REJECTED is reachable from any pre-production state: cheap rejection is
#:   the whole economics of a factory where most hypotheses fail.
#: * DORMANT -> EXPLORATORY is the §38 retest path, and it requires a
#:   materially different condition recorded elsewhere; the transition alone
#:   does not grant it.
#: * DISABLED is handled separately below, not listed here.
ALLOWED_TRANSITIONS: dict[AlphaStatus, frozenset[AlphaStatus]] = {
    AlphaStatus.IDEA: frozenset({AlphaStatus.EXPLORATORY, AlphaStatus.REJECTED}),
    AlphaStatus.EXPLORATORY: frozenset({AlphaStatus.VALIDATED, AlphaStatus.REJECTED}),
    AlphaStatus.VALIDATED: frozenset({AlphaStatus.CONFIRMATORY, AlphaStatus.REJECTED}),
    AlphaStatus.CONFIRMATORY: frozenset({AlphaStatus.FORWARD_SHADOW, AlphaStatus.REJECTED}),
    AlphaStatus.FORWARD_SHADOW: frozenset(
        {AlphaStatus.CANARY_PRODUCTION, AlphaStatus.DORMANT, AlphaStatus.REJECTED}
    ),
    AlphaStatus.CANARY_PRODUCTION: frozenset(
        {AlphaStatus.LIMITED_PRODUCTION, AlphaStatus.HEALTHY, AlphaStatus.FORWARD_SHADOW}
    ),
    AlphaStatus.LIMITED_PRODUCTION: frozenset(
        {AlphaStatus.NORMAL_PRODUCTION, AlphaStatus.HEALTHY, AlphaStatus.CANARY_PRODUCTION}
    ),
    AlphaStatus.NORMAL_PRODUCTION: frozenset(
        {AlphaStatus.HEALTHY, AlphaStatus.WEAKENING, AlphaStatus.LIMITED_PRODUCTION}
    ),
    AlphaStatus.HEALTHY: frozenset({AlphaStatus.WEAKENING, AlphaStatus.LIMITED_PRODUCTION}),
    AlphaStatus.WEAKENING: frozenset(
        {AlphaStatus.HEALTHY, AlphaStatus.DECAYING, AlphaStatus.LIMITED_PRODUCTION}
    ),
    AlphaStatus.DECAYING: frozenset({AlphaStatus.DORMANT, AlphaStatus.CANARY_PRODUCTION}),
    AlphaStatus.DORMANT: frozenset({AlphaStatus.EXPLORATORY, AlphaStatus.REJECTED}),
    AlphaStatus.REJECTED: frozenset(),
    AlphaStatus.DISABLED: frozenset({AlphaStatus.DORMANT, AlphaStatus.REJECTED}),
}


def transition_allowed(current: AlphaStatus, target: AlphaStatus) -> bool:
    """Whether `current -> target` is a legal lifecycle move.

    DISABLED is reachable from **any** non-terminal state and is not listed in
    `ALLOWED_TRANSITIONS`, because an emergency stop that has to walk a ladder
    is not an emergency stop. It is the one transition that must never be
    blocked by the state machine.
    """
    if target is AlphaStatus.DISABLED:
        return current not in TERMINAL_STATES
    return target in ALLOWED_TRANSITIONS[current]


class AlphaSource(BaseModel):
    """One measurable economic effect, with its evidence and its provenance.

    Evidence quantities are `Estimate`s rather than bare floats, for the same
    reason `meta/research_opportunity.py` uses them: an unmeasured number must
    be recorded as unmeasured. A `net_ev` of 0.0 because nothing was measured
    and a `net_ev` of 0.0 because it was measured at zero are different
    claims, and collapsing them is how an empty table becomes confidence.

    Fields that cannot yet be populated are optional rather than defaulted
    (spec §48). Nothing here fabricates a value.
    """

    model_config = ConfigDict(extra="forbid")

    alpha_id: str
    name: str
    family: InformationFamily
    version: str
    description: str
    economic_hypothesis: str

    status: AlphaStatus = AlphaStatus.IDEA

    created_at: TzAwareDatetime
    frozen_at: TzAwareDatetime | None = None

    underlyings: list[str] = Field(default_factory=list)
    horizons: list[str] = Field(default_factory=list)
    directions: list[str] = Field(default_factory=list)
    required_data_sources: list[str] = Field(default_factory=list)

    trial_ids: list[str] = Field(default_factory=list)

    nominal_sample: Estimate
    effective_sample: Estimate
    expected_net_ev: Estimate
    lcb_net_ev: Estimate
    posterior_probability_positive: Estimate
    uncertainty_score: Estimate
    drift_score: Estimate
    decay_score: Estimate

    git_commit: str | None = None
    config_hash: str | None = None
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _frozen_before_confirmatory(self) -> Self:
        """An alpha past EXPLORATORY must have been frozen.

        Confirmatory evidence means nothing if the hypothesis could still move.
        `frozen_at` is the timestamp that makes "the method was fixed before
        the data was seen" checkable rather than asserted.
        """
        needs_freeze = {
            AlphaStatus.CONFIRMATORY,
            AlphaStatus.FORWARD_SHADOW,
            *PRODUCTION_STATES,
        }
        if self.status in needs_freeze and self.frozen_at is None:
            raise ValueError(
                f"status {self.status} requires frozen_at: confirmatory and production "
                "evidence is only meaningful for a hypothesis that was frozen first"
            )
        return self


class EdgeAttribution(BaseModel):
    """Where one decision's economic value actually came from.

    "Did this trade make money" is far less useful than "which part of the
    machine made it". The components exist so that value cannot all hide under
    "model alpha" (spec §8) -- and this repository has already measured why
    that matters: the forecast channel was negative while product selection was
    positive, which a single aggregate number would have concealed.

    `interaction_residual` is not a dumping ground. It is the measure of how
    much is *not* understood, and a large residual is a finding rather than a
    rounding detail.
    """

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    prediction_time: TzAwareDatetime
    underlying: str
    horizon: int = Field(gt=0)

    total_incremental_net_ev: float

    forecast_edge: float = 0.0
    path_edge: float = 0.0
    product_selection_edge: float = 0.0
    cost_selection_edge: float = 0.0
    issuer_edge: float = 0.0
    timing_edge: float = 0.0
    portfolio_edge: float = 0.0

    interaction_residual: float = 0.0

    reconciliation_tolerance: float = Field(default=1e-9, gt=0.0)
    schema_version: str = SCHEMA_VERSION

    @property
    def component_sum(self) -> float:
        return (
            self.forecast_edge
            + self.path_edge
            + self.product_selection_edge
            + self.cost_selection_edge
            + self.issuer_edge
            + self.timing_edge
            + self.portfolio_edge
            + self.interaction_residual
        )

    @model_validator(mode="after")
    def _components_reconcile(self) -> Self:
        """Components plus residual must equal the total, or this raises.

        Deliberately a raise and not a silent adjustment: an attribution that
        quietly balances itself is decoration. If the parts do not add up, the
        decomposition is wrong and the number it produces should not be used.
        """
        gap = abs(self.component_sum - self.total_incremental_net_ev)
        if gap > self.reconciliation_tolerance:
            raise ValueError(
                f"edge components + residual = {self.component_sum!r} do not reconcile to "
                f"total_incremental_net_ev = {self.total_incremental_net_ev!r} "
                f"(gap {gap!r} > tolerance {self.reconciliation_tolerance!r}); "
                "refusing to report an attribution that does not add up"
            )
        return self


__all__ = [
    "ALLOWED_TRANSITIONS",
    "PRODUCTION_STATES",
    "TERMINAL_STATES",
    "AlphaSource",
    "AlphaStatus",
    "EdgeAttribution",
    "transition_allowed",
]
