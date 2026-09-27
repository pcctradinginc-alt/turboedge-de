"""System generations -- making "the system improved" a checkable claim.

A self-optimizing system needs a defence against its own favourite error:
concluding it got better because it *changed*. Weights moved, a model was
replaced, features were added, a backtest looks nicer -- none of that is
improvement, and all of it feels like it.

So a frozen configuration gets a `generation_id`, and the only way to mark one
as improved is forward evidence on data that did not exist when the change was
made. `mark_improved` refuses everything else, including a better backtest,
and refuses loudly rather than quietly declining.

Why this is not paranoia here: `regime_conditional` improved CRPS in 18 of 20
cells at p ~ 0.0135 and measured economically *worse* than the null
(`docs/measured_results.md` §6.15). A generation carrying it would have looked
better by every diagnostic this repository had at the time.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from turboedge.storage.schemas import SCHEMA_VERSION, TzAwareDatetime


class GenerationStatus(StrEnum):
    """What is known about a generation relative to its parent.

    `UNEVALUATED` is the honest default and the one most generations should
    stay in: comparing two generations needs forward data that takes months to
    accumulate. `INCONCLUSIVE` exists so "we looked and could not tell" is
    recordable -- without it, an inconclusive comparison has nowhere to go and
    tends to become a quiet `IMPROVED`.
    """

    UNEVALUATED = "UNEVALUATED"
    IMPROVED = "IMPROVED"
    NOT_IMPROVED = "NOT_IMPROVED"
    INCONCLUSIVE = "INCONCLUSIVE"
    SUPERSEDED = "SUPERSEDED"


class GenerationEvidenceError(RuntimeError):
    """Raised when a generation is marked improved without forward evidence."""


class SystemGeneration(BaseModel):
    """One frozen configuration of the whole system, and what it achieved.

    The versions are recorded rather than the components themselves: a
    generation is a statement about *which* frozen pieces were live together,
    so that a later comparison is between two reproducible configurations and
    not between two moving targets.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    generation_id: str
    created_at: TzAwareDatetime
    parent_generation_id: str | None = None

    git_commit: str
    config_hash: str

    active_model_versions: dict[str, str] = Field(default_factory=dict)
    active_alpha_versions: dict[str, str] = Field(default_factory=dict)
    risk_policy_version: str | None = None
    product_policy_version: str | None = None
    portfolio_policy_version: str | None = None

    #: The forward window this generation was judged over. Both None until an
    #: evaluation has actually run -- never backfilled from a backtest.
    evaluation_start: TzAwareDatetime | None = None
    evaluation_end: TzAwareDatetime | None = None

    #: Forward outcomes. None means not measured, which is different from zero.
    forward_net_ev: float | None = None
    lcb_net_ev: float | None = None
    tail_utility: float | None = None
    regret: float | None = None

    status: GenerationStatus = GenerationStatus.UNEVALUATED
    #: Why the status is what it is, in numbers. Empty while UNEVALUATED.
    status_reason: str = ""

    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _check_evaluation_shape(self) -> Self:
        if (self.evaluation_start is None) != (self.evaluation_end is None):
            raise ValueError(
                "evaluation_start and evaluation_end must be set together: a half-open "
                "evaluation window cannot be compared against anything"
            )
        if (
            self.evaluation_start is not None
            and self.evaluation_end is not None
            and self.evaluation_end <= self.evaluation_start
        ):
            raise ValueError("evaluation_end must be after evaluation_start")
        if self.status is not GenerationStatus.UNEVALUATED and not self.status_reason.strip():
            raise ValueError(
                f"status {self.status} requires status_reason: a verdict with no stated basis "
                "cannot be disagreed with"
            )
        return self

    @property
    def has_forward_evidence(self) -> bool:
        """Whether this generation was measured on a real forward window.

        Requires both a window and a forward net EV. A window with no outcome
        is an evaluation that was started, not one that concluded.
        """
        return (
            self.evaluation_start is not None
            and self.evaluation_end is not None
            and self.forward_net_ev is not None
        )

    def mark_improved(self, *, reason: str, against: SystemGeneration) -> None:
        """Declare this generation better than its parent. Heavily guarded.

        Every check here exists because the alternative is a system that
        congratulates itself for changing:

        * forward evidence is required -- a backtest cannot promote a
          generation, which is the whole point of the object;
        * the comparison must be against this generation's declared parent, so
          a favourable comparison cannot be shopped for;
        * the parent must itself have been measured forward, since "better than
          something unmeasured" is not a comparison;
        * the evaluation windows must not be the same one, because evaluating
          two configurations over identical data is a backtest wearing a
          forward window's clothes;
        * the forward net EV must actually be higher.
        """
        if not self.has_forward_evidence:
            raise GenerationEvidenceError(
                f"{self.generation_id} has no forward evidence (evaluation window "
                f"{self.evaluation_start}..{self.evaluation_end}, forward_net_ev "
                f"{self.forward_net_ev}); a generation is not improved because its backtest is"
            )
        if against.generation_id != self.parent_generation_id:
            raise GenerationEvidenceError(
                f"{self.generation_id} declares parent {self.parent_generation_id!r} but was "
                f"compared against {against.generation_id!r}; comparing against a generation "
                "other than the declared parent is shopping for a favourable baseline"
            )
        if not against.has_forward_evidence:
            raise GenerationEvidenceError(
                f"parent {against.generation_id} has no forward evidence; 'better than something "
                "unmeasured' is not a comparison"
            )
        if (self.evaluation_start, self.evaluation_end) == (
            against.evaluation_start,
            against.evaluation_end,
        ):
            raise GenerationEvidenceError(
                "both generations were evaluated over the identical window; that is a backtest "
                "comparison, not forward evidence that this generation is better"
            )
        assert self.forward_net_ev is not None  # narrowed by has_forward_evidence
        assert against.forward_net_ev is not None
        if self.forward_net_ev <= against.forward_net_ev:
            raise GenerationEvidenceError(
                f"forward_net_ev {self.forward_net_ev} is not above parent's "
                f"{against.forward_net_ev}"
            )
        if not reason.strip():
            raise ValueError("reason is required: a verdict with no stated basis is not reviewable")

        self.status_reason = reason
        self.status = GenerationStatus.IMPROVED


__all__ = ["GenerationEvidenceError", "GenerationStatus", "SystemGeneration"]
