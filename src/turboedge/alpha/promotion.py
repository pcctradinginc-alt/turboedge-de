"""Frozen promotion criteria (Alpha Factory, Phase A).

The human instruction this implements (GOVERNANCE.md §1.1a, 2026-09-27):
*"Human statistical judgement must NOT be required for routine alpha
promotion... Therefore promotion decisions should be made by a deterministic,
non-learning Evidence & Risk Gate."*

That is only safe because of one property, and this module exists to make it
structural rather than a convention: **a candidate can never change the bar it
is measured against.** The adaptive system may propose alpha sources and tune
their parameters during permitted research phases. It may not touch what is in
here. Changing these criteria is a governance act requiring human approval, not
a research act.

Three things follow, and all three are enforced below rather than documented:

* The object is **immutable** (``frozen=True``). A criteria set that a caller
  can mutate in flight is not a gate.
* It carries a ``criteria_version`` and a ``frozen_at``, so a promotion can be
  audited against the exact bar that applied when it happened. A gate whose
  history is unknowable cannot be reviewed after the fact.
* Thresholds this repository has not measured are **required-but-unset**
  (``None``) rather than given a plausible-looking default. An invented number
  in a promotion gate is worse than a missing one: the missing one blocks, the
  invented one passes things.

**Phase A contains definitions only.** There is deliberately no function here
that takes an `AlphaSource` and returns a promotion decision. Building the
evaluator is a later phase; shipping a gate and an evaluator in the same change
would mean the first thing this object ever did was promote something.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from turboedge.storage.schemas import SCHEMA_VERSION, TzAwareDatetime


class PromotionCriteria(BaseModel):
    """One frozen, versioned set of gates an alpha must clear to be promoted.

    Every threshold below is either taken from a number this repository already
    measured and documented -- with its source named -- or left ``None``.
    Nothing here was chosen to make a candidate pass.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    criteria_version: str
    frozen_at: TzAwareDatetime
    source: str = Field(
        default="GOVERNANCE.md §2 (Ladder Rule) + §1.1a",
        description="Where these numbers came from. Not free text: a promotion "
        "audited later must be traceable to the document that set its bar.",
    )

    # --- evidence gates, all from GOVERNANCE.md §2 -----------------------
    #: §2.1. Post-deflation z, i.e. after Benjamini-Hochberg across the
    #: quarter's *primary* hypotheses -- §11.2's correction after an aggregate
    #: model-level p-value was wrongly deflated against 180 cell-level tests.
    min_z_score_after_deflation: float = 1.645
    #: §2.1.
    min_deflated_sharpe_ratio: float = 0.6
    #: §2.1, P(mu > 0).
    min_probabilistic_sharpe_ratio: float = 0.95
    #: §2.2. 10 bps per forward trade. The binding one in practice: the best
    #: lcb_ev this repository has ever produced is -0.00682.
    min_absolute_net_ev_improvement: float = 0.0010
    #: §6.1 / `backtest/excursion_eval.py::MIN_EFFECTIVE_SAMPLE`. Effective,
    #: not nominal: 2,672 ledger rows were worth 1.00 independent observations
    #: (docs/measured_results.md §6.9).
    min_effective_sample: float = 100.0
    #: An LCB below zero is the state every scan has been in since 2026-09-13.
    require_positive_lcb_net_ev: bool = True
    #: §18: exploratory evidence can never promote. Confirmation data must have
    #: been untouched when the hypothesis was frozen.
    require_untouched_confirmation_evidence: bool = True

    # --- risk gates: no measured number exists yet, so none is invented ---
    #: Unset. This repository has never held a position, so it has no measured
    #: expected shortfall to calibrate against.
    max_expected_shortfall: float | None = None
    #: Unset, same reason.
    max_drawdown: float | None = None
    #: Unset. `p_ko` is explicitly not a calibrated probability
    #: (`ranking/ev.py`), so a threshold on it would be a threshold on a number
    #: that does not mean what it says.
    max_ko_risk: float | None = None
    #: Unset. Needs an AlphaCorrelationMatrix, which is a later phase.
    max_alpha_correlation: float | None = None
    #: Unset. BNP is ~92% of the product universe today, so any concentration
    #: limit set now would either block everything or be meaningless.
    max_issuer_concentration: float | None = None
    #: Unset. `quality_score` exists per snapshot but has no promotion-relevant
    #: calibration.
    min_data_quality: float | None = None

    # --- forward evidence ------------------------------------------------
    #: Unset. The forward ledger is 14 days old; any number chosen now would be
    #: chosen to fit that, which is the opposite of a pre-set bar.
    min_forward_shadow_days: int | None = None

    # --- hard blocks: booleans, because these are not tradeable ----------
    #: §40 and Master Spec rule 4/5. A leakage violation is not a risk to be
    #: priced, it invalidates the measurement.
    block_on_unresolved_leakage: bool = True
    #: §42. Includes an over-budget quarter; 2026Q3 stands at 13 of 6.
    block_on_governance_violation: bool = True
    #: §27. Page-Hinkley drift already reduces weights; here it blocks.
    block_on_material_drift: bool = True
    #: §32. The failure mode that produced this gate: an aggregate result
    #: driven by one underlying. Trial 2026Q4-002 returned mean dLCB +0.0153
    #: with DAX, NDX and XAU all negative and EURUSD at +0.1208 -- a PASS by
    #: the letter of its own rule that this block exists to stop.
    block_on_single_regime_dependence: bool = True

    schema_version: str = SCHEMA_VERSION

    @property
    def unset_gates(self) -> tuple[str, ...]:
        """Gates with no threshold yet, in declaration order.

        Exposed rather than hidden because a criteria set with unset gates is
        incomplete, and anything reading it should be able to say so instead of
        treating absence as permission.
        """
        optional = (
            "max_expected_shortfall",
            "max_drawdown",
            "max_ko_risk",
            "max_alpha_correlation",
            "max_issuer_concentration",
            "min_data_quality",
            "min_forward_shadow_days",
        )
        return tuple(name for name in optional if getattr(self, name) is None)

    @property
    def is_complete(self) -> bool:
        """Whether every gate has a threshold.

        False today, and that is the honest state: seven risk and forward gates
        have no measured basis yet. A later evaluator must treat an incomplete
        criteria set as blocking, never as satisfied-by-default -- an unset gate
        means "not yet decided", not "no constraint".
        """
        return not self.unset_gates


__all__ = ["PromotionCriteria"]
