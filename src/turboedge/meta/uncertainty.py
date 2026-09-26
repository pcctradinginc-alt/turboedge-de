"""Uncertainty decomposition and abstention (Phase 1, M4).

Six kinds of not-knowing, kept apart because they behave differently and
call for different responses. Collapsing them into one "confidence: 72%"
would hide the distinction that matters most for a knock-out product: an
error in the tail decides whether the barrier is hit, while an equal error
in the expected return barely moves the payoff.

The abstention rule is deliberately a weighted maximum-and-mean hybrid
rather than a plain average. A single catastrophic unknown -- no usable
product data, a regime never seen before -- must be able to force
abstention on its own; averaging would let five comfortable numbers drown
out one disqualifying one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from turboedge.meta.disagreement import aggregate_disagreement
from turboedge.meta.schemas import DecisionConfidence, ModelTrust
from turboedge.meta.trust import REGIME_FAMILIARITY_TARGET
from turboedge.models.forecast import HorizonForecast

#: Above this, the controller abstains. Not tuned against any outcome --
#: tuning it on results is exactly the parameter fishing this project's
#: governance forbids. It is set so that "no historical evidence at all",
#: which is this repository's current state, lands above it.
ABSTAIN_THRESHOLD = 0.60

#: Below this confidence a PROCEED is downgraded to WATCH_ONLY: readable
#: situation, but not one to act on.
PROCEED_CONFIDENCE_FLOOR = 0.50

_UNCERTAINTY_REFERENCE = 0.02


def epistemic_uncertainty(trusts: Sequence[ModelTrust]) -> float:
    """How little the best available model is trusted.

    Takes the maximum trust, not the mean: the question is whether ANY model
    can be relied on here, and a good model is not discredited by sitting
    next to three poor ones.
    """
    if not trusts:
        return 1.0
    return float(1.0 - max(t.trust_score for t in trusts))


def data_uncertainty(data_quality: float, stale_share: float) -> float:
    """Combines the scan's data-health signal with quote staleness.

    The worse of the two, not their average: fresh quotes do not repair a
    failed health check, and healthy sources do not repair stale quotes.
    """
    return float(min(max(max(1.0 - data_quality, stale_share), 0.0), 1.0))


def regime_uncertainty(observation_count: int) -> float:
    """How unfamiliar this regime is, 1.0 when never observed."""
    if observation_count <= 0:
        return 1.0
    return float(1.0 - min(observation_count / REGIME_FAMILIARITY_TARGET, 1.0))


def calibration_uncertainty(trusts: Sequence[ModelTrust]) -> float:
    """1.0 when calibration is unmeasured for every model.

    Unmeasured is treated as maximally uncertain rather than as neutral:
    with `walkforward_results` empty, a stated probability has never been
    checked against outcomes, and pretending otherwise is how an uncalibrated
    number ends up sizing a position.
    """
    known = [t.calibration_quality for t in trusts if t.calibration_quality is not None]
    if not known:
        return 1.0
    return float(1.0 - max(known))


def product_data_uncertainty(
    *, ratio_unverified_share: float, integrity_fail_share: float
) -> float:
    """Share of candidates whose product terms cannot be relied on.

    Its own axis rather than part of `data_uncertainty` because the failure
    is different in kind: the market data can be perfect while the
    certificate's own Bezugsverhaeltnis is unverified, and that alone is
    enough to make a payoff calculation meaningless.
    """
    return float(min(max(max(ratio_unverified_share, integrity_fail_share), 0.0), 1.0))


def assess(
    *,
    trusts: Sequence[ModelTrust],
    forecasts: Sequence[HorizonForecast],
    data_quality: float,
    stale_share: float,
    regime_observation_count: int,
    ratio_unverified_share: float,
    integrity_fail_share: float,
) -> DecisionConfidence:
    """The full uncertainty vector plus its abstention score."""
    epistemic = epistemic_uncertainty(trusts)
    data = data_uncertainty(data_quality, stale_share)
    regime = regime_uncertainty(regime_observation_count)
    disagreement = aggregate_disagreement(forecasts)
    calibration = calibration_uncertainty(trusts)
    product = product_data_uncertainty(
        ratio_unverified_share=ratio_unverified_share,
        integrity_fail_share=integrity_fail_share,
    )

    components = np.asarray(
        [epistemic, data, regime, disagreement, calibration, product], dtype=np.float64
    )
    # Half the weight on the worst single component, half on the average.
    # One disqualifying unknown should be able to force abstention by
    # itself, while several mild ones should still accumulate.
    abstain = 0.5 * float(components.max()) + 0.5 * float(components.mean())

    return DecisionConfidence(
        epistemic_uncertainty=epistemic,
        data_uncertainty=data,
        regime_uncertainty=regime,
        model_disagreement=disagreement,
        calibration_uncertainty=calibration,
        product_data_uncertainty=product,
        abstain_score=float(min(max(abstain, 0.0), 1.0)),
    )


__all__ = [
    "ABSTAIN_THRESHOLD",
    "PROCEED_CONFIDENCE_FLOOR",
    "assess",
    "calibration_uncertainty",
    "data_uncertainty",
    "epistemic_uncertainty",
    "product_data_uncertainty",
    "regime_uncertainty",
]
