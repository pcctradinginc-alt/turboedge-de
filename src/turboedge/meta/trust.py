"""Per-model trust scoring (Phase 1, M2).

A model's trust in a given situation, built from factors that are each
stored separately so the result can be argued with rather than only
accepted.

The load-bearing design decision: **a factor that cannot be computed is
recorded as missing and treated as untrusted, never silently defaulted to
"fine".** Today `walkforward_results`, `strategy_posteriors` and
`ledger_labels` are all empty in this repository, so `historical_oos_quality`
and `calibration_quality` are genuinely unknowable. A neutral 1.0 there
would manufacture confidence out of an empty table -- precisely the failure
this layer exists to catch. The penalty for missing evidence is therefore
explicit, and `missing_factors` travels with the score.

The multiplicative form is chosen because these factors are closer to
independent necessary conditions than to tradeable strengths: a model in
drift is not rescued by being well calibrated last year.
"""

from __future__ import annotations

from collections.abc import Sequence

from turboedge.meta.schemas import ModelTrust
from turboedge.models.forecast import HorizonForecast
from turboedge.storage.schemas import ModelRegistryEntry, WalkforwardResultRecord

#: Multiplier applied per factor that could not be computed. Deliberately
#: harsh: two unknown factors leave trust at 0.25 of its otherwise-computed
#: value, which is the honest position when the evidence base is empty.
MISSING_FACTOR_PENALTY = 0.5

#: Regime observation count at or above which a regime counts as familiar.
#: Below it, `regime_similarity` scales linearly -- 20 observations is far
#: too few to claim a model works "in this regime", and 0 must yield 0.
REGIME_FAMILIARITY_TARGET = 100

_DRIFT_PENALTY_ACTIVE = 0.3
_UNCERTAINTY_REFERENCE = 0.02
_ECE_REFERENCE = 0.05


def _oos_quality(result: WalkforwardResultRecord | None) -> float | None:
    """Brier improvement over the null model, squashed into [0, 1].

    `None` when no walk-forward record exists for this (model, underlying,
    horizon) -- which is the current state of this repository for every
    model. A model that is *worse* than null scores 0, not a small positive
    number: in this system that is the measured reality for every forecast
    family tested so far (W4, W9), and rounding it up would misrepresent it.
    """
    if result is None or result.brier_null is None or result.brier_null <= 0:
        return None
    improvement = (result.brier_null - result.brier) / result.brier_null
    if improvement <= 0:
        return 0.0
    return float(min(improvement * 10.0, 1.0))


def _calibration_quality(result: WalkforwardResultRecord | None) -> float | None:
    """Expected calibration error, inverted into [0, 1].

    ECE rather than a calibration slope: it measures the gap between stated
    and realised frequencies directly, which is the property that decides
    whether a stated P(KO) or P(up) can be taken at face value. Scaled
    against `_ECE_REFERENCE` -- an ECE of 0.05 is poor for a probability a
    position size depends on, so that is where quality reaches zero.
    """
    if result is None:
        return None
    return float(max(0.0, 1.0 - min(result.ece / _ECE_REFERENCE, 1.0)))


def _regime_similarity(observation_count: int) -> float:
    """Linear in observed frequency up to the familiarity target."""
    if observation_count <= 0:
        return 0.0
    return float(min(observation_count / REGIME_FAMILIARITY_TARGET, 1.0))


def _uncertainty_penalty(forecast: HorizonForecast | None) -> float:
    """Shrinks as the model's own standard error grows.

    Uses the model's stated `uncertainty` (the standard error of its mean),
    scaled against a reference of 0.02 -- roughly the estimation error this
    system's models report at a 14-day horizon.
    """
    if forecast is None:
        return 1.0
    if forecast.uncertainty <= 0:
        return 1.0
    return float(1.0 / (1.0 + forecast.uncertainty / _UNCERTAINTY_REFERENCE))


def score_model_trust(
    entry: ModelRegistryEntry,
    *,
    forecast: HorizonForecast | None,
    walkforward: WalkforwardResultRecord | None,
    regime_observation_count: int,
    data_quality: float,
    in_drift: bool,
) -> ModelTrust:
    """Trust for one model in one situation.

    ``data_quality`` is the scan's own data-health signal, passed in rather
    than recomputed here so the meta layer and the pipeline cannot drift
    apart on what "good data" means.
    """
    missing: list[str] = []
    oos = _oos_quality(walkforward)
    if oos is None:
        missing.append("historical_oos_quality")
    calibration = _calibration_quality(walkforward)
    if calibration is None:
        missing.append("calibration_quality")

    similarity = _regime_similarity(regime_observation_count)
    drift_penalty = _DRIFT_PENALTY_ACTIVE if in_drift else 1.0
    uncertainty_penalty = _uncertainty_penalty(forecast)
    clean_data_quality = float(min(max(data_quality, 0.0), 1.0))

    score = (
        (oos if oos is not None else 1.0)
        * (calibration if calibration is not None else 1.0)
        * similarity
        * clean_data_quality
        * drift_penalty
        * uncertainty_penalty
    )
    score *= MISSING_FACTOR_PENALTY ** len(missing)

    return ModelTrust(
        model_id=entry.model_id,
        signal_family=entry.signal_family,
        historical_oos_quality=oos,
        calibration_quality=calibration,
        regime_similarity=similarity,
        data_quality=clean_data_quality,
        drift_penalty=drift_penalty,
        uncertainty_penalty=uncertainty_penalty,
        trust_score=float(min(max(score, 0.0), 1.0)),
        missing_factors=missing,
    )


def weights_from_trust(trusts: Sequence[ModelTrust]) -> dict[str, float]:
    """Normalise trust into ensemble weights.

    All-zero trust returns all-zero weights rather than a uniform fallback.
    A uniform split would quietly say "trust everyone equally" at the exact
    moment the honest statement is "trust nobody" -- and the controller
    reads that as grounds to abstain.
    """
    total = sum(t.trust_score for t in trusts)
    if total <= 0:
        return {t.model_id: 0.0 for t in trusts}
    return {t.model_id: t.trust_score / total for t in trusts}


__all__ = [
    "MISSING_FACTOR_PENALTY",
    "REGIME_FAMILIARITY_TARGET",
    "score_model_trust",
    "weights_from_trust",
]
