"""Model trust scoring (M2)."""

from __future__ import annotations

from collections.abc import Callable

from turboedge.meta.trust import (
    MISSING_FACTOR_PENALTY,
    REGIME_FAMILIARITY_TARGET,
    score_model_trust,
    weights_from_trust,
)


def test_missing_evidence_is_penalised_not_defaulted_to_neutral(
    entry_factory: Callable[..., object], forecast_factory: Callable[..., object]
) -> None:
    """The decision this whole layer stands on.

    `walkforward_results` is empty in this repository, so OOS quality and
    calibration are genuinely unknowable. Defaulting them to 1.0 would
    manufacture confidence out of an empty table -- exactly the failure the
    meta layer exists to catch.
    """
    trust = score_model_trust(
        entry_factory("m1"),
        forecast=forecast_factory("m1"),
        walkforward=None,
        regime_observation_count=REGIME_FAMILIARITY_TARGET,
        data_quality=1.0,
        in_drift=False,
    )
    assert trust.historical_oos_quality is None
    assert trust.calibration_quality is None
    assert set(trust.missing_factors) == {"historical_oos_quality", "calibration_quality"}
    # Two missing factors => at most 0.25 of the otherwise-computed value.
    assert trust.trust_score <= MISSING_FACTOR_PENALTY**2


def test_model_worse_than_null_scores_zero_not_a_small_positive(
    entry_factory: Callable[..., object],
    forecast_factory: Callable[..., object],
    walkforward_factory: Callable[..., object],
) -> None:
    """W4/W9 measured every family as worse than null. Rounding that up to a
    small positive number would misrepresent the one thing this system has
    most firmly established about itself."""
    trust = score_model_trust(
        entry_factory("m1"),
        forecast=forecast_factory("m1"),
        walkforward=walkforward_factory("m1", brier=0.30, brier_null=0.25),
        regime_observation_count=REGIME_FAMILIARITY_TARGET,
        data_quality=1.0,
        in_drift=False,
    )
    assert trust.historical_oos_quality == 0.0
    assert trust.trust_score == 0.0


def test_unseen_regime_drives_trust_to_zero(
    entry_factory: Callable[..., object],
    forecast_factory: Callable[..., object],
    walkforward_factory: Callable[..., object],
) -> None:
    trust = score_model_trust(
        entry_factory("m1"),
        forecast=forecast_factory("m1"),
        walkforward=walkforward_factory("m1"),
        regime_observation_count=0,
        data_quality=1.0,
        in_drift=False,
    )
    assert trust.regime_similarity == 0.0
    assert trust.trust_score == 0.0


def test_drift_materially_reduces_trust(
    entry_factory: Callable[..., object],
    forecast_factory: Callable[..., object],
    walkforward_factory: Callable[..., object],
) -> None:
    kwargs = dict(
        forecast=forecast_factory("m1"),
        walkforward=walkforward_factory("m1"),
        regime_observation_count=REGIME_FAMILIARITY_TARGET,
        data_quality=1.0,
    )
    calm = score_model_trust(entry_factory("m1"), in_drift=False, **kwargs)
    drifting = score_model_trust(entry_factory("m1"), in_drift=True, **kwargs)
    assert drifting.trust_score < calm.trust_score
    assert drifting.drift_penalty < calm.drift_penalty


def test_all_zero_trust_yields_zero_weights_not_a_uniform_split(
    entry_factory: Callable[..., object], forecast_factory: Callable[..., object]
) -> None:
    """A uniform fallback would say "trust everyone equally" at the exact
    moment the honest statement is "trust nobody"."""
    trusts = [
        score_model_trust(
            entry_factory(m),
            forecast=forecast_factory(m),
            walkforward=None,
            regime_observation_count=0,
            data_quality=1.0,
            in_drift=False,
        )
        for m in ("m1", "m2")
    ]
    assert all(t.trust_score == 0.0 for t in trusts)
    assert weights_from_trust(trusts) == {"m1": 0.0, "m2": 0.0}


def test_weights_normalise_to_one_when_any_trust_exists(
    entry_factory: Callable[..., object],
    forecast_factory: Callable[..., object],
    walkforward_factory: Callable[..., object],
) -> None:
    trusts = [
        score_model_trust(
            entry_factory(m),
            forecast=forecast_factory(m),
            walkforward=walkforward_factory(m, brier=b),
            regime_observation_count=REGIME_FAMILIARITY_TARGET,
            data_quality=1.0,
            in_drift=False,
        )
        for m, b in (("m1", 0.20), ("m2", 0.23))
    ]
    weights = weights_from_trust(trusts)
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert weights["m1"] > weights["m2"], "the better model must get more weight"
