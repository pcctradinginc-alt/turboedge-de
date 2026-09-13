from __future__ import annotations

import math

import pytest

from turboedge.ranking.utility import UtilityConfig, expected_utility, score


def test_expected_utility_hand_example() -> None:
    # Hand-computed example: mean=0.04, ES95=-0.15 (|.|=0.15), P_KO=0.10,
    # uncertainty=0.02, cluster_risk=0.30, default lambdas
    # (es=0.5, ko=0.5, uncertainty=1.0, cluster=0.3).
    cfg = UtilityConfig()
    u = expected_utility(0.04, -0.15, 0.10, 0.02, 0.30, cfg=cfg)
    expected = 0.04 - 0.5 * 0.15 - 0.5 * 0.10 - 1.0 * 0.02 - 0.3 * 0.30
    assert u == pytest.approx(expected)
    assert u == pytest.approx(0.04 - 0.075 - 0.05 - 0.02 - 0.09)


def test_expected_utility_custom_lambdas() -> None:
    cfg = UtilityConfig(lambda_es=1.0, lambda_ko=1.0, lambda_uncertainty=0.5, lambda_cluster=0.0)
    u = expected_utility(0.10, -0.20, 0.05, 0.04, 100.0, cfg=cfg)
    expected = 0.10 - 1.0 * 0.20 - 1.0 * 0.05 - 0.5 * 0.04 - 0.0 * 100.0
    assert u == pytest.approx(expected)


def test_expected_utility_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        expected_utility(0.05, -0.1, 1.5, 0.01, 0.0)
    with pytest.raises(ValueError):
        expected_utility(0.05, -0.1, 0.5, -0.01, 0.0)
    with pytest.raises(ValueError):
        expected_utility(0.05, -0.1, 0.5, 0.01, -0.1)


def test_score_hand_example() -> None:
    # Hand-computed example: this utility comes out *negative*
    # (0.02 - 0.05 - 0.04 - 0.015 - 0.06 = -0.145) -- deliberately, since
    # this is exactly the regime the score-formula fix (module docstring
    # deviation note) applies to: for LCB(U) < 0, the multipliers combine
    # by *division*, not multiplication (see `score`'s docstring).
    cfg = UtilityConfig()
    utility, final = score(
        lcb_net_return=0.02,
        expected_shortfall_95=-0.10,
        p_ko=0.08,
        uncertainty=0.015,
        cluster_risk=0.2,
        liquidity_factor=0.8,
        cfg=cfg,
        calibration_factor=1.0,
        strategy_posterior_factor=1.0,
        positive_memory_factor=1.0,
    )
    expected_u = 0.02 - 0.5 * 0.10 - 0.5 * 0.08 - 1.0 * 0.015 - 0.3 * 0.2
    assert expected_u < 0.0
    assert utility == pytest.approx(expected_u)
    assert final == pytest.approx(expected_u / 0.8)


def test_score_multiplies_all_factors() -> None:
    cfg = UtilityConfig(lambda_es=0.0, lambda_ko=0.0, lambda_uncertainty=0.0, lambda_cluster=0.0)
    utility, final = score(
        lcb_net_return=0.10,
        expected_shortfall_95=0.0,
        p_ko=0.0,
        uncertainty=0.0,
        cluster_risk=0.0,
        liquidity_factor=0.5,
        cfg=cfg,
        calibration_factor=0.9,
        strategy_posterior_factor=0.8,
        positive_memory_factor=1.1,
    )
    assert utility == pytest.approx(0.10)
    assert final == pytest.approx(0.10 * 0.5 * 0.9 * 0.8 * 1.1)


def test_score_default_factors_are_neutral() -> None:
    cfg = UtilityConfig(lambda_es=0.0, lambda_ko=0.0, lambda_uncertainty=0.0, lambda_cluster=0.0)
    utility, final = score(0.05, 0.0, 0.0, 0.0, 0.0, 1.0, cfg=cfg)
    assert utility == pytest.approx(0.05)
    assert final == pytest.approx(0.05)


def test_score_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        score(0.05, -0.1, 0.1, 0.0, 0.0, liquidity_factor=0.0)
    with pytest.raises(ValueError):
        score(0.05, -0.1, 0.1, 0.0, 0.0, liquidity_factor=1.0, calibration_factor=-0.1)


# --- Score-formula order-preservation fix -----------------------------
#
# Master Spec §18 writes score = LCB(U) * liq * calibration * posterior *
# memory unconditionally. That is only order-preserving (better multiplier
# -> better/higher score) when LCB(U) >= 0: for LCB(U) < 0, multiplying by a
# smaller positive factor moves the (negative) score *towards* zero, i.e.
# makes a worse candidate score *higher* -- an order violation. The tests
# below pin down the fix (`score` docstring): multiply when utility >= 0,
# divide by the multiplier when utility < 0, in every factor independently.

_NEG_CASE = dict(
    lcb_net_return=-0.05,
    expected_shortfall_95=0.0,
    p_ko=0.0,
    uncertainty=0.0,
    cluster_risk=0.0,
)
_POS_CASE = dict(
    lcb_net_return=0.05,
    expected_shortfall_95=0.0,
    p_ko=0.0,
    uncertainty=0.0,
    cluster_risk=0.0,
)
_NEUTRAL_CFG = UtilityConfig(
    lambda_es=0.0, lambda_ko=0.0, lambda_uncertainty=0.0, lambda_cluster=0.0
)


def test_score_negative_utility_liquidity_factor_no_longer_inverts_order() -> None:
    """The bug this fix targets, pinned down exactly: two otherwise-identical
    candidates with the same negative LCB(U), one illiquid
    (liquidity_factor=0.3) and one liquid (liquidity_factor=1.0). Under the
    literal spec formula the illiquid one would score *higher* (closer to
    zero) than the liquid one -- backwards. After the fix, the more liquid
    candidate must score at least as high (never worse for being *more*
    liquid).
    """
    _, illiquid_score = score(**_NEG_CASE, liquidity_factor=0.3, cfg=_NEUTRAL_CFG)
    _, liquid_score = score(**_NEG_CASE, liquidity_factor=1.0, cfg=_NEUTRAL_CFG)
    assert liquid_score > illiquid_score


@pytest.mark.parametrize("case", [_POS_CASE, _NEG_CASE], ids=["positive_u", "negative_u"])
@pytest.mark.parametrize(
    "factor_name",
    [
        "liquidity_factor",
        "calibration_factor",
        "strategy_posterior_factor",
        "positive_memory_factor",
    ],
)
def test_score_monotonic_in_each_factor_both_utility_signs(
    case: dict[str, float], factor_name: str
) -> None:
    """Each of the four multipliers must be monotonically non-decreasing in
    the final score, holding everything else fixed -- regardless of whether
    LCB(U) is positive or negative. (Before the fix, this failed for every
    factor whenever LCB(U) < 0.)
    """
    base_kwargs: dict[str, float] = dict(
        liquidity_factor=1.0,
        calibration_factor=1.0,
        strategy_posterior_factor=1.0,
        positive_memory_factor=1.0,
    )
    low = dict(base_kwargs)
    high = dict(base_kwargs)
    low[factor_name] = 0.3
    high[factor_name] = 0.9

    _, score_low = score(**case, **low, cfg=_NEUTRAL_CFG)
    _, score_high = score(**case, **high, cfg=_NEUTRAL_CFG)
    assert score_high >= score_low


def test_score_positive_and_negative_utility_agree_at_boundary_zero() -> None:
    """The multiply/divide branches must agree exactly at LCB(U) == 0 (no
    discontinuity introduced by the fix at the sign boundary)."""
    zero_case = dict(_POS_CASE, lcb_net_return=0.0)
    utility, final = score(**zero_case, liquidity_factor=0.3, cfg=_NEUTRAL_CFG)
    assert utility == pytest.approx(0.0)
    assert final == pytest.approx(0.0)


def test_score_negative_utility_zero_multiplier_uses_eps_floor_not_zero_division() -> None:
    """A multiplier of exactly 0.0 (e.g. positive_memory_factor=0.0) is a
    legal input (Field ge=0.0); dividing a negative utility by it must not
    raise ZeroDivisionError or produce inf/nan -- the eps floor keeps it
    finite (and very negative, correctly penalizing a zero-quality
    multiplier hard)."""
    utility, final = score(
        **_NEG_CASE, liquidity_factor=1.0, positive_memory_factor=0.0, cfg=_NEUTRAL_CFG
    )
    assert utility < 0.0
    assert final < utility  # far more negative than dividing by a real factor would give
    assert math.isfinite(final)
