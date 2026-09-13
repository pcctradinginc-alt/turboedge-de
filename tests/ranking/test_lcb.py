from __future__ import annotations

import pytest

from turboedge.ranking.lcb import LcbConfig, lower_confidence_bound


def test_lcb_never_exceeds_central_mean() -> None:
    lcb = lower_confidence_bound(
        central_mean=0.05, pessimistic_mean=0.01, mc_standard_error=0.005, shrunk_mean=0.04
    )
    assert lcb <= 0.05


def test_lcb_never_exceeds_shrunk_mean() -> None:
    lcb = lower_confidence_bound(
        central_mean=0.05, pessimistic_mean=0.05, mc_standard_error=0.0, shrunk_mean=0.02
    )
    assert lcb <= 0.02


def test_lcb_matches_pessimistic_scenario_reduced_by_mc_error_when_aligned() -> None:
    # When shrunk_mean == central_mean, the LCB reduces exactly to
    # "pessimistic scenario, further reduced by z * mc_standard_error".
    lcb = lower_confidence_bound(
        central_mean=0.05, pessimistic_mean=0.02, mc_standard_error=0.01, shrunk_mean=0.05, z=1.645
    )
    assert lcb == pytest.approx(0.02 - 1.645 * 0.01)


def test_lcb_ignores_upward_pessimistic_noise() -> None:
    # A "pessimistic" scenario that happens to land above central (Monte
    # Carlo noise) must never raise the bound.
    lcb_noisy = lower_confidence_bound(
        central_mean=0.05, pessimistic_mean=0.06, mc_standard_error=0.0, shrunk_mean=0.05
    )
    lcb_aligned = lower_confidence_bound(
        central_mean=0.05, pessimistic_mean=0.05, mc_standard_error=0.0, shrunk_mean=0.05
    )
    assert lcb_noisy == pytest.approx(lcb_aligned)


def test_lcb_monotonic_in_mc_standard_error() -> None:
    low = lower_confidence_bound(0.05, 0.02, 0.001, 0.05)
    high = lower_confidence_bound(0.05, 0.02, 0.05, 0.05)
    assert high < low


def test_lcb_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        lower_confidence_bound(0.05, 0.02, -0.01, 0.05)
    with pytest.raises(ValueError):
        lower_confidence_bound(0.05, 0.02, 0.01, 0.05, z=0.0)


def test_lcb_config_defaults() -> None:
    cfg = LcbConfig()
    assert cfg.z == pytest.approx(1.645)
