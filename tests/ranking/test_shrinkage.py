from __future__ import annotations

import numpy as np
import pytest

from turboedge.ranking.shrinkage import (
    ShrinkageConfig,
    leverage_bucket_for,
    shrink_group_means,
    shrinkage_group_key,
    shrinkage_intensity,
)
from turboedge.storage.schemas import Direction


def test_leverage_bucket_boundaries() -> None:
    assert leverage_bucket_for(2.0) == "2-3"
    assert leverage_bucket_for(2.99) == "2-3"
    assert leverage_bucket_for(3.0) == "3-4"
    assert leverage_bucket_for(7.5) == "6-8"
    assert leverage_bucket_for(15.0) == ">15"
    assert leverage_bucket_for(1000.0) == ">15"


def test_leverage_bucket_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        leverage_bucket_for(0.0)
    with pytest.raises(ValueError):
        leverage_bucket_for(-1.0)


def test_shrinkage_group_key_distinguishes_direction_and_bucket() -> None:
    a = shrinkage_group_key("DAX", Direction.LONG, "5-6")
    b = shrinkage_group_key("DAX", Direction.SHORT, "5-6")
    c = shrinkage_group_key("DAX", Direction.LONG, "6-8")
    assert len({a, b, c}) == 3


def test_shrinkage_intensity_decreases_with_group_size() -> None:
    small = shrinkage_intensity(0.02, 0.001, 0.001, n=2)
    large = shrinkage_intensity(0.02, 0.001, 0.001, n=200)
    assert small > large
    assert 0.0 <= large <= small <= 1.0


def test_shrinkage_intensity_increases_with_noise_terms() -> None:
    base = shrinkage_intensity(0.0, 0.0, 0.0, n=20)
    more_dispersion = shrinkage_intensity(0.05, 0.0, 0.0, n=20)
    more_mc_se = shrinkage_intensity(0.0, 0.05, 0.0, n=20)
    more_uncertainty = shrinkage_intensity(0.0, 0.0, 0.05, n=20)
    assert more_dispersion > base
    assert more_mc_se > base
    assert more_uncertainty > base


def test_shrinkage_intensity_small_groups_are_strongly_shrunk() -> None:
    # Build Contract v2 W7 requirement 3: groups < 5 -> strong shrinkage,
    # even with essentially no measured noise/dispersion.
    intensity = shrinkage_intensity(1e-9, 1e-9, 1e-9, n=3)
    assert intensity > 0.5


def test_shrinkage_intensity_validates_inputs() -> None:
    with pytest.raises(ValueError):
        shrinkage_intensity(-0.01, 0.0, 0.0, n=5)
    with pytest.raises(ValueError):
        shrinkage_intensity(0.0, 0.0, 0.0, n=0)


def test_shrink_group_means_reduces_dispersion_for_identical_true_ev() -> None:
    """If every product in a group has the same *true* EV and raw estimates
    differ only by simulation noise, shrinkage should reduce the spread of
    the resulting estimates (Build Contract v2 W7 test requirement)."""
    rng = np.random.default_rng(7)
    true_ev = 0.03
    raw = (true_ev + rng.normal(0.0, 0.02, size=40)).tolist()
    mc_se = [0.01] * 40
    uncertainty = [0.01] * 40

    shrunk, intensity = shrink_group_means(raw, mc_se, uncertainty)

    assert 0.0 < intensity < 1.0
    assert np.std(shrunk) < np.std(raw)


def test_shrink_group_means_preserves_order_within_group() -> None:
    raw = [0.01, 0.05, -0.02, 0.10, 0.00]
    mc_se = [0.01, 0.02, 0.015, 0.03, 0.01]
    uncertainty = [0.01, 0.01, 0.02, 0.01, 0.015]

    shrunk, intensity = shrink_group_means(raw, mc_se, uncertainty)

    raw_order = sorted(range(len(raw)), key=lambda i: raw[i])
    shrunk_order = sorted(range(len(shrunk)), key=lambda i: shrunk[i])
    assert raw_order == shrunk_order
    assert 0.0 <= intensity <= 1.0


def test_shrink_group_means_validates_lengths() -> None:
    with pytest.raises(ValueError):
        shrink_group_means([0.1, 0.2], [0.01], [0.01, 0.01])
    with pytest.raises(ValueError):
        shrink_group_means([], [], [])


def test_shrink_group_means_custom_config_changes_intensity() -> None:
    raw = [0.01, 0.02, 0.03, 0.015, 0.025]
    mc_se = [0.01] * 5
    uncertainty = [0.01] * 5

    default_shrunk, default_intensity = shrink_group_means(raw, mc_se, uncertainty)
    weak_cfg = ShrinkageConfig(
        prior_pseudo_count=0.1,
        weight_dispersion=0.0,
        weight_mc_standard_error=0.0,
        weight_model_uncertainty=0.0,
    )
    weak_shrunk, weak_intensity = shrink_group_means(raw, mc_se, uncertainty, cfg=weak_cfg)

    assert weak_intensity < default_intensity
    assert np.std(weak_shrunk) > np.std(default_shrunk) - 1e-12
