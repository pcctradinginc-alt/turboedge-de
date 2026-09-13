from __future__ import annotations

import numpy as np
import pytest

from turboedge.backtest.purged_cv import PurgedWalkForwardSplit, average_uniqueness


def test_average_uniqueness_hand_example() -> None:
    # Three overlapping 3-bar windows: [0,2], [1,3], [2,4].
    t0 = np.array([0, 1, 2])
    t1 = np.array([2, 3, 4])
    weights = average_uniqueness(t0, t1)
    # concurrency per bar: 0:1, 1:2, 2:3, 3:2, 4:1
    # sample0 [0,2] -> mean(1/1, 1/2, 1/3) = 0.6111...
    # sample1 [1,3] -> mean(1/2, 1/3, 1/2) = 0.4444...
    # sample2 [2,4] -> mean(1/3, 1/2, 1/1) = 0.6111...
    assert weights[0] == pytest.approx(11.0 / 18.0)
    assert weights[1] == pytest.approx(4.0 / 9.0)
    assert weights[2] == pytest.approx(11.0 / 18.0)


def test_average_uniqueness_disjoint_windows_get_weight_one() -> None:
    t0 = np.array([0, 10, 20])
    t1 = np.array([2, 12, 22])
    weights = average_uniqueness(t0, t1)
    np.testing.assert_allclose(weights, 1.0)


def test_average_uniqueness_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        average_uniqueness(np.array([0, 1]), np.array([1]))


def test_average_uniqueness_rejects_t1_before_t0() -> None:
    with pytest.raises(ValueError):
        average_uniqueness(np.array([5]), np.array([3]))


def test_average_uniqueness_empty() -> None:
    weights = average_uniqueness(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    assert weights.shape == (0,)


def _label_windows(n: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + horizon
    return t0, t1


def test_purged_split_no_overlap_between_train_and_test_windows() -> None:
    n = 300
    horizon = 10
    embargo = 5
    t0, t1 = _label_windows(n, horizon)
    splitter = PurgedWalkForwardSplit(min_train=100, step=20, horizon=horizon, embargo=embargo)
    n_folds = 0
    for train_idx, test_idx in splitter.split(t0, t1):
        n_folds += 1
        test_t0_min = t0[test_idx].min()
        # No training sample's label window may reach into (or past) the
        # test block's earliest label start, minus the embargo buffer.
        assert np.all(t1[train_idx] < test_t0_min - embargo)
        # Train is always strictly before test chronologically.
        assert train_idx.max() < test_idx.min()
    assert n_folds > 0


def test_purged_split_n_splits_mode_produces_requested_fold_count() -> None:
    n = 200
    horizon = 5
    t0, t1 = _label_windows(n, horizon)
    splitter = PurgedWalkForwardSplit(n_splits=4, horizon=horizon, embargo=horizon)
    folds = list(splitter.split(t0, t1))
    assert len(folds) == 4
    # Test blocks are contiguous and chronologically increasing.
    prev_end = -1
    for _train_idx, test_idx in folds:
        assert test_idx.min() > prev_end
        prev_end = test_idx.max()


def test_purged_split_requires_exactly_one_mode() -> None:
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(horizon=5, embargo=1)  # neither mode
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(horizon=5, embargo=1, n_splits=3, min_train=10, step=5)  # both


def test_purged_split_rejects_negative_embargo() -> None:
    with pytest.raises(ValueError):
        PurgedWalkForwardSplit(horizon=5, embargo=-1, n_splits=3)


def test_purged_split_rejects_unsorted_t0() -> None:
    splitter = PurgedWalkForwardSplit(horizon=2, embargo=1, n_splits=2)
    t0 = np.array([5, 1, 3], dtype=np.int64)
    t1 = t0 + 2
    with pytest.raises(ValueError):
        list(splitter.split(t0, t1))


def test_purged_split_embargo_increases_purged_training_samples() -> None:
    n = 200
    horizon = 5
    t0, t1 = _label_windows(n, horizon)
    small_embargo = list(
        PurgedWalkForwardSplit(min_train=100, step=50, horizon=horizon, embargo=0).split(t0, t1)
    )
    large_embargo = list(
        PurgedWalkForwardSplit(min_train=100, step=50, horizon=horizon, embargo=20).split(t0, t1)
    )
    # Matching folds (same test blocks) -> larger embargo can only shrink
    # (never grow) the training set.
    for (train_small, test_small), (train_large, test_large) in zip(
        small_embargo, large_embargo, strict=True
    ):
        assert np.array_equal(test_small, test_large)
        assert train_large.size <= train_small.size
