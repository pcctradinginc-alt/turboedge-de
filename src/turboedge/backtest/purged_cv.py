"""Purged, embargoed walk-forward CV and average-uniqueness sample weights.

Formula reference: Master Spec §9.3, §27.4, §28 (Lopez de Prado, "Advances in
Financial Machine Learning", ch. 4 & 7). CLAUDE.md rules 6-8: no random
time-series splits; purged CV plus embargo; average-uniqueness weights for
overlapping samples.

Every sample here is identified by a half-open-ish label window
``[t0, t1]`` (integer bar indices, ``t1 >= t0``) -- e.g. a horizon-``h``
forecast made "as of" bar ``t0`` is only actually resolved once bar ``t1 =
t0 + h`` closes, so its label secretly depends on every bar in between.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def average_uniqueness(
    t0_index: npt.NDArray[np.int64], t1_index: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """Average-uniqueness sample weight per overlapping label window (Lopez de Prado ch. 4).

    For each sample ``i`` with label window ``[t0_i, t1_i]``, the
    concurrency ``c_t`` at bar ``t`` is the number of samples whose window
    covers ``t``; ``uniqueness_i = mean_{t in [t0_i, t1_i]}(1 / c_t)``. A
    sample whose window never overlaps any other sample's gets weight 1.0
    (fully unique); heavily overlapping samples get weights close to 0.

    Args:
        t0_index: label window start (inclusive), one per sample.
        t1_index: label window end (inclusive), one per sample, ``>= t0_index``.

    Returns:
        One weight per sample, same order as the inputs.
    """
    t0 = np.asarray(t0_index, dtype=np.int64)
    t1 = np.asarray(t1_index, dtype=np.int64)
    if t0.shape != t1.shape:
        raise ValueError(
            f"t0_index and t1_index must have the same shape, got {t0.shape!r} and {t1.shape!r}"
        )
    if t0.ndim != 1:
        raise ValueError(f"t0_index/t1_index must be 1-dimensional, got shape {t0.shape!r}")
    if t0.size == 0:
        return np.zeros(0, dtype=np.float64)
    if np.any(t1 < t0):
        raise ValueError("t1_index must be >= t0_index elementwise")

    span_min = int(t0.min())
    span_max = int(t1.max())
    length = span_max - span_min + 1
    concurrency = np.zeros(length, dtype=np.float64)
    for a, b in zip(t0.tolist(), t1.tolist(), strict=True):
        concurrency[a - span_min : b - span_min + 1] += 1.0

    weights = np.empty(t0.shape[0], dtype=np.float64)
    for i, (a, b) in enumerate(zip(t0.tolist(), t1.tolist(), strict=True)):
        window = concurrency[a - span_min : b - span_min + 1]
        weights[i] = float(np.mean(1.0 / window))
    return weights


@dataclass
class PurgedWalkForwardSplit:
    """Chronological (never random) train/test index generator with purging + embargo.

    Two construction modes:

    - ``n_splits``: split the sample axis into an initial expanding training
      block followed by ``n_splits`` roughly-equal contiguous test blocks.
    - ``min_train`` + ``step``: classic expanding-window walk-forward -- the
      first test block starts right after ``min_train`` samples, and each
      subsequent block covers the next ``step`` samples, until the data is
      exhausted.

    Exactly one of the two modes must be specified.

    For every fold, ``split`` purges any candidate training sample whose
    label window ``[t0, t1]`` overlaps the test block's label-window span,
    *and* additionally excludes training samples ending within ``embargo``
    bars before that span (the combined purge+embargo treatment: purging
    alone removes exact overlap, embargo adds a safety buffer against
    residual serial dependence). Training samples are always strictly
    chronologically before the test block (this is a walk-forward, not a
    k-fold with wraparound), so no "training after test" case can arise.
    """

    horizon: int
    embargo: int
    n_splits: int | None = None
    min_train: int | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon!r}")
        if self.embargo < 0:
            raise ValueError(f"embargo must be >= 0, got {self.embargo!r}")
        uses_n_splits = self.n_splits is not None
        uses_expanding = self.min_train is not None or self.step is not None
        if uses_n_splits == uses_expanding:
            raise ValueError("specify exactly one of: n_splits, or both min_train and step")
        if uses_expanding and (self.min_train is None or self.step is None):
            raise ValueError("min_train and step must both be given together")
        if self.n_splits is not None and self.n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {self.n_splits!r}")
        if self.min_train is not None and self.min_train < 1:
            raise ValueError(f"min_train must be >= 1, got {self.min_train!r}")
        if self.step is not None and self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step!r}")

    def _test_blocks(self, n: int) -> list[tuple[int, int]]:
        """``[start, end)`` sample-index ranges per fold's test block, chronological."""
        if self.min_train is not None and self.step is not None:
            blocks: list[tuple[int, int]] = []
            start = self.min_train
            while start < n:
                end = min(start + self.step, n)
                blocks.append((start, end))
                start = end
            return blocks

        n_splits = self.n_splits
        if n_splits is None:  # pragma: no cover -- guarded by __post_init__ mode check
            raise ValueError("n_splits must be set in n_splits mode")
        block_size = n // (n_splits + 1)
        if block_size < 1:
            raise ValueError(f"not enough samples ({n}) for n_splits={n_splits}")
        blocks = []
        start = block_size
        for i in range(n_splits):
            end = start + block_size if i < n_splits - 1 else n
            blocks.append((start, end))
            start = end
        return blocks

    def split(
        self, t0: npt.NDArray[np.int64], t1: npt.NDArray[np.int64]
    ) -> Iterator[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
        """Yield ``(train_idx, test_idx)`` index arrays into the sample axis ``0..len(t0)-1``.

        ``t0``/``t1`` must be sorted ascending by ``t0`` (samples in
        chronological order) -- the caller's sample axis, not necessarily
        equal to bar indices, though it usually is.
        """
        t0_arr = np.asarray(t0, dtype=np.int64)
        t1_arr = np.asarray(t1, dtype=np.int64)
        n = t0_arr.shape[0]
        if t1_arr.shape[0] != n:
            raise ValueError("t0 and t1 must have the same length")
        if n > 1 and np.any(np.diff(t0_arr) < 0):
            raise ValueError("t0 must be sorted ascending (samples in chronological order)")

        for test_start, test_end in self._test_blocks(n):
            test_idx = np.arange(test_start, test_end, dtype=np.int64)
            if test_idx.size == 0:
                continue
            test_t0_min = int(t0_arr[test_idx].min())
            candidate_train = np.arange(0, test_start, dtype=np.int64)
            if candidate_train.size == 0:
                continue
            purge_boundary = test_t0_min - self.embargo
            keep = t1_arr[candidate_train] < purge_boundary
            train_idx = candidate_train[keep]
            if train_idx.size == 0:
                continue
            yield train_idx, test_idx
