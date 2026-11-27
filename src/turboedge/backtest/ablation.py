"""Read-only paired OOS diagnostics. No automatic promotion or filter changes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite

import numpy as np

from turboedge.backtest.walkforward import WalkForwardResult


@dataclass(frozen=True, slots=True)
class PairedEffect:
    full_rows: int
    reduced_rows: int
    matched_rows: int
    brier_gain: float | None
    gross_return_gain: float | None
    brier_interval: tuple[float, float] | None
    gross_return_interval: tuple[float, float] | None


def _interval(values: list[float], horizon: int, tests: int) -> tuple[float, float] | None:
    # Overlapping labels are not independent. Vary contiguous block length.
    if len(values) < max(60, 10 * horizon):
        return None
    arr = np.asarray(values)
    rng = np.random.default_rng(20261126)
    intervals = []
    for block in (horizon, 2 * horizon):
        count = (len(arr) + block - 1) // block
        starts = rng.integers(0, len(arr) - block + 1, size=(1000, count))
        indices = (starts[:, :, None] + np.arange(block)).reshape(1000, -1)
        means = arr[indices[:, : len(arr)]].mean(axis=1)
        tail = 0.025 / tests
        intervals.append((float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))))
    return min(x[0] for x in intervals), max(x[1] for x in intervals)


def compare(full: WalkForwardResult, reduced: WalkForwardResult, *, tests: int = 1) -> PairedEffect:
    """Positive gains favor the full model; no inference on unmatched rows."""
    if tests < 1:
        raise ValueError("tests must be positive")
    if full.horizon_days != reduced.horizon_days:
        raise ValueError("horizon mismatch")

    def index(result: WalkForwardResult) -> dict[datetime, tuple[float, float]]:
        rows: dict[datetime, tuple[float, float]] = {}
        for at, p, ret in result.oos_predictions:
            if at in rows or not (0 <= p <= 1 and isfinite(ret)):
                raise ValueError("duplicate or invalid OOS observation")
            rows[at] = (p, ret)
        return rows

    a, b = index(full), index(reduced)
    brier: list[float] = []
    gross: list[float] = []
    for at in sorted(a.keys() & b.keys()):
        p, ret = a[at]
        q, other = b[at]
        if ret != other:
            raise ValueError("realized outcome mismatch")
        y = float(ret > 0)
        brier.append((q - y) ** 2 - (p - y) ** 2)
        gross.append(float(np.sign(p - 0.5) - np.sign(q - 0.5)) * ret)
    return PairedEffect(
        full_rows=len(a),
        reduced_rows=len(b),
        matched_rows=len(brier),
        brier_gain=float(np.mean(brier)) if brier else None,
        gross_return_gain=float(np.mean(gross)) if gross else None,
        brier_interval=_interval(brier, full.horizon_days, tests),
        gross_return_interval=_interval(gross, full.horizon_days, tests),
    )
