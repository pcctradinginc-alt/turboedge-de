"""Concept drift detection: Page-Hinkley test (Master Spec §30).

Tracks a stream of calibration errors or realized returns for one
``(signal_family, horizon)`` (or any other stream identity a caller wants,
e.g. per-issuer pricing drift) and flags when its mean has shifted enough to
warrant a weight-reduction recommendation. CLAUDE.md rule 32: drift reduces
weights, it never auto-deletes a model -- this module only ever
*recommends* a weight reduction (via a persisted :class:`DriftEvent`); the
actual reduction is applied by :mod:`turboedge.learning.registry` /
:mod:`turboedge.learning.ensemble_weights`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import DriftEvent


class PageHinkleyConfig(BaseModel):
    """This module's own config (wired into config.py/YAML by the
    integration wave)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Minimum magnitude of change considered meaningful (tolerance against
    # noise); values much smaller than this are treated as no drift.
    delta: float = Field(default=0.005, ge=0)
    # Detection threshold on the cumulative Page-Hinkley statistic; larger
    # lambda_ -> fewer, more confident detections.
    lambda_: float = Field(default=0.05, gt=0)
    # Recommended relative weight reduction on detection (consumed by the
    # caller, e.g. `ModelRegistry.update_weights` via a synthetic negative
    # utility) -- this module does not apply it itself.
    recommended_weight_multiplier: float = Field(default=0.5, gt=0, le=1.0)


@dataclass
class PageHinkley:
    """Online (one-observation-at-a-time) Page-Hinkley change detector.

    Standard formulation (Page 1954 / Hinkley 1971, as used for concept
    drift e.g. in Gama et al.): maintains a running mean ``mean_hat`` of the
    stream and a cumulative sum ``m_t = sum_{i<=t} (x_i - mean_hat_i -
    delta)``; drift is flagged once ``M_t - m_t > lambda_``, where
    ``M_t = max(m_1, ..., m_t)``.

    Usage::

        ph = PageHinkley(config=PageHinkleyConfig())
        for x in stream:
            if ph.update(x):
                ...  # drift detected, ph.reset() to keep monitoring
    """

    config: PageHinkleyConfig = field(default_factory=PageHinkleyConfig)
    _n: int = field(default=0, repr=False)
    _mean_hat: float = field(default=0.0, repr=False)
    _cumulative: float = field(default=0.0, repr=False)
    _min_cumulative: float = field(default=0.0, repr=False)

    @property
    def statistic(self) -> float:
        """Current Page-Hinkley statistic ``PH_t = M_t - m_t`` (>= 0);
        drift is flagged once this exceeds ``config.lambda_``."""
        return self._cumulative - self._min_cumulative

    @property
    def n(self) -> int:
        return self._n

    def reset(self) -> None:
        """Reset all running state (typically called right after a drift
        detection, to keep monitoring the stream going forward)."""
        self._n = 0
        self._mean_hat = 0.0
        self._cumulative = 0.0
        self._min_cumulative = 0.0

    def update(self, x: float) -> bool:
        """Fold in one new observation; returns ``True`` iff drift is
        detected on this observation (``statistic > config.lambda_``)."""
        self._n += 1
        # incremental mean update (Welford-style, no need to keep the full
        # stream in memory)
        self._mean_hat += (x - self._mean_hat) / self._n
        self._cumulative += x - self._mean_hat - self.config.delta
        self._min_cumulative = min(self._min_cumulative, self._cumulative)
        return self.statistic > self.config.lambda_

    def update_many(self, xs: list[float]) -> int | None:
        """Feed a batch of observations in order; returns the 1-based index
        (within ``xs``) of the first observation that triggers detection,
        or ``None`` if none does. Does not auto-reset on detection -- the
        caller decides (mirrors :meth:`update`)."""
        for i, x in enumerate(xs, start=1):
            if self.update(x):
                return i
        return None


def record_drift_event(
    store: Store,
    *,
    stream_id: str,
    signal_family: str | None,
    metric: str,
    detector: PageHinkley,
    detected_at: datetime | None = None,
) -> DriftEvent:
    """Persist a :class:`DriftEvent` for a detector that just fired, with
    the recommended action being a weight reduction (never a deletion --
    CLAUDE.md rule 32)."""
    event = DriftEvent(
        event_id=uuid.uuid4().hex,
        detected_at=detected_at if detected_at is not None else datetime.now(UTC),
        stream_id=stream_id,
        signal_family=signal_family,
        metric=metric,
        ph_statistic=detector.statistic,
        threshold=detector.config.lambda_,
        action="weight_reduction_recommended",
        details={
            "recommended_weight_multiplier": detector.config.recommended_weight_multiplier,
            "n_observations": detector.n,
        },
    )
    store.insert_drift_event(event)
    return event


__all__ = ["PageHinkley", "PageHinkleyConfig", "record_drift_event"]
