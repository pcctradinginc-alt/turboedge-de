"""Rolling correlation clusters of underlyings and cluster-level risk (Master
Spec §31 "Cluster- und Ruin-Risiko").

Two distinct notions of "cluster" appear in this milestone -- this module
owns only the second:

1. The *shrinkage* group ``(underlying_id, direction, leverage_bucket
   [, horizon_days])`` used by ``ranking/shrinkage.py`` -- a within-scan
   grouping of *candidate products*, unrelated to correlation.
2. The *correlation cluster* of *underlyings* computed here: a rolling
   hierarchical clustering of underlyings' daily log returns, used for
   portfolio-level concentration/ruin risk (this module) and consumed by
   ``ranking/sizing.py`` (``max_cluster_fraction``) and
   ``ranking/gates.py`` (via ``ranking/ev.py.to_candidate_gate_input``'s
   ``cluster_risk_pass``).

Fallback (Build Contract v2 W7 requirement 6): an underlying with
insufficient overlapping history against every other underlying gets its
own singleton cluster, with a logged warning -- never silently grouped by
guesswork.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict, Field
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from turboedge.storage.schemas import UnderlyingBar

logger = structlog.get_logger(__name__)

_SINGLETON_PREFIX = "singleton"
_CORR_PREFIX = "corr"


class ClusterConfig(BaseModel):
    """Own config for this module (wired into ``config.py``/YAML by the
    integration wave).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Rolling window (trading days) of daily log returns used to estimate
    #: pairwise correlation (Master Spec §31: "rollierende
    #: Korrelationscluster").
    lookback_days: int = Field(default=120, gt=1)
    #: Minimum number of overlapping daily-return observations between a
    #: pair of underlyings required to trust their correlation estimate at
    #: all; underlyings that never reach this against any peer fall back to
    #: a singleton cluster.
    min_overlap_days: int = Field(default=30, gt=1)
    #: Hierarchical clustering linkage method (``scipy.cluster.hierarchy.
    #: linkage``).
    linkage_method: str = "average"
    #: Distance (``1 - rho``) cutoff for ``fcluster(..., criterion=
    #: "distance")``. Lower = stricter (fewer, tighter clusters); e.g. 0.5
    #: groups underlyings correlated at rho >= 0.5.
    distance_threshold: float = Field(default=0.5, gt=0.0, le=2.0)
    #: Max number of concurrent ACTIONABLE proposals allowed in one
    #: correlation cluster (Master Spec §31 "max_active_positions").
    max_active_positions_per_cluster: int = Field(default=3, gt=0)
    #: Max total capital fraction allowed concurrently deployed in one
    #: correlation cluster (Master Spec §31 "max_capital_fraction").
    max_capital_fraction_per_cluster: float = Field(default=0.25, gt=0.0)


@dataclass(frozen=True, slots=True)
class ClusterAssignment:
    """One underlying's correlation-cluster membership as of a clustering run."""

    underlying_id: str
    cluster_id: str
    #: True iff this underlying could not be reliably correlated against any
    #: peer (insufficient overlapping history) and was placed in its own
    #: singleton cluster as a fallback.
    fallback: bool


@dataclass(frozen=True, slots=True)
class OpenClusterPosition:
    """One existing open position or already-selected candidate this scan,
    for :func:`cluster_risk` to weigh against a new candidate.
    """

    underlying_id: str
    cluster_id: str
    capital_fraction: float
    #: Counts toward ``max_active_positions_per_cluster`` (an actual open
    #: position or an ACTIONABLE proposal already emitted this scan) versus
    #: e.g. a WATCH-only candidate that should not consume the position-
    #: count budget.
    counts_as_active_position: bool = True


def _daily_log_returns(
    bars: Sequence[UnderlyingBar], as_of: datetime
) -> dict[np.datetime64, float]:
    eligible = sorted((b for b in bars if b.available_at <= as_of), key=lambda b: b.ts)
    out: dict[np.datetime64, float] = {}
    prev_close: float | None = None
    for b in eligible:
        if prev_close is not None and prev_close > 0.0 and b.close > 0.0:
            out[np.datetime64(b.ts.date())] = float(np.log(b.close / prev_close))
        prev_close = b.close
    return out


def compute_clusters(
    bars_by_underlying: Mapping[str, Sequence[UnderlyingBar]],
    *,
    as_of: datetime,
    cfg: ClusterConfig | None = None,
) -> dict[str, ClusterAssignment]:
    """Rolling correlation clusters of underlyings (Master Spec §31).

    Uses each underlying's most recent ``cfg.lookback_days`` daily log
    returns as of ``as_of`` (CLAUDE.md rule 5: only bars with
    ``available_at <= as_of``). Underlyings are clustered via hierarchical
    clustering (``scipy.cluster.hierarchy``) on the distance matrix ``1 -
    rho``.

    Every ``underlying_id`` present in ``bars_by_underlying`` gets an entry.
    Falls back to a singleton cluster (with a logged warning) for:

    - fewer than 2 underlyings total (nothing to correlate against),
    - an underlying with fewer than ``cfg.min_overlap_days`` overlapping
      return observations against every other underlying,
    - a non-finite/degenerate correlation matrix (e.g. a constant-price
      series with zero variance).
    """
    c = cfg if cfg is not None else ClusterConfig()
    ids = list(bars_by_underlying.keys())
    result: dict[str, ClusterAssignment] = {}

    if len(ids) < 2:
        for uid in ids:
            result[uid] = ClusterAssignment(uid, f"{_SINGLETON_PREFIX}_{uid}", True)
            if ids:
                logger.warning("cluster_fallback_insufficient_peers", underlying_id=uid)
        return result

    returns_by_id = {
        uid: _daily_log_returns(bars, as_of) for uid, bars in bars_by_underlying.items()
    }
    # Restrict each series to its most recent lookback_days dates (by date key).
    trimmed: dict[str, dict[np.datetime64, float]] = {}
    for uid, series in returns_by_id.items():
        dates_sorted = sorted(series.keys())
        keep = set(dates_sorted[-c.lookback_days :]) if c.lookback_days > 0 else set(dates_sorted)
        trimmed[uid] = {d: series[d] for d in keep}

    n = len(ids)
    overlap_ok = np.ones(n, dtype=bool)
    corr = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            common = trimmed[ids[i]].keys() & trimmed[ids[j]].keys()
            if len(common) < c.min_overlap_days:
                overlap_ok[i] = False
                overlap_ok[j] = False
                corr[i, j] = corr[j, i] = 0.0
                continue
            xi = np.array([trimmed[ids[i]][d] for d in common], dtype=np.float64)
            xj = np.array([trimmed[ids[j]][d] for d in common], dtype=np.float64)
            if np.std(xi) <= 0.0 or np.std(xj) <= 0.0:
                overlap_ok[i] = False
                overlap_ok[j] = False
                corr[i, j] = corr[j, i] = 0.0
                continue
            rho = float(np.corrcoef(xi, xj)[0, 1])
            if not np.isfinite(rho):
                overlap_ok[i] = False
                overlap_ok[j] = False
                rho = 0.0
            corr[i, j] = corr[j, i] = rho

    clusterable_idx = [i for i in range(n) if overlap_ok[i]]
    for i in range(n):
        if not overlap_ok[i]:
            result[ids[i]] = ClusterAssignment(ids[i], f"{_SINGLETON_PREFIX}_{ids[i]}", True)
            logger.warning("cluster_fallback_insufficient_history", underlying_id=ids[i])

    if len(clusterable_idx) < 2:
        for i in clusterable_idx:
            result[ids[i]] = ClusterAssignment(ids[i], f"{_SINGLETON_PREFIX}_{ids[i]}", True)
            logger.warning("cluster_fallback_insufficient_peers", underlying_id=ids[i])
        return result

    sub_corr = corr[np.ix_(clusterable_idx, clusterable_idx)]
    distance = np.clip(1.0 - sub_corr, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    if condensed.size == 0:
        for i in clusterable_idx:
            result[ids[i]] = ClusterAssignment(ids[i], f"{_SINGLETON_PREFIX}_{ids[i]}", True)
        return result

    z = linkage(condensed, method=c.linkage_method)
    labels = fcluster(z, t=c.distance_threshold, criterion="distance")
    for pos, i in enumerate(clusterable_idx):
        result[ids[i]] = ClusterAssignment(ids[i], f"{_CORR_PREFIX}_{int(labels[pos])}", False)

    return result


def cluster_id_for(assignments: Mapping[str, ClusterAssignment], underlying_id: str) -> str:
    """Cluster id for ``underlying_id``, falling back to its own singleton
    cluster (with a logged warning) if it is not present in ``assignments``
    at all (e.g. a brand-new underlying not yet part of any
    :func:`compute_clusters` run).
    """
    assignment = assignments.get(underlying_id)
    if assignment is not None:
        return assignment.cluster_id
    logger.warning("cluster_id_missing_assignment_fallback", underlying_id=underlying_id)
    return f"{_SINGLETON_PREFIX}_{underlying_id}"


def cluster_risk(
    open_positions: Sequence[OpenClusterPosition],
    new_candidate: OpenClusterPosition,
    *,
    cfg: ClusterConfig | None = None,
) -> float:
    """Cluster-level concentration risk for ``new_candidate``'s cluster,
    including ``new_candidate`` itself, as a ``>= 0`` utilization ratio
    (``1.0`` == exactly at a cap; ``> 1.0`` == over a cap).

    ``max(position_count_ratio, capital_fraction_ratio)`` -- the binding
    (larger) of the two Master Spec §31 per-cluster limits.
    """
    c = cfg if cfg is not None else ClusterConfig()
    same_cluster = [p for p in open_positions if p.cluster_id == new_candidate.cluster_id]

    n_active = sum(1 for p in same_cluster if p.counts_as_active_position)
    n_active += 1 if new_candidate.counts_as_active_position else 0
    count_ratio = n_active / c.max_active_positions_per_cluster

    capital_used = sum(p.capital_fraction for p in same_cluster) + new_candidate.capital_fraction
    capital_ratio = capital_used / c.max_capital_fraction_per_cluster

    return float(max(count_ratio, capital_ratio))


def cluster_risk_pass(
    open_positions: Sequence[OpenClusterPosition],
    new_candidate: OpenClusterPosition,
    *,
    cfg: ClusterConfig | None = None,
) -> bool:
    """``True`` iff admitting ``new_candidate`` stays within both per-cluster
    caps (Master Spec §19 gate ``cluster_risk == PASS``).
    """
    return cluster_risk(open_positions, new_candidate, cfg=cfg) <= 1.0


__all__ = [
    "ClusterAssignment",
    "ClusterConfig",
    "OpenClusterPosition",
    "cluster_id_for",
    "cluster_risk",
    "cluster_risk_pass",
    "compute_clusters",
]
