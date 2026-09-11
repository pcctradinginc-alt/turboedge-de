"""Source health scoring and status aggregation.

Computes a health score from availability, freshness, missingness,
schema_consistency, and optional cross_source_agreement metrics.
Weights are distributed proportionally if some metrics are unavailable.
"""

from __future__ import annotations

from datetime import datetime

from turboedge.adapters.base import HealthCheckResult
from turboedge.storage.schemas import HealthStatus, SourceHealthRecord


def score_source(
    source: str,
    checked_at: datetime,
    availability: float,
    freshness: float,
    missingness: float,
    schema_consistency: float,
    cross_source_agreement: float | None,
    message: str,
    warn_threshold: float = 0.8,
    fail_threshold: float = 0.5,
) -> SourceHealthRecord:
    """Compute a weighted health score for a data source.

    Metrics are clipped to [0, 1]. If cross_source_agreement is None,
    its weight (0.1) is redistributed to the other metrics proportionally.
    Availability of 0 always results in status FAIL.

    Weights (when all metrics present):
    - availability: 0.35
    - freshness: 0.2
    - (1 - missingness): 0.15
    - schema_consistency: 0.2
    - cross_source_agreement: 0.1

    Status assignment:
    - score >= warn_threshold: PASS
    - score >= fail_threshold (and < warn_threshold): WARN
    - score < fail_threshold (or availability == 0): FAIL

    Args:
        source: Source name (e.g., "deutsche_boerse").
        checked_at: When this check was performed.
        availability: Fraction of expected data present [0, 1].
        freshness: How recent the data is [0, 1].
        missingness: Fraction of values missing in the dataset [0, 1].
        schema_consistency: Fraction of rows conforming to schema [0, 1].
        cross_source_agreement: Agreement with other sources, or None [0, 1].
        message: Human-readable status message.
        warn_threshold: Score threshold for PASS status (default 0.8).
        fail_threshold: Score threshold for WARN status (default 0.5).

    Returns:
        SourceHealthRecord with computed score and status.
    """
    # Clip all metrics to [0, 1]
    avail = max(0.0, min(1.0, availability))
    fresh = max(0.0, min(1.0, freshness))
    miss = max(0.0, min(1.0, missingness))
    schema = max(0.0, min(1.0, schema_consistency))
    xsa = None if cross_source_agreement is None else max(0.0, min(1.0, cross_source_agreement))

    # Base weights
    weights: dict[str, float] = {
        "availability": 0.35,
        "freshness": 0.2,
        "non_missingness": 0.15,  # 1 - missingness
        "schema_consistency": 0.2,
        "cross_source_agreement": 0.1,
    }

    # Adjust weights if cross_source_agreement is missing
    if xsa is None:
        xsa_weight = weights.pop("cross_source_agreement")
        # Redistribute to other weights proportionally
        remaining_total = sum(weights.values())
        for key in weights:
            weights[key] *= 1 + xsa_weight / remaining_total

    # Compute weighted score
    score = (
        avail * weights["availability"]
        + fresh * weights["freshness"]
        + (1.0 - miss) * weights["non_missingness"]
        + schema * weights["schema_consistency"]
    )

    if xsa is not None:
        score += xsa * weights["cross_source_agreement"]

    # Clip score to [0, 1]
    score = max(0.0, min(1.0, score))

    # Determine status
    if avail == 0:
        status = HealthStatus.FAIL
    elif score >= warn_threshold:
        status = HealthStatus.PASS
    elif score >= fail_threshold:
        status = HealthStatus.WARN
    else:
        status = HealthStatus.FAIL

    return SourceHealthRecord(
        source=source,
        checked_at=checked_at,
        availability=avail,
        freshness=fresh,
        missingness=miss,
        schema_consistency=schema,
        cross_source_agreement=xsa,
        score=score,
        status=status,
        message=message,
    )


def from_healthcheck(result: HealthCheckResult) -> SourceHealthRecord:
    """Convert a HealthCheckResult into a SourceHealthRecord.

    When ok=True (PASS):
    - availability=1, freshness=1, missingness=0, schema_consistency=1

    When ok=False (WARN or FAIL):
    - availability=0, schema_consistency=0.5

    Metrics omitted (cross_source_agreement) are left None for weight redistribution.

    Args:
        result: A HealthCheckResult from an adapter's healthcheck().

    Returns:
        A SourceHealthRecord with inferred metrics.
    """
    if result.ok:
        # PASS: perfect metrics
        return score_source(
            source=result.source,
            checked_at=result.checked_at,
            availability=1.0,
            freshness=1.0,
            missingness=0.0,
            schema_consistency=1.0,
            cross_source_agreement=None,
            message=result.message,
        )
    else:
        # FAIL or WARN: degraded metrics
        return score_source(
            source=result.source,
            checked_at=result.checked_at,
            availability=0.0,
            freshness=1.0,
            missingness=0.0,
            schema_consistency=0.5,
            cross_source_agreement=None,
            message=result.message,
        )


def overall_status(records: list[SourceHealthRecord]) -> HealthStatus:
    """Aggregate status across multiple sources.

    Returns the worst (most severe) status:
    - FAIL if any record is FAIL
    - WARN if any record is WARN (and none are FAIL)
    - PASS if all are PASS (or empty list)

    Args:
        records: List of SourceHealthRecord objects.

    Returns:
        The worst status across all records.
    """
    if not records:
        return HealthStatus.PASS

    worst = HealthStatus.PASS
    for record in records:
        if record.status == HealthStatus.FAIL:
            return HealthStatus.FAIL
        elif record.status == HealthStatus.WARN:
            worst = HealthStatus.WARN

    return worst


def critical_failures(
    records: list[SourceHealthRecord], critical_sources: set[str]
) -> list[SourceHealthRecord]:
    """Filter to FAIL records from critical sources.

    Args:
        records: List of SourceHealthRecord objects.
        critical_sources: Set of source names that are critical (e.g., {"deutsche_boerse"}).

    Returns:
        List of records where status=FAIL and source in critical_sources.
    """
    return [r for r in records if r.status == HealthStatus.FAIL and r.source in critical_sources]


__all__ = [
    "critical_failures",
    "from_healthcheck",
    "overall_status",
    "score_source",
]
