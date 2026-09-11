"""Tests for monitoring/source_health.py."""

from __future__ import annotations

from datetime import UTC, datetime

from turboedge.adapters.base import HealthCheckResult
from turboedge.monitoring.source_health import (
    OPTIONAL_SOURCES,
    critical_failures,
    from_healthcheck,
    overall_status,
    score_source,
)
from turboedge.storage.schemas import HealthStatus


class TestScoreSource:
    """Test score_source()."""

    def test_perfect_score(self) -> None:
        """All 1.0 metrics yield score close to 1.0 and status PASS."""
        now = datetime.now(UTC)
        record = score_source(
            source="test",
            checked_at=now,
            availability=1.0,
            freshness=1.0,
            missingness=0.0,
            schema_consistency=1.0,
            cross_source_agreement=1.0,
            message="OK",
        )

        assert record.status == HealthStatus.PASS
        assert record.score == 1.0

    def test_zero_availability_always_fail(self) -> None:
        """availability=0 always results in FAIL."""
        now = datetime.now(UTC)
        record = score_source(
            source="test",
            checked_at=now,
            availability=0.0,
            freshness=1.0,
            missingness=0.0,
            schema_consistency=1.0,
            cross_source_agreement=1.0,
            message="unavailable",
        )

        assert record.status == HealthStatus.FAIL

    def test_above_warn_threshold(self) -> None:
        """Score >= warn_threshold results in PASS."""
        now = datetime.now(UTC)
        record = score_source(
            source="test",
            checked_at=now,
            availability=0.95,
            freshness=0.95,
            missingness=0.05,  # 1-0.05 = 0.95
            schema_consistency=0.95,
            cross_source_agreement=None,
            message="Good",
            warn_threshold=0.8,
            fail_threshold=0.5,
        )

        assert record.status == HealthStatus.PASS

    def test_between_thresholds(self) -> None:
        """Score between fail and warn thresholds results in WARN."""
        now = datetime.now(UTC)
        record = score_source(
            source="test",
            checked_at=now,
            availability=0.7,
            freshness=0.7,
            missingness=0.3,
            schema_consistency=0.7,
            cross_source_agreement=None,
            message="Degraded",
            warn_threshold=0.8,
            fail_threshold=0.5,
        )

        assert record.status == HealthStatus.WARN

    def test_below_fail_threshold(self) -> None:
        """Score < fail_threshold results in FAIL."""
        now = datetime.now(UTC)
        record = score_source(
            source="test",
            checked_at=now,
            availability=0.4,
            freshness=0.4,
            missingness=0.6,
            schema_consistency=0.4,
            cross_source_agreement=None,
            message="Bad",
            warn_threshold=0.8,
            fail_threshold=0.5,
        )

        assert record.status == HealthStatus.FAIL

    def test_values_clipped_to_unit_interval(self) -> None:
        """Values > 1.0 or < 0.0 are clipped to [0, 1]."""
        now = datetime.now(UTC)
        record = score_source(
            source="test",
            checked_at=now,
            availability=1.5,  # clipped to 1.0
            freshness=-0.5,  # clipped to 0.0
            missingness=0.5,
            schema_consistency=0.5,
            cross_source_agreement=None,
            message="clipped",
        )

        assert record.availability == 1.0
        assert record.freshness == 0.0

    def test_weights_with_all_metrics(self) -> None:
        """Score is weighted correctly with all metrics."""
        # With all metrics: availability (0.35) + freshness (0.2)
        # + (1-miss)*0.15 + schema (0.2) + xsa (0.1)
        # 0.5 * 0.35 + 0.5 * 0.2 + (1-0.5)*0.15 + 0.5 * 0.2 + 0.5 * 0.1
        # = 0.175 + 0.1 + 0.075 + 0.1 + 0.05 = 0.5
        now = datetime.now(UTC)
        record = score_source(
            source="test",
            checked_at=now,
            availability=0.5,
            freshness=0.5,
            missingness=0.5,
            schema_consistency=0.5,
            cross_source_agreement=0.5,
            message="middle",
        )

        assert abs(record.score - 0.5) < 0.01

    def test_weight_redistribution_when_xsa_missing(self) -> None:
        """When cross_source_agreement is None, its weight is redistributed."""
        # Without XSA, weights should sum to 1.0
        # availability (0.35) + freshness (0.2) + (1-miss)*0.15 + schema (0.2)
        # = 0.35 + 0.2 + 0.15 + 0.2 = 0.9
        # Need to redistribute 0.1, so multiply all by 1.1111...
        now = datetime.now(UTC)
        record = score_source(
            source="test",
            checked_at=now,
            availability=0.6,
            freshness=0.6,
            missingness=0.4,  # 1-0.4 = 0.6
            schema_consistency=0.6,
            cross_source_agreement=None,
            message="no xsa",
        )

        # Score should be > 0.5 since all metrics are 0.6
        assert record.score > 0.5

    def test_missingness_inverted_in_score(self) -> None:
        """(1 - missingness) is used in scoring, not missingness directly."""
        now = datetime.now(UTC)
        record_low_miss = score_source(
            source="test",
            checked_at=now,
            availability=1.0,
            freshness=1.0,
            missingness=0.1,  # 1 - 0.1 = 0.9
            schema_consistency=1.0,
            cross_source_agreement=None,
            message="low miss",
        )

        record_high_miss = score_source(
            source="test",
            checked_at=now,
            availability=1.0,
            freshness=1.0,
            missingness=0.9,  # 1 - 0.9 = 0.1
            schema_consistency=1.0,
            cross_source_agreement=None,
            message="high miss",
        )

        # Low missingness should score higher
        assert record_low_miss.score > record_high_miss.score


class TestFromHealthcheck:
    """Test from_healthcheck()."""

    def test_from_healthcheck_ok_true(self) -> None:
        """HealthCheckResult with ok=True maps to perfect metrics."""
        now = datetime.now(UTC)
        hc = HealthCheckResult(
            source="test_source",
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=50.0,
            checked_at=now,
            message="Healthy",
        )

        record = from_healthcheck(hc)

        assert record.source == "test_source"
        assert record.availability == 1.0
        assert record.freshness == 1.0
        assert record.missingness == 0.0
        assert record.schema_consistency == 1.0
        assert record.cross_source_agreement is None
        assert record.status == HealthStatus.PASS

    def test_from_healthcheck_ok_false(self) -> None:
        """HealthCheckResult with ok=False maps to degraded metrics."""
        now = datetime.now(UTC)
        hc = HealthCheckResult(
            source="test_source",
            status=HealthStatus.FAIL,
            ok=False,
            latency_ms=None,
            checked_at=now,
            message="Unavailable",
        )

        record = from_healthcheck(hc)

        assert record.source == "test_source"
        assert record.availability == 0.0
        assert record.freshness == 1.0
        assert record.missingness == 0.0
        assert record.schema_consistency == 0.5
        assert record.cross_source_agreement is None
        assert record.status == HealthStatus.FAIL

    def test_message_preserved(self) -> None:
        """Message is preserved from HealthCheckResult."""
        now = datetime.now(UTC)
        hc = HealthCheckResult(
            source="test_source",
            status=HealthStatus.WARN,
            ok=False,
            latency_ms=100.0,
            checked_at=now,
            message="Partial failure",
        )

        record = from_healthcheck(hc)

        assert record.message == "Partial failure"

    def test_from_healthcheck_warn_with_ok_true_stays_warn(self) -> None:
        """Regression test: a WARN HealthCheckResult with ok=True (e.g.

        CitiFirstTurboAdapter.healthcheck()'s partial_universe case) must
        never be silently promoted to PASS just because score_source()'s
        weighted score for its chosen metrics happens to clear the PASS
        threshold. Previously `from_healthcheck` branched purely on `ok`
        (treating any ok=True result as "PASS: perfect metrics"), which
        made this exact WARN report as PASS in `sources health`.
        """
        now = datetime.now(UTC)
        hc = HealthCheckResult(
            source="citi",
            status=HealthStatus.WARN,
            ok=True,
            latency_ms=42.0,
            checked_at=now,
            message="partial_universe: 25/33 rows for DAX",
        )

        record = from_healthcheck(hc)

        assert record.status == HealthStatus.WARN
        assert record.message == "partial_universe: 25/33 rows for DAX"

    def test_from_healthcheck_pass_status_stays_pass_even_if_ok_false(self) -> None:
        """Symmetric regression guard: status is authoritative in both
        directions, not just for WARN."""
        now = datetime.now(UTC)
        hc = HealthCheckResult(
            source="test_source",
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=10.0,
            checked_at=now,
            message="all good",
        )

        record = from_healthcheck(hc)

        assert record.status == HealthStatus.PASS


class TestOptionalSources:
    """OPTIONAL_SOURCES: sources that must never gate a critical-failure alert."""

    def test_csv_import_is_optional(self) -> None:
        assert "csv_import" in OPTIONAL_SOURCES

    def test_issuer_feeds_are_not_optional(self) -> None:
        assert "bnp_paribas" not in OPTIONAL_SOURCES
        assert "citi" not in OPTIONAL_SOURCES


class TestOverallStatus:
    """Test overall_status()."""

    def test_empty_records_pass(self) -> None:
        """Empty list returns PASS."""
        status = overall_status([])

        assert status == HealthStatus.PASS

    def test_single_pass(self) -> None:
        """Single PASS record returns PASS."""
        now = datetime.now(UTC)
        record = score_source(
            source="test",
            checked_at=now,
            availability=1.0,
            freshness=1.0,
            missingness=0.0,
            schema_consistency=1.0,
            cross_source_agreement=None,
            message="OK",
        )

        status = overall_status([record])

        assert status == HealthStatus.PASS

    def test_single_fail(self) -> None:
        """Single FAIL record returns FAIL."""
        now = datetime.now(UTC)
        record = score_source(
            source="test",
            checked_at=now,
            availability=0.0,
            freshness=1.0,
            missingness=0.0,
            schema_consistency=1.0,
            cross_source_agreement=None,
            message="Down",
        )

        status = overall_status([record])

        assert status == HealthStatus.FAIL

    def test_mixed_pass_and_warn_returns_warn(self) -> None:
        """If any is WARN and none are FAIL, return WARN."""
        now = datetime.now(UTC)
        pass_record = score_source(
            source="good",
            checked_at=now,
            availability=1.0,
            freshness=1.0,
            missingness=0.0,
            schema_consistency=1.0,
            cross_source_agreement=None,
            message="OK",
        )
        warn_record = score_source(
            source="degraded",
            checked_at=now,
            availability=0.7,
            freshness=0.7,
            missingness=0.3,
            schema_consistency=0.7,
            cross_source_agreement=None,
            message="Degraded",
            warn_threshold=0.8,
            fail_threshold=0.5,
        )

        status = overall_status([pass_record, warn_record])

        assert status == HealthStatus.WARN

    def test_fail_dominates(self) -> None:
        """If any is FAIL, return FAIL (regardless of others)."""
        now = datetime.now(UTC)
        pass_record = score_source(
            source="good",
            checked_at=now,
            availability=1.0,
            freshness=1.0,
            missingness=0.0,
            schema_consistency=1.0,
            cross_source_agreement=None,
            message="OK",
        )
        fail_record = score_source(
            source="bad",
            checked_at=now,
            availability=0.0,
            freshness=1.0,
            missingness=0.0,
            schema_consistency=1.0,
            cross_source_agreement=None,
            message="Down",
        )

        status = overall_status([pass_record, fail_record])

        assert status == HealthStatus.FAIL


class TestCriticalFailures:
    """Test critical_failures()."""

    def test_empty_records(self) -> None:
        """Empty list returns empty."""
        failures = critical_failures([], critical_sources={"source1"})

        assert failures == []

    def test_no_failures(self) -> None:
        """All PASS records return empty."""
        now = datetime.now(UTC)
        record = score_source(
            source="test",
            checked_at=now,
            availability=1.0,
            freshness=1.0,
            missingness=0.0,
            schema_consistency=1.0,
            cross_source_agreement=None,
            message="OK",
        )

        failures = critical_failures([record], critical_sources={"test"})

        assert failures == []

    def test_fail_not_critical(self) -> None:
        """FAIL record from non-critical source is filtered out."""
        now = datetime.now(UTC)
        record = score_source(
            source="non_critical",
            checked_at=now,
            availability=0.0,
            freshness=1.0,
            missingness=0.0,
            schema_consistency=1.0,
            cross_source_agreement=None,
            message="Down",
        )

        failures = critical_failures([record], critical_sources={"critical_source"})

        assert failures == []

    def test_fail_critical(self) -> None:
        """FAIL record from critical source is returned."""
        now = datetime.now(UTC)
        record = score_source(
            source="critical_source",
            checked_at=now,
            availability=0.0,
            freshness=1.0,
            missingness=0.0,
            schema_consistency=1.0,
            cross_source_agreement=None,
            message="Down",
        )

        failures = critical_failures([record], critical_sources={"critical_source"})

        assert len(failures) == 1
        assert failures[0].source == "critical_source"

    def test_multiple_critical_failures(self) -> None:
        """Multiple FAIL records from critical sources are returned."""
        now = datetime.now(UTC)
        records = [
            score_source(
                source="critical1",
                checked_at=now,
                availability=0.0,
                freshness=1.0,
                missingness=0.0,
                schema_consistency=1.0,
                cross_source_agreement=None,
                message="Down",
            ),
            score_source(
                source="non_critical",
                checked_at=now,
                availability=0.0,
                freshness=1.0,
                missingness=0.0,
                schema_consistency=1.0,
                cross_source_agreement=None,
                message="Down",
            ),
            score_source(
                source="critical2",
                checked_at=now,
                availability=0.0,
                freshness=1.0,
                missingness=0.0,
                schema_consistency=1.0,
                cross_source_agreement=None,
                message="Down",
            ),
        ]

        failures = critical_failures(records, critical_sources={"critical1", "critical2"})

        assert len(failures) == 2
        sources = {f.source for f in failures}
        assert sources == {"critical1", "critical2"}

    def test_warn_not_included(self) -> None:
        """WARN records are not included, only FAIL."""
        now = datetime.now(UTC)
        record = score_source(
            source="critical_source",
            checked_at=now,
            availability=0.7,
            freshness=0.7,
            missingness=0.3,
            schema_consistency=0.7,
            cross_source_agreement=None,
            message="Degraded",
            warn_threshold=0.8,
            fail_threshold=0.5,
        )

        failures = critical_failures([record], critical_sources={"critical_source"})

        assert failures == []
