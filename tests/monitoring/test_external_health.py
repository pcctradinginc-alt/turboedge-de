"""Pin fail-closed external-source health without endangering saved state."""

from __future__ import annotations

from turboedge.monitoring.external_health import evaluate_report


def test_clean_ingest_is_healthy() -> None:
    healthy, _ = evaluate_report({"mode": "external_ingest", "counts": {"series": 12, "failures": 0}})
    assert healthy


def test_ingest_failures_are_alarm_even_when_pipeline_saved_state() -> None:
    healthy, reason = evaluate_report(
        {"mode": "external_ingest", "counts": {"series": 12, "failures": 11}}
    )
    assert not healthy
    assert "11" in reason


def test_missing_counts_and_invalid_shapes_fail_closed() -> None:
    samples = [
        {},
        {"mode": "external_ingest"},
        {"mode": "other", "counts": {"series": 1, "failures": 0}},
        {"mode": "external_ingest", "counts": {"series": 1, "failures": "0"}},
        {"mode": "external_ingest", "counts": {"series": 0, "failures": 0}},
    ]
    assert all(not evaluate_report(sample)[0] for sample in samples)


def test_intentionally_skipped_optional_sources_are_not_ingest_failures() -> None:
    healthy, _ = evaluate_report(
        {"mode": "external_ingest", "counts": {"series": 0, "failures": 0},
         "skipped": {"entsoe": "AUTH_MISSING"}}
    )
    assert healthy
