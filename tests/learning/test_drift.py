from __future__ import annotations

import numpy as np

from turboedge.learning.drift import PageHinkley, PageHinkleyConfig, record_drift_event


def test_page_hinkley_detects_a_mean_jump() -> None:
    rng = np.random.default_rng(1)
    stable = rng.normal(0.0, 0.01, 100)
    shifted = rng.normal(0.1, 0.01, 100)
    stream = list(stable) + list(shifted)

    ph = PageHinkley(config=PageHinkleyConfig(delta=0.005, lambda_=0.5))
    idx = ph.update_many(stream)

    assert idx is not None
    assert idx > 100  # detected only after the shift, not spuriously early
    assert idx < 130  # detected reasonably promptly after the shift


def test_page_hinkley_no_detection_on_stable_stream() -> None:
    rng = np.random.default_rng(2)
    stream = list(rng.normal(0.0, 0.01, 300))
    ph = PageHinkley(config=PageHinkleyConfig(delta=0.005, lambda_=0.5))
    assert ph.update_many(stream) is None


def test_page_hinkley_reset_clears_state() -> None:
    ph = PageHinkley(config=PageHinkleyConfig(delta=0.0, lambda_=0.01))
    ph.update(1.0)
    ph.update(1.0)
    assert ph.n == 2
    ph.reset()
    assert ph.n == 0
    assert ph.statistic == 0.0


def test_record_drift_event_persists_weight_reduction_recommendation(store) -> None:
    ph = PageHinkley(config=PageHinkleyConfig(delta=0.0, lambda_=0.01))
    ph.update(1.0)
    ph.update(1.0)

    event = record_drift_event(
        store,
        stream_id="calibration_error:tsmom:7d",
        signal_family="tsmom",
        metric="calibration_error",
        detector=ph,
    )
    assert event.action == "weight_reduction_recommended"

    persisted = store.list_drift_events("calibration_error:tsmom:7d")
    assert len(persisted) == 1
    assert persisted[0].event_id == event.event_id
    assert persisted[0].signal_family == "tsmom"
