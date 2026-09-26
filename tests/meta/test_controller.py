"""Shadow meta-controller (M5)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from turboedge.meta.controller import decide
from turboedge.meta.schemas import MetaDecisionKind


def base_kwargs(start: datetime, **overrides: object) -> dict:
    kwargs: dict = dict(
        run_id="r1",
        underlying_id="DAX",
        horizon_days=7,
        prediction_time=start + timedelta(days=299),
        registry=[],
        forecasts=[],
        walkforward=[],
        data_quality=1.0,
        stale_share=0.0,
        ratio_unverified_share=0.0,
        integrity_fail_share=0.0,
        config_hash="c",
        now=start,
    )
    kwargs.update(overrides)
    return kwargs


def test_empty_evidence_base_abstains(
    bars_factory: Callable[..., list],
    entry_factory: Callable[..., object],
    forecast_factory: Callable[..., object],
    start: datetime,
) -> None:
    """This repository's actual state today: no walk-forward results, no
    labels, no posteriors. The controller must abstain rather than proceed
    on an empty evidence base.
    """
    d = decide(
        bars=bars_factory(300),
        **base_kwargs(start, registry=[entry_factory("m1")], forecasts=[forecast_factory("m1")]),
    )
    assert d.decision is MetaDecisionKind.ABSTAIN
    assert d.shadow_mode is True
    assert any("incomplete evidence" in r for r in d.reasons)


def test_decision_is_always_shadow_in_phase_one(
    bars_factory: Callable[..., list],
    entry_factory: Callable[..., object],
    forecast_factory: Callable[..., object],
    walkforward_factory: Callable[..., object],
    start: datetime,
) -> None:
    """Phase 1 must never emit a non-shadow decision, whatever the inputs."""
    d = decide(
        bars=bars_factory(400),
        **base_kwargs(
            start,
            registry=[entry_factory("m1")],
            forecasts=[forecast_factory("m1")],
            walkforward=[walkforward_factory("m1")],
        ),
    )
    assert d.shadow_mode is True


def test_every_reason_is_present_and_non_empty(
    bars_factory: Callable[..., list],
    entry_factory: Callable[..., object],
    forecast_factory: Callable[..., object],
    start: datetime,
) -> None:
    """An explanation that cannot be traced to a number does not belong in
    the list -- and an empty list would make the decision unauditable."""
    d = decide(
        bars=bars_factory(300),
        **base_kwargs(start, registry=[entry_factory("m1")], forecasts=[forecast_factory("m1")]),
    )
    assert d.reasons
    assert all(isinstance(r, str) and r.strip() for r in d.reasons)


def test_unclassifiable_regime_is_reported_and_abstains(
    bars_factory: Callable[..., list],
    entry_factory: Callable[..., object],
    forecast_factory: Callable[..., object],
    start: datetime,
) -> None:
    d = decide(
        bars=bars_factory(20),  # far too short to classify
        **base_kwargs(
            start,
            prediction_time=start + timedelta(days=19),
            registry=[entry_factory("m1")],
            forecasts=[forecast_factory("m1")],
        ),
    )
    assert d.volatility_regime == "unknown"
    assert d.regime_observation_count == 0
    assert d.decision is MetaDecisionKind.ABSTAIN
    assert any("not classifiable" in r for r in d.reasons)


def test_no_models_at_all_abstains(bars_factory: Callable[..., list], start: datetime) -> None:
    d = decide(bars=bars_factory(300), **base_kwargs(start))
    assert d.decision is MetaDecisionKind.ABSTAIN
    assert d.selected_models == []


def test_good_evidence_can_reach_proceed(
    bars_factory: Callable[..., list],
    entry_factory: Callable[..., object],
    forecast_factory: Callable[..., object],
    walkforward_factory: Callable[..., object],
    start: datetime,
) -> None:
    """The counterpart the abstention tests need.

    Without this, a controller that abstained unconditionally would pass
    every other test in this file -- which would make it useless while
    looking rigorous.
    """
    bars = bars_factory(600)
    d = decide(
        bars=bars,
        **base_kwargs(
            start,
            prediction_time=start + timedelta(days=599),
            registry=[entry_factory("m1"), entry_factory("m2")],
            forecasts=[forecast_factory("m1"), forecast_factory("m2")],
            walkforward=[
                walkforward_factory("m1", brier=0.18, brier_null=0.25, ece=0.005),
                walkforward_factory("m2", brier=0.19, brier_null=0.25, ece=0.005),
            ],
        ),
    )
    assert d.decision is not MetaDecisionKind.ABSTAIN, (
        f"abstained despite strong evidence: {d.reasons}"
    )
    assert d.selected_models
    assert abs(sum(d.model_weights.values()) - 1.0) < 1e-12
