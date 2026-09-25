"""Tests for the W12-A Cboe volatility-state features."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from turboedge.features.cboe_volatility import FEATURE_NAMES, build_cboe_features
from turboedge.storage.schemas import ExternalObservation

START = datetime(2024, 1, 2, tzinfo=UTC)


def obs(series_id: str, day: int, value: float) -> ExternalObservation:
    observation_time = START + timedelta(days=day)
    return ExternalObservation(
        series_id=series_id,
        value=value,
        unit="index_points",
        frequency="daily",
        source_version="cboe_daily_prices_csv",
        observation_time=observation_time,
        available_at=observation_time + timedelta(days=1),
        retrieved_at=datetime(2026, 9, 25, tzinfo=UTC),
        source="cboe",
        parser_version="1",
        quality_score=1.0,
    )


def series(series_id: str, values: list[float]) -> list[ExternalObservation]:
    return [obs(series_id, i, v) for i, v in enumerate(values)]


def full_history(n: int = 300) -> list[ExternalObservation]:
    """A complete, non-degenerate history for all four needed series."""
    out: list[ExternalObservation] = []
    out += series("VIX.CLOSE", [15.0 + (i % 7) * 0.5 for i in range(n)])
    out += series("VIX9D.CLOSE", [13.0 + (i % 5) * 0.4 for i in range(n)])
    out += series("VIX3M.CLOSE", [18.0 + (i % 11) * 0.3 for i in range(n)])
    out += series("VVIX.CLOSE", [90.0 + (i % 13) * 1.1 for i in range(n)])
    return out


def prediction_after(day: int) -> datetime:
    """A prediction time just after day `day`'s value became available."""
    return START + timedelta(days=day + 1, hours=1)


def test_all_ten_features_are_produced() -> None:
    got = build_cboe_features(full_history(), prediction_after(299))
    assert set(got) == set(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == 10


def test_features_never_use_an_unavailable_observation() -> None:
    """The central guarantee: tomorrow's close must not move today's feature.

    Same history, but the second call additionally contains a far-future
    VIX spike. If any feature changed, the builder would be reading values
    it could not have had.
    """
    history = full_history()
    at = prediction_after(100)
    baseline = build_cboe_features(history, at)

    # Days 200-204: comfortably after the day-100 prediction time.
    future_spike = [obs("VIX.CLOSE", 200 + i, 99.0) for i in range(5)]
    for o in future_spike:
        assert o.available_at > at, "fixture must actually be in the future"
    with_future = build_cboe_features([*history, *future_spike], at)

    assert baseline == with_future


def test_missing_series_yields_none_not_a_substituted_value() -> None:
    """Rule 29: absent inputs stay absent.

    A zero here would be indistinguishable from a genuine flat term
    structure, which is a real and quite different market state.
    """
    only_vix = series("VIX.CLOSE", [15.0 + (i % 7) * 0.5 for i in range(300)])
    got = build_cboe_features(only_vix, prediction_after(299))
    assert got["vix_level"] is not None
    assert got["vix9d_over_vix_minus_1"] is None
    assert got["vvix_over_vix_minus_1"] is None
    assert got["term_structure_slope"] is None
    assert got["term_structure_curvature"] is None
    assert got["vol_of_vol_regime"] is None


def test_short_history_yields_none_for_window_features() -> None:
    """Too few points is not the same as an average reading."""
    short = full_history(n=5)
    got = build_cboe_features(short, prediction_after(4))
    assert got["vix_level"] is not None
    assert got["vix_zscore_20d"] is None
    assert got["volatility_regime"] is None


def test_constant_series_zscore_is_none_not_zero() -> None:
    """A flat series has no dispersion to standardise by.

    Returning 0.0 would read as "exactly average"; the honest answer is
    "not computable".
    """
    flat = series("VIX.CLOSE", [15.0] * 60)
    got = build_cboe_features(flat, prediction_after(59))
    assert got["vix_zscore_20d"] is None


def test_term_structure_signs_are_readable() -> None:
    """Inverted structure (short end richest) must give positive legs."""
    inverted = (
        series("VIX.CLOSE", [20.0] * 30)
        + series("VIX9D.CLOSE", [25.0] * 30)
        + series("VIX3M.CLOSE", [18.0] * 30)
    )
    got = build_cboe_features(inverted, prediction_after(29))
    assert got["vix9d_over_vix_minus_1"] == 25.0 / 20.0 - 1.0 > 0
    assert got["vix_over_vix3m_minus_1"] == 20.0 / 18.0 - 1.0 > 0
    assert got["term_structure_slope"] == 25.0 / 18.0 - 1.0 > 0


def test_curvature_is_zero_for_a_straight_term_structure() -> None:
    """Curvature must isolate the hump, not restate the slope.

    Equal proportional steps 9d->30d and 30d->3m mean no curvature, whatever
    the slope is -- that is exactly the quantity W9's two-feature version
    could not express.
    """
    straight = (
        series("VIX.CLOSE", [20.0] * 30)
        + series("VIX9D.CLOSE", [22.0] * 30)
        + series("VIX3M.CLOSE", [200.0 / 11.0] * 30)
    )
    got = build_cboe_features(straight, prediction_after(29))
    assert got["term_structure_curvature"] is not None
    assert abs(got["term_structure_curvature"]) < 1e-12


def test_regime_is_a_percentile_in_unit_interval() -> None:
    rising = series("VIX.CLOSE", [10.0 + i * 0.05 for i in range(300)])
    got = build_cboe_features(rising, prediction_after(299))
    regime = got["volatility_regime"]
    assert regime is not None
    assert 0.0 <= regime <= 1.0
    assert regime > 0.99, "a monotonically rising series peaks at its own high"


def test_change_5d_compares_against_the_value_five_days_back() -> None:
    values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0]
    got = build_cboe_features(series("VIX.CLOSE", values), prediction_after(6))
    assert got["vix_change_5d"] is not None
    assert abs(got["vix_change_5d"] - (20.0 / 11.0 - 1.0)) < 1e-12
