"""Regression: missing CFTC observations must never shift other weeks."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

from turboedge.features.cftc_positioning import build_cftc_features, feature_names_for
from turboedge.storage.schemas import ExternalObservation


def test_missing_week_joins_by_observation_time() -> None:
    start = datetime(2026, 1, 5, tzinfo=UTC)
    rows = []
    for week, net, oi in [(0, 100, 1000), (1, None, 2000), (2, 300, 3000)]:
        moment = start + timedelta(weeks=week)
        for series, value in [("OPEN_INTEREST", oi), ("DEALER_NET", net)]:
            if value is not None:
                rows.append(
                    cast(
                        ExternalObservation,
                        SimpleNamespace(
                            series_id=f"SPX.{series}",
                            value=value,
                            observation_time=moment,
                            available_at=moment,
                        ),
                    )
                )
    result = build_cftc_features(rows, start + timedelta(weeks=3), market="SPX")
    assert result["cftc_spx_dealer_share_of_oi"] == 0.1
    assert result["cftc_spx_dealer_change_1w"] is None


def test_newer_unmatched_report_is_not_current() -> None:
    start = datetime(2026, 1, 5, tzinfo=UTC)
    rows = [
        cast(
            ExternalObservation,
            SimpleNamespace(
                series_id=f"SPX.{series}",
                value=value,
                observation_time=moment,
                available_at=moment,
            ),
        )
        for series, value, moment in [
            ("OPEN_INTEREST", 1000, start),
            ("DEALER_NET", 100, start),
            ("DEALER_NET", 200, start + timedelta(days=7)),
        ]
    ]
    result = build_cftc_features(rows, start + timedelta(days=8), market="SPX")
    assert result["cftc_spx_dealer_share_of_oi"] is None


def test_canonical_features_exclude_duplicate_but_legacy_is_explicit() -> None:
    cutoff = datetime(2026, 1, 5, tzinfo=UTC)
    canonical = build_cftc_features([], cutoff, market="SPX")
    legacy = build_cftc_features([], cutoff, market="SPX", include_legacy_alias=True)
    assert set(canonical) == set(feature_names_for("SPX"))
    assert len(canonical) == 18
    assert len(legacy) == 21
    assert set(legacy) == set(feature_names_for("SPX", include_legacy_alias=True))
    assert all(not name.endswith("_velocity") for name in canonical)
