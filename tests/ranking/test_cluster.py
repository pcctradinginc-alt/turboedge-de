from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from turboedge.ranking.cluster import (
    ClusterConfig,
    OpenClusterPosition,
    cluster_id_for,
    cluster_risk,
    cluster_risk_pass,
    compute_clusters,
)
from turboedge.storage.schemas import UnderlyingBar


def _bars_from_returns(underlying_id: str, returns: np.ndarray, start: date) -> list[UnderlyingBar]:
    bars: list[UnderlyingBar] = []
    price = 100.0
    d = start
    for r in returns:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        price = price * float(np.exp(r))
        ts = datetime(d.year, d.month, d.day, 22, 0, tzinfo=UTC)
        bars.append(
            UnderlyingBar(
                underlying_id=underlying_id,
                ts=ts,
                interval="1d",
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price,
                volume=1000.0,
                observation_time=ts,
                available_at=ts,
                retrieved_at=ts,
                source="test",
                parser_version="1",
                quality_score=1.0,
            )
        )
        d += timedelta(days=1)
    return bars


def test_compute_clusters_groups_highly_correlated_underlyings() -> None:
    rng = np.random.default_rng(42)
    n = 150
    common = rng.normal(0.0, 0.01, size=n)
    idio_a = rng.normal(0.0, 0.001, size=n)
    idio_b = rng.normal(0.0, 0.001, size=n)
    idio_c = rng.normal(0.0, 0.01, size=n)  # C: uncorrelated with A/B

    start = date(2024, 1, 2)
    bars = {
        "A": _bars_from_returns("A", common + idio_a, start),
        "B": _bars_from_returns("B", common + idio_b, start),
        "C": _bars_from_returns("C", idio_c, start),
    }
    as_of = datetime(2024, 12, 1, tzinfo=UTC)
    cfg = ClusterConfig(lookback_days=120, min_overlap_days=30, distance_threshold=0.5)

    assignments = compute_clusters(bars, as_of=as_of, cfg=cfg)

    assert assignments["A"].cluster_id == assignments["B"].cluster_id
    assert assignments["A"].cluster_id != assignments["C"].cluster_id
    assert not assignments["A"].fallback
    assert not assignments["C"].fallback


def test_compute_clusters_falls_back_to_singleton_for_insufficient_history() -> None:
    rng = np.random.default_rng(1)
    long_bars = _bars_from_returns("LONG_HIST", rng.normal(0.0, 0.01, size=150), date(2024, 1, 2))
    short_bars = _bars_from_returns("SHORT_HIST", rng.normal(0.0, 0.01, size=5), date(2024, 11, 20))

    as_of = datetime(2024, 12, 1, tzinfo=UTC)
    cfg = ClusterConfig(lookback_days=120, min_overlap_days=30)
    assignments = compute_clusters(
        {"LONG_HIST": long_bars, "SHORT_HIST": short_bars}, as_of=as_of, cfg=cfg
    )

    assert assignments["SHORT_HIST"].fallback
    assert assignments["SHORT_HIST"].cluster_id == "singleton_SHORT_HIST"


def test_compute_clusters_single_underlying_is_singleton() -> None:
    bars = _bars_from_returns(
        "ONLY", np.random.default_rng(2).normal(0.0, 0.01, size=100), date(2024, 1, 2)
    )
    assignments = compute_clusters({"ONLY": bars}, as_of=datetime(2024, 6, 1, tzinfo=UTC))
    assert assignments["ONLY"].fallback
    assert assignments["ONLY"].cluster_id == "singleton_ONLY"


def test_cluster_id_for_missing_assignment_falls_back() -> None:
    assignments = compute_clusters({}, as_of=datetime(2024, 6, 1, tzinfo=UTC))
    assert cluster_id_for(assignments, "NEW_UNDERLYING") == "singleton_NEW_UNDERLYING"


def test_cluster_risk_increases_with_position_count_and_capital() -> None:
    cfg = ClusterConfig(max_active_positions_per_cluster=2, max_capital_fraction_per_cluster=0.10)
    new_candidate = OpenClusterPosition("DAX", "corr_1", capital_fraction=0.02)

    risk_empty = cluster_risk([], new_candidate, cfg=cfg)
    open_positions = [
        OpenClusterPosition("MDAX", "corr_1", capital_fraction=0.03),
        OpenClusterPosition("SDAX", "corr_1", capital_fraction=0.02),
    ]
    risk_crowded = cluster_risk(open_positions, new_candidate, cfg=cfg)

    assert risk_crowded > risk_empty


def test_cluster_risk_ignores_other_clusters() -> None:
    cfg = ClusterConfig(max_active_positions_per_cluster=1, max_capital_fraction_per_cluster=0.05)
    new_candidate = OpenClusterPosition("DAX", "corr_1", capital_fraction=0.01)
    other_cluster_positions = [OpenClusterPosition("GOLD", "corr_2", capital_fraction=0.5)]

    risk = cluster_risk(other_cluster_positions, new_candidate, cfg=cfg)
    assert risk == pytest.approx(1.0)  # only new_candidate itself counts


def test_cluster_risk_pass_boundary() -> None:
    cfg = ClusterConfig(max_active_positions_per_cluster=5, max_capital_fraction_per_cluster=0.10)
    ok_candidate = OpenClusterPosition("DAX", "corr_1", capital_fraction=0.05)
    over_candidate = OpenClusterPosition("DAX", "corr_1", capital_fraction=0.20)

    assert cluster_risk_pass([], ok_candidate, cfg=cfg) is True
    assert cluster_risk_pass([], over_candidate, cfg=cfg) is False
