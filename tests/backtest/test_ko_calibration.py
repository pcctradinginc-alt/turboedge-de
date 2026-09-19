from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from turboedge.backtest.ko_calibration import (
    KoCalibrationMethodResult,
    KoCalibrationMetrics,
    _calibration_improves,
    _decide_promotion,
    _tail_risk_ok,
    run_ko_calibration,
)
from turboedge.backtest.purged_cv import PurgedWalkForwardSplit
from turboedge.simulation.ko_calibration import KoCalibrationObservation
from turboedge.storage.schemas import Direction

_EPOCH = datetime(2020, 1, 1, tzinfo=UTC)


def _obs(
    t0: int,
    *,
    underlying_id: str = "TEST",
    horizon_days: int = 5,
    direction: Direction = Direction.LONG,
    sigma_k: float = 1.5,
    regime_bucket: str = "mid_vol",
    p_ko_raw: float = 0.1,
    realized_ko: bool = False,
) -> KoCalibrationObservation:
    return KoCalibrationObservation(
        underlying_id=underlying_id,
        t0_index=t0,
        prediction_time=_EPOCH + timedelta(days=t0),
        horizon_days=horizon_days,
        direction=direction,
        sigma_k=sigma_k,
        regime_bucket=regime_bucket,
        p_ko_raw=p_ko_raw,
        realized_ko=realized_ko,
    )


def _metrics(
    *, n: int = 1000, brier: float, ece: float, ace: float, mse: float
) -> KoCalibrationMetrics:
    return KoCalibrationMetrics(
        n=n,
        brier=brier,
        calibration_intercept=0.0,
        calibration_slope=1.0,
        ece=ece,
        mean_signed_error=mse,
        absolute_calibration_error=ace,
    )


def _method_result(
    method: str, overall: KoCalibrationMetrics, by_sigma_bucket: dict[float, KoCalibrationMetrics]
) -> KoCalibrationMethodResult:
    return KoCalibrationMethodResult(
        method=method,
        overall=overall,
        by_horizon={},
        by_direction={},
        by_sigma_bucket=by_sigma_bucket,
        by_underlying={},
        by_regime={},
        predictions=[],
    )


# -- walk-forward wiring (folds don't overlap, embargo respected) ------------


def test_run_ko_calibration_walk_forward_folds_match_purged_split() -> None:
    """The module's own fold wiring must reproduce exactly the test-set
    t0_index values a directly-constructed PurgedWalkForwardSplit with the
    same horizon/embargo/min_train/step would produce -- i.e. no
    observation evaluated twice, none skipped, no train/test overlap or
    embargo violation introduced by this module's own bookkeeping."""
    n = 80
    horizon = 5
    observations = [
        _obs(t0, p_ko_raw=0.05 + 0.001 * t0, realized_ko=(t0 % 7 == 0)) for t0 in range(n)
    ]
    result = run_ko_calibration(
        observations, min_train=30, step=10, candidate_methods=("identity",)
    )

    t0_arr = np.arange(n, dtype=np.int64)
    t1_arr = t0_arr + horizon
    splitter = PurgedWalkForwardSplit(horizon=horizon, embargo=horizon, min_train=30, step=10)
    expected_t0: set[int] = set()
    for _train_idx, test_idx in splitter.split(t0_arr, t1_arr):
        expected_t0.update(test_idx.tolist())

    got_t0 = [(p.prediction_time - _EPOCH).days for p in result.raw.predictions]
    assert sorted(got_t0) == sorted(expected_t0)
    # No observation is evaluated twice.
    assert len(got_t0) == len(set(got_t0))


def test_run_ko_calibration_rejects_empty_observations() -> None:
    with pytest.raises(ValueError):
        run_ko_calibration([], min_train=10, step=5)


# -- promotion rule ------------------------------------------------------------


def test_identity_is_never_promoted() -> None:
    observations = [
        _obs(t0, p_ko_raw=0.05 + 0.002 * t0, realized_ko=(t0 % 5 == 0)) for t0 in range(120)
    ]
    result = run_ko_calibration(
        observations, min_train=40, step=10, candidate_methods=("identity",)
    )
    assert result.promoted_method is None
    assert result.promoted_method != "identity"


def test_candidate_better_ece_but_worse_tail_risk_is_rejected() -> None:
    raw = _method_result(
        "raw",
        overall=_metrics(brier=0.10, ece=0.05, ace=0.05, mse=-0.02),
        by_sigma_bucket={
            1.5: _metrics(brier=0.10, ece=0.05, ace=0.05, mse=-0.02),
            2.0: _metrics(brier=0.10, ece=0.05, ace=0.05, mse=-0.02),
        },
    )
    # Better overall calibration (lower brier/ece/ace)...
    candidate = _method_result(
        "isotonic",
        overall=_metrics(brier=0.09, ece=0.03, ace=0.03, mse=0.0),
        # ...but at the trading-relevant sigma buckets it flips from
        # conservative (-0.02) to materially under-predicting (+0.05),
        # crossing the safety tolerance.
        by_sigma_bucket={
            1.5: _metrics(brier=0.09, ece=0.03, ace=0.03, mse=0.05),
            2.0: _metrics(brier=0.09, ece=0.03, ace=0.03, mse=0.05),
        },
    )
    assert _calibration_improves(raw, candidate) is True
    assert _tail_risk_ok(raw, candidate) is False

    promoted, reason = _decide_promotion(raw, {"identity": raw, "isotonic": candidate})
    assert promoted is None
    assert "isotonic" in reason
    assert "tail-risk" in reason or "tail" in reason


def test_candidate_worse_brier_is_rejected_even_with_better_ece() -> None:
    raw = _method_result(
        "raw",
        overall=_metrics(brier=0.10, ece=0.05, ace=0.05, mse=-0.02),
        by_sigma_bucket={
            1.5: _metrics(brier=0.10, ece=0.05, ace=0.05, mse=-0.02),
            2.0: _metrics(brier=0.10, ece=0.05, ace=0.05, mse=-0.02),
        },
    )
    candidate = _method_result(
        "platt",
        overall=_metrics(brier=0.11, ece=0.02, ace=0.02, mse=0.0),
        by_sigma_bucket={
            1.5: _metrics(brier=0.11, ece=0.02, ace=0.02, mse=-0.01),
            2.0: _metrics(brier=0.11, ece=0.02, ace=0.02, mse=-0.01),
        },
    )
    assert _calibration_improves(raw, candidate) is False
    promoted, reason = _decide_promotion(raw, {"identity": raw, "platt": candidate})
    assert promoted is None
    assert "platt" in reason


def test_candidate_that_improves_everything_is_promoted() -> None:
    raw = _method_result(
        "raw",
        overall=_metrics(brier=0.10, ece=0.05, ace=0.05, mse=-0.02),
        by_sigma_bucket={
            1.5: _metrics(brier=0.10, ece=0.05, ace=0.05, mse=-0.02),
            2.0: _metrics(brier=0.10, ece=0.05, ace=0.05, mse=-0.02),
        },
    )
    candidate = _method_result(
        "isotonic",
        overall=_metrics(brier=0.09, ece=0.02, ace=0.02, mse=-0.005),
        by_sigma_bucket={
            1.5: _metrics(brier=0.09, ece=0.02, ace=0.02, mse=-0.01),
            2.0: _metrics(brier=0.09, ece=0.02, ace=0.02, mse=-0.01),
        },
    )
    promoted, reason = _decide_promotion(raw, {"identity": raw, "isotonic": candidate})
    assert promoted == "isotonic"
    assert "isotonic" in reason


def test_decide_promotion_picks_lowest_absolute_calibration_error_among_passing() -> None:
    raw = _method_result(
        "raw",
        overall=_metrics(brier=0.10, ece=0.05, ace=0.05, mse=-0.02),
        by_sigma_bucket={
            1.5: _metrics(brier=0.10, ece=0.05, ace=0.05, mse=-0.02),
            2.0: _metrics(brier=0.10, ece=0.05, ace=0.05, mse=-0.02),
        },
    )
    isotonic = _method_result(
        "isotonic",
        overall=_metrics(brier=0.09, ece=0.03, ace=0.03, mse=-0.005),
        by_sigma_bucket={
            1.5: _metrics(brier=0.09, ece=0.03, ace=0.03, mse=-0.01),
            2.0: _metrics(brier=0.09, ece=0.03, ace=0.03, mse=-0.01),
        },
    )
    platt = _method_result(
        "platt",
        overall=_metrics(brier=0.09, ece=0.02, ace=0.015, mse=-0.005),
        by_sigma_bucket={
            1.5: _metrics(brier=0.09, ece=0.02, ace=0.015, mse=-0.01),
            2.0: _metrics(brier=0.09, ece=0.02, ace=0.015, mse=-0.01),
        },
    )
    promoted, _reason = _decide_promotion(
        raw, {"identity": raw, "isotonic": isotonic, "platt": platt}
    )
    assert promoted == "platt"  # lower absolute_calibration_error (0.015 < 0.03)


def test_tail_risk_ok_skips_missing_buckets() -> None:
    raw = _method_result(
        "raw", overall=_metrics(brier=0.1, ece=0.05, ace=0.05, mse=-0.02), by_sigma_bucket={}
    )
    candidate = _method_result(
        "isotonic",
        overall=_metrics(brier=0.09, ece=0.02, ace=0.02, mse=0.0),
        by_sigma_bucket={},
    )
    # Neither has any sigma-bucket data -> nothing to disqualify on.
    assert _tail_risk_ok(raw, candidate) is True
