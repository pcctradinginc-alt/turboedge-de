"""Walk-forward out-of-sample evaluation of a ``ForecastModel``.

Formula reference: Master Spec §28 ("Walk-Forward Methodik"), §9.3, §38.
CLAUDE.md rules 6-8: no random splits; purged CV + embargo; refit on an
expanding window, never peeking past ``as_of``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from turboedge.backtest import metrics as bt_metrics
from turboedge.backtest.purged_cv import PurgedWalkForwardSplit
from turboedge.backtest.significance import probabilistic_sharpe_ratio
from turboedge.models.forecast import ForecastModel, HorizonForecast
from turboedge.models.quantile import QUANTILE_LEVELS
from turboedge.storage.schemas import UnderlyingBar

#: Nominal coverage of the two central prediction intervals every
#: HorizonForecast's quantiles imply: q05/q95 (90%) and q25/q75 (50%).
_INTERVAL_90 = ("q05", "q95")
_INTERVAL_50 = ("q25", "q75")


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    model_id: str
    horizon_days: int
    n_folds: int
    brier: float
    log_loss: float
    ece: float
    mean_oos_return_when_long: float
    mean_oos_return_when_short: float
    hit_rate: float
    psr: float
    oos_predictions: list[tuple[datetime, float, float]]  # (t, p_up, realized_return) -- diagnostic
    # Phase D (docs/measured_results.md): the full predictive distribution
    # collected per out-of-sample prediction, instead of throwing everything
    # but p_up away before scoring (the "Bewertungsziel auf die Verteilung
    # umstellen" problem statement). Same order/length as ``oos_predictions``.
    oos_forecasts: list[tuple[datetime, HorizonForecast, float]]  # (t, forecast, realized_return)
    crps: float  # mean quantile-based CRPS approximation (bt_metrics.mean_crps_from_quantiles)
    pinball_by_quantile: dict[str, float]  # mean pinball loss per "q05".."q95" key
    coverage_90: float  # empirical coverage of the [q05, q95] interval (nominal 0.90)
    coverage_50: float  # empirical coverage of the [q25, q75] interval (nominal 0.50)


def walk_forward_evaluate(
    model_factory: Callable[[], ForecastModel],
    bars: Sequence[UnderlyingBar],
    horizons: Sequence[int],
    *,
    min_train: int,
    step: int,
    embargo: int,
) -> list[WalkForwardResult]:
    """Expanding-window walk-forward OOS evaluation, refitting ``model_factory()`` every ``step``.

    For each horizon independently: an expanding-window
    :class:`~turboedge.backtest.purged_cv.PurgedWalkForwardSplit` (purged +
    embargoed by ``horizon``) generates train/test folds over the bar axis;
    for each fold, a fresh model is fit on bars up to (and including) the
    fold's last training bar's ``available_at`` (never later -- CLAUDE.md
    rule 5), then used to predict every test bar in that fold, and the
    realized ``h``-day-forward log return is compared against that
    prediction. Predictions/labels from every fold are pooled per horizon
    into one :class:`WalkForwardResult`.
    """
    sorted_bars = sorted(bars, key=lambda b: b.ts)
    n = len(sorted_bars)
    closes = np.array([b.close for b in sorted_bars], dtype=np.float64)

    results: list[WalkForwardResult] = []
    for h in horizons:
        splitter = PurgedWalkForwardSplit(
            min_train=min_train, step=step, horizon=h, embargo=embargo
        )
        t0 = np.arange(n, dtype=np.int64)
        t1 = t0 + h
        oos: list[tuple[datetime, float, float]] = []
        oos_forecasts: list[tuple[datetime, HorizonForecast, float]] = []
        n_folds = 0
        model_id = ""

        for train_idx, test_idx in splitter.split(t0, t1):
            train_end = int(train_idx.max())
            as_of = sorted_bars[train_end].available_at
            model = model_factory()
            model.fit(sorted_bars, as_of)
            model_id = model.model_id
            n_folds += 1
            for i in test_idx.tolist():
                if i + h >= n:
                    continue
                bar = sorted_bars[i]
                realized_return = float(np.log(closes[i + h] / closes[i]))
                forecast = model.predict(sorted_bars, bar.available_at, horizons=[h])[0]
                oos.append((bar.ts, forecast.p_up, realized_return))
                oos_forecasts.append((bar.ts, forecast, realized_return))

        if not oos:
            raise ValueError(
                f"walk_forward_evaluate produced no out-of-sample predictions for horizon {h}d "
                f"(min_train={min_train}, step={step}, embargo={embargo}, n_bars={n}) -- "
                "check that there is enough history"
            )

        p_ups = np.array([o[1] for o in oos], dtype=np.float64)
        rets = np.array([o[2] for o in oos], dtype=np.float64)
        labels = (rets > 0.0).astype(np.float64)

        long_mask = p_ups > 0.5
        short_mask = p_ups < 0.5
        mean_long = float(np.mean(rets[long_mask])) if np.any(long_mask) else float("nan")
        mean_short = float(np.mean(-rets[short_mask])) if np.any(short_mask) else float("nan")

        signal_returns = np.where(long_mask, rets, np.where(short_mask, -rets, 0.0))

        # Phase D: score the full predictive distribution (CRPS/pinball/coverage)
        # against the same realized returns brier/log_loss/ece already use above --
        # p_up/brier are left completely untouched, this is purely additive.
        quantiles_by_tau = [
            {QUANTILE_LEVELS[key]: fc.quantiles[key] for key in QUANTILE_LEVELS}
            for _t, fc, _r in oos_forecasts
        ]
        crps = bt_metrics.mean_crps_from_quantiles(rets, quantiles_by_tau)
        pinball_by_tau = bt_metrics.pinball_loss_by_level(rets, quantiles_by_tau)
        pinball_by_quantile = {
            key: pinball_by_tau[tau]
            for key, tau in QUANTILE_LEVELS.items()
            if tau in pinball_by_tau
        }
        lower90 = np.array([fc.quantiles[_INTERVAL_90[0]] for _t, fc, _r in oos_forecasts])
        upper90 = np.array([fc.quantiles[_INTERVAL_90[1]] for _t, fc, _r in oos_forecasts])
        coverage_90 = bt_metrics.interval_coverage(rets, lower90, upper90)
        lower50 = np.array([fc.quantiles[_INTERVAL_50[0]] for _t, fc, _r in oos_forecasts])
        upper50 = np.array([fc.quantiles[_INTERVAL_50[1]] for _t, fc, _r in oos_forecasts])
        coverage_50 = bt_metrics.interval_coverage(rets, lower50, upper50)

        results.append(
            WalkForwardResult(
                model_id=model_id,
                horizon_days=h,
                n_folds=n_folds,
                brier=bt_metrics.brier_score(p_ups, labels),
                log_loss=bt_metrics.log_loss(p_ups, labels),
                ece=bt_metrics.expected_calibration_error(p_ups, labels),
                mean_oos_return_when_long=mean_long,
                mean_oos_return_when_short=mean_short,
                hit_rate=bt_metrics.hit_rate(signal_returns),
                psr=probabilistic_sharpe_ratio(signal_returns)
                if signal_returns.size >= 3
                else float("nan"),
                oos_predictions=oos,
                oos_forecasts=oos_forecasts,
                crps=crps,
                pinball_by_quantile=pinball_by_quantile,
                coverage_90=coverage_90,
                coverage_50=coverage_50,
            )
        )
    return results
