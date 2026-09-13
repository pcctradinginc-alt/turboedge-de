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
from turboedge.models.forecast import ForecastModel
from turboedge.storage.schemas import UnderlyingBar


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
    oos_predictions: list[tuple[datetime, float, float]]  # (t, p_up, realized_return)


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
            )
        )
    return results
