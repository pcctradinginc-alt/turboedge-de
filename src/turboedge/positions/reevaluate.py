"""Daily re-evaluation of open manual positions (Contract v3 Abschnitt D).

For each open :class:`~turboedge.storage.schemas.ManualPosition`: refresh the
quote, refresh the underlying's forecast, and derive a status
(``HOLD``/``REDUCE``/``EXIT``/``INVALIDATED``) with reasons -- never an
executed action (CLAUDE.md rule 1/2/3: research system only). Persisted as an
append-only :class:`~turboedge.storage.schemas.PositionEvaluation` row per
``(position_id, as_of)``.

Performance note (this runs once per day per open position, not once per
scan): a full Monte-Carlo path re-simulation per position (the same cost as
the main scan's EV pipeline) was judged unnecessary here. Instead this module
uses two cheaper, documented, analytic approximations for its "remaining
horizon" figures --

- ``remaining_lcb_ev``: ``forecast.mean - z * forecast.uncertainty -
  round_trip_spread`` (a simple LCB analogue, not the shrinkage/pessimistic-
  scenario/Monte-Carlo-noise combination ``ranking/lcb.py`` uses for fresh
  candidates).
- ``remaining_p_ko``: :func:`simulation.barrier.brownian_bridge_hit_probability`
  (the analytic first-passage control value ``simulation/barrier.py`` itself
  documents as a sanity check for the full Monte Carlo P(KO), used here
  directly since a lightweight daily re-check does not need the same
  precision as an initial trade proposal).

Both are clearly weaker than the full EV pipeline and are labeled as such in
every rendered email/reason -- this module never claims Monte-Carlo-grade
precision for a metric it computed analytically.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict, Field

from turboedge.config import TurboEdgeConfig, config_hash
from turboedge.features.product import ewma_volatility
from turboedge.models.ensemble import combine_forecasts
from turboedge.models.forecast import HorizonForecast
from turboedge.pipeline.scan import PriceSource, default_forecast_models
from turboedge.provenance import git_commit
from turboedge.simulation.barrier import brownian_bridge_hit_probability
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Direction,
    ManualPosition,
    PositionEvaluation,
    PositionEvaluationStatus,
    PositionStatus,
    UnderlyingBar,
)

logger = structlog.get_logger(__name__)


class ReevaluateConfig(BaseModel):
    """This module's own config (Build Contract v2 rule: every module owns
    its own ``XxxConfig``; wired into ``config.py``/YAML by a future
    integration pass if it turns out this needs tuning -- not part of
    Contract v3's explicitly listed sections, so no YAML file is added for
    it in this wave)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    z: float = Field(default=1.645, gt=0.0)
    reduce_lcb_threshold: float = Field(default=0.0)
    exit_lcb_threshold: float = Field(default=-0.03)
    reduce_p_ko_threshold: float = Field(default=0.35, gt=0.0, lt=1.0)
    exit_p_ko_threshold: float = Field(default=0.6, gt=0.0, lt=1.0)
    max_quote_age_days: int = Field(default=5, gt=0)


@dataclass(frozen=True)
class ReevaluationResult:
    evaluation: PositionEvaluation
    status_changed: bool
    materially_changed: bool


def _round_trip_spread(bid: float | None, ask: float | None) -> float:
    if bid is None or ask is None or ask <= 0:
        return 0.0
    return (ask - bid) / ask


def reevaluate_position(
    *,
    cfg: TurboEdgeConfig,
    store: Store,
    position: ManualPosition,
    price_adapter: PriceSource,
    as_of: datetime,
    config: ReevaluateConfig | None = None,
) -> ReevaluationResult:
    """Re-evaluate one open position as of ``as_of``.

    Never raises for ordinary data problems -- a missing instrument, a stale
    or missing quote, or insufficient underlying history all resolve to
    ``INVALIDATED`` with ``data_quality_ok=False`` and an explanatory
    reason, never a crash (this is meant to run unattended, daily, over
    every open position).
    """
    rcfg = config if config is not None else ReevaluateConfig()
    config_hash_value = config_hash(cfg)
    git_commit_value = git_commit()
    reasons: list[str] = []

    if position.isin is None:
        return _invalidated(position, as_of, "no_isin_on_file", config_hash_value, git_commit_value)

    instrument = store.get_instrument(position.isin)
    if instrument is None or instrument.underlying_id is None:
        return _invalidated(
            position,
            as_of,
            "instrument_master_data_unavailable",
            config_hash_value,
            git_commit_value,
            isin=position.isin,
        )
    underlying_id = instrument.underlying_id

    snapshot = store.latest_product_snapshot_at_or_before(position.isin, as_of)
    if snapshot is None:
        return _invalidated(
            position,
            as_of,
            "no_recent_quote",
            config_hash_value,
            git_commit_value,
            isin=position.isin,
            underlying_id=underlying_id,
        )
    quote_age_days = (as_of - (snapshot.quote_timestamp or snapshot.observation_time)).days
    if quote_age_days > rcfg.max_quote_age_days:
        reasons.append(f"quote_stale:{quote_age_days}d")

    if snapshot.knocked_out:
        return _terminal(
            position,
            as_of,
            PositionEvaluationStatus.EXIT,
            ["knocked_out"],
            snapshot.bid,
            snapshot.quote_timestamp,
            underlying_id,
            config_hash_value,
            git_commit_value,
        )

    try:
        fetched_bars = price_adapter.fetch_daily_bars(underlying_id, lookback_days=400)
    except Exception as exc:
        logger.warning("reevaluate_bars_fetch_failed", underlying_id=underlying_id, error=str(exc))
        fetched_bars = store.latest_underlying_bars(underlying_id, 400)
    usable_bars = sorted((b for b in fetched_bars if b.available_at <= as_of), key=lambda b: b.ts)
    if len(usable_bars) < 30:
        reasons.append("insufficient_underlying_history_for_forecast")
        return _terminal(
            position,
            as_of,
            PositionEvaluationStatus.INVALIDATED,
            reasons,
            snapshot.bid,
            snapshot.quote_timestamp,
            underlying_id,
            config_hash_value,
            git_commit_value,
            data_quality_ok=False,
        )

    forecast = _fit_short_horizon_forecast(usable_bars, as_of, cfg)
    if forecast is None:
        reasons.append("forecast_unavailable")
        return _terminal(
            position,
            as_of,
            PositionEvaluationStatus.INVALIDATED,
            reasons,
            snapshot.bid,
            snapshot.quote_timestamp,
            underlying_id,
            config_hash_value,
            git_commit_value,
            data_quality_ok=False,
        )

    direction = instrument.direction
    round_trip = _round_trip_spread(snapshot.bid, snapshot.ask)
    directional_mean = forecast.mean if direction == Direction.LONG else -forecast.mean
    remaining_lcb_ev = directional_mean - rcfg.z * forecast.uncertainty - round_trip
    remaining_p_profit = float(
        forecast.p_up if direction == Direction.LONG else 1.0 - forecast.p_up
    )

    remaining_p_ko: float | None = None
    barrier = snapshot.knockout_barrier
    if barrier is not None:
        closes = np.array([b.close for b in usable_bars], dtype=np.float64)
        log_returns = np.diff(np.log(closes))
        if log_returns.size >= 2:
            sigma = float(ewma_volatility(log_returns, lam=0.94)[-1])
            spot = closes[-1]
            still_safe = spot > barrier if direction == Direction.LONG else spot < barrier
            if sigma > 0 and still_safe:
                s1 = spot * math.exp(directional_mean)
                remaining_p_ko = brownian_bridge_hit_probability(
                    spot, s1, barrier, sigma, direction
                )
            elif sigma > 0:
                # Barrier already breached in the underlying's own bar
                # history but the source has not (yet) flagged
                # knocked_out=True -- treat as certain (CLAUDE.md rule 17,
                # never optimistic).
                remaining_p_ko = 1.0

    status = PositionEvaluationStatus.HOLD
    if remaining_p_ko is not None and remaining_p_ko >= rcfg.exit_p_ko_threshold:
        status = PositionEvaluationStatus.EXIT
        reasons.append(f"remaining_p_ko_high:{remaining_p_ko:.3f}")
    elif remaining_lcb_ev <= rcfg.exit_lcb_threshold:
        status = PositionEvaluationStatus.EXIT
        reasons.append(f"remaining_lcb_ev_negative:{remaining_lcb_ev:.4f}")
    elif (remaining_p_ko is not None and remaining_p_ko >= rcfg.reduce_p_ko_threshold) or (
        remaining_lcb_ev <= rcfg.reduce_lcb_threshold
    ):
        status = PositionEvaluationStatus.REDUCE
        reasons.append(f"remaining_lcb_ev_marginal:{remaining_lcb_ev:.4f}")
    else:
        reasons.append(f"remaining_lcb_ev_positive:{remaining_lcb_ev:.4f}")
    reasons.append(
        "p_ko_analytic_approximation_see_docs"
        if remaining_p_ko is not None
        else "p_ko_unavailable_no_barrier_on_file"
    )

    unrealized_return = (
        snapshot.bid / position.entry_price - 1.0 if snapshot.bid is not None else None
    )

    evaluation = PositionEvaluation(
        position_id=position.position_id,
        as_of=as_of,
        wkn=position.wkn,
        isin=position.isin,
        underlying_id=underlying_id,
        status=status,
        reasons=reasons,
        current_bid=snapshot.bid,
        quote_timestamp=snapshot.quote_timestamp,
        remaining_horizon_days=forecast.horizon_days,
        remaining_lcb_ev=remaining_lcb_ev,
        remaining_p_ko=remaining_p_ko,
        remaining_p_profit=remaining_p_profit,
        unrealized_return=unrealized_return,
        data_quality_ok=True,
        config_hash=config_hash_value,
        git_commit=git_commit_value,
    )
    return _finalize(store, evaluation)


def _fit_short_horizon_forecast(
    bars: Sequence[UnderlyingBar], as_of: datetime, cfg: TurboEdgeConfig
) -> HorizonForecast | None:
    """Fit the default model ensemble and return the ensemble forecast at
    the *shortest* configured horizon (a position re-check cares about the
    nearest-term outlook, not the full horizon ladder)."""
    horizon = min(cfg.forecast.horizons)
    forecasts: list[HorizonForecast] = []
    for model in default_forecast_models():
        try:
            model.fit(bars, as_of)
            forecasts.extend(model.predict(bars, as_of, horizons=[horizon]))
        except Exception as exc:
            logger.debug(
                "reevaluate_model_fit_failed",
                model_id=getattr(model, "model_id", "?"),
                error=str(exc),
            )
            continue
    if not forecasts:
        return None
    weights = {f.model_id: 1.0 for f in forecasts}
    return combine_forecasts(forecasts, weights)


def _invalidated(
    position: ManualPosition,
    as_of: datetime,
    reason: str,
    config_hash_value: str,
    git_commit_value: str | None,
    *,
    isin: str | None = None,
    underlying_id: str | None = None,
) -> ReevaluationResult:
    return _terminal(
        position,
        as_of,
        PositionEvaluationStatus.INVALIDATED,
        [reason],
        None,
        None,
        underlying_id,
        config_hash_value,
        git_commit_value,
        data_quality_ok=False,
        isin=isin,
    )


def _terminal(
    position: ManualPosition,
    as_of: datetime,
    status: PositionEvaluationStatus,
    reasons: list[str],
    current_bid: float | None,
    quote_timestamp: datetime | None,
    underlying_id: str | None,
    config_hash_value: str,
    git_commit_value: str | None,
    *,
    data_quality_ok: bool = True,
    isin: str | None = None,
) -> ReevaluationResult:
    evaluation = PositionEvaluation(
        position_id=position.position_id,
        as_of=as_of,
        wkn=position.wkn,
        isin=isin if isin is not None else position.isin,
        underlying_id=underlying_id,
        status=status,
        reasons=reasons,
        current_bid=current_bid,
        quote_timestamp=quote_timestamp,
        remaining_horizon_days=None,
        remaining_lcb_ev=None,
        remaining_p_ko=None,
        remaining_p_profit=None,
        unrealized_return=(
            current_bid / position.entry_price - 1.0 if current_bid is not None else None
        ),
        data_quality_ok=data_quality_ok,
        config_hash=config_hash_value,
        git_commit=git_commit_value,
    )
    return ReevaluationResult(evaluation=evaluation, status_changed=True, materially_changed=True)


def _finalize(store: Store, evaluation: PositionEvaluation) -> ReevaluationResult:
    previous = store.latest_position_evaluation(evaluation.position_id)
    status_changed = previous is None or previous.status != evaluation.status
    materially_changed = status_changed or _material_change(previous, evaluation)
    return ReevaluationResult(
        evaluation=evaluation,
        status_changed=status_changed,
        materially_changed=materially_changed,
    )


def _material_change(previous: PositionEvaluation | None, current: PositionEvaluation) -> bool:
    if previous is None:
        return True
    for field_name in ("remaining_lcb_ev", "remaining_p_ko"):
        prev_val = getattr(previous, field_name)
        cur_val = getattr(current, field_name)
        if prev_val is None or cur_val is None:
            continue
        if abs(prev_val - cur_val) > 0.02:  # 2 percentage points
            return True
    return False


def reevaluate_open_positions(
    *,
    cfg: TurboEdgeConfig,
    store: Store,
    price_adapter: PriceSource,
    as_of: datetime | None = None,
    config: ReevaluateConfig | None = None,
) -> list[ReevaluationResult]:
    """Re-evaluate every currently OPEN manual position (``turboedge
    position reevaluate``). Persists every evaluation (append-only,
    ``position_evaluations``); returns the full list so the CLI can decide
    which ones warrant a mail (``status_changed`` or ``materially_changed``)."""
    now = as_of if as_of is not None else datetime.now(UTC)
    open_positions = store.list_positions(status=PositionStatus.OPEN)
    results: list[ReevaluationResult] = []
    for position in open_positions:
        result = reevaluate_position(
            cfg=cfg,
            store=store,
            position=position,
            price_adapter=price_adapter,
            as_of=now,
            config=config,
        )
        store.append_position_evaluation(result.evaluation)
        results.append(result)
    return results


__all__ = [
    "ReevaluateConfig",
    "ReevaluationResult",
    "reevaluate_open_positions",
    "reevaluate_position",
]
