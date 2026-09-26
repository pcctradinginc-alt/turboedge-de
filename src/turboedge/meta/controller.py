"""Shadow meta-controller (Phase 1, M5).

Runs alongside the existing pipeline and records what it *would* have
decided. It changes no model weight, no gate, no threshold -- `shadow_mode`
is true on every decision it emits, and nothing in `pipeline/scan.py` reads
its output.

That restraint is the point rather than a limitation. This layer's own
claims are untested: whether low model disagreement actually predicts better
outcomes, whether an unfamiliar regime really warrants abstention, whether
the weights below are sensible at all -- none of it is measured yet. Letting
it steer production before that evidence exists would repeat, one level up,
exactly the mistake this project spent W4 and W9 documenting: acting on a
model that has never beaten its null.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from turboedge.meta.disagreement import (
    direction_disagreement,
    forecast_dispersion,
    tail_risk_disagreement,
)
from turboedge.meta.regimes import (
    UNKNOWN_REGIME,
    classify_trend_regime,
    classify_volatility_regime,
    count_similar_history,
)
from turboedge.meta.schemas import (
    DecisionConfidence,
    MetaDecision,
    MetaDecisionKind,
    ModelTrust,
)
from turboedge.meta.trust import score_model_trust, weights_from_trust
from turboedge.meta.uncertainty import (
    ABSTAIN_THRESHOLD,
    PROCEED_CONFIDENCE_FLOOR,
    assess,
)
from turboedge.models.forecast import HorizonForecast
from turboedge.storage.schemas import (
    ModelRegistryEntry,
    UnderlyingBar,
    WalkforwardResultRecord,
)

#: A model with trust at or below this is dropped from `selected_models`.
#: Not zero: a model can be technically usable and still contribute nothing
#: but noise, and the ensemble is better off without it.
MIN_SELECTION_TRUST = 0.05


def _reasons(
    confidence: DecisionConfidence,
    trusts: Sequence[ModelTrust],
    forecasts: Sequence[HorizonForecast],
    *,
    volatility_regime: str,
    trend_regime: str,
    regime_observation_count: int,
) -> list[str]:
    """Explanations rendered from the numbers, never written freehand.

    Every line carries the value it rests on, so a reader can disagree with
    the reasoning rather than only with the verdict. An explanation that
    cannot be traced to a computed value does not belong in this list.
    """
    out: list[str] = []
    if confidence.epistemic_uncertainty >= 0.9:
        best = max((t.trust_score for t in trusts), default=0.0)
        out.append(f"no model is trusted here (best trust {best:.2f})")
    missing = sorted({f for t in trusts for f in t.missing_factors})
    if missing:
        out.append(
            f"trust rests on incomplete evidence: {', '.join(missing)} unavailable "
            f"for {sum(1 for t in trusts if t.missing_factors)}/{len(trusts)} model(s)"
        )
    if volatility_regime == UNKNOWN_REGIME or trend_regime == UNKNOWN_REGIME:
        out.append("regime not classifiable (insufficient causal price history)")
    elif confidence.regime_uncertainty >= 0.5:
        out.append(
            f"regime {volatility_regime}/{trend_regime} rarely observed "
            f"({regime_observation_count} comparable historical days)"
        )
    if confidence.model_disagreement >= 0.4 and len(forecasts) >= 2:
        out.append(
            f"models disagree (direction split {direction_disagreement(forecasts):.2f}, "
            f"tail spread {tail_risk_disagreement(forecasts):.2f} sigma, "
            f"mean spread {forecast_dispersion(forecasts):.2f} sigma)"
        )
    elif len(forecasts) >= 2 and confidence.model_disagreement <= 0.15:
        out.append(
            f"models agree (disagreement {confidence.model_disagreement:.2f}) -- "
            "note agreement is not evidence of correctness, it is untested here"
        )
    if confidence.calibration_uncertainty >= 0.9:
        out.append("calibration never measured out-of-sample for any model")
    if confidence.data_uncertainty >= 0.4:
        out.append(f"data quality degraded ({confidence.data_uncertainty:.2f})")
    if confidence.product_data_uncertainty >= 0.3:
        out.append(
            f"product terms unverified for a material share "
            f"({confidence.product_data_uncertainty:.2f})"
        )
    if not out:
        worst = max(
            (
                confidence.epistemic_uncertainty,
                confidence.data_uncertainty,
                confidence.regime_uncertainty,
                confidence.model_disagreement,
                confidence.calibration_uncertainty,
                confidence.product_data_uncertainty,
            )
        )
        out.append(f"no elevated uncertainty component (worst {worst:.2f})")
    return out


def decide(
    *,
    run_id: str,
    underlying_id: str,
    horizon_days: int,
    prediction_time: datetime,
    bars: Sequence[UnderlyingBar],
    registry: Sequence[ModelRegistryEntry],
    forecasts: Sequence[HorizonForecast],
    walkforward: Sequence[WalkforwardResultRecord],
    data_quality: float,
    stale_share: float,
    ratio_unverified_share: float,
    integrity_fail_share: float,
    drifting_families: Sequence[str] = (),
    config_hash: str = "",
    git_commit: str | None = None,
    now: datetime | None = None,
) -> MetaDecision:
    """One shadow decision. Reads only what was available at
    ``prediction_time``; never mutates anything."""
    decided_at = now if now is not None else datetime.now(UTC)

    volatility_regime = classify_volatility_regime(bars, prediction_time)
    trend_regime = classify_trend_regime(bars, prediction_time)
    regime_count = count_similar_history(bars, prediction_time, volatility_regime, trend_regime)

    forecast_by_model = {f.model_id: f for f in forecasts}
    wf_by_model = {
        w.model_id: w
        for w in walkforward
        if w.underlying_id == underlying_id and w.horizon_days == horizon_days
    }
    drifting = set(drifting_families)

    trusts = [
        score_model_trust(
            entry,
            forecast=forecast_by_model.get(entry.model_id),
            walkforward=wf_by_model.get(entry.model_id),
            regime_observation_count=regime_count,
            data_quality=data_quality,
            in_drift=entry.signal_family in drifting,
        )
        for entry in registry
    ]

    confidence = assess(
        trusts=trusts,
        forecasts=forecasts,
        data_quality=data_quality,
        stale_share=stale_share,
        regime_observation_count=regime_count,
        ratio_unverified_share=ratio_unverified_share,
        integrity_fail_share=integrity_fail_share,
    )

    selected = [t.model_id for t in trusts if t.trust_score > MIN_SELECTION_TRUST]
    weights = weights_from_trust([t for t in trusts if t.model_id in selected])
    final_confidence = float(max(0.0, 1.0 - confidence.abstain_score))

    if confidence.abstain_score >= ABSTAIN_THRESHOLD or not selected:
        decision = MetaDecisionKind.ABSTAIN
    elif final_confidence < PROCEED_CONFIDENCE_FLOOR:
        decision = MetaDecisionKind.WATCH_ONLY
    else:
        decision = MetaDecisionKind.PROCEED

    return MetaDecision(
        run_id=run_id,
        underlying_id=underlying_id,
        horizon_days=horizon_days,
        prediction_time=prediction_time,
        decided_at=decided_at,
        volatility_regime=volatility_regime,
        trend_regime=trend_regime,
        regime_observation_count=regime_count,
        available_models=[e.model_id for e in registry],
        selected_models=selected,
        model_weights=weights,
        model_trust=trusts,
        confidence=confidence,
        final_confidence=final_confidence,
        decision=decision,
        reasons=_reasons(
            confidence,
            trusts,
            forecasts,
            volatility_regime=volatility_regime,
            trend_regime=trend_regime,
            regime_observation_count=regime_count,
        ),
        shadow_mode=True,
        config_hash=config_hash,
        git_commit=git_commit,
    )


__all__ = ["MIN_SELECTION_TRUST", "decide"]
