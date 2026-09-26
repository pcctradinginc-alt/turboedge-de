"""Human-readable rendering of a shadow meta decision (Phase 1, M8).

Plain text, no dependencies, no colour. Every number shown is one the
controller computed; nothing here derives, rounds away or editorialises a
value. The point of the report is that a reader can disagree with the
reasoning, which requires seeing the inputs rather than the verdict alone.
"""

from __future__ import annotations

from collections.abc import Sequence

from turboedge.meta.schemas import MetaDecision


def render_meta_decision(decision: MetaDecision) -> str:
    """One decision as a readable block."""
    c = decision.confidence
    lines = [
        f"META DECISION: {decision.decision.value}"
        + ("  [SHADOW -- changes nothing]" if decision.shadow_mode else ""),
        "",
        f"  {decision.underlying_id} h={decision.horizon_days}d  "
        f"as of {decision.prediction_time.isoformat()}",
        f"  regime: {decision.volatility_regime} / {decision.trend_regime}  "
        f"({decision.regime_observation_count} comparable historical days)",
        f"  confidence: {decision.final_confidence:.2f}   abstain score: {c.abstain_score:.2f}",
        "",
        "  Uncertainty:",
        f"    epistemic (best model untrusted)  {c.epistemic_uncertainty:.2f}",
        f"    data quality / staleness          {c.data_uncertainty:.2f}",
        f"    regime unfamiliarity              {c.regime_uncertainty:.2f}",
        f"    model disagreement                {c.model_disagreement:.2f}",
        f"    calibration unmeasured            {c.calibration_uncertainty:.2f}",
        f"    product terms unverified          {c.product_data_uncertainty:.2f}",
        "",
        f"  Models: {len(decision.selected_models)}/{len(decision.available_models)} selected",
    ]
    for trust in decision.model_trust:
        weight = decision.model_weights.get(trust.model_id, 0.0)
        missing = (
            f"  [unmeasured: {', '.join(trust.missing_factors)}]" if trust.missing_factors else ""
        )
        lines.append(
            f"    {trust.model_id:24} trust {trust.trust_score:.3f}  weight {weight:.3f}{missing}"
        )
    lines += ["", "  Reasons:"]
    lines += [f"    - {r}" for r in decision.reasons]
    return "\n".join(lines)


def render_summary(decisions: Sequence[MetaDecision]) -> str:
    """Counts across a scan.

    The abstention rate is the number worth watching in Phase 1: it says how
    often the system admits it cannot read the situation, which is the claim
    this layer exists to make measurable -- not a performance figure.
    """
    if not decisions:
        return "META: no decisions recorded."
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.decision.value] = counts.get(d.decision.value, 0) + 1
    total = len(decisions)
    abstained = counts.get("ABSTAIN", 0)
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    return (
        f"META (shadow): {total} decision(s)  " + "  ".join(parts) + f"\n"
        f"  abstention rate {abstained / total:.0%}  "
        f"mean confidence {sum(d.final_confidence for d in decisions) / total:.2f}"
    )


__all__ = ["render_meta_decision", "render_summary"]
