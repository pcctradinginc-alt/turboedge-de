"""Human-readable rendering of a shadow meta decision (Phase 1, M8).

Plain text, no dependencies, no colour. Every number shown is one the
controller computed; nothing here derives, rounds away or editorialises a
value. The point of the report is that a reader can disagree with the
reasoning, which requires seeing the inputs rather than the verdict alone.
"""

from __future__ import annotations

from collections.abc import Sequence

from turboedge.meta.research_opportunity import (
    EstimateBasis,
    ResearchStatus,
    StoredOpportunity,
)
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


def render_research_queue(
    entries: Sequence[StoredOpportunity],
    *,
    limit: int | None = None,
) -> str:
    """The ranked research queue as readable text (Phase 2, §13).

    Two things are shown that a bare ranking would hide, and they are the
    reason this renderer exists rather than a sorted list of ids.

    `evidence` is the share of ranking inputs that could actually be
    measured. Most entries sit low, because the inputs to an *unrun*
    experiment are judgements by construction -- a queue that presented
    those scores without saying so would read as measurement.

    `?` marks the inputs that are unknown. They are penalties in the score,
    not gaps papered over with a neutral value, so an entry can rank low
    purely because nobody knows enough about it yet -- which is useful
    information and should be visible.
    """
    if not entries:
        return "RESEARCH QUEUE: empty."

    shown = entries if limit is None else entries[:limit]
    lines = [
        f"RESEARCH QUEUE ({len(entries)} open, showing {len(shown)})"
        "   [priorities only -- implementation needs human approval]",
        "",
        f"  {'#':>2}  {'hypothesis':28} {'status':9} {'score':>7}  {'evidence':>8}  family",
    ]
    for rank, entry in enumerate(shown, start=1):
        o = entry.opportunity
        p = entry.priority
        score = "  --   " if p is None else f"{p.score:7.4f}"
        evidence = "    --  " if p is None else f"{p.evidence_completeness:7.0%} "
        lines.append(
            f"  {rank:>2}. {o.hypothesis_id:28} {o.status.value:9} {score}  {evidence}  "
            f"{o.information_family.value}"
        )

    lines += ["", "  Detail:"]
    for entry in shown:
        o = entry.opportunity
        p = entry.priority
        lines.append(f"    {o.hypothesis_id}  ({o.information_family.value})")
        lines.append(f"      {o.description}")
        declared = sorted(
            name for name in _RANKING_INPUTS if getattr(o, name).basis is EstimateBasis.DECLARED
        )
        if p is None:
            lines.append("      not scored yet")
        else:
            lines.append(
                f"      score {p.score:.4f}  (base {p.base_score:.4f})   "
                f"evidence {p.evidence_completeness:.0%}"
            )
            if p.unknown_inputs:
                lines.append(f"      ? unknown: {', '.join(sorted(p.unknown_inputs))}")
            for reason in p.reasons:
                lines.append(f"      - {reason}")
        if declared:
            lines.append(f"      declared (judgement, not measured): {', '.join(declared)}")
        if o.status is not ResearchStatus.PROPOSED:
            lines.append(
                f"      {o.status.value} by {o.approved_by or 'unknown'}"
                + (f" -- {o.status_note}" if o.status_note else "")
            )
        lines.append("")
    return "\n".join(lines).rstrip()


#: The ranking-input field names, for the provenance line above. Kept beside
#: the renderer rather than imported from storage so that a new input added to
#: the schema shows up as absent here instead of being quietly omitted.
_RANKING_INPUTS: tuple[str, ...] = (
    "expected_information_gain",
    "expected_economic_value",
    "probability_of_resolving_uncertainty",
    "implementation_cost",
    "implementation_complexity",
    "estimated_sample_size",
    "current_uncertainty",
    "data_availability",
    "leakage_risk",
    "overlap_with_existing_research",
)


__all__ = ["render_meta_decision", "render_research_queue", "render_summary"]
