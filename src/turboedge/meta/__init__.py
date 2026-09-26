"""Meta decision layer.

Phase 1 answers "does the system know enough here?" alongside the existing
pipeline, without influencing it -- see `controller.decide`, shadow mode only.

Phase 2 answers "what should be investigated next?" -- see
`research_queue.ResearchQueue`. It may reorder priorities freely and may never
leave an entry anywhere but PROPOSED; every later state needs a named human.
"""

from turboedge.meta.controller import decide
from turboedge.meta.report import (
    render_meta_decision,
    render_research_queue,
    render_summary,
)
from turboedge.meta.research_opportunity import (
    Estimate,
    EstimateBasis,
    InformationFamily,
    ResearchOpportunity,
    ResearchPriority,
    ResearchStatus,
    StoredOpportunity,
    SuccessfulResearchPattern,
)
from turboedge.meta.research_queue import IllegalTransition, ResearchQueue
from turboedge.meta.schemas import (
    DecisionConfidence,
    MetaDecision,
    MetaDecisionKind,
    ModelTrust,
)

__all__ = [
    "DecisionConfidence",
    "Estimate",
    "EstimateBasis",
    "IllegalTransition",
    "InformationFamily",
    "MetaDecision",
    "MetaDecisionKind",
    "ModelTrust",
    "ResearchOpportunity",
    "ResearchPriority",
    "ResearchQueue",
    "ResearchStatus",
    "StoredOpportunity",
    "SuccessfulResearchPattern",
    "decide",
    "render_meta_decision",
    "render_research_queue",
    "render_summary",
]
