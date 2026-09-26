"""Meta decision layer.

Phase 1 answers "does the system know enough here?" alongside the existing
pipeline, without influencing it -- see `controller.decide`, shadow mode only.

Phase 2 answers "what should be investigated next?" -- see
`turboedge.meta.research_queue.ResearchQueue`, imported from its own module
rather than re-exported here. `storage.duckdb` imports the research schemas
from this package, so anything this `__init__` pulls in is loaded *before*
`Store` exists. `research_queue` reaches back into `storage` and
`learning`, so exporting it here made `from turboedge.storage.duckdb import
Store` fail as a first import -- while leaving the CLI and the test suite
green, because both happen to load `learning` earlier. This `__init__` stays
limited to modules that do not import `storage.duckdb`.

It may reorder priorities freely and may never leave an entry anywhere but
PROPOSED; every later state needs a named human.
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
    "InformationFamily",
    "MetaDecision",
    "MetaDecisionKind",
    "ModelTrust",
    "ResearchOpportunity",
    "ResearchPriority",
    "ResearchStatus",
    "StoredOpportunity",
    "SuccessfulResearchPattern",
    "decide",
    "render_meta_decision",
    "render_research_queue",
    "render_summary",
]
