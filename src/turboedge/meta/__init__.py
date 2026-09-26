"""Meta decision layer (Phase 1: shadow mode only).

Answers "does the system know enough here?" alongside the existing
pipeline, without influencing it. See `controller.decide`.
"""

from turboedge.meta.controller import decide
from turboedge.meta.schemas import (
    DecisionConfidence,
    MetaDecision,
    MetaDecisionKind,
    ModelTrust,
)

__all__ = [
    "DecisionConfidence",
    "MetaDecision",
    "MetaDecisionKind",
    "ModelTrust",
    "decide",
]
