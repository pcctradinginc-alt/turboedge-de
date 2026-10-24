"""The confirmatory-autonomy proposal is a document, not a mechanism.

`docs/proposal_confirmatory_autonomy.md` proposes rules under which the
loop could open confirmatory trials without per-trial approval. The brief
that asked for it was explicit that the system may not activate such a
change itself, so these tests pin that it has not: nothing reads the
document, the loop still cannot start a confirmatory trial, and promotion
is still locked.

A proposal to loosen a constraint on oneself is exactly the kind of
artefact whose incentives are suspect. These tests are what makes the
distinction between "proposed" and "in effect" checkable rather than a
matter of trusting the document's own heading.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

from turboedge.research.controller import ActionKind, decide_next_action

PROPOSAL = Path("docs/proposal_confirmatory_autonomy.md")
SRC = Path("src/turboedge")
_NOW = datetime(2026, 10, 24, tzinfo=UTC)


def test_the_proposal_exists_and_says_it_is_not_active() -> None:
    assert PROPOSAL.is_file()
    head = PROPOSAL.read_text(encoding="utf-8")[:400]

    assert "NOT ACTIVE" in head
    assert "NOT ACTIVATABLE BY THIS SYSTEM" in head


def test_no_source_file_reads_the_proposal() -> None:
    # The difference between a proposal and a rule is whether anything acts
    # on it. If a module ever reads this file, it has stopped being a
    # proposal regardless of what its heading says.
    readers = [
        path.relative_to(SRC).as_posix()
        for path in sorted(SRC.rglob("*.py"))
        if "proposal_confirmatory_autonomy" in path.read_text(encoding="utf-8")
    ]

    assert readers == [], readers


def test_the_governance_document_records_no_approval_of_it() -> None:
    # Activation would be a dated amendment in GOVERNANCE.md naming the
    # approver, the same mechanism as §1.1a. There is none.
    governance = Path("GOVERNANCE.md")
    if not governance.is_file():  # pragma: no cover - the file is in the repo
        return

    text = governance.read_text(encoding="utf-8")

    assert "proposal_confirmatory_autonomy" not in text
    assert "confirmatory autonomy" not in text.lower()


def test_the_controller_still_has_exactly_five_actions() -> None:
    # A sixth action is how confirmatory autonomy would arrive. There is no
    # sixth.
    assert set(ActionKind) == {
        ActionKind.START_EXPLORATORY,
        ActionKind.AWAIT_DATA,
        ActionKind.AWAIT_CONFIRMATION_APPROVAL,
        ActionKind.SHELVE_FAMILY,
        ActionKind.NO_ACTION,
    }
    assert not any("CONFIRMATORY" in k and "AWAIT" not in k for k in ActionKind)


def test_promotion_is_still_locked_by_default() -> None:
    import inspect

    signature = inspect.signature(decide_next_action)
    default = signature.parameters["promotion_unlocked"].default

    assert default is False, (
        "promotion_unlocked exists only to be False, so that 'this is locked' "
        "is recorded in the evidence rather than being invisible."
    )


def test_support_still_stops_at_a_human() -> None:
    from turboedge.research.controller import Candidate

    supported = Candidate(
        hypothesis_id="RO-A",
        family="macro",
        priority=0.9,
        effective_sample=500.0,
        required_sample=48.0,
        data_ready=True,
        exploratory_done=True,
        has_exploratory_support=True,
    )

    action = decide_next_action(
        candidates=[supported],
        exploratory_screens_this_quarter=0,
        confirmatory_trials_this_quarter=0,
        confirmatory_budget=6,
        now=_NOW,
    )

    assert action.kind is ActionKind.AWAIT_CONFIRMATION_APPROVAL
    assert "named human" in action.reason
    assert "does not start them" in action.reason


def test_no_module_calls_the_approval_function() -> None:
    # `meta/research_queue.approve` refuses to run without a named human.
    # Nothing in the loop may call it, with a name or otherwise.
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if path.as_posix().endswith("meta/research_queue.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "approve" and "cli" not in path.as_posix():
                offenders.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno}")

    assert offenders == [], offenders
