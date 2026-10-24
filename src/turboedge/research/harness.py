"""The harnesses the autonomous loop is allowed to execute, and nothing else.

The standing constraint this module exists to honour is that the system may
not generate and launch executable procedures of its own. So the loop does
not build an experiment for a question; it looks the question up in this
registry, and if nothing here answers it, it records
`NO_APPROVED_HARNESS` and moves to the next question.

The registry is a module-level constant of frozen dataclasses holding
direct references to already-reviewed functions. There is no name lookup,
no import by string, no parameter the caller may vary. Adding an entry is a
source change a human reviews -- which is the only mechanism by which this
loop can ever acquire a new capability.

WHAT IS IN HERE, AND WHY SO LITTLE
----------------------------------
One harness, covering two of the catalogue's fourteen questions. That is
the honest extent of what exists, and the gap is the finding, not a
shortfall in this module:

* `excursion` (`backtest/excursion_eval.py`) answers RO-MAE-PREDICTION and
  RO-MFE-PREDICTION. It is read-only, promotes nothing, changes no weight,
  gate or threshold, and is allowed to answer "not enough data to tell".
  That makes it exploratory in the sense `research/partitions.py` means.

Two existing harnesses are deliberately NOT here:

* `research synthetic-ev` is confirmatory. It is pre-registered, runs
  exactly once, and consumes the quarterly primary-hypothesis budget. The
  loop may not start a confirmatory trial, so it may not call this.
* `research ko-calibration` persists a promotion decision
  (`ko_calibration_promotion`). Promotion is human-gated while
  `alpha/promotion.py` still has unset gates, so the loop may not call it
  either -- even though the measurement part would be harmless, because
  the write is not.

The other twelve catalogue questions have no harness at all. They will
report `NO_APPROVED_HARNESS` until someone writes one and registers it
here, and the loop will skip past them to a question it can answer. That
is the intended behaviour: a loop that invented a procedure for an
unanswerable question would be the failure mode, not the feature.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from turboedge.backtest.excursion_eval import build_excursion_dataset, evaluate_excursion
from turboedge.backtest.purged_cv import PurgedWalkForwardSplit
from turboedge.models.excursion import ExcursionTarget

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from turboedge.storage.duckdb import Store

#: Verdicts that mean a screen found something worth confirming. Taken from
#: `evaluate_excursion`'s documented vocabulary, not invented here:
#: INSUFFICIENT_SAMPLE, NO_IMPROVEMENT and IMPROVEMENT_NOT_SIGNIFICANT are
#: all legitimate answers and none of them is support.
SUPPORTING_VERDICTS: frozenset[str] = frozenset({"CANDIDATE"})


@dataclass(frozen=True, slots=True)
class HarnessOutcome:
    """What a harness measured, in the terms the loop records.

    `data_version` identifies the input the harness saw. Without it a
    screen result cannot be told apart from the same screen on more data a
    month later, which is exactly the comparison the loop needs to make
    when deciding whether a question is worth revisiting.
    """

    verdict: str
    metrics: dict[str, Any]
    data_version: str
    supported: bool
    spec: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ApprovedHarness:
    """One reviewed procedure the loop may run, pinned to a version.

    `version` is bumped by hand when the procedure's behaviour changes. It
    is part of the screen id, so a screen run under an older version does
    not suppress a re-run under a newer one -- the loop would otherwise
    treat "already answered" as permanent across a method change.
    """

    name: str
    version: str
    hypothesis_ids: frozenset[str]
    description: str
    spec: Mapping[str, Any]
    run: Callable[[Store, str, datetime], HarnessOutcome]


#: Walk-forward parameters for the excursion screen. Fixed here rather than
#: taken from the caller: a loop that could vary the split until something
#: looked significant would be tuning, not screening. These match the
#: `research excursion` CLI defaults, which is where they were reviewed.
_EXCURSION_SPEC: Mapping[str, Any] = {
    "horizon": 3,
    "embargo": 3,
    "min_train": 200,
    "step": 100,
    "underlying": None,
    "partition": "DEVELOPMENT",
}


def _data_version(*parts: object) -> str:
    """A short, deterministic fingerprint of the data a harness consumed."""
    material = "|".join(str(p) for p in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _run_excursion(store: Store, hypothesis_id: str, now: datetime) -> HarnessOutcome:
    """Screen MAE or MFE predictability against its unconditional null.

    Raises on a genuine failure (including the `ValueError` that
    `build_excursion_dataset` raises when nothing has matured and been
    labelled yet). The caller records the error against the screen; it is
    not swallowed here, because a harness that reported success on no data
    would be worse than one that failed.
    """
    target = ExcursionTarget.MAE if "MAE" in hypothesis_id else ExcursionTarget.MFE
    dataset = build_excursion_dataset(store, underlying=_EXCURSION_SPEC["underlying"], as_of=now)
    result = evaluate_excursion(
        dataset,
        splitter=PurgedWalkForwardSplit(
            horizon=int(_EXCURSION_SPEC["horizon"]),
            embargo=int(_EXCURSION_SPEC["embargo"]),
            min_train=int(_EXCURSION_SPEC["min_train"]),
            step=int(_EXCURSION_SPEC["step"]),
        ),
    )

    per_target = next((r for r in result.results if r.target is target), None)
    metrics: dict[str, Any] = {
        "target": target.value,
        "n_distinct_dates": result.n_distinct_dates,
        "effective_sample": result.effective_sample,
        "feature_sample_ratio": result.feature_sample_ratio,
        "dropped_rows": result.dropped_rows,
        # Carried verbatim: the top-level verdict is the stronger of the two
        # targets, and these reasons are what name which target achieved
        # what. Collapsing them away would make the verdict unattributable.
        "verdict_reasons": result.verdict_reasons,
    }
    if per_target is not None:
        metrics |= {
            "crps_null": per_target.crps_null,
            "crps_conditional": per_target.crps_conditional,
            "mean_pinball_null": per_target.mean_pinball_null,
            "mean_pinball_conditional": per_target.mean_pinball_conditional,
            "coverage_null": per_target.coverage_null,
            "coverage_conditional": per_target.coverage_conditional,
            "significance_p_value": per_target.significance_p_value,
            "n_significance_folds": per_target.n_significance_folds,
            "n_test_samples": per_target.n_test_samples,
        }

    # `evaluate_excursion` reports ONE verdict for both targets, the
    # stronger of the two. Attributing a CANDIDATE to this hypothesis
    # therefore needs the additional, necessary condition that this
    # target's own CRPS actually improved -- otherwise an MFE candidate
    # would register as support for the MAE question. This is a guard
    # against misattribution, not a second decision rule: it can only ever
    # withhold support, never grant it.
    target_improved = per_target is not None and per_target.crps_conditional < per_target.crps_null
    supported = result.verdict in SUPPORTING_VERDICTS and target_improved

    return HarnessOutcome(
        verdict=result.verdict,
        metrics=metrics,
        data_version=_data_version(
            "excursion",
            len(dataset.y_mae),
            dataset.n_distinct_dates,
            f"{dataset.effective_sample:.6f}",
            sorted(dataset.dropped_rows.items()),
            dataset.feature_names,
        ),
        supported=supported,
        spec={**_EXCURSION_SPEC, "target": target.value, "as_of": now.isoformat()},
    )


#: Every harness the loop may execute. A frozen tuple of direct function
#: references -- extending it is a reviewed source change, which is the
#: only way this loop gains a capability.
APPROVED_HARNESSES: tuple[ApprovedHarness, ...] = (
    ApprovedHarness(
        name="excursion",
        version="1",
        hypothesis_ids=frozenset({"RO-MAE-PREDICTION", "RO-MFE-PREDICTION"}),
        description=(
            "Does a model conditioning on entry-time features predict a turbo "
            "position's maximum adverse/favourable excursion better than the "
            "unconditional empirical distribution, out of sample under purged "
            "walk-forward CV? Read-only; promotes nothing."
        ),
        spec=_EXCURSION_SPEC,
        run=_run_excursion,
    ),
)


def resolve_harness(hypothesis_id: str) -> ApprovedHarness | None:
    """The approved harness for this question, or `None` if there is none.

    `None` is a normal outcome for twelve of the fourteen catalogue
    questions and must be handled as one. The caller records
    `NO_APPROVED_HARNESS` and moves on; it does not substitute a different
    harness, and it does not build one.
    """
    for harness in APPROVED_HARNESSES:
        if hypothesis_id in harness.hypothesis_ids:
            return harness
    return None


def harnessable_hypothesis_ids() -> frozenset[str]:
    """Every question some approved harness can answer."""
    return frozenset().union(*(h.hypothesis_ids for h in APPROVED_HARNESSES))


__all__ = [
    "APPROVED_HARNESSES",
    "SUPPORTING_VERDICTS",
    "ApprovedHarness",
    "HarnessOutcome",
    "harnessable_hypothesis_ids",
    "resolve_harness",
]
