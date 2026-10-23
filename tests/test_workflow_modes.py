"""The pipeline workflow's modes must stay consistent in four places.

A mode has to be declared in the `workflow_dispatch` choice list, mapped
from a cron in `resolve-mode`, accepted by that job's own validation case,
and implemented as a job gated on `needs.resolve-mode.outputs.mode`. Four
places, no mechanism keeping them aligned.

Measured 2026-10-23: adding the `external` mode updated three of the four.
The dispatch list was missed, so `gh workflow run -f mode=external` failed
with HTTP 422 -- which at least failed loudly. The dangerous version of the
same mistake is a cron that resolves to a mode with no job: `resolve-mode`
succeeds, every mode job evaluates its `if:` to false and is skipped, and
the run reports success having done nothing at all. That is indistinguishable
from a healthy run in the Actions list.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW = Path(".github/workflows/pipeline.yml")

#: Jobs that are not pipeline modes: the resolver itself and the watchdog,
#: which runs in every mode and is deliberately ungated.
NON_MODE_JOBS = frozenset({"resolve-mode", "health-watch"})


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    if not WORKFLOW.is_file():
        pytest.skip(f"{WORKFLOW} not present")
    loaded = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML resolves the bare key `on:` to the boolean True.
    return workflow[True] if True in workflow else workflow["on"]


def _mode_jobs(workflow: dict[str, Any]) -> set[str]:
    return set(workflow["jobs"]) - NON_MODE_JOBS


def _resolver_script(workflow: dict[str, Any]) -> str:
    return str(workflow["jobs"]["resolve-mode"]["steps"][0]["run"])


def test_dispatch_options_and_mode_jobs_are_the_same_set(
    workflow: dict[str, Any],
) -> None:
    options = set(_triggers(workflow)["workflow_dispatch"]["inputs"]["mode"]["options"])

    assert options == _mode_jobs(workflow)


def _mode_for_cron(script: str, cron: str) -> str | None:
    """The mode a shell `case` arm assigns to `cron`, or None.

    Walks the lines rather than matching one clever pattern: several crons
    share a single arm (`"a"|"b"|"c")` for the five scan times), and a regex
    that handles that correctly is harder to read than the loop.
    """
    lines = script.splitlines()
    for i, line in enumerate(lines):
        if f'"{cron}"' not in line:
            continue
        for following in lines[i:]:
            if "mode=" in following:
                rest = following.split("mode=", 1)[1]
                found = re.match(r'\s*"?(\w+)', rest)
                return found.group(1) if found else None
    return None


def test_every_cron_resolves_to_a_mode_that_has_a_job(workflow: dict[str, Any]) -> None:
    script = _resolver_script(workflow)
    jobs = _mode_jobs(workflow)

    for entry in _triggers(workflow)["schedule"]:
        cron = entry["cron"]
        mode = _mode_for_cron(script, cron)
        assert mode is not None, f"cron {cron!r} is not mapped to any mode"
        assert mode in jobs, (
            f"cron {cron!r} resolves to mode {mode!r}, which has no job; the run "
            "would report success having skipped every job"
        )


def test_every_mode_job_is_reachable_from_some_cron(workflow: dict[str, Any]) -> None:
    # The reverse direction: a job nobody schedules is a job that never
    # runs, and nothing in the Actions list would say so.
    script = _resolver_script(workflow)
    scheduled = {_mode_for_cron(script, entry["cron"]) for entry in _triggers(workflow)["schedule"]}

    assert _mode_jobs(workflow) <= scheduled


def test_the_resolver_validation_accepts_exactly_the_mode_jobs(
    workflow: dict[str, Any],
) -> None:
    script = _resolver_script(workflow)
    # The second `case` validates the resolved mode: `scan|eod|...) ;;`
    match = re.search(r"^\s*([a-z|]+)\)\s*;;\s*$", script, re.M)
    assert match is not None, "no mode-validation case found in resolve-mode"
    accepted = set(match.group(1).split("|"))

    assert accepted == _mode_jobs(workflow)


def test_every_mode_job_is_gated_on_the_resolved_mode(workflow: dict[str, Any]) -> None:
    for name in _mode_jobs(workflow):
        spec = workflow["jobs"][name]
        condition = str(spec.get("if", ""))
        assert "resolve-mode" in condition, f"{name} is not gated on the resolver"
        assert f"'{name}'" in condition, f"{name} is gated on another mode's name"
        assert "resolve-mode" in (spec.get("needs") or []), f"{name} does not need the resolver"


def test_the_watchdog_cannot_affect_the_data_path(workflow: dict[str, Any]) -> None:
    # It exists to make silence audible, so it must not be able to gate,
    # delay or break a mode job -- and it must not touch the state.
    watchdog = workflow["jobs"]["health-watch"]

    assert "needs" not in watchdog, "the watchdog must not gate on anything"
    assert "if" not in watchdog, "the watchdog runs in every mode"
    for name in _mode_jobs(workflow):
        needs = workflow["jobs"][name].get("needs") or []
        assert "health-watch" not in needs, f"{name} must not depend on the watchdog"

    steps = yaml.dump(watchdog["steps"])
    assert "turboedge-state" not in steps, "the watchdog must not restore or pack state"


def test_the_external_job_has_room_for_a_first_backfill(workflow: dict[str, Any]) -> None:
    # The whole point of separating it. A first run fetches full history
    # from every source, and PortWatch alone honours a 60-second crawl
    # delay across 11 series; measured 2026-10-23, 30 minutes was not
    # enough and the job was cancelled 19 weekdays running.
    external = workflow["jobs"]["external"]
    eod = workflow["jobs"]["eod"]

    assert external["timeout-minutes"] >= 60
    assert external["timeout-minutes"] > eod["timeout-minutes"]


def test_no_state_save_is_loosened_to_always(workflow: dict[str, Any]) -> None:
    # `if: always()` on a save looks like insurance and is not: the restore
    # action searches `gh run list --status success`, so an artifact from a
    # cancelled run is never read again. Loosening it would buy nothing and
    # contradict turboedge-state-pack's documented contract.
    for name, spec in workflow["jobs"].items():
        for step in spec.get("steps", []):
            uses = str(step.get("uses", ""))
            if "turboedge-state-pack" in uses:
                assert step.get("if") == "success()", (
                    f"{name}/{step.get('name')!r} must gate its state save on "
                    "success(), not always()"
                )
