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

#: Jobs that are not pipeline modes. Only the resolver: the watchdog moved
#: to its own workflow on 2026-10-24 (see `WATCHDOG_WORKFLOW`).
NON_MODE_JOBS = frozenset({"resolve-mode"})

#: The watchdog's own workflow.
WATCHDOG_WORKFLOW = Path(".github/workflows/health-watch.yml")


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


@pytest.fixture(scope="module")
def watchdog() -> dict[str, Any]:
    if not WATCHDOG_WORKFLOW.is_file():
        pytest.skip(f"{WATCHDOG_WORKFLOW} not present")
    loaded = yaml.safe_load(WATCHDOG_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _watchdog_steps() -> Any:
    loaded = yaml.safe_load(WATCHDOG_WORKFLOW.read_text(encoding="utf-8"))
    return loaded["jobs"]["health-watch"]["steps"]


def test_the_watchdog_is_not_a_job_of_the_pipeline(workflow: dict[str, Any]) -> None:
    # The reason is measured, not stylistic. A failing job fails the run,
    # and `.github/actions/turboedge-state` restores from
    # `gh run list --workflow pipeline.yml --status success` -- so a run
    # whose watchdog failed became invisible to the restore even though its
    # data jobs had succeeded and packed state. Checked against the Actions
    # API on 2026-10-24: all eight runs reported `failure` on 2026-10-23/24
    # had uploaded `turboedge-state-enc`, every one of those artifacts was
    # unreachable, and the newest state the pipeline could restore was from
    # 12:49 that day -- discarding a successful 65-minute external ingest
    # and a successful eod. A watchdog built to make silent data loss
    # audible was causing it.
    assert "health-watch" not in workflow["jobs"], (
        "the watchdog is a job of pipeline.yml again. Its failure would fail "
        "the run, and the state restore only considers successful runs, so "
        "every data job's packed state in that run becomes unreachable."
    )


def test_the_watchdog_cannot_affect_the_data_path(
    workflow: dict[str, Any], watchdog: dict[str, Any]
) -> None:
    # It exists to make silence audible, so it must not be able to gate,
    # delay or break a mode job -- and it must not touch the state.
    for name in workflow["jobs"]:
        needs = workflow["jobs"][name].get("needs") or []
        assert "health-watch" not in needs, f"{name} must not depend on the watchdog"

    steps = yaml.dump(watchdog["jobs"]["health-watch"]["steps"])
    assert "turboedge-state" not in steps, "the watchdog must not restore or pack state"
    assert "write" not in yaml.dump(watchdog.get("permissions", {})), (
        "the watchdog reads the Actions API and nothing else"
    )


def test_the_watchdog_runs_on_a_schedule_of_its_own(watchdog: dict[str, Any]) -> None:
    # A watchdog triggered only by pipeline runs cannot detect the pipeline
    # not running at all -- and a cron that silently stops firing is a real
    # failure mode here: on 2026-09-14 only one of three scheduled scans
    # fired, and that one 37 minutes late.
    triggers = watchdog[True] if True in watchdog else watchdog["on"]

    assert "schedule" in triggers, "without its own schedule, total silence is undetectable"
    assert "workflow_run" in triggers, "it should also check right after each pipeline run"
    assert triggers["workflow_run"]["workflows"] == ["Pipeline"]


def test_the_watchdog_is_never_cancelled_in_progress(watchdog: dict[str, Any]) -> None:
    # A cancelled watchdog is a watchdog that said nothing.
    assert watchdog["concurrency"]["cancel-in-progress"] is False


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


def test_the_watchdog_monitors_exactly_the_jobs_that_move_data(
    workflow: dict[str, Any],
) -> None:
    # If a mode is added to the pipeline and not to MONITORED_JOBS, its
    # failures are invisible -- which is the condition this watchdog was
    # built to end, reintroduced one mode at a time.
    from turboedge.monitoring.pipeline_health import MONITORED_JOBS, WATCHDOG_JOB

    assert WATCHDOG_JOB not in workflow["jobs"]
    assert set(workflow["jobs"]) == MONITORED_JOBS, (
        "the watchdog's monitored-job set no longer matches the workflow's jobs. "
        "Every job except the watchdog itself either moves data or gates the "
        "jobs that do."
    )


def test_the_watchdog_decides_in_python_not_in_yaml(workflow: dict[str, Any]) -> None:
    # The jq version of this check was wrong in three ways that no test
    # could catch, because a filter inside a YAML `run:` block is not
    # testable. Keeping the decision in a module keeps it tested; see
    # tests/monitoring/test_pipeline_health.py.
    steps = yaml.dump(_watchdog_steps())

    assert "pipeline_health.py" in steps, (
        "the watchdog no longer calls the tested evaluator. Collect in the "
        "workflow if you must, but decide in Python."
    )
    assert "::error::" not in steps, (
        "the watchdog annotates from the workflow again. The annotation belongs "
        "with the decision, where it is tested."
    )


def test_the_watchdog_runs_without_installing_the_package(
    workflow: dict[str, Any],
) -> None:
    # The last line of defence must not depend on a dependency install of
    # the package it is watching, so the evaluator imports nothing outside
    # the standard library and is run as a plain script.
    import ast
    import sys

    steps = yaml.dump(_watchdog_steps())
    assert "uv run" not in steps, "the watchdog must not need the project environment"
    assert "uv sync" not in steps

    source = Path("src/turboedge/monitoring/pipeline_health.py").read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    non_stdlib = sorted(imported - sys.stdlib_module_names)
    assert non_stdlib == [], (
        f"the watchdog now imports {non_stdlib}. It is run as a bare script with "
        "no dependency install, so a third-party import makes it fail exactly "
        "when the environment is what broke."
    )


# -- the state restore must not filter on the run conclusion -------------

STATE_ACTION = Path(".github/actions/turboedge-state/action.yml")


def test_the_state_restore_does_not_filter_on_run_conclusion() -> None:
    """The measured data loss of 2026-10-23/24, structurally prevented.

    "Pack and upload state" is the LAST step of every data job and is gated
    on `if: success()`, so the artifact can only exist if every data step of
    that job completed. Its existence is a stronger and more precise signal
    than the run's conclusion, which is the maximum over ALL jobs in the run
    -- including jobs that touch no state.

    Filtering candidates on `--status success` discarded exactly that: eight
    runs were reported `failure` while every data job in them succeeded and
    all eight had uploaded `turboedge-state-enc`, and the chain resumed from
    a day-old state. Four scans' worth of bid/ask snapshots went with it,
    and a quote from a given minute cannot be fetched again later.
    """
    if not STATE_ACTION.is_file():
        pytest.skip(f"{STATE_ACTION} not present")
    # Comment lines are stripped: the file explains the filter it no longer
    # uses, and a test that forbade the words would forbid the explanation.
    source = "\n".join(
        line
        for line in STATE_ACTION.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )

    assert "--status success" not in source, (
        "the state restore filters candidate runs on the run-level conclusion "
        "again. A run can be red for a reason that has nothing to do with the "
        "state it packed, and that filter is what made eight runs' data "
        "unreachable on 2026-10-23/24."
    )


def test_the_state_pack_is_the_last_step_of_every_data_job(
    workflow: dict[str, Any],
) -> None:
    # The restore's safety now rests on this: the artifact exists only if
    # every preceding step of that job succeeded. If packing moved earlier,
    # an artifact could represent a job that then went on to fail partway,
    # and "artifact exists" would stop meaning "the data work completed".
    for name in ("scan", "eod", "external", "research", "weekly", "monthly"):
        spec = workflow["jobs"].get(name)
        if spec is None:
            continue
        steps = spec["steps"]
        packs = [
            i
            for i, step in enumerate(steps)
            if "pack" in str(step.get("name", "")).lower()
            and "state" in str(step.get("name", "")).lower()
            and "snapshot" not in str(step.get("name", "")).lower()
        ]
        if not packs:
            continue
        last_pack = max(packs)
        after = [
            str(step.get("name", step.get("uses")))
            for step in steps[last_pack + 1 :]
            if str(step.get("if", "")) != "always()"
        ]
        assert after == [], (
            f"{name} runs {after} after packing its state. The restore treats the "
            "artifact's existence as proof the job's data work completed, which "
            "only holds while packing is last."
        )
        assert "success()" in str(steps[last_pack].get("if", "")), (
            f"{name}'s state pack is no longer gated on success(), so the artifact "
            "would no longer prove that the data steps completed."
        )


# -- the recovery workflow must not be able to damage anything -----------

RECOVERY_WORKFLOW = Path(".github/workflows/state-recovery.yml")


@pytest.fixture(scope="module")
def recovery() -> dict[str, Any]:
    if not RECOVERY_WORKFLOW.is_file():
        pytest.skip(f"{RECOVERY_WORKFLOW} not present")
    loaded = yaml.safe_load(RECOVERY_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_the_recovery_runs_only_when_a_human_dispatches_it(
    recovery: dict[str, Any],
) -> None:
    # It decrypts and merges production state. Nothing should be able to
    # start that except a person deciding to.
    triggers = recovery[True] if True in recovery else recovery["on"]

    assert set(triggers) == {"workflow_dispatch"}, (
        "the state recovery gained a trigger other than workflow_dispatch"
    )


def test_the_recovery_defaults_to_a_dry_run(recovery: dict[str, Any]) -> None:
    # A dispatch made without reading the inputs must do no harm.
    triggers = recovery[True] if True in recovery else recovery["on"]
    default = triggers["workflow_dispatch"]["inputs"]["dry_run"]["default"]

    assert default == "true", "dry_run no longer defaults to true"
    assert isinstance(default, str), (
        "workflow_dispatch inputs arrive as strings; a boolean default would make "
        "the `!= 'true'` comparison in the pack step always true"
    )


def test_the_recovered_state_is_not_uploaded_as_the_live_artifact(
    recovery: dict[str, Any],
) -> None:
    # `pipeline.yml`'s restore now selects the newest run holding
    # `turboedge-state-enc`. Uploading under that name -- even from another
    # workflow -- would be one rename away from the live chain silently
    # adopting an unreviewed merge.
    steps = recovery["jobs"]["recover"]["steps"]
    pack = [s for s in steps if "turboedge-state-pack" in str(s.get("uses", ""))]

    assert pack, "the recovery no longer packs anything"
    for step in pack:
        name = str((step.get("with") or {}).get("artifact_name", ""))
        assert name and name != "turboedge-state-enc", (
            f"the recovery uploads under {name!r}; the live restore selects "
            "`turboedge-state-enc`, so promoting a merge must stay a deliberate act"
        )
        assert str((step.get("with") or {}).get("push_to_backend", "false")) != "true", (
            "the recovery pushes to the durable state backend, which is a restore source of its own"
        )


def test_the_recovery_only_packs_after_a_clean_verification(
    recovery: dict[str, Any],
) -> None:
    # A merge that introduced duplicate keys must never be packed: a
    # duplicated snapshot doubles the observation counts that retention
    # thinning and every effective-sample figure derive from.
    steps = recovery["jobs"]["recover"]["steps"]
    names = [str(s.get("name", s.get("uses"))) for s in steps]
    verify = next(i for i, n in enumerate(names) if n.startswith("Verify"))
    pack = next(i for i, s in enumerate(steps) if "turboedge-state-pack" in str(s.get("uses", "")))

    assert pack > verify, "the recovery packs before verifying the merge"
    condition = str(steps[pack].get("if", ""))
    assert "success()" in condition, "the pack step is no longer gated on success()"
    assert "dry_run" in condition, "the pack step no longer honours dry_run"


def test_the_recovery_never_unpacks_into_the_live_state(
    recovery: dict[str, Any],
) -> None:
    # Each archive goes into its own directory. An unpack over `./state`
    # would replace the live database instead of merging into it.
    steps = yaml.dump(recovery["jobs"]["recover"]["steps"])

    assert "state unpack" in steps
    assert "--state-dir ./state state unpack" not in steps, (
        "the recovery unpacks an archive over the live state directory"
    )


def test_the_recovery_merges_rather_than_replaces(recovery: dict[str, Any]) -> None:
    from turboedge.state.recovery import merge_recovered_state

    steps = yaml.dump(recovery["jobs"]["recover"]["steps"])
    assert "state merge-recovered" in steps

    # And the merge itself must stay insert-only.
    import inspect

    source = inspect.getsource(merge_recovered_state)
    for statement in ("DELETE", "UPDATE ", "DROP ", "INSERT OR REPLACE"):
        assert statement not in source.upper().replace("INSERT INTO", ""), (
            f"{statement} appears in the merge; it must only ever insert rows the "
            "current database lacks"
        )
