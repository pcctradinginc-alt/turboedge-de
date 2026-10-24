"""Decide whether recent pipeline runs warrant an alarm.

WHY THIS EXISTS
---------------
From 2026-09-29 to 2026-10-23 the `eod` job was cancelled by its timeout on
19 consecutive weekdays, discarding each night's label/learn output, and
nobody was told. The reason is specific: **GitHub emails the owner when a
run FAILS, not when it is CANCELLED, and a job timeout is reported as
cancelled.** The loudest failure mode this pipeline had was also its
quietest. A watchdog that turns "a recent data job did not succeed" into an
actual failure fixes that, because a failure does notify.

WHY IT IS PYTHON AND NOT A SHELL STEP
-------------------------------------
The first version was a jq one-liner in the workflow, and it was wrong in
three ways that no test could have caught, because there was nothing to
test:

1. It filtered `.conclusion != null`. `gh run list --json conclusion`
   reports an in-progress run as the empty string, not null, so the
   watchdog counted the very run it was part of -- whose `external` job was
   still running and went on to succeed -- as a failure. Observed on run
   39999355264.
2. It judged the *run* conclusion. Its own failure therefore made the run
   fail, so the next run saw a non-successful run in the window and failed
   too: one false alarm became an unbroken chain of them, each one looking
   like a fresh data incident.
3. It re-reported every incident in a 36-hour window on every run. The 19
   cancelled `eod` runs would have produced an alarm on all ~45 runs of the
   following day and a half, which is the shape of alert fatigue that makes
   the 20th silence go unnoticed.

So the rules are: only *completed* runs, only the jobs that actually move
data, never the watchdog's own result, and never an incident already
reported by an earlier watchdog run.

WHAT IT DELIBERATELY STILL ALARMS ON
------------------------------------
A cancelled data job, including one a human cancelled. Distinguishing a
human cancel from a timeout is not reliably possible from the Actions API,
and the asymmetry is not close: a missed timeout costs a night of data
silently, while a redundant alarm about a cancel the operator performed
costs one email they already expected.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

#: Jobs whose failure means data was not collected or not learned from.
#: `resolve-mode` is included because every data job is gated on its output:
#: if it fails, the run does nothing at all.
MONITORED_JOBS: frozenset[str] = frozenset(
    {"resolve-mode", "scan", "eod", "external", "research", "weekly", "monthly"}
)

#: The watchdog's own job. Never monitored -- see reason 2 in the module
#: docstring. A broken watchdog is a watchdog problem, not a data incident,
#: and must not be able to manufacture an endless chain of alarms.
WATCHDOG_JOB = "health-watch"

#: Job conclusions that mean the job did not do its work. `cancelled` is
#: here because that is how a job timeout is reported -- the whole reason
#: this module exists. `skipped` is absent: a mode job not selected for this
#: run is the normal case.
FAILING_CONCLUSIONS: frozenset[str] = frozenset(
    {"failure", "cancelled", "timed_out", "startup_failure", "action_required", "stale"}
)

#: Covers a full day of scans plus eod and external with room for a late
#: start, without reaching back far enough to keep alarming about something
#: already fixed.
WINDOW_HOURS = 36


@dataclass(frozen=True, slots=True)
class JobObservation:
    """One job of one run, as the Actions API reports it."""

    name: str
    status: str
    conclusion: str | None

    @property
    def completed(self) -> bool:
        # The empty string is what `gh` prints for a job that has not
        # finished. Treating it as a conclusion is exactly the bug that
        # made the watchdog fail on a run whose job went on to succeed.
        return self.status == "completed" and bool(self.conclusion)

    @property
    def failed(self) -> bool:
        return self.completed and self.conclusion in FAILING_CONCLUSIONS


@dataclass(frozen=True, slots=True)
class RunObservation:
    """One pipeline run and its jobs."""

    run_id: int
    created_at: datetime
    status: str
    jobs: tuple[JobObservation, ...] = ()
    #: The run-level conclusion. Not used to decide whether a data job
    #: failed -- that was the original bug -- but it is what the state
    #: restore filters on, so it is what decides whether a run's packed
    #: state can ever be read again.
    conclusion: str | None = None

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def job(self, name: str) -> JobObservation | None:
        for candidate in self.jobs:
            if candidate.name == name:
                return candidate
        return None


@dataclass(frozen=True, slots=True)
class Incident:
    """A completed monitored job that did not do its work."""

    run_id: int
    created_at: datetime
    job: str
    conclusion: str

    def describe(self) -> str:
        return f"{self.created_at.isoformat()}  {self.job}  {self.conclusion}  run {self.run_id}"


@dataclass(frozen=True, slots=True)
class StallVerdict:
    """Whether the pipeline appears to have stopped running altogether."""

    stalled: bool
    last_run_at: datetime | None
    hours_since: float | None


@dataclass(frozen=True, slots=True)
class StateChainVerdict:
    """Whether any recent run's packed state is reachable by the restore.

    `.github/actions/turboedge-state` restores from
    `gh run list --workflow pipeline.yml --status success`, so a run that
    went red for ANY reason takes its packed state out of reach -- even
    when every data job in it succeeded. That is not hypothetical: a
    watchdog job added on 2026-10-23 to make silent data loss audible made
    eight consecutive runs red, all eight had uploaded
    `turboedge-state-enc`, and the newest reachable state was from 12:49
    that day, behind a successful 65-minute external ingest and a
    successful eod.

    That specific cause is fixed by moving the watchdog to its own
    workflow. This check is here because the *class* of cause is not: any
    future job that goes red hides the state the same way, and the
    symptom -- runs happening, none of them succeeding -- is visible
    without knowing why.
    """

    stale: bool
    last_success_at: datetime | None
    completed_runs_in_window: int
    #: Runs that did successful data work which the restore cannot reach,
    #: because the run as a whole went red and is therefore filtered out.
    unreachable_runs: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class HealthVerdict:
    """What the watchdog found, and whether it should fail the run."""

    alarm: bool
    incidents: tuple[Incident, ...]
    window_start: datetime
    window_floor: datetime
    last_alarm_at: datetime | None
    considered_runs: int
    skipped_in_progress: tuple[int, ...]
    already_reported: tuple[Incident, ...]
    reason: str
    stall: StallVerdict = StallVerdict(stalled=False, last_run_at=None, hours_since=None)
    state_chain: StateChainVerdict = StateChainVerdict(
        stale=False, last_success_at=None, completed_runs_in_window=0
    )


def _is_watchdog_alarm(run: RunObservation) -> bool:
    watchdog = run.job(WATCHDOG_JOB)
    return watchdog is not None and watchdog.failed


def _evaluate_stall(
    history: Sequence[RunObservation] | None,
    *,
    now: datetime,
    window_floor: datetime,
) -> StallVerdict:
    """Has the pipeline stopped running, as opposed to never having run?

    This is the one incident a watchdog triggered only by pipeline runs
    could never see: if the cron stops firing, nothing calls the watchdog,
    so total silence is the one state it cannot report. The scheduled
    trigger exists for this, and `history` -- every run the listing
    returned, in or out of the window -- is how it tells a repository that
    has never run from one that has stopped.

    A cron that silently stops is not hypothetical here: on 2026-09-14 only
    one of three scheduled scans fired, and that one 37 minutes late.
    """
    if not history:
        # No history at all. A first run, or a listing that returned
        # nothing; neither is evidence that something broke.
        return StallVerdict(stalled=False, last_run_at=None, hours_since=None)
    last = max(run.created_at for run in history)
    hours = (now - last).total_seconds() / 3600.0
    return StallVerdict(stalled=last < window_floor, last_run_at=last, hours_since=hours)


def evaluate(
    runs: list[RunObservation],
    *,
    now: datetime,
    current_run_id: int | None = None,
    window_hours: int = WINDOW_HOURS,
    history: Sequence[RunObservation] | None = None,
) -> HealthVerdict:
    """Which monitored jobs failed recently and have not been reported yet.

    The window starts at the later of `now - window_hours` and the moment
    of the most recent run whose own watchdog already alarmed. That second
    bound is the de-duplication: anything older has been reported, and
    reporting it again on every subsequent run is how a real incident gets
    lost in a wall of repeats.

    The alarming run itself is *included* in the window, not excluded. Its
    watchdog ran in parallel with its data jobs and so never judged them --
    on purpose, so that a watchdog outage cannot stop a scan -- which means
    this is the first evaluation that can see them.
    """
    window_floor = now - timedelta(hours=window_hours)

    in_window = [
        run for run in runs if run.created_at >= window_floor and run.run_id != current_run_id
    ]
    skipped_in_progress = tuple(run.run_id for run in in_window if not run.completed)
    completed = [run for run in in_window if run.completed]

    alarm_times = [run.created_at for run in completed if _is_watchdog_alarm(run)]
    last_alarm_at = max(alarm_times) if alarm_times else None
    window_start = max(window_floor, last_alarm_at) if last_alarm_at else window_floor

    incidents: list[Incident] = []
    already_reported: list[Incident] = []
    for run in completed:
        for job in run.jobs:
            if job.name not in MONITORED_JOBS or not job.failed:
                continue
            assert job.conclusion is not None  # `failed` implies a conclusion
            incident = Incident(
                run_id=run.run_id,
                created_at=run.created_at,
                job=job.name,
                conclusion=job.conclusion,
            )
            if run.created_at >= window_start:
                incidents.append(incident)
            else:
                already_reported.append(incident)

    incidents.sort(key=lambda i: (i.created_at, i.job))
    already_reported.sort(key=lambda i: (i.created_at, i.job))

    stall = _evaluate_stall(history, now=now, window_floor=window_floor)

    considered = list(history) if history is not None else in_window
    successes = [run.created_at for run in considered if run.conclusion == "success"]
    last_success = max(successes) if successes else None
    # Absent conclusions are absent data, not evidence of a problem. A
    # watchdog that alarms when it cannot see is a watchdog that cries wolf
    # the first time a field is renamed upstream.
    conclusions_seen = any(run.conclusion is not None for run in considered)
    # The precise condition, not a time window: a red run that nonetheless
    # did successful data work, newer than the last run the restore can
    # select. That work exists, it was packed, and it cannot be read again.
    # A coarser "the last success is old" rule would have missed the real
    # incident, where the last success was only twelve hours behind.
    unreachable = tuple(
        run.run_id
        for run in completed
        if run.conclusion is not None
        and run.conclusion != "success"
        and (last_success is None or run.created_at > last_success)
        and any(job.name in MONITORED_JOBS and job.conclusion == "success" for job in run.jobs)
    )
    state_chain = StateChainVerdict(
        stale=bool(unreachable) and conclusions_seen,
        last_success_at=last_success,
        completed_runs_in_window=len(completed),
        unreachable_runs=unreachable,
    )

    if incidents:
        reason = (
            f"{len(incidents)} monitored job(s) did not succeed since {window_start.isoformat()}"
        )
    elif state_chain.stale:
        reason = (
            f"{len(state_chain.unreachable_runs)} run(s) did successful data work that "
            "the state restore cannot reach, because the run as a whole went red and "
            "the restore selects only successful runs: "
            f"{', '.join(str(r) for r in state_chain.unreachable_runs)}. Whatever is "
            "making those runs red is also hiding every data job's output in them."
        )
    elif stall.stalled and stall.last_run_at is not None and stall.hours_since is not None:
        reason = (
            f"the pipeline has not run since {stall.last_run_at.isoformat()} "
            f"({stall.hours_since:.0f}h ago), past the {window_hours}h window. A "
            "schedule that stops firing produces no failing job to notice."
        )
    elif already_reported:
        # Ordered after the two real findings: this is a non-finding, and
        # letting it take the reason line made the annotation say "already
        # reported" while the error said state was unreachable.
        reason = (
            f"{len(already_reported)} earlier incident(s) in the window were already "
            "reported by a previous watchdog alarm"
        )
    elif not completed:
        reason = "no completed runs in the window"
    else:
        reason = f"all monitored jobs in {len(completed)} completed run(s) did their work"

    return HealthVerdict(
        alarm=bool(incidents) or stall.stalled or state_chain.stale,
        incidents=tuple(incidents),
        window_start=window_start,
        window_floor=window_floor,
        last_alarm_at=last_alarm_at,
        considered_runs=len(completed),
        skipped_in_progress=skipped_in_progress,
        already_reported=tuple(already_reported),
        reason=reason,
        stall=stall,
        state_chain=state_chain,
    )


def _parse_timestamp(value: Any) -> datetime:
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_runs(payload: Any) -> list[RunObservation]:
    """Build run observations from the shape `gh` produces.

    Accepts both `databaseId` (what `gh run list --json` prints) and `id`
    (what `gh api` returns), because the workflow uses both commands.
    """
    runs: list[RunObservation] = []
    for raw in payload or []:
        if not isinstance(raw, dict):
            continue
        run_id = raw.get("databaseId", raw.get("id"))
        created = raw.get("createdAt", raw.get("created_at"))
        if run_id is None or created is None:
            continue
        jobs = tuple(
            JobObservation(
                name=str(job.get("name", "")),
                status=str(job.get("status", "")),
                conclusion=(
                    str(job["conclusion"]) if job.get("conclusion") not in (None, "") else None
                ),
            )
            for job in raw.get("jobs") or []
            if isinstance(job, dict)
        )
        run_conclusion = raw.get("conclusion")
        runs.append(
            RunObservation(
                run_id=int(run_id),
                created_at=_parse_timestamp(created),
                status=str(raw.get("status", "")),
                jobs=jobs,
                conclusion=(str(run_conclusion) if run_conclusion not in (None, "") else None),
            )
        )
    return runs


def markdown_summary(verdict: HealthVerdict) -> str:
    """The GitHub step summary, written whether or not there is an alarm."""
    lines = [
        "## Pipeline health",
        "",
        f"- Window: `{verdict.window_start.isoformat()}` .. now",
        f"- Completed runs considered: {verdict.considered_runs}",
        f"- Verdict: {'ALARM' if verdict.alarm else 'ok'} -- {verdict.reason}",
    ]
    if verdict.last_alarm_at is not None:
        lines.append(
            f"- Window shortened to start at the last watchdog alarm "
            f"(`{verdict.last_alarm_at.isoformat()}`) so reported incidents are not repeated."
        )
    if verdict.stall.stalled:
        lines.append(f"- **The pipeline has stopped running.** {verdict.reason}")
    if verdict.state_chain.stale:
        lines.append(
            "- **Successful data work is unreachable.** The state restore selects "
            f"only successful runs; the last one was "
            f"`{verdict.state_chain.last_success_at}`, and "
            f"{len(verdict.state_chain.unreachable_runs)} newer red run(s) packed "
            "state that cannot be read again."
        )
    if verdict.skipped_in_progress:
        lines.append(
            "- Runs still in progress, not judged: "
            + ", ".join(str(r) for r in verdict.skipped_in_progress)
        )
    if verdict.already_reported:
        lines.append(
            f"- Suppressed as already reported: {len(verdict.already_reported)} incident(s)."
        )
    if verdict.incidents:
        lines += ["", "### New incidents", ""]
        lines += [
            f"- `{i.conclusion}` **{i.job}** at {i.created_at.isoformat()} "
            f"([run](../../actions/runs/{i.run_id}))"
            for i in verdict.incidents
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Read the collected Actions data on stdin, report, and exit 1 on alarm.

    Input is one JSON object: `{"now": ..., "current_run_id": ..., "runs":
    [...]}`. The workflow collects it with `gh`; everything that decides
    anything happens here, where it is tested.
    """
    _ = argv
    document = json.loads(sys.stdin.read() or "{}")
    now = _parse_timestamp(document["now"]) if document.get("now") else datetime.now(UTC)
    current = document.get("current_run_id")
    verdict = evaluate(
        parse_runs(document.get("runs")),
        now=now,
        current_run_id=int(current) if current not in (None, "") else None,
        history=parse_runs(document.get("history")) if document.get("history") else None,
    )

    print(markdown_summary(verdict))
    for incident in verdict.incidents:
        print(f"  {incident.describe()}")

    if not verdict.alarm:
        return 0

    if verdict.stall.stalled:
        print(f"::error::the pipeline has stopped running. {verdict.reason}")
    if verdict.state_chain.stale:
        print(
            f"::error::successful data work is unreachable to the state restore. {verdict.reason}"
        )
    if verdict.incidents:
        print(
            f"::error::{len(verdict.incidents)} monitored pipeline job(s) did not succeed. "
            "A job timeout is reported as 'cancelled', which GitHub does NOT email about -- "
            "that is why 19 such runs went unnoticed for 25 days from 2026-09-29. This step "
            "fails on purpose so the notification happens."
        )
    return 1


if __name__ == "__main__":  # pragma: no cover - exercised via the workflow
    raise SystemExit(main())
