"""Regression tests for the pipeline watchdog.

Each of the first four tests pins one way the shell version of this
watchdog was wrong on 2026-10-24. They are the reason the logic moved out
of the workflow: a jq filter inside a YAML `run:` block cannot be tested,
and all three of its bugs were the kind a test catches instantly.

The last group pins what must keep working: a real timeout still alarms.
A watchdog that never cries wolf by never barking is the failure mode the
19 silent nights already demonstrated.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from turboedge.monitoring.pipeline_health import (
    FAILING_CONCLUSIONS,
    MONITORED_JOBS,
    WATCHDOG_JOB,
    JobObservation,
    RunObservation,
    evaluate,
    main,
    markdown_summary,
    parse_runs,
)

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


def _job(name: str, conclusion: str | None, *, status: str = "completed") -> JobObservation:
    return JobObservation(name=name, status=status, conclusion=conclusion)


def _run(
    run_id: int,
    *,
    hours_ago: float,
    jobs: list[JobObservation],
    status: str = "completed",
    conclusion: str | None = None,
) -> RunObservation:
    # `conclusion` defaults to what the jobs imply, which is what GitHub
    # reports: any failing job makes the run fail. The state-chain tests
    # pass it explicitly, because the whole point there is that the two can
    # disagree -- a run can be red while every data job in it succeeded.
    if conclusion is None and status == "completed":
        conclusion = (
            "failure" if any(j.conclusion in FAILING_CONCLUSIONS for j in jobs) else "success"
        )
    return RunObservation(
        run_id=run_id,
        created_at=NOW - timedelta(hours=hours_ago),
        status=status,
        jobs=tuple(jobs),
        conclusion=conclusion,
    )


def _ok_run(run_id: int, *, hours_ago: float, job: str = "eod") -> RunObservation:
    return _run(
        run_id,
        hours_ago=hours_ago,
        jobs=[
            _job("resolve-mode", "success"),
            _job(job, "success"),
            _job(WATCHDOG_JOB, "success"),
        ],
    )


# -- bug 1: an in-progress run counted as a failure -----------------------


def test_a_run_still_in_progress_is_not_an_incident() -> None:
    # Observed on run 39999355264: `gh run list --json conclusion` prints
    # the empty string for a run that has not finished, not null, so the
    # filter `.conclusion != null` let it through and the watchdog failed
    # over its own run -- whose `external` job went on to succeed 65
    # minutes later.
    runs = [
        _run(
            1,
            hours_ago=0.05,
            status="in_progress",
            jobs=[_job("external", "", status="in_progress")],
        )
    ]

    verdict = evaluate(runs, now=NOW)

    assert not verdict.alarm
    assert verdict.skipped_in_progress == (1,)
    assert verdict.incidents == ()


def test_an_unfinished_job_inside_a_finished_run_is_not_an_incident() -> None:
    # The same empty-string trap one level down.
    runs = [
        _run(
            1,
            hours_ago=2,
            jobs=[_job("eod", "success"), _job("external", None, status="in_progress")],
        )
    ]

    assert not evaluate(runs, now=NOW).alarm


def test_the_current_run_is_never_judged() -> None:
    # Belt and braces for the same bug: even a run that somehow reports
    # itself completed must not be able to alarm about itself.
    runs = [_run(99, hours_ago=0, jobs=[_job("external", "cancelled")])]

    assert not evaluate(runs, now=NOW, current_run_id=99).alarm
    assert evaluate(runs, now=NOW, current_run_id=None).alarm


# -- bug 2: the watchdog's own failure read as a data incident ------------


def test_a_failed_watchdog_is_not_itself_a_data_incident() -> None:
    # Otherwise one false alarm becomes an unbroken chain of them: the
    # watchdog fails, which fails the run, so the next run sees a
    # non-successful run and fails too, each link looking like a fresh
    # data incident.
    runs = [
        _run(
            1,
            hours_ago=2,
            jobs=[
                _job("resolve-mode", "success"),
                _job("eod", "success"),
                _job(WATCHDOG_JOB, "failure"),
            ],
        )
    ]

    verdict = evaluate(runs, now=NOW)

    assert verdict.incidents == ()
    # The run is red, so its packed state is out of the restore's reach --
    # which is a separate, real finding, and the reason the watchdog no
    # longer runs inside pipeline.yml at all.
    assert verdict.state_chain.stale


def test_the_watchdog_job_is_not_monitored() -> None:
    assert WATCHDOG_JOB not in MONITORED_JOBS


def test_a_chain_of_watchdog_failures_never_compounds() -> None:
    # Ten consecutive runs in which only the watchdog failed. Under the old
    # run-level rule each one would alarm about the previous.
    runs = [
        _run(
            i,
            hours_ago=float(i),
            jobs=[_job("scan", "success"), _job(WATCHDOG_JOB, "failure")],
        )
        for i in range(1, 11)
    ]

    assert evaluate(runs, now=NOW).incidents == ()


# -- bug 3: the same incident re-reported on every later run --------------


def test_an_incident_already_reported_is_not_reported_again() -> None:
    # Run 2 failed its eod job. Run 1's watchdog alarmed about it. The next
    # run must not alarm about it a second time -- 19 cancelled nights
    # would otherwise produce an alarm on all ~45 runs of the following day
    # and a half, which is the alert fatigue that makes the 20th silence go
    # unnoticed.
    runs = [
        _run(2, hours_ago=10, jobs=[_job("eod", "cancelled"), _job(WATCHDOG_JOB, "success")]),
        _run(
            1,
            hours_ago=8,
            jobs=[_job("scan", "success"), _job(WATCHDOG_JOB, "failure")],
            conclusion="success",
        ),
    ]

    verdict = evaluate(runs, now=NOW)

    assert not verdict.alarm
    assert len(verdict.already_reported) == 1
    assert verdict.already_reported[0].job == "eod"
    assert verdict.last_alarm_at == NOW - timedelta(hours=8)


def test_a_new_incident_after_an_alarm_still_reports() -> None:
    # Suppression must be bounded by time, not by "an alarm happened once".
    runs = [
        _run(2, hours_ago=10, jobs=[_job("eod", "cancelled"), _job(WATCHDOG_JOB, "success")]),
        _run(1, hours_ago=8, jobs=[_job("scan", "success"), _job(WATCHDOG_JOB, "failure")]),
        _run(3, hours_ago=2, jobs=[_job("eod", "cancelled"), _job(WATCHDOG_JOB, "success")]),
    ]

    verdict = evaluate(runs, now=NOW)

    assert verdict.alarm
    assert [i.run_id for i in verdict.incidents] == [3]


def test_the_alarming_run_s_own_data_jobs_are_still_judged() -> None:
    # The watchdog runs in parallel with the data jobs, deliberately, so a
    # watchdog outage cannot stop a scan. That means it never judged its own
    # run's data jobs, so the next evaluation is the first that can -- and
    # the de-duplication window must not swallow them.
    runs = [
        _run(
            1,
            hours_ago=8,
            jobs=[_job("external", "cancelled"), _job(WATCHDOG_JOB, "failure")],
        )
    ]

    verdict = evaluate(runs, now=NOW)

    assert verdict.alarm
    assert [(i.run_id, i.job) for i in verdict.incidents] == [(1, "external")]


def test_each_fresh_nightly_failure_is_its_own_incident() -> None:
    # A failure that recurs every night is a new incident every night, even
    # though the previous one was reported. Suppressing those is how a
    # 25-day outage stays quiet.
    runs = [
        # Night one fails. Its own watchdog ran in parallel and so did not
        # judge it.
        _run(1, hours_ago=30, jobs=[_job("eod", "cancelled"), _job(WATCHDOG_JOB, "success")]),
        # The next run alarms about night one.
        _run(2, hours_ago=20, jobs=[_job("scan", "success"), _job(WATCHDOG_JOB, "failure")]),
        # Night two fails as well. A separate night, a separate incident.
        _run(3, hours_ago=6, jobs=[_job("eod", "cancelled"), _job(WATCHDOG_JOB, "success")]),
    ]

    verdict = evaluate(runs, now=NOW)

    assert verdict.alarm
    assert [i.run_id for i in verdict.incidents] == [3]
    assert [i.run_id for i in verdict.already_reported] == [1]


# -- what must keep working ----------------------------------------------


def test_a_timed_out_data_job_alarms() -> None:
    # The whole purpose. A job timeout is reported as `cancelled`, and
    # GitHub does not email about a cancelled run -- that is why 19 of them
    # went unnoticed for 25 days from 2026-09-29.
    runs = [_run(1, hours_ago=3, jobs=[_job("eod", "cancelled")])]

    verdict = evaluate(runs, now=NOW)

    assert verdict.alarm
    assert verdict.incidents[0].conclusion == "cancelled"


def test_cancelled_counts_as_failing() -> None:
    assert "cancelled" in FAILING_CONCLUSIONS
    assert "timed_out" in FAILING_CONCLUSIONS
    assert "success" not in FAILING_CONCLUSIONS
    assert "skipped" not in FAILING_CONCLUSIONS


def test_every_data_job_and_the_mode_gate_are_monitored() -> None:
    # If a mode is added to the pipeline and not here, its failures are
    # invisible. `resolve-mode` is monitored because every data job is
    # gated on its output.
    assert {
        "resolve-mode",
        "scan",
        "eod",
        "external",
        "research",
        "weekly",
        "monthly",
    } == MONITORED_JOBS


def test_a_skipped_mode_job_is_not_an_incident() -> None:
    # Only one mode job runs per run; the rest are skipped, which is the
    # normal case and not a failure.
    runs = [
        _run(
            1,
            hours_ago=1,
            jobs=[
                _job("resolve-mode", "success"),
                _job("scan", "success"),
                _job("eod", "skipped"),
                _job("external", "skipped"),
                _job("weekly", "skipped"),
            ],
        )
    ]

    assert not evaluate(runs, now=NOW).alarm


def test_a_failed_mode_gate_alarms() -> None:
    # If resolve-mode fails, the run does nothing at all -- which looks
    # like silence, not like a failure, unless it is monitored.
    runs = [_run(1, hours_ago=1, jobs=[_job("resolve-mode", "failure")])]

    assert evaluate(runs, now=NOW).alarm


def test_an_incident_outside_the_window_is_not_reported() -> None:
    runs = [_run(1, hours_ago=40, jobs=[_job("eod", "cancelled")])]

    assert not evaluate(runs, now=NOW).alarm


def test_no_runs_at_all_is_not_an_alarm() -> None:
    # An empty history means the watchdog learned nothing, which is not the
    # same as learning that something broke.
    verdict = evaluate([], now=NOW)

    assert not verdict.alarm
    assert "no completed runs" in verdict.reason


def test_a_healthy_history_is_not_an_alarm() -> None:
    runs = [_ok_run(i, hours_ago=float(i) * 3) for i in range(1, 8)]

    verdict = evaluate(runs, now=NOW)

    assert not verdict.alarm
    assert verdict.considered_runs == 7


# -- the data contract with `gh` ------------------------------------------


def test_parse_runs_accepts_both_gh_shapes() -> None:
    # `gh run list --json` prints camelCase `databaseId`/`createdAt`;
    # `gh api` returns snake_case `id`/`created_at`. The workflow uses both.
    parsed = parse_runs(
        [
            {
                "databaseId": 1,
                "createdAt": "2026-10-24T10:00:00Z",
                "status": "completed",
                "jobs": [{"name": "eod", "status": "completed", "conclusion": "cancelled"}],
            },
            {
                "id": 2,
                "created_at": "2026-10-24T11:00:00Z",
                "status": "in_progress",
                "jobs": [{"name": "external", "status": "in_progress", "conclusion": ""}],
            },
        ]
    )

    assert [r.run_id for r in parsed] == [1, 2]
    assert parsed[0].jobs[0].failed
    assert parsed[1].jobs[0].conclusion is None
    assert not parsed[1].jobs[0].completed


def test_parse_runs_ignores_malformed_entries() -> None:
    # A watchdog that crashes on a surprise in someone else's API response
    # is a watchdog that is not watching.
    assert parse_runs([None, "nonsense", {}, {"databaseId": 5}]) == []


def test_main_exits_zero_when_healthy(capsys, monkeypatch) -> None:
    document = json.dumps(
        {
            "now": "2026-10-24T12:00:00Z",
            "current_run_id": 99,
            "runs": [
                {
                    "databaseId": 1,
                    "createdAt": "2026-10-24T10:00:00Z",
                    "status": "completed",
                    "jobs": [{"name": "eod", "status": "completed", "conclusion": "success"}],
                }
            ],
        }
    )
    monkeypatch.setattr("sys.stdin", _Stdin(document))

    assert main([]) == 0
    assert "::error::" not in capsys.readouterr().out


def test_main_exits_one_and_annotates_on_alarm(capsys, monkeypatch) -> None:
    document = json.dumps(
        {
            "now": "2026-10-24T12:00:00Z",
            "current_run_id": 99,
            "runs": [
                {
                    "databaseId": 1,
                    "createdAt": "2026-10-24T10:00:00Z",
                    "status": "completed",
                    "jobs": [{"name": "eod", "status": "completed", "conclusion": "cancelled"}],
                }
            ],
        }
    )
    monkeypatch.setattr("sys.stdin", _Stdin(document))

    assert main([]) == 1
    out = capsys.readouterr().out
    assert "::error::" in out
    assert "cancelled" in out


def test_main_survives_an_empty_input(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", _Stdin(""))

    assert main([]) == 0


def test_the_summary_names_the_suppression_when_it_applies() -> None:
    runs = [
        _run(2, hours_ago=10, jobs=[_job("eod", "cancelled"), _job(WATCHDOG_JOB, "success")]),
        _run(1, hours_ago=8, jobs=[_job("scan", "success"), _job(WATCHDOG_JOB, "failure")]),
    ]

    summary = markdown_summary(evaluate(runs, now=NOW))

    assert "already reported" in summary
    assert "last watchdog alarm" in summary


class _Stdin:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> str:
        return self._text


# -- the observed cascade, replayed --------------------------------------

#: The job conclusions actually recorded for pipeline.yml on 2026-10-23/24,
#: read from the Actions API on 2026-10-24. Eight consecutive runs are
#: reported as `failure` at run level, and in every one of them *every data
#: job succeeded* -- the only failing job is the watchdog itself. One
#: genuine incident exists in the window: `external` was cancelled at
#: 15:30. That is the false-alarm chain this module was rewritten to end,
#: and the single real event it must still catch.
OBSERVED_RUNS: list[tuple[int, str, list[tuple[str, str]]]] = [
    (39956274493, "2026-10-23T12:49:40Z", [("resolve-mode", "success"), ("scan", "success")]),
    (
        39968420761,
        "2026-10-23T14:17:53Z",
        [("resolve-mode", "success"), ("external", "success"), (WATCHDOG_JOB, "failure")],
    ),
    (
        39973892098,
        "2026-10-23T14:56:11Z",
        [("resolve-mode", "success"), ("scan", "success"), (WATCHDOG_JOB, "failure")],
    ),
    (
        39978874688,
        "2026-10-23T15:30:36Z",
        [("resolve-mode", "success"), ("external", "cancelled"), (WATCHDOG_JOB, "failure")],
    ),
    (
        39999355264,
        "2026-10-23T17:57:40Z",
        [("resolve-mode", "success"), ("scan", "success"), (WATCHDOG_JOB, "failure")],
    ),
    (
        40012410917,
        "2026-10-23T19:34:03Z",
        [("resolve-mode", "success"), ("scan", "success"), (WATCHDOG_JOB, "failure")],
    ),
    (
        40025944915,
        "2026-10-23T21:19:43Z",
        [("resolve-mode", "success"), ("scan", "success"), (WATCHDOG_JOB, "failure")],
    ),
    (
        40042567854,
        "2026-10-23T23:58:19Z",
        [("resolve-mode", "success"), ("eod", "success"), (WATCHDOG_JOB, "failure")],
    ),
    (
        40047953810,
        "2026-10-24T01:02:41Z",
        [("resolve-mode", "success"), ("external", "success"), (WATCHDOG_JOB, "failure")],
    ),
]


def _observed(*, before: str | None = None) -> list[RunObservation]:
    return [
        RunObservation(
            run_id=run_id,
            created_at=datetime.fromisoformat(created.replace("Z", "+00:00")),
            status="completed",
            jobs=tuple(_job(name, conclusion) for name, conclusion in jobs),
            # As GitHub reported them: every one of these runs except the
            # first was red, and all of them had packed state.
            conclusion="success" if run_id == 39956274493 else "failure",
        )
        for run_id, created, jobs in OBSERVED_RUNS
        if before is None or created < before
    ]


def test_the_observed_cascade_produces_no_data_incident() -> None:
    # Replayed at the moment run 40047953810's watchdog ran. The old rule
    # gave eight consecutive red runs here; seven of them said nothing
    # except that the watchdog had failed. None of them contains a failed
    # data job, so none is reported as one.
    verdict = evaluate(
        _observed(),
        now=datetime(2026, 10, 24, 1, 3, tzinfo=UTC),
        current_run_id=40047953810,
    )

    assert verdict.incidents == (), [i.describe() for i in verdict.incidents]
    assert verdict.already_reported, "the real external cancel should still be accounted for"
    # It does alarm, but for the right reason and only once: every one of
    # those red runs packed state the restore cannot reach. See
    # `test_the_observed_cascade_would_have_been_caught_as_a_hidden_state_chain`.
    assert verdict.state_chain.stale


def test_the_observed_external_cancel_is_detected_once() -> None:
    # The one genuine incident in that window. Replayed at the moment the
    # next run's watchdog would have looked.
    verdict = evaluate(
        _observed(before="2026-10-23T16:00:00Z"),
        now=datetime(2026, 10, 23, 16, 0, tzinfo=UTC),
        current_run_id=39973892098,
    )

    assert verdict.alarm
    assert [(i.run_id, i.job, i.conclusion) for i in verdict.incidents] == [
        (39978874688, "external", "cancelled")
    ]


def test_no_data_job_failed_in_the_observed_cascade() -> None:
    # Pins the finding itself, so the fixture cannot be misread later as
    # evidence that the data path was broken. It was not: the watchdog was.
    data_failures = [
        (run_id, name, conclusion)
        for run_id, _, jobs in OBSERVED_RUNS
        for name, conclusion in jobs
        if name in MONITORED_JOBS and conclusion in FAILING_CONCLUSIONS
    ]

    assert data_failures == [(39978874688, "external", "cancelled")]


# -- the pipeline not running at all -------------------------------------


def test_a_pipeline_that_has_stopped_running_alarms() -> None:
    # The one incident a watchdog triggered only by pipeline runs could
    # never see: if the cron stops firing, nothing calls the watchdog, so
    # total silence is the one state it cannot report. Not hypothetical --
    # on 2026-09-14 only one of three scheduled scans fired, and that one
    # 37 minutes late.
    history = [_ok_run(1, hours_ago=60), _ok_run(2, hours_ago=80)]

    verdict = evaluate([], now=NOW, history=history)

    assert verdict.alarm
    assert verdict.stall.stalled
    assert verdict.stall.hours_since is not None and verdict.stall.hours_since > 36
    assert "has not run since" in verdict.reason


def test_a_repository_that_has_never_run_does_not_alarm() -> None:
    # No history is not evidence that something broke. A first run would
    # otherwise alarm about itself.
    verdict = evaluate([], now=NOW, history=[])

    assert not verdict.alarm
    assert not verdict.stall.stalled
    assert verdict.stall.last_run_at is None


def test_a_recently_run_pipeline_is_not_stalled() -> None:
    history = [_ok_run(1, hours_ago=2), _ok_run(2, hours_ago=40)]

    verdict = evaluate([_ok_run(1, hours_ago=2)], now=NOW, history=history)

    assert not verdict.alarm
    assert not verdict.stall.stalled


def test_a_stall_is_reported_even_with_no_job_incidents() -> None:
    # The window has nothing in it precisely because nothing ran. An
    # incident count of zero must not be read as health.
    history = [_ok_run(1, hours_ago=50)]

    verdict = evaluate([], now=NOW, history=history)

    assert verdict.incidents == ()
    assert verdict.alarm
    assert "stopped running" in markdown_summary(verdict)


def test_a_stall_annotates_and_exits_one(capsys, monkeypatch) -> None:
    document = json.dumps(
        {
            "now": "2026-10-24T12:00:00Z",
            "runs": [],
            "history": [
                {
                    "databaseId": 1,
                    "createdAt": "2026-10-21T10:00:00Z",
                    "status": "completed",
                    "jobs": [{"name": "eod", "status": "completed", "conclusion": "success"}],
                }
            ],
        }
    )
    monkeypatch.setattr("sys.stdin", _Stdin(document))

    assert main([]) == 1
    out = capsys.readouterr().out
    assert "stopped running" in out
    assert "::error::" in out


def test_the_window_bound_still_applies_to_incidents_during_a_stall() -> None:
    # A stall and an old incident together: the stall is the finding, and
    # the old incident is not re-reported on top of it.
    history = [_ok_run(1, hours_ago=50)]
    runs = [_run(1, hours_ago=50, jobs=[_job("eod", "cancelled")])]

    verdict = evaluate(runs, now=NOW, history=history)

    assert verdict.alarm
    assert verdict.stall.stalled
    assert verdict.incidents == ()


# -- the state chain -----------------------------------------------------


def test_runs_happening_but_none_succeeding_alarms() -> None:
    # `.github/actions/turboedge-state` restores from
    # `gh run list --workflow pipeline.yml --status success`, so a run that
    # went red for ANY reason takes its packed state out of reach -- even
    # when every data job in it succeeded. The symptom is visible without
    # knowing the cause: runs are happening and none of them succeeds.
    runs = [
        _run(i, hours_ago=float(i * 2), jobs=[_job("scan", "success")], conclusion="failure")
        for i in range(1, 6)
    ]

    verdict = evaluate(runs, now=NOW)

    assert verdict.alarm
    assert verdict.state_chain.stale
    assert verdict.incidents == (), "no data job failed; the finding is the hidden state"
    assert "cannot reach" in verdict.reason
    assert len(verdict.state_chain.unreachable_runs) == 5


def test_the_observed_cascade_would_have_been_caught_as_a_hidden_state_chain() -> None:
    # The incident this check exists for, replayed on the recorded data.
    # Eight consecutive red runs, every data job in them successful, every
    # one of them having uploaded `turboedge-state-enc`, and the newest
    # reachable state from 12:49 that day.
    verdict = evaluate(
        _observed(),
        now=datetime(2026, 10, 24, 1, 3, tzinfo=UTC),
        current_run_id=40047953810,
    )

    assert verdict.state_chain.stale
    assert verdict.state_chain.last_success_at == datetime(2026, 10, 23, 12, 49, 40, tzinfo=UTC)
    assert verdict.alarm
    # Every red run after the last success did successful data work.
    assert set(verdict.state_chain.unreachable_runs) == {
        39968420761,
        39973892098,
        39978874688,
        39999355264,
        40012410917,
        40025944915,
        40042567854,
    }


def test_a_red_run_older_than_the_last_success_is_not_flagged() -> None:
    # The chain only needs to be reachable. A red run whose work a later
    # successful run already carried forward is not a loss: the restore
    # picks that later run, which built on the same state.
    runs = [
        _run(1, hours_ago=20, jobs=[_job("scan", "success")], conclusion="failure"),
        _run(2, hours_ago=10, jobs=[_job("scan", "success")], conclusion="success"),
    ]

    verdict = evaluate(runs, now=NOW)

    assert not verdict.state_chain.stale
    assert not verdict.alarm


def test_a_red_run_newer_than_the_last_success_is_flagged() -> None:
    # Its packed state is the newest there is, and the restore cannot see
    # it. This is the exact shape of the 2026-10-23 incident.
    runs = [
        _run(1, hours_ago=10, jobs=[_job("scan", "success")], conclusion="success"),
        _run(2, hours_ago=2, jobs=[_job("external", "success")], conclusion="failure"),
    ]

    verdict = evaluate(runs, now=NOW)

    assert verdict.state_chain.stale
    assert verdict.state_chain.unreachable_runs == (2,)


def test_absent_conclusions_do_not_alarm() -> None:
    # A watchdog that alarms when it cannot see is one that cries wolf the
    # first time a field is renamed upstream.
    runs = [
        RunObservation(
            run_id=1,
            created_at=NOW - timedelta(hours=2),
            status="completed",
            jobs=(_job("scan", "success"),),
            conclusion=None,
        )
    ]

    verdict = evaluate(runs, now=NOW)

    assert not verdict.state_chain.stale
    assert not verdict.alarm


def test_a_stalled_pipeline_is_not_also_reported_as_a_hidden_chain() -> None:
    # With nothing completing in the window the finding is the stall. Two
    # alarms for one cause is noise.
    history = [_ok_run(1, hours_ago=50)]

    verdict = evaluate([], now=NOW, history=history)

    assert verdict.stall.stalled
    assert not verdict.state_chain.stale


def test_the_summary_names_the_hidden_state_chain() -> None:
    runs = [
        _run(1, hours_ago=2, jobs=[_job("scan", "success")], conclusion="failure"),
    ]

    summary = markdown_summary(evaluate(runs, now=NOW))

    assert "Successful data work is unreachable" in summary
