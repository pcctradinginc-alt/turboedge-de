"""SUCCESS, INSUFFICIENT_DATA and ERROR must be distinguishable.

Until 2026-10-25 the weekly MAE/MFE step carried a blanket
`continue-on-error: true`, and `research excursion` exited 1 both when the
forward ledger had nothing matured and when something had genuinely broken.
A technical fault in the only research harness this system has was
therefore indistinguishable from a young ledger -- and the ledger *is*
young, so the fault would have been the one nobody looked for.

The exit code now carries the distinction:

    0  SUCCESS            a valid evaluation ran, whatever its verdict
    3  INSUFFICIENT_DATA  working, not enough matured observations yet
    1  ERROR              a technical fault

These tests pin all three, and that the workflow step fails on the third
while accepting the second.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import typer
import yaml
from typer.testing import CliRunner

from turboedge.cli import app
from turboedge.cli_learn import EXCURSION_INSUFFICIENT_DATA
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Category,
    Direction,
    ExitReason,
    LedgerEntry,
    LedgerEntryStatus,
    LedgerLabel,
    UnderlyingBar,
)

_NOW = datetime(2026, 9, 20, 20, 0, tzinfo=UTC)
WORKFLOW = Path(".github/workflows/pipeline.yml")

#: The five entry-side features `build_excursion_dataset` reads from
#: `feature_snapshot`. An entry missing any of them is dropped as
#: `null_feature` however good its label is.
_FEATURES = {
    "cost_rank_score": 0.1,
    "leverage": 2.0,
    "premium_over_fair": 0.0,
    "signal_score": 0.2,
    "spread_pct": 0.001,
}


def _entry(index: int, *, day: int) -> LedgerEntry:
    prediction_time = datetime(2026, 9, day, 8, 0, tzinfo=UTC)
    return LedgerEntry(
        run_id=f"run-{index}",
        candidate_id=f"cand-{index}",
        signal_id="sig-1",
        signal_version_hash="h",
        trial_id="TR-test",
        prediction_time=prediction_time,
        underlying="DAX",
        direction=Direction.LONG,
        horizon_days=3,
        regime_bucket=None,
        cluster_id=None,
        feature_hash="fh",
        model_hash="mh",
        config_hash="ch",
        git_commit=None,
        category=Category.WATCH,
        selected_wkn=None,
        selected_isin=f"DE000EXC{index:04d}",
        issuer="BankA",
        entry_bid=4.98,
        entry_ask=5.0,
        entry_spread=0.004,
        entry_quote_timestamp=prediction_time,
        entry_underlying_timestamp=prediction_time,
        financing_level_entry=18_000.0,
        barrier_entry=18_000.0,
        ratio=0.01,
        fx=1.0,
        predicted_return=0.0,
        p_profit=0.5,
        p_ko=0.05,
        expected_shortfall=-0.1,
        lcb_ev=0.0,
        uncertainty=0.02,
        shrinkage_intensity=0.5,
        is_shadow=False,
        shadow_stratum=None,
        suggested_position_fraction=None,
        exit_due=date(2026, 9, day) + timedelta(days=3),
        alternatives=[],
        feature_snapshot=dict(_FEATURES),
        status=LedgerEntryStatus.OPEN,
    )


def _known_label(entry_id: str, *, mae: float, mfe: float, at: datetime) -> LedgerLabel:
    """A label with MAE and MFE chosen by the test, not computed."""
    return LedgerLabel(
        entry_id=entry_id,
        labeled_at=at,
        exit_bid=5.1,
        exit_quote_timestamp=at,
        financing_level_exit=18_000.0,
        exit_reason=ExitReason.HORIZON,
        realized_selected_pnl=0.02,
        underlying_pnl=0.005,
        median_turbo_pnl=0.01,
        best_turbo_pnl=0.03,
        ideal_turbo_pnl=0.04,
        mfe=mfe,
        mae=mae,
        ko_hit=False,
        time_to_ko_days=None,
        ambiguous_path=False,
    )


@pytest.fixture
def seeded(tmp_path: Path) -> Callable[[int], Path]:
    """A state directory whose ledger holds `n` labelled entries."""

    def _seed(n: int) -> Path:
        state_dir = tmp_path / f"state_{n}"
        state_dir.mkdir(parents=True, exist_ok=True)
        with Store(state_dir / "turboedge.duckdb") as store:
            store.init_schema()
            entries = [_entry(i, day=1 + (i % 15)) for i in range(n)]
            store.append_ledger_entries(entries)
            for i, entry in enumerate(entries):
                store.attach_ledger_label(
                    _known_label(
                        entry.entry_id,
                        # Deliberately varied and known, so the dataset's
                        # targets can be checked against the inputs.
                        mae=-0.01 * (1 + i % 5),
                        mfe=0.02 * (1 + i % 5),
                        at=datetime(2026, 9, 1 + (i % 15), 20, 0, tzinfo=UTC) + timedelta(days=3),
                    )
                )
            store.append_underlying_bars(
                [
                    UnderlyingBar(
                        underlying_id="DAX",
                        ts=datetime(2026, 9, day, tzinfo=UTC),
                        interval="1d",
                        open=18_000.0,
                        high=18_200.0,
                        low=17_900.0,
                        close=18_100.0,
                        volume=1000.0,
                        observation_time=datetime(2026, 9, day, tzinfo=UTC),
                        available_at=datetime(2026, 9, day, 22, 0, tzinfo=UTC),
                        retrieved_at=datetime(2026, 9, day, 22, 0, tzinfo=UTC),
                        source="fake",
                        parser_version="1",
                        is_stale=False,
                        quality_score=1.0,
                    )
                    for day in range(1, 25)
                ]
            )
        return state_dir

    return _seed


def _run(state_dir: Path) -> Any:
    return CliRunner().invoke(
        app, ["--state-dir", str(state_dir), "research", "excursion"], catch_exceptions=False
    )


# -- the three outcomes ---------------------------------------------------


def test_an_empty_ledger_reports_insufficient_data_not_an_error(
    seeded: Callable[[int], Path],
) -> None:
    result = _run(seeded(0))

    assert result.exit_code == EXCURSION_INSUFFICIENT_DATA
    assert "INSUFFICIENT_DATA" in result.stdout
    assert "ERROR" not in result.stdout


def test_a_few_labelled_entries_still_report_insufficient_data(
    seeded: Callable[[int], Path],
) -> None:
    # The evaluation runs and reaches a verdict of INSUFFICIENT_SAMPLE.
    # Reported as INSUFFICIENT_DATA rather than as a success, so "it ran" is
    # not mistaken for "it measured something".
    result = _run(seeded(6))

    assert result.exit_code == EXCURSION_INSUFFICIENT_DATA


def test_a_technical_fault_is_reported_as_an_error(
    seeded: Callable[[int], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The case the blanket `continue-on-error` hid. A fault must exit 1, not
    # the data-shortage code.
    import turboedge.cli_learn as cli_learn

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("the splitter is misconfigured")

    monkeypatch.setattr(cli_learn, "evaluate_excursion", _boom)
    result = _run(seeded(400))

    assert result.exit_code == 1
    assert "ERROR" in result.stdout
    assert result.exit_code != EXCURSION_INSUFFICIENT_DATA


def test_a_dataset_build_fault_is_also_an_error(
    seeded: Callable[[int], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import turboedge.cli_learn as cli_learn

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise KeyError("column vanished")

    monkeypatch.setattr(cli_learn, "build_excursion_dataset", _boom)
    result = _run(seeded(10))

    assert result.exit_code == 1
    assert "ERROR" in result.stdout


def test_an_overdue_entry_makes_a_data_shortage_an_error(
    tmp_path: Path,
) -> None:
    # The shortage is only legitimate while nothing is overdue. An entry
    # past its exit_due means the labeling loop is at fault, and reporting
    # that as "waiting for data" would hide it behind the one state that is
    # expected to persist for months.
    state_dir = tmp_path / "overdue"
    state_dir.mkdir()
    with Store(state_dir / "turboedge.duckdb") as store:
        store.init_schema()
        store.append_ledger_entries([_entry(1, day=1)])  # exit_due 2026-09-04

    result = _run(state_dir)

    assert result.exit_code == 1, (
        "an overdue unlabelled entry is a labeling fault, not a maturity question"
    )


# -- the known-value integration test ------------------------------------


def test_the_dataset_carries_the_mae_and_mfe_it_was_given(
    seeded: Callable[[int], Path],
) -> None:
    # Known inputs, checked outputs: the targets must arrive exactly as
    # recorded, and the effective sample must count distinct dates rather
    # than rows.
    from turboedge.backtest.excursion_eval import build_excursion_dataset

    state_dir = seeded(15)
    with Store(state_dir / "turboedge.duckdb") as store:
        dataset = build_excursion_dataset(store, underlying="DAX", as_of=_NOW)

    assert len(dataset.y_mae) == 15
    assert set(dataset.y_mae) == {-0.01, -0.02, -0.03, -0.04, -0.05}
    assert set(dataset.y_mfe) == {0.02, 0.04, 0.06, 0.08, 0.10}
    # 15 entries spread over 15 distinct prediction dates.
    assert dataset.n_distinct_dates == 15
    # Effective sample accounts for the overlapping 3-day label windows and
    # is therefore below the row count, never equal to it.
    assert 0 < dataset.effective_sample <= 15
    assert dataset.effective_sample < len(dataset.y_mae)


def test_a_knocked_out_entry_keeps_its_label_and_is_usable(
    seeded: Callable[[int], Path],
) -> None:
    # A knock-out is an outcome, not an error: the position lost its value
    # and that is exactly what the excursion research is about.
    from turboedge.backtest.excursion_eval import build_excursion_dataset

    state_dir = seeded(0)
    with Store(state_dir / "turboedge.duckdb") as store:
        entry = _entry(99, day=2)
        store.append_ledger_entries([entry])
        store.attach_ledger_label(
            _known_label(entry.entry_id, mae=-1.0, mfe=0.0, at=_NOW).model_copy(
                update={"ko_hit": True, "time_to_ko_days": 2, "exit_reason": ExitReason.KO}
            )
        )
        dataset = build_excursion_dataset(store, underlying="DAX", as_of=_NOW)

    assert len(dataset.y_mae) == 1
    assert dataset.y_mae[0] == pytest.approx(-1.0)


def test_an_ambiguous_path_is_excluded_rather_than_resolved(
    seeded: Callable[[int], Path],
) -> None:
    # CLAUDE.md rule 17: an ambiguous bar is never resolved optimistically.
    # Here that means the observation is dropped, not guessed at.
    from turboedge.backtest.excursion_eval import build_excursion_dataset

    state_dir = seeded(0)
    with Store(state_dir / "turboedge.duckdb") as store:
        entry = _entry(98, day=2)
        store.append_ledger_entries([entry])
        store.attach_ledger_label(
            _known_label(entry.entry_id, mae=-0.02, mfe=0.03, at=_NOW).model_copy(
                update={"ambiguous_path": True}
            )
        )
        with pytest.raises(ValueError, match="ambiguous_path"):
            build_excursion_dataset(store, underlying="DAX", as_of=_NOW)


# -- the workflow must act on the distinction ----------------------------


def test_the_weekly_step_no_longer_swallows_every_outcome() -> None:
    if not WORKFLOW.is_file():
        pytest.skip(f"{WORKFLOW} not present")
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    step = next(
        s
        for s in workflow["jobs"]["weekly"]["steps"]
        if "excursion" in str(s.get("name", "")).lower()
    )

    assert step.get("continue-on-error") is not True, (
        "the weekly excursion step swallows every outcome again, so a technical "
        "fault in the only research harness looks like a young ledger"
    )
    run = str(step["run"])
    assert "-eq 3" in run, "the step no longer recognises the INSUFFICIENT_DATA code"
    assert "::error::" in run, "a technical fault must annotate and fail"
    assert 'exit "${status}"' in run, "a technical fault must propagate"


def test_the_insufficient_data_code_is_not_one() -> None:
    # If it were 1 it could not be told from a fault, which is the entire
    # point of having a separate code.
    assert EXCURSION_INSUFFICIENT_DATA != 1
    assert EXCURSION_INSUFFICIENT_DATA != 0
    assert isinstance(typer.Exit(code=EXCURSION_INSUFFICIENT_DATA).exit_code, int)
