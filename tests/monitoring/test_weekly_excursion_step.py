"""The weekly MAE/MFE step's exit-code handling, run as the shell runs it.

WHY THIS TEST EXISTS AT ALL
---------------------------
On 2026-10-31 the scheduled weekly run (41197107832) died at this step on
exit code 3 -- INSUFFICIENT_DATA, a normal data shortage -- and took the
research tournament, the queue re-score, the KO-calibration and the
**state upload** down with it. The step already contained the branch that
handles exit 3. It never ran.

The cause is that Actions invokes every ``run:`` block as
``bash --noprofile --norc -e -o pipefail {0}``. Errexit comes from the
*invocation*, so ``set -uo pipefail`` inside the script cannot turn it off
-- it only adds nounset and pipefail. A bare command followed by
``status=$?`` is therefore dead code on any non-zero exit.

HOW IT IS TESTED
----------------
The script is read out of ``pipeline.yml`` rather than copied here, so
this cannot pass while the real step is broken, and it is executed under
the same shell flags Actions uses with a stub ``uv`` that exits with a
chosen code. The repository's own doctrine is that a decision embedded in
YAML is untestable and belongs in Python
(``monitoring/pipeline_health.py`` exists for that reason); this is the
narrow exception where the decision cannot move out, because it is about
the step's own exit status.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

#: Exactly how GitHub Actions runs a `run:` block on Linux. Errexit is
#: part of it, which is the whole point.
_ACTIONS_SHELL = ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail"]

_STEP_NAME = "Measure MAE/MFE excursion prediction"


def _weekly_excursion_script() -> str:
    """The step's script, read from the workflow that actually runs it."""
    workflow = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / ".github" / "workflows" / "pipeline.yml").read_text()
    )
    steps = workflow["jobs"]["weekly"]["steps"]
    for step in steps:
        if step.get("name") == _STEP_NAME:
            return str(step["run"])
    raise AssertionError(f"step {_STEP_NAME!r} not found in pipeline.yml's weekly job")


def _run_step(script: str, *, exit_code: int, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Execute the step with a stub `uv` that exits with `exit_code`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "uv"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'echo "stub uv: $*"\n'
        # Mimic the real command: it prints its verdict and writes its
        # report before exiting, so a non-zero exit here is not an absence
        # of output.
        'echo "EXCURSION INSUFFICIENT_SAMPLE  (all underlyings)"\n'
        f"exit {exit_code}\n"
    )
    stub.chmod(0o755)

    script_path = tmp_path / "step.sh"
    script_path.write_text(script)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(
        [*_ACTIONS_SHELL, str(script_path)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )


def test_the_script_captures_the_status_in_a_compound_list() -> None:
    """A structural check, so the defect cannot come back by refactoring.

    `cmd || status=$?` is errexit-exempt because the command sits on the
    left of `||`. A bare `cmd` followed by `status=$?` is not, and that is
    what failed run 41197107832.
    """
    script = _weekly_excursion_script()

    assert "|| status=$?" in script, "the command's status must be captured, not assumed"
    assert "\n          status=$?" not in script, (
        "a bare `status=$?` on its own line is unreachable under errexit"
    )


def test_exit_zero_is_a_successful_evaluation(tmp_path: Path) -> None:
    result = _run_step(_weekly_excursion_script(), exit_code=0, tmp_path=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "a valid evaluation ran" in result.stdout
    assert "::notice::" not in result.stdout
    assert "::error::" not in result.stdout


def test_exit_three_is_a_data_shortage_and_does_not_fail_the_step(tmp_path: Path) -> None:
    """The regression. Run 41197107832 exited 3 here and lost the state upload."""
    result = _run_step(_weekly_excursion_script(), exit_code=3, tmp_path=tmp_path)

    assert result.returncode == 0, (
        "INSUFFICIENT_DATA is a normal data shortage; failing the step skips "
        f"every later step including the state upload. stderr: {result.stderr}"
    )
    assert "::notice::" in result.stdout
    assert "INSUFFICIENT_DATA" in result.stdout
    assert "Not a fault" in result.stdout
    assert "::error::" not in result.stdout


@pytest.mark.parametrize("exit_code", [1, 2, 4, 70])
def test_a_technical_fault_still_fails_the_step(exit_code: int, tmp_path: Path) -> None:
    """The half of the fix that must not be lost.

    Treating exit 3 as normal is only safe if everything else still fails:
    a crash in the harness must not be absorbed as "not enough data".
    """
    result = _run_step(_weekly_excursion_script(), exit_code=exit_code, tmp_path=tmp_path)

    assert result.returncode == exit_code, result.stdout
    assert "::error::" in result.stdout
    assert "a technical fault, not a data shortage" in result.stdout
    assert "::notice::" not in result.stdout


def test_the_step_does_not_disable_errexit_wholesale(tmp_path: Path) -> None:
    """`set +e` would have fixed the symptom and hidden the next bug.

    The exemption has to stay local to the one command whose non-zero exit
    is expected, so that a genuine failure later in the script still fails
    the step.
    """
    script = _weekly_excursion_script()

    assert "set +e" not in script

    # And demonstrably so: a failing command appended after the branch must
    # still fail, which it could not if errexit had been switched off.
    probe = script + "\nfalse\necho 'should not be reached'\n"
    result = _run_step(probe, exit_code=3, tmp_path=tmp_path)

    assert result.returncode != 0
    assert "should not be reached" not in result.stdout


def test_every_step_after_the_excursion_measurement_depends_on_it_passing() -> None:
    """Why exit 3 mattered so much: the state upload comes later.

    This is the consequence the fix exists for, pinned so that nobody
    "simplifies" the ordering without noticing. `continue-on-error` on the
    census is deliberate and is not a substitute -- it protects that one
    step, not the ones after the excursion measurement.
    """
    workflow = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / ".github" / "workflows" / "pipeline.yml").read_text()
    )
    names = [str(s.get("name", "")) for s in workflow["jobs"]["weekly"]["steps"]]

    excursion = names.index(_STEP_NAME)
    tournament = next(i for i, n in enumerate(names) if "research tournament" in n)
    pack = next(i for i, n in enumerate(names) if n.startswith("Pack and upload state"))

    assert excursion < tournament < pack
