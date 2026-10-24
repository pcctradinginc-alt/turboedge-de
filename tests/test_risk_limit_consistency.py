"""The risk limits must agree between GOVERNANCE.md, config and code.

Reconciled 2026-10-24 (GOVERNANCE.md §5.2a). What that found matters for
how these tests are shaped: the "four divergences" were three different
kinds of problem, and only two of them were a disagreement about one
quantity.

    cap                      GOVERNANCE.md     config/code     kind
    max_position_fraction    2%                5%              same quantity, UNRESOLVED
    max_cluster_fraction     5%                15%             same quantity, UNRESOLVED
    KO loss                  8% portfolio-wide 2% per position NOT the same quantity
    total exposure           100% notional     30% capital     NOT the same quantity

The earlier version of this file asserted all three as numeric conflicts,
which claimed a comparability that does not exist for the KO row: the
implementation caps one position's expected KO loss contribution, while the
document described a portfolio-wide budget that is **not implemented at
all**. Read as the same per-position quantity, 8% could never bind (it would
need `p_ko > 4`) -- which is itself the evidence that the two mean different
things.

So this file now does four separate jobs:

1. Keeps the two genuine conflicts as a standing expected failure, naming
   both numbers, until a human records a decision.
2. Hard-stops the one direction that could happen without review: a
   configured cap looser than the code's own default.
3. Pins the semantic findings in GOVERNANCE.md, so the document cannot drift
   back to describing limits the code does not have.
4. Pins that the two portfolio-level caps are actually *supplied*. They were
   inert until 2026-10-24 -- nothing passed `cluster_fraction_used` or
   `total_risk_used`, so both collapsed to per-position caps looser than the
   single-position cap and neither ever bound. A limit nobody passes is not
   a limit, and that failure is invisible to any test that only compares
   numbers.

None of these tests changes a limit. Changing one needs an explicit human
decision -- tightening is still a change to a hard risk limit, and §11
reserves that.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

from turboedge.ranking.cluster import ClusterConfig
from turboedge.ranking.sizing import SizingConfig, suggested_position_fraction

GOVERNANCE = Path("GOVERNANCE.md")
RANKING_YAML = Path("configs/ranking.yaml")
SCAN = Path("src/turboedge/pipeline/scan.py")
SCAN_ALL = Path("src/turboedge/pipeline/scan_all.py")

#: The two caps where GOVERNANCE.md §5.2 and the config describe the *same*
#: quantity with different numbers. Transcribed from the §5.2 table; the test
#: below re-reads the document so a silent edit there is caught.
CONFLICTING_CAPS: dict[str, float] = {
    "max_position_fraction": 0.02,
    "max_cluster_fraction": 0.05,
}


def _governance_text() -> str:
    if not GOVERNANCE.is_file():
        pytest.skip("GOVERNANCE.md not present")
    return GOVERNANCE.read_text(encoding="utf-8")


def _governance_caps() -> dict[str, float]:
    """Parse the §5.2 table rather than trusting the copy above."""
    text = _governance_text()
    found: dict[str, float] = {}
    for name in CONFLICTING_CAPS:
        match = re.search(rf"^\|\s*`?{name}`?\s*\|\s*([0-9.]+)\s*%", text, re.M)
        if match:
            found[name] = float(match.group(1)) / 100.0
    return found


def _configured() -> dict[str, float]:
    if not RANKING_YAML.is_file():
        pytest.skip("configs/ranking.yaml not present")
    raw = yaml.safe_load(RANKING_YAML.read_text(encoding="utf-8")) or {}
    sizing = raw.get("sizing") or {}
    return {k: float(v) for k, v in sizing.items() if isinstance(v, int | float)}


# -- 1. the two genuine conflicts, kept visible --------------------------


def test_the_governance_document_still_states_the_caps_this_test_knows() -> None:
    # If the table is reworded or a row removed, the comparison below would
    # silently stop comparing anything.
    assert _governance_caps() == CONFLICTING_CAPS


@pytest.mark.parametrize("name", sorted(CONFLICTING_CAPS))
def test_the_known_divergence_between_governance_and_config_is_reported(
    name: str,
) -> None:
    # Deliberately a single expected-failure surface rather than an
    # assertion that the numbers match: they do not, and relaxing the
    # expectation to the looser value would be exactly the "quietly adjust
    # the threshold to the observation" move GOVERNANCE.md forbids. This
    # records the open conflict with both numbers, so it cannot be forgotten
    # and cannot be resolved by accident.
    governance = CONFLICTING_CAPS[name]
    configured = _configured().get(name)
    if configured is None:
        pytest.skip(f"{name} is not set in configs/ranking.yaml")

    if configured != pytest.approx(governance):
        pytest.xfail(
            f"UNRESOLVED: {name} is {configured} in configs/ranking.yaml and "
            f"{governance} in GOVERNANCE.md §5.2. Provenance (§5.2a): the stricter "
            "figure predates the sizing code; the looser one arrived with it, in a "
            "commit that edited GOVERNANCE.md but not §5.2. Reconciling needs a "
            "dated human decision; the safety-relevant direction is covered by "
            "test_no_configured_cap_is_looser_than_the_code_default."
        )


# -- 2. the direction that must never happen quietly ---------------------


def test_no_configured_cap_is_looser_than_the_code_default() -> None:
    # A YAML edit that widens a cap past what the code ships as its own
    # default is the change that could happen without any governance entry,
    # and it is the only one this suite stops outright.
    configured = _configured()
    defaults = SizingConfig()

    for name in (
        "max_position_fraction",
        "max_cluster_fraction",
        "max_total_risk",
        "max_ko_loss_contribution",
    ):
        if name not in configured:
            continue
        assert configured[name] <= getattr(defaults, name) + 1e-12, (
            f"configs/ranking.yaml sets {name}={configured[name]}, looser than the "
            f"code default {getattr(defaults, name)}. Widening a risk cap needs a "
            "GOVERNANCE.md entry and a human decision, not a YAML edit."
        )


# -- 3. the semantic findings, pinned in the document --------------------


def test_the_governance_document_names_the_total_exposure_cap_the_code_has() -> None:
    # It described `max_total_notional` (100% of notionals), which the code
    # does not have, and omitted `max_total_risk` (30% of capital), which it
    # does. A document naming a limit that does not exist is worse than one
    # naming none.
    text = _governance_text()

    assert "max_total_risk" in text, (
        "GOVERNANCE.md §5.2 no longer names the total-exposure cap the code "
        "actually applies (`max_total_risk`)."
    )
    assert "max_total_notional" not in text or "not the same quantity" in text, (
        "GOVERNANCE.md still presents `max_total_notional` as an implemented "
        "limit. The code has no such cap."
    )


def test_the_governance_document_records_the_ko_cap_as_per_position() -> None:
    # The implementation caps one position's expected KO loss contribution.
    # Presenting 8% and 2% as the same number in conflict claims a
    # comparability that does not exist.
    text = _governance_text()

    assert "per position" in text
    assert "not implemented" in text, (
        "GOVERNANCE.md should still record that the portfolio-wide KO loss "
        "budget it describes has no implementation."
    )


def test_the_portfolio_wide_ko_budget_is_still_unimplemented() -> None:
    # The claim above must stay true. `max_cluster_fraction` and
    # `max_total_risk` are netted against a "used" argument; the KO cap is
    # not, which is exactly what makes it per-position.
    source = Path("src/turboedge/ranking/sizing.py").read_text(encoding="utf-8")
    parameters = ast.parse(source)
    names = {
        arg.arg
        for node in ast.walk(parameters)
        if isinstance(node, ast.FunctionDef)
        for arg in list(node.args.args) + list(node.args.kwonlyargs)
    }

    assert "cluster_fraction_used" in names
    assert "total_risk_used" in names
    assert "ko_loss_used" not in names, (
        "a portfolio-wide KO budget now exists in sizing.py -- update "
        "GOVERNANCE.md §5.2a, which records it as unimplemented."
    )


def test_the_third_cluster_number_still_has_no_governance_counterpart() -> None:
    # `cluster.max_capital_fraction_per_cluster` (25%) is the limit the
    # separate `cluster_risk_pass` gate uses, distinct from
    # `sizing.max_cluster_fraction`. Recorded in §5.2a, not resolved.
    raw = yaml.safe_load(RANKING_YAML.read_text(encoding="utf-8")) or {}
    cluster = (raw.get("cluster") or {}).get("max_capital_fraction_per_cluster")

    assert cluster == pytest.approx(ClusterConfig().max_capital_fraction_per_cluster)
    assert "max_capital_fraction_per_cluster" in _governance_text(), (
        "§5.2a should still record this third cluster number and that it has no "
        "counterpart in the §5.2 table."
    )


# -- 4. the caps are actually supplied -----------------------------------


def _kwargs_at_call(path: Path, func_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == func_name:
            found |= {kw.arg for kw in node.keywords if kw.arg}
    return found


def test_the_scan_supplies_both_portfolio_level_caps() -> None:
    # The defect this pins: until 2026-10-24 neither argument was passed,
    # both defaulted to 0.0, and two of the four caps never bound. A limit
    # nobody passes is not a limit, and no number-comparing test can see it.
    supplied = _kwargs_at_call(SCAN, "evaluate_product_horizons")

    assert "cluster_fraction_used" in supplied, (
        "pipeline/scan.py no longer passes cluster_fraction_used, so "
        "max_cluster_fraction collapses to a per-position cap of 15% -- looser "
        "than the 5% single-position cap, i.e. inert."
    )
    assert "total_risk_used" in supplied, (
        "pipeline/scan.py no longer passes total_risk_used, so max_total_risk "
        "collapses to a per-position cap of 30%, i.e. inert."
    )


def test_the_batch_scan_supplies_the_portfolio_total() -> None:
    supplied = _kwargs_at_call(SCAN_ALL, "run_scan")

    assert "portfolio_fraction_used" in supplied


def test_the_batch_scan_seeds_held_positions() -> None:
    # The cluster accumulator started empty on every run, so the cluster cap
    # was net of this run's own proposals and knew nothing about positions
    # actually held.
    source = SCAN_ALL.read_text(encoding="utf-8")

    assert "portfolio_usage" in source, (
        "scan_all no longer reads the held-position usage, so the portfolio-level "
        "caps are again net of nothing but this run's own proposals."
    )


def test_the_cluster_cap_binds_when_the_used_fraction_is_supplied() -> None:
    # The behaviour the wiring exists to enable, at the sizing boundary.
    cfg = SizingConfig()
    returns = [0.4] * 60 + [-0.2] * 40

    unused = suggested_position_fraction(returns, p_ko=0.01, uncertainty=0.0, cfg=cfg)
    nearly_full = suggested_position_fraction(
        returns,
        p_ko=0.01,
        uncertainty=0.0,
        cluster_fraction_used=cfg.max_cluster_fraction - 0.001,
        cfg=cfg,
    )
    exhausted = suggested_position_fraction(
        returns,
        p_ko=0.01,
        uncertainty=0.0,
        cluster_fraction_used=cfg.max_cluster_fraction,
        cfg=cfg,
    )

    assert unused > 0.0
    assert nearly_full == pytest.approx(0.001)
    assert exhausted == 0.0


def test_the_total_risk_cap_binds_when_the_used_fraction_is_supplied() -> None:
    # Previously untested at any level, because nothing ever passed it.
    cfg = SizingConfig()
    returns = [0.4] * 60 + [-0.2] * 40

    exhausted = suggested_position_fraction(
        returns,
        p_ko=0.01,
        uncertainty=0.0,
        total_risk_used=cfg.max_total_risk,
        cfg=cfg,
    )

    assert exhausted == 0.0
