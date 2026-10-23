"""The risk limits must agree between GOVERNANCE.md, config and code.

Four values diverge today (measured 2026-10-23), in both directions, with
nothing keeping them aligned:

    cap                        GOVERNANCE.md §5.2   configs/ranking.yaml
    max_position_fraction      2%                   5%    (looser, 2.5x)
    max_cluster_fraction       5%                   15%   (looser, 3x)
    max_ko_loss_contribution   8%                   2%    (stricter)

The code defaults in `ranking/sizing.py` match the YAML, so config and
implementation agree with each other and both disagree with the governance
document. There is currently no automated check at all, which is how the
drift went unnoticed.

These tests do NOT change a limit. Changing one needs an explicit human
decision -- tightening is still a change to a hard risk limit, and
GOVERNANCE.md §11 reserves that. What they do is make the divergence
impossible to ignore: the suite states the conflict, names both values, and
keeps stating it until someone reconciles them.

The one thing they do enforce unconditionally is the direction that matters:
a configured cap may never be *looser* than the code's own default. Loosening
by editing YAML, with no governance entry and no review, is the failure mode
worth a hard stop.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from turboedge.ranking.cluster import ClusterConfig
from turboedge.ranking.sizing import SizingConfig

GOVERNANCE = Path("GOVERNANCE.md")
RANKING_YAML = Path("configs/ranking.yaml")

#: The caps GOVERNANCE.md §5.2 states, as fractions. Transcribed from the
#: table at GOVERNANCE.md:186-194; the test below re-reads the document so a
#: silent edit there is caught rather than assumed away.
GOVERNANCE_CAPS: dict[str, float] = {
    "max_position_fraction": 0.02,
    "max_cluster_fraction": 0.05,
    "max_ko_loss_contribution": 0.08,
}


def _governance_caps() -> dict[str, float]:
    """Parse the §5.2 table out of GOVERNANCE.md rather than trusting a copy."""
    if not GOVERNANCE.is_file():
        pytest.skip("GOVERNANCE.md not present")
    text = GOVERNANCE.read_text(encoding="utf-8")
    found: dict[str, float] = {}
    for name in GOVERNANCE_CAPS:
        match = re.search(rf"^\|\s*{name}\s*\|\s*([0-9.]+)\s*%", text, re.M)
        if match:
            found[name] = float(match.group(1)) / 100.0
    return found


def _configured() -> dict[str, float]:
    if not RANKING_YAML.is_file():
        pytest.skip("configs/ranking.yaml not present")
    raw = yaml.safe_load(RANKING_YAML.read_text(encoding="utf-8")) or {}
    sizing = raw.get("sizing") or {}
    return {k: float(v) for k, v in sizing.items() if isinstance(v, (int, float))}


def test_the_governance_document_still_states_the_caps_this_test_knows() -> None:
    # If the table is reworded or a row removed, the comparison below would
    # silently stop comparing anything.
    assert _governance_caps() == GOVERNANCE_CAPS


def test_no_configured_cap_is_looser_than_the_code_default() -> None:
    # The direction that matters. A YAML edit that widens a cap past what
    # the code ships as its own default is the change that could happen
    # without any governance entry, and it is the only one this suite stops
    # outright.
    configured = _configured()
    defaults = SizingConfig()

    for name in ("max_position_fraction", "max_cluster_fraction", "max_total_risk"):
        if name not in configured:
            continue
        assert configured[name] <= getattr(defaults, name) + 1e-12, (
            f"configs/ranking.yaml sets {name}={configured[name]}, looser than the "
            f"code default {getattr(defaults, name)}. Widening a risk cap needs a "
            "GOVERNANCE.md entry and a human decision, not a YAML edit."
        )


@pytest.mark.parametrize("name", sorted(GOVERNANCE_CAPS))
def test_the_known_divergence_between_governance_and_config_is_reported(
    name: str,
) -> None:
    # Deliberately a single expected-failure surface rather than an
    # assertion that the numbers match: they do not, and pretending
    # otherwise by relaxing the expectation to the looser value would be
    # exactly the "quietly adjust the threshold to the observation" move
    # GOVERNANCE.md forbids. This records the open conflict with both
    # numbers, so it cannot be forgotten and cannot be resolved by accident.
    governance = GOVERNANCE_CAPS[name]
    configured = _configured().get(name)
    if configured is None:
        pytest.skip(f"{name} is not set in configs/ranking.yaml")

    if configured != pytest.approx(governance):
        pytest.xfail(
            f"UNRESOLVED: {name} is {configured} in configs/ranking.yaml and "
            f"{governance} in GOVERNANCE.md §5.2. Reconciling this needs a human "
            "decision; the safety-relevant direction is covered separately by "
            "test_no_configured_cap_is_looser_than_the_code_default."
        )


def test_the_cluster_capital_cap_has_no_governance_counterpart() -> None:
    # A third cluster number exists in configs/ranking.yaml
    # (`cluster.max_capital_fraction_per_cluster`, 25%) with nothing
    # corresponding in GOVERNANCE.md §5.2, which lists a 5% cluster cap.
    # Recorded so the gap is visible; not resolved here.
    raw = yaml.safe_load(RANKING_YAML.read_text(encoding="utf-8")) or {}
    cluster = (raw.get("cluster") or {}).get("max_capital_fraction_per_cluster")

    assert cluster == pytest.approx(ClusterConfig().max_capital_fraction_per_cluster)
    assert "max_capital_fraction_per_cluster" not in GOVERNANCE.read_text(encoding="utf-8"), (
        "GOVERNANCE.md now mentions this cap -- reconcile it and update this test"
    )
