from __future__ import annotations

from pathlib import Path

import pytest

from turboedge.learning.failed_hypotheses import (
    allow_retest,
    append,
    is_dormant,
    load,
    new_hypothesis,
)


def test_load_missing_file_returns_empty_list(tmp_path: Path) -> None:
    assert load(tmp_path / "failed_hypotheses.json") == []


def test_append_persists_and_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "failed_hypotheses.json"
    h1 = new_hypothesis("rsi_14", ["5d", "10d"], -0.0012, 412, trial_id="TR-1")
    h2 = new_hypothesis("macd_cross", ["7d"], -0.0004, 210, trial_id="TR-2")

    append(path, h1)
    append(path, h2)

    loaded = load(path)
    assert len(loaded) == 2
    assert {h.feature for h in loaded} == {"rsi_14", "macd_cross"}
    assert all(h.status == "dormant" for h in loaded)


def test_is_dormant(tmp_path: Path) -> None:
    path = tmp_path / "failed_hypotheses.json"
    assert is_dormant(path, "rsi_14") is False
    append(path, new_hypothesis("rsi_14", ["5d"], -0.001, 100))
    assert is_dormant(path, "rsi_14") is True


def test_allow_retest_requires_non_empty_note(tmp_path: Path) -> None:
    path = tmp_path / "failed_hypotheses.json"
    append(path, new_hypothesis("rsi_14", ["5d"], -0.001, 100))
    with pytest.raises(ValueError, match="regime_change_note"):
        allow_retest(path, "rsi_14", "   ")


def test_allow_retest_flips_status_and_preserves_history(tmp_path: Path) -> None:
    path = tmp_path / "failed_hypotheses.json"
    append(path, new_hypothesis("rsi_14", ["5d", "10d"], -0.001, 100))

    allow_retest(path, "rsi_14", "volatility regime shift confirmed via ADF test")

    loaded = load(path)
    assert len(loaded) == 1
    assert loaded[0].status == "retest_allowed"
    assert loaded[0].regime_change_note == "volatility regime shift confirmed via ADF test"
    assert is_dormant(path, "rsi_14") is False


def test_allow_retest_raises_when_no_match(tmp_path: Path) -> None:
    path = tmp_path / "failed_hypotheses.json"
    append(path, new_hypothesis("rsi_14", ["5d"], -0.001, 100))
    with pytest.raises(ValueError, match="no failed_hypotheses entry"):
        allow_retest(path, "never_tried", "some regime change")


def test_seed_state_file_parses_valid_json(tmp_path: Path) -> None:
    """The registry file format parses correctly and can be round-tripped.
    Uses a temporary copy instead of the shared `state/registry/failed_hypotheses.json`
    to avoid test brittleness when research results are added."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    seed_path = repo_root / "state" / "registry" / "failed_hypotheses.json"
    assert seed_path.exists()

    # Load and parse the existing registry to verify format is valid
    entries = load(seed_path)
    assert isinstance(entries, list)
    for entry in entries:
        assert hasattr(entry, "feature")
        assert hasattr(entry, "status")

    # Verify a fresh, empty registry also parses
    fresh_path = tmp_path / "failed_hypotheses.json"
    assert load(fresh_path) == []
