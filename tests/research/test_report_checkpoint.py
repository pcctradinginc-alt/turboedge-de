"""Checkpoint recovery preserves results and never treats a missing report as success."""

import json
from pathlib import Path

from turboedge.research.report_checkpoint import finalize, write_checkpoint


def test_timeout_preserves_partial_results(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_checkpoint(path, {"status": "PARTIAL", "effects": [{"feature": "test"}]})
    finalize(path, 124)
    result = json.loads(path.read_text())
    assert result["status"] == "TIMEOUT"
    assert result["effects"] == [{"feature": "test"}]
    assert result["promotion_allowed"] is False
    assert not path.with_suffix(".json.tmp").exists()


def test_missing_report_is_failure_even_after_zero_exit(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    finalize(path, 0)
    assert json.loads(path.read_text())["status"] == "FAILED"
