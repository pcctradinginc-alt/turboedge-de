"""Atomic research checkpoints and explicit workflow failure classification."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def finalize(path: Path, exit_code: int) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except (OSError, ValueError):
        payload = {}
    if exit_code:
        payload["status"] = "TIMEOUT" if exit_code in (124, 137) else "FAILED"
    elif payload.get("status") not in ("EXPLORATORY_ONLY", "INCOMPLETE_DATA_OR_EVALUATION"):
        payload["status"] = "FAILED"
    payload["exit_code"] = exit_code
    payload["finished_at"] = datetime.now(UTC).isoformat()
    payload["promotion_allowed"] = False
    write_checkpoint(path, payload)


if __name__ == "__main__":
    finalize(Path(sys.argv[1]), int(sys.argv[2]))
