"""Independent, count-only external ingestion verdict.

A state-changing Pipeline workflow must remain successful after publishing a
usable encrypted state artifact: the restore chain intentionally selects only
successful runs. A series ingest can still fail under `continue-on-error`
without losing the successful series from that run. This small *separate*
watchdog verifies the published counter artifact and alarms independently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def evaluate_report(document: Any) -> tuple[bool, str]:
    """Return (healthy, reason); malformed/missing evidence fails closed."""
    if not isinstance(document, dict) or document.get("mode") != "external_ingest":
        return False, "external ingest summary is missing or has an unexpected mode"
    counts = document.get("counts")
    if not isinstance(counts, dict):
        return False, "external ingest counts missing"
    failures = counts.get("failures")
    series = counts.get("series")
    if type(failures) is not int or failures < 0:
        return False, "invalid external ingest failures count"
    if type(series) is not int or series < 0:
        return False, "invalid external ingest series count"
    if failures:
        return False, f"{failures} external data series failed ingestion"
    if series == 0 and not document.get("skipped"):
        return False, "external ingest processed no series and reported no skipped sources"
    return True, f"external ingest completed: {series} series, zero failures"


def main(argv: list[str] | None = None) -> int:
    arguments = argv if argv is not None else sys.argv[1:]
    print("## External data health")
    if len(arguments) != 1:
        print("::error::expected one external health summary path")
        return 1
    try:
        report = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"::error::external data health evidence unavailable: {type(exc).__name__}")
        return 1
    healthy, reason = evaluate_report(report)
    print(f"- Verdict: {'ok' if healthy else 'ALARM'} -- {reason}")
    if not healthy:
        print(f"::error::{reason}")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
