"""Conservative GitHub Actions scan wake-up decisions (never reset state).

This is deliberately independent of third-party packages so a watchdog can
make its decision even before uv installs the project dependencies.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")


def should_dispatch_scan(
    runs: list[dict[str, object]],
    *,
    now: datetime,
    quiet_hours: int = 3,
) -> bool:
    """Only replace absent weekday market-hours runs; never double-start recent work."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(BERLIN)
    minute = local.hour * 60 + local.minute
    if local.weekday() >= 5 or not (9 * 60 + 5 <= minute <= 17 * 60 + 15):
        return False

    cutoff = now.astimezone(UTC) - timedelta(hours=quiet_hours)
    for run in runs:
        created = run.get("createdAt")
        if not isinstance(created, str):
            raise ValueError("workflow run has no createdAt timestamp")
        started = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if started.tzinfo is None:
            raise ValueError("workflow run timestamp must be timezone-aware")
        if started >= cutoff:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-file", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.runs_file.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("expected an array of GitHub Actions runs")
    print("dispatch" if should_dispatch_scan(data, now=datetime.now(UTC)) else "skip")


if __name__ == "__main__":
    main()
