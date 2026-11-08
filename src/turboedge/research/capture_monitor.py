"""Research quote-capture E2E monitor stored in the EXISTING encrypted state.

No extra database or external service. The status marker travels in
state/registry/ with the already-encrypted lean state backup. A missing or
rolled-back PIT tape cannot silently pass as an empty research dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from turboedge.research.data_factory import TapeCapture, TapeHealth, audit_research_tape
from turboedge.research.market_session import is_research_market_session

_MONITOR_PATH = Path("registry/research_capture_monitor.json")
_VERSION = 1
_CONSECUTIVE_FAILURE_ALERT_AT = 3
_MISSING_EXIT_ALERT_AT = 2


@dataclass(frozen=True, slots=True)
class CaptureMonitorState:
    version: int = _VERSION
    attempts: int = 0
    no_eligible_pair_streak: int = 0
    expected_quotes: int = 0
    expected_entry_days: int = 0
    last_attempt_at: str | None = None
    last_success_at: str | None = None
    last_missing_exit_alert_count: int = 0


@dataclass(frozen=True, slots=True)
class CaptureMonitorReport:
    status: str
    reason: str
    consecutive_no_pair: int
    quotes: int
    independent_days: int
    complete_exits: int
    missing_exits: int
    pending_exits: int
    alert: bool


def _read_state(state_dir: Path) -> CaptureMonitorState:
    path = state_dir / _MONITOR_PATH
    if not path.exists():
        return CaptureMonitorState()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("corrupt research capture state: not an object")
    state = CaptureMonitorState(**raw)
    if state.version != _VERSION:
        raise ValueError("unsupported research capture monitor state version")
    if min(state.attempts, state.no_eligible_pair_streak, state.expected_quotes) < 0:
        raise ValueError("negative capture monitor counter")
    return state


def _write_state(state_dir: Path, value: CaptureMonitorState) -> None:
    path = state_dir / _MONITOR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    try:
        temp.write_text(json.dumps(asdict(value), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def verify_restored_tape(state_dir: Path, *, now: datetime) -> TapeHealth:
    """Prevent rolling research state backwards to a smaller/missing archive."""
    state = _read_state(state_dir)
    health = audit_research_tape(state_dir, as_of=now)
    if (
        health.quote_rows < state.expected_quotes
        or health.entry_days < state.expected_entry_days
        or health.invalid_temporal_rows
    ):
        raise ValueError(
            "research capture tape is missing, rolled back, or has invalid "
            "chronology after state restore; refusing silent reset"
        )
    return health


def record_capture_attempt(
    state_dir: Path,
    *,
    now: datetime,
    result: TapeCapture,
) -> CaptureMonitorReport:
    """Validate new archive, persist attempt, and expose alert decision.

    A pair can be observable even if the 3-day cohort spacing forbids writing
    a new entry. This is not misclassified as a broken data source.
    """
    if not is_research_market_session(now):
        raise ValueError("capture monitoring is only allowed during an open session")
    before = _read_state(state_dir)
    after = verify_restored_tape(state_dir, now=now)
    if result.quotes_written and after.quote_rows <= before.expected_quotes:
        raise ValueError("capture reported quotes but restored PIT tape did not advance")
    same_utc_day = (
        before.last_attempt_at is not None
        and datetime.fromisoformat(before.last_attempt_at).date() == now.date()
    )
    # A manual retry on the SAME calendar day must not count as a second
    # failed independent capture. It may still add real quotes to the tape.
    # Friday is a deliberate follow-up-only day, not a missed new-pair
    # acquisition. Nor should market closures consume the failure budget.
    entry_expected = is_research_market_session(now, entry=True)
    streak = (
        0
        if result.eligible_pair_found
        else before.no_eligible_pair_streak + 1
        if entry_expected and not same_utc_day
        else before.no_eligible_pair_streak
    )
    alert_no_pairs = (
        streak >= _CONSECUTIVE_FAILURE_ALERT_AT
        and streak % _CONSECUTIVE_FAILURE_ALERT_AT == 0
        and not same_utc_day
    )
    alert_missing = (
        after.missing_exit_days >= _MISSING_EXIT_ALERT_AT
        and after.missing_exit_days > before.last_missing_exit_alert_count
    )
    issue = (
        "NO_FRESH_CROSS_ISSUER_PAIRS"
        if alert_no_pairs
        else "MISSING_REAL_EXITS"
        if alert_missing
        else "OK"
    )
    updated = CaptureMonitorState(
        attempts=before.attempts + (0 if same_utc_day else 1),
        no_eligible_pair_streak=streak,
        expected_quotes=after.quote_rows,
        expected_entry_days=after.entry_days,
        last_attempt_at=now.isoformat(),
        last_success_at=now.isoformat() if result.eligible_pair_found else before.last_success_at,
        last_missing_exit_alert_count=(
            after.missing_exit_days if alert_missing else before.last_missing_exit_alert_count
        ),
    )
    _write_state(state_dir, updated)
    return CaptureMonitorReport(
        status="DEGRADED" if issue != "OK" else "OBSERVING",
        reason=issue,
        consecutive_no_pair=streak,
        quotes=after.quote_rows,
        independent_days=after.independent_entry_days,
        complete_exits=after.complete_exit_days,
        missing_exits=after.missing_exit_days,
        pending_exits=after.pending_exit_days,
        alert=issue != "OK",
    )


__all__ = [
    "CaptureMonitorReport",
    "CaptureMonitorState",
    "record_capture_attempt",
    "verify_restored_tape",
]
