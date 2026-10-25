"""Why the labeler produced the number it did.

THE DIAGNOSTIC GAP THIS CLOSES
------------------------------
`turboedge label` reported outcome counts only::

    label  labeled=0  ko=0  ambiguous=0  missing_data=0

Every one of those is zero in two completely different situations: nothing
was due, or everything was due and none of it could be labelled. The first
is the system correctly waiting for real observations to mature; the second
is a defect. The log could not tell them apart, and on 2026-10-23 that
ambiguity sat next to a forward ledger holding 3,670 entries and zero
labels -- a state that looks alarming and may be entirely correct.

So this reports the **population**, not just the outcome: how many entries
are open, how many are due, how many are *overdue and still unlabelled*,
and when the next one matures. `overdue_unlabeled > 0` is the condition
that means something is broken. `due_now == 0` with a `next_due_date` in
the future means the system is waiting, which is not a bug and must not be
reported as one.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It computes no label, estimates no exit price and shortens no horizon. It
only counts what is there, so that a count of zero can be attributed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from turboedge.storage.duckdb import Store

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LedgerHealth:
    """The forward ledger's state, in the terms that explain a label count."""

    as_of: datetime
    entries_total: int = 0
    open_entries: int = 0
    labeled_entries: int = 0
    expired_no_data: int = 0
    #: Open entries whose `exit_due` has arrived. What the labeler will act
    #: on the next time it runs.
    due_now: int = 0
    #: Open entries whose `exit_due` passed *before* today. These should
    #: have been labelled by an earlier run; if any exist, the loop is not
    #: keeping up and the reason has to be established.
    overdue_unlabeled: int = 0
    oldest_overdue_days: int | None = None
    #: The earliest `exit_due` still ahead. With `due_now == 0` this is the
    #: date on which labelling can next possibly produce anything.
    next_due_date: date | None = None
    entries_created_today: int = 0
    newest_label_age_days: int | None = None
    #: Labels that pass the **label-side** filters
    #: `build_excursion_dataset` applies: labelled, not EXPIRED_NO_DATA,
    #: not ambiguous, with both excursion targets and the realised P&L
    #: present.
    #:
    #: Not the same as "rows research will keep". That function applies a
    #: second, independent filter on the *entry* side -- every name in
    #: `FEATURE_NAMES` must be finite, and five of them come from
    #: `feature_snapshot`. An entry written with an empty or partial
    #: snapshot is dropped as `null_feature` however good its label is.
    #: Checking that here would mean restating `FEATURE_NAMES` in SQL, so
    #: this number is an upper bound and the harness reports the rest.
    usable_for_research: int = 0
    ko_hits: int = 0
    ambiguous_paths: int = 0
    missing_exit_quote: int = 0
    by_exit_reason: Mapping[str, int] = field(default_factory=dict)

    @property
    def waiting_for_data(self) -> bool:
        """True when zero labels is the correct outcome rather than a fault."""
        return self.due_now == 0 and self.overdue_unlabeled == 0

    @property
    def healthy(self) -> bool:
        """True unless an entry is overdue and still unlabelled."""
        return self.overdue_unlabeled == 0

    def summary(self) -> str:
        if self.overdue_unlabeled:
            return (
                f"{self.overdue_unlabeled} entry(ies) overdue and unlabelled, oldest "
                f"{self.oldest_overdue_days}d -- the labeling loop is not keeping up"
            )
        if self.due_now:
            return f"{self.due_now} entry(ies) due for labeling now"
        if self.next_due_date is not None:
            return (
                f"nothing due; {self.open_entries} open entry(ies), the next matures "
                f"{self.next_due_date.isoformat()}"
            )
        if not self.entries_total:
            return "the forward ledger is empty"
        return f"nothing due and nothing open; {self.labeled_entries} labelled entry(ies)"


def _scalar(store: Store, sql: str, params: list[object] | None = None) -> int:
    row = store._conn.execute(sql, params or []).fetchone()
    return int(row[0]) if row is not None and row[0] is not None else 0


def ledger_health(store: Store, *, as_of: datetime) -> LedgerHealth:
    """Count the forward ledger's state as of `as_of`.

    Aggregate queries rather than materialising the ledger: this runs in
    every end-of-day job, and the point is a cheap, always-available
    explanation of the label count.
    """
    today = as_of.astimezone(UTC).date()

    by_status = {
        str(status): int(count)
        for status, count in store._conn.execute(
            "SELECT status, count(*) FROM forward_ledger GROUP BY status"
        ).fetchall()
    }
    by_exit_reason = {
        str(reason): int(count)
        for reason, count in store._conn.execute(
            "SELECT exit_reason, count(*) FROM ledger_labels GROUP BY exit_reason"
        ).fetchall()
    }

    due_now = _scalar(
        store,
        "SELECT count(*) FROM forward_ledger WHERE status = 'open' AND exit_due <= ?",
        [today],
    )
    overdue = _scalar(
        store,
        "SELECT count(*) FROM forward_ledger WHERE status = 'open' AND exit_due < ?",
        [today],
    )
    oldest_overdue = store._conn.execute(
        "SELECT min(exit_due) FROM forward_ledger WHERE status = 'open' AND exit_due < ?",
        [today],
    ).fetchone()
    next_due = store._conn.execute(
        "SELECT min(exit_due) FROM forward_ledger WHERE status = 'open' AND exit_due > ?",
        [today],
    ).fetchone()
    newest_label = store._conn.execute("SELECT max(labeled_at) FROM ledger_labels").fetchone()

    health = LedgerHealth(
        as_of=as_of,
        entries_total=sum(by_status.values()),
        open_entries=by_status.get("open", 0),
        labeled_entries=by_status.get("labeled", 0),
        expired_no_data=by_status.get("expired_no_data", 0),
        due_now=due_now,
        overdue_unlabeled=overdue,
        oldest_overdue_days=(
            (today - oldest_overdue[0]).days
            if oldest_overdue is not None and oldest_overdue[0] is not None
            else None
        ),
        next_due_date=(next_due[0] if next_due is not None else None),
        entries_created_today=_scalar(
            store,
            "SELECT count(*) FROM forward_ledger WHERE CAST(prediction_time AS DATE) = ?",
            [today],
        ),
        newest_label_age_days=(
            (as_of - newest_label[0]).days
            if newest_label is not None and newest_label[0] is not None
            else None
        ),
        usable_for_research=_scalar(
            store,
            "SELECT count(*) FROM ledger_labels WHERE exit_reason <> 'expired_no_data' "
            "AND NOT ambiguous_path AND mae IS NOT NULL AND mfe IS NOT NULL "
            "AND realized_selected_pnl IS NOT NULL",
        ),
        ko_hits=_scalar(store, "SELECT count(*) FROM ledger_labels WHERE ko_hit"),
        ambiguous_paths=_scalar(store, "SELECT count(*) FROM ledger_labels WHERE ambiguous_path"),
        missing_exit_quote=_scalar(
            store, "SELECT count(*) FROM ledger_labels WHERE exit_bid IS NULL"
        ),
        by_exit_reason=by_exit_reason,
    )

    logger.info(
        "ledger_health",
        open_entries=health.open_entries,
        due_now=health.due_now,
        overdue_unlabeled=health.overdue_unlabeled,
        next_due=health.next_due_date.isoformat() if health.next_due_date else None,
        usable_for_research=health.usable_for_research,
        newest_label_age_days=health.newest_label_age_days,
        waiting_for_data=health.waiting_for_data,
    )
    return health


__all__ = ["LedgerHealth", "ledger_health"]
