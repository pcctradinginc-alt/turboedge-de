"""Forward Ledger: the system's append-only central memory (Master Spec §25).

Every scan run's candidates -- ACTIONABLE, WATCH, REJECT and a stratified
shadow sample of otherwise-discarded candidates -- are recorded here before
anything is known about their outcome, so later evaluation of the system's
skill is never contaminated by hindsight or selection bias (Master Spec §25:
"Nicht nur ACTIONABLE-Kandidaten speichern... Eine stratified Shadow Sample
verworfener Kandidaten ist Pflicht, um Selection Bias zu reduzieren.").

Entries are never overwritten or deleted once recorded; the only mutation an
entry ever undergoes is the ``open -> labeled`` status transition performed
by :meth:`ForwardLedger.attach_label` when its (separately append-only)
:class:`~turboedge.storage.schemas.LedgerLabel` is attached.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime

import numpy as np

from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import Category, LedgerEntry, LedgerEntryStatus, LedgerLabel


class ForwardLedger:
    """Thin, append-only wrapper around the ``forward_ledger``/
    ``ledger_labels`` tables (``storage/duckdb.py``).

    Usage::

        ledger = ForwardLedger(store)
        ledger.record(entries)
        due = ledger.due_for_labeling(datetime.now(UTC))
        ledger.attach_label(label)
        rows = ledger.entries(run_id="run-1")
    """

    def __init__(self, store: Store) -> None:
        self.store = store

    def record(self, entries: Sequence[LedgerEntry]) -> int:
        """Persist new entries.

        Idempotent per ``entry_id`` (``sha256(run_id|candidate_id|horizon)``,
        Build Contract v2 W6 requirement 1/2): recording the same
        ``(run_id, candidate_id, horizon_days)`` triple twice is a no-op the
        second time -- never a duplicate row, never an overwrite. Returns
        the number of rows actually newly inserted.
        """
        return self.store.append_ledger_entries(entries)

    def due_for_labeling(self, as_of: datetime) -> list[LedgerEntry]:
        """Open entries whose ``exit_due`` date has arrived by ``as_of``."""
        return self.store.ledger_entries_due_for_labeling(as_of)

    def attach_label(self, label: LedgerLabel) -> None:
        """Attach the append-only exit-side label to its entry, flipping
        that entry's status to ``labeled``. Raises ``StoreError`` if the
        entry does not exist or is already labeled (a label is a final
        fact, recorded exactly once)."""
        self.store.attach_ledger_label(label)

    def entries(
        self,
        *,
        run_id: str | None = None,
        underlying: str | None = None,
        status: LedgerEntryStatus | None = None,
        category: Category | None = None,
        is_shadow: bool | None = None,
    ) -> list[tuple[LedgerEntry, LedgerLabel | None]]:
        """Entries matching the given filters (AND-combined; an omitted
        filter is unconstrained), each paired with its label if one has
        been attached yet, ordered by ``prediction_time`` ascending."""
        return self.store.list_ledger_entries(
            run_id=run_id,
            underlying=underlying,
            status=status,
            category=category,
            is_shadow=is_shadow,
        )


def select_shadow_sample[T](
    candidates: Sequence[T],
    rng: np.random.Generator,
    per_stratum: int,
    strata_fn: Callable[[T], tuple[str, str, str]],
) -> list[T]:
    """Draw a stratified shadow sample of (typically discarded) candidates.

    Master Spec §25 requires a *stratified* shadow sample of rejected
    candidates be stored alongside ACTIONABLE ones, to reduce selection bias
    when later evaluating the system's overall skill. Candidates are grouped
    by ``strata_fn(candidate)`` -- by convention
    ``(category, direction, leverage_bucket)`` per Build Contract v2 -- and
    up to ``per_stratum`` candidates are drawn uniformly at random (without
    replacement) from each group via the caller-supplied ``rng``.

    Deterministic given ``rng``'s state (CLAUDE.md rule 16): strata are
    visited in a stable, sorted order, and within a stratum the selected
    positions are returned in their original relative order.

    Args:
        candidates: The pool to sample from (any type; typically
            ``CandidateEvaluation``).
        rng: Seeded ``numpy.random.Generator`` -- never a bare, unseeded
            random source.
        per_stratum: Maximum number of candidates drawn per stratum. A
            stratum with fewer members than this contributes all of them.
        strata_fn: Maps one candidate to its stratum key, e.g.
            ``lambda c: (c.category.value, c.direction.value, c.leverage_bucket or "unknown")``.

    Returns:
        The sampled candidates, grouped by stratum (strata in sorted key
        order), each stratum's members in their original relative order.
    """
    if per_stratum <= 0 or not candidates:
        return []
    groups: dict[tuple[str, str, str], list[int]] = {}
    for idx, candidate in enumerate(candidates):
        key = strata_fn(candidate)
        groups.setdefault(key, []).append(idx)

    selected: list[T] = []
    for key in sorted(groups):
        indices = np.asarray(groups[key], dtype=np.int64)
        n = min(per_stratum, len(indices))
        chosen = rng.choice(indices, size=n, replace=False)
        for i in sorted(chosen.tolist()):
            selected.append(candidates[i])
    return selected


__all__ = ["ForwardLedger", "select_shadow_sample"]
