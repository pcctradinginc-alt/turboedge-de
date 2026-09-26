"""Research queue -- the governance boundary of Phase 2 (§8).

Everything upstream of this module (`value_of_information`, `research_memory`,
`catalog`) only ever produces numbers: a priority score, an enriched
estimate, a ranked list. None of that is dangerous on its own. What makes a
research queue dangerous is what happens *after* the ranking -- whether a
high-scoring idea can turn into running code, a promoted model, or a live
trial without a human ever looking at it.

The user's rule, verbatim in intent: the system may PRIORITISE research
automatically, but it may NEVER change production code on its own; actually
implementing an idea requires human approval. This module is where that
rule either holds or quietly stops holding, so it is enforced as a type-level
and runtime constraint rather than a naming convention:

* `seed_from_catalog` -- the only way a NEW row can appear -- can only ever
  write `SYSTEM_ASSIGNABLE_STATES` (today: `{PROPOSED}`). It is checked
  against the catalog's own declared status, not assumed.
* `rescore` -- the only thing that runs on every scan without anyone asking
  it to -- is wired so it is *structurally impossible* for it to move a
  status, set `approved_by`/`approved_at`, or touch `trial_id`: those five
  fields are pinned back onto whatever `research_memory.enrich` returns
  before anything is persisted, and the whole object is re-validated through
  `ResearchOpportunity`'s own constructor (not `model_copy`, which would
  skip that check) so `_check_approval_recorded` runs on every write this
  module makes, not just the ones it remembers to check itself.
* `approve` is the only function in the entire codebase allowed to move an
  entry out of `PROPOSED`, and it refuses to run without a named human and a
  stated reason.
* `transition` moves an already-approved entry through the rest of its
  life, but only along edges in `ALLOWED_TRANSITIONS`, and it explicitly
  refuses `PROPOSED` as a source status even though `PROPOSED -> APPROVED`
  is a legal edge in that table -- the table describes the lifecycle graph,
  not who is allowed to walk which edge.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from turboedge.learning import failed_hypotheses
from turboedge.meta import catalog, research_memory, value_of_information
from turboedge.meta.catalog import MeasuredOutcome
from turboedge.meta.research_opportunity import (
    SYSTEM_ASSIGNABLE_STATES,
    ResearchOpportunity,
    ResearchStatus,
    StoredOpportunity,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    # `storage.duckdb` imports the research schemas from this package, so a
    # runtime import here closes a cycle: duckdb -> meta/__init__ ->
    # research_queue -> duckdb. The queue only needs `Store` as a type; it is
    # always handed a live one.
    from turboedge.storage.duckdb import Store


class IllegalTransition(Exception):
    """A status move that either is not in `ALLOWED_TRANSITIONS`, or is one
    of the extra restrictions this module layers on top of that table (most
    importantly: `PROPOSED` may only be left via `approve`)."""


#: Which status may legally follow which. This is the *complete* lifecycle
#: graph -- both `approve` and `transition` check membership here, plus
#: their own extra rules (`approve` is the only walker of `PROPOSED ->
#: APPROVED`; `transition` additionally requires a `trial_id` before
#: `RUNNING`). Read as: for each key, running research may legally continue
#: into any status in the value set. Reasoning for every edge, and every
#: absence, below.
ALLOWED_TRANSITIONS: Mapping[ResearchStatus, frozenset[ResearchStatus]] = {
    # The only way out of the system-writable state. Reachable only through
    # `approve`, never through `transition` -- see IllegalTransition raised
    # there. Kept in this table anyway because the table describes the
    # graph, and `apply_measured_outcomes` path-finds over it to decide
    # *which* function to call at each hop.
    ResearchStatus.PROPOSED: frozenset({ResearchStatus.APPROVED}),
    # An approved idea can: start running (RUNNING), be shelved without
    # ever running -- e.g. a resourcing decision, not a research result
    # (DORMANT), or be turned down outright before any trial is spent on it
    # (REJECTED). It cannot jump straight to MEASURED/PROMOTED: a result
    # with no RUNNING phase in between would have no trial_id and violate
    # Master Spec §23 trial discipline.
    ResearchStatus.APPROVED: frozenset(
        {ResearchStatus.RUNNING, ResearchStatus.DORMANT, ResearchStatus.REJECTED}
    ),
    # A running trial ends in a measured result, or is abandoned mid-flight
    # (REJECTED -- e.g. the adapter it depended on broke). It cannot go to
    # DORMANT directly: "dormant" means "we have a documented reason not to
    # run this right now", which for something already running is exactly
    # what MEASURED -> DORMANT (a negative result, parked rather than
    # rejected outright) is for.
    ResearchStatus.RUNNING: frozenset({ResearchStatus.MEASURED, ResearchStatus.REJECTED}),
    # A measured result can be PROMOTED (positive, human decides to act on
    # it -- the one place code/model changes are even eligible to follow, and
    # only ever outside this module), parked as DORMANT (negative result,
    # cheap to revisit later -- CLAUDE.md rule 31: negative patterns are
    # never deleted), or REJECTED (negative result, not worth revisiting).
    ResearchStatus.MEASURED: frozenset(
        {ResearchStatus.PROMOTED, ResearchStatus.DORMANT, ResearchStatus.REJECTED}
    ),
    # A dormant idea's only way back to life is a fresh approval -- Master
    # Spec §23's "Nur bei klar dokumentiertem Regimewechsel erneut testen"
    # mirrors `failed_hypotheses.allow_retest`'s required
    # `regime_change_note`; here that documentation is the `note` argument
    # `transition` requires on every hop, so the retest reason is on the
    # record before the entry can run again. It cannot jump back to RUNNING
    # or MEASURED directly -- a retest is a new approval decision, not a
    # continuation of the old one.
    ResearchStatus.DORMANT: frozenset({ResearchStatus.APPROVED}),
    # Terminal: a promoted idea's future is production code/model changes,
    # which happen outside this queue entirely (§8 -- this module never
    # touches production code).
    ResearchStatus.PROMOTED: frozenset(),
    # Terminal, deliberately distinct from DORMANT: REJECTED is a human
    # judgement that this is not worth revisiting (vs. DORMANT's "not now,
    # but see above for how"). Collapsing the two would lose the one
    # distinction that lets `ranked` treat them differently from a
    # cheap-to-revisit parked idea -- except both are equally excluded from
    # `ranked` today, since re-opening either requires the same explicit,
    # documented human act.
    ResearchStatus.REJECTED: frozenset(),
}

#: Terminal statuses excluded from `ranked`: the queue is what to do next,
#: not a history of everything ever proposed.
_TERMINAL_STATES = frozenset({ResearchStatus.PROMOTED, ResearchStatus.REJECTED})


def _append_note(existing: str, *, actor: str, note: str, now: datetime) -> str:
    """Append one timestamped, attributed line to a status history.

    Never overwrites: `status_note` is meant to read like a log, mirroring
    `failed_hypotheses`'s own append-only discipline (CLAUDE.md rule 31).
    """
    entry = f"[{now.isoformat()}] {actor}: {note.strip()}"
    return entry if not existing else f"{existing}\n{entry}"


def _revalidate(opportunity: ResearchOpportunity, **updates: object) -> ResearchOpportunity:
    """Apply `updates` and re-run every `ResearchOpportunity` validator.

    `model_copy(update=...)` deliberately skips validation, which would let
    a bug here silently produce a status without an `approved_by` -- exactly
    the invariant `_check_approval_recorded` exists to prevent. Every
    lifecycle mutation in this module goes through this helper instead, so
    that guarantee is enforced by the constructor on every write, not by
    this module remembering to check it.
    """
    data = {**opportunity.model_dump(), **updates}
    return ResearchOpportunity.model_validate(data)


def _shortest_path(frm: ResearchStatus, to: ResearchStatus) -> list[ResearchStatus] | None:
    """BFS over `ALLOWED_TRANSITIONS` for the hops from `frm` to `to`.

    Used by `apply_measured_outcomes` as its fallback path-finder (see
    `_measured_outcome_path`), and whenever it must land a status without
    ever writing one directly. If no path exists, that is a hole in
    `ALLOWED_TRANSITIONS`, not a reason to bypass it -- the caller raises.
    """
    if frm == to:
        return []
    seen = {frm}
    queue: deque[tuple[ResearchStatus, list[ResearchStatus]]] = deque([(frm, [])])
    while queue:
        node, path = queue.popleft()
        for nxt in ALLOWED_TRANSITIONS.get(node, frozenset()):
            if nxt in seen:
                continue
            new_path = [*path, nxt]
            if nxt == to:
                return new_path
            seen.add(nxt)
            queue.append((nxt, new_path))
    return None


def _measured_outcome_path(
    current: ResearchStatus, outcome: MeasuredOutcome
) -> list[ResearchStatus] | None:
    """The hops from `current` to `outcome.status` for `apply_measured_outcomes`.

    Not merely the shortest legal path: `ALLOWED_TRANSITIONS[APPROVED]`
    includes a direct edge to `DORMANT` for an idea shelved as a resourcing
    decision *before* it ever ran, which is a shorter route to `DORMANT`
    than going through `RUNNING`/`MEASURED` first. Taking that shortcut for
    an outcome that carries a `trial_id` -- meaning a trial genuinely ran
    and was measured -- would misrepresent a measured negative result as an
    idea nobody bothered to try. So whenever `trial_id` is set and the
    entry is still `PROPOSED`, the path is forced through `APPROVED ->
    RUNNING -> MEASURED` before its final resting state, regardless of
    whether a shorter edge exists.

    Falls back to plain shortest-path BFS whenever that narrative does not
    apply: `trial_id` is unset (Eurex/Euwax: no trial ever ran, so the
    shortest legal path already tells the right story), or the entry is not
    starting from `PROPOSED` (already partway through its lifecycle from an
    earlier call -- resume from wherever it is rather than assuming the
    full canonical narrative still applies).
    """
    if outcome.trial_id is not None and current is ResearchStatus.PROPOSED:
        path = [ResearchStatus.APPROVED, ResearchStatus.RUNNING, ResearchStatus.MEASURED]
        if outcome.status is not ResearchStatus.MEASURED:
            path.append(outcome.status)
        return path
    return _shortest_path(current, outcome.status)


class ResearchQueue:
    """Orchestrates the Phase 2 research queue: seeding, scoring, and the
    governed lifecycle every entry moves through (§8).

    Holds no state of its own beyond the `Store` and the failed-hypotheses
    registry path -- every method reads current state from the `Store`
    immediately before acting, so entries never go stale between calls.
    """

    def __init__(self, store: Store, *, failed_hypotheses_path: Path) -> None:
        self._store = store
        self._failed_hypotheses_path = failed_hypotheses_path

    def _require(self, hypothesis_id: str) -> StoredOpportunity:
        stored = self._store.get_research_opportunity(hypothesis_id)
        if stored is None:
            raise KeyError(f"no research opportunity {hypothesis_id!r} in the store")
        return stored

    # -- seeding -------------------------------------------------------

    def seed_from_catalog(self, *, now: datetime | None = None) -> list[str]:
        """Insert every catalog entry not already in the store.

        `now` is accepted for interface symmetry with the other lifecycle
        methods (every meta-layer method takes an explicit clock -- no
        look-ahead) but unused: seeding copies static catalog data and
        stamps no date of its own.

        Idempotent by construction: an id already present is skipped
        entirely, whatever its current status, `approved_by` or
        `status_note` -- re-seeding must never reset work a human has
        already done on that entry (§8). This also means a still-`PROPOSED`
        entry's *declared* estimates stay stable across re-seeds rather
        than being silently swapped out from under whoever is currently
        evaluating it; a catalog author who needs to correct a declared
        estimate does so by editing the stored entry directly (via
        `rescore`'s persistence path, or manually), not by re-seeding.

        Returns the ids it newly inserted.
        """
        del now  # unused -- see docstring
        existing_ids = {
            s.opportunity.hypothesis_id for s in self._store.list_research_opportunities()
        }
        inserted: list[str] = []
        for opportunity in catalog.CATALOG:
            if opportunity.status not in SYSTEM_ASSIGNABLE_STATES:
                raise ValueError(
                    f"catalog entry {opportunity.hypothesis_id!r} has status "
                    f"{opportunity.status}, which is not in SYSTEM_ASSIGNABLE_STATES; "
                    "seed_from_catalog refuses to write a pre-approved entry (§8)"
                )
            if opportunity.hypothesis_id in existing_ids:
                continue
            self._store.upsert_research_opportunity(opportunity)
            inserted.append(opportunity.hypothesis_id)
        return inserted

    # -- applying already-authorised results ---------------------------

    def apply_measured_outcomes(self, *, now: datetime | None = None) -> list[str]:
        """Land `catalog.W12_MEASURED_OUTCOMES` on their intended statuses.

        These four questions were already researched and authorised by the
        user in September 2026 -- without this, the queue would keep
        ranking work that is already finished, which is worse than not
        ranking it at all (it would waste review attention on a solved
        question dressed up as a new priority).

        Walks each outcome from its current stored status to
        `outcome.status` one legal hop at a time, via `approve` for the
        `PROPOSED -> APPROVED` hop and `transition` for every hop after
        that -- never by writing a status directly, so every guard those
        two functions enforce (non-blank approver/note, `trial_id` before
        `RUNNING`, `ALLOWED_TRANSITIONS` membership) applies here exactly as
        it would to a human-driven change. If `outcome.status` is not
        reachable from the current status, that is a bug in
        `ALLOWED_TRANSITIONS` and this raises rather than bypassing it.

        Skips ids not yet seeded, and ids already at the target status
        (idempotent). Returns the ids it changed.
        """
        now = now if now is not None else datetime.now(UTC)
        changed: list[str] = []
        for hypothesis_id, outcome in catalog.W12_MEASURED_OUTCOMES.items():
            stored = self._store.get_research_opportunity(hypothesis_id)
            if stored is None:
                continue  # not seeded yet -- nothing to apply the outcome to
            current = stored.opportunity.status
            if current == outcome.status:
                continue  # already applied
            path = _measured_outcome_path(current, outcome)
            if path is None:
                raise IllegalTransition(
                    f"W12 measured outcome for {hypothesis_id!r} targets "
                    f"{outcome.status}, which is not reachable from {current} via "
                    "any legal transition -- fix ALLOWED_TRANSITIONS, do not "
                    "bypass it"
                )
            frm = current
            for step_to in path:
                if frm is ResearchStatus.PROPOSED:
                    self.approve(
                        hypothesis_id,
                        approved_by=outcome.approved_by,
                        note=outcome.note,
                        trial_id=outcome.trial_id,
                        now=now,
                    )
                else:
                    self.transition(
                        hypothesis_id,
                        step_to,
                        actor=outcome.approved_by,
                        note=outcome.note,
                        now=now,
                    )
                frm = step_to
            changed.append(hypothesis_id)
        return changed

    # -- scoring --------------------------------------------------------

    def rescore(
        self,
        *,
        source_available: Mapping[str, bool] | None = None,
        now: datetime | None = None,
    ) -> list[StoredOpportunity]:
        """Recompute priority for every stored entry, in any status.

        This is the one thing the system is free to do on its own: update
        `priority_score`/`priority_detail`/`scored_at`. It must never move a
        status or touch `approved_by`/`approved_at`/`trial_id` -- enforced
        below by pinning those five fields back onto whatever
        `research_memory.enrich` returns, then re-validating (see
        `_revalidate`), before anything is persisted.

        `source_available` defaults to an empty mapping, not to "assume
        available": an unlisted source's availability is UNKNOWN and must be
        penalised as such by `research_memory.enrich`/`value_of_information`,
        never silently guessed as `True`.

        Persists the *enriched* opportunity (not the original) so that the
        measured `prior_failure_similarity`/`family_redundancy`/
        `pattern_support` estimates `research_memory.enrich` computes are
        durable and reviewable later, not recomputed-and-discarded on every
        call.
        """
        now = now if now is not None else datetime.now(UTC)
        source_available = source_available if source_available is not None else {}

        failed = failed_hypotheses.load(self._failed_hypotheses_path)
        patterns = self._store.list_successful_research_patterns()
        decisions = self._store.list_meta_decisions()

        stored_before = self._store.list_research_opportunities()
        all_opportunities = [s.opportunity for s in stored_before]

        results: list[StoredOpportunity] = []
        for stored in stored_before:
            original = stored.opportunity
            others = [o for o in all_opportunities if o.hypothesis_id != original.hypothesis_id]
            enriched, estimates = research_memory.enrich(
                original,
                failed=failed,
                others=others,
                patterns=patterns,
                decisions=decisions,
                source_available=source_available,
                now=now,
            )
            enriched = _revalidate(
                enriched,
                status=original.status,
                trial_id=original.trial_id,
                approved_by=original.approved_by,
                approved_at=original.approved_at,
                status_note=original.status_note,
            )
            priority = value_of_information.score_opportunity(
                enriched,
                prior_failure_similarity=estimates["prior_failure_similarity"],
                family_redundancy=estimates["family_redundancy"],
                pattern_support=estimates["pattern_support"],
            )
            self._store.upsert_research_opportunity(enriched, priority=priority, scored_at=now)
            results.append(
                StoredOpportunity(opportunity=enriched, priority=priority, scored_at=now)
            )
        return results

    def ranked(self, *, limit: int | None = None) -> list[StoredOpportunity]:
        """Active entries, highest last-computed priority first.

        Delegates ordering to `Store.list_research_opportunities` (`priority_score
        DESC NULLS LAST, hypothesis_id ASC`), so unscored entries sort last
        rather than first -- "never scored yet" is not the same as
        "unimportant", but ranking it above measured work would misrepresent
        it. Excludes `PROMOTED`/`REJECTED`: this method answers "what should
        be worked on next", and a terminal entry is no longer a candidate for
        that -- it belongs in history, not in the queue.
        """
        active = [
            s
            for s in self._store.list_research_opportunities()
            if s.opportunity.status not in _TERMINAL_STATES
        ]
        return active[:limit] if limit is not None else active

    # -- governed lifecycle ----------------------------------------------

    def approve(
        self,
        hypothesis_id: str,
        *,
        approved_by: str,
        note: str,
        trial_id: str | None = None,
        now: datetime | None = None,
    ) -> StoredOpportunity:
        """The only function allowed to move an entry out of `PROPOSED` (§8).

        Requires a named human and a stated reason -- both raise `ValueError`
        if blank or whitespace-only, since an approval without either is not
        distinguishable from the system approving itself.
        """
        if not approved_by.strip():
            raise ValueError(
                "approve requires a non-empty approved_by (§8: human approval "
                "is required to leave PROPOSED)"
            )
        if not note.strip():
            raise ValueError(
                "approve requires a non-empty note (§8: human approval is "
                "required to leave PROPOSED)"
            )
        now = now if now is not None else datetime.now(UTC)
        stored = self._require(hypothesis_id)
        opportunity = stored.opportunity
        if opportunity.status is not ResearchStatus.PROPOSED:
            raise IllegalTransition(
                f"{opportunity.status} -> {ResearchStatus.APPROVED}: approve() only "
                f"accepts entries in PROPOSED; {hypothesis_id!r} is currently "
                f"{opportunity.status} (use transition() for later moves)"
            )
        updated = _revalidate(
            opportunity,
            status=ResearchStatus.APPROVED,
            approved_by=approved_by,
            approved_at=now,
            trial_id=trial_id if trial_id is not None else opportunity.trial_id,
            status_note=_append_note(
                opportunity.status_note, actor=approved_by, note=note, now=now
            ),
        )
        self._store.upsert_research_opportunity(
            updated, priority=stored.priority, scored_at=stored.scored_at
        )
        return StoredOpportunity(
            opportunity=updated, priority=stored.priority, scored_at=stored.scored_at
        )

    def transition(
        self,
        hypothesis_id: str,
        to: ResearchStatus,
        *,
        actor: str,
        note: str,
        now: datetime | None = None,
    ) -> StoredOpportunity:
        """Move an already-approved entry along `ALLOWED_TRANSITIONS`.

        `PROPOSED` is refused as a source even though `PROPOSED -> APPROVED`
        is a legal edge in the table: that specific hop is `approve`'s alone
        (§8), so a caller reaching for the general-purpose mover here cannot
        accidentally take the one path that is supposed to require a human
        name and reason with the stricter checks `approve` applies.

        Moving to `RUNNING` requires `trial_id` to already be set on the
        entry (normally by `approve`, which accepts one) -- a research run
        with no trial id would break Master Spec §23 trial discipline and
        the adjustment budget it feeds.

        `actor` and `note` are both required and appended to `status_note`
        with a timestamp, so the lifecycle history is reviewable without
        needing a separate audit table. `approved_by`/`approved_at` are
        deliberately left untouched by every hop after the first: they
        record who authorised the entry to leave `PROPOSED` at all, which
        would be lost if every later hop overwrote it with whoever happened
        to run that step.
        """
        if not actor.strip():
            raise ValueError("transition requires a non-empty actor")
        if not note.strip():
            raise ValueError("transition requires a non-empty note")
        now = now if now is not None else datetime.now(UTC)
        stored = self._require(hypothesis_id)
        opportunity = stored.opportunity
        frm = opportunity.status
        if frm is ResearchStatus.PROPOSED:
            raise IllegalTransition(
                f"{frm} -> {to}: PROPOSED may only be left via approve() (§8), not transition()"
            )
        if to not in ALLOWED_TRANSITIONS.get(frm, frozenset()):
            raise IllegalTransition(f"{frm} -> {to} is not a legal transition")
        if to is ResearchStatus.RUNNING and not opportunity.trial_id:
            raise ValueError(
                f"{hypothesis_id!r} has no trial_id; moving to RUNNING requires one "
                "to already be set (Master Spec §23 trial discipline / adjustment "
                "budget) -- set it via approve(trial_id=...) first"
            )
        updated = _revalidate(
            opportunity,
            status=to,
            status_note=_append_note(opportunity.status_note, actor=actor, note=note, now=now),
        )
        self._store.upsert_research_opportunity(
            updated, priority=stored.priority, scored_at=stored.scored_at
        )
        return StoredOpportunity(
            opportunity=updated, priority=stored.priority, scored_at=stored.scored_at
        )


__all__ = [
    "ALLOWED_TRANSITIONS",
    "IllegalTransition",
    "ResearchQueue",
]
