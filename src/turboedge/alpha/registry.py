"""The alpha registry -- what has been tried, how strong, and what failed.

`ModelRegistry` answers "which predictor is champion". It cannot answer the
questions that decide whether this project ever earns anything: which
economically measurable effects have been tested, how large they were, how
uncertain, on which underlyings and horizons they held, whether they are
decaying, and -- most importantly -- which ones were tried and did **not**
work.

That last one is the reason this module refuses several things a normal CRUD
layer would allow:

* A registered `(alpha_id, version)` cannot be overwritten. New evidence for
  the same frozen hypothesis goes through `record_evidence`, which may only
  touch the eight evidence estimates; a changed hypothesis is a new version.
* A REJECTED alpha is immutable. No status move, no evidence update, no
  deletion. There is no `delete` in this module at all. Failure memory that
  can be edited away is not memory (Master Spec §7, §55.3, §55.4).
* Every status change appends to `alpha_status_history` rather than replacing
  the previous one, so "it was in production, then it was rejected, then it
  came back" stays visible instead of collapsing into its last frame.

Phase A scope: this stores and constrains. It evaluates nothing. There is no
promotion decision here and `promotion.PromotionCriteria` still has seven
unset gates -- an alpha reaches CANARY_PRODUCTION only because a caller that
does not yet exist said so, and that caller cannot be written until those
gates have measured values.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from turboedge.alpha.schemas import (
    TERMINAL_STATES,
    AlphaSource,
    AlphaStatus,
    transition_allowed,
)
from turboedge.meta.research_opportunity import Estimate

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    # `storage.duckdb` imports `AlphaSource` from `alpha.schemas`, so importing
    # `Store` at runtime would close the loop duckdb -> alpha -> duckdb. The
    # registry only needs the type; it is always handed a live store.
    from turboedge.storage.duckdb import Store


class AlphaNotFound(KeyError):
    """No such `(alpha_id, version)` in the registry."""


class AlphaAlreadyRegistered(Exception):
    """`register` was called for an `(alpha_id, version)` that already exists.

    Raised rather than upserted on purpose: the overwrite this prevents is
    exactly how a failed alpha would disappear.
    """


class IllegalAlphaTransition(Exception):
    """A status move that `transition_allowed` rejects, or one out of a
    terminal state."""


class AlphaImmutable(Exception):
    """An attempt to change an alpha that is permanently frozen (REJECTED)."""


#: The only status a newly registered alpha may carry. Everything past IDEA
#: has to be reached through `transition`, which records who moved it and why.
#: Registering something directly as VALIDATED would let an alpha acquire a
#: history it never earned.
REGISTRABLE_STATES: frozenset[AlphaStatus] = frozenset({AlphaStatus.IDEA})

#: The evidence fields `record_evidence` is allowed to touch. Deliberately not
#: "every float field": the hypothesis, the universe, the freeze timestamp and
#: the status are not evidence and must not drift as measurements come in.
EVIDENCE_FIELDS: tuple[str, ...] = (
    "nominal_sample",
    "effective_sample",
    "expected_net_ev",
    "lcb_net_ev",
    "posterior_probability_positive",
    "uncertainty_score",
    "drift_score",
    "decay_score",
)


class AlphaRegistry:
    """Append-mostly store of alpha sources and their status history."""

    def __init__(self, store: Store) -> None:
        self._store = store

    # -- reads -------------------------------------------------------------

    def get(self, alpha_id: str, version: str) -> AlphaSource:
        alpha = self._store.get_alpha_source(alpha_id, version)
        if alpha is None:
            raise AlphaNotFound(f"{alpha_id}@{version}")
        return alpha

    def list_all(self, *, status: AlphaStatus | None = None) -> list[AlphaSource]:
        return self._store.list_alpha_sources(status=None if status is None else str(status))

    def in_states(self, states: Sequence[AlphaStatus]) -> list[AlphaSource]:
        wanted = frozenset(states)
        return [a for a in self._store.list_alpha_sources() if a.status in wanted]

    def history(
        self, alpha_id: str, version: str
    ) -> list[tuple[datetime, str | None, str, str, str]]:
        """`(recorded_at, from_status, to_status, actor, note)`, oldest first."""
        self.get(alpha_id, version)  # raises AlphaNotFound rather than returning []
        return self._store.list_alpha_status_history(alpha_id, version)

    def versions_of(self, alpha_id: str) -> list[AlphaSource]:
        return [a for a in self._store.list_alpha_sources() if a.alpha_id == alpha_id]

    # -- writes ------------------------------------------------------------

    def register(self, alpha: AlphaSource, *, actor: str, now: datetime | None = None) -> None:
        """Add a new alpha at IDEA. Refuses to overwrite an existing version."""
        if alpha.status not in REGISTRABLE_STATES:
            raise IllegalAlphaTransition(
                f"an alpha may only be registered in {sorted(REGISTRABLE_STATES)}, "
                f"not {alpha.status}: every later state has to be reached through "
                "transition() so that who moved it, when, and why is recorded"
            )
        if self._store.get_alpha_source(alpha.alpha_id, alpha.version) is not None:
            raise AlphaAlreadyRegistered(
                f"{alpha.alpha_id}@{alpha.version} already exists; register a new "
                "version instead of overwriting the evidence behind the old one"
            )
        at = _now(now)
        self._store.upsert_alpha_source(alpha)
        self._store.append_alpha_status_change(
            alpha_id=alpha.alpha_id,
            version=alpha.version,
            recorded_at=at,
            from_status=None,
            to_status=str(alpha.status),
            actor=actor,
            note="registered",
        )

    def transition(
        self,
        alpha_id: str,
        version: str,
        *,
        to_status: AlphaStatus,
        actor: str,
        note: str,
        now: datetime | None = None,
    ) -> AlphaSource:
        """Move an alpha along a legal lifecycle edge, appending to history.

        `actor` and `note` are required and must be non-empty. A status change
        with no stated reason cannot be reviewed later, and every review this
        project has survived turned on being able to reconstruct why a decision
        was made at the time rather than afterwards.
        """
        alpha = self.get(alpha_id, version)
        if not actor.strip():
            raise ValueError("transition requires a named actor")
        if not note.strip():
            raise ValueError("transition requires a stated reason")
        if alpha.status in TERMINAL_STATES:
            raise AlphaImmutable(
                f"{alpha_id}@{version} is {alpha.status}, which is terminal: a "
                "recorded failure stays recorded"
            )
        if to_status is alpha.status:
            raise IllegalAlphaTransition(
                f"{alpha_id}@{version} is already {to_status}; a no-op transition "
                "would add a history entry that describes nothing"
            )
        if not transition_allowed(alpha.status, to_status):
            raise IllegalAlphaTransition(
                f"{alpha.status} -> {to_status} is not a legal move for {alpha_id}@{version}"
            )
        at = _now(now)
        # Reconstructed through the constructor rather than `model_copy` so
        # that AlphaSource's own validators run -- in particular the one that
        # refuses CONFIRMATORY and production states without `frozen_at`.
        updated = AlphaSource(**{**alpha.model_dump(), "status": to_status})
        self._store.upsert_alpha_source(updated)
        self._store.append_alpha_status_change(
            alpha_id=alpha_id,
            version=version,
            recorded_at=at,
            from_status=str(alpha.status),
            to_status=str(to_status),
            actor=actor,
            note=note,
        )
        return updated

    def disable(
        self,
        alpha_id: str,
        version: str,
        *,
        actor: str,
        note: str,
        now: datetime | None = None,
    ) -> AlphaSource:
        """Emergency stop. Legal from any non-terminal state.

        Separate from `transition` only so that it reads as what it is at the
        call site; it goes through the same recording path.
        """
        return self.transition(
            alpha_id, version, to_status=AlphaStatus.DISABLED, actor=actor, note=note, now=now
        )

    def record_evidence(
        self,
        alpha_id: str,
        version: str,
        *,
        evidence: dict[str, Estimate],
        trial_ids: Sequence[str] = (),
    ) -> AlphaSource:
        """Update measured evidence for an existing version. Status unchanged.

        Only `EVIDENCE_FIELDS` may be written, and `trial_ids` is unioned
        rather than replaced: the trials that produced an earlier, weaker
        reading are part of the record. A status move is a separate,
        separately-recorded act -- evidence arriving is not a promotion.
        """
        alpha = self.get(alpha_id, version)
        if alpha.status in TERMINAL_STATES:
            raise AlphaImmutable(
                f"{alpha_id}@{version} is {alpha.status}: its evidence is the record "
                "of why it was rejected and is not editable"
            )
        unknown = sorted(set(evidence) - set(EVIDENCE_FIELDS))
        if unknown:
            raise ValueError(
                f"not evidence fields: {unknown}; record_evidence may only touch "
                f"{list(EVIDENCE_FIELDS)}, because the hypothesis, universe and freeze "
                "timestamp must not drift as measurements arrive"
            )
        merged = list(alpha.trial_ids) + [t for t in trial_ids if t not in alpha.trial_ids]
        updated = AlphaSource(**{**alpha.model_dump(), **evidence, "trial_ids": merged})
        self._store.upsert_alpha_source(updated)
        return updated


def _now(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(UTC)
