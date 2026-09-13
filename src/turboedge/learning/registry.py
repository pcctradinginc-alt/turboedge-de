"""Model registry: champion/challenger/dormant/protected lifecycle (Master
Spec §20-21).

Selection between forecast models is one of the ways the system "learns"
without rewriting source code (Master Spec §20): weights shift, statuses
change, but the protected TSMOM baseline is always registered and visible
(CLAUDE.md rule 10), and promotion only ever happens through the Ladder Rule
(Master Spec §27.3, `promote_if_ladder`).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from turboedge.learning.ensemble_weights import update_weights as _compute_weights
from turboedge.storage.duckdb import Store, StoreError
from turboedge.storage.schemas import ModelRegistryEntry, ModelStatus


class ModelRegistry:
    """Thin, persistence-backed wrapper around the ``model_registry`` /
    ``model_weight_history`` tables.

    Usage::

        registry = ModelRegistry(store)
        registry.register("tsmom_horizon_norm_v1", model_hash, "tsmom", {}, None,
                           status=ModelStatus.PROTECTED)
        registry.register("logit_v1", model_hash2, "logit", {"C": 1.0}, trial_id)
        registry.update_weights({"tsmom_horizon_norm_v1": 0.02, "logit_v1": -0.01})
        weights = registry.weights()
    """

    def __init__(self, store: Store) -> None:
        self.store = store

    def register(
        self,
        model_id: str,
        model_hash: str,
        signal_family: str,
        params: dict[str, Any],
        trial_id: str | None,
        *,
        status: ModelStatus = ModelStatus.CHALLENGER,
        initial_weight: float = 1.0,
        now: datetime | None = None,
    ) -> ModelRegistryEntry:
        """Register a new model, or re-register an existing ``model_id``
        (e.g. after a retrain that changed ``model_hash``/``params``).

        Re-registering preserves the existing row's ``created_at`` and
        ``status`` is only changed if explicitly passed (defaults to
        ``CHALLENGER``, so re-registering never silently un-promotes/
        un-protects a model -- callers that want to keep an existing
        model's status across a re-register should read it first via
        :meth:`get` and pass it back in).
        """
        ts = now if now is not None else datetime.now(UTC)
        entry = ModelRegistryEntry(
            model_id=model_id,
            model_hash=model_hash,
            signal_family=signal_family,
            status=status,
            weight=initial_weight,
            params=params,
            trial_id=trial_id,
            created_at=ts,
            updated_at=ts,
        )
        self.store.upsert_model_registry_entry(entry)
        stored = self.store.get_model_registry_entry(model_id)
        assert stored is not None  # just upserted
        return stored

    def get(self, model_id: str) -> ModelRegistryEntry | None:
        return self.store.get_model_registry_entry(model_id)

    def list(self, signal_family: str | None = None) -> list[ModelRegistryEntry]:
        return self.store.list_model_registry_entries(signal_family)

    def set_status(
        self,
        model_id: str,
        status: ModelStatus,
        *,
        trial_id: str | None = None,
        now: datetime | None = None,
    ) -> ModelRegistryEntry:
        """Change one model's status.

        Refuses (``ValueError``) to move a currently ``PROTECTED`` model to
        any other status -- protected baselines are never demoted or
        deleted (CLAUDE.md rule 10); the reverse (promoting some other
        model *to* ``PROTECTED``) is likewise refused, since "protected" is
        a designation made once at registration, not something a
        promotion/demotion round should be able to grant.
        """
        existing = self.store.get_model_registry_entry(model_id)
        if existing is None:
            raise StoreError(f"cannot set status: no registered model {model_id!r}")
        if existing.status is ModelStatus.PROTECTED and status is not ModelStatus.PROTECTED:
            raise ValueError(
                f"cannot change status of protected model {model_id!r} away from "
                "protected (CLAUDE.md rule 10: 'Protected TSMOM Baseline niemals "
                "löschen.')"
            )
        if status is ModelStatus.PROTECTED and existing.status is not ModelStatus.PROTECTED:
            raise ValueError(
                f"cannot grant protected status to {model_id!r} via set_status; "
                "protected status is only assigned at register() time"
            )
        ts = now if now is not None else datetime.now(UTC)
        updated = existing.model_copy(
            update={
                "status": status,
                "trial_id": trial_id if trial_id is not None else existing.trial_id,
                "updated_at": ts,
            }
        )
        self.store.upsert_model_registry_entry(updated)
        return updated

    def weights(self) -> dict[str, float]:
        """Current weight of every registered model, protected/dormant
        ones included -- "Protected Baseline bleibt unabhängig davon immer
        sichtbar" (Master Spec §21)."""
        return {entry.model_id: entry.weight for entry in self.list()}

    def update_weights(
        self,
        utilities: Mapping[str, float],
        eta: float = 0.5,
        w_min: float = 0.01,
        *,
        trial_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, float]:
        """Exponentially reweight exactly the models present in
        ``utilities`` (Master Spec §21: ``w_i,t+1 ∝ w_i,t * exp(eta *
        utility_i,t)``, floored at ``w_min`` and renormalized -- see
        :func:`turboedge.learning.ensemble_weights.update_weights` for the
        pure math).

        A model not present in ``utilities`` is not evaluated this round
        and keeps its current weight. Every reweighted model's new weight
        is recorded in ``model_weight_history`` (append-only audit trail),
        tagged with ``trial_id`` when this reweighting round is itself part
        of a research trial. Returns the full, updated ``weights()``
        (including models untouched this round).

        Raises:
            StoreError: If ``utilities`` names a model that was never
                registered.
        """
        if not utilities:
            return self.weights()
        ts = now if now is not None else datetime.now(UTC)
        current: dict[str, float] = {}
        entries: dict[str, ModelRegistryEntry] = {}
        for model_id in utilities:
            entry = self.store.get_model_registry_entry(model_id)
            if entry is None:
                raise StoreError(f"cannot reweight unregistered model {model_id!r}")
            entries[model_id] = entry
            current[model_id] = entry.weight

        new_weights = _compute_weights(current, utilities, eta=eta, w_min=w_min)

        for model_id, new_weight in new_weights.items():
            entry = entries[model_id]
            updated = entry.model_copy(update={"weight": new_weight, "updated_at": ts})
            self.store.upsert_model_registry_entry(updated)
            self.store.append_model_weight_history(
                model_id,
                new_weight,
                utility=utilities[model_id],
                trial_id=trial_id,
                recorded_at=ts,
            )
        return self.weights()


def promote_if_ladder(
    registry: ModelRegistry,
    challenger_model_id: str,
    champion_model_id: str | None,
    oos_improvement: float,
    min_improvement: float,
    *,
    trial_id: str,
    now: datetime | None = None,
) -> bool:
    """Ladder Rule promotion (Master Spec §27.3, GOVERNANCE.md §2): promote
    ``challenger_model_id`` to ``champion`` only if its out-of-sample
    improvement exceeds ``min_improvement``.

    CLAUDE.md rule 26 ("Keine Verbesserung nur anhand In-Sample behaupten"):
    ``oos_improvement`` must already be an out-of-sample number -- this
    function has no way to check that itself, callers (the walk-forward /
    significance-testing code) are responsible for that.

    If promoted and there was a previous champion, that model is demoted to
    ``challenger`` (never ``dormant`` merely for having been replaced --
    demotion-to-dormant is a separate, performance-triggered decision, see
    GOVERNANCE.md §6.2). A ``PROTECTED`` champion is never displaced by
    this function -- raises ``ValueError`` if ``champion_model_id`` refers
    to one, since the protected baseline is a permanent benchmark, not a
    slot promotion ever vacates.

    Returns:
        ``True`` iff the promotion happened.
    """
    if oos_improvement < min_improvement:
        return False
    if champion_model_id is not None:
        champion = registry.get(champion_model_id)
        if champion is not None:
            if champion.status is ModelStatus.PROTECTED:
                raise ValueError(
                    f"cannot promote {challenger_model_id!r} over protected model "
                    f"{champion_model_id!r}; protected baselines are never "
                    "displaced (CLAUDE.md rule 10)"
                )
            registry.set_status(
                champion_model_id, ModelStatus.CHALLENGER, trial_id=trial_id, now=now
            )
    registry.set_status(challenger_model_id, ModelStatus.CHAMPION, trial_id=trial_id, now=now)
    return True


__all__ = ["ModelRegistry", "promote_if_ladder"]
