"""Tests for `AlphaRegistry` -- above all, that failure memory is durable."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from turboedge.alpha.registry import (
    EVIDENCE_FIELDS,
    AlphaAlreadyRegistered,
    AlphaImmutable,
    AlphaNotFound,
    AlphaRegistry,
    IllegalAlphaTransition,
)
from turboedge.alpha.schemas import AlphaSource, AlphaStatus
from turboedge.meta.research_opportunity import Estimate, InformationFamily
from turboedge.storage.duckdb import Store

_NOW = datetime(2026, 9, 27, 10, 0, tzinfo=UTC)


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[AlphaRegistry]:
    with Store(tmp_path / "turboedge.duckdb") as store:
        store.init_schema()
        yield AlphaRegistry(store)


def _alpha(**over: object) -> AlphaSource:
    unknown = Estimate.unknown("not measured yet")
    defaults: dict[str, object] = dict(
        alpha_id="A-1",
        name="issuer spread widening",
        family=InformationFamily.MICROSTRUCTURE,
        version="1",
        description="d",
        economic_hypothesis="h",
        created_at=_NOW,
        underlyings=["DAX"],
        horizons=["5"],
        directions=["long"],
        required_data_sources=["gettex"],
        nominal_sample=unknown,
        effective_sample=unknown,
        expected_net_ev=unknown,
        lcb_net_ev=unknown,
        posterior_probability_positive=unknown,
        uncertainty_score=unknown,
        drift_score=unknown,
        decay_score=unknown,
    )
    defaults.update(over)
    return AlphaSource(**defaults)  # type: ignore[arg-type]


def _advance(registry: AlphaRegistry, alpha: AlphaSource, *states: AlphaStatus) -> AlphaSource:
    at = _NOW
    out = alpha
    for state in states:
        at += timedelta(minutes=1)
        out = registry.transition(
            alpha.alpha_id,
            alpha.version,
            to_status=state,
            actor="tester",
            note=f"to {state}",
            now=at,
        )
    return out


def test_register_then_read_back_preserves_every_field(registry: AlphaRegistry) -> None:
    alpha = _alpha(
        expected_net_ev=Estimate.measured(0.0031),
        effective_sample=Estimate.measured(118.0),
        trial_ids=["T-1"],
        git_commit="abc123",
        config_hash="cfg",
    )
    registry.register(alpha, actor="tester", now=_NOW)

    stored = registry.get("A-1", "1")
    assert stored == alpha


def test_registration_is_recorded_in_history(registry: AlphaRegistry) -> None:
    registry.register(_alpha(), actor="tester", now=_NOW)

    history = registry.history("A-1", "1")
    assert history == [(_NOW, None, "IDEA", "tester", "registered")]


def test_register_refuses_to_overwrite_an_existing_version(registry: AlphaRegistry) -> None:
    registry.register(_alpha(name="original"), actor="tester", now=_NOW)

    with pytest.raises(AlphaAlreadyRegistered):
        registry.register(_alpha(name="rewritten"), actor="tester", now=_NOW)

    assert registry.get("A-1", "1").name == "original"


def test_register_refuses_a_status_the_alpha_has_not_earned(registry: AlphaRegistry) -> None:
    with pytest.raises(IllegalAlphaTransition):
        registry.register(_alpha(status=AlphaStatus.VALIDATED), actor="tester", now=_NOW)


def test_a_new_version_coexists_with_the_old_one(registry: AlphaRegistry) -> None:
    registry.register(_alpha(version="1"), actor="tester", now=_NOW)
    registry.register(_alpha(version="2", economic_hypothesis="revised"), actor="t", now=_NOW)

    assert {a.version for a in registry.versions_of("A-1")} == {"1", "2"}


def test_get_raises_for_an_unknown_alpha(registry: AlphaRegistry) -> None:
    with pytest.raises(AlphaNotFound):
        registry.get("nope", "1")


def test_history_of_an_unknown_alpha_raises_rather_than_returning_empty(
    registry: AlphaRegistry,
) -> None:
    with pytest.raises(AlphaNotFound):
        registry.history("nope", "1")


def test_transition_follows_the_lifecycle_and_appends_history(registry: AlphaRegistry) -> None:
    alpha = _alpha()
    registry.register(alpha, actor="tester", now=_NOW)

    _advance(registry, alpha, AlphaStatus.EXPLORATORY, AlphaStatus.VALIDATED)

    assert registry.get("A-1", "1").status is AlphaStatus.VALIDATED
    assert [(h[1], h[2]) for h in registry.history("A-1", "1")] == [
        (None, "IDEA"),
        ("IDEA", "EXPLORATORY"),
        ("EXPLORATORY", "VALIDATED"),
    ]


def test_transition_rejects_an_illegal_edge(registry: AlphaRegistry) -> None:
    registry.register(_alpha(), actor="tester", now=_NOW)

    with pytest.raises(IllegalAlphaTransition):
        registry.transition(
            "A-1", "1", to_status=AlphaStatus.NORMAL_PRODUCTION, actor="t", note="n"
        )

    assert registry.get("A-1", "1").status is AlphaStatus.IDEA


def test_transition_rejects_a_no_op(registry: AlphaRegistry) -> None:
    registry.register(_alpha(), actor="tester", now=_NOW)

    with pytest.raises(IllegalAlphaTransition):
        registry.transition("A-1", "1", to_status=AlphaStatus.IDEA, actor="t", note="n")


@pytest.mark.parametrize(("actor", "note"), [("", "n"), ("  ", "n"), ("t", ""), ("t", "   ")])
def test_transition_requires_an_actor_and_a_reason(
    registry: AlphaRegistry, actor: str, note: str
) -> None:
    registry.register(_alpha(), actor="tester", now=_NOW)

    with pytest.raises(ValueError, match="requires a"):
        registry.transition("A-1", "1", to_status=AlphaStatus.EXPLORATORY, actor=actor, note=note)

    assert registry.get("A-1", "1").status is AlphaStatus.IDEA


def test_transition_still_enforces_the_frozen_at_rule(registry: AlphaRegistry) -> None:
    alpha = _alpha()
    registry.register(alpha, actor="tester", now=_NOW)
    _advance(registry, alpha, AlphaStatus.EXPLORATORY, AlphaStatus.VALIDATED)

    # CONFIRMATORY without frozen_at must fail in AlphaSource's own validator,
    # which only runs because transition() rebuilds rather than model_copy()s.
    with pytest.raises(ValueError, match="frozen_at"):
        registry.transition(
            "A-1", "1", to_status=AlphaStatus.CONFIRMATORY, actor="t", note="confirm"
        )

    assert registry.get("A-1", "1").status is AlphaStatus.VALIDATED


def test_a_frozen_alpha_can_reach_confirmatory(registry: AlphaRegistry) -> None:
    alpha = _alpha(frozen_at=_NOW)
    registry.register(alpha, actor="tester", now=_NOW)

    out = _advance(
        registry,
        alpha,
        AlphaStatus.EXPLORATORY,
        AlphaStatus.VALIDATED,
        AlphaStatus.CONFIRMATORY,
    )

    assert out.status is AlphaStatus.CONFIRMATORY


def test_rejection_is_permanent(registry: AlphaRegistry) -> None:
    alpha = _alpha()
    registry.register(alpha, actor="tester", now=_NOW)
    _advance(registry, alpha, AlphaStatus.REJECTED)

    with pytest.raises(AlphaImmutable):
        registry.transition("A-1", "1", to_status=AlphaStatus.EXPLORATORY, actor="t", note="retry")
    with pytest.raises(AlphaImmutable):
        registry.record_evidence("A-1", "1", evidence={"expected_net_ev": Estimate.measured(0.05)})

    stored = registry.get("A-1", "1")
    assert stored.status is AlphaStatus.REJECTED
    assert not stored.expected_net_ev.is_known


def test_registry_exposes_no_way_to_delete_an_alpha() -> None:
    # A failed alpha that can be removed is not failure memory (spec §7).
    assert not [n for n in dir(AlphaRegistry) if "delete" in n or "remove" in n]


def test_disable_is_reachable_from_a_production_state(registry: AlphaRegistry) -> None:
    alpha = _alpha(frozen_at=_NOW)
    registry.register(alpha, actor="tester", now=_NOW)
    _advance(
        registry,
        alpha,
        AlphaStatus.EXPLORATORY,
        AlphaStatus.VALIDATED,
        AlphaStatus.CONFIRMATORY,
        AlphaStatus.FORWARD_SHADOW,
        AlphaStatus.CANARY_PRODUCTION,
    )

    out = registry.disable(
        "A-1", "1", actor="human", note="spread blowout", now=_NOW + timedelta(hours=1)
    )

    assert out.status is AlphaStatus.DISABLED
    assert registry.history("A-1", "1")[-1][1:4] == ("CANARY_PRODUCTION", "DISABLED", "human")


def test_record_evidence_updates_estimates_without_moving_status(registry: AlphaRegistry) -> None:
    alpha = _alpha()
    registry.register(alpha, actor="tester", now=_NOW)
    _advance(registry, alpha, AlphaStatus.EXPLORATORY)

    out = registry.record_evidence(
        "A-1",
        "1",
        evidence={
            "expected_net_ev": Estimate.measured(0.0042),
            "lcb_net_ev": Estimate.measured(-0.0011),
        },
        trial_ids=["T-9"],
    )

    assert out.status is AlphaStatus.EXPLORATORY
    assert out.expected_net_ev.value == pytest.approx(0.0042)
    assert out.lcb_net_ev.value == pytest.approx(-0.0011)
    assert out.trial_ids == ["T-9"]
    assert len(registry.history("A-1", "1")) == 2  # evidence is not a status event


def test_record_evidence_unions_trial_ids_rather_than_replacing(registry: AlphaRegistry) -> None:
    registry.register(_alpha(trial_ids=["T-1"]), actor="tester", now=_NOW)

    out = registry.record_evidence("A-1", "1", evidence={}, trial_ids=["T-1", "T-2"])

    assert out.trial_ids == ["T-1", "T-2"]


def test_record_evidence_refuses_non_evidence_fields(registry: AlphaRegistry) -> None:
    registry.register(_alpha(), actor="tester", now=_NOW)

    with pytest.raises(ValueError, match="not evidence fields"):
        registry.record_evidence(
            "A-1", "1", evidence={"economic_hypothesis": Estimate.measured(1.0)}
        )


def test_evidence_fields_match_the_schema(registry: AlphaRegistry) -> None:
    # If an evidence estimate is added to AlphaSource, record_evidence must
    # learn about it here rather than silently refusing to write it.
    estimate_fields = {
        name for name, field in AlphaSource.model_fields.items() if field.annotation is Estimate
    }
    assert set(EVIDENCE_FIELDS) == estimate_fields


def test_list_and_filter_by_status(registry: AlphaRegistry) -> None:
    a = _alpha(alpha_id="A-1")
    b = _alpha(alpha_id="A-2")
    registry.register(a, actor="t", now=_NOW)
    registry.register(b, actor="t", now=_NOW + timedelta(seconds=1))
    _advance(registry, b, AlphaStatus.EXPLORATORY)

    assert {x.alpha_id for x in registry.list_all()} == {"A-1", "A-2"}
    assert [x.alpha_id for x in registry.list_all(status=AlphaStatus.IDEA)] == ["A-1"]
    assert [x.alpha_id for x in registry.in_states([AlphaStatus.EXPLORATORY])] == ["A-2"]


def test_evidence_survives_a_store_reopen(tmp_path: Path) -> None:
    path = tmp_path / "turboedge.duckdb"
    with Store(path) as store:
        store.init_schema()
        AlphaRegistry(store).register(
            _alpha(expected_net_ev=Estimate.declared(0.01, "prior")), actor="t", now=_NOW
        )
    with Store(path) as store:
        store.init_schema()
        stored = AlphaRegistry(store).get("A-1", "1")

    assert stored.expected_net_ev.value == pytest.approx(0.01)
    assert stored.expected_net_ev.note == "prior"
    assert not stored.drift_score.is_known
