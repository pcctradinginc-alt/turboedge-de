from __future__ import annotations

import pytest

from turboedge.learning.ensemble_weights import update_weights as compute_weights
from turboedge.learning.registry import ModelRegistry, promote_if_ladder
from turboedge.storage.duckdb import StoreError
from turboedge.storage.schemas import ModelStatus

# -- pure math (ensemble_weights.compute_weights) ---------------------------


def test_compute_weights_normalizes_to_one() -> None:
    current = {"a": 0.5, "b": 0.5}
    utilities = {"a": 0.02, "b": -0.01}
    result = compute_weights(current, utilities, eta=0.5, w_min=0.01)
    assert sum(result.values()) == pytest.approx(1.0)


def test_compute_weights_higher_utility_gets_higher_weight_monotonic() -> None:
    current = {"a": 1.0, "b": 1.0, "c": 1.0}
    utilities = {"a": -0.02, "b": 0.0, "c": 0.03}
    result = compute_weights(current, utilities, eta=1.0, w_min=0.01)
    assert result["a"] < result["b"] < result["c"]


def test_compute_weights_floor_holds_when_w_min_small_relative_to_n() -> None:
    current = {f"m{i}": 1.0 for i in range(5)}
    # One model gets crushed by a very negative utility.
    utilities = {f"m{i}": (-10.0 if i == 0 else 0.0) for i in range(5)}
    result = compute_weights(current, utilities, eta=1.0, w_min=0.01)
    assert all(w >= 0.01 - 1e-9 for w in result.values())
    assert sum(result.values()) == pytest.approx(1.0)


def test_compute_weights_requires_matching_key_sets() -> None:
    with pytest.raises(ValueError, match="same model ids"):
        compute_weights({"a": 1.0}, {"b": 1.0})


def test_compute_weights_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compute_weights({}, {})


def test_compute_weights_rejects_non_positive_current_weight() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        compute_weights({"a": 0.0}, {"a": 0.1})


# -- ModelRegistry -----------------------------------------------------------


def test_register_and_get_roundtrip(store) -> None:
    registry = ModelRegistry(store)
    entry = registry.register("logit_v1", "hash1", "logit", {"C": 1.0}, "TR-1")
    fetched = registry.get("logit_v1")
    assert fetched is not None
    assert fetched == entry
    assert fetched.status == ModelStatus.CHALLENGER
    assert fetched.weight == 1.0


def test_weights_includes_all_registered_models_regardless_of_status(store) -> None:
    registry = ModelRegistry(store)
    registry.register("tsmom", "h0", "tsmom", {}, None, status=ModelStatus.PROTECTED)
    registry.register("logit_v1", "h1", "logit", {}, "TR-1")
    registry.set_status("logit_v1", ModelStatus.DORMANT)

    weights = registry.weights()
    assert set(weights) == {"tsmom", "logit_v1"}


def test_protected_model_can_lose_weight_but_never_status(store) -> None:
    registry = ModelRegistry(store)
    registry.register("tsmom", "h0", "tsmom", {}, None, status=ModelStatus.PROTECTED)
    registry.register("logit_v1", "h1", "logit", {}, "TR-1")

    updated = registry.update_weights({"tsmom": -0.05, "logit_v1": 0.05}, eta=1.0)
    assert updated["tsmom"] < updated["logit_v1"]
    # still registered, still protected, never removed
    fetched = registry.get("tsmom")
    assert fetched is not None
    assert fetched.status == ModelStatus.PROTECTED
    assert fetched.weight > 0.0


def test_set_status_refuses_to_demote_protected_model(store) -> None:
    registry = ModelRegistry(store)
    registry.register("tsmom", "h0", "tsmom", {}, None, status=ModelStatus.PROTECTED)
    with pytest.raises(ValueError, match="protected"):
        registry.set_status("tsmom", ModelStatus.DORMANT)


def test_set_status_refuses_to_grant_protected_via_set_status(store) -> None:
    registry = ModelRegistry(store)
    registry.register("logit_v1", "h1", "logit", {}, "TR-1")
    with pytest.raises(ValueError, match="protected"):
        registry.set_status("logit_v1", ModelStatus.PROTECTED)


def test_update_weights_records_history(store) -> None:
    registry = ModelRegistry(store)
    registry.register("logit_v1", "h1", "logit", {}, "TR-1")
    registry.update_weights({"logit_v1": 0.02}, trial_id="TR-2")
    history = store.model_weight_history("logit_v1")
    assert len(history) == 1
    _recorded_at, weight, utility, trial_id = history[0]
    assert weight == pytest.approx(1.0)  # only one model -> normalizes to 1.0
    assert utility == pytest.approx(0.02)
    assert trial_id == "TR-2"


def test_update_weights_unregistered_model_raises(store) -> None:
    registry = ModelRegistry(store)
    with pytest.raises(StoreError):
        registry.update_weights({"never-registered": 0.1})


def test_update_weights_empty_utilities_is_noop(store) -> None:
    registry = ModelRegistry(store)
    registry.register("logit_v1", "h1", "logit", {}, "TR-1")
    result = registry.update_weights({})
    assert result == {"logit_v1": 1.0}


# -- promote_if_ladder --------------------------------------------------------


def test_promote_if_ladder_promotes_when_improvement_exceeds_minimum(store) -> None:
    registry = ModelRegistry(store)
    registry.register("champion_v1", "h0", "logit", {}, "TR-1", status=ModelStatus.CHAMPION)
    registry.register("challenger_v1", "h1", "logit", {}, "TR-2")

    promoted = promote_if_ladder(
        registry,
        "challenger_v1",
        "champion_v1",
        oos_improvement=0.002,
        min_improvement=0.001,
        trial_id="TR-3",
    )
    assert promoted is True
    assert registry.get("challenger_v1").status == ModelStatus.CHAMPION
    assert registry.get("champion_v1").status == ModelStatus.CHALLENGER


def test_promote_if_ladder_rejects_below_minimum_improvement(store) -> None:
    registry = ModelRegistry(store)
    registry.register("champion_v1", "h0", "logit", {}, "TR-1", status=ModelStatus.CHAMPION)
    registry.register("challenger_v1", "h1", "logit", {}, "TR-2")

    promoted = promote_if_ladder(
        registry,
        "challenger_v1",
        "champion_v1",
        oos_improvement=0.0001,
        min_improvement=0.001,
        trial_id="TR-3",
    )
    assert promoted is False
    assert registry.get("challenger_v1").status == ModelStatus.CHALLENGER
    assert registry.get("champion_v1").status == ModelStatus.CHAMPION


def test_promote_if_ladder_never_displaces_protected_champion(store) -> None:
    registry = ModelRegistry(store)
    registry.register("tsmom", "h0", "tsmom", {}, None, status=ModelStatus.PROTECTED)
    registry.register("challenger_v1", "h1", "logit", {}, "TR-2")

    with pytest.raises(ValueError, match="protected"):
        promote_if_ladder(
            registry,
            "challenger_v1",
            "tsmom",
            oos_improvement=1.0,
            min_improvement=0.001,
            trial_id="TR-3",
        )


def test_promote_if_ladder_with_no_prior_champion(store) -> None:
    registry = ModelRegistry(store)
    registry.register("challenger_v1", "h1", "logit", {}, "TR-2")
    promoted = promote_if_ladder(
        registry,
        "challenger_v1",
        None,
        oos_improvement=0.01,
        min_improvement=0.001,
        trial_id="TR-3",
    )
    assert promoted is True
    fetched = registry.get("challenger_v1")
    assert fetched is not None
    assert fetched.status == ModelStatus.CHAMPION
