"""Pure exponential ensemble-weight update math (Master Spec §21).

Kept separate from ``learning/registry.py`` (which owns persistence and the
`ModelRegistry.update_weights` orchestration) so the weighting arithmetic
itself is trivially unit-testable without a `Store`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def update_weights(
    current_weights: Mapping[str, float],
    utilities: Mapping[str, float],
    eta: float = 0.5,
    w_min: float = 0.01,
) -> dict[str, float]:
    """Exponential ensemble reweighting (Master Spec §21)::

        w_i,t+1 ∝ w_i,t * exp(eta * utility_i,t)

    normalized to sum to 1, then floored at ``w_min`` via "water-filling":
    any model whose normalized weight would fall below ``w_min`` is pinned
    at exactly ``w_min``, and the remaining probability mass is
    re-distributed proportionally among the rest (repeated until no further
    model falls below the floor). This guarantees ``sum(weights) == 1``
    *and* every weight ``>= w_min`` exactly (up to floating-point epsilon),
    unlike a single floor-then-renormalize pass, which can pull a floored
    weight slightly back under ``w_min`` on the final renormalization.

    Old models can regain relevance under this scheme when a regime shift
    makes their utility positive again ("Alte Modelle können dadurch bei
    Regimewechsel wieder relevant werden.") -- and a model's weight can
    shrink (even the protected TSMOM baseline's) without the model itself
    ever being removed from ``current_weights``/the registry (CLAUDE.md
    rule 10 / rule 32).

    ``current_weights`` and ``utilities`` must share exactly the same key
    set: the models being reweighted this round. A model absent from a
    given round's ``utilities`` is left untouched by the caller
    (``ModelRegistry.update_weights``) -- this function only ever computes
    new weights for the models it is given.

    Raises:
        ValueError: If the two mappings' key sets differ, either is empty,
            any current weight is not strictly positive (a non-positive
            weight can never recover under exponential reweighting, which
            is a modeling error upstream, not something this function
            should silently paper over), or ``w_min * len(current_weights)
            > 1`` (no floor-respecting distribution summing to 1 exists).
    """
    if set(current_weights) != set(utilities):
        raise ValueError(
            "current_weights and utilities must cover exactly the same model "
            f"ids (got {sorted(current_weights)} vs {sorted(utilities)})"
        )
    if not current_weights:
        raise ValueError("current_weights/utilities must not be empty")

    model_ids = list(current_weights)
    n = len(model_ids)
    if w_min * n > 1.0 + 1e-9:
        raise ValueError(
            f"w_min ({w_min}) * n_models ({n}) exceeds 1: no distribution over "
            "these models can respect the floor and still sum to 1"
        )

    weights_arr = np.array([current_weights[m] for m in model_ids], dtype=np.float64)
    if not np.all(weights_arr > 0):
        raise ValueError("all current_weights must be > 0")

    utilities_arr = np.array([utilities[m] for m in model_ids], dtype=np.float64)
    raw = weights_arr * np.exp(eta * utilities_arr)
    raw_map = {m: float(r) for m, r in zip(model_ids, raw, strict=True)}

    return _water_fill_floor(raw_map, w_min)


def _water_fill_floor(raw: Mapping[str, float], w_min: float) -> dict[str, float]:
    """Normalize ``raw`` (positive, unnormalized weights) to sum to 1 while
    guaranteeing every result is ``>= w_min``, by iteratively pinning any
    model that would fall below the floor at exactly ``w_min`` and
    re-normalizing the remaining probability mass across the rest.

    Terminates in at most ``len(raw)`` iterations (each iteration pins at
    least one more model, or finishes).
    """
    remaining_ids = list(raw)
    remaining_mass = 1.0
    result: dict[str, float] = {}

    for _ in range(len(raw)):
        if not remaining_ids:
            break
        total_raw = sum(raw[m] for m in remaining_ids)
        proportional = {m: (raw[m] / total_raw) * remaining_mass for m in remaining_ids}
        newly_floored = [m for m in remaining_ids if proportional[m] < w_min]
        if not newly_floored:
            result.update(proportional)
            remaining_ids = []
            break
        for m in newly_floored:
            result[m] = w_min
        remaining_mass -= w_min * len(newly_floored)
        remaining_ids = [m for m in remaining_ids if m not in newly_floored]

    # Any ids left over after the loop bound (should not happen given the
    # per-iteration progress guarantee) fall back to an equal split of
    # whatever mass remains, so the function always returns a complete,
    # sum-to-1 result rather than raising on a pathological edge case.
    if remaining_ids:
        share = remaining_mass / len(remaining_ids)
        for m in remaining_ids:
            result[m] = share

    return result


__all__ = ["update_weights"]
