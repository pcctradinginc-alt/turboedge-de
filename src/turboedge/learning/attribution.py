"""Which model said what, and how it turned out.

WHY THIS IS A JOIN AND NOT A NEW TABLE
--------------------------------------
The operator asked for the ensemble's composition and its
prediction-time weights to be stored immutably so that each model can
later be evaluated against the real outcome. Almost all of it already is.
``pipeline/scan.py`` writes one ``forecasts`` row per
``(run_id, underlying_id, horizon_days, model_id)`` for every component
model, plus an ``ensemble`` row whose ``component_weights`` column holds
the weight map as it stood when the forecast was made. Each row carries
the model's id, its hash, its own mean/sigma/quantiles, ``n_train``,
``n_effective``, ``prediction_time``, ``frozen_at``, ``config_hash`` and
``git_commit``. Inserts are ``ON CONFLICT ... DO NOTHING``, so a row is
written once and never rewritten.

What was missing is the join. ``LedgerEntry`` carries ``run_id``,
``underlying`` and ``horizon_days``, which is exactly the key those rows
are stored under -- so nothing needed duplicating, only connecting.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not split the ensemble's realised P&L across its members. That
number belongs to one decision, taken by the combination; apportioning it
by weight would invent a counterfactual in which each model traded alone,
which it did not. `cli_learn.py` reports `labeled_entries_unattributable`
precisely so that the absence stays visible.

What it does instead is evaluate each model's **own forecast** against the
realised outcome -- its predicted mean against the realised return, its
``p_up`` against the realised direction. That is a model-level
measurement, it needs no counterfactual, and it is what makes a later
weight update rest on evidence rather than on apportionment.

It also updates no weight. Turning these scores into weights is a
governance step with its own statistical requirements
(`GOVERNANCE.md`), and this module only measures.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from turboedge.storage.schemas import ForecastRecord, LedgerEntry, LedgerLabel

#: The ``model_id`` under which ``pipeline/scan.py`` writes the combined
#: forecast. Everything else in a run's rows is a component.
ENSEMBLE_MODEL_ID = "ensemble"


@dataclass(frozen=True, slots=True)
class ComponentForecast:
    """One model's own forecast, as it stood at prediction time."""

    model_id: str
    model_hash: str
    signal_family: str
    mean: float
    sigma: float | None
    p_up: float | None
    weight_at_prediction: float | None
    n_effective: float | None


@dataclass(frozen=True, slots=True)
class EnsembleAttribution:
    """The composition behind one ledger entry, resolved from the archive."""

    entry_id: str
    run_id: str
    underlying_id: str
    horizon_days: int
    ensemble_model_hash: str | None
    ensemble_mean: float | None
    components: tuple[ComponentForecast, ...]
    #: Weights as recorded on the ensemble row. Empty when the row is
    #: absent, which is reported rather than treated as equal weighting.
    weights: Mapping[str, float]

    @property
    def complete(self) -> bool:
        """Whether every weighted model has a forecast row to evaluate.

        Incomplete attribution is usable for the models that are present
        and must not be reported as if it covered the whole ensemble, so
        the distinction is explicit rather than left to a count.
        """
        if not self.weights or not self.components:
            return False
        present = {c.model_id for c in self.components}
        return all(model_id in present for model_id in self.weights)

    @property
    def missing_models(self) -> tuple[str, ...]:
        present = {c.model_id for c in self.components}
        return tuple(sorted(m for m in self.weights if m not in present))


def resolve_attribution(
    entry: LedgerEntry, forecasts: Sequence[ForecastRecord]
) -> EnsembleAttribution:
    """Resolve one entry's ensemble composition from that run's forecast rows.

    ``forecasts`` is the run's rows; the caller fetches them once per run
    rather than once per entry, because a scan writes a handful of rows and
    a run can carry thousands of entries.

    Rows for other horizons are ignored. An entry whose run wrote no rows
    comes back with no components and no weights, which `complete` reports
    as False -- the pre-2026-10-26 entries are exactly this case and stay
    so, because nothing can reconstruct weights that were never written.
    """
    same = [
        f
        for f in forecasts
        if f.underlying_id == entry.underlying and f.horizon_days == entry.horizon_days
    ]
    ensemble_row = next((f for f in same if f.model_id == ENSEMBLE_MODEL_ID), None)
    weights: dict[str, float] = {}
    if ensemble_row is not None:
        raw = ensemble_row.component_weights
        source = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        weights = {
            str(k): float(v)
            for k, v in source.items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        }

    components = tuple(
        ComponentForecast(
            model_id=f.model_id,
            model_hash=f.model_hash,
            signal_family=f.signal_family,
            mean=f.mean,
            sigma=f.sigma,
            p_up=f.p_up,
            weight_at_prediction=weights.get(f.model_id),
            n_effective=f.n_effective,
        )
        for f in sorted(same, key=lambda f: f.model_id)
        if f.model_id != ENSEMBLE_MODEL_ID
    )
    return EnsembleAttribution(
        entry_id=entry.entry_id,
        run_id=entry.run_id,
        underlying_id=entry.underlying,
        horizon_days=entry.horizon_days,
        ensemble_model_hash=None if ensemble_row is None else ensemble_row.model_hash,
        ensemble_mean=None if ensemble_row is None else ensemble_row.mean,
        components=components,
        weights=weights,
    )


@dataclass(frozen=True, slots=True)
class ModelScore:
    """How one model's own forecasts fared against realised outcomes."""

    model_id: str
    n: int
    #: Mean squared error of the predicted mean against the realised
    #: underlying return. The underlying return, not the product's P&L:
    #: the forecast is about the market, and judging it by a leveraged
    #: product's payoff would score the product selection too.
    mse: float
    #: Share of entries whose realised direction matched ``p_up > 0.5``.
    #: ``nan`` when no entry carried a ``p_up``.
    directional_accuracy: float
    mean_weight: float | None

    @property
    def interpretable(self) -> bool:
        """Whether this score rests on enough entries to read.

        Twenty is a convention and a lenient one. Below it the mean
        squared error of a handful of forecasts says more about which days
        happened to mature than about the model.
        """
        return self.n >= 20


@dataclass(frozen=True, slots=True)
class AttributionReport:
    """What a batch of labelled entries says about each model."""

    entries_considered: int
    entries_attributable: int
    entries_incomplete: int
    model_scores: tuple[ModelScore, ...]
    ensemble_score: ModelScore | None
    missing_models: Mapping[str, int]

    @property
    def attributable_share(self) -> float:
        if self.entries_considered == 0:
            return float("nan")
        return self.entries_attributable / self.entries_considered

    def summary(self) -> str:
        if self.entries_considered == 0:
            return "no labelled entry to attribute"
        return (
            f"{self.entries_attributable} of {self.entries_considered} labelled "
            f"entr(ies) attributable ({100 * self.attributable_share:.1f} %), "
            f"{len(self.model_scores)} model(s) scored"
        )


def _score(
    model_id: str,
    predictions: Sequence[tuple[float, float | None, float | None]],
) -> ModelScore:
    """Score one model from (predicted mean, p_up, realised return) triples."""
    errors = [
        (mean - realised) ** 2 for mean, _p_up, realised in predictions if realised is not None
    ]
    hits = [
        (p_up > 0.5) == (realised > 0.0)
        for _mean, p_up, realised in predictions
        if p_up is not None and realised is not None
    ]
    # `mean_weight` is left None here and filled by the caller, which is
    # the only place that has seen the weights. Scoring and weighting are
    # kept apart on purpose: a score must not depend on how heavily the
    # model was trusted at the time.
    return ModelScore(
        model_id=model_id,
        n=len(errors),
        mse=float(np.mean(errors)) if errors else float("nan"),
        directional_accuracy=float(np.mean(hits)) if hits else float("nan"),
        mean_weight=None,
    )


def attribution_report(
    rows: Sequence[tuple[LedgerEntry, LedgerLabel, EnsembleAttribution]],
) -> AttributionReport:
    """Score every model that appears in an attributable labelled entry.

    The realised quantity is ``LedgerLabel.underlying_pnl`` -- the
    underlying's own return over the entry's window. A forecast is a claim
    about the market, so scoring it against a leveraged product's P&L
    would be scoring the product selection as well, and the two are
    separate questions (`research/lottery_pricing.py` is the second one).

    An entry with no attribution contributes to `entries_considered` and
    to nothing else, so a report cannot look complete by excluding what it
    could not resolve.
    """
    by_model: dict[str, list[tuple[float, float | None, float | None]]] = {}
    weights_seen: dict[str, list[float]] = {}
    ensemble_preds: list[tuple[float, float | None, float | None]] = []
    missing: dict[str, int] = {}
    attributable = 0
    incomplete = 0

    for entry, label, attribution in rows:
        realised = label.underlying_pnl
        if not attribution.components or not attribution.weights:
            for model_id in attribution.missing_models:
                missing[model_id] = missing.get(model_id, 0) + 1
            continue
        attributable += 1
        if not attribution.complete:
            incomplete += 1
            for model_id in attribution.missing_models:
                missing[model_id] = missing.get(model_id, 0) + 1
        for component in attribution.components:
            by_model.setdefault(component.model_id, []).append(
                (component.mean, component.p_up, realised)
            )
            if component.weight_at_prediction is not None:
                weights_seen.setdefault(component.model_id, []).append(
                    component.weight_at_prediction
                )
        if attribution.ensemble_mean is not None:
            ensemble_preds.append((attribution.ensemble_mean, entry.p_profit, realised))

    scores: list[ModelScore] = []
    for model_id, predictions in sorted(by_model.items()):
        base = _score(model_id, predictions)
        seen = weights_seen.get(model_id)
        scores.append(
            ModelScore(
                model_id=base.model_id,
                n=base.n,
                mse=base.mse,
                directional_accuracy=base.directional_accuracy,
                mean_weight=float(np.mean(seen)) if seen else None,
            )
        )

    ensemble = _score(ENSEMBLE_MODEL_ID, ensemble_preds) if ensemble_preds else None
    return AttributionReport(
        entries_considered=len(rows),
        entries_attributable=attributable,
        entries_incomplete=incomplete,
        model_scores=tuple(scores),
        ensemble_score=ensemble,
        missing_models=missing,
    )


def beats_null(score: ModelScore, *, null_mse: float) -> bool | None:
    """Whether a model's forecasts beat an unconditional null, or unknown.

    `None` when the score is not interpretable or either figure is absent.
    A `False` would claim a comparison was made; `None` says it was not.
    """
    if not score.interpretable or math.isnan(score.mse) or math.isnan(null_mse):
        return None
    return score.mse < null_mse


__all__ = [
    "ENSEMBLE_MODEL_ID",
    "AttributionReport",
    "ComponentForecast",
    "EnsembleAttribution",
    "ModelScore",
    "attribution_report",
    "beats_null",
    "resolve_attribution",
]
