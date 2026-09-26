"""Dataset assembly and out-of-sample evaluation for MAE/MFE excursion models.

Research questions RO-MAE-PREDICTION / RO-MFE-PREDICTION (catalog,
``meta/catalog.py``): can a model conditioning on **entry-time** features
forecast a turbo position's MAE (maximum adverse excursion) or MFE (maximum
favourable excursion) better than the unconditional empirical distribution
the labeler already implies? ``turboedge.models.excursion`` supplies both
models (:class:`~turboedge.models.excursion.NullExcursionModel` is the
unconditional baseline, :class:`~turboedge.models.excursion.ConditionalExcursionModel`
the challenger); this module supplies the honest comparison between them --
dataset assembly with a hard leakage guard, purged walk-forward evaluation,
and a verdict that refuses to present a comparison number as meaningful
below a measured sample-size floor (0/20 forecast-model cells and 0/80
challenger-signal cells have ever cleared the promotion ladder in this
repo -- ``docs/measured_results.md`` -- and the honest default expectation
here is the same NO until proven otherwise, out of sample).

Everything in ``forward_ledger`` is entry-time by construction; everything
in ``ledger_labels`` is exit-side, and "mae"/"mfe" are this module's only
legitimate targets from it -- see :func:`_check_no_label_leakage`, which
turns that rule into a loud runtime assertion instead of a comment next to
a hand-maintained column list.

``verdict == "CANDIDATE"`` never means promoted (CLAUDE.md rule 23,
GOVERNANCE.md §6): promotion is a separate, human-gated decision. This
module only ever measures.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from turboedge.backtest import metrics as bt_metrics
from turboedge.backtest.purged_cv import PurgedWalkForwardSplit, average_uniqueness
from turboedge.backtest.significance import bootstrap_p_value
from turboedge.models.excursion import (
    DEFAULT_QUANTILES,
    ConditionalExcursionModel,
    ExcursionPrediction,
    ExcursionTarget,
    NullExcursionModel,
)
from turboedge.pricing.intrinsic import implied_underlying
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import Direction, ExitReason, LedgerEntry, LedgerLabel

__all__ = [
    "FEATURE_NAMES",
    "MIN_DISTINCT_DATES",
    "MIN_EFFECTIVE_SAMPLE",
    "ExcursionDataset",
    "ExcursionEvalResult",
    "ExcursionFoldResult",
    "ExcursionTargetResult",
    "build_excursion_dataset",
    "evaluate_excursion",
]

# --------------------------------------------------------------------------
# Feature allow-list and the leakage guard
# --------------------------------------------------------------------------

#: Entry-time features fed to both excursion models.
#:
#: Deliberately NOT the whole entry-time column list ``MAEMFE_CONTRACT.md``'s
#: Data section names as verified/available (``predicted_return``, ``p_ko``,
#: ``p_profit``, ``uncertainty``, ``lcb_ev``, ``ratio``, ``horizon_days``,
#: ...) -- only what the contract's "What to build" section actually asks
#: for: the five ``feature_snapshot`` keys, a derived barrier-distance
#: feature, and direction. Keeping this list short also keeps
#: ``ConditionalExcursionModel``'s own per-tau sample floor
#: (``(n_features + 1) * 10``) low, which matters a great deal on a
#: research question the codebase's own track record says will usually
#: fail the sample-size gate before it ever reaches a model comparison.
#:
#: ``regime_bucket`` is excluded on purpose, not by oversight: it is
#: frequently NULL (100% NULL in the local 2,672-row sample; not verified
#: populated in production either) and a free-form string of unbounded
#: cardinality with no declared "unknown" sentinel. Rule 29 forbids
#: silently imputing a pricing-relevant value for a missing field --
#: one-hot encoding it would require inventing exactly such a value for
#: every NULL row, and dropping every NULL row would (being "frequently"
#: null) throw away most of the dataset for a field that is not itself
#: proven predictive of anything. Leaving it out of the numeric feature
#: matrix entirely is the one option that neither imputes nor mass-drops.
FEATURE_NAMES: tuple[str, ...] = (
    "cost_rank_score",
    "leverage",
    "premium_over_fair",
    "signal_score",
    "spread_pct",
    "barrier_distance",
    "direction_sign",
)


def _check_no_label_leakage(feature_names: Sequence[str]) -> None:
    """Raise if any name in ``feature_names`` is also a ``LedgerLabel`` field.

    ``LedgerLabel`` (``storage/schemas.py``) is the EXIT-side table --
    ``mae``/``mfe`` are targets, and "nothing from this table may ever
    become a feature" (``MAEMFE_CONTRACT.md``'s Data section). Checking the
    actual pydantic field set here -- rather than trusting a hand-written
    comment next to ``FEATURE_NAMES`` -- means an accidental future edit
    that adds e.g. ``"realized_selected_pnl"`` to the allow-list fails
    loudly (an ``AssertionError``, at import time via the module-level call
    below, or explicitly in a test) instead of silently leaking exit-side
    information into ``x``.
    """
    label_columns = set(LedgerLabel.model_fields.keys())
    leaked = set(feature_names) & label_columns
    if leaked:
        raise AssertionError(
            f"feature_names contains ledger_labels (exit-side) columns: {sorted(leaked)!r} -- "
            "these can never be entry-time features (MAEMFE_CONTRACT.md, rule: nothing from "
            "ledger_labels except mae/mfe as targets may ever reach x)"
        )


_check_no_label_leakage(FEATURE_NAMES)

#: Reasons a candidate row can be dropped before it reaches the dataset,
#: fixed so ``dropped_rows`` always reports every category (even at 0)
#: rather than only the ones that happened to fire -- silent row loss is
#: exactly how sample-size claims become fiction (MAEMFE_CONTRACT.md).
#: Checked in this priority order (first match wins) so a row that could
#: match more than one reason (e.g. ``EXPIRED_NO_DATA`` always also has
#: ``ambiguous_path=True`` and null mae/mfe -- verified against
#: ``learning/labeler.py::resolve_exit``) is counted exactly once, under
#: its most specific reason:
#:
#: 1. ``no_label``: entry not labeled yet (open position).
#: 2. ``exit_reason_expired_no_data``: no exit information at all.
#: 3. ``ambiguous_path``: an ambiguous bar was resolved conservatively
#:    (Master Spec rule 17 -- never resolved optimistically, and never
#:    used as if it were an unambiguous observation either).
#: 4. ``null_target``: mae/mfe/realized_selected_pnl missing despite none
#:    of the above (defensive; not expected to fire given 1-3, since KO/
#:    HORIZON resolutions -- the only ``ambiguous_path=False`` reasons --
#:    always set all three per ``resolve_exit``).
#: 5. ``null_feature``: an entry-time feature could not be computed
#:    (missing ``feature_snapshot`` key, or barrier-distance undefined
#:    because ``financing_level_entry``/``barrier_entry`` is NULL).
_DROP_REASONS: tuple[str, ...] = (
    "no_label",
    "exit_reason_expired_no_data",
    "ambiguous_path",
    "null_target",
    "null_feature",
)


def _barrier_distance(entry: LedgerEntry) -> float:
    """Relative distance from the implied entry-time underlying level to ``barrier_entry``.

    ``forward_ledger`` never stores the underlying's own spot level at
    entry -- only the certificate's own entry price (``entry_ask``, per
    rule 11: entry is always ask) and structural terms
    (``financing_level_entry``, ``ratio``, ``fx``, ``direction``). Those
    are exactly the inputs ``turboedge.pricing.intrinsic.implied_underlying``
    already uses to invert the turbo/knock-out linear pricing identity
    (Master Spec §13.1) back to an implied spot::

        implied_spot = F + entry_ask * fx / ratio   (LONG)
        implied_spot = F - entry_ask * fx / ratio   (SHORT)

    This feature is that implied spot's distance to ``barrier_entry``,
    using the same moneyness sign convention as
    ``turboedge.pricing.intrinsic.intrinsic_value`` (positive means further
    from breaching the barrier, for either direction) and normalized by
    the implied spot so the value is comparable in scale across products
    and underlyings::

        barrier_distance = (implied_spot - barrier_entry) / implied_spot   (LONG)
        barrier_distance = (barrier_entry - implied_spot) / implied_spot  (SHORT)

    Returns ``nan`` (never an imputed guess -- rule 29) when
    ``financing_level_entry`` or ``barrier_entry`` is NULL, or when the
    implied spot is non-positive (degenerate input); the caller drops any
    row whose feature vector contains a ``nan``.
    """
    if entry.financing_level_entry is None or entry.barrier_entry is None:
        return float("nan")
    implied_spot = implied_underlying(
        price=entry.entry_ask,
        financing_level=entry.financing_level_entry,
        ratio=entry.ratio,
        direction=entry.direction,
        fx=entry.fx,
    )
    if not (implied_spot > 0.0):
        return float("nan")
    moneyness = (
        implied_spot - entry.barrier_entry
        if entry.direction is Direction.LONG
        else entry.barrier_entry - implied_spot
    )
    return float(moneyness / implied_spot)


def _row_features(entry: LedgerEntry) -> npt.NDArray[np.float64]:
    """One row of ``x``, in :data:`FEATURE_NAMES` order; may contain ``nan``."""
    snapshot = entry.feature_snapshot
    values = {
        "cost_rank_score": snapshot.get("cost_rank_score", float("nan")),
        "leverage": snapshot.get("leverage", float("nan")),
        "premium_over_fair": snapshot.get("premium_over_fair", float("nan")),
        "signal_score": snapshot.get("signal_score", float("nan")),
        "spread_pct": snapshot.get("spread_pct", float("nan")),
        "barrier_distance": _barrier_distance(entry),
        "direction_sign": 1.0 if entry.direction is Direction.LONG else -1.0,
    }
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64)


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------


class ExcursionDataset(BaseModel):
    """Assembled, leak-checked, purge-ready MAE/MFE training/evaluation set.

    One row per surviving ``forward_ledger`` x ``ledger_labels`` join (see
    :func:`build_excursion_dataset` for the join and drop logic), sorted by
    ``t0`` ascending -- required by
    :meth:`~turboedge.backtest.purged_cv.PurgedWalkForwardSplit.split`.

    ``underlying``/``as_of`` are carried here (beyond the fields a plain
    join would need) purely so :func:`evaluate_excursion` -- which only
    receives this dataset and a splitter -- can still stamp them onto its
    ``ExcursionEvalResult`` without a second parameter threading them
    through separately.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    x: npt.NDArray[np.float64]
    y_mae: npt.NDArray[np.float64]
    y_mfe: npt.NDArray[np.float64]
    t0: npt.NDArray[np.int64]
    t1: npt.NDArray[np.int64]
    weights: npt.NDArray[np.float64]
    is_shadow: npt.NDArray[np.bool_]
    realized_pnl: npt.NDArray[np.float64]
    feature_names: tuple[str, ...]
    n_distinct_dates: int
    effective_sample: float
    dropped_rows: dict[str, int]
    underlying: str | None
    as_of: datetime


def build_excursion_dataset(
    store: Store, *, underlying: str | None = None, as_of: datetime
) -> ExcursionDataset:
    """Join ``forward_ledger``/``ledger_labels`` on ``entry_id`` into an :class:`ExcursionDataset`.

    See the module docstring for the leakage guard, and :data:`FEATURE_NAMES`
    for which entry-time columns become features.

    No look-ahead (rule 4/5): entries with ``prediction_time > as_of`` are
    excluded outright -- not counted in ``dropped_rows`` (they are simply
    not part of the ``as_of`` snapshot, the same convention
    ``bars_as_of``/every ``ForecastModel.fit(..., as_of)`` in this codebase
    already uses), whereas every row in ``dropped_rows`` below IS as-of
    eligible but structurally unusable. See :data:`_DROP_REASONS` for the
    five counted-and-dropped categories and their priority order.

    ``t0``/``t1`` are day indices relative to the earliest surviving
    ``prediction_time`` (``t0`` from ``prediction_time.date()``, ``t1``
    from ``exit_due``, both integer day offsets) -- exactly the units
    :func:`~turboedge.backtest.purged_cv.average_uniqueness` and
    :class:`~turboedge.backtest.purged_cv.PurgedWalkForwardSplit` expect.
    ``t1`` is floored at ``t0`` as a defensive guard (``average_uniqueness``
    requires ``t1 >= t0`` elementwise); this should never bind for real
    data since ``exit_due`` is always strictly after ``prediction_time``.
    """
    pairs = store.list_ledger_entries(underlying=underlying)
    dropped = {reason: 0 for reason in _DROP_REASONS}
    kept: list[tuple[LedgerEntry, LedgerLabel, npt.NDArray[np.float64]]] = []

    for entry, label in pairs:
        if entry.prediction_time > as_of:
            continue
        if label is None:
            dropped["no_label"] += 1
            continue
        if label.exit_reason is ExitReason.EXPIRED_NO_DATA:
            dropped["exit_reason_expired_no_data"] += 1
            continue
        if label.ambiguous_path:
            dropped["ambiguous_path"] += 1
            continue
        if label.mae is None or label.mfe is None or label.realized_selected_pnl is None:
            dropped["null_target"] += 1
            continue
        features = _row_features(entry)
        if not np.all(np.isfinite(features)):
            dropped["null_feature"] += 1
            continue
        kept.append((entry, label, features))

    if not kept:
        raise ValueError(
            f"build_excursion_dataset(underlying={underlying!r}, as_of={as_of!r}) produced zero "
            f"surviving rows out of {len(pairs)} candidate entries (dropped={dropped!r})"
        )

    kept.sort(key=lambda row: row[0].prediction_time)
    min_date = kept[0][0].prediction_time.date()

    x = np.stack([row[2] for row in kept]).astype(np.float64)
    y_mae = np.array([row[1].mae for row in kept], dtype=np.float64)
    y_mfe = np.array([row[1].mfe for row in kept], dtype=np.float64)
    t0 = np.array([(row[0].prediction_time.date() - min_date).days for row in kept], dtype=np.int64)
    t1_raw = np.array([(row[0].exit_due - min_date).days for row in kept], dtype=np.int64)
    t1 = np.maximum(t1_raw, t0)
    weights = average_uniqueness(t0, t1)
    is_shadow = np.array([row[0].is_shadow for row in kept], dtype=np.bool_)
    realized_pnl = np.array([row[1].realized_selected_pnl for row in kept], dtype=np.float64)

    return ExcursionDataset(
        x=x,
        y_mae=y_mae,
        y_mfe=y_mfe,
        t0=t0,
        t1=t1,
        weights=weights,
        is_shadow=is_shadow,
        realized_pnl=realized_pnl,
        feature_names=FEATURE_NAMES,
        n_distinct_dates=int(np.unique(t0).size),
        effective_sample=float(np.sum(weights)),
        dropped_rows=dropped,
        underlying=underlying,
        as_of=as_of,
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

#: GOVERNANCE.md Sec 2's own promotion floor -- reused here as the gate
#: below which no comparison number may be presented as a result at all,
#: not only as the bar for a later promotion decision.
MIN_EFFECTIVE_SAMPLE = 100.0

#: Minimum distinct entry-time calendar dates a dataset must span before a
#: purged walk-forward split over it can mean anything, independent of raw
#: effective sample size.
#:
#: ``embargo=horizon`` is the convention this codebase already uses for
#: every other purged-CV caller (``models/challengers.py``,
#: ``models/directional.py``: ``PurgedWalkForwardSplit(horizon=h,
#: embargo=h, ...)``), and horizons up to 14d are in active use
#: (``turboedge scan --horizon ... 14d``). A single purge+embargo buffer at
#: that horizon can therefore consume up to 28 calendar days on its own; a
#: dataset spanning fewer distinct dates than that cannot even remove one
#: such buffer's worth of history from the training set without exhausting
#: the whole sample, so no walk-forward split over it can claim to test
#: genuine out-of-sample generalization, regardless of row count. 30 is a
#: deliberately modest floor just above that single-buffer minimum -- not a
#: target sample size, only the point below which the split's own
#: chronological structure is not yet meaningful. (This is exactly the
#: failure mode a same-day-only dataset -- the local database's actual
#: state, one distinct prediction date -- hits hardest:
#: :func:`~turboedge.backtest.purged_cv.average_uniqueness` gives every
#: row weight ``1/n`` when every label window fully overlaps, so
#: ``effective_sample`` alone already collapses to ~1.0 in that case; this
#: floor is a second, independent line of defense for datasets that spread
#: over a handful of dates without spanning enough of them.)
MIN_DISTINCT_DATES = 30

#: Two-sided bootstrap-hypothesis-test significance bar for the paired-fold
#: CRPS-improvement check, matching GOVERNANCE.md §3.1's own FDR alpha
#: (0.10) rather than inventing a separate threshold for this one overlay.
_SIGNIFICANCE_ALPHA = 0.10

#: Folds below this count cannot support a meaningful paired-fold bootstrap
#: (with 1-2 points, a percentile bootstrap is either degenerate or trivially
#: "significant" from resampling noise, not real evidence) -- matches
#: ``backtest/significance.py``'s own ``n >= 3`` floor for PSR/DSR.
_MIN_SIGNIFICANCE_FOLDS = 3

_BOOTSTRAP_RESAMPLES = 2000
#: Fixed so a re-run reproduces byte-for-byte (Master Spec rule 33).
_BOOTSTRAP_SEED = 20260926

#: Minimum *relative* CRPS/mean-pinball reduction the conditional model must
#: show before it counts as beating the null at all -- a directional "win"
#: alone is not enough (GOVERNANCE.md §2.2 requires a minimum effect size on
#: top of significance for exactly this reason: "small improvements... remain
#: experimental"). Without this floor, two models scoring within noise of each
#: other on held-out data (a sub-0.1% CRPS difference, measured on a
#: pure-noise synthetic dataset in this module's own test suite --
#: ``test_pure_noise_dataset_does_not_yield_candidate``) can still land on the
#: "improving" side of zero by chance, and with only a handful of
#: walk-forward folds a paired bootstrap can occasionally call that
#: direction "significant" too (exactly the false-positive rate
#: ``_SIGNIFICANCE_ALPHA`` implies it will, some fraction of the time). 5% is
#: a conservative floor -- far above plausible noise-level differences, far
#: below what a real, useful predictive relationship produces (an 80%+
#: relative CRPS reduction in this module's own signal-dependent test).
_MIN_RELATIVE_IMPROVEMENT = 0.05

#: Rank quantile used by the selective-abstention overlay (see
#: :func:`_selective_abstention`): the model's predicted 5th-percentile MAE,
#: i.e. its belief about the tail-downside outcome.
_ABSTENTION_TAU = 0.05
#: Entries in the "less bad predicted downside" half are kept, the other
#: half rejected -- a fixed, parameter-free 50/50 split, not tuned against
#: realized P&L (tuning it against the very outcome it is being scored
#: against would be exactly the parameter fishing GOVERNANCE.md forbids).
_ABSTENTION_KEEP_FRACTION = 0.5

_VERDICT_RANK: dict[str, int] = {
    "NO_IMPROVEMENT": 0,
    "IMPROVEMENT_NOT_SIGNIFICANT": 1,
    "CANDIDATE": 2,
}


class ExcursionFoldResult(BaseModel):
    """One purged walk-forward fold's null-vs-conditional comparison, for one target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fold_index: int
    n_train: int
    n_test: int
    pinball_null: dict[float, float]
    pinball_conditional: dict[float, float]
    crps_null: float
    crps_conditional: float
    coverage_null: float
    coverage_conditional: float
    quantile_crossing_rate: float


class ExcursionTargetResult(BaseModel):
    """Aggregated (across folds, weighted by test-fold size) result for one target (MAE or MFE)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: ExcursionTarget
    pinball_by_tau_null: dict[float, float]
    pinball_by_tau_conditional: dict[float, float]
    crps_null: float
    crps_conditional: float
    coverage_null: float
    coverage_conditional: float
    quantile_crossing_rate: float
    folds: list[ExcursionFoldResult]
    n_test_samples: int
    effective_sample: float
    #: ``max(quantile_levels) - min(quantile_levels)``; 0.90 for
    #: :data:`~turboedge.models.excursion.DEFAULT_QUANTILES` (q05/q95).
    nominal_coverage: float
    mean_pinball_null: float
    mean_pinball_conditional: float
    #: Two-sided bootstrap p-value (H0: mean per-fold CRPS difference is 0)
    #: over the folds that actually produced a paired comparison; ``nan``
    #: if fewer than :data:`_MIN_SIGNIFICANCE_FOLDS` did.
    significance_p_value: float
    n_significance_folds: int


class ExcursionEvalResult(BaseModel):
    """Top-level result of :func:`evaluate_excursion`.

    ``verdict`` is one of ``"INSUFFICIENT_SAMPLE"``, ``"NO_IMPROVEMENT"``,
    ``"IMPROVEMENT_NOT_SIGNIFICANT"``, ``"CANDIDATE"`` -- see
    :func:`evaluate_excursion`'s docstring for exactly what each requires.
    Since MAE and MFE are two separate research questions
    (RO-MAE-PREDICTION / RO-MFE-PREDICTION) evaluated in one call, the
    top-level verdict is the strongest of the two per-target verdicts
    (``verdict_reasons`` names which target achieved what, with the actual
    numbers, so this collapsing is never a hidden step).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    underlying: str | None
    as_of: datetime
    n_distinct_dates: int
    effective_sample: float
    dropped_rows: dict[str, int]
    results: list[ExcursionTargetResult]
    selective_abstention: dict[str, float]
    verdict: str
    verdict_reasons: list[str]


def _fold_predictions(
    model: NullExcursionModel | ConditionalExcursionModel, x_test: npt.NDArray[np.float64]
) -> list[ExcursionPrediction]:
    return model.predict(x_test)


def _evaluate_target(
    target: ExcursionTarget,
    y: npt.NDArray[np.float64],
    dataset: ExcursionDataset,
    folds_idx: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
    quantile_levels: tuple[float, ...],
) -> ExcursionTargetResult:
    """Run every fold for one target, fitting both models on the *same* train/test split.

    Both models get ``dataset.weights[train_idx]`` as ``sample_weight`` --
    the null must get them too, or the comparison is rigged in the
    conditional model's favour by construction, not by evidence.
    """
    fold_results: list[ExcursionFoldResult] = []
    fold_test_idx: list[npt.NDArray[np.int64]] = []
    crps_diffs: list[float] = []

    for i, (train_idx, test_idx) in enumerate(folds_idx):
        x_train, x_test = dataset.x[train_idx], dataset.x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = dataset.weights[train_idx]

        null_model = NullExcursionModel(target, quantile_levels=quantile_levels)
        cond_model = ConditionalExcursionModel(target, quantile_levels=quantile_levels)
        try:
            null_model.fit(x_train, y_train, sample_weight=w_train)
            cond_model.fit(x_train, y_train, sample_weight=w_train)
        except ValueError:
            # Fold too small for a stable fit of at least one model (e.g.
            # an early expanding-window fold, or a fold whose y happens to
            # have zero variance). Both models must succeed together or the
            # fold is dropped entirely -- a null-only number for a fold the
            # conditional model could not even fit would misrepresent the
            # comparison as complete when it is not.
            continue

        null_preds = _fold_predictions(null_model, x_test)
        cond_preds = _fold_predictions(cond_model, x_test)
        null_q = [p.quantiles for p in null_preds]
        cond_q = [p.quantiles for p in cond_preds]

        crps_null = bt_metrics.mean_crps_from_quantiles(y_test, null_q)
        crps_cond = bt_metrics.mean_crps_from_quantiles(y_test, cond_q)
        pinball_null = bt_metrics.pinball_loss_by_level(y_test, null_q)
        pinball_cond = bt_metrics.pinball_loss_by_level(y_test, cond_q)

        lo_tau, hi_tau = min(quantile_levels), max(quantile_levels)
        coverage_null = bt_metrics.interval_coverage(
            y_test,
            np.array([p.quantiles[lo_tau] for p in null_preds]),
            np.array([p.quantiles[hi_tau] for p in null_preds]),
        )
        coverage_cond = bt_metrics.interval_coverage(
            y_test,
            np.array([p.quantiles[lo_tau] for p in cond_preds]),
            np.array([p.quantiles[hi_tau] for p in cond_preds]),
        )

        fold_results.append(
            ExcursionFoldResult(
                fold_index=i,
                n_train=int(train_idx.size),
                n_test=int(test_idx.size),
                pinball_null=pinball_null,
                pinball_conditional=pinball_cond,
                crps_null=crps_null,
                crps_conditional=crps_cond,
                coverage_null=coverage_null,
                coverage_conditional=coverage_cond,
                quantile_crossing_rate=cond_model.crossing_rate,
            )
        )
        fold_test_idx.append(test_idx)
        crps_diffs.append(crps_cond - crps_null)

    if not fold_results:
        raise ValueError(
            f"evaluate_excursion: no usable folds for target={target.value!r} -- every fold was "
            "too small for a stable model fit despite the dataset passing MIN_EFFECTIVE_SAMPLE/"
            "MIN_DISTINCT_DATES; reduce the splitter's n_splits, or increase min_train/step"
        )

    total_test = sum(f.n_test for f in fold_results)

    def _agg(values: list[float]) -> float:
        return sum(v * f.n_test for v, f in zip(values, fold_results, strict=True)) / total_test

    def _agg_tau_dict(per_fold: list[dict[float, float]]) -> dict[float, float]:
        out: dict[float, float] = {}
        for tau in quantile_levels:
            num = sum(
                d[tau] * f.n_test for d, f in zip(per_fold, fold_results, strict=True) if tau in d
            )
            den = sum(f.n_test for d, f in zip(per_fold, fold_results, strict=True) if tau in d)
            if den:
                out[tau] = num / den
        return out

    pinball_by_tau_null = _agg_tau_dict([f.pinball_null for f in fold_results])
    pinball_by_tau_conditional = _agg_tau_dict([f.pinball_conditional for f in fold_results])
    mean_pinball_null = float(np.mean(list(pinball_by_tau_null.values())))
    mean_pinball_conditional = float(np.mean(list(pinball_by_tau_conditional.values())))

    if len(crps_diffs) >= _MIN_SIGNIFICANCE_FOLDS:
        rng = np.random.default_rng(_BOOTSTRAP_SEED)
        p_value = bootstrap_p_value(
            np.array(crps_diffs, dtype=np.float64),
            stat=lambda v: float(np.mean(v)),
            n=_BOOTSTRAP_RESAMPLES,
            rng=rng,
            null_value=0.0,
        )
    else:
        p_value = float("nan")

    effective_test_sample = float(sum(float(np.sum(dataset.weights[idx])) for idx in fold_test_idx))

    return ExcursionTargetResult(
        target=target,
        pinball_by_tau_null=pinball_by_tau_null,
        pinball_by_tau_conditional=pinball_by_tau_conditional,
        crps_null=_agg([f.crps_null for f in fold_results]),
        crps_conditional=_agg([f.crps_conditional for f in fold_results]),
        coverage_null=_agg([f.coverage_null for f in fold_results]),
        coverage_conditional=_agg([f.coverage_conditional for f in fold_results]),
        quantile_crossing_rate=_agg([f.quantile_crossing_rate for f in fold_results]),
        folds=fold_results,
        n_test_samples=total_test,
        effective_sample=effective_test_sample,
        nominal_coverage=max(quantile_levels) - min(quantile_levels),
        mean_pinball_null=mean_pinball_null,
        mean_pinball_conditional=mean_pinball_conditional,
        significance_p_value=p_value,
        n_significance_folds=len(crps_diffs),
    )


def _selective_abstention(
    dataset: ExcursionDataset,
    folds_idx: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
    quantile_levels: tuple[float, ...],
) -> dict[str, float]:
    """The one economically meaningful use of a predicted MAE computable from this data.

    Ranks pooled out-of-sample test entries by their predicted
    :data:`_ABSTENTION_TAU` (q05) MAE quantile and compares the mean
    ``realized_selected_pnl`` of the entries a
    :data:`_ABSTENTION_KEEP_FRACTION`-based filter would KEEP (the half
    with the less severe predicted downside) against the mean over all
    test entries -- plus the kept fraction and shadow/non-shadow counts,
    so the prefiltering selection effect (``is_shadow=False`` entries are
    already survivors of the pipeline's own gates; ``is_shadow=True`` ones
    are the unbiased sample) is visible rather than hidden inside a single
    pooled number.

    What is explicitly NOT computable here, and not faked: a stop-loss or
    take-profit overlay. ``mae``/``mfe`` are each the extremum of the whole
    exit-window path (``learning/labeler.py::resolve_exit``:
    ``mfe = max(product_returns)``, ``mae = min(product_returns)``) with no
    record of which extremum occurred *first* in time. A stop-loss rule's
    actual exit price depends on that ordering (if MAE happens first, the
    position is stopped out at the MAE price and never sees the later MFE;
    if MFE happens first, the reverse) -- and that ordering is simply not
    in this data. Any number claiming to show what a stop/take-profit rule
    "would have returned" from mae/mfe alone would be silently assuming an
    ordering this dataset cannot support.
    """
    if _ABSTENTION_TAU not in quantile_levels:
        return {}

    q05_pred: list[float] = []
    realized: list[float] = []
    shadow_flags: list[bool] = []
    for train_idx, test_idx in folds_idx:
        x_train, y_train = dataset.x[train_idx], dataset.y_mae[train_idx]
        w_train = dataset.weights[train_idx]
        model = ConditionalExcursionModel(ExcursionTarget.MAE, quantile_levels=quantile_levels)
        try:
            model.fit(x_train, y_train, sample_weight=w_train)
        except ValueError:
            continue
        preds = model.predict(dataset.x[test_idx])
        q05_pred.extend(p.quantiles[_ABSTENTION_TAU] for p in preds)
        realized.extend(dataset.realized_pnl[test_idx].tolist())
        shadow_flags.extend(bool(v) for v in dataset.is_shadow[test_idx].tolist())

    if not q05_pred:
        return {}

    q05_arr = np.array(q05_pred, dtype=np.float64)
    realized_arr = np.array(realized, dtype=np.float64)
    shadow_arr = np.array(shadow_flags, dtype=np.bool_)

    # A higher predicted q05(MAE) means a less severe predicted downside
    # (mae is typically <= 0; closer to 0 is "less bad"). Keep the upper
    # half by that ranking, i.e. the entries the model believes are safer.
    threshold = float(np.median(q05_arr))
    keep_mask = q05_arr >= threshold

    n_all = int(q05_arr.size)
    n_kept = int(np.sum(keep_mask))
    return {
        "kept_fraction": n_kept / n_all,
        "mean_pnl_kept": float(np.mean(realized_arr[keep_mask])) if n_kept else float("nan"),
        "mean_pnl_all": float(np.mean(realized_arr)),
        "n_kept": float(n_kept),
        "n_all": float(n_all),
        "n_shadow_all": float(np.sum(shadow_arr)),
        "n_shadow_kept": float(np.sum(shadow_arr & keep_mask)),
        "n_nonshadow_all": float(np.sum(~shadow_arr)),
        "n_nonshadow_kept": float(np.sum(~shadow_arr & keep_mask)),
    }


def _target_verdict(result: ExcursionTargetResult) -> tuple[str, list[str]]:
    """Per-target verdict. See :func:`evaluate_excursion`'s docstring for the exact ladder.

    Three tiers, checked in order:

    1. ``NO_IMPROVEMENT`` -- the conditional model does not beat the null by
       at least :data:`_MIN_RELATIVE_IMPROVEMENT` on *both* CRPS and mean
       pinball. A directional "win" alone is not enough (see that constant's
       docstring for why a relative-margin floor is needed even before
       significance is checked).
    2. ``IMPROVEMENT_NOT_SIGNIFICANT`` -- clears that margin on both, but
       either interval coverage is worse than the null's, or the paired-fold
       bootstrap does not reject "no true CRPS difference" at
       ``_SIGNIFICANCE_ALPHA``.
    3. ``CANDIDATE`` -- clears the margin on both, coverage is no worse, and
       the significance check passes. Never means promoted.
    """
    reasons: list[str] = []
    crps_improves = result.crps_conditional <= result.crps_null * (1.0 - _MIN_RELATIVE_IMPROVEMENT)
    pinball_improves = result.mean_pinball_conditional <= result.mean_pinball_null * (
        1.0 - _MIN_RELATIVE_IMPROVEMENT
    )
    coverage_no_worse = abs(result.coverage_conditional - result.nominal_coverage) <= abs(
        result.coverage_null - result.nominal_coverage
    )
    reasons.append(
        f"{result.target.value}: crps conditional={result.crps_conditional:.6f} "
        f"vs null={result.crps_null:.6f} (needs <= {1.0 - _MIN_RELATIVE_IMPROVEMENT:.0%} of null "
        f"to count as improving)"
    )
    reasons.append(
        f"{result.target.value}: mean pinball conditional={result.mean_pinball_conditional:.6f} "
        f"vs null={result.mean_pinball_null:.6f}"
    )
    if not (crps_improves and pinball_improves):
        reasons.append(
            f"{result.target.value}: fails the minimum {_MIN_RELATIVE_IMPROVEMENT:.0%} "
            "relative CRPS+pinball improvement over the null"
        )
        return "NO_IMPROVEMENT", reasons

    reasons.append(
        f"{result.target.value}: coverage conditional={result.coverage_conditional:.4f} "
        f"vs null={result.coverage_null:.4f} (nominal={result.nominal_coverage:.2f})"
    )
    reasons.append(
        f"{result.target.value}: significance p={result.significance_p_value:.4f} over "
        f"{result.n_significance_folds} folds (alpha={_SIGNIFICANCE_ALPHA})"
    )
    if (
        coverage_no_worse
        and result.n_significance_folds >= _MIN_SIGNIFICANCE_FOLDS
        and result.significance_p_value < _SIGNIFICANCE_ALPHA
    ):
        return "CANDIDATE", reasons
    reasons.append(
        f"{result.target.value}: improvement present but not a CANDIDATE -- needs coverage no "
        f"worse than the null AND significance at alpha={_SIGNIFICANCE_ALPHA} over "
        f">= {_MIN_SIGNIFICANCE_FOLDS} folds"
    )
    return "IMPROVEMENT_NOT_SIGNIFICANT", reasons


def _decide_verdict(results: list[ExcursionTargetResult]) -> tuple[str, list[str]]:
    best = "NO_IMPROVEMENT"
    all_reasons: list[str] = []
    for result in results:
        verdict, reasons = _target_verdict(result)
        all_reasons.extend(reasons)
        if _VERDICT_RANK[verdict] > _VERDICT_RANK[best]:
            best = verdict
    return best, all_reasons


def evaluate_excursion(
    dataset: ExcursionDataset,
    *,
    splitter: PurgedWalkForwardSplit,
    quantile_levels: Sequence[float] = DEFAULT_QUANTILES,
) -> ExcursionEvalResult:
    """Purged walk-forward null-vs-conditional evaluation of MAE and MFE prediction.

    If ``dataset.effective_sample < MIN_EFFECTIVE_SAMPLE`` or
    ``dataset.n_distinct_dates < MIN_DISTINCT_DATES``, returns
    ``verdict="INSUFFICIENT_SAMPLE"`` immediately with ``results=[]`` --
    *no* comparison number is computed or presented in that case. This is
    the expected outcome on the local database (one distinct prediction
    date), and it must come out cleanly rather than as a crash or a
    misleading number.

    Otherwise, per target (MAE, MFE independently) and per fold: fits
    both models on the training indices (both get
    ``dataset.weights[train_idx]`` as ``sample_weight``), predicts on the
    test indices, and accumulates CRPS/pinball/coverage/quantile-crossing.
    Per-fold metrics are aggregated weighted by test-fold size.

    A target's verdict is ``"CANDIDATE"`` only if the conditional model
    beats the null by at least :data:`_MIN_RELATIVE_IMPROVEMENT` on CRPS
    *and* on mean pinball across ``quantile_levels`` (a directional win
    alone is not enough -- see that constant's docstring), *and* is no
    worse on ``[q_min, q_max]`` interval coverage, *and* a paired-fold
    bootstrap test (:data:`_MIN_SIGNIFICANCE_FOLDS` folds minimum) rejects
    "no true CRPS difference" at ``alpha=_SIGNIFICANCE_ALPHA``. Clearing the
    CRPS/pinball margin without the coverage/significance requirements gives
    ``"IMPROVEMENT_NOT_SIGNIFICANT"`` instead of ``"NO_IMPROVEMENT"``.
    The top-level ``verdict`` is the strongest of the two targets' verdicts
    (RO-MAE-PREDICTION and RO-MFE-PREDICTION are separate research
    questions bundled into one evaluation run); ``verdict_reasons`` names
    which target produced which numbers either way.

    ``verdict == "CANDIDATE"`` never means promoted -- see this module's
    docstring and GOVERNANCE.md §6 for the (separate, human-gated)
    promotion process.
    """
    levels = tuple(float(q) for q in quantile_levels)
    reasons: list[str] = []
    insufficient = False
    if dataset.effective_sample < MIN_EFFECTIVE_SAMPLE:
        reasons.append(
            f"effective_sample={dataset.effective_sample:.2f} < "
            f"MIN_EFFECTIVE_SAMPLE={MIN_EFFECTIVE_SAMPLE:.2f} (GOVERNANCE.md Sec 2 promotion floor)"
        )
        insufficient = True
    if dataset.n_distinct_dates < MIN_DISTINCT_DATES:
        reasons.append(
            f"n_distinct_dates={dataset.n_distinct_dates} < MIN_DISTINCT_DATES={MIN_DISTINCT_DATES}"
        )
        insufficient = True
    if insufficient:
        return ExcursionEvalResult(
            underlying=dataset.underlying,
            as_of=dataset.as_of,
            n_distinct_dates=dataset.n_distinct_dates,
            effective_sample=dataset.effective_sample,
            dropped_rows=dataset.dropped_rows,
            results=[],
            selective_abstention={},
            verdict="INSUFFICIENT_SAMPLE",
            verdict_reasons=reasons,
        )

    folds_idx = list(splitter.split(dataset.t0, dataset.t1))
    if not folds_idx:
        raise ValueError(
            "evaluate_excursion: splitter produced zero folds even though the dataset passed "
            f"the sample-size gates (n_distinct_dates={dataset.n_distinct_dates}, "
            f"effective_sample={dataset.effective_sample:.2f}) -- check the splitter's "
            "min_train/step/n_splits/embargo against this dataset's date span"
        )

    results = [
        _evaluate_target(
            target,
            dataset.y_mae if target is ExcursionTarget.MAE else dataset.y_mfe,
            dataset,
            folds_idx,
            levels,
        )
        for target in (ExcursionTarget.MAE, ExcursionTarget.MFE)
    ]
    abstention = _selective_abstention(dataset, folds_idx, levels)
    verdict, verdict_reasons = _decide_verdict(results)
    return ExcursionEvalResult(
        underlying=dataset.underlying,
        as_of=dataset.as_of,
        n_distinct_dates=dataset.n_distinct_dates,
        effective_sample=dataset.effective_sample,
        dropped_rows=dataset.dropped_rows,
        results=results,
        selective_abstention=abstention,
        verdict=verdict,
        verdict_reasons=verdict_reasons,
    )
