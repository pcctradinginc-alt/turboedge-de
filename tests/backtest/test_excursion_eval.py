"""Tests for :mod:`turboedge.backtest.excursion_eval`.

Uses a real ``Store(":memory:")`` with ``init_schema()`` and synthetic
``LedgerEntry``/``LedgerLabel`` rows built directly (not via
``learning/labeler.py``, so the priority-ordered drop-reason logic can be
exercised for combinations the real labeler would never itself produce --
e.g. a ``null_target`` row with ``ambiguous_path=False``, which is the
defensive branch documented in ``excursion_eval._DROP_REASONS``).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import pytest

from turboedge.backtest import excursion_eval
from turboedge.backtest.excursion_eval import (
    FEATURE_NAMES,
    MIN_DISTINCT_DATES,
    MIN_EFFECTIVE_SAMPLE,
    ExcursionDataset,
    build_excursion_dataset,
    evaluate_excursion,
)
from turboedge.backtest.purged_cv import PurgedWalkForwardSplit, average_uniqueness
from turboedge.models.excursion import (
    DEFAULT_QUANTILES,
    ConditionalExcursionModel,
    ExcursionPrediction,
    ExcursionTarget,
    NullExcursionModel,
)
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Category,
    Direction,
    ExitReason,
    LedgerEntry,
    LedgerEntryStatus,
    LedgerLabel,
)

_FAR_FUTURE = datetime(2030, 1, 1, tzinfo=UTC)


@pytest.fixture
def store() -> Iterator[Store]:
    with Store(":memory:") as s:
        s.init_schema()
        yield s


@dataclass
class _RowSpec:
    """One synthetic ``forward_ledger``/``ledger_labels`` row, with sane defaults."""

    day_offset: int
    direction: Direction = Direction.LONG
    horizon_days: int = 3
    entry_ask: float = 5.0
    financing_level_entry: float | None = 100.0
    barrier_entry: float | None = 100.0
    ratio: float = 1.0
    fx: float = 1.0
    is_shadow: bool = False
    feature_snapshot: dict[str, float] = field(
        default_factory=lambda: {
            "cost_rank_score": 0.1,
            "leverage": 2.0,
            "premium_over_fair": 0.0,
            "signal_score": 0.0,
            "spread_pct": 0.001,
        }
    )
    # Entry-time forecast/gate columns added to FEATURE_NAMES alongside
    # feature_snapshot: overridable per-row (default matching the old
    # hardcoded _make_entry literals) so tests can exercise column order
    # and NULL-handling for these specifically.
    predicted_return: float = 0.0
    p_profit: float = 0.5
    p_ko: float = 0.05
    uncertainty: float = 0.02
    label: bool = True
    exit_reason: ExitReason = ExitReason.HORIZON
    ambiguous_path: bool = False
    mae: float | None = -0.05
    mfe: float | None = 0.05
    realized_selected_pnl: float | None = 0.0


def _make_entry(idx: int, spec: _RowSpec, base_date: date) -> LedgerEntry:
    entry_date = base_date + timedelta(days=spec.day_offset)
    prediction_time = datetime.combine(entry_date, time(12, 0), tzinfo=UTC)
    exit_due = entry_date + timedelta(days=spec.horizon_days)
    return LedgerEntry(
        run_id="run-1",
        candidate_id=f"cand-{idx}",
        signal_id="sig-1",
        signal_version_hash="sighash-1",
        trial_id="TR-2026Q3-test",
        prediction_time=prediction_time,
        underlying="DAX",
        direction=spec.direction,
        horizon_days=spec.horizon_days,
        regime_bucket=None,
        cluster_id=None,
        feature_hash="fh-1",
        model_hash="mh-1",
        config_hash="ch-1",
        git_commit=None,
        category=Category.WATCH,
        selected_wkn=None,
        selected_isin=f"DE000TEST{idx:03d}",
        issuer="TestBank",
        entry_bid=spec.entry_ask * 0.998,
        entry_ask=spec.entry_ask,
        entry_spread=0.002,
        entry_quote_timestamp=prediction_time,
        entry_underlying_timestamp=prediction_time,
        financing_level_entry=spec.financing_level_entry,
        barrier_entry=spec.barrier_entry,
        ratio=spec.ratio,
        fx=spec.fx,
        predicted_return=spec.predicted_return,
        p_profit=spec.p_profit,
        p_ko=spec.p_ko,
        expected_shortfall=-0.1,
        lcb_ev=0.0,
        uncertainty=spec.uncertainty,
        shrinkage_intensity=0.5,
        is_shadow=spec.is_shadow,
        shadow_stratum=None,
        suggested_position_fraction=None,
        exit_due=exit_due,
        alternatives=[],
        feature_snapshot=spec.feature_snapshot,
        status=LedgerEntryStatus.OPEN,
    )


def _make_label(entry_id: str, spec: _RowSpec, labeled_at: datetime) -> LedgerLabel:
    return LedgerLabel(
        entry_id=entry_id,
        labeled_at=labeled_at,
        exit_bid=None,
        exit_quote_timestamp=None,
        financing_level_exit=None,
        exit_reason=spec.exit_reason,
        realized_selected_pnl=spec.realized_selected_pnl,
        underlying_pnl=None,
        median_turbo_pnl=None,
        best_turbo_pnl=None,
        ideal_turbo_pnl=None,
        mfe=spec.mfe,
        mae=spec.mae,
        ko_hit=False,
        time_to_ko_days=None,
        ambiguous_path=spec.ambiguous_path,
    )


def _populate(
    store_: Store, specs: list[_RowSpec], base_date: date = date(2026, 1, 1)
) -> list[LedgerEntry]:
    entries = [_make_entry(i, s, base_date) for i, s in enumerate(specs)]
    inserted = store_.append_ledger_entries(entries)
    assert inserted == len(entries)
    for entry, spec in zip(entries, specs, strict=True):
        if spec.label:
            store_.attach_ledger_label(_make_label(entry.entry_id, spec, entry.prediction_time))
    return entries


def _signal_specs(n: int, seed: int, *, mae_mode: str, mfe_mode: str) -> list[_RowSpec]:
    """``n`` non-overlapping (5-day-spaced, 3-day horizon) rows.

    ``mae_mode``/``mfe_mode`` of ``"signal"`` makes that target a (nearly
    noiseless) linear function of the ``signal_score`` feature;
    ``"noise"`` makes it independent random noise unrelated to any feature.
    """
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(n):
        signal = float(rng.uniform(-1.0, 1.0))
        snapshot = {
            "cost_rank_score": float(rng.uniform(0.0, 1.0)),
            "leverage": float(rng.uniform(1.0, 5.0)),
            "premium_over_fair": float(rng.uniform(-0.02, 0.02)),
            "signal_score": signal,
            "spread_pct": float(rng.uniform(0.0001, 0.01)),
        }
        direction = Direction.LONG if i % 2 == 0 else Direction.SHORT
        entry_ask = float(rng.uniform(4.0, 6.0))

        mae = (
            -0.5 - 0.3 * signal + float(rng.normal(0.0, 0.01))
            if mae_mode == "signal"
            else float(rng.normal(-0.3, 0.1))
        )
        mfe = (
            0.5 + 0.3 * signal + float(rng.normal(0.0, 0.01))
            if mfe_mode == "signal"
            else float(rng.normal(0.3, 0.1))
        )
        if mae > mfe:
            mae, mfe = mfe, mae

        realized_pnl = float(rng.normal(0.0, 0.05))

        specs.append(
            _RowSpec(
                day_offset=i * 5,
                direction=direction,
                entry_ask=entry_ask,
                feature_snapshot=snapshot,
                mae=mae,
                mfe=mfe,
                realized_selected_pnl=realized_pnl,
            )
        )
    return specs


# --------------------------------------------------------------------------
# Leakage guard
# --------------------------------------------------------------------------


def test_leak_assertion_fires_if_a_label_column_is_added_to_the_allow_list() -> None:
    with pytest.raises(AssertionError, match="mae"):
        excursion_eval._check_no_label_leakage((*FEATURE_NAMES, "mae"))
    with pytest.raises(AssertionError, match="realized_selected_pnl"):
        excursion_eval._check_no_label_leakage((*FEATURE_NAMES, "realized_selected_pnl"))
    # The real allow-list itself must not raise (also exercised at import time).
    excursion_eval._check_no_label_leakage(FEATURE_NAMES)


# --------------------------------------------------------------------------
# Dataset assembly: drop-row counting, ordering, ambiguous exclusion
# --------------------------------------------------------------------------


def test_dropped_row_counting_per_reason(store: Store) -> None:
    specs = [
        _RowSpec(day_offset=0),  # kept
        _RowSpec(day_offset=1, label=False),  # no_label: never labeled
        _RowSpec(
            day_offset=2,
            exit_reason=ExitReason.EXPIRED_NO_DATA,
            ambiguous_path=True,
            mae=None,
            mfe=None,
            realized_selected_pnl=None,
        ),  # exit_reason_expired_no_data
        _RowSpec(
            day_offset=3,
            exit_reason=ExitReason.NO_EXIT_QUOTE_CONSERVATIVE,
            ambiguous_path=True,
        ),  # ambiguous_path (mae/mfe/pnl still present, per labeler.py)
        _RowSpec(day_offset=4, mae=None),  # null_target (defensive branch)
        _RowSpec(day_offset=5, financing_level_entry=None),  # null_feature
        _RowSpec(day_offset=6),  # kept
    ]
    _populate(store, specs)

    ds = build_excursion_dataset(store, as_of=_FAR_FUTURE)

    assert ds.dropped_rows == {
        "no_label": 1,
        "exit_reason_expired_no_data": 1,
        "ambiguous_path": 1,
        "null_target": 1,
        "null_feature": 1,
    }
    assert ds.x.shape == (2, len(FEATURE_NAMES))


def test_ambiguous_path_rows_excluded(store: Store) -> None:
    """Master Spec rule 17: an ambiguous bar is never optimistically resolved --
    and never silently treated as if it were an unambiguous observation either."""
    specs = [
        _RowSpec(day_offset=0),
        _RowSpec(
            day_offset=1, exit_reason=ExitReason.NO_EXIT_QUOTE_CONSERVATIVE, ambiguous_path=True
        ),
        _RowSpec(
            day_offset=2, exit_reason=ExitReason.NO_EXIT_QUOTE_CONSERVATIVE, ambiguous_path=True
        ),
        _RowSpec(day_offset=3),
    ]
    _populate(store, specs)

    ds = build_excursion_dataset(store, as_of=_FAR_FUTURE)

    assert ds.dropped_rows["ambiguous_path"] == 2
    assert ds.x.shape[0] == 2


def test_t0_sorted_ascending_regardless_of_insertion_order(store: Store) -> None:
    specs = [_RowSpec(day_offset=20), _RowSpec(day_offset=0), _RowSpec(day_offset=10)]
    _populate(store, specs)

    ds = build_excursion_dataset(store, as_of=_FAR_FUTURE)

    assert ds.x.shape[0] == 3
    assert np.all(np.diff(ds.t0) >= 0)
    assert ds.t0.tolist() == [0, 10, 20]


def test_new_entry_time_features_reach_matrix_in_column_order() -> None:
    """p_ko/predicted_return/p_profit/uncertainty must land at exactly their
    :data:`FEATURE_NAMES` index, not be dropped, reordered, or conflated with
    each other -- the whole point of the fix is that these four specific
    columns reach ``x``.
    """
    assert FEATURE_NAMES[-4:] == ("p_ko", "predicted_return", "p_profit", "uncertainty")

    spec = _RowSpec(
        day_offset=0, p_ko=0.37, predicted_return=0.021, p_profit=0.64, uncertainty=0.09
    )
    entry = _make_entry(0, spec, date(2026, 1, 1))

    row = excursion_eval._row_features(entry)

    assert row.shape == (len(FEATURE_NAMES),)
    assert row[FEATURE_NAMES.index("p_ko")] == pytest.approx(0.37)
    assert row[FEATURE_NAMES.index("predicted_return")] == pytest.approx(0.021)
    assert row[FEATURE_NAMES.index("p_profit")] == pytest.approx(0.64)
    assert row[FEATURE_NAMES.index("uncertainty")] == pytest.approx(0.09)


def test_null_new_feature_counted_in_dropped_rows_not_imputed(store: Store) -> None:
    """A non-finite value in one of the new columns must be dropped
    (``null_feature``), never silently imputed (Master Spec rule 29).

    ``predicted_return`` is a plain, unconstrained ``float`` on
    ``LedgerEntry`` (unlike ``p_ko``/``p_profit``, which are ``UnitFloat``,
    or ``uncertainty``, which has ``Field(ge=0)`` -- both reject ``NaN``
    via their own range validators before a row could ever reach this far).
    It is therefore the one new column that can carry a non-finite value
    into ``_row_features`` if the ``DOUBLE NOT NULL`` database constraint
    were ever violated, and this test exercises exactly that path.
    """
    specs = [
        _RowSpec(day_offset=0),
        _RowSpec(day_offset=1, predicted_return=float("nan")),
        _RowSpec(day_offset=2),
    ]
    _populate(store, specs)

    ds = build_excursion_dataset(store, as_of=_FAR_FUTURE)

    assert ds.dropped_rows["null_feature"] == 1
    assert ds.x.shape[0] == 2
    assert np.all(np.isfinite(ds.x))


# --------------------------------------------------------------------------
# Sample-size gate
# --------------------------------------------------------------------------


def test_insufficient_sample_on_one_distinct_date(store: Store) -> None:
    rng = np.random.default_rng(3)
    specs = [
        _RowSpec(
            day_offset=0,
            direction=Direction.LONG if i % 2 == 0 else Direction.SHORT,
            entry_ask=float(rng.uniform(4.0, 6.0)),
            feature_snapshot={
                "cost_rank_score": float(rng.uniform(0.0, 1.0)),
                "leverage": float(rng.uniform(1.0, 5.0)),
                "premium_over_fair": float(rng.uniform(-0.02, 0.02)),
                "signal_score": float(rng.uniform(-1.0, 1.0)),
                "spread_pct": float(rng.uniform(0.0001, 0.01)),
            },
        )
        for i in range(50)
    ]
    _populate(store, specs)

    ds = build_excursion_dataset(store, as_of=_FAR_FUTURE)
    assert ds.n_distinct_dates == 1

    splitter = PurgedWalkForwardSplit(horizon=3, embargo=3, n_splits=1)
    result = evaluate_excursion(ds, splitter=splitter)

    assert result.verdict == "INSUFFICIENT_SAMPLE"
    assert result.results == []
    assert any("effective_sample" in r for r in result.verdict_reasons)
    assert any("n_distinct_dates" in r for r in result.verdict_reasons)

    # Same-day dataset -> effective_sample collapses to ~1.0 (average
    # uniqueness) -> feature_sample_ratio is necessarily far past the thin
    # bar, and that must show up in verdict_reasons, not just the field.
    expected_ratio = len(FEATURE_NAMES) / ds.effective_sample
    assert result.feature_sample_ratio == pytest.approx(expected_ratio)
    assert any("feature_sample_ratio" in r for r in result.verdict_reasons)


def test_multi_date_dataset_produces_a_real_comparison(store: Store) -> None:
    specs = _signal_specs(200, seed=5, mae_mode="signal", mfe_mode="signal")
    _populate(store, specs)

    ds = build_excursion_dataset(store, as_of=_FAR_FUTURE)
    assert ds.n_distinct_dates >= MIN_DISTINCT_DATES
    assert ds.effective_sample >= MIN_EFFECTIVE_SAMPLE

    splitter = PurgedWalkForwardSplit(horizon=3, embargo=3, n_splits=4)
    result = evaluate_excursion(ds, splitter=splitter)

    assert result.verdict != "INSUFFICIENT_SAMPLE"
    assert len(result.results) == 2
    for target_result in result.results:
        assert target_result.folds
        assert target_result.n_test_samples > 0
        assert np.isfinite(target_result.crps_null)
        assert np.isfinite(target_result.crps_conditional)
        assert 0.0 <= target_result.coverage_null <= 1.0
        assert 0.0 <= target_result.coverage_conditional <= 1.0
    assert result.selective_abstention  # non-empty: 0.05 is in DEFAULT_QUANTILES

    expected_ratio = len(FEATURE_NAMES) / ds.effective_sample
    assert result.feature_sample_ratio == pytest.approx(expected_ratio)
    assert np.isfinite(result.feature_sample_ratio)


# --------------------------------------------------------------------------
# Average-uniqueness weights reach both models
# --------------------------------------------------------------------------


def test_uniqueness_weights_applied_to_both_models(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(7)
    # n=150/train=130 (not the old 100/80): ConditionalExcursionModel's own
    # per-tau sample floor is (n_features + 1) * 10 = 120 with 11 features
    # (up from 80 at 7), so the train split must clear that or fitting
    # raises and the fold is silently dropped by `_evaluate_target`'s
    # try/except, leaving nothing recorded.
    n = 150
    x = rng.normal(size=(n, len(FEATURE_NAMES)))
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 3  # 1-day spacing, 4-day windows -> genuine overlap, weights << 1
    weights = average_uniqueness(t0, t1)
    assert not np.allclose(weights, 1.0)  # sanity: overlap actually produced non-trivial weights

    dataset = ExcursionDataset(
        x=x,
        y_mae=rng.normal(size=n),
        y_mfe=rng.normal(size=n),
        t0=t0,
        t1=t1,
        weights=weights,
        is_shadow=np.zeros(n, dtype=np.bool_),
        realized_pnl=rng.normal(size=n),
        feature_names=FEATURE_NAMES,
        n_distinct_dates=n,
        effective_sample=float(weights.sum()),
        dropped_rows=dict.fromkeys(excursion_eval._DROP_REASONS, 0),
        underlying=None,
        as_of=_FAR_FUTURE,
    )
    train_idx = np.arange(0, 130, dtype=np.int64)
    test_idx = np.arange(130, 150, dtype=np.int64)

    recorded: dict[str, list[np.ndarray]] = {"null": [], "conditional": []}

    class _SpyNull(NullExcursionModel):
        def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            *,
            sample_weight: np.ndarray | None = None,
        ) -> None:
            assert sample_weight is not None
            recorded["null"].append(np.array(sample_weight, copy=True))
            super().fit(x, y, sample_weight=sample_weight)

    class _SpyConditional(ConditionalExcursionModel):
        def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            *,
            sample_weight: np.ndarray | None = None,
        ) -> None:
            assert sample_weight is not None
            recorded["conditional"].append(np.array(sample_weight, copy=True))
            super().fit(x, y, sample_weight=sample_weight)

    monkeypatch.setattr(excursion_eval, "NullExcursionModel", _SpyNull)
    monkeypatch.setattr(excursion_eval, "ConditionalExcursionModel", _SpyConditional)

    excursion_eval._evaluate_target(
        ExcursionTarget.MAE, dataset.y_mae, dataset, [(train_idx, test_idx)], DEFAULT_QUANTILES
    )

    assert len(recorded["null"]) == 1
    assert len(recorded["conditional"]) == 1
    np.testing.assert_array_equal(recorded["null"][0], weights[train_idx])
    np.testing.assert_array_equal(recorded["conditional"][0], weights[train_idx])


# --------------------------------------------------------------------------
# Selective abstention overlay: hand-checkable example
# --------------------------------------------------------------------------


def test_selective_abstention_hand_checkable(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 8
    x = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    y_mae = np.zeros(n, dtype=np.float64)
    t0 = np.arange(n, dtype=np.int64)
    t1 = t0 + 1
    weights = np.ones(n, dtype=np.float64)
    is_shadow = np.array([True, True, True, True, False, False, False, False])
    realized_pnl = np.array([10.0, 20.0, 30.0, 40.0, -5.0, -15.0, -25.0, -35.0])
    # Predicted q05(MAE): the 4 rows with the best realized P&L (indices 0-3,
    # all shadow) get the model's least-severe predicted downside; the other
    # 4 get the worst. This is deliberately clean so "keep the upper half by
    # predicted q05" is hand-verifiable against "keep the 4 best P&L rows".
    predicted_q05 = np.array([-0.01, -0.02, -0.03, -0.04, -0.10, -0.11, -0.12, -0.13])

    dataset = ExcursionDataset(
        x=x,
        y_mae=y_mae,
        y_mfe=y_mae.copy(),
        t0=t0,
        t1=t1,
        weights=weights,
        is_shadow=is_shadow,
        realized_pnl=realized_pnl,
        feature_names=FEATURE_NAMES,
        n_distinct_dates=n,
        effective_sample=float(n),
        dropped_rows=dict.fromkeys(excursion_eval._DROP_REASONS, 0),
        underlying=None,
        as_of=_FAR_FUTURE,
    )

    class _FakeConditional:
        """Bypasses real quantile regression: returns a fixed, known q05 per row."""

        model_id = "fake_conditional"

        def __init__(
            self,
            target: ExcursionTarget,
            *,
            quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
        ) -> None:
            self._levels = quantile_levels

        def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            *,
            sample_weight: np.ndarray | None = None,
        ) -> None:
            return None

        def predict(self, x: np.ndarray) -> list[ExcursionPrediction]:
            preds = []
            for i in range(x.shape[0]):
                running = float("-inf")
                quantiles: dict[float, float] = {}
                for tau in sorted(self._levels):
                    raw = predicted_q05[i] if tau == 0.05 else 0.0
                    value = max(raw, running)
                    quantiles[tau] = value
                    running = value
                preds.append(ExcursionPrediction(target=ExcursionTarget.MAE, quantiles=quantiles))
            return preds

    monkeypatch.setattr(excursion_eval, "ConditionalExcursionModel", _FakeConditional)

    folds_idx = [(np.arange(0, dtype=np.int64), np.arange(n, dtype=np.int64))]
    result = excursion_eval._selective_abstention(dataset, folds_idx, DEFAULT_QUANTILES)

    median = float(np.median(predicted_q05))
    keep_mask = predicted_q05 >= median
    expected_kept_fraction = float(np.sum(keep_mask)) / n
    expected_mean_kept = float(np.mean(realized_pnl[keep_mask]))
    expected_mean_all = float(np.mean(realized_pnl))

    assert result["kept_fraction"] == pytest.approx(expected_kept_fraction)
    assert result["mean_pnl_kept"] == pytest.approx(expected_mean_kept)
    assert result["mean_pnl_all"] == pytest.approx(expected_mean_all)
    assert result["mean_pnl_kept"] == pytest.approx(25.0)
    assert result["mean_pnl_all"] == pytest.approx(2.5)
    assert result["n_all"] == float(n)
    assert result["n_kept"] == 4.0
    assert result["n_shadow_all"] == 4.0
    assert result["n_shadow_kept"] == 4.0
    assert result["n_nonshadow_all"] == 4.0
    assert result["n_nonshadow_kept"] == 0.0


# --------------------------------------------------------------------------
# Mandatory anti-triviality tests
# --------------------------------------------------------------------------


def test_signal_dependent_mae_does_not_yield_no_improvement(store: Store) -> None:
    """A harness that always reports NO_IMPROVEMENT must fail this test.

    MAE here is (almost) a deterministic linear function of ``signal_score``;
    a working evaluation harness must detect that the conditional model beats
    the unconditional null on at least one metric bundle, i.e. the per-target
    verdict for MAE must clear ``NO_IMPROVEMENT`` (it need not reach
    ``CANDIDATE``, since that also requires fold-level significance).
    """
    specs = _signal_specs(300, seed=11, mae_mode="signal", mfe_mode="noise")
    _populate(store, specs)

    ds = build_excursion_dataset(store, as_of=_FAR_FUTURE)
    splitter = PurgedWalkForwardSplit(horizon=3, embargo=3, n_splits=5)
    result = evaluate_excursion(ds, splitter=splitter)

    assert result.verdict != "INSUFFICIENT_SAMPLE"
    assert result.verdict != "NO_IMPROVEMENT", result.verdict_reasons


def test_pure_noise_dataset_does_not_yield_candidate(store: Store) -> None:
    """A harness that fabricates significance must fail this test.

    Both MAE and MFE are pure noise, independent of every feature, with
    adequate sample size. The conditional model must not be reported as a
    ``CANDIDATE`` on data that carries no real signal.
    """
    specs = _signal_specs(300, seed=12, mae_mode="noise", mfe_mode="noise")
    _populate(store, specs)

    ds = build_excursion_dataset(store, as_of=_FAR_FUTURE)
    splitter = PurgedWalkForwardSplit(horizon=3, embargo=3, n_splits=5)
    result = evaluate_excursion(ds, splitter=splitter)

    assert result.verdict != "INSUFFICIENT_SAMPLE"
    assert result.verdict != "CANDIDATE", result.verdict_reasons
