"""Tests for reporting/weekly.py: research tournament, BH correction,
protected-baseline visibility, promotion/demotion suggestions."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
from scipy import stats

from turboedge.backtest.significance import probabilistic_sharpe_ratio
from turboedge.learning.registry import ModelRegistry
from turboedge.reporting.weekly import WeeklyTournamentConfig, run_research_tournament
from turboedge.storage.schemas import Category, ModelStatus

_AS_OF = datetime(2026, 9, 1, 6, 0, tzinfo=UTC)


def _register(
    store,
    model_id: str,
    model_hash: str,
    signal_family: str,
    status: ModelStatus,
    *,
    weight: float = 1.0,
    trial_id: str | None = None,
) -> None:
    registry = ModelRegistry(store)
    registry.register(
        model_id,
        model_hash,
        signal_family,
        {},
        trial_id,
        status=status,
        initial_weight=weight,
        now=datetime(2026, 8, 1, tzinfo=UTC),
    )


def test_protected_family_always_visible_even_without_trades(store) -> None:
    _register(store, "tsmom_horizon_norm_v1", "hash-tsmom", "tsmom", ModelStatus.PROTECTED)

    report = run_research_tournament(
        store, as_of=_AS_OF, cfg=WeeklyTournamentConfig(lookback_days=30)
    )

    families = {f.signal_family: f for f in report.families}
    assert "tsmom" in families
    assert families["tsmom"].protected is True
    assert families["tsmom"].n == 0
    assert families["tsmom"].reliable is False
    assert families["tsmom"].sharpe is None


def test_identical_window_and_bh_correction_ranks_families(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    _register(store, "tsmom_horizon_norm_v1", "hash-tsmom", "tsmom", ModelStatus.PROTECTED)
    _register(store, "champion_v1", "hash-champ", "trend_vol", ModelStatus.CHAMPION)
    _register(
        store,
        "good_v1",
        "hash-good",
        "good_family",
        ModelStatus.CHALLENGER,
        trial_id="TR-2026Q3-good1",
    )
    _register(
        store,
        "noisy_v1",
        "hash-noisy",
        "noisy_family",
        ModelStatus.CHALLENGER,
        trial_id="TR-2026Q3-noisy1",
    )

    def _fill(model_hash: str, signal_id: str, prefix: str, bids: list[float]) -> None:
        for i, bid in enumerate(bids):
            pred = datetime(2026, 8, 1 + i, 8, 0, tzinfo=UTC)
            exit_due = date(2026, 8, 1 + i) + timedelta(days=3)
            record_labeled(
                entry_overrides=dict(
                    candidate_id=f"{prefix}-{i}",
                    category=Category.ACTIONABLE,
                    model_hash=model_hash,
                    signal_id=signal_id,
                    prediction_time=pred,
                    exit_due=exit_due,
                    horizon_days=3,
                ),
                label_overrides=dict(exit_bid=bid, realized_selected_pnl=bid / 4.86 - 1),
            )

    # good_family: consistently positive with low (but nonzero) variance
    # relative to its mean -> a small p-value under the one-sample t-test.
    _fill("hash-good", "good_family_v1", "good", [5.15, 5.25] * 7 + [5.20])
    # noisy_family: alternating strongly +/- -> mean near zero, large p-value.
    _fill("hash-noisy", "noisy_family_v1", "noisy", [5.50, 4.20] * 8)
    # champion baseline (trend_vol): mild, mixed performance.
    _fill("hash-champ", "trend_vol_v1", "champ", [4.90, 4.82] * 6)

    cfg = WeeklyTournamentConfig(lookback_days=45, min_trades_for_comparison=10)
    report = run_research_tournament(store, as_of=_AS_OF, cfg=cfg)
    families = {f.signal_family: f for f in report.families}

    assert families["good_family"].reliable is True
    assert families["noisy_family"].reliable is True
    assert families["good_family"].p_value is not None
    assert families["noisy_family"].p_value is not None
    assert families["good_family"].p_value < families["noisy_family"].p_value
    assert families["good_family"].bh_rejected is not None
    assert families["noisy_family"].bh_rejected is not None
    # protected tsmom is never included in the tested-family p-value set.
    assert families["tsmom"].p_value is None
    assert families["tsmom"].bh_rejected is None

    promo_by_model = {p.challenger_model_id: p for p in report.promotions}
    assert set(promo_by_model) == {"good_v1", "noisy_v1"}
    assert promo_by_model["good_v1"].trial_id == "TR-2026Q3-good1"
    assert promo_by_model["noisy_v1"].trial_id == "TR-2026Q3-noisy1"
    # Promotion is only ever a suggestion -- registry status is untouched.
    assert store.get_model_registry_entry("good_v1").status == ModelStatus.CHALLENGER

    assert report.champion_model_id == "champion_v1"
    assert {w.model_id for w in report.weight_preview} == {
        "tsmom_horizon_norm_v1",
        "champion_v1",
        "good_v1",
        "noisy_v1",
    }
    assert any("nicht im Store persistiert" in n for n in report.notes)


def test_demotion_suggested_for_poor_champion(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    _register(store, "tsmom_horizon_norm_v1", "hash-tsmom", "tsmom", ModelStatus.PROTECTED)
    _register(store, "bad_champ_v1", "hash-bad", "bad_family", ModelStatus.CHAMPION)

    # Alternating (but always losing) exit prices: negative mean return with
    # nonzero variance, so sharpe() does not hit its "flat series -> 0.0"
    # special case -- a genuinely negative Sharpe is what should trip the
    # demotion trigger (GOVERNANCE.md §6.2: "Rolling 4-week Sharpe < 0.0").
    bids = [3.80, 4.20]
    for i in range(12):
        pred = datetime(2026, 8, 1 + i, 8, 0, tzinfo=UTC)
        exit_due = date(2026, 8, 1 + i) + timedelta(days=3)
        bid = bids[i % 2]
        record_labeled(
            entry_overrides=dict(
                candidate_id=f"bad-{i}",
                category=Category.ACTIONABLE,
                model_hash="hash-bad",
                signal_id="bad_family_v1",
                prediction_time=pred,
                exit_due=exit_due,
                horizon_days=3,
            ),
            label_overrides=dict(exit_bid=bid, realized_selected_pnl=bid / 4.86 - 1),
        )

    report = run_research_tournament(
        store, as_of=_AS_OF, cfg=WeeklyTournamentConfig(lookback_days=30)
    )
    demoted = {d.model_id for d in report.demotions}
    assert "bad_champ_v1" in demoted
    # Protected baseline is never a demotion candidate (CLAUDE.md rule 10).
    assert "tsmom_horizon_norm_v1" not in demoted


def test_no_registered_models_returns_empty_but_valid_report(store) -> None:
    report = run_research_tournament(store, as_of=_AS_OF)
    assert report.champion_model_id is None
    assert report.weight_preview == []
    assert any("Kein Modell mit status=champion" in n for n in report.notes)
    # Protected family is still synthesized into the comparison, even with
    # no registry row and no trades.
    assert any(f.signal_family == "tsmom" for f in report.families)


def test_deterministic_across_repeated_calls(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    _register(store, "tsmom_horizon_norm_v1", "hash-tsmom", "tsmom", ModelStatus.PROTECTED)
    # Slight variance (not a literal constant series) so PSR/DSR resolve to
    # finite numbers rather than NaN (a perfectly flat return series makes
    # scipy's skew/kurtosis divide by a zero variance) -- this test checks
    # reproducibility, not degenerate-input handling.
    bids = [5.00, 5.02]
    for i in range(10):
        pred = datetime(2026, 8, 1 + i, 8, 0, tzinfo=UTC)
        exit_due = date(2026, 8, 1 + i) + timedelta(days=3)
        bid = bids[i % 2]
        record_labeled(
            entry_overrides=dict(
                candidate_id=f"det-{i}",
                category=Category.ACTIONABLE,
                model_hash="hash-tsmom",
                signal_id="tsmom_horizon_norm_v1",
                prediction_time=pred,
                exit_due=exit_due,
                horizon_days=3,
            ),
            label_overrides=dict(exit_bid=bid, realized_selected_pnl=bid / 4.86 - 1),
        )
    r1 = run_research_tournament(store, as_of=_AS_OF)
    r2 = run_research_tournament(store, as_of=_AS_OF)
    assert r1.families == r2.families


def test_walkforward_fallback_when_no_forward_ledger(store) -> None:
    """Family with walk-forward backtest results but no forward-ledger trades
    should show walk-forward data with correct evidence_source marker."""
    from turboedge.storage.schemas import WalkforwardResultRecord

    _register(store, "test_model", "hash-test", "test_family", ModelStatus.PROTECTED)

    # Add a walk-forward result with no corresponding forward-ledger trades
    wf_record = WalkforwardResultRecord(
        model_id="test_model",
        model_hash="hash-test",
        signal_family="test_family",
        underlying_id="DAX",
        horizon_days=5,
        evaluated_at=datetime(2026, 8, 28, 12, 0, tzinfo=UTC),
        n_folds=5,
        brier=0.22,
        brier_null=0.25,
        log_loss=0.65,
        ece=0.08,
        hit_rate=0.58,
        mean_oos_return=0.0025,
        psr=0.72,
        n_effective=42.5,
        config_hash="cfg-hash",
        git_commit="abc1234",
        params={},
    )
    store.append_walkforward_results([wf_record])

    report = run_research_tournament(
        store, as_of=_AS_OF, cfg=WeeklyTournamentConfig(lookback_days=30)
    )

    families = {f.signal_family: f for f in report.families}
    assert "test_family" in families
    family_result = families["test_family"]
    assert family_result.evidence_source == "walkforward_backtest"
    assert family_result.mean_return == 0.0025
    # n_effective=42.5 is below the 100-effective-sample promotion floor
    # (GOVERNANCE.md §6.1) that now applies uniformly to PSR/DSR regardless
    # of evidence_source -- the stored walk-forward psr=0.72 must NOT be
    # surfaced as a number here, the same as a forward-ledger family below
    # the floor (see test_psr_dsr_withheld_below_effective_sample_floor).
    assert family_result.psr is None
    assert any("n_effective=42.50" in note and "100" in note for note in family_result.notes)
    assert any("Walk-Forward-Backtest" in note for note in family_result.notes)
    assert any("keine automatische Promotion" in note for note in family_result.notes)


def test_forward_ledger_preferred_over_walkforward(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    """When both forward-ledger and walk-forward data exist for a family,
    forward-ledger takes precedence and evidence_source is marked correctly."""
    from turboedge.storage.schemas import WalkforwardResultRecord

    _register(store, "champ_model", "hash-champ", "dual_family", ModelStatus.CHAMPION)

    # Add forward-ledger trades
    pred = datetime(2026, 8, 15, 8, 0, tzinfo=UTC)
    exit_due = date(2026, 8, 15) + timedelta(days=5)
    record_labeled(
        entry_overrides=dict(
            candidate_id="dual-1",
            category=Category.ACTIONABLE,
            model_hash="hash-champ",
            signal_id="dual_family_v1",
            prediction_time=pred,
            exit_due=exit_due,
            horizon_days=5,
        ),
        label_overrides=dict(exit_bid=5.10, realized_selected_pnl=0.05),
    )

    # Also add walk-forward results
    wf_record = WalkforwardResultRecord(
        model_id="champ_model",
        model_hash="hash-champ",
        signal_family="dual_family",
        underlying_id="DAX",
        horizon_days=5,
        evaluated_at=datetime(2026, 8, 28, 12, 0, tzinfo=UTC),
        n_folds=5,
        brier=0.20,
        brier_null=0.25,
        log_loss=0.60,
        ece=0.06,
        hit_rate=0.62,
        mean_oos_return=0.0015,  # Different from forward-ledger
        psr=0.68,
        n_effective=45.0,
        config_hash="cfg-hash",
        git_commit="abc1234",
        params={},
    )
    store.append_walkforward_results([wf_record])

    report = run_research_tournament(
        store, as_of=_AS_OF, cfg=WeeklyTournamentConfig(lookback_days=30)
    )

    families = {f.signal_family: f for f in report.families}
    assert "dual_family" in families
    family_result = families["dual_family"]
    # Forward-ledger should win
    assert family_result.evidence_source == "forward_ledger"
    assert family_result.n == 1  # One forward-ledger trade
    # Mean return should come from forward-ledger, not walk-forward
    assert family_result.mean_return == 0.05


def test_backtest_evidence_blocks_promotion(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    """A challenger model with only backtest evidence should never be promoted,
    even if it passes all other ladder checks."""
    from turboedge.storage.schemas import WalkforwardResultRecord

    _register(store, "champion_model", "hash-champ", "champ_fam", ModelStatus.CHAMPION)
    _register(
        store,
        "challenger_backtest_only",
        "hash-chall-bt",
        "chall_fam",
        ModelStatus.CHALLENGER,
        trial_id="TR-2026Q3-backtest",
    )

    # Champion has some forward-ledger trades
    for i in range(12):
        pred = datetime(2026, 8, 1 + i, 8, 0, tzinfo=UTC)
        exit_due = date(2026, 8, 1 + i) + timedelta(days=3)
        record_labeled(
            entry_overrides=dict(
                candidate_id=f"champ-{i}",
                category=Category.ACTIONABLE,
                model_hash="hash-champ",
                signal_id="champ_fam_v1",
                prediction_time=pred,
                exit_due=exit_due,
                horizon_days=3,
            ),
            label_overrides=dict(exit_bid=5.00, realized_selected_pnl=0.01),
        )

    # Challenger has only backtest results, even with great metrics
    wf_record = WalkforwardResultRecord(
        model_id="challenger_backtest_only",
        model_hash="hash-chall-bt",
        signal_family="chall_fam",
        underlying_id="DAX",
        horizon_days=3,
        evaluated_at=datetime(2026, 8, 28, 12, 0, tzinfo=UTC),
        n_folds=5,
        brier=0.15,
        brier_null=0.25,
        log_loss=0.45,
        ece=0.03,
        hit_rate=0.75,
        mean_oos_return=0.050,  # Excellent return
        psr=0.95,  # Passes PSR threshold
        n_effective=50.0,
        config_hash="cfg-hash",
        git_commit="abc1234",
        params={},
    )
    store.append_walkforward_results([wf_record])

    report = run_research_tournament(
        store, as_of=_AS_OF, cfg=WeeklyTournamentConfig(lookback_days=30)
    )

    promo_by_model = {p.challenger_model_id: p for p in report.promotions}
    assert "challenger_backtest_only" in promo_by_model
    # Should NOT pass ladder due to backtest_only check
    promo = promo_by_model["challenger_backtest_only"]
    assert promo.passes_ladder is False
    assert any("backtest_only: failed" in r for r in promo.reasons)


def test_same_day_family_with_positive_mean_is_not_manufactured_significant(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    """MANDATORY anti-triviality pin: the exact 2026-09-26 false-positive
    tournament bug (docs/measured_results.md §6.9/§6.11).

    68-ish same-day positions from a single scan run were reported as
    Benjamini-Hochberg significant with Sharpe 9.6-10.5, purely because the
    tournament fed PSR/DSR/the t-test a row count (n=68) instead of the
    effective sample (~1, since every row shares one label window). This
    builds exactly that shape -- 70 rows, one prediction date, one horizon,
    a *positive* mean return with real (non-flat) dispersion large enough
    that the old row-count path calls it significant -- and pins that the
    new path does not.
    """
    _register(store, "tsmom_horizon_norm_v1", "hash-tsmom", "tsmom", ModelStatus.PROTECTED)
    _register(store, "champion_v1", "hash-champ", "champ_family", ModelStatus.CHAMPION)
    _register(
        store,
        "same_day_v1",
        "hash-sameday",
        "same_day_family",
        ModelStatus.CHALLENGER,
        trial_id="TR-2026Q3-sameday",
    )

    pred = datetime(2026, 8, 15, 8, 0, tzinfo=UTC)
    exit_due = date(2026, 8, 18)
    n_rows = 70
    # Alternating, never-flat, always-positive returns (never a constant
    # series -- see backtest/significance.py's _MIN_RELATIVE_DISPERSION
    # docstring for why that distinction matters).
    bids = [5.05, 5.15]
    for i in range(n_rows):
        bid = bids[i % 2]
        record_labeled(
            entry_overrides=dict(
                candidate_id=f"sameday-{i}",
                category=Category.ACTIONABLE,
                model_hash="hash-sameday",
                signal_id="same_day_family_v1",
                prediction_time=pred,
                exit_due=exit_due,
                horizon_days=3,
            ),
            label_overrides=dict(exit_bid=bid, realized_selected_pnl=bid / 4.86 - 1),
        )

    cfg = WeeklyTournamentConfig(lookback_days=30, min_trades_for_comparison=10)
    report = run_research_tournament(store, as_of=_AS_OF, cfg=cfg)
    families = {f.signal_family: f for f in report.families}
    result = families["same_day_family"]

    # -- BEFORE: what the old row-count-as-sample-size path reports --
    returns = np.array([bids[i % 2] / 4.86 - 1 for i in range(n_rows)], dtype=np.float64)
    old_mean = float(np.mean(returns))
    old_std = float(np.std(returns, ddof=1))
    old_t_stat = old_mean / (old_std / math.sqrt(n_rows))
    old_p_value = float(1.0 - stats.t.cdf(old_t_stat, df=n_rows - 1))
    old_psr = probabilistic_sharpe_ratio(returns)
    assert old_p_value < 0.01  # old path: comfortably "significant"
    assert old_psr > cfg.ladder_min_psr  # old path: comfortably clears the PSR>=0.95 ladder gate

    # -- AFTER: the actual report --
    assert result.n == n_rows
    # All 70 rows share one 4-day label window -> exactly one independent
    # observation, regardless of n.
    assert result.n_effective == pytest.approx(1.0, abs=1e-6)
    assert result.psr is None
    assert result.dsr is None
    assert result.z_score is None
    assert result.p_value is None
    assert result.bh_rejected is None
    assert any("n_effective=" in note for note in result.notes)

    promo = next(p for p in report.promotions if p.challenger_model_id == "same_day_v1")
    assert promo.passes_ladder is False
    assert any("psr>=threshold: failed" in r for r in promo.reasons)
    assert any("dsr>=threshold: failed" in r for r in promo.reasons)
    assert any("bh_rejected: failed" in r for r in promo.reasons)


def test_psr_dsr_withheld_below_effective_sample_floor(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    """A family with a *genuinely* well-separated, non-degenerate n_effective
    that is still below the 100-effective-sample promotion floor
    (GOVERNANCE.md §6.1) must show PSR/DSR as ``None`` with a stated reason
    -- distinct from the anti-triviality case above (which collapses to
    n_effective~1): here n_effective is meaningfully large (~15) but simply
    not enough to clear the floor."""
    _register(store, "tsmom_horizon_norm_v1", "hash-tsmom", "tsmom", ModelStatus.PROTECTED)
    n_rows = 15
    bids = [5.00, 5.10]
    for i in range(n_rows):
        pred_date = date(2026, 1, 1) + timedelta(days=5 * i)
        pred = datetime(pred_date.year, pred_date.month, pred_date.day, 8, 0, tzinfo=UTC)
        exit_due = pred_date + timedelta(days=2)
        record_labeled(
            entry_overrides=dict(
                candidate_id=f"floor-{i}",
                category=Category.ACTIONABLE,
                model_hash="hash-floor",
                signal_id="floor_family_v1",
                prediction_time=pred,
                exit_due=exit_due,
                horizon_days=2,
            ),
            label_overrides=dict(
                exit_bid=bids[i % 2], realized_selected_pnl=bids[i % 2] / 4.86 - 1
            ),
        )
    as_of = datetime(2026, 4, 1, 6, 0, tzinfo=UTC)
    cfg = WeeklyTournamentConfig(lookback_days=120, min_trades_for_comparison=10)
    report = run_research_tournament(store, as_of=as_of, cfg=cfg)
    families = {f.signal_family: f for f in report.families}
    result = families["floor_family"]

    assert result.n == n_rows
    # Well-separated windows -> effective sample ~= row count, i.e. this is
    # a real, non-degenerate n_effective -- just below the floor.
    assert result.n_effective == pytest.approx(float(n_rows), rel=1e-6)
    assert result.n_effective < 100.0
    assert result.psr is None
    assert result.dsr is None
    assert any(
        "n_effective=" in note and "100" in note and "PSR/DSR" in note for note in result.notes
    )


def test_psr_dsr_computed_once_effective_sample_clears_the_floor(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    """Positive control for the floor: once n_effective genuinely clears
    GOVERNANCE.md §6.1's 100-distinct-outcomes bar, PSR/DSR are computed
    (not withheld), sourced from the effective sample rather than the row
    count (which happen to coincide here, since every window is
    well-separated)."""
    n_rows = 110
    bids = [5.00, 5.05]
    for i in range(n_rows):
        pred_date = date(2026, 1, 1) + timedelta(days=2 * i)
        pred = datetime(pred_date.year, pred_date.month, pred_date.day, 8, 0, tzinfo=UTC)
        exit_due = pred_date + timedelta(days=1)
        record_labeled(
            entry_overrides=dict(
                candidate_id=f"indep-{i}",
                category=Category.ACTIONABLE,
                model_hash="hash-indep",
                signal_id="indep_family_v1",
                prediction_time=pred,
                exit_due=exit_due,
                horizon_days=1,
            ),
            label_overrides=dict(
                exit_bid=bids[i % 2], realized_selected_pnl=bids[i % 2] / 4.86 - 1
            ),
        )
    as_of = datetime(2026, 1, 1, 6, 0, tzinfo=UTC) + timedelta(days=2 * n_rows + 10)
    cfg = WeeklyTournamentConfig(lookback_days=2 * n_rows + 20, min_trades_for_comparison=10)
    report = run_research_tournament(store, as_of=as_of, cfg=cfg)
    families = {f.signal_family: f for f in report.families}
    result = families["indep_family"]

    assert result.n == n_rows
    assert result.n_effective == pytest.approx(float(n_rows), rel=1e-6)
    assert result.psr is not None
    assert result.dsr is not None
    assert 0.0 <= result.psr <= 1.0
    assert 0.0 <= result.dsr <= 1.0


def test_sharpe_is_per_trade_and_not_annualized(
    store, make_ledger_entry, make_ledger_label, record_labeled
) -> None:
    """docs/measured_results.md §6.9/§6.11: annualizing per-trade returns
    held 3-14 days manufactures a large number from a modest effect (a
    per-trade Sharpe of 0.66 was reported as 10.49, a sqrt(252)x inflation).
    ``FamilyResult.sharpe`` must be the plain per-trade ratio, not scaled by
    any periods-per-year assumption.
    """
    _register(store, "tsmom_horizon_norm_v1", "hash-tsmom", "tsmom", ModelStatus.PROTECTED)
    bids = [5.00, 5.30, 4.90, 5.10, 5.20, 4.95, 5.05, 5.25, 4.85, 5.15]
    for i, bid in enumerate(bids):
        # 5-day spacing with a 3-day horizon -> non-overlapping windows, so
        # weighted mean == plain mean and this admits an exact comparison.
        pred_date = date(2026, 8, 1) + timedelta(days=5 * i)
        pred = datetime(pred_date.year, pred_date.month, pred_date.day, 8, 0, tzinfo=UTC)
        exit_due = pred_date + timedelta(days=3)
        record_labeled(
            entry_overrides=dict(
                candidate_id=f"sh-{i}",
                category=Category.ACTIONABLE,
                model_hash="hash-sh",
                signal_id="sharpe_family_v1",
                prediction_time=pred,
                exit_due=exit_due,
                horizon_days=3,
            ),
            label_overrides=dict(exit_bid=bid, realized_selected_pnl=bid / 4.86 - 1),
        )
    as_of = datetime(2026, 10, 1, 6, 0, tzinfo=UTC)
    report = run_research_tournament(
        store,
        as_of=as_of,
        cfg=WeeklyTournamentConfig(lookback_days=70, min_trades_for_comparison=10),
    )
    families = {f.signal_family: f for f in report.families}
    result = families["sharpe_family"]

    returns = np.array([bid / 4.86 - 1 for bid in bids], dtype=np.float64)
    plain_sharpe = float(np.mean(returns) / np.std(returns, ddof=1))
    assert result.sharpe is not None
    assert result.sharpe == pytest.approx(plain_sharpe, abs=1e-6)
    # The old sqrt(252)-annualized (or sqrt(252/holding_days)-annualized)
    # convention would put this at ~9-16x plain_sharpe; a bound well below
    # that pins the regression.
    assert abs(result.sharpe) < 5.0
