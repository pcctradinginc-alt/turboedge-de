"""Tests for reporting/weekly.py: research tournament, BH correction,
protected-baseline visibility, promotion/demotion suggestions."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

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
    assert family_result.psr == 0.72
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
