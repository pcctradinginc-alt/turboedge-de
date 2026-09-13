"""Weekly research tournament (Master Spec §36 "Research Tournament", §27.3
"Ladder Rule", §27.4 "Multiple Testing").

Compares every registered signal family's forward-ledger performance over an
*identical* walk-forward window ("Immer identische Walk-forward-Fenster"),
applies Benjamini-Hochberg FDR correction across the tested (non-protected)
families, and turns the result into promotion/demotion *suggestions* plus an
ensemble-weight-update *preview* -- never an actual promotion or reweighting.
Only :func:`turboedge.learning.registry.promote_if_ladder` (called
separately, by a human-in-the-loop CLI step in the integration wave) and
:meth:`turboedge.learning.registry.ModelRegistry.update_weights` mutate the
registry; this module only ever reads.

W4's walk-forward evaluation (``backtest/walkforward.py``) is not currently
persisted anywhere the ``Store`` can read back (no such table/method exists
yet), so this always takes the "otherwise use forward-ledger data per signal
family" branch described in Build Contract v2 W8 -- the moment a store-backed
walk-forward result becomes available, this module is the natural place to
prefer it.

Protected TSMOM is always shown in the comparison table, even when it has too
few matured trades in the window to be "tested" (Master Spec §21: "Protected
Baseline bleibt unabhängig davon immer sichtbar").
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from turboedge.backtest.metrics import brier_score, expected_calibration_error, sharpe
from turboedge.backtest.significance import (
    benjamini_hochberg,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from turboedge.learning.ensemble_weights import update_weights as _compute_weights_preview
from turboedge.learning.ledger import ForwardLedger
from turboedge.reporting._common import (
    average_uniqueness_weights,
    holding_days,
    registry_hash_map,
    signal_family_for,
)
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    LedgerEntry,
    LedgerEntryStatus,
    LedgerLabel,
    ModelStatus,
)

_TRADING_DAYS_PER_YEAR = 252.0


class WeeklyTournamentConfig(BaseModel):
    """This module's own config (wired into config.py/YAML by the
    integration wave). Ladder thresholds default to GOVERNANCE.md §2/§3."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lookback_days: int = Field(default=60, ge=1)
    min_trades_for_comparison: int = Field(default=10, ge=1)
    fdr_alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    ladder_min_zscore: float = Field(default=1.645)
    ladder_min_dsr: float = Field(default=0.6)
    ladder_min_psr: float = Field(default=0.95, gt=0.0, le=1.0)
    ladder_min_ev_improvement: float = Field(default=0.0010)
    ladder_min_winrate_improvement: float = Field(default=0.02)
    reweight_eta: float = Field(default=0.5)
    reweight_w_min: float = Field(default=0.01, gt=0.0)
    protected_signal_family: str = "tsmom"
    demotion_sharpe_threshold: float = Field(default=0.0)
    demotion_regret_threshold: float = Field(default=0.0050)  # 50 bps, GOVERNANCE.md §6.2
    demotion_ece_threshold: float = Field(default=0.15)  # GOVERNANCE.md §6.2


class FamilyResult(BaseModel):
    """One signal family's in-window performance from forward-ledger or walk-forward backtest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    signal_family: str
    protected: bool
    n: int
    n_effective: float
    reliable: bool
    evidence_source: str = "forward_ledger"  # "forward_ledger" or "walkforward_backtest"
    mean_return: float | None = None
    sharpe: float | None = None
    psr: float | None = None
    dsr: float | None = None
    z_score: float | None = None
    p_value: float | None = None
    bh_rejected: bool | None = None
    brier: float | None = None
    ece: float | None = None
    mean_regret: float | None = None
    notes: list[str] = Field(default_factory=list)


class PromotionSuggestion(BaseModel):
    """Ladder Rule (Master Spec §27.3) evaluation for one challenger model.

    Never applied automatically -- ``passes_ladder=True`` is a
    recommendation; an operator (or a future CLI step) still has to call
    ``promote_if_ladder`` explicitly. ``trial_id`` is the challenger's
    *existing* registry trial id (Master Spec §27.1: every model change
    already carries one from registration) -- this report never mints a new
    trial merely by running.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    challenger_model_id: str
    challenger_signal_family: str
    champion_model_id: str | None
    trial_id: str | None
    passes_ladder: bool
    reasons: list[str]
    ev_improvement: float | None
    winrate_improvement: float | None


class DemotionSuggestion(BaseModel):
    """A live (champion/challenger) model tripping a GOVERNANCE.md §6.2
    demotion trigger. CLAUDE.md rule 32: this is a recommendation to reduce
    weight, never an automatic deletion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    signal_family: str
    reasons: list[str]


class WeightPreviewEntry(BaseModel):
    """Preview of what ``ModelRegistry.update_weights`` *would* produce this
    round, using each family's in-window mean return as utility -- computed
    with the pure ``learning.ensemble_weights.update_weights`` function, the
    registry is never actually written to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    signal_family: str
    current_weight: float
    previewed_weight: float
    delta: float


class TournamentReport(BaseModel):
    """A complete weekly research tournament report (Master Spec §36)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime
    generated_at: datetime
    window_start: date
    window_end: date
    fdr_alpha: float
    champion_model_id: str | None
    families: list[FamilyResult]
    promotions: list[PromotionSuggestion]
    demotions: list[DemotionSuggestion]
    weight_preview: list[WeightPreviewEntry]
    notes: list[str]


def _win_rate(pairs: list[tuple[LedgerEntry, LedgerLabel]]) -> float | None:
    if not pairs:
        return None
    return float(
        np.mean(
            [
                1.0
                if lbl.realized_selected_pnl is not None and lbl.realized_selected_pnl > 0
                else 0.0
                for _e, lbl in pairs
            ]
        )
    )


def run_research_tournament(
    store: Store,
    *,
    as_of: datetime,
    cfg: WeeklyTournamentConfig | None = None,
) -> TournamentReport:
    """Compare every registered signal family's forward-ledger performance
    over the identical trailing ``cfg.lookback_days`` window ending at
    ``as_of``, and produce promotion/demotion suggestions plus an
    ensemble-weight preview (Master Spec §36). Read-only: never mutates the
    store or the model registry. Deterministic given the same store
    contents (CLAUDE.md rule 16 -- no randomness is used here).
    """
    cfg = cfg if cfg is not None else WeeklyTournamentConfig()
    generated_at = datetime.now(UTC)
    window_end = as_of.date()
    window_start = window_end - timedelta(days=cfg.lookback_days)

    ledger = ForwardLedger(store)
    all_pairs = ledger.entries()
    matured = [
        (e, lbl)
        for e, lbl in all_pairs
        if e.status == LedgerEntryStatus.LABELED
        and lbl is not None
        and window_start <= e.exit_due <= window_end
    ]

    registry_entries = store.list_model_registry_entries()
    hash_map = registry_hash_map(registry_entries)
    weight_map = average_uniqueness_weights(matured)

    by_family: dict[str, list[tuple[LedgerEntry, LedgerLabel]]] = defaultdict(list)
    for e, lbl in matured:
        by_family[signal_family_for(e, hash_map)].append((e, lbl))

    families_present = (
        {m.signal_family for m in registry_entries} | set(by_family) | {cfg.protected_signal_family}
    )

    results: dict[str, FamilyResult] = {}
    p_values: dict[str, float] = {}
    # Fetch walk-forward results for fallback when forward-ledger is empty
    wf_results = store.latest_walkforward_results()
    wf_by_family: dict[str, dict[int, Any]] = defaultdict(dict)
    for wf_rec in wf_results:
        wf_by_family[wf_rec.signal_family][wf_rec.horizon_days] = wf_rec

    for family in sorted(families_present):
        pairs = by_family.get(family, [])
        n = len(pairs)
        n_eff = float(sum(weight_map.get(e.entry_id, 1.0) for e, _l in pairs))
        reliable = n >= cfg.min_trades_for_comparison
        protected = family == cfg.protected_signal_family
        notes: list[str] = []
        evidence_source = "forward_ledger"

        mean_return: float | None = None
        brier_val: float | None = None
        ece_val: float | None = None
        mean_regret: float | None = None
        sharpe_val: float | None = None
        psr_val: float | None = None
        dsr_val: float | None = None
        z_score: float | None = None
        p_value: float | None = None

        if n > 0:
            # Prefer forward-ledger data when available
            returns = np.array([lbl.realized_selected_pnl for _e, lbl in pairs], dtype=np.float64)
            mean_return = float(np.mean(returns))
            p_arr = np.array([e.p_profit for e, _l in pairs], dtype=np.float64)
            y_arr = np.array(
                [
                    1.0
                    if lbl.realized_selected_pnl is not None and lbl.realized_selected_pnl > 0
                    else 0.0
                    for _e, lbl in pairs
                ],
                dtype=np.float64,
            )
            brier_val = brier_score(p_arr, y_arr)
            ece_val = expected_calibration_error(p_arr, y_arr)
            regrets = [
                lbl.best_turbo_pnl - lbl.realized_selected_pnl
                for _e, lbl in pairs
                if lbl.best_turbo_pnl is not None and lbl.realized_selected_pnl is not None
            ]
            mean_regret = statistics.mean(regrets) if regrets else None

        if reliable and n >= 3:
            std = float(np.std(returns, ddof=1))
            if std > 0.0:
                t_stat = (
                    float(mean_return / (std / math.sqrt(n))) if mean_return is not None else 0.0
                )
                p_value = float(1.0 - stats.t.cdf(t_stat, df=n - 1))
                p_value = min(max(p_value, 1e-12), 1.0 - 1e-12)
                z_score = float(stats.norm.ppf(1.0 - p_value))
            else:
                p_value = 0.5
                z_score = 0.0
            mean_holding = max(1.0, statistics.mean(holding_days(e, lbl) for e, lbl in pairs))
            periods_per_year = _TRADING_DAYS_PER_YEAR / mean_holding
            sharpe_val = sharpe(returns, periods_per_year=periods_per_year)
            psr_val = probabilistic_sharpe_ratio(returns)
            dsr_val = deflated_sharpe_ratio(returns, max(1, len(families_present)))
            if not protected:
                p_values[family] = p_value
        elif n == 0 and family in wf_by_family:
            # Fallback: use walk-forward results if no forward-ledger data
            wf_best = max(wf_by_family[family].values(), key=lambda x: x.evaluated_at)
            evidence_source = "walkforward_backtest"
            mean_return = wf_best.mean_oos_return
            sharpe_val = None  # Walk-forward doesn't provide Sharpe directly
            psr_val = wf_best.psr
            dsr_val = None  # DSR not available from backtest
            z_score = None  # Can't compute z-score without returns array
            p_value = None  # Can't compute p-value from backtest alone
            brier_val = wf_best.brier
            ece_val = wf_best.ece
            mean_regret = None  # Not available from backtest
            reliable = False  # Backtest data is less reliable than forward
            n = wf_best.n_folds  # Number of folds as proxy for sample size
            n_eff = wf_best.n_effective
            # Mark in notes that this is backtest data
            notes.append(
                f"Keine Forward-Ledger-Daten; Walk-Forward-Backtest-Ergebnisse verwendet "
                f"(evaluiert {wf_best.evaluated_at.date()}, {wf_best.n_folds} Folds). "
                "keine automatische Promotion basierend auf Backtest-Evidenz (Spec §29)."
            )
        else:
            if n < cfg.min_trades_for_comparison and n > 0:
                notes.append(
                    f"n={n} < min_trades_for_comparison={cfg.min_trades_for_comparison}: kein "
                    "z-/p-Wert, keine BH-Korrektur, kein Sharpe/PSR/DSR."
                )
            elif n == 0 and family not in wf_by_family:
                notes.append("Keine Forward-Ledger- oder Walk-Forward-Backtest-Daten.")

        results[family] = FamilyResult(
            signal_family=family,
            protected=protected,
            n=n,
            n_effective=n_eff,
            reliable=reliable,
            evidence_source=evidence_source,
            mean_return=mean_return,
            sharpe=sharpe_val,
            psr=psr_val,
            dsr=dsr_val,
            z_score=z_score,
            p_value=p_value,
            bh_rejected=None,
            brier=brier_val,
            ece=ece_val,
            mean_regret=mean_regret,
            notes=notes,
        )

    tested_families = sorted(p_values)
    if tested_families:
        pvals_arr = np.array([p_values[f] for f in tested_families], dtype=np.float64)
        rejected = benjamini_hochberg(pvals_arr, alpha=cfg.fdr_alpha)
        for family, rej in zip(tested_families, rejected, strict=True):
            results[family] = results[family].model_copy(update={"bh_rejected": bool(rej)})

    champion_entry = next((m for m in registry_entries if m.status == ModelStatus.CHAMPION), None)
    champion_model_id = champion_entry.model_id if champion_entry is not None else None
    champion_result = (
        results.get(champion_entry.signal_family) if champion_entry is not None else None
    )
    champion_pairs = (
        by_family.get(champion_entry.signal_family, []) if champion_entry is not None else []
    )

    promotions: list[PromotionSuggestion] = []
    for m in registry_entries:
        if m.status is not ModelStatus.CHALLENGER:
            continue
        fam_result = results.get(m.signal_family)
        if fam_result is None:
            continue
        challenger_pairs = by_family.get(m.signal_family, [])

        ev_improvement: float | None = None
        if (
            fam_result.mean_return is not None
            and champion_result is not None
            and champion_result.mean_return is not None
        ):
            ev_improvement = fam_result.mean_return - champion_result.mean_return

        winrate_improvement: float | None = None
        wr_challenger = _win_rate(challenger_pairs)
        wr_champion = _win_rate(champion_pairs)
        if wr_challenger is not None and wr_champion is not None:
            winrate_improvement = wr_challenger - wr_champion

        # Never promote based on backtest data alone (Spec §29)
        backtest_only = fam_result.evidence_source == "walkforward_backtest"
        checks = {
            "reliable_sample": fam_result.reliable,
            "bh_rejected": bool(fam_result.bh_rejected),
            "z_score>=threshold": fam_result.z_score is not None
            and fam_result.z_score >= cfg.ladder_min_zscore,
            "dsr>=threshold": fam_result.dsr is not None and fam_result.dsr >= cfg.ladder_min_dsr,
            "psr>=threshold": fam_result.psr is not None and fam_result.psr >= cfg.ladder_min_psr,
            "ev_improvement>=threshold": ev_improvement is not None
            and ev_improvement >= cfg.ladder_min_ev_improvement,
            "winrate_improvement>=threshold": (
                winrate_improvement is not None
                and winrate_improvement >= cfg.ladder_min_winrate_improvement
            ),
        }
        if backtest_only:
            checks["backtest_only"] = False
        reasons = [f"{name}: {'passed' if passed else 'failed'}" for name, passed in checks.items()]
        passes_ladder = all(checks.values())
        promotions.append(
            PromotionSuggestion(
                challenger_model_id=m.model_id,
                challenger_signal_family=m.signal_family,
                champion_model_id=champion_model_id,
                trial_id=m.trial_id,
                passes_ladder=passes_ladder,
                reasons=reasons,
                ev_improvement=ev_improvement,
                winrate_improvement=winrate_improvement,
            )
        )

    drift_in_window = [
        d for d in store.list_drift_events() if window_start <= d.detected_at.date() <= window_end
    ]
    drift_families = {d.signal_family for d in drift_in_window if d.signal_family is not None}

    demotions: list[DemotionSuggestion] = []
    for m in registry_entries:
        if m.status not in (ModelStatus.CHAMPION, ModelStatus.CHALLENGER):
            continue
        fam_result = results.get(m.signal_family)
        if fam_result is None:
            continue
        demotion_reasons: list[str] = []
        if fam_result.sharpe is not None and fam_result.sharpe < cfg.demotion_sharpe_threshold:
            demotion_reasons.append(
                f"Sharpe {fam_result.sharpe:.3f} < {cfg.demotion_sharpe_threshold:.3f} "
                "im Beobachtungsfenster (GOVERNANCE.md §6.2)"
            )
        if (
            fam_result.mean_regret is not None
            and fam_result.mean_regret > cfg.demotion_regret_threshold
        ):
            demotion_reasons.append(
                f"mittleres Regret {fam_result.mean_regret:.4f} > "
                f"{cfg.demotion_regret_threshold:.4f} (GOVERNANCE.md §6.2)"
            )
        if fam_result.ece is not None and fam_result.ece > cfg.demotion_ece_threshold:
            demotion_reasons.append(
                f"ECE {fam_result.ece:.4f} > {cfg.demotion_ece_threshold:.4f} (GOVERNANCE.md §6.2)"
            )
        if m.signal_family in drift_families:
            demotion_reasons.append(
                "Drift-Ereignis (Page-Hinkley) im Beobachtungsfenster erkannt (Master Spec §30)"
            )
        if demotion_reasons:
            demotions.append(
                DemotionSuggestion(
                    model_id=m.model_id, signal_family=m.signal_family, reasons=demotion_reasons
                )
            )

    notes = [
        "Vergleich nutzt Forward-Ledger-Daten, wo verfügbar; Walk-Forward-Backtest-Ergebnisse "
        "als Fallback (nicht im Store persistiert), wenn Forward-Ledger leer (Spec §29). "
        f"Fenster: {window_start.isoformat()}..{window_end.isoformat()}.",
    ]
    if champion_model_id is None:
        notes.append("Kein Modell mit status=champion im Registry gefunden.")

    current_weights = {m.model_id: m.weight for m in registry_entries}
    weight_preview: list[WeightPreviewEntry] = []
    if current_weights:
        if all(w > 0.0 for w in current_weights.values()):
            utilities = {
                m.model_id: (results[m.signal_family].mean_return or 0.0)
                if m.signal_family in results
                else 0.0
                for m in registry_entries
            }
            try:
                previewed = _compute_weights_preview(
                    current_weights, utilities, eta=cfg.reweight_eta, w_min=cfg.reweight_w_min
                )
            except ValueError as exc:
                notes.append(f"Gewichts-Vorschau nicht berechenbar: {exc}")
                previewed = dict(current_weights)
        else:
            notes.append(
                "Gewichts-Vorschau übersprungen: mindestens ein registriertes Modell hat "
                "weight<=0 (ensemble_weights.update_weights erfordert weight>0)."
            )
            previewed = dict(current_weights)
        weight_preview = [
            WeightPreviewEntry(
                model_id=m.model_id,
                signal_family=m.signal_family,
                current_weight=m.weight,
                previewed_weight=previewed.get(m.model_id, m.weight),
                delta=previewed.get(m.model_id, m.weight) - m.weight,
            )
            for m in registry_entries
        ]

    return TournamentReport(
        as_of=as_of,
        generated_at=generated_at,
        window_start=window_start,
        window_end=window_end,
        fdr_alpha=cfg.fdr_alpha,
        champion_model_id=champion_model_id,
        families=[results[f] for f in sorted(results)],
        promotions=promotions,
        demotions=demotions,
        weight_preview=weight_preview,
        notes=notes,
    )


__all__ = [
    "DemotionSuggestion",
    "FamilyResult",
    "PromotionSuggestion",
    "TournamentReport",
    "WeeklyTournamentConfig",
    "WeightPreviewEntry",
    "run_research_tournament",
]
