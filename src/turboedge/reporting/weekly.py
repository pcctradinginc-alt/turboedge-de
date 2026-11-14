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
    family_average_uniqueness_weights,
    registry_hash_map,
    signal_family_for,
    weighted_mean_return,
)
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Category,
    ExitReason,
    LedgerEntry,
    LedgerEntryStatus,
    LedgerLabel,
    ModelStatus,
)

#: Below this effective sample, a family's PSR/DSR are withheld entirely
#: (set to ``None``, never computed and shown as if meaningful) -- the same
#: "no comparison number below the floor" discipline as
#: ``backtest/excursion_eval.py``'s ``MIN_EFFECTIVE_SAMPLE``/``INSUFFICIENT_SAMPLE``.
#:
#: GOVERNANCE.md §6.1 ("Promotion (Experimental -> Live)") already lists
#: "Effective sample >= 100 distinct outcomes" as a promotion criterion --
#: this reuses that exact, already-governance-approved number rather than
#: inventing a second threshold, and applies it one step earlier: PSR/DSR
#: computed on fewer effective observations than promotion itself requires
#: are not just insufficient to promote on, they are not a measurement at
#: all (a probability statement computed from ~1 independent draw, as the
#: 2,672-row/n_eff=1.00 case in docs/measured_results.md §6.9 is, has no
#: content). This is also exactly the mechanism that produced the
#: 2026-09-26 false-positive tournament report: n=68 same-day rows (from one
#: scan run) reported Sharpe 9.6-10.5 and "bh_rejected=True" only because
#: their row count, not their effective sample (~1), was fed to PSR/DSR/the
#: t-test.
_MIN_EFFECTIVE_SAMPLE_FOR_SIGNIFICANCE = 100.0


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


class WalkforwardCell(BaseModel):
    """One historical diagnostic cell; it is NOT a turbo P&L observation.

    A DAX/5d cell must not be overwritten by an NDX/5d or DAX/10d cell.
    Neither pooling folds nor selecting the latest timestamp provides an
    unbiased strategy-level net-EV estimate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    underlying_id: str
    horizon_days: int
    n_folds: int
    n_effective: float
    mean_gross_direction_return: float
    brier: float
    brier_null: float | None
    psr: float | None
    evaluated_at: datetime


class FamilyResult(BaseModel):
    """One signal family's in-window performance from forward-ledger or walk-forward backtest.

    ``n`` is a row count; ``n_effective`` is the average-uniqueness-weighted
    effective sample (``family_average_uniqueness_weights``,
    ``backtest.purged_cv.average_uniqueness``) -- the two are shown side by
    side deliberately (docs/measured_results.md §6.9/§6.11): a family whose
    2,672 rows share one label window has ``n=2672`` and ``n_effective=1.00``,
    and reading the first number as the sample size is exactly the
    2026-09-26 false-positive tournament bug. ``mean_return`` is the
    ``n_effective``-weighted mean (see ``weighted_mean_return``), not the
    plain row mean, for the same reason. ``sharpe`` is the *per-trade*,
    **not annualized** Sharpe ratio (see ``run_research_tournament`` for
    why). ``psr``/``dsr`` are computed from ``n_effective`` rather than
    ``n`` (average uniqueness over overlapping label windows) and are ``None`` --
    never a number -- whenever ``n_effective`` is below
    ``_MIN_EFFECTIVE_SAMPLE_FOR_SIGNIFICANCE``; ``notes`` says why in that
    case.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    signal_family: str
    protected: bool
    n: int
    n_effective: float
    research_only_n: int = 0
    reliable: bool
    evidence_source: str = "forward_ledger"  # "forward_ledger" or "walkforward_backtest"
    mean_return: float | None = None
    sharpe: float | None = None  # per-trade, NOT annualized -- see run_research_tournament
    psr: float | None = None
    dsr: float | None = None
    z_score: float | None = None
    p_value: float | None = None
    bh_rejected: bool | None = None
    brier: float | None = None
    ece: float | None = None
    mean_regret: float | None = None
    notes: list[str] = Field(default_factory=list)
    walkforward_cells: list[WalkforwardCell] = Field(default_factory=list)


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
    matched_effective_sample: float | None = None


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


def _matched_economic_edge(
    challenger: list[tuple[LedgerEntry, LedgerLabel]],
    champion: list[tuple[LedgerEntry, LedgerLabel]],
) -> tuple[float | None, float | None, float]:
    """Compare only identical entry events, never unrelated model populations.

    Multiple products for the same prediction/event are one observation:
    first average product returns within each event, then compare the two
    models' event means. Average uniqueness is applied across matched
    event label windows before computing any decision-relevant improvement.
    This is research evidence only; confirmed fill/cost evidence is separate.
    """
    from turboedge.backtest.purged_cv import average_uniqueness

    def grouped(
        pairs: list[tuple[LedgerEntry, LedgerLabel]],
    ) -> dict[tuple[str, str, int, datetime, date], tuple[float, int]]:
        data: dict[tuple[str, str, int, datetime, date], list[float]] = defaultdict(list)
        for entry, label in pairs:
            if label.realized_selected_pnl is None:
                continue
            key = (
                entry.underlying,
                entry.direction.value,
                entry.horizon_days,
                entry.prediction_time,
                entry.exit_due,
            )
            data[key].append(label.realized_selected_pnl)
        return {key: (float(np.mean(vals)), len(vals)) for key, vals in data.items()}

    challenger_events = grouped(challenger)
    champion_events = grouped(champion)
    shared = sorted(set(challenger_events) & set(champion_events))
    if not shared:
        return None, None, 0.0
    start = min(key[3].date() for key in shared)
    windows = [((key[3].date() - start).days, (key[4] - start).days) for key in shared]
    t0 = np.asarray([w[0] for w in windows], dtype=np.int64)
    t1 = np.asarray([w[1] for w in windows], dtype=np.int64)
    weights = average_uniqueness(t0, t1)
    diff = np.asarray(
        [challenger_events[key][0] - champion_events[key][0] for key in shared],
        dtype=np.float64,
    )
    win_diff = np.asarray(
        [
            float(challenger_events[key][0] > 0.0) - float(champion_events[key][0] > 0.0)
            for key in shared
        ],
        dtype=np.float64,
    )
    return (
        float(np.average(diff, weights=weights)),
        float(np.average(win_diff, weights=weights)),
        float(np.sum(weights)),
    )


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
    matured_all = [
        (e, lbl)
        for e, lbl in all_pairs
        if e.status == LedgerEntryStatus.LABELED
        and lbl is not None
        and window_start <= e.exit_due <= window_end
    ]
    # A discounted last-known quote is deliberately conservative for
    # position management, but is NOT a measured executable exit. Keep
    # ambiguous/missing exits out of significance and economic comparisons.
    matured = [
        (e, lbl)
        for e, lbl in matured_all
        if lbl is not None
        and lbl.realized_selected_pnl is not None
        and not lbl.ambiguous_path
        and lbl.exit_reason in (ExitReason.HORIZON, ExitReason.KO)
    ]
    excluded_exit_count = len(matured_all) - len(matured)

    registry_entries = store.list_model_registry_entries()
    hash_map = registry_hash_map(registry_entries)

    by_family: dict[str, list[tuple[LedgerEntry, LedgerLabel]]] = defaultdict(list)
    research_only_counts: dict[str, int] = defaultdict(int)
    for e, lbl in matured:
        family = signal_family_for(e, hash_map)
        if e.category == Category.ACTIONABLE and not e.is_shadow:
            by_family[family].append((e, lbl))
        else:
            # WATCH/REJECT/stratified shadows are counterfactual research,
            # not a portfolio that was ever proposed for trading. Never use
            # these observations to claim a strategy return or a promotion.
            research_only_counts[family] += 1

    families_present = (
        {m.signal_family for m in registry_entries}
        | set(by_family)
        | set(research_only_counts)
        | {cfg.protected_signal_family}
    )

    results: dict[str, FamilyResult] = {}
    p_values: dict[str, float] = {}
    # Fetch walk-forward results for fallback when forward-ledger is empty
    wf_results = store.latest_walkforward_results()
    wf_by_family: dict[str, list[Any]] = defaultdict(list)
    for wf_rec in wf_results:
        wf_by_family[wf_rec.signal_family].append(wf_rec)

    for family in sorted(families_present):
        pairs = by_family.get(family, [])
        wf_cells = [
            WalkforwardCell(
                model_id=wf.model_id,
                underlying_id=wf.underlying_id,
                horizon_days=wf.horizon_days,
                n_folds=wf.n_folds,
                n_effective=wf.n_effective,
                mean_gross_direction_return=wf.mean_oos_return,
                brier=wf.brier,
                brier_null=wf.brier_null,
                psr=wf.psr if wf.n_effective >= _MIN_EFFECTIVE_SAMPLE_FOR_SIGNIFICANCE else None,
                evaluated_at=wf.evaluated_at,
            )
            for wf in sorted(
                wf_by_family.get(family, []),
                key=lambda w: (w.model_id, w.underlying_id, w.horizon_days),
            )
        ]
        n = len(pairs)
        research_only_n = research_only_counts.get(family, 0)
        # Per-family, NOT pooled across families (family_average_uniqueness_weights'
        # docstring explains why that distinction matters): family A's own
        # significance must not be diluted by family B's trades on the same
        # underlying/day.
        family_weights = family_average_uniqueness_weights(pairs)
        n_eff = float(sum(family_weights.values()))
        reliable = (
            n >= cfg.min_trades_for_comparison and n_eff >= _MIN_EFFECTIVE_SAMPLE_FOR_SIGNIFICANCE
        )
        protected = family == cfg.protected_signal_family
        notes: list[str] = []
        if research_only_n:
            notes.append(
                f"{research_only_n} WATCH/REJECT/Shadow-Exits nur Forschungsbeobachtung, "
                "nicht als Strategie-Trade/Netto-Alpha gewertet."
            )
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
            # Uniqueness-weighted, not the plain row mean: 2,672 rows sharing
            # one label window must not move the reported mean 2,672x as much
            # as one independent observation would (docs/measured_results.md
            # §6.9/§6.11; see weighted_mean_return's docstring).
            mean_return = weighted_mean_return(pairs, family_weights)
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

        if n >= cfg.min_trades_for_comparison and n >= 3:
            std = float(np.std(returns, ddof=1))
            if (
                std > 0.0
                and n_eff >= _MIN_EFFECTIVE_SAMPLE_FOR_SIGNIFICANCE
                and mean_return is not None
            ):
                # Standard error and degrees of freedom use n_eff, not n: a
                # t-test's validity comes from the number of *independent*
                # draws, and row count overstates that whenever labels
                # overlap (docs/measured_results.md §6.9/§6.11). scipy's t
                # distribution takes a continuous df, so no rounding is
                # needed here (contrast the PSR/DSR path below,
                # which must round because an array's length is an int).
                t_stat = float(mean_return / (std / math.sqrt(n_eff)))
                p_value = float(1.0 - stats.t.cdf(t_stat, df=n_eff - 1.0))
                p_value = min(max(p_value, 1e-12), 1.0 - 1e-12)
                z_score = float(stats.norm.ppf(1.0 - p_value))
            elif n_eff < _MIN_EFFECTIVE_SAMPLE_FOR_SIGNIFICANCE:
                notes.append(
                    f"n_effective={n_eff:.2f} < 100: kein z-/p-Wert, keine "
                    "BH-Korrektur und keine belastbare Inferenz trotz "
                    f"n={n} Rohzeilen; 100 unabh. Outcomes erforderlich."
                )
            else:
                notes.append("Renditekonstanz/fehlende Streuung: kein valider Inferenztest.")

            if mean_return is not None:
                # Sharpe: per-trade, deliberately NOT annualized (Master
                # Spec's "Sharpe" here always meant this ratio, never a
                # calendar-year one). The tournament's positions are held
                # 3-14 days and overlap heavily (the whole reason n_eff
                # exists); scaling by sqrt(any assumed periods-per-year) --
                # metrics.sharpe's default of sqrt(252), or a per-family
                # sqrt(252/mean_holding_days) -- manufactures a large
                # annualized number from a modest per-trade effect precisely
                # because it pretends these trades tile a calendar year
                # independently, which the effective-sample computation
                # above has just shown they do not (docs/measured_results.md
                # §6.9/§6.11: "a per-trade Sharpe of 0.66 is reported as
                # 10.49"). Reporting the unannualized per-trade ratio avoids
                # inventing a trading-frequency assumption this repository
                # has never measured. The array is re-centered on the
                # weighted mean_return (a location shift -- std/skew/
                # kurtosis are unaffected) so Sharpe's implied mean matches
                # the mean_return actually reported alongside it.
                returns_for_mean = returns - float(np.mean(returns)) + mean_return
                sharpe_val = sharpe(returns_for_mean, periods_per_year=1.0)

                # PSR/DSR read the effective sample as T directly. This
                # previously went through a length-`round(n_eff)` surrogate
                # array built by quantile interpolation, because these
                # functions took T only from `returns.shape[0]`; they now
                # accept `n_effective`. The surrogate was conservative
                # (measured: +5.4% dispersion at n_eff=100, +105% at n_eff=5,
                # which lowers the Sharpe and so the PSR) but it distorted
                # exactly the dispersion the skew/kurtosis correction reads,
                # and an approximation has no place inside a significance
                # calculation when an exact parameter will do.
                # Guarded by the sample floor rather than computed and then
                # discarded: PSR raises below three observations, and at
                # n_eff < 100 the answer is withheld anyway (see the notes
                # block further down). Computing it first only to throw it
                # away would crash on exactly the same-day families this
                # whole change exists to catch, where n_eff is 1.00.
                if n_eff >= _MIN_EFFECTIVE_SAMPLE_FOR_SIGNIFICANCE:
                    psr_val = probabilistic_sharpe_ratio(returns_for_mean, n_effective=n_eff)
                    tested_hypotheses = max(
                        1,
                        len(families_present),
                        len(store.list_research_trials()),
                        len(store.list_research_screens()),
                    )
                    dsr_val = deflated_sharpe_ratio(
                        returns_for_mean, tested_hypotheses, n_effective=n_eff
                    )

            if (
                not protected
                and p_value is not None
                and any(m.signal_family == family and m.trial_id for m in registry_entries)
            ):
                p_values[family] = p_value
        elif n == 0 and len(wf_cells) == 1:
            # Historical single-cell diagnostic only: not a 60-day net-turbo
            # outcome, not comparable against another family's forward ledger.
            wf_best = wf_by_family[family][0]
            evidence_source = "walkforward_backtest"
            mean_return = wf_best.mean_oos_return
            sharpe_val = None  # Walk-forward doesn't provide Sharpe directly
            psr_val = None  # historical raw PSR is not evidence of a turbo net edge
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
                "Rendite ist brutto Richtungsrendite des Basiswerts, KEIN Netto-Turbo-EV; "
                "keine automatische Promotion basierend auf Backtest-Evidenz (Spec §29)."
            )
        elif n == 0 and len(wf_cells) > 1:
            evidence_source = "walkforward_backtest"
            notes.append(
                f"{len(wf_cells)} historische OOS-Zellen getrennt nach Modell/Basiswert/Horizont; "
                "kein Cherry-Picking der neuesten Zelle, keine vermischte Mean/PSR. "
                "Historische Basiswert-Richtungsrenditen sind kein Netto-Turbo-EV."
            )
        else:
            if n < cfg.min_trades_for_comparison and n > 0:
                notes.append(
                    f"n={n} < min_trades_for_comparison={cfg.min_trades_for_comparison}: kein "
                    "z-/p-Wert, keine BH-Korrektur, kein Sharpe/PSR/DSR."
                )
            elif n == 0 and family not in wf_by_family:
                notes.append("Keine Forward-Ledger- oder Walk-Forward-Backtest-Daten.")

        # Sample floor for PSR/DSR specifically, applied uniformly regardless
        # of evidence_source (forward-ledger computed above, or a stored
        # walk-forward n_effective): "the family's PSR/DSR must not be
        # presented as numbers at all" below the floor, the same way
        # backtest/excursion_eval.py returns INSUFFICIENT_SAMPLE instead of a
        # comparison figure -- a probability statement computed from an
        # effective sample under _MIN_EFFECTIVE_SAMPLE_FOR_SIGNIFICANCE is
        # not a weak result, it is not a result (docs/measured_results.md
        # §6.9).
        # The note is emitted whenever the floor bites, not only when there
        # was a value to discard. The forward-ledger branch now declines to
        # compute PSR/DSR below the floor in the first place, and a silent
        # `None` would read as "not applicable" rather than "withheld, and
        # here is the number that withheld it".
        has_evidence = n > 0 or family in wf_by_family
        if n_eff < _MIN_EFFECTIVE_SAMPLE_FOR_SIGNIFICANCE and has_evidence:
            notes.append(
                f"n_effective={n_eff:.2f} < {_MIN_EFFECTIVE_SAMPLE_FOR_SIGNIFICANCE:.0f}: "
                "PSR/DSR nicht ausgewiesen (GOVERNANCE.md §6.1 Promotion-Kriterium "
                "'Effective sample >= 100 distinct outcomes' -- angewendet, bevor statt "
                "erst bei der Promotion selbst; docs/measured_results.md §6.9/§6.11)."
            )
            psr_val = None
            dsr_val = None

        results[family] = FamilyResult(
            signal_family=family,
            protected=protected,
            n=n,
            n_effective=n_eff,
            research_only_n=research_only_n,
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
            walkforward_cells=wf_cells,
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
        winrate_improvement: float | None = None
        paired_effective_sample: float | None = None
        if (
            fam_result.evidence_source == "forward_ledger"
            and champion_result is not None
            and champion_result.evidence_source == "forward_ledger"
        ):
            ev_improvement, winrate_improvement, paired_effective_sample = _matched_economic_edge(
                challenger_pairs, champion_pairs
            )
        elif champion_entry is None and fam_result.evidence_source == "forward_ledger":
            # First champion's reference is explicit NO_TRADE (return 0),
            # not a fictitious reference with a made-up hit rate.
            ev_improvement = fam_result.mean_return
            paired_effective_sample = fam_result.n_effective

        # Never promote based on backtest data alone (Spec §29)
        backtest_only = fam_result.evidence_source == "walkforward_backtest"
        valid_entry_quotes = bool(challenger_pairs) and all(
            0.0 <= (entry.prediction_time - entry.entry_quote_timestamp).total_seconds() <= 300.0
            and entry.entry_ask > 0.0
            and (entry.entry_bid is None or entry.entry_bid <= entry.entry_ask)
            for entry, _ in challenger_pairs
        )
        checks = {
            "quote_freshness_and_bid_ask": valid_entry_quotes,
            "reliable_sample": fam_result.reliable
            and fam_result.n_effective >= _MIN_EFFECTIVE_SAMPLE_FOR_SIGNIFICANCE,
            "registered_trial_id": m.trial_id is not None,
            "matched_effective_sample>=100": (
                paired_effective_sample is not None
                and paired_effective_sample >= _MIN_EFFECTIVE_SAMPLE_FOR_SIGNIFICANCE
            ),
            "independent_confirmation_and_risk_gates": False,  # not shown by weekly ledger
            "bh_rejected": bool(fam_result.bh_rejected),
            "z_score>=threshold": fam_result.z_score is not None
            and fam_result.z_score >= cfg.ladder_min_zscore,
            "dsr>=threshold": fam_result.dsr is not None and fam_result.dsr >= cfg.ladder_min_dsr,
            "psr>=threshold": fam_result.psr is not None and fam_result.psr >= cfg.ladder_min_psr,
            "ev_improvement>=threshold": ev_improvement is not None
            and ev_improvement >= cfg.ladder_min_ev_improvement,
            "winrate_improvement>=threshold": (
                champion_entry is None  # first champion: no artificial champion winrate
                or (
                    winrate_improvement is not None
                    and winrate_improvement >= cfg.ladder_min_winrate_improvement
                )
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
                matched_effective_sample=paired_effective_sample,
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
        if (
            fam_result.reliable
            and fam_result.sharpe is not None
            and fam_result.sharpe < cfg.demotion_sharpe_threshold
        ):
            demotion_reasons.append(
                f"Sharpe {fam_result.sharpe:.3f} < {cfg.demotion_sharpe_threshold:.3f} "
                "im Beobachtungsfenster (GOVERNANCE.md §6.2)"
            )
        if (
            fam_result.reliable
            and fam_result.mean_regret is not None
            and fam_result.mean_regret > cfg.demotion_regret_threshold
        ):
            demotion_reasons.append(
                f"mittleres Regret {fam_result.mean_regret:.4f} > "
                f"{cfg.demotion_regret_threshold:.4f} (GOVERNANCE.md §6.2)"
            )
        if (
            fam_result.reliable
            and fam_result.ece is not None
            and fam_result.ece > cfg.demotion_ece_threshold
        ):
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
        "Historische Walk-Forward-Zellen sind nur Basiswert-Richtungsdiagnostik. "
        "Forward-Ledger besteht aus tatsächlichen Quote-/KO-Exit-Labels; "
        "keine gemeinsame Nettorendite oder gemeinsame Signifikanz zwischen beiden Datenarten. "
        f"Fenster: {window_start.isoformat()}..{window_end.isoformat()}.",
    ]
    notes.append(
        f"Exit-Qualitaet: {len(matured)} bestaetigte Quote-/KO-Exits im Fenster; "
        f"{sum(map(len, by_family.values()))} ACTIONABLE-Nicht-Shadow Outcomes; "
        f"{sum(research_only_counts.values())} WATCH/REJECT/Shadow-Outcomes nur Forschung; "
        f"{excluded_exit_count} fehlende/ambigue Exits aus der Performance-Messung ausgeschlossen."
    )
    notes.append(
        "Produktkosten: Forward-Rendite verwendet Einstiegs-Ask und beobachteten "
        "Ausstiegs-Bid oder KO, aber keine nachgewiesenen echten Fills/Steuern/Gebuehren; "
        "keine Trading-Freigabe. OOS-Brutto-Basiswert-Return ist keine Turbo-Rendite."
    )
    quarter = f"{as_of.year}Q{(as_of.month - 1) // 3 + 1}"
    quarter_trials = store.list_research_trials(quarter=quarter)
    exploratory_trials = sum(t.kind == "exploratory_screen" for t in quarter_trials)
    confirmatory_trials = sum(t.kind != "exploratory_screen" for t in quarter_trials)
    research_screens = store.list_research_screens(quarter=quarter)
    notes.append(
        f"Research-Budget {quarter}: {exploratory_trials} registrierte explorative "
        f"Trials, {len(research_screens)} ausgefuehrte Screens (Limit 12), "
        f"{confirmatory_trials} nicht-explorative Trials (Limit 6). "
        "Keine automatische Freigabe; bestaetigende Evidenz separat erforderlich."
    )
    if len(research_screens) > 12 or confirmatory_trials > 6:
        notes.append("GOVERNANCE_BUDGET_EXCEEDED: weiterer Trial/Freigabe blockieren.")
    if champion_model_id is None:
        notes.append("Kein Modell mit status=champion im Registry gefunden.")

    current_weights = {m.model_id: m.weight for m in registry_entries}
    weight_preview: list[WeightPreviewEntry] = []
    if current_weights:
        if all(w > 0.0 for w in current_weights.values()):
            utilities = {
                m.model_id: (results[m.signal_family].mean_return or 0.0)
                if (
                    m.signal_family in results
                    and results[m.signal_family].evidence_source == "forward_ledger"
                    and results[m.signal_family].n_effective
                    >= _MIN_EFFECTIVE_SAMPLE_FOR_SIGNIFICANCE
                )
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
    "WalkforwardCell",
    "WeeklyTournamentConfig",
    "WeightPreviewEntry",
    "run_research_tournament",
]
