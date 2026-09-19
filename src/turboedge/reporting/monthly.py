"""Monthly performance report (Master Spec §38 "Metrics", §24 "Counterfactual
Learning", §27.4 "Multiple Testing", §46 "Shadow Portfolio").

Builds a self-contained :class:`MonthlyReport` from everything the Forward
Ledger (``forward_ledger`` + ``ledger_labels``) and the learning tables
(``strategy_posteriors``, ``model_registry``, ``model_weight_history``,
``research_trials``, ``drift_events``, ``shadow_portfolio``) know about one
calendar month, entirely offline (no network, no mutation of the store).

Honesty is mandatory (Master Spec §27.4, §9.3; CLAUDE.md rules 25/26):
return-based metrics (success rate, mean/median return, profit factor,
Sharpe/Sortino, PSR/DSR, Expected Shortfall) are computed from *labeled*
forward-ledger trades only, every such block reports its raw ``n`` *and* an
average-uniqueness-adjusted effective sample size, a ``reliable`` flag, and
-- below ``cfg.min_trades_for_stats`` -- Sharpe/Sortino/PSR/DSR are withheld
entirely (never "a Sharpe ratio from 3 trades") rather than merely flagged.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from turboedge.backtest.metrics import (
    brier_score,
    expected_calibration_error,
    profit_factor,
    sharpe,
    sortino,
)
from turboedge.backtest.significance import (
    bootstrap_ci,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from turboedge.learning.ledger import ForwardLedger
from turboedge.learning.posterior import PosteriorConfig, StrategyPosterior
from turboedge.learning.trials import TrialsConfig, effective_number_of_trials
from turboedge.models.forecast import HORIZONS
from turboedge.pricing.instrument_compare import (
    InputValue,
    InstrumentComparisonResult,
    SimulationParams,
    compare_instruments,
    load_instruments_config,
)
from turboedge.reporting._common import (
    WilsonInterval,
    average_uniqueness_weights,
    expected_shortfall,
    financing_drag_pct,
    holding_days,
    last_day_of_month,
    leverage_bucket,
    quarter_label,
    registry_hash_map,
    signal_family_for,
    wilson_interval,
)
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Category,
    Direction,
    LedgerEntry,
    LedgerEntryStatus,
    LedgerLabel,
    ShadowPortfolioKind,
    TrialStatus,
)

_TRADING_DAYS_PER_YEAR = 252.0
_GROUP_DIMENSIONS: tuple[str, ...] = (
    "underlying",
    "direction",
    "horizon",
    "issuer",
    "leverage_bucket",
    "signal_family",
)


class MonthlyReportConfig(BaseModel):
    """This module's own config (wired into config.py/YAML by the
    integration wave)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Below this many labeled trades, return-based metrics are marked
    # unreliable and Sharpe/Sortino/PSR/DSR are withheld entirely (Master
    # Spec §9.3/§27.4: "keine Sharpe-Ratio aus 3 Trades").
    min_trades_for_stats: int = Field(default=10, ge=1)
    wilson_confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    ci_confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    n_bootstrap: int = Field(default=2000, ge=100)
    bootstrap_seed: int = Field(default=1234)
    es_alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    leverage_bucket_edges: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)
    posterior_config: PosteriorConfig = Field(default_factory=PosteriorConfig)
    trials_config: TrialsConfig = Field(default_factory=TrialsConfig)
    shadow_portfolio_kinds: tuple[ShadowPortfolioKind, ...] = (
        ShadowPortfolioKind.TOP1,
        ShadowPortfolioKind.TOP3,
        ShadowPortfolioKind.TOP5,
        ShadowPortfolioKind.RANDOM_VALID_TURBO,
        ShadowPortfolioKind.LOWEST_SPREAD,
        ShadowPortfolioKind.LOWEST_FINANCING_COST,
        ShadowPortfolioKind.HIGHEST_LEVERAGE,
        ShadowPortfolioKind.LOWEST_LEVERAGE,
        ShadowPortfolioKind.MEDIAN_PRODUCT,
    )

    # -- instrument-class cost comparison (Systemkonzept 1.2 §12.1/§23) ------
    # ``None`` (the default) skips this block entirely -- every existing
    # caller/test that builds a plain ``MonthlyReportConfig()`` is therefore
    # unaffected. Set to an underlying id (e.g. "DAX") to include it.
    instrument_compare_underlying: str | None = None
    instrument_compare_notional_eur: float = Field(default=24000.0, gt=0.0)
    instrument_compare_horizon_days: int = Field(default=15, gt=0)
    # Direction.value ("long"/"short") rather than the enum itself, so a
    # plain reporting.yaml "monthly: {...}" dict can set it without this
    # module requiring an enum-aware YAML loader.
    instrument_compare_direction: str = "long"
    # Where to find configs/instruments.yaml -- this report is built from
    # only a Store (no config_dir threaded through build_monthly_report's
    # existing signature/callers), so this defaults to the same relative
    # "configs" path AppContext.config_dir defaults to (cli.py) when the
    # process's cwd is the project root, as every documented invocation
    # (README "Development Commands") assumes.
    instrument_compare_config_dir: str = "configs"
    # This report is built entirely offline (module docstring: "no network,
    # no mutation of the store") -- unlike `turboedge compare instruments`
    # (cli.py), which fetches a live ECB rate, there is no persisted ECB
    # rate history to read here (adapters/ecb.py's rate is fetched
    # transiently at scan time and never stored). The reference rate is
    # therefore ALWAYS `assumed` in this report, from this fallback --
    # mirrors configs/risk.yaml's reference_rate_fallback default exactly
    # (duplicated, not imported, to keep this module free of a
    # turboedge.config dependency it does not otherwise have).
    instrument_compare_reference_rate_fallback: float = Field(default=0.03, ge=0.0)
    # Mirrors configs/risk.yaml's default_financing_spread default -- used
    # only when a representative Turbo ISIN has no measurable financing-
    # level history (pricing/instrument_compare.py prefers the measured
    # value whenever local state supports it).
    instrument_compare_financing_spread_fallback: float = 0.025
    # Mirrors configs/risk.yaml's financing_adjustment_jump_threshold_pct.
    instrument_compare_financing_adjustment_jump_threshold_pct: float = Field(default=0.08, gt=0.0)
    # Mirrors configs/simulation.yaml's defaults (path-simulation params for
    # the KO-probability estimate).
    instrument_compare_n_paths: int = Field(default=2000, gt=0)
    instrument_compare_path_method: str = "vol_scaled_bootstrap"
    instrument_compare_block_size: int = Field(default=5, gt=0)
    instrument_compare_lookback_days: int = Field(default=750, gt=0)
    instrument_compare_seed: int = 20260101


class ReturnStats(BaseModel):
    """Realized-outcome statistics for one group of labeled forward-ledger
    trades (Master Spec §38). Always populated with ``n``/``n_effective``
    and honesty notes; return-shape metrics are ``None`` when there is no
    data, and Sharpe/Sortino/PSR/DSR are ``None`` whenever ``not reliable``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    n: int
    n_effective: float
    reliable: bool
    notes: list[str] = Field(default_factory=list)

    success_rate: float | None = None
    success_rate_ci: WilsonInterval | None = None
    mean_return: float | None = None
    mean_return_ci: tuple[float, float] | None = None
    median_return: float | None = None
    profit_factor: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    psr: float | None = None
    dsr: float | None = None
    expected_shortfall_95: float | None = None

    ko_frequency: float | None = None
    mean_holding_days: float | None = None
    median_holding_days: float | None = None
    mean_spread_pct: float | None = None
    mean_financing_drag_pct: float | None = None
    financing_drag_n: int = 0

    brier: float | None = None
    ece: float | None = None
    calibration_n: int = 0

    product_selection_edge: float | None = None
    product_selection_edge_n: int = 0
    product_selection_regret: float | None = None
    product_selection_regret_n: int = 0
    issuer_drag: float | None = None
    issuer_drag_n: int = 0

    ambiguous_path_rate: float | None = None


class Grouping(BaseModel):
    """One grouping dimension's per-bucket :class:`ReturnStats` (Master Spec
    §38 "Gruppieren nach ...")."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: str
    buckets: dict[str, ReturnStats]


class PosteriorSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    signal_family: str
    horizon_days: int
    p_mu_positive: float
    posterior_mean: float
    posterior_std: float | None  # None when undefined (alpha <= 1 -> infinite)
    n: float


class ModelWeightSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    signal_family: str
    status: str
    weight: float
    weight_at_month_start: float | None
    weight_change: float | None


class TrialBudgetSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    quarter: str
    used: int
    budget: int


class DriftEventSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    detected_at: datetime
    stream_id: str
    signal_family: str | None
    metric: str
    ph_statistic: float
    threshold: float
    action: str


class ShadowPortfolioSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    portfolio: str
    n: int
    n_realized: int
    mean_net_return: float | None


class MonthlyReport(BaseModel):
    """A complete monthly performance report (Master Spec §38)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    month: date  # first of the reported calendar month
    month_end: date
    as_of: datetime
    generated_at: datetime
    status_only: bool

    n_scans_in_month: int
    category_counts_in_month: dict[str, int]

    actionable: ReturnStats
    shadow: ReturnStats
    groupings_actionable: list[Grouping]
    groupings_shadow: list[Grouping]

    strategy_posteriors: list[PosteriorSummary]
    model_weights: list[ModelWeightSummary]
    open_trials: list[str]
    trial_budget: TrialBudgetSummary
    drift_events: list[DriftEventSummary]
    shadow_portfolio: list[ShadowPortfolioSummary]

    # Instrument-class cost comparison block (Systemkonzept 1.2 §12.1
    # "Instrumentenneutralitaet", §23: required on the cost side of this
    # report). ``None`` when ``cfg.instrument_compare_underlying`` is unset
    # (the default) or the comparison could not be built at all -- see
    # ``data_quality_notes`` for the reason in the latter case. When
    # present, every number inside it is itself already tagged measured/
    # assumed/missing (``pricing/instrument_compare.py``) -- this field
    # being non-``None`` does NOT mean every line in it was measured.
    instrument_comparison: InstrumentComparisonResult | None = None

    data_quality_notes: list[str]
    narrative: list[str]


class _Item:
    __slots__ = ("entry", "label", "weight")

    def __init__(self, entry: LedgerEntry, label: LedgerLabel, weight: float) -> None:
        self.entry = entry
        self.label = label
        self.weight = weight


def _compute_return_stats(
    items: list[_Item],
    cfg: MonthlyReportConfig,
    rng: np.random.Generator,
    n_trials_for_deflation: int,
) -> ReturnStats:
    n = len(items)
    if n == 0:
        return ReturnStats(
            n=0,
            n_effective=0.0,
            reliable=False,
            notes=["Keine gelabelten Trades in diesem Zeitraum/dieser Gruppe."],
        )

    returns: npt.NDArray[np.float64] = np.array(
        [it.label.realized_selected_pnl for it in items], dtype=np.float64
    )
    n_effective = float(sum(it.weight for it in items))
    reliable = n >= cfg.min_trades_for_stats
    notes: list[str] = []

    successes = int(np.sum(returns > 0))
    wilson = wilson_interval(successes, n, cfg.wilson_confidence)
    success_rate = successes / n

    mean_return = float(np.mean(returns))
    median_return = float(np.median(returns))
    mean_ci: tuple[float, float] | None = None
    if n >= 2:
        mean_ci = bootstrap_ci(
            returns,
            lambda x: float(np.mean(x)),
            cfg.n_bootstrap,
            rng,
            alpha=1.0 - cfg.ci_confidence,
        )
    pf = profit_factor(returns)

    sharpe_val: float | None = None
    sortino_val: float | None = None
    psr_val: float | None = None
    dsr_val: float | None = None
    # `reliable` normally implies n >= cfg.min_trades_for_stats (default 10),
    # but a misconfigured (very low) threshold must never crash into
    # sharpe()/probabilistic_sharpe_ratio()'s own n>=2/n>=3 minimums -- guard
    # explicitly rather than relying on the config default.
    if reliable and n >= 3:
        mean_holding = max(1.0, statistics.mean(holding_days(it.entry, it.label) for it in items))
        periods_per_year = _TRADING_DAYS_PER_YEAR / mean_holding
        sharpe_val = sharpe(returns, periods_per_year=periods_per_year)
        sortino_val = sortino(returns, periods_per_year=periods_per_year)
        psr_val = probabilistic_sharpe_ratio(returns)
        dsr_val = deflated_sharpe_ratio(returns, max(1, n_trials_for_deflation))
    else:
        ci_str = (
            f"95%-Bootstrap-CI der mittleren Rendite: [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}]."
            if mean_ci is not None
            else "n=1: keine Bootstrap-CI berechenbar."
        )
        notes.append(
            f"Renditekennzahlen nicht aussagekraeftig: nur n={n} gelabelte Trades "
            f"(< min_trades_for_stats={cfg.min_trades_for_stats}); Sharpe/Sortino/PSR/DSR "
            f"nicht ausgewiesen (effektive Stichprobe n_effective={n_effective:.2f}). {ci_str}"
        )

    es95 = expected_shortfall(returns, cfg.es_alpha)
    ko_frequency = float(np.mean([1.0 if it.label.ko_hit else 0.0 for it in items]))
    holding = [holding_days(it.entry, it.label) for it in items]
    mean_holding_days = float(np.mean(holding))
    median_holding_days = float(np.median(holding))
    mean_spread_pct = float(np.mean([it.entry.entry_spread for it in items]))

    fin_vals = [v for it in items if (v := financing_drag_pct(it.entry, it.label)) is not None]
    mean_financing = float(np.mean(fin_vals)) if fin_vals else None

    p_arr = np.array([it.entry.p_profit for it in items], dtype=np.float64)
    y_arr = np.array(
        [
            1.0
            if it.label.realized_selected_pnl is not None and it.label.realized_selected_pnl > 0
            else 0.0
            for it in items
        ],
        dtype=np.float64,
    )
    brier = brier_score(p_arr, y_arr)
    ece = expected_calibration_error(p_arr, y_arr)

    edges = [
        it.label.realized_selected_pnl - it.label.median_turbo_pnl
        for it in items
        if it.label.median_turbo_pnl is not None and it.label.realized_selected_pnl is not None
    ]
    regrets = [
        it.label.best_turbo_pnl - it.label.realized_selected_pnl
        for it in items
        if it.label.best_turbo_pnl is not None and it.label.realized_selected_pnl is not None
    ]
    drags = [
        it.label.realized_selected_pnl - it.label.ideal_turbo_pnl
        for it in items
        if it.label.ideal_turbo_pnl is not None and it.label.realized_selected_pnl is not None
    ]
    ambiguous_rate = float(np.mean([1.0 if it.label.ambiguous_path else 0.0 for it in items]))

    return ReturnStats(
        n=n,
        n_effective=n_effective,
        reliable=reliable,
        notes=notes,
        success_rate=success_rate,
        success_rate_ci=wilson,
        mean_return=mean_return,
        mean_return_ci=mean_ci,
        median_return=median_return,
        profit_factor=pf,
        sharpe=sharpe_val,
        sortino=sortino_val,
        psr=psr_val,
        dsr=dsr_val,
        expected_shortfall_95=es95,
        ko_frequency=ko_frequency,
        mean_holding_days=mean_holding_days,
        median_holding_days=median_holding_days,
        mean_spread_pct=mean_spread_pct,
        mean_financing_drag_pct=mean_financing,
        financing_drag_n=len(fin_vals),
        brier=brier,
        ece=ece,
        calibration_n=n,
        product_selection_edge=statistics.mean(edges) if edges else None,
        product_selection_edge_n=len(edges),
        product_selection_regret=statistics.mean(regrets) if regrets else None,
        product_selection_regret_n=len(regrets),
        issuer_drag=statistics.mean(drags) if drags else None,
        issuer_drag_n=len(drags),
        ambiguous_path_rate=ambiguous_rate,
    )


def _dimension_key(
    entry: LedgerEntry, dim: str, cfg: MonthlyReportConfig, hash_map: dict[str, str]
) -> str:
    if dim == "underlying":
        return entry.underlying
    if dim == "direction":
        return entry.direction.value
    if dim == "horizon":
        return f"{entry.horizon_days}d"
    if dim == "issuer":
        return entry.issuer
    if dim == "leverage_bucket":
        return leverage_bucket(entry, cfg.leverage_bucket_edges)
    if dim == "signal_family":
        return signal_family_for(entry, hash_map)
    raise ValueError(f"unknown grouping dimension {dim!r}")


def _build_groupings(
    pairs: list[tuple[LedgerEntry, LedgerLabel]],
    weight_map: dict[str, float],
    cfg: MonthlyReportConfig,
    rng: np.random.Generator,
    n_trials: int,
    hash_map: dict[str, str],
) -> list[Grouping]:
    groupings: list[Grouping] = []
    for dim in _GROUP_DIMENSIONS:
        buckets: dict[str, list[_Item]] = defaultdict(list)
        for entry, label in pairs:
            key = _dimension_key(entry, dim, cfg, hash_map)
            buckets[key].append(_Item(entry, label, weight_map.get(entry.entry_id, 1.0)))
        bucket_stats = {
            key: _compute_return_stats(items, cfg, rng, n_trials)
            for key, items in sorted(buckets.items())
        }
        groupings.append(Grouping(dimension=dim, buckets=bucket_stats))
    return groupings


def _build_instrument_comparison(
    store: Store,
    cfg: MonthlyReportConfig,
    as_of: datetime,
    data_quality_notes: list[str],
) -> InstrumentComparisonResult | None:
    """Build the Systemkonzept 1.2 §12.1 instrument-class cost comparison
    block for this report, or ``None`` when it is disabled
    (``cfg.instrument_compare_underlying is None``, the default) or cannot
    be built at all (a ``data_quality_notes`` entry then explains why --
    this never raises out of ``build_monthly_report``, matching that
    function's existing "never crashes the report" contract for every other
    optional block).

    Entirely offline, like the rest of this module: the reference rate is
    always ``assumed`` here (see ``MonthlyReportConfig``'s own comment on
    ``instrument_compare_reference_rate_fallback`` for why), never fetched
    live -- that only happens in ``turboedge compare instruments``
    (``cli.py``).
    """
    if cfg.instrument_compare_underlying is None:
        return None
    try:
        direction = Direction(cfg.instrument_compare_direction)
    except ValueError as exc:
        data_quality_notes.append(
            f"Instrumentenklassenvergleich uebersprungen: ungueltige "
            f"instrument_compare_direction {cfg.instrument_compare_direction!r} ({exc})."
        )
        return None

    instruments_path = Path(cfg.instrument_compare_config_dir) / "instruments.yaml"
    try:
        instruments_cfg = load_instruments_config(instruments_path)
    except (OSError, ValueError) as exc:
        data_quality_notes.append(
            f"Instrumentenklassenvergleich uebersprungen: {instruments_path} nicht ladbar ({exc})."
        )
        return None

    ref_rate = InputValue.assumed(
        cfg.instrument_compare_reference_rate_fallback,
        "Monatsbericht ist offline (kein Netzwerkzugriff, siehe Modul-Docstring); kein "
        "live ECB-Datenpunkt verfuegbar -- konfigurierter Fallback "
        "(instrument_compare_reference_rate_fallback) verwendet.",
    )
    financing_fallback = InputValue.assumed(
        cfg.instrument_compare_financing_spread_fallback,
        "configs/risk.yaml-aequivalenter Fallback (instrument_compare_financing_spread_fallback).",
    )
    sim_params = SimulationParams(
        n_paths=cfg.instrument_compare_n_paths,
        method=cfg.instrument_compare_path_method,
        block_size=cfg.instrument_compare_block_size,
        lookback_days=cfg.instrument_compare_lookback_days,
        seed=cfg.instrument_compare_seed,
    )
    try:
        return compare_instruments(
            store,
            underlying_id=cfg.instrument_compare_underlying,
            notional_eur=cfg.instrument_compare_notional_eur,
            horizon_days=cfg.instrument_compare_horizon_days,
            direction=direction,
            ref_rate=ref_rate,
            financing_spread_fallback=financing_fallback,
            financing_adjustment_jump_threshold_pct=(
                cfg.instrument_compare_financing_adjustment_jump_threshold_pct
            ),
            instruments_cfg=instruments_cfg,
            sim=sim_params,
            as_of=as_of,
        )
    except ValueError as exc:
        data_quality_notes.append(f"Instrumentenklassenvergleich fehlgeschlagen: {exc}")
        return None


def build_monthly_report(
    store: Store,
    *,
    month: date,
    as_of: datetime,
    cfg: MonthlyReportConfig | None = None,
) -> MonthlyReport:
    """Build the monthly performance report for the calendar month
    containing ``month`` (only ``year``/``month`` are used).

    Data basis: ``forward_ledger`` + ``ledger_labels`` entries whose horizon
    ended within the month (``exit_due`` in ``[month_start, month_end]``),
    split into ACTIONABLE proposals and the stratified shadow sample (Master
    Spec §25/§46). If nothing matured this month at all, still returns a
    coherent status report (``status_only=True``): scan count, candidate
    category counts, and data-quality notes for the month, per Build
    Contract v2 W8 requirement 1.

    Never touches the network or mutates the store. Deterministic given the
    same store contents and ``cfg.bootstrap_seed`` (CLAUDE.md rule 16).
    """
    cfg = cfg if cfg is not None else MonthlyReportConfig()
    rng = np.random.default_rng(cfg.bootstrap_seed)
    generated_at = datetime.now(UTC)

    month_start = month.replace(day=1)
    month_end = last_day_of_month(month_start)

    ledger = ForwardLedger(store)
    all_pairs = ledger.entries()

    predicted_this_month = [
        (e, lbl) for e, lbl in all_pairs if month_start <= e.prediction_time.date() <= month_end
    ]
    n_scans_in_month = len({e.run_id for e, _ in predicted_this_month})
    category_counts = Counter(e.category.value for e, _ in predicted_this_month)

    matured_this_month = [
        (e, lbl) for e, lbl in all_pairs if month_start <= e.exit_due <= month_end
    ]
    labeled_matured = [
        (e, lbl)
        for e, lbl in matured_this_month
        if e.status == LedgerEntryStatus.LABELED and lbl is not None
    ]
    expired_matured = [
        (e, lbl) for e, lbl in matured_this_month if e.status == LedgerEntryStatus.EXPIRED_NO_DATA
    ]
    open_matured = [(e, lbl) for e, lbl in matured_this_month if e.status == LedgerEntryStatus.OPEN]

    data_quality_notes: list[str] = []
    if expired_matured:
        data_quality_notes.append(
            f"{len(expired_matured)} fällige Ledger-Einträge ohne jegliche Exit-Daten "
            "(status=expired_no_data): kein Bid, kein Underlying-Bar verfügbar."
        )
    if open_matured:
        data_quality_notes.append(
            f"{len(open_matured)} fällige Ledger-Einträge sind zum Berichtszeitpunkt "
            f"({as_of.isoformat()}) noch nicht gelabelt (status=open; label-Job vermutlich "
            "noch nicht gelaufen)."
        )
    ambiguous_n = sum(1 for _e, lbl in labeled_matured if lbl is not None and lbl.ambiguous_path)
    if labeled_matured and ambiguous_n:
        data_quality_notes.append(
            f"{ambiguous_n}/{len(labeled_matured)} gelabelte Einträge mit ambiguous_path=True "
            "(Master Spec §26): konservativ (nie optimistisch) aufgelöst."
        )

    weight_map = average_uniqueness_weights(labeled_matured)
    quarter = quarter_label(as_of)
    n_trials = effective_number_of_trials(store, quarter)
    registry_entries = store.list_model_registry_entries()
    hash_map = registry_hash_map(registry_entries)

    actionable_pairs = [(e, lbl) for e, lbl in labeled_matured if e.category == Category.ACTIONABLE]
    shadow_pairs = [(e, lbl) for e, lbl in labeled_matured if e.is_shadow]

    def _items(pairs: list[tuple[LedgerEntry, LedgerLabel]]) -> list[_Item]:
        return [_Item(e, lbl, weight_map.get(e.entry_id, 1.0)) for e, lbl in pairs]

    actionable_stats = _compute_return_stats(_items(actionable_pairs), cfg, rng, n_trials)
    shadow_stats = _compute_return_stats(_items(shadow_pairs), cfg, rng, n_trials)

    groupings_actionable = _build_groupings(
        actionable_pairs, weight_map, cfg, rng, n_trials, hash_map
    )
    groupings_shadow = _build_groupings(shadow_pairs, weight_map, cfg, rng, n_trials, hash_map)

    # -- strategy posteriors ---------------------------------------------
    families = sorted({m.signal_family for m in registry_entries}) or ["tsmom"]
    posteriors: list[PosteriorSummary] = []
    for family in families:
        for horizon in HORIZONS:
            posterior = StrategyPosterior.load(store, family, horizon, config=cfg.posterior_config)
            if posterior.n <= 0.0:
                continue
            std = posterior.posterior_std
            posteriors.append(
                PosteriorSummary(
                    signal_family=family,
                    horizon_days=horizon,
                    p_mu_positive=posterior.p_mu_positive(),
                    posterior_mean=posterior.posterior_mean,
                    posterior_std=std if np.isfinite(std) else None,
                    n=posterior.n,
                )
            )

    # -- model weights + change since month start -------------------------
    model_weights: list[ModelWeightSummary] = []
    for m in registry_entries:
        history = store.model_weight_history(m.model_id)
        weight_at_start: float | None = None
        for recorded_at, weight, _utility, _trial_id in history:
            if recorded_at.date() <= month_start:
                weight_at_start = weight
        change = (m.weight - weight_at_start) if weight_at_start is not None else None
        model_weights.append(
            ModelWeightSummary(
                model_id=m.model_id,
                signal_family=m.signal_family,
                status=m.status.value,
                weight=m.weight,
                weight_at_month_start=weight_at_start,
                weight_change=change,
            )
        )

    # -- trials / budget ----------------------------------------------------
    open_trials = [
        t.trial_id for t in store.list_research_trials() if t.status == TrialStatus.EXPERIMENTAL
    ]
    trial_budget = TrialBudgetSummary(
        quarter=quarter,
        used=store.count_research_trials_in_quarter(quarter),
        budget=cfg.trials_config.quarterly_budget,
    )

    # -- drift events detected in-month --------------------------------------
    drift_events = [
        DriftEventSummary(
            event_id=d.event_id,
            detected_at=d.detected_at,
            stream_id=d.stream_id,
            signal_family=d.signal_family,
            metric=d.metric,
            ph_statistic=d.ph_statistic,
            threshold=d.threshold,
            action=d.action,
        )
        for d in store.list_drift_events()
        if month_start <= d.detected_at.date() <= month_end
    ]

    # -- shadow portfolio comparison -----------------------------------------
    shadow_by_kind: dict[str, list[float]] = defaultdict(list)
    shadow_n_total: dict[str, int] = defaultdict(int)
    for sp in store.list_shadow_positions():
        if not (month_start <= sp.exit_due <= month_end):
            continue
        shadow_n_total[sp.portfolio.value] += 1
        if sp.realized_net_return is not None:
            shadow_by_kind[sp.portfolio.value].append(sp.realized_net_return)
    shadow_portfolio = [
        ShadowPortfolioSummary(
            portfolio=kind.value,
            n=shadow_n_total.get(kind.value, 0),
            n_realized=len(shadow_by_kind.get(kind.value, [])),
            mean_net_return=(
                float(np.mean(shadow_by_kind[kind.value]))
                if shadow_by_kind.get(kind.value)
                else None
            ),
        )
        for kind in cfg.shadow_portfolio_kinds
    ]

    # -- instrument-class cost comparison (Systemkonzept 1.2 §12.1/§23) -----
    instrument_comparison = _build_instrument_comparison(store, cfg, as_of, data_quality_notes)

    status_only = len(labeled_matured) == 0

    narrative: list[str] = []
    if status_only:
        narrative.append(
            f"Keine gelabelten Vorschläge fällig im Zeitraum {month_start.isoformat()}"
            f"-{month_end.isoformat()}."
        )
        if predicted_this_month:
            counts_str = ", ".join(f"{k}={v}" for k, v in sorted(category_counts.items()))
            narrative.append(
                f"{n_scans_in_month} Scan-Lauf/Läufe im Monat; Kandidaten: {counts_str}."
            )
        else:
            narrative.append("Keine Scan-Läufe mit Vorhersagen in diesem Monat protokolliert.")
        if data_quality_notes:
            narrative.append("Siehe data_quality_notes für Details zu offenen/fehlenden Labels.")
    else:
        narrative.append(
            f"{len(actionable_pairs)} ACTIONABLE- und {len(shadow_pairs)} Shadow-Trades "
            f"im Zeitraum {month_start.isoformat()}-{month_end.isoformat()} gelabelt."
        )
        if not actionable_stats.reliable:
            narrative.append(
                "ACTIONABLE-Stichprobe zu klein für belastbare Renditekennzahlen "
                f"(n={actionable_stats.n})."
            )
    if instrument_comparison is not None and instrument_comparison.factor_vs_future is not None:
        narrative.append(
            f"Instrumentenklassenvergleich ({instrument_comparison.underlying_id}, "
            f"{instrument_comparison.horizon_days}d, "
            f"{instrument_comparison.notional_eur:.0f} EUR): {instrument_comparison.factor_note}"
        )

    return MonthlyReport(
        month=month_start,
        month_end=month_end,
        as_of=as_of,
        generated_at=generated_at,
        status_only=status_only,
        n_scans_in_month=n_scans_in_month,
        category_counts_in_month=dict(category_counts),
        actionable=actionable_stats,
        shadow=shadow_stats,
        groupings_actionable=groupings_actionable,
        groupings_shadow=groupings_shadow,
        strategy_posteriors=posteriors,
        model_weights=model_weights,
        open_trials=open_trials,
        trial_budget=trial_budget,
        drift_events=drift_events,
        shadow_portfolio=shadow_portfolio,
        instrument_comparison=instrument_comparison,
        data_quality_notes=data_quality_notes,
        narrative=narrative,
    )


__all__ = [
    "DriftEventSummary",
    "Grouping",
    "ModelWeightSummary",
    "MonthlyReport",
    "MonthlyReportConfig",
    "PosteriorSummary",
    "ReturnStats",
    "ShadowPortfolioSummary",
    "TrialBudgetSummary",
    "build_monthly_report",
]
