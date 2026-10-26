"""Integration-wave CLI commands (Contract v3 Abschnitt E): ``scan-all``,
``label``, ``learn``, ``position reevaluate``, ``report monthly``,
``research tournament``, ``forecast``, ``backtest``.

Kept as a separate module (mirrors ``cli_state.py``'s own rationale) so
``cli.py`` -- a shared file -- only needs a short registration call. Every
command here writes ``reports/summary.json`` (``{"mode": ..., "counts":
{...}}``, plus command-specific extra fields) and, when
``TURBOEDGE_PUBLIC_LOGS`` is truthy, prints only counters to stdout (never
ISIN/WKN/price detail -- the full detail still reaches the operator via the
private Gmail report where applicable).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import structlog
import typer
from rich.console import Console
from rich.table import Table

from turboedge.adapters.ecb import EcbEstrAdapter
from turboedge.adapters.fallback_prices import YFinancePriceAdapter
from turboedge.adapters.registry import (
    build_http_client,
    build_product_adapters,
    build_reference_healthchecks,
)
from turboedge.backtest.excursion_eval import build_excursion_dataset, evaluate_excursion
from turboedge.backtest.ko_calibration import KoCalibrationMetrics, run_ko_calibration
from turboedge.backtest.purged_cv import PurgedWalkForwardSplit
from turboedge.backtest.synthetic_turbo_ev import (
    SyntheticTurboConfig,
    run_synthetic_net_ev_trial,
)
from turboedge.backtest.walkforward import walk_forward_evaluate
from turboedge.config import config_hash
from turboedge.learning.drift import PageHinkley, PageHinkleyConfig, record_drift_event
from turboedge.learning.failed_hypotheses import default_path as failed_hypotheses_default_path
from turboedge.learning.labeler import label_due_entries
from turboedge.learning.ledger import ForwardLedger
from turboedge.learning.ledger_health import ledger_health
from turboedge.learning.posterior import StrategyPosterior
from turboedge.learning.registry import ModelRegistry
from turboedge.learning.trials import backfill_w9_trials
from turboedge.meta import render_research_queue
from turboedge.meta.research_queue import IllegalTransition, ResearchQueue
from turboedge.models.baselines import (
    RegimeConditionalEmpiricalModel,
    RegularizedLinearLocationModel,
    RobustLocationScaleModel,
)
from turboedge.models.directional import (
    LogisticDirectionModel,
    NullModel,
    TsmomForecastModel,
)
from turboedge.models.forecast import ForecastModel
from turboedge.notifications.dedup import NotificationDeduplicator, notification_hash
from turboedge.notifications.gmail import EmailMessageSpec, GmailCredentials, GmailNotifier
from turboedge.notifications.templates import PositionUpdateContext, render_position_update
from turboedge.pipeline.archive import run_product_archive
from turboedge.pipeline.scan_all import run_scan_all
from turboedge.positions.reevaluate import reevaluate_open_positions
from turboedge.provenance import git_commit
from turboedge.reporting.html import save_monthly_report, save_weekly_report, summary_counts
from turboedge.reporting.monthly import build_monthly_report
from turboedge.reporting.redaction import redact_console_enabled
from turboedge.reporting.weekly import run_research_tournament
from turboedge.research.data_audit import (
    PLAUSIBLE_SCAN_SECONDS,
    ledger_census,
    point_in_time_audit,
    quote_freshness,
    snapshot_staleness_advantage,
)
from turboedge.research.lottery_pricing import MIN_INDEPENDENT_DAYS
from turboedge.research.lottery_pricing import evaluate as evaluate_lottery_pricing
from turboedge.research.product_pairs import ComparabilityCriteria, census
from turboedge.research.step import run_research_step
from turboedge.simulation.ko_calibration import build_ko_calibration_dataset
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    KoCalibrationPromotionRecord,
    KoCalibrationResultRecord,
    LedgerEntryStatus,
    WalkforwardResultRecord,
)
from turboedge.universe.underlying_map import resolve_underlying_id

logger = structlog.get_logger(__name__)

console = Console()
err_console = Console(stderr=True)

_SIGNAL_FAMILY_VERSION_RE = re.compile(r"^(?P<family>.+)_v\d+$")
_DEFAULT_MODEL_FACTORIES: dict[str, Any] = {
    "tsmom_forecast_v1": TsmomForecastModel,
    "logistic_direction_v1": LogisticDirectionModel,
    "null_v1": NullModel,
    # Phase D pre-registered baselines (docs/measured_results.md), measured
    # against the null model before any new challenger -- never part of the
    # live scan ensemble (models.forecast.build_default_models), only this
    # measurement/persistence path.
    "regime_conditional_empirical_v1": RegimeConditionalEmpiricalModel,
    "regularized_linear_location_v1": RegularizedLinearLocationModel,
    "robust_location_scale_t_v1": RobustLocationScaleModel,
}


def _write_summary(mode: str, counts: dict[str, int], **extra: Any) -> Path:
    path = Path("reports") / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"mode": mode, "counts": counts}
    payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _print_counts(label: str, counts: dict[str, Any]) -> None:
    line = "  ".join(f"{k}={v}" for k, v in counts.items())
    console.print(f"[bold]{label}[/bold]  {line}")


def _build_scan_dependencies(
    app_ctx: Any,
) -> tuple[Any, YFinancePriceAdapter, EcbEstrAdapter, list[Any]]:
    product_adapters = build_product_adapters(app_ctx.cfg)
    price_adapter = YFinancePriceAdapter()
    ecb_source_cfg = app_ctx.cfg.sources.get("ecb")
    ecb_http = build_http_client(ecb_source_cfg) if ecb_source_cfg is not None else None
    if ecb_http is None:
        from turboedge.adapters.base import HttpClient

        ecb_http = HttpClient(
            user_agent=app_ctx.cfg.default.http_default_user_agent,
            timeout_s=app_ctx.cfg.default.http_default_timeout_s,
            min_interval_s=0.0,
            max_retries=3,
        )
    estr_adapter = EcbEstrAdapter(ecb_http, fallback_rate=app_ctx.cfg.risk.reference_rate_fallback)
    reference_checks = build_reference_healthchecks(app_ctx.cfg)
    return product_adapters, price_adapter, estr_adapter, reference_checks


def _build_notifier(app_ctx: Any) -> GmailNotifier:
    creds = GmailCredentials.from_env()
    return GmailNotifier(app_ctx.cfg.gmail, creds)


# --------------------------------------------------------------------------
# scan-all
# --------------------------------------------------------------------------


def scan_all_cmd(
    ctx: typer.Context,
    underlying: Annotated[
        list[str] | None,
        typer.Option("--underlying", help="Restrict to these underlying id(s) (repeatable)"),
    ] = None,
    email: bool = typer.Option(False, "--email", help="Send ACTIONABLE trade-proposal emails"),
    json_out: str = typer.Option(None, "--json-out", help="Write full scan-all JSON to PATH"),
) -> None:
    """Scan every active underlying with the full forecast/EV pipeline
    (Contract v3 Abschnitt B). ``pipeline.yml``'s scan mode calls this
    exactly (``turboedge scan-all --email``)."""
    app_ctx = ctx.obj
    product_adapters, price_adapter, estr_adapter, reference_checks = _build_scan_dependencies(
        app_ctx
    )
    notifier = _build_notifier(app_ctx) if email else None

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        result = run_scan_all(
            app_ctx.cfg,
            store,
            app_ctx.state_dir,
            underlying_ids=underlying,
            product_adapters=product_adapters,
            price_adapter=price_adapter,
            estr_adapter=estr_adapter,
            reference_healthchecks=reference_checks,
            notifier=notifier,
            email=email,
        )

    if redact_console_enabled():
        _print_counts("scan-all", result.counts)
    else:
        table = Table(title="scan-all")
        table.add_column("Underlying")
        for cat in ("ACTIONABLE", "WATCH", "REJECT", "DATA_QUALITY"):
            table.add_column(cat, justify="right")
        table.add_column("elapsed_s", justify="right")
        for uid in result.counts_by_underlying:
            row_counts = result.counts_by_underlying[uid]
            table.add_row(
                uid,
                *(
                    str(row_counts.get(cat, 0))
                    for cat in ("ACTIONABLE", "WATCH", "REJECT", "DATA_QUALITY")
                ),
                f"{result.elapsed_by_underlying.get(uid, 0.0):.1f}",
            )
        console.print(table)
        if result.skipped:
            console.print("[yellow]Skipped underlyings:[/yellow]")
            for uid, reason in result.skipped.items():
                console.print(f"  - {uid}: {reason}")
        console.print(f"Total elapsed: {result.elapsed_s:.1f}s")

    if json_out:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_ids": result.run_ids,
            "counts": result.counts,
            "counts_by_underlying": result.counts_by_underlying,
            "skipped": result.skipped,
            "warnings": result.warnings,
            "elapsed_s": result.elapsed_s,
            "elapsed_by_underlying": result.elapsed_by_underlying,
            "candidates_by_underlying": {
                uid: [c.model_dump(mode="json") for c in scan_result.candidates]
                for uid, scan_result in result.results.items()
            },
        }
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2, default=str)

    _write_summary(
        "scan",
        result.counts,
        counts_by_underlying=result.counts_by_underlying,
        skipped=result.skipped,
        elapsed_s=round(result.elapsed_s, 2),
        # Consumed by pipeline.yml's scan job (state pack-snapshots --run-id
        # ...) to build the incremental per-scan Parquet snapshot artifact --
        # see state/archive.py's module docstring ("Parquet archiving").
        run_ids=result.run_ids,
    )

    if not result.results:
        err_console.print("[red]scan-all: every underlying was skipped[/red]")
        raise typer.Exit(code=3)


# --------------------------------------------------------------------------
# label
# --------------------------------------------------------------------------


def label_cmd(ctx: typer.Context) -> None:
    """Label every forward-ledger entry whose ``exit_due`` has arrived
    (Master Spec §25)."""
    app_ctx = ctx.obj
    now = datetime.now(UTC)

    with Store(app_ctx.db_path) as store:
        store.init_schema()

        def price_bars_lookup(underlying_id: str) -> list[Any]:
            return store.latest_underlying_bars(underlying_id, 2000)

        result = label_due_entries(store, now, price_bars_lookup=price_bars_lookup)
        # Reported alongside the outcome, because every outcome count is
        # zero both when nothing was due and when everything was due and
        # none of it could be labelled. Without the population, `labeled=0`
        # is unattributable -- and on 2026-10-23 that ambiguity sat next to
        # a ledger holding 3,670 entries and no labels.
        health = ledger_health(store, as_of=now)

    counts = {
        "labeled": result.labeled,
        "ko": result.ko,
        "ambiguous": result.ambiguous,
        "missing_data": result.missing_data,
        "shadow_positions_labeled": result.shadow_positions_labeled,
    }
    _print_counts("label", counts)
    console.print(f"  ledger: {health.summary()}", markup=False, highlight=False)
    console.print(
        f"  open {health.open_entries}  due {health.due_now}  "
        f"overdue {health.overdue_unlabeled}  labelled {health.labeled_entries}  "
        f"usable for research {health.usable_for_research}",
        markup=False,
        highlight=False,
    )
    if not health.healthy:
        err_console.print(
            f"[red]{health.overdue_unlabeled} forward-ledger entry(ies) are overdue and "
            f"still unlabelled (oldest {health.oldest_overdue_days} days).[/red] That is "
            "not a data-maturity question -- an entry past its exit_due should have been "
            "labelled or marked expired_no_data by now."
        )
    _write_summary(
        "label",
        counts,
        ledger_open=health.open_entries,
        ledger_due_now=health.due_now,
        ledger_overdue_unlabeled=health.overdue_unlabeled,
        ledger_oldest_overdue_days=health.oldest_overdue_days,
        ledger_next_due=health.next_due_date.isoformat() if health.next_due_date else None,
        ledger_entries_created_today=health.entries_created_today,
        ledger_newest_label_age_days=health.newest_label_age_days,
        ledger_usable_for_research=health.usable_for_research,
        ledger_ko_hits=health.ko_hits,
        ledger_ambiguous=health.ambiguous_paths,
        ledger_missing_exit_quote=health.missing_exit_quote,
        ledger_waiting_for_data=health.waiting_for_data,
    )


# --------------------------------------------------------------------------
# learn
# --------------------------------------------------------------------------


def _signal_family_for(model_hash: str, signal_id: str, hash_map: dict[str, str]) -> str:
    family = hash_map.get(model_hash)
    if family is not None:
        return family
    m = _SIGNAL_FAMILY_VERSION_RE.match(signal_id)
    return m.group("family") if m is not None else signal_id


def learn_cmd(ctx: typer.Context) -> None:
    """Update strategy posteriors, ensemble weights and drift detection from
    every labeled forward-ledger entry (Master Spec §21/§30).

    Idempotent by design: each run rebuilds every touched
    ``StrategyPosterior`` from the *complete* labeled history for its
    ``(signal_family, horizon_days)`` rather than incrementally folding in
    "new since last run" rows (no such tracking flag exists on
    ``LedgerLabel``) -- safe to call repeatedly, never double-counts.
    """
    app_ctx = ctx.obj
    now = datetime.now(UTC)

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        registry = ModelRegistry(store)
        hash_map = {e.model_hash: e.signal_family for e in registry.list()}

        pairs = ForwardLedger(store).entries(status=LedgerEntryStatus.LABELED)
        usable = [
            (entry, label)
            for entry, label in pairs
            if label is not None and label.realized_selected_pnl is not None
        ]

        by_family_horizon: dict[tuple[str, int], list[float]] = {}
        by_model: dict[str, list[float]] = {}
        by_family_chrono: dict[str, list[tuple[datetime, float]]] = {}
        for entry, label in usable:
            assert label is not None and label.realized_selected_pnl is not None
            family = _signal_family_for(entry.model_hash, entry.signal_id, hash_map)
            by_family_horizon.setdefault((family, entry.horizon_days), []).append(
                label.realized_selected_pnl
            )
            by_model.setdefault(entry.model_hash, []).append(label.realized_selected_pnl)
            by_family_chrono.setdefault(family, []).append(
                (entry.prediction_time, label.realized_selected_pnl)
            )

        posteriors_updated = 0
        for (family, horizon), returns in by_family_horizon.items():
            posterior = StrategyPosterior.new(
                family, horizon, config=app_ctx.cfg.learning.posterior
            )
            posterior.update(np.asarray(returns, dtype=np.float64))
            posterior.save(store, updated_at=now)
            posteriors_updated += 1

        model_id_by_hash = {e.model_hash: e.model_id for e in registry.list()}
        utilities = {
            model_id_by_hash[model_hash]: float(np.mean(returns))
            for model_hash, returns in by_model.items()
            if model_hash in model_id_by_hash
        }
        models_reweighted = 0
        if utilities:
            registry.update_weights(
                utilities, eta=app_ctx.cfg.learning.eta, w_min=app_ctx.cfg.learning.w_min
            )
            models_reweighted = len(utilities)
        elif usable:
            # Measured 2026-10-26, run 40377514500: 2,785 labelled entries
            # produced `models_reweighted=0`, which read as "nothing needed
            # reweighting" and is in fact "nothing could be attributed".
            #
            # `by_model` is keyed by `LedgerEntry.model_hash`, which
            # `pipeline/scan.py` fills with `ensemble.model_hash` -- a hash
            # of the *combination*. `model_id_by_hash` is keyed by the
            # hashes of individual models as the scan registered them. Two
            # different namespaces, so the intersection is empty by
            # construction, not by accident. On top of that a model's hash
            # covers `training_end` and its fitted coefficients, so it
            # changes on every refit while the registry keeps only the
            # newest -- a matured entry's hash could not match even within
            # one namespace.
            #
            # Attributing an ensemble's realised P&L back to its members
            # needs the per-model weights as they stood at prediction time,
            # and the ledger does not record them. So this is reported
            # rather than guessed: an invented attribution would move real
            # model weights on made-up evidence.
            logger.warning(
                "learn_weights_unattributable",
                labeled_entries=len(usable),
                distinct_ledger_model_hashes=len(by_model),
                registered_model_hashes=len(model_id_by_hash),
                reason=(
                    "no labelled entry's model_hash matches a registered model; the "
                    "ledger records the ensemble hash while the registry records "
                    "per-model hashes, and a model's hash changes on every refit"
                ),
            )

        drift_events = 0
        for family, chrono in by_family_chrono.items():
            chrono.sort(key=lambda item: item[0])
            detector = PageHinkley(config=PageHinkleyConfig())
            fired_at = detector.update_many([r for _t, r in chrono])
            if fired_at is not None:
                record_drift_event(
                    store,
                    stream_id=f"realized_return:{family}",
                    signal_family=family,
                    metric="realized_return",
                    detector=detector,
                    detected_at=now,
                )
                drift_events += 1

    counts = {
        "posteriors_updated": posteriors_updated,
        "models_reweighted": models_reweighted,
        "drift_events": drift_events,
        "labeled_entries_considered": len(usable),
        # Reported alongside the count so that a zero can be read. A
        # `models_reweighted=0` with zero unattributable entries means
        # there was nothing to learn from; with thousands it means the
        # attribution failed, and those are opposite situations.
        "labeled_entries_unattributable": len(usable) if not models_reweighted else 0,
    }
    _print_counts("learn", counts)
    _write_summary("learn", counts)


# --------------------------------------------------------------------------
# position reevaluate
# --------------------------------------------------------------------------


def position_reevaluate_cmd(
    ctx: typer.Context,
    email: bool = typer.Option(False, "--email", help="Send position-update emails on change"),
) -> None:
    """Re-evaluate every open manual position (Contract v3 Abschnitt D)."""
    app_ctx = ctx.obj
    price_adapter = YFinancePriceAdapter()
    now = datetime.now(UTC)

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        results = reevaluate_open_positions(
            cfg=app_ctx.cfg, store=store, price_adapter=price_adapter, as_of=now
        )

        notifier = _build_notifier(app_ctx) if email else None
        sent = 0
        if notifier is not None:
            dedup = NotificationDeduplicator(store)
            for result in results:
                if not (result.status_changed or result.materially_changed):
                    continue
                ev = result.evaluation
                context = PositionUpdateContext(
                    wkn=ev.wkn,
                    isin=ev.isin,
                    underlying_id=ev.underlying_id,
                    status=ev.status.value,
                    reasons=ev.reasons,
                    current_bid=ev.current_bid,
                    remaining_horizon_days=ev.remaining_horizon_days,
                    remaining_lcb_ev=ev.remaining_lcb_ev,
                    remaining_p_ko=ev.remaining_p_ko,
                    unrealized_return=ev.unrealized_return,
                )
                subject, body = render_position_update(context)
                key_values = {
                    "status": ev.status.value,
                    "lcb": round(ev.remaining_lcb_ev or 0.0, 3),
                    "p_ko": round(ev.remaining_p_ko or 0.0, 3),
                }
                n_hash = notification_hash(ev.position_id, "POSITION_UPDATE", key_values)
                if not dedup.should_send(n_hash):
                    continue
                recipients = notifier.credentials.recipients if notifier.credentials else []
                spec = EmailMessageSpec(subject=subject, body_text=body, to=recipients)
                send_result = notifier.send(spec)
                dedup.mark_sent(n_hash, ev.position_id, "POSITION_UPDATE", subject, sent_at=now)
                if send_result.sent:
                    sent += 1

    counts: dict[str, int] = {}
    for result in results:
        counts[result.evaluation.status.value] = counts.get(result.evaluation.status.value, 0) + 1
    counts["mails_sent"] = sent
    if redact_console_enabled():
        _print_counts("position reevaluate", counts)
    else:
        table = Table(title="Position Reevaluation")
        table.add_column("WKN")
        table.add_column("Status")
        table.add_column("Changed", justify="center")
        table.add_column("Reasons")
        for result in results:
            ev = result.evaluation
            table.add_row(
                ev.wkn,
                ev.status.value,
                "yes" if result.status_changed else "no",
                "; ".join(ev.reasons[:3]),
            )
        console.print(table)
    _write_summary("position_reevaluate", counts)


# --------------------------------------------------------------------------
# report monthly / research tournament
# --------------------------------------------------------------------------


def report_monthly_cmd(
    ctx: typer.Context,
    month: str = typer.Option(None, "--month", help="YYYY-MM; defaults to the current month"),
    email: bool = typer.Option(False, "--email", help="Email the monthly report"),
) -> None:
    """Build (and persist) the monthly performance report (Master Spec §38)."""
    app_ctx = ctx.obj
    now = datetime.now(UTC)
    target_month = date.fromisoformat(f"{month}-01") if month else now.date().replace(day=1)

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        report = build_monthly_report(
            store, month=target_month, as_of=now, cfg=app_ctx.cfg.reporting.monthly_config()
        )
        json_path, html_path = save_monthly_report(report, reports_dir=Path("reports"))

        if email:
            from turboedge.reporting.html import to_plain_text

            notifier = _build_notifier(app_ctx)
            dedup = NotificationDeduplicator(store)
            body = to_plain_text(report)
            subject = f"TurboEdge Monthly Report — {target_month.strftime('%Y-%m')}"
            key_values: dict[str, float | str | None] = {
                "month": target_month.isoformat(),
                "status_only": int(report.status_only),
            }
            n_hash = notification_hash(None, "MONTHLY_REPORT", key_values)
            if dedup.should_send(n_hash):
                recipients = notifier.credentials.recipients if notifier.credentials else []
                spec = EmailMessageSpec(subject=subject, body_text=body, to=recipients)
                notifier.send(spec)
                dedup.mark_sent(n_hash, None, "MONTHLY_REPORT", subject, sent_at=now)

    counts = summary_counts(report)
    _print_counts("report monthly", counts)
    console.print(f"Wrote {json_path} / {html_path}")
    _write_summary("report_monthly", counts, month=target_month.isoformat())


#: One entry per pre-registered synthetic-EV trial: result path, the date it
#: opens, and which sigma channel it tests. The result file's existence is the
#: idempotency guard -- a pre-registered trial runs exactly once, and a
#: scheduled job that repeats it would be multiple testing by the back door.
_SigmaSource = Literal["historical", "forecast", "null_forecast"]

_SYNTHETIC_EV_TRIALS: dict[str, tuple[Path, str, _SigmaSource]] = {
    # 2026Q4-001: the mean channel. Run 2026-09-26 under its Amendment C and
    # charged to Q3 as the 12th of 6. Result: FAIL_NULL_RESULT, -0.020547.
    "2026Q4-001": (Path("docs/results_2026Q4_001.json"), "2026-10-01", "historical"),
    # 2026Q4-002: the width channel -- each arm's path dispersion set to its own
    # forecast sigma, which `simulate_paths` only became able to express on
    # 2026-09-26. Deliberately NOT brought forward: the mechanism it depends on
    # is days old (see its §7), and its Amendment A already found the premise in
    # its own §2 to be backwards.
    "2026Q4-002": (Path("docs/results_2026Q4_002.json"), "2026-10-01", "forecast"),
}


def research_synthetic_ev_cmd(
    ctx: typer.Context,
    trial: str = typer.Option("2026Q4-001", "--trial", help="Which pre-registered trial to run"),
    force: bool = typer.Option(
        False, "--force", help="Re-run even though a result already exists (voids the trial)"
    ),
    not_before: str | None = typer.Option(
        None, "--not-before", help="Override the trial's own opening date (ISO)"
    ),
) -> None:
    """Run one pre-registered synthetic-EV trial (`docs/preregistration_<trial>.md`).

    Exactly once. The pre-registration fixes the primary hypothesis, the
    method, the decision rule and the denominator before the data was seen;
    what makes it confirmatory rather than retrospective is that nobody gets
    to choose when to run it, or to run it again.

    Two guards enforce that, and both refuse rather than warn:

    * A result file already on disk means the trial has been run. `--force`
      overrides it and says in the output that doing so voids the trial.
    * The trial's own opening date refuses an early run. Q3 is closed at 11 of
      6 (GOVERNANCE.md §11.1), so a Q3 run would either consume a budget unit
      that does not exist or charge Q3 work to Q4 -- the two errors §11.1 and
      §11.5 already record. `--not-before` overrides the date, and an override
      needs a written amendment in the pre-registration, as 2026Q4-001's
      Amendment C is.
    """
    app_ctx = ctx.obj
    now = datetime.now(UTC)

    if trial not in _SYNTHETIC_EV_TRIALS:
        console.print(
            f"[red]Unknown trial {trial!r}.[/red] Registered: "
            f"{', '.join(sorted(_SYNTHETIC_EV_TRIALS))}"
        )
        raise typer.Exit(code=4)
    result_path, opens_on, sigma_source = _SYNTHETIC_EV_TRIALS[trial]
    threshold = date.fromisoformat(not_before or opens_on)

    if now.date() < threshold:
        console.print(
            f"[yellow]Not run:[/yellow] today is {now.date()}, trial {trial} opens "
            f"{threshold}. Q3 is closed at 11 of 6; running now would mis-charge the quarter."
        )
        raise typer.Exit(code=2)

    if result_path.exists() and not force:
        console.print(
            f"[yellow]Not run:[/yellow] {result_path} already exists, so trial {trial} has "
            "been run. Re-running it would turn one test into many. Use --force only if you "
            "intend to void it."
        )
        raise typer.Exit(code=3)

    price_adapter = YFinancePriceAdapter()
    bars_by_underlying = {
        u: price_adapter.fetch_daily_bars(u, lookback_days=4000)
        for u in ("DAX", "NDX", "EURUSD", "XAU")
    }
    provenance = {u: len(b) for u, b in bars_by_underlying.items()}
    console.print(f"Bars fetched {now.date()}: {provenance}")

    result = run_synthetic_net_ev_trial(
        bars_by_underlying, config=SyntheticTurboConfig(sigma_source=sigma_source)
    )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump(mode="json")
    payload["provenance"] = {
        "fetched_at": now.isoformat(),
        "bars_per_underlying": provenance,
        "lookback_days": 4000,
        "git_commit": app_ctx.git_commit,
        "forced": force,
        "trial": trial,
        "sigma_source": sigma_source,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    console.print(
        f"\n{trial} {result.verdict}  "
        f"mean dLCB_NetEV={result.mean_delta_lcb_net_ev:+.6f}  "
        f"p={result.p_value:.4f} (alpha={result.alpha})  n_dates={result.n_dates}",
        markup=False,
        highlight=False,
    )
    console.print(f"  {result.verdict_reason}", markup=False, highlight=False)
    console.print(f"Wrote {result_path}")
    _write_summary(
        "research_synthetic_ev",
        {"n_dates": result.n_dates},
        verdict=result.verdict,
        p_value=result.p_value,
        mean_delta_lcb_net_ev=result.mean_delta_lcb_net_ev,
    )


#: Exit code for "the evaluation works, the data has not matured". Chosen
#: distinct from 1 so a workflow can accept it while still failing on a real
#: fault -- see `.github/workflows/pipeline.yml`'s weekly job.
EXCURSION_INSUFFICIENT_DATA = 3


def research_excursion_cmd(
    ctx: typer.Context,
    underlying: str | None = typer.Option(
        None, "--underlying", help="Restrict to one underlying (default: all pooled)"
    ),
    horizon: int = typer.Option(3, "--horizon", help="Label horizon in days, for purge/embargo"),
    min_train: int = typer.Option(200, "--min-train", help="Walk-forward initial training size"),
    step: int = typer.Option(100, "--step", help="Walk-forward test block size"),
    json_out: str = typer.Option(None, "--json-out", help="Write the full result JSON to PATH"),
) -> None:
    """Measure MAE/MFE prediction against its unconditional null (queue RO-MAE/MFE-PREDICTION).

    Read-only: it changes no model weight, no gate and no threshold, and
    promotes nothing. It answers one question -- does a model conditioning on
    entry-time features beat the unconditional excursion distribution
    out-of-sample -- and is allowed to answer "there is not enough data to
    tell", which is the expected outcome until the forward ledger spans
    enough independent label windows.
    """
    app_ctx = ctx.obj
    now = datetime.now(UTC)

    # Three outcomes, three exit codes, because a workflow that cannot tell
    # them apart hides the only one that matters:
    #
    #   0  SUCCESS            -- a valid evaluation ran, whatever its verdict
    #   EXCURSION_INSUFFICIENT_DATA  -- working, but not enough real
    #                          observations have matured yet. Expected for a
    #                          long time, and not a defect.
    #   1  ERROR              -- a technical fault. Must fail the job.
    #
    # Until 2026-10-25 "no usable rows" and a genuine crash both exited 1 and
    # the weekly step carried a blanket `continue-on-error`, so a real fault
    # was indistinguishable from a young ledger.
    with Store(app_ctx.db_path) as store:
        store.init_schema()
        try:
            dataset = build_excursion_dataset(store, underlying=underlying, as_of=now)
        except ValueError as exc:
            health = ledger_health(store, as_of=now)
            console.print(f"EXCURSION INSUFFICIENT_DATA  ({underlying or 'all underlyings'})")
            console.print(f"  {exc}", markup=False, highlight=False)
            console.print(f"  ledger: {health.summary()}", markup=False, highlight=False)
            _write_summary(
                "research_excursion",
                {"rows": 0},
                verdict="INSUFFICIENT_DATA",
                ledger_open=health.open_entries,
                ledger_overdue_unlabeled=health.overdue_unlabeled,
                ledger_next_due=(
                    health.next_due_date.isoformat() if health.next_due_date else None
                ),
                ledger_usable_for_research=health.usable_for_research,
            )
            # Only legitimate while nothing is overdue. An overdue entry means
            # the shortage is a labeling fault, not a maturity question, and
            # that must not be reported as "waiting for data".
            raise typer.Exit(code=EXCURSION_INSUFFICIENT_DATA if health.healthy else 1) from exc
        except Exception as exc:
            console.print(f"EXCURSION ERROR  ({underlying or 'all underlyings'})")
            err_console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
            _write_summary("research_excursion", {"rows": 0}, verdict="ERROR", error=str(exc))
            raise typer.Exit(code=1) from exc
        try:
            result = evaluate_excursion(
                dataset,
                splitter=PurgedWalkForwardSplit(
                    horizon=horizon, embargo=horizon, min_train=min_train, step=step
                ),
            )
        except Exception as exc:
            console.print(f"EXCURSION ERROR  ({underlying or 'all underlyings'})")
            err_console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
            _write_summary(
                "research_excursion",
                {"rows": len(dataset.y_mae)},
                verdict="ERROR",
                error=str(exc),
            )
            raise typer.Exit(code=1) from exc

    console.print(f"EXCURSION {result.verdict}  ({underlying or 'all underlyings'})")
    console.print(
        f"  rows {len(dataset.y_mae)}  distinct dates {result.n_distinct_dates}  "
        f"effective sample {result.effective_sample:.2f}"
    )
    dropped = {k: v for k, v in result.dropped_rows.items() if v}
    console.print(f"  dropped: {dropped or 'none'}")
    for reason in result.verdict_reasons:
        console.print(f"  - {reason}", markup=False, highlight=False)
    for target_result in result.results:
        mean_null = sum(target_result.pinball_by_tau_null.values()) / len(
            target_result.pinball_by_tau_null
        )
        mean_cond = sum(target_result.pinball_by_tau_conditional.values()) / len(
            target_result.pinball_by_tau_conditional
        )
        console.print(
            f"  {target_result.target.value}: CRPS null {target_result.crps_null:.6f} vs "
            f"conditional {target_result.crps_conditional:.6f} | "
            f"mean pinball {mean_null:.6f} vs {mean_cond:.6f} | "
            f"coverage {target_result.coverage_null:.2f} vs "
            f"{target_result.coverage_conditional:.2f} | "
            f"crossing {target_result.quantile_crossing_rate:.2%}"
        )

    if json_out:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"Wrote {json_path}")

    _write_summary(
        "research_excursion",
        {"distinct_dates": result.n_distinct_dates, "rows": len(dataset.y_mae)},
        verdict=result.verdict,
        effective_sample=result.effective_sample,
    )

    if result.verdict == "INSUFFICIENT_SAMPLE":
        # The evaluation ran and is working; there is simply not enough
        # independent observation yet. Reported as such rather than as a
        # success, so "it ran" is not mistaken for "it measured something".
        raise typer.Exit(code=EXCURSION_INSUFFICIENT_DATA)


def research_product_pairs_cmd(
    ctx: typer.Context,
    underlying: str | None = typer.Option(
        None, "--underlying", help="Restrict to one underlying (default: all)"
    ),
    since: str | None = typer.Option(
        None, "--since", help="Only snapshots at or after this ISO timestamp"
    ),
    barrier_tolerance: float = typer.Option(
        0.10, "--barrier-tolerance", help="Relative barrier-distance tolerance"
    ),
    leverage_tolerance: float = typer.Option(
        0.15, "--leverage-tolerance", help="Relative leverage tolerance"
    ),
    window_seconds: int = typer.Option(
        60, "--window-seconds", help="Quote window within which products count as simultaneous"
    ),
    json_out: str = typer.Option(None, "--json-out", help="Write the full census to PATH"),
) -> None:
    """Count comparable cross-issuer product pairs in the stored archive.

    The precondition for pre-registering the product-selection cost question
    (`docs/research_plan_product_selection_cost_edge.md` §4): before asking
    whether choosing between comparable products of different issuers buys a
    real cost advantage, it has to be established how often the archive even
    contains such a pair.

    Read-only. Computes no cost, tests no hypothesis, opens no trial and
    consumes no research budget. Every count is a lower bound -- matching
    buckets at the tolerance width rather than comparing all pairs, so two
    products either side of a bucket edge are not counted.
    """
    app_ctx = ctx.obj
    criteria = ComparabilityCriteria(
        barrier_pct_tolerance=barrier_tolerance,
        leverage_tolerance=leverage_tolerance,
        window_seconds=window_seconds,
    )
    since_dt = datetime.fromisoformat(since).astimezone(UTC) if since else None

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        snapshots = store.list_product_snapshots(underlying_id=underlying, since=since_dt)
        result = census(snapshots, criteria=criteria)

    console.print(f"COMPARABLE PAIRS  {result.summary()}")
    console.print(
        f"  snapshots stored {len(snapshots)}  usable {result.snapshots_considered}",
        markup=False,
        highlight=False,
    )
    if result.dropped:
        for reason, count in sorted(result.dropped.items(), key=lambda kv: -kv[1]):
            console.print(f"    dropped {reason}: {count}", markup=False, highlight=False)
    for uid, count in sorted(result.pairs_by_underlying.items()):
        dates = result.distinct_dates_by_underlying.get(uid, 0)
        console.print(
            f"  {uid}: {count} pair(s) on {dates} distinct date(s)",
            markup=False,
            highlight=False,
        )
    for (issuer_a, issuer_b), count in sorted(
        result.pairs_by_issuer_pair.items(), key=lambda kv: -kv[1]
    ):
        console.print(f"  {issuer_a} vs {issuer_b}: {count}", markup=False, highlight=False)
    if not result.pairs:
        console.print(
            "  [dim]no cross-issuer pair: the question is not measurable on this "
            "archive yet, and issuer coverage is the reason to check first[/dim]"
        )

    payload = {
        "pair_count": result.pair_count,
        "snapshots_stored": len(snapshots),
        "snapshots_considered": result.snapshots_considered,
        "dropped": result.dropped,
        "pairs_by_underlying": dict(result.pairs_by_underlying),
        "distinct_dates_by_underlying": dict(result.distinct_dates_by_underlying),
        "pairs_by_issuer_pair": {
            f"{a}|{b}": n for (a, b), n in result.pairs_by_issuer_pair.items()
        },
        "criteria": {
            "barrier_pct_tolerance": criteria.barrier_pct_tolerance,
            "leverage_tolerance": criteria.leverage_tolerance,
            "window_seconds": criteria.window_seconds,
            "require_verified_ratio": criteria.require_verified_ratio,
        },
    }
    if json_out:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"Wrote {json_path}")

    _write_summary("research_product_pairs", {"pairs": result.pair_count}, **payload)


def research_queue_cmd(
    ctx: typer.Context,
    limit: int = typer.Option(10, "--limit", help="How many entries to show"),
    rescore: bool = typer.Option(
        True, "--rescore/--no-rescore", help="Recompute priorities before printing"
    ),
) -> None:
    """Show the ranked research queue (Phase 2, §8).

    Priorities only. Nothing here authorises work: the system may reorder the
    queue as often as it likes, but only ``research approve`` can move an entry
    out of PROPOSED, and only a named human can run that.
    """
    app_ctx = ctx.obj
    now = datetime.now(UTC)

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        queue = ResearchQueue(
            store,
            failed_hypotheses_path=failed_hypotheses_default_path(app_ctx.state_dir),
        )
        seeded = queue.seed_from_catalog(now=now)
        if rescore:
            queue.rescore(now=now)
        entries = queue.ranked()
        # markup=False: the rendered text contains square brackets (the
        # "[priorities only -- implementation needs human approval]" header and
        # the unknown-input lists), which rich would otherwise parse as markup
        # tags and silently swallow -- losing exactly the line that says this
        # queue is not permission to act.
        console.print(render_research_queue(entries, limit=limit), markup=False, highlight=False)

    if seeded:
        console.print(
            f"\nSeeded {len(seeded)} new catalog entr{'y' if len(seeded) == 1 else 'ies'}."
        )
    _write_summary("research_queue", {"open": len(entries), "seeded": len(seeded)})


def research_approve_cmd(
    ctx: typer.Context,
    hypothesis_id: str = typer.Argument(..., help="Which opportunity to approve"),
    approved_by: str = typer.Option(..., "--by", help="Who is approving this (required)"),
    note: str = typer.Option(..., "--note", help="Why it is being approved (required)"),
    trial_id: str | None = typer.Option(
        None, "--trial-id", help="Existing trial id, if one has been registered"
    ),
) -> None:
    """Approve one research opportunity for implementation (§8).

    This is the human-approval gate. It is a CLI command rather than anything
    automatic on purpose: the system is allowed to argue for a question, never
    to authorise it. ``--by`` and ``--note`` are required because an approval
    with no named approver and no stated reason is not an audit trail.

    Approving does not start any work and changes no production code. It only
    records that a human considers this question worth running.
    """
    app_ctx = ctx.obj
    now = datetime.now(UTC)

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        queue = ResearchQueue(
            store,
            failed_hypotheses_path=failed_hypotheses_default_path(app_ctx.state_dir),
        )
        queue.seed_from_catalog(now=now)
        try:
            stored = queue.approve(
                hypothesis_id,
                approved_by=approved_by,
                note=note,
                trial_id=trial_id,
                now=now,
            )
        except (ValueError, IllegalTransition, KeyError) as exc:
            console.print(f"[red]Not approved:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        console.print(
            f"{stored.opportunity.hypothesis_id} -> "
            f"{stored.opportunity.status.value} (by {approved_by})"
        )
        if trial_id is None:
            console.print(
                "No trial id recorded. One is required before this can move to RUNNING "
                "(Master Spec §23/§24: every research change needs a trial id against "
                "the adjustment budget)."
            )
    _write_summary("research_approve", {"approved": 1}, hypothesis_id=hypothesis_id)


def research_tournament_cmd(
    ctx: typer.Context,
    email: bool = typer.Option(False, "--email", help="Email the tournament report"),
) -> None:
    """Run the weekly research tournament (Master Spec §36)."""
    app_ctx = ctx.obj
    now = datetime.now(UTC)

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        report = run_research_tournament(
            store, as_of=now, cfg=app_ctx.cfg.reporting.weekly_config()
        )
        json_path, html_path = save_weekly_report(report, reports_dir=Path("reports"))

        if email:
            from turboedge.reporting.html import to_plain_text

            notifier = _build_notifier(app_ctx)
            dedup = NotificationDeduplicator(store)
            body = to_plain_text(report)
            subject = f"TurboEdge Research Tournament — {report.window_end.isoformat()}"
            key_values = {"window_end": report.window_end.isoformat()}
            n_hash = notification_hash(None, "RESEARCH_TOURNAMENT", key_values)
            if dedup.should_send(n_hash):
                recipients = notifier.credentials.recipients if notifier.credentials else []
                spec = EmailMessageSpec(subject=subject, body_text=body, to=recipients)
                notifier.send(spec)
                dedup.mark_sent(n_hash, None, "RESEARCH_TOURNAMENT", subject, sent_at=now)

    counts = summary_counts(report)
    _print_counts("research tournament", counts)
    console.print(f"Wrote {json_path} / {html_path}")
    _write_summary("research_tournament", counts)


# --------------------------------------------------------------------------
# forecast (diagnostic)
# --------------------------------------------------------------------------


def forecast_cmd(
    ctx: typer.Context,
    underlying: str = typer.Option(..., "--underlying", help="Underlying id, e.g. DAX"),
) -> None:
    """Diagnostic: fit every default forecast model against one underlying
    and print its per-horizon forecast (no persistence, no EV/paths)."""
    app_ctx = ctx.obj
    underlying_id = resolve_underlying_id(underlying)
    if underlying_id is None:
        err_console.print(f"[red]Unknown underlying: {underlying!r}[/red]")
        raise typer.Exit(code=2)

    price_adapter = YFinancePriceAdapter()
    try:
        bars = price_adapter.fetch_daily_bars(underlying_id, lookback_days=800)
    except Exception as exc:
        err_console.print(f"[red]Could not fetch bars for {underlying_id}: {exc}[/red]")
        raise typer.Exit(code=3) from exc

    as_of = datetime.now(UTC)
    models: list[ForecastModel] = [TsmomForecastModel(), LogisticDirectionModel(), NullModel()]
    table = Table(title=f"Forecast diagnostic — {underlying_id}")
    table.add_column("Model")
    table.add_column("Horizon", justify="right")
    table.add_column("p_up", justify="right")
    table.add_column("mean", justify="right")
    table.add_column("sigma", justify="right")
    table.add_column("uncertainty", justify="right")

    fitted = 0
    for model in models:
        try:
            model.fit(bars, as_of)
            forecasts = model.predict(bars, as_of, horizons=app_ctx.cfg.forecast.horizons)
        except Exception as exc:
            logger.warning("forecast_cmd_model_failed", model_id=model.model_id, error=str(exc))
            continue
        fitted += 1
        for f in forecasts:
            table.add_row(
                model.model_id,
                f"{f.horizon_days}d",
                f"{f.p_up:.3f}",
                f"{f.mean:.5f}",
                f"{f.sigma:.5f}",
                f"{f.uncertainty:.5f}",
            )
    console.print(table)
    counts = {"models_fit": fitted, "models_total": len(models)}
    _write_summary("forecast", counts, underlying_id=underlying_id)


# --------------------------------------------------------------------------
# backtest (walk-forward, persisted)
# --------------------------------------------------------------------------


def _mean_oos_return(r: Any) -> float:
    vals = [
        v for v in (r.mean_oos_return_when_long, r.mean_oos_return_when_short) if not math.isnan(v)
    ]
    return float(np.mean(vals)) if vals else 0.0


def backtest_cmd(
    ctx: typer.Context,
    underlying: str = typer.Option(None, "--underlying", help="Restrict to one underlying id"),
) -> None:
    """Walk-forward evaluate every default forecast model against every
    active (or ``--underlying``-selected) underlying's history, persisting
    every ``(model, horizon)`` result to ``walkforward_results`` (Contract
    v3 coordinator addition -- makes W4's results readable by
    ``reporting.weekly.run_research_tournament`` once that module is wired
    to prefer this table; see Kurzbericht)."""
    app_ctx = ctx.obj
    if underlying:
        resolved = resolve_underlying_id(underlying)
        if resolved is None:
            err_console.print(f"[red]Unknown underlying: {underlying!r}[/red]")
            raise typer.Exit(code=2)
        ids = [resolved]
    else:
        ids = app_ctx.cfg.universe.enabled_ids()

    price_adapter = YFinancePriceAdapter()
    now = datetime.now(UTC)
    hash_value = config_hash(app_ctx.cfg)
    commit = git_commit()
    fcfg = app_ctx.cfg.forecast

    rows_written = 0
    table = Table(title="Backtest (walk-forward)")
    table.add_column("Underlying")
    table.add_column("Model")
    table.add_column("Horizon", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("Brier(null)", justify="right")
    table.add_column("Hit rate", justify="right")
    table.add_column("CRPS", justify="right")
    table.add_column("Cov90", justify="right")

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        for uid in ids:
            try:
                bars = price_adapter.fetch_daily_bars(uid, lookback_days=1500)
                store.append_underlying_bars(bars)
            except Exception as exc:
                logger.warning("backtest_bars_fetch_failed", underlying_id=uid, error=str(exc))
                bars = store.latest_underlying_bars(uid, 1500)
            if len(bars) < fcfg.min_train + fcfg.embargo + max(fcfg.horizons):
                logger.warning("backtest_insufficient_history", underlying_id=uid, n_bars=len(bars))
                continue

            try:
                null_results = walk_forward_evaluate(
                    NullModel,
                    bars,
                    fcfg.horizons,
                    min_train=fcfg.min_train,
                    step=fcfg.step,
                    embargo=fcfg.embargo,
                )
            except ValueError as exc:
                logger.warning("backtest_null_failed", underlying_id=uid, error=str(exc))
                continue
            null_brier = {r.horizon_days: r.brier for r in null_results}

            for model_id, factory in _DEFAULT_MODEL_FACTORIES.items():
                try:
                    results = walk_forward_evaluate(
                        factory,
                        bars,
                        fcfg.horizons,
                        min_train=fcfg.min_train,
                        step=fcfg.step,
                        embargo=fcfg.embargo,
                    )
                except ValueError as exc:
                    logger.warning(
                        "backtest_model_failed",
                        underlying_id=uid,
                        model_id=model_id,
                        error=str(exc),
                    )
                    continue
                signal_family = factory().signal_family
                records = [
                    WalkforwardResultRecord(
                        model_id=r.model_id,
                        model_hash=None,
                        signal_family=signal_family,
                        underlying_id=uid,
                        horizon_days=r.horizon_days,
                        evaluated_at=now,
                        n_folds=r.n_folds,
                        brier=r.brier,
                        brier_null=null_brier.get(r.horizon_days),
                        log_loss=r.log_loss,
                        ece=r.ece,
                        hit_rate=r.hit_rate,
                        mean_oos_return=_mean_oos_return(r),
                        psr=r.psr,
                        n_effective=float(len(r.oos_predictions)),
                        config_hash=hash_value,
                        git_commit=commit,
                        params={},
                        crps=r.crps,
                        pinball_loss=r.pinball_by_quantile,
                        coverage_90=r.coverage_90,
                        coverage_50=r.coverage_50,
                    )
                    for r in results
                ]
                rows_written += store.append_walkforward_results(records)
                for r in results:
                    table.add_row(
                        uid,
                        r.model_id,
                        f"{r.horizon_days}d",
                        f"{r.brier:.4f}",
                        f"{null_brier.get(r.horizon_days, float('nan')):.4f}",
                        f"{r.hit_rate:.3f}",
                        f"{r.crps:.5f}",
                        f"{r.coverage_90:.3f}",
                    )

    if redact_console_enabled():
        _print_counts("backtest", {"rows_written": rows_written})
    else:
        console.print(table)
    _write_summary("backtest", {"rows_written": rows_written}, underlyings=ids)


# --------------------------------------------------------------------------
# research backfill-trials
# --------------------------------------------------------------------------


def research_backfill_trials_cmd(
    ctx: typer.Context,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report which trial_ids would be inserted, without writing"
    ),
) -> None:
    """Backfill the six GOVERNANCE.md §11.1 W9 2026Q3 trials into
    ``research_trials`` (idempotent; a second run adds nothing)."""
    app_ctx = ctx.obj
    with Store(app_ctx.db_path) as store:
        store.init_schema()
        before = store.count_research_trials_in_quarter("2026Q3")
        inserted = backfill_w9_trials(store, dry_run=dry_run)
        after = before if dry_run else store.count_research_trials_in_quarter("2026Q3")

    counts: dict[str, Any] = {"trials_before": before}
    counts["trials_would_insert" if dry_run else "trials_inserted"] = len(inserted)
    counts["trials_after"] = after
    _print_counts("research backfill-trials", counts)
    if inserted:
        console.print(f"  {'would insert' if dry_run else 'inserted'}: {', '.join(inserted)}")
    _write_summary("research_backfill_trials", counts, dry_run=dry_run, trial_ids=inserted)


# --------------------------------------------------------------------------
# research ko-calibration
# --------------------------------------------------------------------------

# ~2010-2026 (matches the W4/W9 walk-forward window documented in
# docs/measured_results.md §1-2).
_KO_CALIBRATION_LOOKBACK_DAYS = 4200
# Every 5th business day is used as an "as of" bar -- see
# simulation/ko_calibration.py's own docstring on step_days: a documented,
# applied-before-any-result-is-seen stride, not post-hoc cherry-picking.
# Measured (~4200 bars, n_paths=2000): ~800 "as of" bars per underlying in a
# few seconds.
_KO_CALIBRATION_STEP_DAYS = 5
# Walk-forward min_train/step, expressed in "as of" bars (t0s); multiplied
# by the number of (direction, sigma_bucket) observations sharing each t0
# (2 directions x len(sigma_levels)) to get the actual sample-count
# min_train/step PurgedWalkForwardSplit expects.
_KO_CALIBRATION_MIN_TRAIN_T0 = 300
_KO_CALIBRATION_STEP_T0 = 100


def _ko_result_record(
    run_id: str,
    method: str,
    breakdown_dim: str,
    breakdown_value: str,
    m: KoCalibrationMetrics,
    evaluated_at: datetime,
    config_hash_value: str,
    commit: str | None,
) -> KoCalibrationResultRecord:
    return KoCalibrationResultRecord(
        run_id=run_id,
        method=method,
        breakdown_dim=breakdown_dim,
        breakdown_value=breakdown_value,
        n=m.n,
        brier=m.brier,
        calibration_intercept=None
        if math.isnan(m.calibration_intercept)
        else m.calibration_intercept,
        calibration_slope=None if math.isnan(m.calibration_slope) else m.calibration_slope,
        ece=m.ece,
        mean_signed_error=m.mean_signed_error,
        absolute_calibration_error=m.absolute_calibration_error,
        evaluated_at=evaluated_at,
        config_hash=config_hash_value,
        git_commit=commit,
    )


def research_ko_calibration_cmd(
    ctx: typer.Context,
    underlying: Annotated[
        list[str] | None,
        typer.Option("--underlying", help="Restrict to these underlying id(s) (repeatable)"),
    ] = None,
    json_out: str = typer.Option(None, "--json-out", help="Write full results JSON to PATH"),
) -> None:
    """Out-of-sample calibration of the path-simulation P(KO) (Workstream
    W10, docs/measured_results.md §3): builds the empirical dataset,
    walk-forward fits/evaluates every candidate calibrator (identity,
    isotonic, Platt) against raw P(KO), persists per-breakdown results and
    the promotion decision, and prints a summary. Measurement only --
    ranking/gates.py and pipeline/scan.py are not touched by this command,
    even when a calibrator is promoted."""
    app_ctx = ctx.obj
    if underlying:
        ids: list[str] = []
        for u in underlying:
            resolved = resolve_underlying_id(u)
            if resolved is None:
                err_console.print(f"[red]Unknown underlying: {u!r}[/red]")
                raise typer.Exit(code=2)
            ids.append(resolved)
    else:
        ids = app_ctx.cfg.universe.enabled_ids()

    price_adapter = YFinancePriceAdapter()
    now = datetime.now(UTC)
    hash_value = config_hash(app_ctx.cfg)
    commit = git_commit()
    sim_cfg = app_ctx.cfg.simulation
    run_id = f"ko-calibration-{now.strftime('%Y%m%dT%H%M%SZ')}"

    all_observations = []
    obs_per_underlying: dict[str, int] = {}
    with Store(app_ctx.db_path) as store:
        store.init_schema()
        for uid in ids:
            try:
                bars = price_adapter.fetch_daily_bars(
                    uid, lookback_days=_KO_CALIBRATION_LOOKBACK_DAYS
                )
                store.append_underlying_bars(bars)
            except Exception as exc:
                logger.warning(
                    "ko_calibration_bars_fetch_failed", underlying_id=uid, error=str(exc)
                )
                bars = store.latest_underlying_bars(uid, _KO_CALIBRATION_LOOKBACK_DAYS)
            rng = np.random.default_rng(sim_cfg.seed)
            try:
                obs = build_ko_calibration_dataset(
                    bars,
                    uid,
                    n_paths=sim_cfg.n_paths,
                    method=sim_cfg.method,
                    block_size=sim_cfg.block_size,
                    lookback_days=sim_cfg.lookback_days,
                    rng=rng,
                    step_days=_KO_CALIBRATION_STEP_DAYS,
                )
            except ValueError as exc:
                logger.warning("ko_calibration_dataset_failed", underlying_id=uid, error=str(exc))
                continue
            obs_per_underlying[uid] = len(obs)
            all_observations.extend(obs)

        if not all_observations:
            err_console.print(
                "[red]research ko-calibration: no usable observations for any underlying[/red]"
            )
            raise typer.Exit(code=3)

        n_directions = len({o.direction for o in all_observations})
        n_sigma = len({o.sigma_k for o in all_observations})
        obs_per_t0 = max(n_directions * n_sigma, 1)
        min_train = _KO_CALIBRATION_MIN_TRAIN_T0 * obs_per_t0
        step = _KO_CALIBRATION_STEP_T0 * obs_per_t0

        try:
            result = run_ko_calibration(all_observations, min_train=min_train, step=step)
        except ValueError as exc:
            err_console.print(f"[red]research ko-calibration: {exc}[/red]")
            raise typer.Exit(code=3) from exc

        method_results = {"raw": result.raw, **result.candidates}
        records: list[KoCalibrationResultRecord] = []
        for method, method_result in method_results.items():
            records.append(
                _ko_result_record(
                    run_id,
                    method,
                    "overall",
                    "overall",
                    method_result.overall,
                    now,
                    hash_value,
                    commit,
                )
            )
            for h, m in method_result.by_horizon.items():
                records.append(
                    _ko_result_record(run_id, method, "horizon", str(h), m, now, hash_value, commit)
                )
            for d, m in method_result.by_direction.items():
                records.append(
                    _ko_result_record(
                        run_id, method, "direction", str(d), m, now, hash_value, commit
                    )
                )
            for s, m in method_result.by_sigma_bucket.items():
                records.append(
                    _ko_result_record(
                        run_id, method, "sigma_bucket", str(s), m, now, hash_value, commit
                    )
                )
            for u, m in method_result.by_underlying.items():
                records.append(
                    _ko_result_record(
                        run_id, method, "underlying", str(u), m, now, hash_value, commit
                    )
                )
            for r, m in method_result.by_regime.items():
                records.append(
                    _ko_result_record(run_id, method, "regime", str(r), m, now, hash_value, commit)
                )
        rows_written = store.append_ko_calibration_results(records)
        store.insert_ko_calibration_promotion(
            KoCalibrationPromotionRecord(
                run_id=run_id,
                promoted_method=result.promoted_method,
                reason=result.promotion_reason,
                evaluated_at=now,
                config_hash=hash_value,
                git_commit=commit,
            )
        )

    if redact_console_enabled():
        pass
    else:
        table = Table(title="KO calibration (out-of-sample)")
        table.add_column("Method")
        table.add_column("n", justify="right")
        table.add_column("Brier", justify="right")
        table.add_column("Intercept", justify="right")
        table.add_column("Slope", justify="right")
        table.add_column("ECE", justify="right")
        table.add_column("MeanSignedErr", justify="right")
        table.add_column("AbsCalErr", justify="right")
        for method, method_result in method_results.items():
            m = method_result.overall
            table.add_row(
                method,
                str(m.n),
                f"{m.brier:.4f}",
                f"{m.calibration_intercept:.3f}",
                f"{m.calibration_slope:.3f}",
                f"{m.ece:.4f}",
                f"{m.mean_signed_error:+.4f}",
                f"{m.absolute_calibration_error:.4f}",
            )
        console.print(table)
        console.print(f"Promotion: {result.promoted_method or 'NONE'} -- {result.promotion_reason}")

    counts = {
        "underlyings": len(obs_per_underlying),
        "observations": len(all_observations),
        "ko_calibration_result_rows": rows_written,
        "promoted": 1 if result.promoted_method else 0,
    }
    _print_counts("research ko-calibration", counts)

    if json_out:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "run_id": run_id,
            "underlyings": ids,
            "obs_per_underlying": obs_per_underlying,
            "n_observations": len(all_observations),
            "promoted_method": result.promoted_method,
            "promotion_reason": result.promotion_reason,
            "methods": {
                method: {
                    "overall": asdict(method_result.overall),
                    "by_horizon": {str(k): asdict(v) for k, v in method_result.by_horizon.items()},
                    "by_direction": {
                        str(k): asdict(v) for k, v in method_result.by_direction.items()
                    },
                    "by_sigma_bucket": {
                        str(k): asdict(v) for k, v in method_result.by_sigma_bucket.items()
                    },
                    "by_underlying": {
                        str(k): asdict(v) for k, v in method_result.by_underlying.items()
                    },
                    "by_regime": {str(k): asdict(v) for k, v in method_result.by_regime.items()},
                }
                for method, method_result in method_results.items()
            },
        }
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2, default=str)

    _write_summary(
        "research_ko_calibration",
        counts,
        run_id=run_id,
        obs_per_underlying=obs_per_underlying,
        promoted_method=result.promoted_method,
        promotion_reason=result.promotion_reason,
    )


def archive_products_cmd(
    ctx: typer.Context,
    underlying: Annotated[
        list[str] | None,
        typer.Option(
            "--underlying",
            help=(
                "Underlying id(s) to archive (repeatable). Default: every "
                "enabled id from configs/universe.yaml"
            ),
        ),
    ] = None,
    max_pages: int = typer.Option(
        150,
        "--max-pages",
        help=(
            "Deep gettex page cap for this archive run only (default 150, "
            "~15,000 rows at 100 rows/page). Never affects the live scan "
            "path, which keeps its own configs/sources.yaml value."
        ),
    ),
) -> None:
    """Deep, slow, once-daily full-depth product fetch for research only.

    Decoupled from the decision path on purpose (see
    ``pipeline/archive.py`` module docstring): writes ``product_snapshots``/
    ``instruments`` at full depth, never touches pricing, EV, gates, the
    forward ledger or notifications. Belongs in the ``eod`` job, never one
    of the five daily ``scan-all`` runs.
    """
    app_ctx = ctx.obj
    underlying_ids = list(underlying) if underlying else app_ctx.cfg.universe.enabled_ids()
    if not underlying_ids:
        err_console.print(
            "[red]No underlyings enabled in configs/universe.yaml and none given via "
            "--underlying[/red]"
        )
        raise typer.Exit(code=2)

    product_adapters = build_product_adapters(app_ctx.cfg)

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        result = run_product_archive(
            app_ctx.cfg,
            store,
            product_adapters=product_adapters,
            underlying_ids=underlying_ids,
            max_pages=max_pages,
        )

    counts = {
        "snapshots_written": result.snapshots_written,
        "instruments_upserted": result.instruments_upserted,
        "issuers_seen": len(result.issuers_seen),
    }

    if redact_console_enabled():
        _print_counts("archive products", counts)
    else:
        table = Table(title="archive products -- issuer breadth")
        table.add_column("Issuer")
        table.add_column("Snapshots", justify="right")
        for issuer, n in sorted(result.issuers_seen.items(), key=lambda kv: -kv[1]):
            table.add_row(issuer, str(n))
        console.print(table)

        by_underlying = Table(title="archive products -- rows by underlying")
        by_underlying.add_column("Underlying")
        by_underlying.add_column("Rows", justify="right")
        for uid, n in result.rows_fetched_by_underlying.items():
            by_underlying.add_row(uid, str(n))
        console.print(by_underlying)

        console.print(
            f"Snapshots written: {result.snapshots_written}  "
            f"Instruments upserted: {result.instruments_upserted}  "
            f"Duration: {result.duration_s:.1f}s"
        )
        if len(result.issuers_seen) <= 1:
            console.print(
                "[yellow]Only one issuer this run -- the point of this command is "
                "issuer BREADTH, so this is a disappointing result, not a quiet "
                "success.[/yellow]"
            )
        if result.warnings:
            console.print(f"[yellow]Warnings: {', '.join(result.warnings)}[/yellow]")

    _write_summary(
        "archive_products",
        counts,
        underlyings=list(result.underlyings),
        issuers_seen=result.issuers_seen,
        rows_fetched_by_underlying=result.rows_fetched_by_underlying,
        warnings=result.warnings,
        duration_s=result.duration_s,
    )


def research_step_cmd(
    ctx: typer.Context,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Decide and print, but record nothing."),
    ] = False,
) -> None:
    """One cycle of the autonomous research loop: decide, run, record.

    Reads the research queue, the quarter's recorded trials and screens, the
    persisted shelf and the negative memory. Picks the highest-ranked
    admissible question, and when the decision is to screen it, executes the
    harness registered for it in `research/harness.py` -- never a procedure
    built on the spot. A question with no approved harness is recorded as
    such and the loop moves to the next one.

    Idempotent within a quarter: a re-run that reaches the same conclusion
    writes nothing new and re-runs no experiment, so a retried or
    interrupted scheduled run cannot inflate the research ledger -- which is
    the denominator the multiple-testing correction depends on.

    It never approves, never promotes and never starts a confirmatory
    trial. Those remain human-gated (`research approve`), and promotion is
    additionally blocked while `alpha/promotion.py` has unset gates and
    GOVERNANCE.md disagrees with configs/ranking.yaml about risk limits.
    """
    app_ctx = ctx.obj

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        # The notification path is wired here rather than left to a
        # separate command: a screen that finds something has to reach the
        # operator without anyone asking it to. Credentials absent means a
        # dry-run send, which records nothing and stays eligible.
        creds = GmailCredentials.from_env()
        action = run_research_step(
            store,
            failed_hypotheses_path=failed_hypotheses_default_path(app_ctx.state_dir),
            now=datetime.now(UTC),
            dry_run=dry_run,
            notifier=None if dry_run else GmailNotifier(app_ctx.cfg.gmail, creds),
            recipients=creds.recipients if creds else (),
            run_id=os.environ.get("GITHUB_RUN_ID"),
            run_url=(
                f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
                f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
                if {"GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"} <= os.environ.keys()
                else None
            ),
        )
        # Read back rather than threaded through the return value: the step
        # may record several screens in one run (walking past questions it
        # has no harness for) and the table is the record either way.
        screens = [] if dry_run else store.list_research_screens()

    console.print(f"[bold]{action.kind}[/bold]")
    if action.hypothesis_id:
        console.print(f"  hypothesis: {action.hypothesis_id}  (family: {action.family})")
    console.print(f"  reason: {action.reason}", markup=False, highlight=False)

    executed = sorted(screens, key=lambda s: s.started_at)[-3:]
    for screen in executed:
        console.print(
            f"  screen {screen.status.value}: {screen.hypothesis_id} via "
            f"{screen.harness}@{screen.harness_version} -> {screen.verdict or '-'} "
            f"(data {screen.data_version}, {screen.cost_seconds:.1f}s)",
            markup=False,
            highlight=False,
        )
        if screen.error:
            console.print(f"    {screen.error}", markup=False, highlight=False)
    console.print(
        "  [dim]this loop never approves, promotes, or starts a confirmatory "
        "trial -- those need a named human[/dim]"
    )
    _write_summary(
        "research_step",
        {"decided": 1, "recorded": 0 if dry_run else 1, "screens": len(executed)},
        kind=str(action.kind),
        hypothesis_id=action.hypothesis_id,
        reason=action.reason,
        evidence=action.evidence,
        screens=[
            {
                "hypothesis_id": s.hypothesis_id,
                "harness": f"{s.harness}@{s.harness_version}",
                "status": s.status.value,
                "verdict": s.verdict,
                "data_version": s.data_version,
                "cost_seconds": round(s.cost_seconds, 3),
                "error": s.error,
            }
            for s in executed
        ],
    )


def research_data_audit_cmd(
    ctx: typer.Context,
    underlying: str | None = typer.Option(
        None, "--underlying", help="Restrict the snapshot analysis to one underlying"
    ),
    since: str | None = typer.Option(
        None, "--since", help="Only snapshots at or after this ISO timestamp"
    ),
    window_seconds: int = typer.Option(
        60, "--window-seconds", help="Window within which products count as simultaneous"
    ),
    barrier_tolerance: float = typer.Option(
        0.10, "--barrier-tolerance", help="Relative barrier-distance tolerance"
    ),
    json_out: str = typer.Option(None, "--json-out", help="Write the full audit to PATH"),
) -> None:
    """Audit the forward dataset: what is missing, and how stale the quotes were.

    Answers two questions that come before any forward result means
    anything (`docs/data_audit_2026-10-25.md`):

    1. Is the forward sample representative? Every ledger row is classified
       into one of eight observation states, and the ones carrying a real
       exit are contrasted against the ones carrying an assumed value. A
       young ledger and a ledger losing data look identical in a bare
       "labeled=0" and are reported separately here.
    2. Were the quotes the decisions were made on actually current? Quote
       age needs no labels, so it is measurable even on a fully unlabeled
       ledger, and the association between apparent cheapness and quote age
       says whether a price advantage is partly just staleness.

    Read-only. Opens no trial, consumes no research budget, promotes
    nothing and writes nothing to the store.
    """
    app_ctx = ctx.obj
    since_dt = datetime.fromisoformat(since).astimezone(UTC) if since else None
    now = datetime.now(UTC)

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        rows = store.list_ledger_entries()
        snapshots = store.list_product_snapshots(underlying_id=underlying, since=since_dt)

    census = ledger_census(rows, now=now)
    freshness = quote_freshness([entry for entry, _ in rows])
    # Run the comparison both with and without the verified-ratio
    # requirement. The schema default is UNVERIFIED, so requiring it can
    # legitimately drop the entire archive -- and an empty result must not
    # be reportable as "no association found".
    strict = snapshot_staleness_advantage(
        snapshots,
        criteria=ComparabilityCriteria(
            barrier_pct_tolerance=barrier_tolerance,
            window_seconds=window_seconds,
            require_verified_ratio=True,
        ),
    )
    permissive = snapshot_staleness_advantage(
        snapshots,
        criteria=ComparabilityCriteria(
            barrier_pct_tolerance=barrier_tolerance,
            window_seconds=window_seconds,
            require_verified_ratio=False,
        ),
    )

    console.print(f"FORWARD POPULATION  {census.summary()}", markup=False, highlight=False)
    for state, count in sorted(census.by_state.items(), key=lambda kv: -kv[1]):
        console.print(f"  {state}: {count}", markup=False, highlight=False)
    console.print("  covariate contrasts (real exit vs assumed/missing):")
    for contrast in census.contrasts:
        if not contrast.comparable:
            console.print(
                f"    {contrast.name}: not comparable "
                f"(n={contrast.n_trustworthy}/{contrast.n_assumed})",
                markup=False,
                highlight=False,
            )
            continue
        console.print(
            f"    {contrast.name}: real {contrast.mean_trustworthy:.6f} vs "
            f"assumed {contrast.mean_assumed:.6f} (diff {contrast.difference:+.6f}, "
            f"n={contrast.n_trustworthy}/{contrast.n_assumed})",
            markup=False,
            highlight=False,
        )

    console.print(f"QUOTE AGE AT DECISION  {freshness.summary()}", markup=False, highlight=False)
    pit = point_in_time_audit([entry for entry, _ in rows])
    console.print(f"POINT-IN-TIME ORDERING  {pit.summary()}", markup=False, highlight=False)
    console.print(
        f"    leak suspected: {pit.leak_suspected}  "
        f"(a case beyond the scan window cannot be explained by ordering)",
        markup=False,
        highlight=False,
    )
    if pit.age_quantiles:
        console.print(
            "    prediction_time - quote_timestamp, seconds: "
            + "  ".join(f"{k}={v:+.1f}" for k, v in pit.age_quantiles.items()),
            markup=False,
            highlight=False,
        )
    for issuer, counts in sorted(pit.by_issuer.items()):
        console.print(f"    {issuer}: {counts}", markup=False, highlight=False)
    for label, report in (("verified-ratio only", strict), ("all ratios", permissive)):
        assoc = report.association
        if assoc.interpretable:
            verdict = f"spearman {assoc.spearman:+.4f}"
        elif (flat := assoc.degenerate_side) is not None:
            # A constant input is a data defect, not a null result.
            verdict = (
                f"n={assoc.n} but {flat} has no variation "
                f"(distinct ages {assoc.distinct_ages}, cheapness {assoc.distinct_advantages}) "
                "-- not answerable from this archive"
            )
        else:
            verdict = f"n={assoc.n} too small to read"
        console.print(
            f"STALENESS vs CHEAPNESS ({label})  {report.groups} group(s), pooled {verdict}",
            markup=False,
            highlight=False,
        )
        wg = report.within_group
        if wg.groups_used:
            console.print(
                f"    within-group mean {wg.mean:+.4f} "
                f"[{wg.ci_low:+.4f},{wg.ci_high:+.4f}] over {wg.groups_used} group(s), "
                f"{wg.members} member(s); excludes zero: {wg.excludes_zero}",
                markup=False,
                highlight=False,
            )
        else:
            console.print(
                "    within-group: no group had enough varying members to correlate",
                markup=False,
                highlight=False,
            )
        for reason, count in sorted(report.dropped.items(), key=lambda kv: -kv[1])[:6]:
            console.print(f"    dropped {reason}: {count}", markup=False, highlight=False)

    payload = {
        "generated_at": now.isoformat(),
        "forward_population": {
            "total": census.total,
            "by_state": dict(census.by_state),
            "trustworthy": census.trustworthy,
            "assumed": census.assumed,
            "missing": census.missing,
            "by_underlying": {k: dict(v) for k, v in census.by_underlying.items()},
            "by_issuer": {k: dict(v) for k, v in census.by_issuer.items()},
            "contrasts": [
                {
                    "name": c.name,
                    "n_trustworthy": c.n_trustworthy,
                    "n_assumed": c.n_assumed,
                    "mean_trustworthy": c.mean_trustworthy,
                    "mean_assumed": c.mean_assumed,
                    "comparable": c.comparable,
                }
                for c in census.contrasts
            ],
        },
        "point_in_time_ordering": {
            "n": pit.n,
            "by_ordering": dict(pit.by_ordering),
            "by_issuer": {k: dict(v) for k, v in pit.by_issuer.items()},
            "age_quantiles": dict(pit.age_quantiles),
            "min_age_seconds": pit.min_age_seconds,
            "max_age_seconds": pit.max_age_seconds,
            "beyond_window": pit.beyond_window,
            "leak_suspected": pit.leak_suspected,
            "plausible_scan_seconds": PLAUSIBLE_SCAN_SECONDS,
        },
        "quote_age_at_decision": {
            "n": freshness.n,
            "n_negative": freshness.n_negative,
            "median_seconds": freshness.median_seconds,
            "p90_seconds": freshness.p90_seconds,
            "max_seconds": freshness.max_seconds,
        },
        "staleness_vs_cheapness": {
            key: {
                "groups": report.groups,
                "n": report.association.n,
                "spearman": report.association.spearman,
                "interpretable": report.association.interpretable,
                "distinct_ages": report.association.distinct_ages,
                "distinct_advantages": report.association.distinct_advantages,
                "degenerate_side": report.association.degenerate_side,
                "within_group": {
                    "groups_used": report.within_group.groups_used,
                    "members": report.within_group.members,
                    "mean": report.within_group.mean,
                    "ci_low": report.within_group.ci_low,
                    "ci_high": report.within_group.ci_high,
                    "excludes_zero": report.within_group.excludes_zero,
                },
                "dropped": dict(report.dropped),
            }
            for key, report in (("verified_ratio_only", strict), ("all_ratios", permissive))
        },
        "snapshots_read": len(snapshots),
    }
    if json_out:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"  wrote {json_path}", markup=False, highlight=False)


def research_lottery_pricing_cmd(
    ctx: typer.Context,
    lookback_days: int = typer.Option(
        30, "--lookback-days", help="Window of archived quotes to read"
    ),
    json_out: str = typer.Option(None, "--json-out", help="Write the full result to PATH"),
) -> None:
    """Run the A02 lottery-pricing screen (Pi-CAPM) on the archived quotes.

    Do more lottery-like turbos -- those whose knock-out barrier sits
    closer to the spot -- carry systematically worse executable round-trip
    costs? Estimated within (issuer, underlying, direction, minute) cells,
    so issuer pricing style and the market moment are differenced out, and
    regressed on barrier distance rather than on an ask-derived leverage.

    WHY THIS IS A COMMAND AND NOT A LOOP STEP. The question's family is
    `product_selection`, which is deliberately absent from
    `controller.PERMITTED_EXPLORATORY_FAMILIES`. The autonomous loop may
    therefore not start it, and widening that set is a governance decision
    rather than a research one. The harness stays registered for the day
    the operator permits the family; until then the screen runs here, when
    a person asks for it.

    Read-only: computes no EV, opens no trial, promotes nothing and writes
    nothing to the store.
    """
    app_ctx = ctx.obj
    now = datetime.now(UTC)
    since = now - timedelta(days=lookback_days)

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        snapshots = store.list_product_snapshots(since=since, until=now)

    result = evaluate_lottery_pricing(snapshots)

    console.print(f"A02 LOTTERY PRICING  verdict {result.verdict}", markup=False, highlight=False)
    console.print(
        f"  {len(snapshots)} snapshot(s) -> {result.cells_used} cell(s), "
        f"{result.products_used} quote(s) used",
        markup=False,
        highlight=False,
    )
    console.print(
        f"  independence: {result.distinct_days} day(s), {result.distinct_minutes} minute(s), "
        f"{result.clusters_used} (day, issuer) cluster(s); needs "
        f"{MIN_INDEPENDENT_DAYS} day(s)",
        markup=False,
        highlight=False,
    )
    console.print(
        f"  spread slope {result.mean_spread_slope:+.6f} "
        f"[{result.spread_ci_low:+.6f}, {result.spread_ci_high:+.6f}]  "
        f"excludes zero: {result.interval_excludes_zero}",
        markup=False,
        highlight=False,
    )
    console.print(
        f"  premium slope {result.mean_premium_slope:+.6f} "
        f"[{result.premium_ci_low:+.6f}, {result.premium_ci_high:+.6f}]",
        markup=False,
        highlight=False,
    )
    console.print(
        f"  economic size: median {result.median_economic_size_bps:+.2f} bps, "
        f"mean {result.mean_economic_size_bps:+.2f} bps across the barrier range",
        markup=False,
        highlight=False,
    )
    console.print(
        f"  issuers: {sorted({c.issuer for c in result.cells})}", markup=False, highlight=False
    )
    for reason, count in sorted(result.dropped.items(), key=lambda kv: -kv[1])[:8]:
        console.print(f"    dropped {reason}: {count}", markup=False, highlight=False)

    payload = {
        "generated_at": now.isoformat(),
        "lookback_days": lookback_days,
        "verdict": result.verdict,
        "snapshots_read": len(snapshots),
        "cells_used": result.cells_used,
        "distinct_days": result.distinct_days,
        "distinct_minutes": result.distinct_minutes,
        "clusters_used": result.clusters_used,
        "products_used": result.products_used,
        "mean_spread_slope": result.mean_spread_slope,
        "spread_ci_low": result.spread_ci_low,
        "spread_ci_high": result.spread_ci_high,
        "interval_excludes_zero": result.interval_excludes_zero,
        "mean_premium_slope": result.mean_premium_slope,
        "premium_ci_low": result.premium_ci_low,
        "premium_ci_high": result.premium_ci_high,
        "mean_economic_size_bps": result.mean_economic_size_bps,
        "median_economic_size_bps": result.median_economic_size_bps,
        "issuers": sorted({c.issuer for c in result.cells}),
        "dropped": dict(result.dropped),
        "cells": [
            {
                "issuer": c.issuer,
                "underlying": c.underlying_id,
                "direction": str(c.direction),
                "minute": c.minute.isoformat(),
                "n_products": c.n_products,
                "spread_slope": c.spread_slope,
                "premium_slope": c.premium_slope,
                "economic_size_bps": c.economic_size_bps,
            }
            for c in result.cells[:200]
        ],
    }
    if json_out:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        console.print(f"  wrote {json_path}", markup=False, highlight=False)


def register_learn_commands(app: typer.Typer, position_app: typer.Typer) -> None:
    """Wire every Contract v3 integration-wave command into the main CLI.
    Called once from ``cli.py``::

        from turboedge.cli_learn import register_learn_commands
        register_learn_commands(app, position_app)
    """
    app.command("scan-all")(scan_all_cmd)
    app.command("label")(label_cmd)
    app.command("learn")(learn_cmd)
    app.command("forecast")(forecast_cmd)
    app.command("backtest")(backtest_cmd)
    position_app.command("reevaluate")(position_reevaluate_cmd)

    report_app = typer.Typer(help="Performance reports")
    report_app.command("monthly")(report_monthly_cmd)
    app.add_typer(report_app, name="report")

    research_app = typer.Typer(help="Research governance")
    research_app.command("queue")(research_queue_cmd)
    research_app.command("excursion")(research_excursion_cmd)
    research_app.command("synthetic-ev")(research_synthetic_ev_cmd)
    research_app.command("approve")(research_approve_cmd)
    research_app.command("tournament")(research_tournament_cmd)
    research_app.command("backfill-trials")(research_backfill_trials_cmd)
    research_app.command("ko-calibration")(research_ko_calibration_cmd)
    research_app.command("step")(research_step_cmd)
    research_app.command("product-pairs")(research_product_pairs_cmd)
    research_app.command("data-audit")(research_data_audit_cmd)
    research_app.command("lottery-pricing")(research_lottery_pricing_cmd)
    app.add_typer(research_app, name="research")

    archive_app = typer.Typer(help="Deep, slow, once-daily research-only product archive")
    archive_app.command("products")(archive_products_cmd)
    app.add_typer(archive_app, name="archive")


__all__ = ["register_learn_commands"]
