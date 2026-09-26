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
import re
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any

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
from turboedge.backtest.ko_calibration import KoCalibrationMetrics, run_ko_calibration
from turboedge.backtest.walkforward import walk_forward_evaluate
from turboedge.config import config_hash
from turboedge.learning.drift import PageHinkley, PageHinkleyConfig, record_drift_event
from turboedge.learning.failed_hypotheses import default_path as failed_hypotheses_default_path
from turboedge.learning.labeler import label_due_entries
from turboedge.learning.ledger import ForwardLedger
from turboedge.learning.posterior import StrategyPosterior
from turboedge.learning.registry import ModelRegistry
from turboedge.learning.trials import backfill_w9_trials
from turboedge.meta import IllegalTransition, ResearchQueue, render_research_queue
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
from turboedge.pipeline.scan_all import run_scan_all
from turboedge.positions.reevaluate import reevaluate_open_positions
from turboedge.provenance import git_commit
from turboedge.reporting.html import save_monthly_report, save_weekly_report, summary_counts
from turboedge.reporting.monthly import build_monthly_report
from turboedge.reporting.redaction import redact_console_enabled
from turboedge.reporting.weekly import run_research_tournament
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

    counts = {
        "labeled": result.labeled,
        "ko": result.ko,
        "ambiguous": result.ambiguous,
        "missing_data": result.missing_data,
        "shadow_positions_labeled": result.shadow_positions_labeled,
    }
    if redact_console_enabled():
        _print_counts("label", counts)
    else:
        _print_counts("label", counts)
    _write_summary("label", counts)


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
    research_app.command("approve")(research_approve_cmd)
    research_app.command("tournament")(research_tournament_cmd)
    research_app.command("backfill-trials")(research_backfill_trials_cmd)
    research_app.command("ko-calibration")(research_ko_calibration_cmd)
    app.add_typer(research_app, name="research")


__all__ = ["register_learn_commands"]
