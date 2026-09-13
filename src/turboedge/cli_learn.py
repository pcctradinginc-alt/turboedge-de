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
from turboedge.backtest.walkforward import walk_forward_evaluate
from turboedge.config import config_hash
from turboedge.learning.drift import PageHinkley, PageHinkleyConfig, record_drift_event
from turboedge.learning.labeler import label_due_entries
from turboedge.learning.ledger import ForwardLedger
from turboedge.learning.posterior import StrategyPosterior
from turboedge.learning.registry import ModelRegistry
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
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import LedgerEntryStatus, WalkforwardResultRecord
from turboedge.universe.underlying_map import resolve_underlying_id

logger = structlog.get_logger(__name__)

console = Console()
err_console = Console(stderr=True)

_SIGNAL_FAMILY_VERSION_RE = re.compile(r"^(?P<family>.+)_v\d+$")
_DEFAULT_MODEL_FACTORIES: dict[str, Any] = {
    "tsmom_forecast_v1": TsmomForecastModel,
    "logistic_direction_v1": LogisticDirectionModel,
    "null_v1": NullModel,
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
                    )

    if redact_console_enabled():
        _print_counts("backtest", {"rows_written": rows_written})
    else:
        console.print(table)
    _write_summary("backtest", {"rows_written": rows_written}, underlyings=ids)


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
    research_app.command("tournament")(research_tournament_cmd)
    app.add_typer(research_app, name="research")


__all__ = ["register_learn_commands"]
