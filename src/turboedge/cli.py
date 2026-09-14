"""TurboEdge-DE command-line interface.

typer-based CLI with subcommands for sources health, notifications, positions,
and database queries. Configuration and state directories are configurable via
environment variables and CLI flags.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import structlog
import typer
from rich.console import Console
from rich.table import Table

from turboedge import __version__
from turboedge.adapters.base import HttpClient
from turboedge.adapters.ecb import EcbEstrAdapter
from turboedge.adapters.fallback_prices import YFinancePriceAdapter
from turboedge.adapters.registry import (
    build_http_client,
    build_product_adapters,
    build_reference_healthchecks,
)
from turboedge.cli_learn import register_learn_commands
from turboedge.cli_state import register_state_commands
from turboedge.config import ConfigError, config_hash, load_config
from turboedge.logging import configure_logging
from turboedge.monitoring.source_health import (
    OPTIONAL_SOURCES,
    critical_failures,
    from_healthcheck,
    overall_status,
)
from turboedge.notifications.dedup import NotificationDeduplicator, notification_hash
from turboedge.notifications.gmail import GmailCredentials, GmailNotifier
from turboedge.pipeline.scan import ScanOptions, run_scan
from turboedge.pipeline.universe import NoProductsError, run_universe
from turboedge.positions.ledger import PositionLedger
from turboedge.provenance import git_commit, new_run_id
from turboedge.reporting.console import (
    render_scan,
    render_universe,
    scan_report_text,
    scan_result_to_json,
)
from turboedge.reporting.redaction import redact_console_enabled
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import Direction, HealthStatus, SourceHealthRecord
from turboedge.universe.underlying_map import resolve_underlying_id

logger = structlog.get_logger(__name__)

app = typer.Typer(no_args_is_help=True)
console = Console()
# `stderr=True` (not `file=sys.stderr`) so the Console resolves `sys.stderr`
# lazily on each write. `typer.testing.CliRunner` swaps `sys.stderr` for an
# isolated stream per-invocation; binding to the original stream object at
# import time would make error output invisible to CliRunner-based tests.
err_console = Console(stderr=True)

_FOOTER = "Research system — manual execution only."
_HORIZON_TO_DAYS: dict[str, int] = {"3d": 3, "5d": 5, "7d": 7, "10d": 10, "14d": 14}
# Placeholder recipient used only when GmailCredentials.from_env() returns
# None (dry-run: no GMAIL_USER/GMAIL_APP_PASSWORD/TURBOEDGE_EMAIL_TO set).
# Must never be `cfg.gmail.subject_prefix` -- that was a copy/paste bug that
# made dry-run "notification_dry_run" log lines report the subject prefix
# (e.g. "[TurboEdge-DE]") as the recipient list instead of an actual address.
_DRY_RUN_FALLBACK_RECIPIENT = "no-recipients-configured@example.invalid"


class AppContext:
    """Application context passed through CLI commands."""

    def __init__(self, cfg_dir: Path, state_dir: Path, log_fmt: str | None = None) -> None:

        self.config_dir = cfg_dir
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "turboedge.duckdb"

        configure_logging(fmt=log_fmt)  # type: ignore
        self.cfg = load_config(self.config_dir)
        self.config_hash = config_hash(self.cfg)
        self.git_commit = git_commit()


def _load_app_context(ctx: typer.Context) -> AppContext:
    """Extract AppContext from typer Context."""
    return ctx.obj


@app.callback()
def main(
    ctx: typer.Context,
    config_dir: str = typer.Option(
        "./configs",
        "--config-dir",
        envvar="TURBOEDGE_CONFIG_DIR",
        help="Path to configs directory",
    ),
    state_dir: str = typer.Option(
        "./state",
        "--state-dir",
        envvar="TURBOEDGE_STATE_DIR",
        help="Path to state directory",
    ),
    log_format: str | None = typer.Option(
        None,
        "--log-format",
        envvar="TURBOEDGE_LOG_FORMAT",
        help="Log format: console or json",
    ),
) -> None:
    """TurboEdge-DE: Research system for German turbocertificates.

    Environment variables:
      TURBOEDGE_CONFIG_DIR: Path to configs directory
      TURBOEDGE_STATE_DIR: Path to state directory
      TURBOEDGE_LOG_FORMAT: Log format (console or json)
      GMAIL_USER: Gmail account for notifications
      GMAIL_APP_PASSWORD: Gmail app-specific password
      TURBOEDGE_EMAIL_TO: Recipient email(s, comma-separated)
    """
    try:
        app_ctx = AppContext(Path(config_dir), Path(state_dir), log_fmt=log_format)
        ctx.obj = app_ctx
    except ConfigError as exc:
        err_console.print(f"[red]Configuration error: {exc}[/red]")
        raise typer.Exit(code=2)  # noqa: B904


# --- universe command ---


@app.command("universe")
def universe_cmd(
    ctx: typer.Context,
    # `Annotated` (rather than the plain `= typer.Option(...)` default used
    # elsewhere in this file) so ruff's B008 does not flag the call: with a
    # mutable `list[str]` annotation, B008 fires on a call used directly as
    # the default value (unlike `str`/`bool`/`int` params, whose immutable
    # annotations already exempt them) -- moving the `typer.Option(...)`
    # call into the annotation's metadata sidesteps that entirely.
    underlying: Annotated[
        list[str] | None,
        typer.Option(
            "--underlying",
            help=(
                "Underlying id(s) to fetch (repeatable). Default: the enabled "
                "ids from configs/universe.yaml"
            ),
        ),
    ] = None,
    source: str = typer.Option(
        "all",
        "--source",
        help="Product source to use: 'all' or a single adapter name (e.g. 'csv_import')",
    ),
) -> None:
    """Fetch the product universe from every enabled source, merge, dedupe, persist."""
    app_ctx = _load_app_context(ctx)

    underlying_ids = list(underlying) if underlying else app_ctx.cfg.universe.enabled_ids()
    if not underlying_ids:
        err_console.print(
            "[red]No underlyings enabled in configs/universe.yaml and none given via "
            "--underlying[/red]"
        )
        raise typer.Exit(code=2)

    try:
        product_adapters = build_product_adapters(
            app_ctx.cfg, only=None if source == "all" else source
        )
    except ValueError as exc:
        err_console.print(f"[red]Configuration error: {exc}[/red]")
        raise typer.Exit(code=2)  # noqa: B904

    run_id = new_run_id()
    with Store(app_ctx.db_path) as store:
        store.init_schema()
        try:
            result = run_universe(
                app_ctx.cfg,
                store,
                app_ctx.state_dir,
                product_adapters,
                underlying_ids,
                run_id=run_id,
            )
        except NoProductsError as exc:
            err_console.print("[red]No products retrieved from any source.[/red]")
            for name, err in sorted(exc.source_errors.items()):
                err_console.print(f"  - {name}: {err}")
            raise typer.Exit(code=3)  # noqa: B904

        if redact_console_enabled():
            console.print(
                f"[bold]Universe run[/bold] {result.run_id}  "
                f"merged={len(result.products)} conflicts={len(result.conflicts)} "
                "(TURBOEDGE_PUBLIC_LOGS=1: candidate detail redacted from console)"
            )
        else:
            render_universe(result, console)


# --- scan command ---


def _write_scan_failure_reports(
    *,
    run_id: str,
    underlying_id: str,
    source_errors: dict[str, str],
    health: list[SourceHealthRecord],
    json_out: str | None,
    report_out: str | None,
) -> None:
    """Write --json-out/--report-out even when every product source failed.

    NoProductsError aborts run_scan() before any ScanResult exists, so this
    builds the minimal equivalent by hand from what IS available at that
    point: the per-source errors and whatever source health was collected in
    a pre-flight pass -- CI artifacts stay meaningful (source of the failure
    is inspectable) instead of the run leaving no report files at all.
    """
    if json_out:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "underlying_id": underlying_id,
            "signal": None,
            "candidates": [],
            "counts": {},
            "warnings": [
                f"product_source_failed:{name}:{err}" for name, err in sorted(source_errors.items())
            ],
            "health": [h.model_dump(mode="json") for h in health],
            "notification": None,
            "error": "no products from any source (all sources failed)",
            "source_errors": dict(sorted(source_errors.items())),
            "footer": _FOOTER,
        }
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2, default=str)
        console.print(f"[blue]Wrote JSON report to {json_out}[/blue]")

    if report_out:
        report_path = Path(report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "TurboEdge-DE Scan Report -- FAILED",
            f"Run ID: {run_id}",
            f"Underlying: {underlying_id}",
            "",
            "All product sources failed:",
        ]
        for name, err in sorted(source_errors.items()):
            lines.append(f"  - {name}: {err}")
        lines.append("")
        lines.append("Source health:")
        if health:
            for h in health:
                lines.append(f"  - {h.source}: {h.status.value} ({h.message})")
        else:
            lines.append("  (none collected)")
        lines.append("")
        lines.append(_FOOTER)
        report_path.write_text("\n".join(lines) + "\n")
        console.print(f"[blue]Wrote report to {report_out}[/blue]")


@app.command("scan")
def scan_cmd(
    ctx: typer.Context,
    underlying: str = typer.Option(..., "--underlying", help="Underlying id, e.g. DAX"),
    direction: str = typer.Option(
        None, "--direction", help="Restrict candidates to 'long' or 'short'"
    ),
    horizon: str = typer.Option(
        "7d", "--horizon", help="Cost horizon: one of 3d, 5d, 7d, 10d, 14d"
    ),
    top: int = typer.Option(20, "--top", help="Number of top candidates to show/report"),
    email: bool = typer.Option(
        False, "--email", help="Send a deduplicated scan report by email (dry-run without creds)"
    ),
    report_out: str = typer.Option(
        None, "--report-out", help="Write the plain-text scan report to PATH"
    ),
    json_out: str = typer.Option(None, "--json-out", help="Write the JSON scan report to PATH"),
) -> None:
    """Scan one underlying's turbo/knockout universe for cost-ranked candidates.

    ACTIONABLE is technically reachable; in practice, no ACTIONABLE candidate
    is produced today because no forecast model has a measured out-of-sample
    advantage over the null model (see docs/measured_results.md). Exit codes:
    0 ok (even with 0 candidates), 2 invalid --underlying/--horizon/
    --direction or adapter configuration, 3 every product source failed
    (NoProductsError) -- in that case --json-out/--report-out are still
    written with whatever source-health and per-source-error detail is
    available, so CI artifacts stay meaningful.
    """
    app_ctx = _load_app_context(ctx)

    underlying_id = resolve_underlying_id(underlying)
    if underlying_id is None:
        err_console.print(f"[red]Unknown underlying: {underlying!r}[/red]")
        raise typer.Exit(code=2)

    horizon_days = _HORIZON_TO_DAYS.get(horizon)
    if horizon_days is None:
        err_console.print(
            f"[red]Invalid --horizon: {horizon!r}; must be one of "
            f"{', '.join(sorted(_HORIZON_TO_DAYS))}[/red]"
        )
        raise typer.Exit(code=2)

    direction_value: Direction | None = None
    if direction is not None:
        try:
            direction_value = Direction(direction)
        except ValueError:
            err_console.print(
                f"[red]Invalid --direction: {direction!r}; must be 'long' or 'short'[/red]"
            )
            raise typer.Exit(code=2)  # noqa: B904

    try:
        product_adapters = build_product_adapters(app_ctx.cfg)
    except ValueError as exc:
        err_console.print(f"[red]Configuration error: {exc}[/red]")
        raise typer.Exit(code=2)  # noqa: B904

    price_adapter = YFinancePriceAdapter()

    ecb_source_cfg = app_ctx.cfg.sources.get("ecb")
    ecb_http = (
        build_http_client(ecb_source_cfg)
        if ecb_source_cfg is not None
        else HttpClient(
            user_agent=app_ctx.cfg.default.http_default_user_agent,
            timeout_s=app_ctx.cfg.default.http_default_timeout_s,
            min_interval_s=0.0,
            max_retries=3,
        )
    )
    estr_adapter = EcbEstrAdapter(ecb_http, fallback_rate=app_ctx.cfg.risk.reference_rate_fallback)
    reference_checks = build_reference_healthchecks(app_ctx.cfg)

    notifier: GmailNotifier | None = None
    if email:
        creds = GmailCredentials.from_env()
        notifier = GmailNotifier(app_ctx.cfg.gmail, creds)

    # Pre-flight health snapshot, used only to enrich the failure-path report
    # below if every product source fails; run_scan() performs its own
    # authoritative health check internally regardless (persisted either
    # way).
    preflight_health: list[SourceHealthRecord] = []
    for adapter in product_adapters:
        try:
            preflight_health.append(from_healthcheck(adapter.healthcheck()))
        except Exception as exc:  # a broken healthcheck must not block reporting
            logger.warning("preflight_healthcheck_failed", adapter=adapter.name, error=str(exc))
    for check in reference_checks:
        try:
            preflight_health.append(from_healthcheck(check()))
        except Exception as exc:
            logger.warning("preflight_reference_healthcheck_failed", error=str(exc))

    run_id = new_run_id()
    with Store(app_ctx.db_path) as store:
        store.init_schema()
        options = ScanOptions(
            underlying_id=underlying_id,
            direction=direction_value,
            horizon_days=horizon_days,
            top=top,
            email=email,
        )
        try:
            result = run_scan(
                app_ctx.cfg,
                store,
                app_ctx.state_dir,
                options=options,
                product_adapters=product_adapters,
                price_adapter=price_adapter,
                estr_adapter=estr_adapter,
                reference_healthchecks=reference_checks,
                notifier=notifier,
                run_id=run_id,
            )
        except NoProductsError as exc:
            err_console.print("[red]Scan aborted: no products retrieved from any source.[/red]")
            for name, err in sorted(exc.source_errors.items()):
                err_console.print(f"  - {name}: {err}")
            _write_scan_failure_reports(
                run_id=run_id,
                underlying_id=underlying_id,
                source_errors=exc.source_errors,
                health=preflight_health,
                json_out=json_out,
                report_out=report_out,
            )
            raise typer.Exit(code=3)  # noqa: B904

        if redact_console_enabled():
            counts_line = "  ".join(f"{cat.value}={count}" for cat, count in result.counts.items())
            console.print(
                f"[bold]Candidates[/bold]  {counts_line}  "
                "(TURBOEDGE_PUBLIC_LOGS=1: candidate detail redacted from console -- see email)"
            )
        else:
            render_scan(result, console, top)

        if json_out:
            json_path = Path(json_out)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w") as f:
                json.dump(scan_result_to_json(result), f, indent=2, default=str)
            console.print(f"[blue]Wrote JSON report to {json_out}[/blue]")

        if report_out:
            report_path = Path(report_out)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(scan_report_text(result, top))
            console.print(f"[blue]Wrote report to {report_out}[/blue]")


# --- sources subcommand ---


sources_app = typer.Typer(help="Source data management and health")
app.add_typer(sources_app, name="sources")


@sources_app.command("health")
def sources_health(
    ctx: typer.Context,
    json_out: str = typer.Option(
        None,
        "--json-out",
        help="Write JSON report to file",
    ),
    email_on_fail: bool = typer.Option(
        False,
        "--email-on-fail",
        help="Send email alert if any source fails",
    ),
    fail_on_error: bool = typer.Option(
        False,
        "--fail-on-error",
        help="Exit with code 4 if any source fails",
    ),
) -> None:
    """Check health of all enabled data sources."""
    app_ctx = _load_app_context(ctx)

    run_id = new_run_id()
    with Store(app_ctx.db_path) as store:
        store.init_schema()
        store.start_run(
            run_id,
            command="sources health",
            config_hash=app_ctx.config_hash,
            git_commit=app_ctx.git_commit,
        )

        # Build healthchecks
        product_adapters = build_product_adapters(app_ctx.cfg)
        reference_checks = build_reference_healthchecks(app_ctx.cfg)

        if not product_adapters and not reference_checks:
            msg = (
                "[yellow]WARNING: No product adapters registered and no "
                "reference checks available[/yellow]"
            )
            err_console.print(msg)

        health_records: list[SourceHealthRecord] = []

        if not product_adapters:
            # No product source can be assessed at all -- overall_status()
            # over an empty/product-less record list would otherwise report
            # a silent PASS (Build Contract Task 1 review finding). A
            # synthetic WARN record forces the aggregate status to at least
            # WARN so this is never mistaken for "all sources healthy".
            health_records.append(
                SourceHealthRecord(
                    source="product_adapters",
                    checked_at=datetime.now(UTC),
                    availability=0.0,
                    freshness=0.0,
                    missingness=1.0,
                    schema_consistency=0.0,
                    cross_source_agreement=None,
                    score=0.0,
                    status=HealthStatus.WARN,
                    message=(
                        "No product adapters registered or enabled; "
                        "product-source health cannot be assessed."
                    ),
                )
            )

        # Run product adapter healthchecks
        for adapter in product_adapters:
            try:
                result = adapter.healthcheck()
                record = from_healthcheck(result)
                health_records.append(record)
            except Exception as exc:
                logger.error(
                    "healthcheck_exception",
                    adapter=adapter.name,
                    error=str(exc),
                )
                # Convert to FAIL record
                record = SourceHealthRecord(
                    source=adapter.name,
                    checked_at=datetime.now(UTC),
                    availability=0.0,
                    freshness=0.0,
                    missingness=0.0,
                    schema_consistency=0.0,
                    cross_source_agreement=None,
                    score=0.0,
                    status=HealthStatus.FAIL,
                    message=f"Exception: {exc}",
                )
                health_records.append(record)

        # Run reference rate healthchecks
        for check_fn in reference_checks:
            try:
                result = check_fn()
                record = from_healthcheck(result)
                health_records.append(record)
            except Exception as exc:
                logger.error(
                    "reference_healthcheck_exception",
                    error=str(exc),
                )
                record = SourceHealthRecord(
                    source="reference_rate",
                    checked_at=datetime.now(UTC),
                    availability=0.0,
                    freshness=0.0,
                    missingness=0.0,
                    schema_consistency=0.0,
                    cross_source_agreement=None,
                    score=0.0,
                    status=HealthStatus.FAIL,
                    message=f"Exception: {exc}",
                )
                health_records.append(record)

        # Persist
        store.append_source_health(health_records)

        # Display table
        table = Table(title="Source Health Report")
        table.add_column("Source", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Availability", justify="right")
        table.add_column("Freshness", justify="right")
        table.add_column("Message", style="green")

        for record in health_records:
            status_color = (
                "green"
                if record.status == HealthStatus.PASS
                else "yellow"
                if record.status == HealthStatus.WARN
                else "red"
            )
            table.add_row(
                record.source,
                f"[{status_color}]{record.status.value}[/{status_color}]",
                f"{record.availability:.2%}",
                f"{record.freshness:.2%}",
                record.message,
            )

        console.print(table)

        # JSON output
        if json_out:
            json_path = Path(json_out)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w") as f:
                json.dump(
                    [record.model_dump(mode="json") for record in health_records],
                    f,
                    indent=2,
                    default=str,
                )
            console.print(f"[blue]Wrote JSON report to {json_out}[/blue]")

        # Determine overall status
        overall = overall_status(health_records)
        store.finish_run(
            run_id,
            status="success" if overall == HealthStatus.PASS else "degraded",
        )

        # Critical sources: every enabled product adapter except the
        # OPTIONAL_SOURCES (currently just csv_import, a user-curated
        # manual-fallback source whose everyday state is "nothing
        # uploaded"), plus the two reference-rate sources scan pricing
        # depends on. A FAIL from a non-critical/optional source must never
        # trigger the email alert or --fail-on-error exit -- that was the
        # cause of the daily false-alarm alert (csv_import FAIL on an empty
        # CI runner). overall/overall_status above is still computed and
        # persisted/displayed unchanged (used for the run's stored
        # "success"/"degraded" status and the console table), only the
        # alert/exit-code gating below is narrowed to critical sources.
        critical_sources = {a.name for a in product_adapters if a.name not in OPTIONAL_SOURCES} | {
            "yfinance",
            "ecb_estr",
        }
        crit_fails = critical_failures(health_records, critical_sources)

        # Email on failure -- only for a critical-source FAIL, deduplicated
        # per calendar day (UTC) so re-running the workflow, or a persisting
        # failure across consecutive scheduled runs on the same day, does
        # not re-send the identical alert.
        if email_on_fail and crit_fails:
            today = datetime.now(UTC).date().isoformat()
            failing_sources = sorted(r.source for r in crit_fails)
            dedup = NotificationDeduplicator(store)
            n_hash = notification_hash(
                None,
                "SOURCE_HEALTH_ALERT",
                {"date": today, "sources": ",".join(failing_sources)},
            )
            if not dedup.should_send(n_hash):
                console.print(
                    "[yellow]Source health alert already sent today for "
                    f"{', '.join(failing_sources)}; skipping duplicate[/yellow]"
                )
            else:
                creds = GmailCredentials.from_env()
                notifier = GmailNotifier(app_ctx.cfg.gmail, creds)
                spec_body = (
                    f"Source health check: {overall.value}\n"
                    f"Critical source failure(s): {', '.join(failing_sources)}\n\n"
                    "Sources:\n"
                )
                for record in health_records:
                    spec_body += f"  {record.source}: {record.status.value} ({record.message})\n"
                from turboedge.notifications.gmail import EmailMessageSpec

                spec = EmailMessageSpec(
                    subject="TurboEdge Source Health Alert",
                    body_text=spec_body,
                    to=creds.recipients if creds else [_DRY_RUN_FALLBACK_RECIPIENT],
                )
                try:
                    send_result = notifier.send(spec)
                    if send_result.sent:
                        console.print("[green]Email sent[/green]")
                    else:
                        console.print(f"[yellow]Dry-run: {send_result.message}[/yellow]")
                    dedup.mark_sent(
                        n_hash, None, "SOURCE_HEALTH_ALERT", spec.subject, sent_at=datetime.now(UTC)
                    )
                except Exception as exc:
                    err_console.print(f"[red]Failed to send email: {exc}[/red]")
                    raise typer.Exit(code=1)  # noqa: B904

        # Exit code -- only for a critical-source FAIL (see above).
        if fail_on_error and crit_fails:
            raise typer.Exit(code=4)


# --- notify subcommand ---


notify_app = typer.Typer(help="Send notifications")
app.add_typer(notify_app, name="notify")


@notify_app.command("test")
def notify_test(ctx: typer.Context) -> None:
    """Send a test notification."""
    app_ctx = _load_app_context(ctx)

    from turboedge.notifications.gmail import EmailMessageSpec, GmailNotifier
    from turboedge.notifications.templates import render_test_email

    creds = GmailCredentials.from_env()
    notifier = GmailNotifier(app_ctx.cfg.gmail, creds)

    subject, body = render_test_email(datetime.now(UTC), __version__)
    spec = EmailMessageSpec(
        subject=subject,
        body_text=body,
        to=creds.recipients if creds else ["test@example.com"],
    )

    try:
        result = notifier.send(spec)
    except Exception as exc:
        err_console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[blue]Test notification: {result.message}[/blue]")
    if not result.sent:
        console.print("[yellow]Dry-run mode[/yellow]")
    raise typer.Exit(code=0)


# --- position subcommand ---


position_app = typer.Typer(help="Manual position ledger")
app.add_typer(position_app, name="position")


@position_app.command("add")
def position_add(
    ctx: typer.Context,
    wkn: str = typer.Option(..., help="Warrant Kennnummer (6 alphanumeric)"),
    qty: float = typer.Option(..., help="Quantity (must be > 0)"),
    price: float = typer.Option(..., help="Entry price (must be > 0)"),
    date_str: str = typer.Option(..., "--date", help="Entry date (YYYY-MM-DD)"),
    isin: str = typer.Option(None, help="Optional ISIN"),
) -> None:
    """Add a new position."""
    app_ctx = _load_app_context(ctx)

    try:
        from datetime import date

        entry_date = date.fromisoformat(date_str)
        with Store(app_ctx.db_path) as store:
            store.init_schema()
            ledger = PositionLedger(store)
            position = ledger.add(wkn, qty, price, entry_date, isin=isin)
            console.print(
                f"[green]Position added: {position.position_id} ({wkn} x{qty} @ {price})[/green]"
            )
    except ValueError as exc:
        err_console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=2)  # noqa: B904


@position_app.command("list")
def position_list(
    ctx: typer.Context,
    status: str = typer.Option(
        None,
        "--status",
        help="Filter: open or closed",
    ),
) -> None:
    """List positions."""
    app_ctx = _load_app_context(ctx)

    from turboedge.storage.schemas import PositionStatus

    try:
        filter_status = None
        if status:
            filter_status = PositionStatus(status)

        with Store(app_ctx.db_path) as store:
            store.init_schema()
            ledger = PositionLedger(store)
            positions = ledger.list(status=filter_status)

            if not positions:
                console.print("[yellow]No positions[/yellow]")
                return

            table = Table(title="Positions")
            table.add_column("WKN", style="cyan")
            table.add_column("Qty", justify="right")
            table.add_column("Entry Price", justify="right")
            table.add_column("Entry Date")
            table.add_column("Status", style="magenta")
            table.add_column("Exit Price", justify="right")
            table.add_column("Exit Date")
            table.add_column("Realized P&L", justify="right")
            table.add_column("Return", justify="right")

            for pos in positions:
                pnl = ledger.realized_pnl(pos)
                ret = ledger.realized_return(pos)
                pnl_str = f"{pnl:.2f}" if pnl is not None else "n/a"
                ret_str = f"{ret * 100:.2f}%" if ret is not None else "n/a"

                table.add_row(
                    pos.wkn,
                    f"{pos.qty:.0f}",
                    f"{pos.entry_price:.2f}",
                    pos.entry_date.isoformat(),
                    pos.status.value,
                    f"{pos.exit_price:.2f}" if pos.exit_price else "n/a",
                    pos.exit_date.isoformat() if pos.exit_date else "n/a",
                    pnl_str,
                    ret_str,
                )

            console.print(table)

    except ValueError as exc:
        err_console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=2)  # noqa: B904


@position_app.command("close")
def position_close(
    ctx: typer.Context,
    wkn: str = typer.Option(..., help="Warrant Kennnummer"),
    price: float = typer.Option(..., help="Exit price (must be > 0)"),
    date_str: str = typer.Option(..., "--date", help="Exit date (YYYY-MM-DD)"),
) -> None:
    """Close an open position."""
    app_ctx = _load_app_context(ctx)

    try:
        from datetime import date

        exit_date = date.fromisoformat(date_str)
        with Store(app_ctx.db_path) as store:
            store.init_schema()
            ledger = PositionLedger(store)
            closed = ledger.close(wkn, price, exit_date)
            pnl = ledger.realized_pnl(closed)
            ret = ledger.realized_return(closed)
            ret_pct = f"{ret * 100:.2f}%" if ret is not None else "n/a"
            pnl_str = f"{pnl:.2f}" if pnl is not None else "n/a"
            console.print(
                f"[green]Position closed: {closed.position_id} P&L={pnl_str} ({ret_pct})[/green]"
            )
    except ValueError as exc:
        err_console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(code=2)  # noqa: B904


# --- db subcommand ---


db_app = typer.Typer(help="Database queries")
app.add_typer(db_app, name="db")

# --- state subcommand + `db compact` (turboedge.cli_state) ---
register_state_commands(app, db_app)

# --- scan-all/label/learn/forecast/backtest/report/research/position
# reevaluate (turboedge.cli_learn, Contract v3 integration wave) ---
register_learn_commands(app, position_app)


@db_app.command("info")
def db_info(ctx: typer.Context) -> None:
    """Display database information."""
    app_ctx = _load_app_context(ctx)

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        counts = store.table_counts()

        table = Table(title=f"Database: {app_ctx.db_path}")
        table.add_column("Table", style="cyan")
        table.add_column("Rows", justify="right", style="magenta")

        for table_name, count in counts.items():
            table.add_row(table_name, str(count))

        console.print(table)


if __name__ == "__main__":
    app()
