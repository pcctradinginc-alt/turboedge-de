"""CLI for the External Data Factory (spec §27, §66).

Four verbs, mirroring the lifecycle: what sources exist, fetch them, what
state the data are in, and what the system has decided is usable. The
readiness view is the one the user should never have to reason about
themselves -- "when can we finally use this dataset?" is a question the
software answers (spec §84).
"""

from __future__ import annotations

import importlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from turboedge.adapters.base import HttpClient
from turboedge.config import external_data_path
from turboedge.external.adapter import ExternalSeriesAdapter
from turboedge.external.catalog import ExternalDataConfig, load_external_data_config
from turboedge.external.ingest import run_external_ingest
from turboedge.external.raw_archive import archive_size_bytes
from turboedge.external.readiness import READINESS_LEVEL
from turboedge.learning.failed_hypotheses import (
    default_path as failed_hypotheses_default_path,
)
from turboedge.meta.research_queue import ResearchQueue
from turboedge.storage.duckdb import Store

console = Console()


def _load(ctx: typer.Context) -> tuple[Any, ExternalDataConfig]:
    app_ctx = ctx.obj
    cfg = load_external_data_config(external_data_path(app_ctx.config_dir))
    return app_ctx, cfg


def build_external_adapters(cfg: ExternalDataConfig) -> dict[str, ExternalSeriesAdapter]:
    """Instantiate an adapter for every source that has one.

    Imported lazily and one at a time so that a single adapter failing to
    import -- a missing optional dependency, say -- costs that one source
    rather than the whole command.
    """
    adapters: dict[str, ExternalSeriesAdapter] = {}
    for source_id in cfg.sources:
        adapter = _try_build(source_id)
        if adapter is not None:
            adapters[source_id] = adapter
    return adapters


#: `source_id -> (module, class)`. A table rather than a match statement so
#: that adding a source is a one-line change here and nowhere else, and so
#: that an adapter that fails to import cannot stop the others from running.
_ADAPTER_CLASSES: dict[str, tuple[str, str]] = {
    "ecb": ("turboedge.adapters.ecb_data", "EcbDataAdapter"),
    "bundesbank": ("turboedge.adapters.bundesbank", "BundesbankAdapter"),
    "fred": ("turboedge.adapters.fred", "FredAdapter"),
    "eu_bcs": ("turboedge.adapters.eu_bcs", "EuBcsAdapter"),
    "destatis_truck": ("turboedge.adapters.destatis_truck", "DestatisTruckAdapter"),
    "estat": ("turboedge.adapters.estat", "EstatAdapter"),
}

#: Minimum seconds between requests to one host, per source. The project's
#: baseline is 1 request/second; `destatis_truck` is 30 because
#: `www.destatis.de/robots.txt` declares `Crawl-delay: 30`, and a declared
#: crawl delay is honoured rather than negotiated. The cost is one slow
#: fetch a day for a file that is republished weekly.
_MIN_INTERVAL_S: dict[str, float] = {
    "ecb": 1.0,
    "bundesbank": 1.0,
    "fred": 1.0,
    "eu_bcs": 1.0,
    "destatis_truck": 30.0,
    "estat": 1.0,
}

#: Honest, identifiable, and stating that this system never executes orders.
USER_AGENT = "TurboEdge-DE-Research/0.1 (+research-only; no-execution)"


def _http_for(source_id: str) -> HttpClient:
    return HttpClient(
        user_agent=USER_AGENT,
        timeout_s=90.0 if source_id == "destatis_truck" else 30.0,
        min_interval_s=_MIN_INTERVAL_S.get(source_id, 1.0),
    )


def _try_build(source_id: str) -> ExternalSeriesAdapter | None:
    """Build one adapter, or report why it is unavailable and carry on.

    A missing optional dependency or an adapter that is not written yet must
    cost that one source, not the whole command -- the other five still have
    data to collect today.
    """
    entry = _ADAPTER_CLASSES.get(source_id)
    if entry is None:
        return None
    module_name, class_name = entry
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        console.print(f"[yellow]adapter for {source_id} unavailable: {exc}[/yellow]")
        return None
    factory = getattr(module, class_name, None)
    if factory is None:
        console.print(
            f"[yellow]adapter for {source_id}: {module_name} has no {class_name}[/yellow]"
        )
        return None
    try:
        instance = factory(_http_for(source_id))
    except TypeError as exc:
        console.print(
            f"[yellow]adapter for {source_id} does not take the standard "
            f"HttpClient argument: {exc}[/yellow]"
        )
        return None
    if not isinstance(instance, ExternalSeriesAdapter):
        console.print(f"[yellow]{class_name} does not implement ExternalSeriesAdapter[/yellow]")
        return None
    return instance


def _credentials_present(cfg: ExternalDataConfig) -> dict[str, bool]:
    """Which credential-requiring sources have their variable set.

    Reads only whether the variable is non-empty. The value is never read
    into a log line, a report or the database.
    """
    out: dict[str, bool] = {}
    for manifest in cfg.sources.values():
        if manifest.requires_auth and manifest.auth_environment_variable:
            out[manifest.source_id] = bool(
                os.environ.get(manifest.auth_environment_variable, "").strip()
            )
    return out


def external_sources_cmd(ctx: typer.Context) -> None:
    """List every registered external source and its access status."""
    _, cfg = _load(ctx)
    present = _credentials_present(cfg)

    table = Table(title="External sources")
    for column in ("source", "status", "enabled", "credential", "PIT class", "series"):
        table.add_column(column)
    for manifest in cfg.sources.values():
        credential = "n/a"
        if manifest.requires_auth:
            credential = f"{manifest.auth_environment_variable} " + (
                "set" if present.get(manifest.source_id) else "MISSING"
            )
        table.add_row(
            manifest.source_id,
            str(manifest.status),
            "yes" if manifest.enabled else "no",
            credential,
            str(manifest.backfill_class),
            str(len(cfg.series_for(manifest.source_id))),
        )
    console.print(table)


def external_ingest_cmd(
    ctx: typer.Context,
    source: Annotated[
        list[str] | None,
        typer.Option("--source", help="Limit the run to these source ids (repeatable)."),
    ] = None,
) -> None:
    """Fetch, archive, normalise and evaluate every enabled series."""
    app_ctx, cfg = _load(ctx)
    adapters = build_external_adapters(cfg)

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        result = run_external_ingest(
            cfg,
            store,
            adapters=adapters,
            state_dir=app_ctx.state_dir,
            sources=source,
            credentials_present=_credentials_present(cfg),
        )
        handed_off = _hand_off_to_research(result, store, state_dir=app_ctx.state_dir)

    table = Table(title="Ingestion")
    for column in ("series", "rows", "state", "triggers", "note"):
        table.add_column(column)
    for item in result.series:
        note = item.error or ("; ".join(item.warnings) if item.warnings else "")
        table.add_row(
            item.series_id,
            str(item.observations_written),
            str(item.readiness.state) if item.readiness else "-",
            ",".join(str(t.trigger_type) for t in item.triggers) or "-",
            note[:60],
        )
    console.print(table)
    for source_id, reason in result.skipped_sources.items():
        console.print(f"[yellow]skipped {source_id}: {reason}[/yellow]")
    for hypothesis_id in handed_off:
        console.print(
            f"[green]proposed research question {hypothesis_id}[/green] "
            "(PROPOSED only -- a human must approve before anything runs)"
        )

    _write_summary(
        "external_ingest",
        {
            "series": len(result.series),
            "observations": result.observations_written,
            "failures": len(result.failures),
            "triggers": len(result.new_triggers),
            "questions_proposed": len(handed_off),
        },
        skipped=result.skipped_sources,
        raw_archive_bytes=archive_size_bytes(app_ctx.state_dir),
    )
    if result.failures:
        raise typer.Exit(code=1)


def _hand_off_to_research(result: Any, store: Store, *, state_dir: Path) -> list[str]:
    """Turn newly ready datasets into PROPOSED research questions.

    The hand-off spec §82.10 asks for, and nothing more. `ResearchQueue`
    owns the governance boundary, so the entry is created there, in
    `PROPOSED`, by the one method allowed to do it. Nothing here approves
    anything, starts a trial, consumes a research budget unit or touches a
    CONFIRMATION or FORWARD partition.

    The trigger is marked handed off only after the question exists, so an
    interrupted run re-proposes rather than silently losing the dataset.
    """
    if not result.new_triggers:
        return []

    queue = ResearchQueue(store, failed_hypotheses_path=failed_hypotheses_default_path(state_dir))
    inserted = queue.seed_from_data_readiness(result.new_triggers)
    by_key = {t.dedup_key: t for t in result.new_triggers}
    now = datetime.now(UTC)
    for hypothesis_id, dedup_key in inserted:
        trigger = by_key[dedup_key]
        store.upsert_research_trigger(
            trigger.model_copy(update={"handed_off_at": now, "hypothesis_id": hypothesis_id})
        )
    return [hypothesis_id for hypothesis_id, _ in inserted]


def external_readiness_cmd(
    ctx: typer.Context,
    json_out: Annotated[
        Path | None, typer.Option("--json-out", help="Also write the table as JSON.")
    ] = None,
) -> None:
    """Show, per series, whether the data may be used yet -- and why not."""
    app_ctx, _cfg = _load(ctx)

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        records = store.list_latest_readiness()

    table = Table(title="Data readiness")
    for column in ("series", "state", "lvl", "eff N", "span d", "fwd d", "blocking"):
        table.add_column(column)
    for record in records:
        blocking = "; ".join(record.blocking_reasons)
        table.add_row(
            f"{record.source}.{record.series_id}",
            str(record.state),
            str(READINESS_LEVEL[record.state]),
            f"{record.effective_n:.0f}",
            str(record.calendar_span_days),
            str(record.forward_span_days),
            blocking[:70] or "-",
        )
    console.print(table)
    console.print(
        "[dim]Level 5 (production eligible) is deliberately unreachable from data "
        "alone: it also needs a validated alpha, confirmation, positive economic "
        "Turbo evidence, a forward shadow and the Evidence & Risk Gate.[/dim]"
    )

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps([r.model_dump(mode="json") for r in records], indent=2))


def external_triggers_cmd(ctx: typer.Context) -> None:
    """Show every readiness event the system has emitted, and its hand-off."""
    app_ctx, _cfg = _load(ctx)

    with Store(app_ctx.db_path) as store:
        store.init_schema()
        triggers = store.list_research_triggers()

    if not triggers:
        console.print("no readiness events emitted yet")
        return

    table = Table(title="Research triggers")
    for column in ("series", "event", "emitted", "eff N", "handed off"):
        table.add_column(column)
    for trigger in triggers:
        table.add_row(
            f"{trigger.source}.{trigger.series_id}",
            str(trigger.trigger_type),
            trigger.emitted_at.astimezone(UTC).strftime("%Y-%m-%d"),
            f"{trigger.effective_n:.0f}",
            trigger.hypothesis_id or "pending",
        )
    console.print(table)
    console.print(
        "[dim]A trigger is a proposal, never an approval: it records that data "
        "became usable and says nothing about whether an alpha is there.[/dim]"
    )


def _write_summary(mode: str, counts: dict[str, int], **extra: Any) -> Path:
    path = Path("reports") / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"mode": mode, "counts": counts}
    payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def register_external_commands(app: typer.Typer) -> None:
    external_app = typer.Typer(help="External Data Factory: ingestion and readiness")
    external_app.command("sources")(external_sources_cmd)
    external_app.command("ingest")(external_ingest_cmd)
    external_app.command("readiness")(external_readiness_cmd)
    external_app.command("triggers")(external_triggers_cmd)
    app.add_typer(external_app, name="external")


__all__ = ["build_external_adapters", "register_external_commands"]
