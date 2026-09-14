"""Rich console tables, JSON and plain-text rendering for pipeline results.

``render_universe``/``render_scan`` print to a provided ``rich.console.Console``
(``turboedge universe``/``turboedge scan``); ``scan_result_to_json`` and
``scan_report_text`` produce the ``--json-out``/``--report-out`` file
contents. Every renderer ends with the same "research system, no execution"
reminder (CLAUDE.md rule 3), and JSON/text reports contain no ``ACTIONABLE``
candidates in practice today (no forecast model has measured edge; see
``docs/measured_results.md``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from rich.table import Table

from turboedge.notifications.templates import ScanReportContext, ScanReportRow, render_scan_report
from turboedge.pipeline.scan import ScanResult
from turboedge.pipeline.universe import UniverseResult
from turboedge.storage.schemas import CandidateEvaluation

_FOOTER = "Research system — manual execution only."
_MAX_CONFLICTS_SHOWN = 20


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _fmt_num(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


# --------------------------------------------------------------------------
# console renderers
# --------------------------------------------------------------------------


def render_universe(result: UniverseResult, console: Console) -> None:
    """Render a :class:`~turboedge.pipeline.universe.UniverseResult` to ``console``."""
    console.print(f"[bold]Universe run[/bold] {result.run_id}")

    table = Table(title="Products by source")
    table.add_column("Source")
    table.add_column("Products", justify="right")
    for source, count in sorted(result.counts_by_source.items()):
        table.add_row(source, str(count))
    table.add_row("[bold]merged (deduplicated)[/bold]", str(len(result.products)))
    console.print(table)

    if result.source_errors:
        console.print("[yellow]Source errors:[/yellow]")
        for source, err in sorted(result.source_errors.items()):
            console.print(f"  - {source}: {err}")

    if result.conflicts:
        console.print(f"[yellow]{len(result.conflicts)} field conflict(s) detected[/yellow]")
        for conflict in result.conflicts[:_MAX_CONFLICTS_SHOWN]:
            console.print(
                f"  - {conflict.isin} {conflict.field}: "
                f"values={conflict.values} sources={conflict.sources}"
            )
        if len(result.conflicts) > _MAX_CONFLICTS_SHOWN:
            console.print(f"  ... and {len(result.conflicts) - _MAX_CONFLICTS_SHOWN} more")

    if result.snapshot is not None:
        console.print(
            f"Snapshot: {result.snapshot.path} "
            f"({result.snapshot.row_count} rows, "
            f"hash={result.snapshot.data_snapshot_hash[:12]}...)"
        )

    console.print(f"[dim]{_FOOTER}[/dim]")


def render_scan(result: ScanResult, console: Console, top: int) -> None:
    """Render a :class:`~turboedge.pipeline.scan.ScanResult` to ``console``."""
    if result.signal is not None:
        s = result.signal
        hint = s.direction_hint.value if s.direction_hint is not None else "none"
        console.print(
            f"[bold]Signal[/bold] {s.underlying_id}: score={s.score:.4f} "
            f"direction_hint={hint} threshold={s.threshold:.2f}"
        )
    else:
        console.print(
            "[yellow]Signal unavailable (insufficient underlying history) -- "
            "cost analysis only, no directional hint[/yellow]"
        )

    counts_line = "  ".join(f"{cat.value}={count}" for cat, count in result.counts.items())
    console.print(f"[bold]Candidates[/bold]  {counts_line}")

    table = Table(title=f"Top {top} candidates")
    for column in (
        "#",
        "WKN/ISIN",
        "Issuer",
        "Dir",
        "Category",
        "Lev",
        "Spread%",
        "Barrier %/sigma",
        "Margin%",
        "Fin. 7d%",
        "Gap%",
        "Cost per exposure (h)",
        "Liquidity",
        "Top reasons",
    ):
        table.add_column(column)

    for rank, c in enumerate(result.candidates[:top], start=1):
        wkn_isin = c.wkn or c.isin
        if c.distance_to_barrier_pct is not None and c.distance_to_barrier_sigma is not None:
            pct_part = f"{c.distance_to_barrier_pct * 100:.2f}%"
            sigma_part = f"{c.distance_to_barrier_sigma:.2f}sigma"
            barrier = f"{pct_part} / {sigma_part}"
        else:
            barrier = "n/a"
        table.add_row(
            str(rank),
            wkn_isin,
            c.issuer,
            c.direction.value,
            c.category.value,
            _fmt_num(c.leverage, "x"),
            _fmt_pct(c.costs.spread_pct if c.costs is not None else None),
            barrier,
            _fmt_pct(c.costs.issuer_margin_pct if c.costs is not None else None),
            _fmt_pct(c.financing_cost_horizon_pct.get("7d")),
            _fmt_pct(c.costs.gap_premium_pct if c.costs is not None else None),
            _fmt_pct(c.cost_rank_score),
            _fmt_num(c.liquidity_factor),
            "; ".join(c.reasons[:3]),
        )
    console.print(table)

    if result.warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  - {warning}")

    if result.notification is not None:
        n = result.notification
        status = "sent" if n.sent else ("dry-run" if n.dry_run else "not sent")
        console.print(f"Notification: {status} ({n.message})")

    console.print(f"[dim]{_FOOTER}[/dim]")


# --------------------------------------------------------------------------
# JSON / text export
# --------------------------------------------------------------------------


def scan_result_to_json(result: ScanResult) -> dict[str, Any]:
    """Serialize a :class:`ScanResult` for ``--json-out``."""
    return {
        "run_id": result.run_id,
        "signal": result.signal.model_dump(mode="json") if result.signal is not None else None,
        "candidates": [c.model_dump(mode="json") for c in result.candidates],
        "counts": {cat.value: count for cat, count in result.counts.items()},
        "warnings": list(result.warnings),
        "health": [h.model_dump(mode="json") for h in result.health],
        "notification": (
            {
                "sent": result.notification.sent,
                "dry_run": result.notification.dry_run,
                "recipients": list(result.notification.recipients),
                "message": result.notification.message,
            }
            if result.notification is not None
            else None
        ),
        "footer": _FOOTER,
    }


def _resolve_underlying_id(result: ScanResult) -> str:
    if result.signal is not None:
        return result.signal.underlying_id
    if result.candidates:
        return result.candidates[0].underlying_id
    return "unknown"


def _to_report_row(rank: int, c: CandidateEvaluation) -> ScanReportRow:
    return ScanReportRow(
        rank=rank,
        isin=c.isin,
        wkn=c.wkn,
        issuer=c.issuer,
        direction=c.direction.value,
        category=c.category.value,
        leverage=c.leverage,
        spread_pct=c.costs.spread_pct if c.costs is not None else None,
        distance_to_barrier_pct=c.distance_to_barrier_pct,
        issuer_margin_pct=c.costs.issuer_margin_pct if c.costs is not None else None,
        financing_cost_7d_pct=c.financing_cost_horizon_pct.get("7d"),
        liquidity_factor=c.liquidity_factor,
        cost_rank_score=c.cost_rank_score,
        reasons=c.reasons,
    )


def scan_report_text(result: ScanResult, top: int) -> str:
    """Plain-text scan report for ``--report-out`` (footer included, never ACTIONABLE)."""
    underlying_id = _resolve_underlying_id(result)
    rows = [_to_report_row(i + 1, c) for i, c in enumerate(result.candidates[:top])]

    context = ScanReportContext(
        run_id=result.run_id,
        underlying_id=underlying_id,
        generated_at=datetime.now(UTC),
        signal_score=result.signal.score if result.signal is not None else None,
        direction_hint=(
            result.signal.direction_hint.value
            if result.signal is not None and result.signal.direction_hint is not None
            else None
        ),
        counts={cat.value: count for cat, count in result.counts.items()},
        rows=rows,
        warnings=result.warnings,
    )
    _subject, body = render_scan_report(context)
    return body


__all__ = [
    "render_scan",
    "render_universe",
    "scan_report_text",
    "scan_result_to_json",
]
