"""Human-facing rendering of pipeline results: rich console tables, JSON, plain text.

This package never itself talks to the network, DuckDB or Gmail -- it only
formats already-computed :class:`~turboedge.pipeline.universe.UniverseResult`
and :class:`~turboedge.pipeline.scan.ScanResult` objects for a human (console)
or for a file (``--json-out`` / ``--report-out``). CLAUDE.md rule 3:
productive output is exclusively via Gmail/reports, never live execution --
every renderer here ends with the same reminder footer.
"""

from __future__ import annotations

from turboedge.reporting.console import (
    render_scan,
    render_universe,
    scan_report_text,
    scan_result_to_json,
)

__all__ = [
    "render_scan",
    "render_universe",
    "scan_report_text",
    "scan_result_to_json",
]
