"""Human-facing rendering of pipeline and learning results: rich console
tables, JSON, HTML and plain text.

This package never itself talks to the network or Gmail -- it only reads an
already-open :class:`~turboedge.storage.duckdb.Store` (monthly/weekly
reports) or formats already-computed
:class:`~turboedge.pipeline.universe.UniverseResult` /
:class:`~turboedge.pipeline.scan.ScanResult` objects (console/JSON/text), for
a human or for a file. CLAUDE.md rule 3: productive output is exclusively via
Gmail/reports, never live execution -- every renderer here ends with the same
reminder footer.
"""

from __future__ import annotations

from turboedge.reporting.console import (
    render_scan,
    render_universe,
    scan_report_text,
    scan_result_to_json,
)
from turboedge.reporting.html import (
    render_monthly_html,
    render_weekly_html,
    save_monthly_report,
    save_weekly_report,
    summary_counts,
    to_plain_text,
)
from turboedge.reporting.monthly import MonthlyReport, MonthlyReportConfig, build_monthly_report
from turboedge.reporting.weekly import (
    TournamentReport,
    WeeklyTournamentConfig,
    run_research_tournament,
)

__all__ = [
    "MonthlyReport",
    "MonthlyReportConfig",
    "TournamentReport",
    "WeeklyTournamentConfig",
    "build_monthly_report",
    "render_monthly_html",
    "render_scan",
    "render_universe",
    "render_weekly_html",
    "run_research_tournament",
    "save_monthly_report",
    "save_weekly_report",
    "scan_report_text",
    "scan_result_to_json",
    "summary_counts",
    "to_plain_text",
]
