"""Tests for reporting/html.py: rendering, plain text, persistence, summaries."""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from turboedge.learning.registry import ModelRegistry
from turboedge.reporting.html import (
    render_monthly_html,
    render_weekly_html,
    save_monthly_report,
    save_weekly_report,
    summary_counts,
    to_plain_text,
)
from turboedge.reporting.monthly import build_monthly_report
from turboedge.reporting.weekly import WeeklyTournamentConfig, run_research_tournament
from turboedge.storage.schemas import Category, ModelStatus

_URL_RE = re.compile(r"https?://", re.IGNORECASE)
_AS_OF = datetime(2026, 9, 1, 6, 30, tzinfo=UTC)
_MONTH = date(2026, 8, 1)
_FOOTER_SNIPPET = "Research system — manual execution only."


def _populated_monthly_report(record_labeled, store):
    for i in range(5):
        pred = datetime(2026, 8, 1 + i, 8, 0, tzinfo=UTC)
        exit_due = date(2026, 8, 1 + i) + timedelta(days=5)
        bid = 5.10 if i % 2 == 0 else 4.60
        record_labeled(
            entry_overrides=dict(
                candidate_id=f"html-{i}",
                category=Category.ACTIONABLE,
                prediction_time=pred,
                exit_due=exit_due,
            ),
            label_overrides=dict(exit_bid=bid, realized_selected_pnl=bid / 4.86 - 1),
        )
    return build_monthly_report(store, month=_MONTH, as_of=_AS_OF)


def _populated_weekly_report(store):
    registry = ModelRegistry(store)
    registry.register(
        "tsmom_horizon_norm_v1", "hash-tsmom", "tsmom", {}, None, status=ModelStatus.PROTECTED
    )
    return run_research_tournament(
        store, as_of=_AS_OF, cfg=WeeklyTournamentConfig(lookback_days=30)
    )


def test_render_monthly_html_has_no_external_resources(store, record_labeled) -> None:
    report = _populated_monthly_report(record_labeled, store)
    out = render_monthly_html(report)
    assert not _URL_RE.search(out)
    assert "<script" not in out.lower()
    assert "<img" not in out.lower()
    assert "<style" in out.lower()
    assert _FOOTER_SNIPPET in out


def test_render_weekly_html_has_no_external_resources(store) -> None:
    report = _populated_weekly_report(store)
    out = render_weekly_html(report)
    assert not _URL_RE.search(out)
    assert "<script" not in out.lower()
    assert "<img" not in out.lower()
    assert _FOOTER_SNIPPET in out


def test_monthly_html_escapes_free_text_fields(store, record_labeled) -> None:
    record_labeled(
        entry_overrides=dict(
            candidate_id="esc-1",
            category=Category.ACTIONABLE,
            issuer="<script>alert(1)</script>",
            prediction_time=datetime(2026, 8, 2, 8, 0, tzinfo=UTC),
            exit_due=date(2026, 8, 9),
        ),
        label_overrides=dict(exit_bid=5.10, realized_selected_pnl=5.10 / 4.86 - 1),
    )
    report = build_monthly_report(store, month=_MONTH, as_of=_AS_OF)
    out = render_monthly_html(report)
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


def test_monthly_plain_text_has_footer(store, record_labeled) -> None:
    report = _populated_monthly_report(record_labeled, store)
    text = to_plain_text(report)
    assert _FOOTER_SNIPPET in text
    assert "<table" not in text and "<tr" not in text  # no HTML markup leaked in
    assert "Anlageberatung" in text


def test_weekly_plain_text_has_footer(store) -> None:
    report = _populated_weekly_report(store)
    text = to_plain_text(report)
    assert _FOOTER_SNIPPET in text
    assert "<table" not in text and "<tr" not in text


def test_summary_counts_monthly(store, record_labeled) -> None:
    report = _populated_monthly_report(record_labeled, store)
    counts = summary_counts(report)
    assert counts["actionable_n"] == 5
    assert counts["status_only"] == 0
    assert all(isinstance(v, int) for v in counts.values())


def test_summary_counts_weekly(store) -> None:
    report = _populated_weekly_report(store)
    counts = summary_counts(report)
    assert counts["families"] >= 1
    assert all(isinstance(v, int) for v in counts.values())


def test_save_monthly_report_writes_json_and_html(store, record_labeled, tmp_path: Path) -> None:
    report = _populated_monthly_report(record_labeled, store)
    reports_dir = tmp_path / "state" / "reports"
    json_path, html_path = save_monthly_report(report, reports_dir=reports_dir)

    assert json_path == reports_dir / "monthly" / "2026-08.json"
    assert html_path == reports_dir / "monthly" / "2026-08.html"
    assert json_path.exists()
    assert html_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["month"] == "2026-08-01"
    assert payload["actionable"]["n"] == 5
    assert _FOOTER_SNIPPET in html_path.read_text(encoding="utf-8")


def test_save_weekly_report_uses_iso_week(store, tmp_path: Path) -> None:
    report = _populated_weekly_report(store)
    reports_dir = tmp_path / "state" / "reports"
    json_path, html_path = save_weekly_report(report, reports_dir=reports_dir)

    iso_year, iso_week, _ = report.window_end.isocalendar()
    expected_stem = f"{iso_year}-W{iso_week:02d}"
    assert json_path == reports_dir / "weekly" / f"{expected_stem}.json"
    assert html_path == reports_dir / "weekly" / f"{expected_stem}.html"
    assert json_path.exists()
    assert html_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["fdr_alpha"] == report.fdr_alpha


def test_save_monthly_report_overwrites_on_rerun(store, record_labeled, tmp_path: Path) -> None:
    report = _populated_monthly_report(record_labeled, store)
    reports_dir = tmp_path / "state" / "reports"
    save_monthly_report(report, reports_dir=reports_dir)
    json_path, html_path = save_monthly_report(report, reports_dir=reports_dir)
    assert json_path.exists()
    assert html_path.exists()


def test_weekly_plain_text_shows_effective_sample_beside_the_row_count(store) -> None:
    """The emailed rendering must show the gap, not just the HTML table.

    The 2026-09-26 tournament email printed `n=68` for positions opened in a
    single scan on a single day — worth about one independent observation —
    and reported them as Benjamini-Hochberg significant. Showing `n_effective`
    only in the HTML report would leave the one rendering that actually gets
    sent as misleading as it was.
    """
    report = _populated_weekly_report(store)
    text = to_plain_text(report)

    assert "n_effective=" in text
    # Per-trade, not a sqrt(252)-annualized number built on the same fiction.
    assert "sharpe_per_trade=" in text
    assert "sharpe=" not in text.replace("sharpe_per_trade=", "")
