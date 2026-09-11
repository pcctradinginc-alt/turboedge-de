"""Jinja2 plaintext email templates for scan reports and test emails."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from jinja2 import DictLoader, Environment, StrictUndefined


def _pct_filter(value: float | None) -> str:
    """Format decimal (0.0123) as percentage string (1.23%), or 'n/a' if None."""
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _num_filter(value: float | None) -> str:
    """Format a number, or 'n/a' if None."""
    if value is None:
        return "n/a"
    return f"{value:.4f}".rstrip("0").rstrip(".")


@dataclass
class ScanReportRow:
    """One product in a scan report."""

    rank: int
    isin: str
    wkn: str | None
    issuer: str
    direction: str
    category: str
    leverage: float | None
    spread_pct: float | None
    distance_to_barrier_pct: float | None
    issuer_margin_pct: float | None
    financing_cost_7d_pct: float | None
    liquidity_factor: float | None
    reasons: list[str]


@dataclass
class ScanReportContext:
    """Context for rendering a scan report template."""

    run_id: str
    underlying_id: str
    generated_at: datetime
    signal_score: float | None
    direction_hint: str | None
    counts: dict[str, int]  # category -> count
    rows: list[ScanReportRow]
    warnings: list[str]


# Jinja2 environment with strict undefined and plaintext templates
_env = Environment(
    loader=DictLoader({}),
    autoescape=False,
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)
_env.filters["pct"] = _pct_filter
_env.filters["num"] = _num_filter


_TEST_EMAIL_SUBJECT = "TurboEdge Test — {{ now.strftime('%Y-%m-%d %H:%M') }}"

_TEST_EMAIL_BODY = """TurboEdge-DE Test Notification

Timestamp: {{ now.isoformat() }}
Version: {{ version }}
Status: OK

This is a test notification to verify Gmail credentials and delivery.

Research system — manual execution only."""


_SCAN_REPORT_SUBJECT = (
    "TurboEdge SCAN — {{ underlying_id }}"
    "{% if counts %} — {% for cat, count in counts.items() %}"
    "{{ count }} {{ cat }}"
    '{{ "/ " if not loop.last else "" }}'
    "{% endfor %}{% endif %}"
)

_SCAN_REPORT_BODY = """TurboEdge-DE Scan Report

Run ID: {{ run_id }}
Underlying: {{ underlying_id }}
Generated: {{ generated_at.isoformat() }}
{% if signal_score is not none %}Signal Score: {{ signal_score | num }}
{% endif %}{% if direction_hint %}Direction Hint: {{ direction_hint }}
{% endif %}
Summary:
{% for cat, count in counts.items() %}  {{ cat }}: {{ count }}
{% endfor %}
{% if warnings %}Warnings:
{% for warn in warnings %}  - {{ warn }}
{% endfor %}
{% endif %}
{% if rows %}Candidates (ranked by cost):
{% for row in rows %}
{{ row.rank }}. {{ row.isin }}{% if row.wkn %} ({{ row.wkn }}){% endif %} — {{ row.issuer }}
   Direction: {{ row.direction }}, Category: {{ row.category }}
   Leverage: {{ row.leverage | num }}x
   Spread: {{ row.spread_pct | pct }}
   Distance to Barrier: {{ row.distance_to_barrier_pct | pct }}
   Issuer Margin: {{ row.issuer_margin_pct | pct }}
   Financing (7d): {{ row.financing_cost_7d_pct | pct }}
   Liquidity Factor: {{ row.liquidity_factor | num }}
   Reasons:
{% for reason in row.reasons %}     - {{ reason }}
{% endfor %}
{% endfor %}
{% else %}No candidates in this milestone.
{% endif %}
No ACTIONABLE category in this milestone: path model / LCB(EV) not yet implemented (Phase 3/4).

Research system — manual execution only."""


def render_test_email(now: datetime, version: str) -> tuple[str, str]:
    """Render a test email.

    Args:
        now: Current timestamp (tz-aware UTC).
        version: Version string (e.g., "0.1.0").

    Returns:
        Tuple of (subject, body_text).
    """
    subject_tmpl = _env.from_string(_TEST_EMAIL_SUBJECT)
    body_tmpl = _env.from_string(_TEST_EMAIL_BODY)

    subject = subject_tmpl.render(now=now)
    body = body_tmpl.render(now=now, version=version)

    return subject, body


def render_scan_report(context: ScanReportContext) -> tuple[str, str]:
    """Render a scan report.

    Args:
        context: ScanReportContext with run data and candidates.

    Returns:
        Tuple of (subject, body_text).
    """
    subject_tmpl = _env.from_string(_SCAN_REPORT_SUBJECT)
    body_tmpl = _env.from_string(_SCAN_REPORT_BODY)

    subject = subject_tmpl.render(
        underlying_id=context.underlying_id,
        counts=context.counts,
    )
    body = body_tmpl.render(
        run_id=context.run_id,
        underlying_id=context.underlying_id,
        generated_at=context.generated_at,
        signal_score=context.signal_score,
        direction_hint=context.direction_hint,
        counts=context.counts,
        rows=context.rows,
        warnings=context.warnings,
    )

    return subject, body


__all__ = [
    "ScanReportContext",
    "ScanReportRow",
    "render_scan_report",
    "render_test_email",
]
