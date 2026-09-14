"""Jinja2 plaintext email templates for scan reports, test emails, trade
proposals (Master Spec §34), position updates and the optional no-actionable
digest."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    # "Cost per exposure (h)": total round-trip cost over the scan horizon as
    # a % of underlying exposure (leverage-normalized), used to rank
    # candidates within WATCH. See CandidateEvaluation.cost_rank_score
    # (storage/schemas.py) and pipeline/scan.py for the formula (Build
    # Contract BEFUND 2).
    cost_rank_score: float | None
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
   Cost per exposure (h): {{ row.cost_rank_score | pct }}
   Liquidity Factor: {{ row.liquidity_factor | num }}
   Reasons:
{% for reason in row.reasons %}     - {{ reason }}
{% endfor %}
{% endfor %}
{% else %}No candidates in this milestone.
{% endif %}
No ACTIONABLE candidates produced in practice (no forecast model has
measured edge; see docs/measured_results.md).

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


# --------------------------------------------------------------------------
# Trade proposal (Master Spec §34) -- Contract v3 Abschnitt C
# --------------------------------------------------------------------------


@dataclass
class TradeProposalContext:
    """Everything :func:`render_trade_proposal` needs for one ACTIONABLE
    candidate. Any field the source could not determine (e.g. ``mae`` /
    ``sigma`` in return units when the underlying's model does not expose
    them) is left ``None`` and the template omits that line entirely rather
    than guessing (CLAUDE.md rule 29)."""

    underlying_id: str
    issuer: str
    wkn: str | None
    isin: str
    direction: str
    horizon_days: int
    ask: float
    bid: float
    spread_pct: float
    leverage: float
    distance_to_barrier_pct: float | None
    distance_to_barrier_sigma: float | None
    p_profit: float
    expected_net_return: float
    lcb_net_return: float
    p_ko: float
    es95: float
    spread_cost_pct: float
    financing_cost_pct: float
    gap_premium_pct: float
    issuer_margin_pct: float
    suggested_position_fraction: float
    reasons: list[str]
    no_model_beats_null_disclosure: str
    mae_return_units: float | None = None


_TRADE_PROPOSAL_SUBJECT = (
    "TurboEdge ACTIONABLE — {{ underlying_id }} {{ direction | upper }} — "
    "{{ horizon_days }}d — WKN {{ wkn or isin }}"
)

_TRADE_PROPOSAL_BODY = """TurboEdge-DE Trade Proposal (ACTIONABLE)

Underlying: {{ underlying_id }}
Direction: {{ direction | upper }}
Horizon: {{ horizon_days }} trading days
Issuer: {{ issuer }}
WKN: {{ wkn or "n/a" }}    ISIN: {{ isin }}

Quote:
  Ask: {{ ask | num }}    Bid: {{ bid | num }}    Spread: {{ spread_pct | pct }}
  Leverage: {{ leverage | num }}x
{% if ko_distance_line %}  {{ ko_distance_line }}
{% endif %}{% if mae_return_units is not none %}  MAE (return units): {{ mae_return_units | num }}
{% endif %}
Forecast / EV (Horizont {{ horizon_days }}d):
  P(net profit): {{ p_profit | pct }}
  Expected net return: {{ expected_net_return | pct }}
  LCB(EV): {{ lcb_net_return | pct }}
  P(KO): {{ p_ko | pct }}  -- konservativ, nicht kalibriert (W5-Kalibrierungsstudie:
    simulierte P(KO) überschätzt das realisierte P(KO) im relevanten Bereich; siehe docs).
    Keine Nachjustierung per Fudge-Faktor.
  ES95: {{ es95 | pct }}

Kostenaufstellung (% vom Ask):
  Spread: {{ spread_cost_pct | pct }}
  Finanzierung über Horizont: {{ financing_cost_pct | pct }}
  Gap-Prämie: {{ gap_premium_pct | pct }}
  Emittenten-Aufschlag: {{ issuer_margin_pct | pct }}

Vorgeschlagener Kapitalanteil: {{ suggested_position_fraction | pct }} des Portfolios
(fractional Kelly, hart limitiert -- niemals ausgeführt, nur Vorschlag).

PRO:
{% for reason in pro_reasons %}  - {{ reason }}
{% endfor %}
CONTRA / zu beachten:
  - P(KO) = {{ p_ko | pct }} (konservativ, siehe oben)
  - ES95 = {{ es95 | pct }}
  - {{ no_model_beats_null_disclosure }}

EXIT-Bedingungen (manuell zu prüfen):
  - Horizont erreicht ({{ horizon_days }} Handelstage) -> Position schließen/neu bewerten.
  - Signal dreht (Richtungswechsel des Basiswert-Signals) -> EXIT prüfen.
  - LCB(EV) wird negativ oder P(KO) steigt materiell -> REDUCE/EXIT prüfen
    (`turboedge position reevaluate`).
  - Knock-out-Barriere erreicht -> automatischer Verfall, keine Handlung möglich.

Research system — manual execution only. Kein automatischer Handel, keine Order
wird ausgeführt."""

_TRADE_PROPOSAL_EXCLUDED_REASON_PREFIXES = (
    "lcb_ev_not_positive",
    "cluster_risk",
    "quote_age_at_decision",
    "source_quote_stale",
    "spread_too_high",
)


def render_trade_proposal(context: TradeProposalContext) -> tuple[str, str]:
    """Render the §34 ACTIONABLE trade-proposal email (subject, body)."""
    subject_tmpl = _env.from_string(_TRADE_PROPOSAL_SUBJECT)
    body_tmpl = _env.from_string(_TRADE_PROPOSAL_BODY)
    pro_reasons = [
        reason
        for reason in context.reasons
        if not reason.startswith(_TRADE_PROPOSAL_EXCLUDED_REASON_PREFIXES)
    ]
    ko_distance_line = ""
    if context.distance_to_barrier_pct is not None:
        ko_distance_line = f"KO distance: {_pct_filter(context.distance_to_barrier_pct)}"
        if context.distance_to_barrier_sigma is not None:
            ko_distance_line += f" ({_num_filter(context.distance_to_barrier_sigma)} sigma)"
    ctx = {
        "pro_reasons": pro_reasons,
        "ko_distance_line": ko_distance_line,
        "underlying_id": context.underlying_id,
        "issuer": context.issuer,
        "wkn": context.wkn,
        "isin": context.isin,
        "direction": context.direction,
        "horizon_days": context.horizon_days,
        "ask": context.ask,
        "bid": context.bid,
        "spread_pct": context.spread_pct,
        "leverage": context.leverage,
        "mae_return_units": context.mae_return_units,
        "p_profit": context.p_profit,
        "expected_net_return": context.expected_net_return,
        "lcb_net_return": context.lcb_net_return,
        "p_ko": context.p_ko,
        "es95": context.es95,
        "spread_cost_pct": context.spread_cost_pct,
        "financing_cost_pct": context.financing_cost_pct,
        "gap_premium_pct": context.gap_premium_pct,
        "issuer_margin_pct": context.issuer_margin_pct,
        "suggested_position_fraction": context.suggested_position_fraction,
        "reasons": context.reasons,
        "no_model_beats_null_disclosure": context.no_model_beats_null_disclosure,
    }
    subject = subject_tmpl.render(**ctx)
    body = body_tmpl.render(**ctx)
    return subject, body


# --------------------------------------------------------------------------
# Position update (HOLD/REDUCE/EXIT/INVALIDATED) -- Contract v3 Abschnitt D
# --------------------------------------------------------------------------


@dataclass
class PositionUpdateContext:
    wkn: str
    isin: str | None
    underlying_id: str | None
    status: str  # HOLD | REDUCE | EXIT | INVALIDATED
    reasons: list[str] = field(default_factory=list)
    current_bid: float | None = None
    remaining_horizon_days: int | None = None
    remaining_lcb_ev: float | None = None
    remaining_p_ko: float | None = None
    unrealized_return: float | None = None


_POSITION_UPDATE_SUBJECT = "TurboEdge Position — {{ wkn }} — {{ status }}"

_POSITION_UPDATE_BODY = """TurboEdge-DE Position Update

WKN: {{ wkn }}{% if isin %}    ISIN: {{ isin }}{% endif %}
{% if underlying_id %}Underlying: {{ underlying_id }}
{% endif %}
Status: {{ status }}

{% if current_bid is not none %}Current bid: {{ current_bid | num }}
{% endif %}{% if unrealized_return is not none %}Unrealized return: {{ unrealized_return | pct }}
{% endif %}{% if remaining_horizon_days is not none %}
Remaining horizon: {{ remaining_horizon_days }} trading days
{% endif %}{% if remaining_lcb_ev is not none %}Remaining LCB(EV): {{ remaining_lcb_ev | pct }}
{% endif %}{% if remaining_p_ko is not none %}
Remaining P(KO): {{ remaining_p_ko | pct }} -- konservativ, nicht kalibriert.
{% endif %}
Begründung:
{% for reason in reasons %}  - {{ reason }}
{% endfor %}
Research system — manual execution only."""


def render_position_update(context: PositionUpdateContext) -> tuple[str, str]:
    """Render a HOLD/REDUCE/EXIT/INVALIDATED position-update email."""
    subject_tmpl = _env.from_string(_POSITION_UPDATE_SUBJECT)
    body_tmpl = _env.from_string(_POSITION_UPDATE_BODY)
    ctx = {
        "wkn": context.wkn,
        "isin": context.isin,
        "underlying_id": context.underlying_id,
        "status": context.status,
        "reasons": context.reasons,
        "current_bid": context.current_bid,
        "remaining_horizon_days": context.remaining_horizon_days,
        "remaining_lcb_ev": context.remaining_lcb_ev,
        "remaining_p_ko": context.remaining_p_ko,
        "unrealized_return": context.unrealized_return,
    }
    subject = subject_tmpl.render(**ctx)
    body = body_tmpl.render(**ctx)
    return subject, body


# --------------------------------------------------------------------------
# No-actionable digest (opt-in only -- Contract v3 Abschnitt C)
# --------------------------------------------------------------------------


@dataclass
class NoActionableDigestContext:
    """Only rendered when the operator has explicitly opted into a periodic
    "nothing to trade" summary -- never sent by default (Contract v3: "Kein
    täglicher 'kein Trade'-Spam")."""

    period_label: str
    underlyings_scanned: list[str]
    watch_counts: dict[str, int]
    top_reasons: list[str]


_NO_ACTIONABLE_DIGEST_SUBJECT = "TurboEdge Digest — {{ period_label }} — no ACTIONABLE"

_NO_ACTIONABLE_DIGEST_BODY = """TurboEdge-DE Digest ({{ period_label }})

No ACTIONABLE candidates this period.

Underlyings scanned: {{ underlyings_scanned | join(", ") }}

WATCH counts:
{% for underlying, count in watch_counts.items() %}  {{ underlying }}: {{ count }}
{% endfor %}
Most common reasons candidates did not reach ACTIONABLE:
{% for reason in top_reasons %}  - {{ reason }}
{% endfor %}
This is expected and correct whenever no forecast model has a demonstrated
out-of-sample edge (see W4 measurement) -- the system is designed to say
"no trade" often (Master Spec).

Research system — manual execution only."""


def render_no_actionable_digest(context: NoActionableDigestContext) -> tuple[str, str]:
    """Render the opt-in periodic "no ACTIONABLE" digest."""
    subject_tmpl = _env.from_string(_NO_ACTIONABLE_DIGEST_SUBJECT)
    body_tmpl = _env.from_string(_NO_ACTIONABLE_DIGEST_BODY)
    ctx = {
        "period_label": context.period_label,
        "underlyings_scanned": context.underlyings_scanned,
        "watch_counts": context.watch_counts,
        "top_reasons": context.top_reasons,
    }
    subject = subject_tmpl.render(**ctx)
    body = body_tmpl.render(**ctx)
    return subject, body


__all__ = [
    "NoActionableDigestContext",
    "PositionUpdateContext",
    "ScanReportContext",
    "ScanReportRow",
    "TradeProposalContext",
    "render_no_actionable_digest",
    "render_position_update",
    "render_scan_report",
    "render_test_email",
    "render_trade_proposal",
]
