"""Self-contained HTML and plain-text rendering, plus on-disk persistence,
for the monthly performance report (:mod:`turboedge.reporting.monthly`) and
the weekly research tournament (:mod:`turboedge.reporting.weekly`).

The HTML output embeds its own ``<style>`` block and nothing else -- no
external stylesheet, script, font, image, or tracking pixel of any kind
(Build Contract v2 W8 requirement 3: "kein externes CSS/JS, keine
Tracking-Pixel, druckbar") -- so it renders identically offline, in an email
client, or printed, and never phones home. ``to_plain_text`` produces the
plain-text counterpart used as the plain-text MIME part of the
corresponding email (``notifications/gmail.py`` sends both parts;
HTML-composition happens only here, W8 never sends mail itself).
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from turboedge.reporting._common import WilsonInterval
from turboedge.reporting.monthly import ModelWeightSummary, MonthlyReport, ReturnStats
from turboedge.reporting.weekly import TournamentReport

_FOOTER_TEXT = (
    "Research system — manual execution only.\n"
    "Alle Angaben sind Forschungsergebnisse dieses Systems, keine "
    "Anlageberatung und keine Aufforderung zum Kauf oder Verkauf von "
    "Finanzinstrumenten."
)

_FOOTER_HTML = (
    "<footer><p>Research system — manual execution only.</p>"
    "<p>Alle Angaben sind Forschungsergebnisse dieses Systems, keine "
    "Anlageberatung und keine Aufforderung zum Kauf oder Verkauf von "
    "Finanzinstrumenten.</p></footer>"
)

_HTML_STYLE = """<style>
body { font-family: Georgia, "Times New Roman", serif; color: #1a1a1a;
       max-width: 960px; margin: 0 auto; padding: 1.5em; background: #ffffff; }
h1 { font-size: 1.4em; margin-bottom: .1em; }
h2 { font-size: 1.15em; border-bottom: 1px solid #ccc; padding-bottom: .2em; margin-top: 1.6em; }
h4 { margin-bottom: .2em; }
table { border-collapse: collapse; width: 100%; margin: .5em 0 1em 0; font-size: .92em; }
th, td { border: 1px solid #ccc; padding: .3em .5em; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
p.meta { color: #555555; font-size: .9em; }
p.banner { background: #fff3cd; border: 1px solid #f0c36d; padding: .6em; }
p.notes { color: #7a3b00; font-size: .85em; }
ul.notes { color: #7a3b00; }
footer { margin-top: 2em; padding-top: 1em; border-top: 1px solid #ccc;
         font-size: .85em; color: #555555; }
@media print { body { padding: 0; } }
</style>"""


def _esc(value: str) -> str:
    return html.escape(value, quote=True)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _wilson_str(interval: WilsonInterval | None) -> str:
    if interval is None or interval.n == 0:
        return "n/a"
    return f"[{interval.lower * 100:.2f}%, {interval.upper * 100:.2f}%] (n={interval.n})"


def _ci_str(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "n/a"
    return f"[{ci[0] * 100:.2f}%, {ci[1] * 100:.2f}%]"


_NOT_RELIABLE_LABEL = "nicht ausgewiesen (kleine Stichprobe)"


def _return_stats_table(stats: ReturnStats) -> str:
    sharpe_str = _num(stats.sharpe, 2) if stats.reliable else _NOT_RELIABLE_LABEL
    sortino_str = _num(stats.sortino, 2) if stats.reliable else _NOT_RELIABLE_LABEL
    psr_str = _num(stats.psr, 3) if stats.reliable else _NOT_RELIABLE_LABEL
    dsr_str = _num(stats.dsr, 3) if stats.reliable else _NOT_RELIABLE_LABEL
    reliable_label = "ja" if stats.reliable else "NEIN — geringe Stichprobe"
    rows: list[tuple[str, str]] = [
        ("n (gelabelte Trades)", str(stats.n)),
        ("n_effective (average uniqueness)", f"{stats.n_effective:.2f}"),
        ("aussagekräftig (n &ge; Schwelle)", reliable_label),
        ("Erfolgsquote", _pct(stats.success_rate)),
        ("Erfolgsquote 95%-CI (Wilson)", _wilson_str(stats.success_rate_ci)),
        ("Ø Netto-Rendite", _pct(stats.mean_return)),
        ("Ø Netto-Rendite Bootstrap-CI", _ci_str(stats.mean_return_ci)),
        ("Median Netto-Rendite", _pct(stats.median_return)),
        ("Profit-Faktor", _num(stats.profit_factor, 2)),
        ("Sharpe (annualisiert)", sharpe_str),
        ("Sortino (annualisiert)", sortino_str),
        ("Probabilistic Sharpe Ratio", psr_str),
        ("Deflated Sharpe Ratio", dsr_str),
        ("Expected Shortfall (5%)", _pct(stats.expected_shortfall_95)),
        ("KO-Häufigkeit", _pct(stats.ko_frequency)),
        ("Ø Haltedauer (Handelstage)", _num(stats.mean_holding_days, 1)),
        ("Median Haltedauer (Handelstage)", _num(stats.median_holding_days, 1)),
        ("Ø Spread", _pct(stats.mean_spread_pct)),
        (
            f"Ø Finanzierungskosten (indikativ, n={stats.financing_drag_n})",
            _pct(stats.mean_financing_drag_pct),
        ),
        ("Brier Score", _num(stats.brier, 4)),
        ("Expected Calibration Error", _num(stats.ece, 4)),
        (
            f"Produktauswahl-Edge (selected-median, n={stats.product_selection_edge_n})",
            _pct(stats.product_selection_edge),
        ),
        (
            f"Produktauswahl-Regret (best-selected, n={stats.product_selection_regret_n})",
            _pct(stats.product_selection_regret),
        ),
        (f"Issuer-Drag (selected-ideal, n={stats.issuer_drag_n})", _pct(stats.issuer_drag)),
        ("Anteil ambiguous_path", _pct(stats.ambiguous_path_rate)),
    ]
    body = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)
    notes_html = ""
    if stats.notes:
        items = "".join(f"<li>{_esc(n)}</li>" for n in stats.notes)
        notes_html = f"<ul class='notes'>{items}</ul>"
    return f"<table>{body}</table>{notes_html}"


def _grouping_table(dimension: str, buckets: dict[str, ReturnStats]) -> str:
    header = (
        "<tr><th>Wert</th><th>n</th><th>n_eff</th><th>Erfolgsquote</th>"
        "<th>Ø Rendite</th><th>Sharpe</th></tr>"
    )
    rows = []
    for key, s in buckets.items():
        sharpe_cell = _num(s.sharpe, 2) if s.reliable else "-"
        rows.append(
            f"<tr><td>{_esc(key)}</td><td>{s.n}</td><td>{s.n_effective:.1f}</td>"
            f"<td>{_pct(s.success_rate)}</td><td>{_pct(s.mean_return)}</td>"
            f"<td>{sharpe_cell}</td></tr>"
        )
    return f"<h4>nach {_esc(dimension)}</h4><table>{header}{''.join(rows)}</table>"


def render_monthly_html(report: MonthlyReport) -> str:
    """Self-contained HTML for one :class:`MonthlyReport` (Build Contract
    v2 W8 requirement 3)."""
    parts: list[str] = [_HTML_STYLE]
    parts.append(f"<h1>TurboEdge-DE Monatsbericht {report.month.strftime('%Y-%m')}</h1>")
    parts.append(
        "<p class='meta'>Zeitraum: "
        f"{report.month.isoformat()} - {report.month_end.isoformat()} · Stand: "
        f"{_esc(report.as_of.isoformat())} · erzeugt: {_esc(report.generated_at.isoformat())}</p>"
    )
    if report.status_only:
        parts.append(
            "<p class='banner'>Keine gelabelten Vorschläge in diesem Zeitraum fällig — "
            "Statusbericht.</p>"
        )
    parts.append(f"<p>Scan-Läufe im Monat: {report.n_scans_in_month}</p>")
    if report.category_counts_in_month:
        cats = ", ".join(
            f"{_esc(k)}={v}" for k, v in sorted(report.category_counts_in_month.items())
        )
        parts.append(f"<p>Kandidaten-Kategorien im Monat: {cats}</p>")
    if report.narrative:
        items = "".join(f"<li>{_esc(n)}</li>" for n in report.narrative)
        parts.append(f"<ul>{items}</ul>")
    if report.data_quality_notes:
        items = "".join(f"<li>{_esc(n)}</li>" for n in report.data_quality_notes)
        parts.append(f"<h2>Datenqualität</h2><ul class='notes'>{items}</ul>")

    parts.append("<h2>ACTIONABLE-Vorschläge</h2>")
    parts.append(_return_stats_table(report.actionable))
    parts.append("<h2>Shadow-Sample</h2>")
    parts.append(_return_stats_table(report.shadow))

    if report.groupings_actionable:
        parts.append("<h2>ACTIONABLE nach Gruppierung</h2>")
        for g in report.groupings_actionable:
            parts.append(_grouping_table(g.dimension, g.buckets))
    if report.groupings_shadow:
        parts.append("<h2>Shadow nach Gruppierung</h2>")
        for g in report.groupings_shadow:
            parts.append(_grouping_table(g.dimension, g.buckets))

    if report.strategy_posteriors:
        parts.append("<h2>Strategy-Posteriors</h2>")
        rows = "".join(
            f"<tr><td>{_esc(p.signal_family)}</td><td>{p.horizon_days}d</td>"
            f"<td>{_pct(p.p_mu_positive)}</td><td>{_pct(p.posterior_mean)}</td>"
            f"<td>{_num(p.posterior_std, 4) if p.posterior_std is not None else 'undefiniert'}</td>"
            f"<td>{p.n:.1f}</td></tr>"
            for p in report.strategy_posteriors
        )
        parts.append(
            "<table><tr><th>Signalfamilie</th><th>Horizont</th><th>P(μ&gt;0)</th>"
            f"<th>Posterior-Mittel</th><th>Posterior-Std</th><th>n</th></tr>{rows}</table>"
        )

    if report.model_weights:
        parts.append("<h2>Modellgewichte</h2>")

        def _weight_row(m: ModelWeightSummary) -> str:
            start = (
                _num(m.weight_at_month_start, 4) if m.weight_at_month_start is not None else "n/a"
            )
            change = _num(m.weight_change, 4) if m.weight_change is not None else "n/a"
            return (
                f"<tr><td>{_esc(m.model_id)}</td><td>{_esc(m.signal_family)}</td>"
                f"<td>{_esc(m.status)}</td><td>{_num(m.weight, 4)}</td>"
                f"<td>{start}</td><td>{change}</td></tr>"
            )

        rows = "".join(_weight_row(m) for m in report.model_weights)
        parts.append(
            "<table><tr><th>Modell</th><th>Signalfamilie</th><th>Status</th><th>Gewicht</th>"
            f"<th>Gewicht Monatsanfang</th><th>Änderung</th></tr>{rows}</table>"
        )

    parts.append("<h2>Research-Trials</h2>")
    parts.append(
        f"<p>Quartal {_esc(report.trial_budget.quarter)}: {report.trial_budget.used}/"
        f"{report.trial_budget.budget} Anpassungsbudget verwendet.</p>"
    )
    if report.open_trials:
        trials = ", ".join(_esc(t) for t in report.open_trials)
        parts.append(f"<p>Offene (experimentelle) Trials: {trials}</p>")
    else:
        parts.append("<p>Keine offenen experimentellen Trials.</p>")

    parts.append("<h2>Drift-Ereignisse</h2>")
    if report.drift_events:
        rows = "".join(
            f"<tr><td>{_esc(d.detected_at.isoformat())}</td><td>{_esc(d.stream_id)}</td>"
            f"<td>{_esc(d.signal_family or '-')}</td><td>{_esc(d.metric)}</td>"
            f"<td>{_num(d.ph_statistic, 4)}</td><td>{_num(d.threshold, 4)}</td>"
            f"<td>{_esc(d.action)}</td></tr>"
            for d in report.drift_events
        )
        parts.append(
            "<table><tr><th>Erkannt</th><th>Stream</th><th>Signalfamilie</th><th>Metrik</th>"
            f"<th>PH-Statistik</th><th>Schwelle</th><th>Empfehlung</th></tr>{rows}</table>"
        )
    else:
        parts.append("<p>Keine Drift-Ereignisse in diesem Monat.</p>")

    parts.append("<h2>Shadow-Portfolio-Vergleich</h2>")
    rows = "".join(
        f"<tr><td>{_esc(s.portfolio)}</td><td>{s.n}</td><td>{s.n_realized}</td>"
        f"<td>{_pct(s.mean_net_return)}</td></tr>"
        for s in report.shadow_portfolio
    )
    parts.append(
        "<table><tr><th>Portfolio</th><th>n</th><th>n realisiert</th>"
        f"<th>Ø Netto-Rendite</th></tr>{rows}</table>"
    )

    parts.append(_FOOTER_HTML)
    return "\n".join(parts)


def render_weekly_html(report: TournamentReport) -> str:
    """Self-contained HTML for one :class:`TournamentReport` (Build
    Contract v2 W8 requirement 3)."""
    parts: list[str] = [_HTML_STYLE]
    parts.append("<h1>TurboEdge-DE Weekly Research Tournament</h1>")
    parts.append(
        "<p class='meta'>Fenster: "
        f"{report.window_start.isoformat()} - {report.window_end.isoformat()} · Stand: "
        f"{_esc(report.as_of.isoformat())} · erzeugt: {_esc(report.generated_at.isoformat())} · "
        f"FDR alpha={report.fdr_alpha:.2f}</p>"
    )
    champion = _esc(report.champion_model_id) if report.champion_model_id else "keiner registriert"
    parts.append(f"<p>Champion-Modell: {champion}</p>")

    parts.append("<h2>Signalfamilien</h2>")
    header = (
        "<tr><th>Familie</th><th>protected</th><th>n</th><th>n_eff</th><th>Ø Rendite</th>"
        "<th>Sharpe</th><th>PSR</th><th>DSR</th><th>z</th><th>p</th><th>BH rejected</th>"
        "<th>Brier</th><th>ECE</th><th>Ø Regret</th></tr>"
    )
    rows = "".join(
        f"<tr><td>{_esc(f.signal_family)}</td><td>{'ja' if f.protected else ''}</td>"
        f"<td>{f.n}</td><td>{f.n_effective:.1f}</td><td>{_pct(f.mean_return)}</td>"
        f"<td>{_num(f.sharpe, 2) if f.sharpe is not None else '-'}</td>"
        f"<td>{_num(f.psr, 3) if f.psr is not None else '-'}</td>"
        f"<td>{_num(f.dsr, 3) if f.dsr is not None else '-'}</td>"
        f"<td>{_num(f.z_score, 2) if f.z_score is not None else '-'}</td>"
        f"<td>{_num(f.p_value, 4) if f.p_value is not None else '-'}</td>"
        f"<td>{'ja' if f.bh_rejected else ('nein' if f.bh_rejected is not None else '-')}</td>"
        f"<td>{_num(f.brier, 4) if f.brier is not None else '-'}</td>"
        f"<td>{_num(f.ece, 4) if f.ece is not None else '-'}</td>"
        f"<td>{_pct(f.mean_regret)}</td></tr>"
        for f in report.families
    )
    parts.append(f"<table>{header}{rows}</table>")
    for f in report.families:
        if f.notes:
            note_str = "; ".join(_esc(n) for n in f.notes)
            parts.append(f"<p class='notes'>{_esc(f.signal_family)}: {note_str}</p>")

    parts.append("<h2>Promotion-Vorschläge</h2>")
    if report.promotions:
        rows = "".join(
            f"<tr><td>{_esc(p.challenger_model_id)}</td><td>{_esc(p.challenger_signal_family)}</td>"
            f"<td>{_esc(p.champion_model_id) if p.champion_model_id else '-'}</td>"
            f"<td>{_esc(p.trial_id) if p.trial_id else '-'}</td>"
            f"<td>{'JA' if p.passes_ladder else 'nein'}</td>"
            f"<td>{_pct(p.ev_improvement)}</td><td>{_pct(p.winrate_improvement)}</td>"
            f"<td>{'; '.join(_esc(r) for r in p.reasons)}</td></tr>"
            for p in report.promotions
        )
        parts.append(
            "<table><tr><th>Challenger</th><th>Familie</th><th>Champion</th><th>Trial-ID</th>"
            "<th>Ladder Rule</th><th>EV-Verbesserung</th><th>Winrate-Verbesserung</th>"
            f"<th>Details</th></tr>{rows}</table>"
        )
    else:
        parts.append("<p>Keine Challenger-Modelle registriert.</p>")

    parts.append("<h2>Demotion-Vorschläge</h2>")
    if report.demotions:
        rows = "".join(
            f"<tr><td>{_esc(d.model_id)}</td><td>{_esc(d.signal_family)}</td>"
            f"<td>{'; '.join(_esc(r) for r in d.reasons)}</td></tr>"
            for d in report.demotions
        )
        parts.append(
            f"<table><tr><th>Modell</th><th>Familie</th><th>Gründe</th></tr>{rows}</table>"
        )
    else:
        parts.append("<p>Keine Demotion-Trigger ausgelöst.</p>")

    parts.append("<h2>Gewichts-Vorschau</h2>")
    if report.weight_preview:
        rows = "".join(
            f"<tr><td>{_esc(w.model_id)}</td><td>{_esc(w.signal_family)}</td>"
            f"<td>{_num(w.current_weight, 4)}</td><td>{_num(w.previewed_weight, 4)}</td>"
            f"<td>{_num(w.delta, 4)}</td></tr>"
            for w in report.weight_preview
        )
        parts.append(
            "<table><tr><th>Modell</th><th>Familie</th><th>Aktuell</th><th>Vorschau</th>"
            f"<th>Δ</th></tr>{rows}</table>"
        )
    else:
        parts.append("<p>Keine Gewichts-Vorschau verfügbar.</p>")

    if report.notes:
        items = "".join(f"<li>{_esc(n)}</li>" for n in report.notes)
        parts.append(f"<h2>Hinweise</h2><ul class='notes'>{items}</ul>")

    parts.append(_FOOTER_HTML)
    return "\n".join(parts)


def _return_stats_plain(stats: ReturnStats) -> list[str]:
    lines = [
        f"  n={stats.n}  n_effective={stats.n_effective:.2f}  "
        f"aussagekraeftig={'ja' if stats.reliable else 'NEIN'}"
    ]
    if stats.success_rate is not None:
        ci = stats.success_rate_ci
        ci_str = (
            f" 95%-CI [{ci.lower * 100:.2f}%, {ci.upper * 100:.2f}%]"
            if ci is not None and ci.n > 0
            else ""
        )
        lines.append(f"  Erfolgsquote: {stats.success_rate * 100:.2f}%{ci_str}")
    if stats.mean_return is not None:
        mean_ci_str = (
            f" 95%-CI [{stats.mean_return_ci[0] * 100:.2f}%, {stats.mean_return_ci[1] * 100:.2f}%]"
            if stats.mean_return_ci is not None
            else ""
        )
        median_str = (
            f"  (Median {stats.median_return * 100:.2f}%)"
            if stats.median_return is not None
            else ""
        )
        lines.append(
            f"  Mittlere Nettorendite: {stats.mean_return * 100:.2f}%{mean_ci_str}{median_str}"
        )
    if (
        stats.reliable
        and stats.sharpe is not None
        and stats.psr is not None
        and stats.dsr is not None
    ):
        lines.append(
            f"  Sharpe={stats.sharpe:.2f}  Sortino={_num(stats.sortino, 2)}  "
            f"PSR={stats.psr:.3f}  DSR={stats.dsr:.3f}"
        )
    else:
        lines.append("  Sharpe/Sortino/PSR/DSR: nicht ausgewiesen (kleine Stichprobe)")
    for note in stats.notes:
        lines.append(f"  Hinweis: {note}")
    return lines


def _monthly_plain_text(report: MonthlyReport) -> str:
    lines = [
        f"TurboEdge-DE Monatsbericht {report.month.strftime('%Y-%m')}",
        f"Zeitraum: {report.month.isoformat()} - {report.month_end.isoformat()}",
        f"Stand: {report.as_of.isoformat()}  erzeugt: {report.generated_at.isoformat()}",
        "",
    ]
    if report.status_only:
        lines.append("STATUSBERICHT: keine gelabelten Vorschlaege in diesem Zeitraum faellig.")
        lines.append("")
    lines.append(f"Scan-Laeufe im Monat: {report.n_scans_in_month}")
    if report.category_counts_in_month:
        cats = ", ".join(f"{k}={v}" for k, v in sorted(report.category_counts_in_month.items()))
        lines.append(f"Kandidaten-Kategorien: {cats}")
    lines.append("")
    for line in report.narrative:
        lines.append(f"- {line}")
    lines.append("")
    lines.append("ACTIONABLE:")
    lines.extend(_return_stats_plain(report.actionable))
    lines.append("")
    lines.append("SHADOW:")
    lines.extend(_return_stats_plain(report.shadow))
    if report.data_quality_notes:
        lines.append("")
        lines.append("Datenqualitaet:")
        for note in report.data_quality_notes:
            lines.append(f"- {note}")
    lines.append("")
    lines.append(_FOOTER_TEXT)
    return "\n".join(lines)


def _weekly_plain_text(report: TournamentReport) -> str:
    lines = [
        "TurboEdge-DE Weekly Research Tournament",
        f"Fenster: {report.window_start.isoformat()} - {report.window_end.isoformat()}",
        f"Champion: {report.champion_model_id or 'keiner registriert'}",
        "",
        "Signalfamilien:",
    ]
    for f in report.families:
        # `n_effective` belongs next to `n`, not only in the HTML table: this
        # is the rendering that gets emailed, and the gap between the two is
        # the whole point. The 2026-09-26 report showed n=68 for positions
        # opened in one scan on one day, worth ~1 independent observation.
        # Sharpe is per-trade and unannualized (see reporting/weekly.py).
        lines.append(
            f"  {f.signal_family}{' [protected]' if f.protected else ''}: "
            f"n={f.n} n_effective={_num(f.n_effective, 2)} "
            f"mean_return={_num(f.mean_return, 4)} sharpe_per_trade={_num(f.sharpe, 2)} "
            f"psr={_num(f.psr, 3)} dsr={_num(f.dsr, 3)} bh_rejected={f.bh_rejected}"
        )
        for note in f.notes:
            if "n_effective" in note:
                lines.append(f"      {note}")
    lines.append("")
    lines.append("Promotion-Vorschlaege:")
    if report.promotions:
        for p in report.promotions:
            lines.append(
                f"  {p.challenger_model_id} ({p.challenger_signal_family}): "
                f"passes_ladder={p.passes_ladder} trial_id={p.trial_id}"
            )
    else:
        lines.append("  keine Challenger registriert")
    lines.append("")
    lines.append("Demotion-Vorschlaege:")
    if report.demotions:
        for d in report.demotions:
            lines.append(f"  {d.model_id}: " + "; ".join(d.reasons))
    else:
        lines.append("  keine")
    lines.append("")
    lines.append(_FOOTER_TEXT)
    return "\n".join(lines)


def to_plain_text(report: MonthlyReport | TournamentReport) -> str:
    """Plain-text rendering of ``report``, for the plain-text MIME part of
    the corresponding email (Master Spec §34's plaintext convention)."""
    if isinstance(report, MonthlyReport):
        return _monthly_plain_text(report)
    if isinstance(report, TournamentReport):
        return _weekly_plain_text(report)
    raise TypeError(f"unsupported report type {type(report)!r}")


def summary_counts(report: MonthlyReport | TournamentReport) -> dict[str, int]:
    """Small integer summary for the ``reports/summary.json`` convention
    used by the scheduled workflows (Build Contract v2 W8 requirement 4)."""
    if isinstance(report, MonthlyReport):
        return {
            "status_only": int(report.status_only),
            "n_scans_in_month": report.n_scans_in_month,
            "actionable_n": report.actionable.n,
            "shadow_n": report.shadow.n,
            "open_trials": len(report.open_trials),
            "drift_events": len(report.drift_events),
            "data_quality_notes": len(report.data_quality_notes),
        }
    if isinstance(report, TournamentReport):
        return {
            "families": len(report.families),
            "tested_families": sum(1 for f in report.families if f.p_value is not None),
            "challengers_evaluated": len(report.promotions),
            "promotions_suggested": sum(1 for p in report.promotions if p.passes_ladder),
            "demotions_suggested": len(report.demotions),
        }
    raise TypeError(f"unsupported report type {type(report)!r}")


def save_monthly_report(report: MonthlyReport, *, reports_dir: Path) -> tuple[Path, Path]:
    """Persist ``report`` as ``<reports_dir>/monthly/<YYYY-MM>.json`` and
    ``.html`` (Build Contract v2 W8 requirement 4). Creates the directory if
    needed; overwrites any existing file for the same month. Returns
    ``(json_path, html_path)``.
    """
    target_dir = reports_dir / "monthly"
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = report.month.strftime("%Y-%m")
    json_path = target_dir / f"{stem}.json"
    html_path = target_dir / f"{stem}.html"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    html_path.write_text(render_monthly_html(report), encoding="utf-8")
    return json_path, html_path


def save_weekly_report(report: TournamentReport, *, reports_dir: Path) -> tuple[Path, Path]:
    """Persist ``report`` as ``<reports_dir>/weekly/<YYYY-Www>.json`` and
    ``.html`` (ISO week of ``report.window_end``; Build Contract v2 W8
    requirement 4). Creates the directory if needed; overwrites any
    existing file for the same ISO week. Returns ``(json_path, html_path)``.
    """
    target_dir = reports_dir / "weekly"
    target_dir.mkdir(parents=True, exist_ok=True)
    iso_year, iso_week, _iso_weekday = report.window_end.isocalendar()
    stem = f"{iso_year}-W{iso_week:02d}"
    json_path = target_dir / f"{stem}.json"
    html_path = target_dir / f"{stem}.html"
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    html_path.write_text(render_weekly_html(report), encoding="utf-8")
    return json_path, html_path


__all__ = [
    "render_monthly_html",
    "render_weekly_html",
    "save_monthly_report",
    "save_weekly_report",
    "summary_counts",
    "to_plain_text",
]
