"""The three alpha-status mails, and the rules that keep them honest.

The operator asked to be told automatically when a research result is
worth their attention, and not otherwise. Three classes, deliberately
unequal in tone:

* ``ALPHA_CANDIDATE`` -- an exploratory screen found something. This is the
  one that will fire most often and mean least, so its subject says
  *Kandidat* and its body says, in the first line of the verdict block,
  that nothing is confirmed. A screen runs on the DEVELOPMENT partition,
  which `research/partitions.py` defines as never being evidence.
* ``ALPHA_CONFIRMED`` -- every pre-registered statistical *and* economic
  condition passed on untouched data. It carries the fifteen items the
  operator asked for, including what is still not known.
* ``ALPHA_DEGRADED`` -- something previously confirmed stopped holding.
  Says what changed, since when, on what evidence, and whether the system
  already downgraded it.

WHAT THIS MODULE REFUSES TO DO
------------------------------
It will not render a confirmed-alpha mail without an effective sample, an
interval and a cost basis: :func:`render_alpha_alert` raises instead. A
"confirmed" mail whose numbers were optional would be the single most
misleading artefact this system could produce, so the check is structural
rather than a convention. A candidate mail may legitimately have a thinner
interval, and a degradation mail must be sendable precisely when the
numbers have gone bad.

Every mail carries a GitHub run reference, because a claim about measured
evidence that cannot be traced back to the run that measured it is not
evidence.

DEDUPLICATION
-------------
The hash covers the kind, the hypothesis and the rounded headline numbers,
so an unchanged status does not re-send daily -- the operator asked for
"keine Benachrichtigungsflut" explicitly. It deliberately does *not* cover
the free text, so rewording a template cannot resurrect an already-sent
alert. A genuinely changed effect size does produce a new hash, which is
the intended behaviour: the number moving is the news.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum

import structlog
from jinja2 import Environment

from turboedge.meta.research_opportunity import ResearchKind
from turboedge.notifications.dedup import NotificationDeduplicator, notification_hash
from turboedge.notifications.gmail import EmailMessageSpec, GmailNotifier, SendResult

logger = structlog.get_logger(__name__)

_env = Environment(autoescape=False)  # plaintext mails only


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.3f} %"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


_env.filters["pct"] = _pct
_env.filters["num"] = _num


class AlphaAlertKind(StrEnum):
    """The three notification classes, in increasing weight."""

    CANDIDATE = "ALPHA_CANDIDATE"
    CONFIRMED = "ALPHA_CONFIRMED"
    DEGRADED = "ALPHA_DEGRADED"


@dataclass(frozen=True, slots=True)
class AlphaAlert:
    """Everything a mail needs, with the numbers kept apart from the prose.

    ``is_test`` marks an isolated delivery check. It changes the subject and
    the first body line so a test can never be mistaken for a discovery,
    and it is part of the dedup hash so a test cannot suppress a later real
    alert about the same hypothesis.
    """

    kind: AlphaAlertKind
    #: What a positive result on this question would entitle anyone to
    #: conclude. It decides the wording: a COST_EDGE must never be
    #: announced as an alpha discovery, which is the condition the
    #: operator's 2026-10-26 authorisation for autonomous
    #: product-selection research was granted under.
    research_kind: ResearchKind
    hypothesis_id: str
    #: One plain-language sentence a non-developer can read.
    signal_description: str
    #: Why it should work, in words rather than symbols.
    mechanism: str
    underlyings: tuple[str, ...]
    horizon_days: int | None
    regime: str | None
    #: The measured economic effect, in `effect_unit`. For a cost finding
    #: this is a cost, and `effect_unit` has to say so.
    effect: float | None
    effect_unit: str
    ci_low: float | None
    ci_high: float | None
    #: Independent observations, not rows. The two differ by an order of
    #: magnitude in this repository and the mail must not quote the
    #: flattering one.
    effective_sample: float | None
    observations: int | None
    #: Which real costs the figure is net of. Empty means "none", and the
    #: template prints that rather than omitting the line.
    costs_included: tuple[str, ...] = ()
    null_comparison: str | None = None
    p_ko: float | None = None
    status: str = "FORWARD_SHADOW"
    limitations: tuple[str, ...] = ()
    run_id: str | None = None
    run_url: str | None = None
    screen_id: str | None = None
    #: Degradation only.
    previous_effect: float | None = None
    changed_since: date | None = None
    evidence: str | None = None
    safeguard: str | None = None
    auto_downgraded: bool | None = None
    is_test: bool = False
    extra: Mapping[str, float | str | None] = field(default_factory=dict)

    @property
    def shadow_only(self) -> bool:
        return self.status.upper().startswith("FORWARD_SHADOW")


_TEST_BANNER = (
    "*** ISOLIERTER ZUSTELLTEST — keine Alpha-Entdeckung. "
    "Diese Nachricht beschreibt kein echtes Ergebnis. ***"
)

#: What each kind of result is called in a subject line. The point is that
#: a cost finding and a directional edge do not share a word: "Alpha" is
#: reserved for `ALPHA_HYPOTHESIS`, and a COST_EDGE subject says
#: "Kostenvorteil" and nothing else.
_KIND_NOUN: dict[ResearchKind, str] = {
    ResearchKind.ALPHA_HYPOTHESIS: "Alpha",
    ResearchKind.COST_EDGE: "Kostenvorteil",
    ResearchKind.RISK_IMPROVEMENT: "Risikomessung",
    ResearchKind.RESEARCH_METHOD: "Forschungsmethode",
    ResearchKind.DATA_QUALITY_METHOD: "Datenqualitätsmethode",
}

_SUBJECTS: dict[AlphaAlertKind, str] = {
    AlphaAlertKind.CANDIDATE: "TurboEdge-DE: Neuer {{ noun }}-Kandidat — {{ hypothesis_id }}",
    AlphaAlertKind.CONFIRMED: (
        "TurboEdge-DE: {{ noun }} wissenschaftlich bestätigt — {{ hypothesis_id }}"
    ),
    AlphaAlertKind.DEGRADED: (
        "TurboEdge-DE: {{ noun }} verschlechtert oder nicht mehr bestätigt — {{ hypothesis_id }}"
    ),
}

#: Prepended to the verdict block for anything that is not a directional
#: hypothesis, so the reader cannot take it for one even if they read only
#: the first lines.
_NOT_ALPHA_BANNER: dict[ResearchKind, str] = {
    ResearchKind.COST_EDGE: (
        "DIES IST KEIN ALPHA. Es ist ein Kostenbefund: er sagt nichts über die\n"
        "Richtung und kann keinen Trade profitabel machen. Er könnte höchstens\n"
        "rechtfertigen, bei einer anderweitig getroffenen Entscheidung das\n"
        "günstigere Produkt zu wählen."
    ),
    ResearchKind.RISK_IMPROVEMENT: (
        "DIES IST KEIN ALPHA. Es ist eine bessere Risikomessung: sie verbessert\n"
        "Positionsgröße und Ausstieg, erzeugt aber selbst keinen Ertrag."
    ),
    ResearchKind.RESEARCH_METHOD: (
        "DIES IST KEIN ALPHA. Es ist eine Methode, um echte Ergebnisse von\n"
        "Artefakten zu unterscheiden."
    ),
    ResearchKind.DATA_QUALITY_METHOD: (
        "DIES IST KEIN ALPHA. Es ist eine Methode, um zu prüfen, ob die Daten\n"
        "eine Frage überhaupt tragen."
    ),
}

_VERDICT_LINES: dict[AlphaAlertKind, str] = {
    AlphaAlertKind.CANDIDATE: (
        "Status: KANDIDAT. Noch NICHT bestätigt. Gemessen auf Entwicklungsdaten,\n"
        "die ausdrücklich nicht als Nachweis gelten. Es folgt eine unabhängige\n"
        "Prüfung auf unangetasteten Daten; bis dahin ändert sich nichts am Handeln."
    ),
    AlphaAlertKind.CONFIRMED: (
        "Status: BESTÄTIGT. Alle vorab festgelegten statistischen und\n"
        "wirtschaftlichen Bedingungen wurden auf unangetasteten Daten erfüllt."
    ),
    AlphaAlertKind.DEGRADED: (
        "Status: ZURÜCKGESTUFT beziehungsweise nicht mehr bestätigt.\n"
        "Die bisherige Evidenz trägt das Ergebnis nicht mehr."
    ),
}

_BODY = """{% if is_test %}{{ test_banner }}

{% endif %}{{ verdict }}

Hypothese: {{ hypothesis_id }}
{% if screen_id %}Experiment: {{ screen_id }}
{% endif %}
Was gemessen wurde
  {{ signal_description }}

Warum es wirken könnte
  {{ mechanism }}

Zahlen
  Wirkung:            {{ effect | num }} {{ effect_unit }}
  95%-Intervall:      {{ interval_text }}
{% if previous_text %}  Vorher:             {{ previous_text }}
{% endif %}  Effektive Stichprobe: {{ effective_sample | num }} unabhängige Beobachtung(en)
{% if observations is not none %}  Rohbeobachtungen:   {{ observations }}
{% endif %}{% if p_ko is not none %}  Knock-out-Risiko: {{ p_ko | pct }}
{% endif %}
Berücksichtigte Kosten
{% if costs_included %}{% for item in costs_included %}  - {{ item }}
{% endfor %}{% else %}  KEINE. Die Zahl ist nicht um Kosten bereinigt und daher kein
  handelbarer Vorteil.
{% endif %}
Märkte und Horizont
  Basiswerte: {{ underlyings | join(', ') }}
{% if horizon_days is not none %}  Horizont:   {{ horizon_days }} Handelstage
{% endif %}{% if regime %}  Marktregime: {{ regime }}
{% endif %}{% if null_comparison %}
Vergleich mit dem Nullmodell
  {{ null_comparison }}
{% endif %}{% if changed_since %}
Veränderung
  Seit: {{ changed_since }}
{% if evidence %}  Belege: {{ evidence }}
{% endif %}{% if safeguard %}  Ausgelöste Sicherheitsmaßnahme: {{ safeguard }}
{% endif %}{% if auto_downgraded is not none %}  Automatisch zurückgestuft: {{ downgraded_text }}
{% endif %}{% endif %}
Verbleibende Einschränkungen
{% if limitations %}{% for item in limitations %}  - {{ item }}
{% endfor %}{% else %}  Keine benannt. Das ist selbst eine Einschränkung.
{% endif %}
Nachweis
{% if run_url %}  {{ run_url }}
{% elif run_id %}  GitHub-Actions-Lauf {{ run_id }}
{% else %}  Kein Lauf referenziert.
{% endif %}
{% if shadow_only %}Dieses Ergebnis läuft ausschließlich im Shadow-Modus. Es werden keine
Orders erzeugt und keine Handelsfreigabe erteilt.
{% else %}Status außerhalb des Shadow-Modus: {{ status }}
{% endif %}
Forschungssystem — Ausführung ausschließlich manuell."""


def render_alpha_alert(alert: AlphaAlert) -> tuple[str, str]:
    """Render one alpha mail as (subject, body).

    Raises:
        ValueError: if a CONFIRMED alert is missing any of the three things
            that make "confirmed" mean something -- an effect, an interval
            and an effective sample -- or names no cost basis. A confirmed
            mail with optional numbers would be the most misleading output
            this system could produce, so this is a structural refusal and
            not a style rule. CANDIDATE and DEGRADED alerts are exempt by
            design: a candidate's interval may be wide, and a degradation
            has to be reportable exactly when the numbers have gone bad.
    """
    if alert.kind is AlphaAlertKind.CONFIRMED:
        missing = [
            name
            for name, value in (
                ("effect", alert.effect),
                ("ci_low", alert.ci_low),
                ("ci_high", alert.ci_high),
                ("effective_sample", alert.effective_sample),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"a CONFIRMED alpha alert needs {missing}; refusing to render a "
                "confirmation whose evidence is absent"
            )
        if not alert.costs_included:
            raise ValueError(
                "a CONFIRMED alpha alert must name the costs the effect is net of; "
                "an unadjusted figure is not a tradeable advantage"
            )
    if not alert.underlyings:
        raise ValueError("an alpha alert must name at least one underlying")

    noun = _KIND_NOUN[alert.research_kind]
    subject = _env.from_string(_SUBJECTS[alert.kind]).render(
        hypothesis_id=alert.hypothesis_id, noun=noun
    )
    if alert.is_test:
        subject = f"[TEST] {subject}"
    body = _env.from_string(_BODY).render(
        is_test=alert.is_test,
        test_banner=_TEST_BANNER,
        verdict=(
            _NOT_ALPHA_BANNER[alert.research_kind] + "\n\n" + _VERDICT_LINES[alert.kind]
            if alert.research_kind in _NOT_ALPHA_BANNER
            else _VERDICT_LINES[alert.kind]
        ),
        hypothesis_id=alert.hypothesis_id,
        screen_id=alert.screen_id,
        signal_description=alert.signal_description,
        mechanism=alert.mechanism,
        effect=alert.effect,
        effect_unit=alert.effect_unit,
        interval_text=(
            f"[{_num(alert.ci_low)}, {_num(alert.ci_high)}]"
            if alert.ci_low is not None and alert.ci_high is not None
            else "nicht bestimmbar"
        ),
        previous_text=(
            None
            if alert.previous_effect is None
            else f"{_num(alert.previous_effect)} {alert.effect_unit}"
        ),
        effective_sample=alert.effective_sample,
        observations=alert.observations,
        p_ko=alert.p_ko,
        costs_included=list(alert.costs_included),
        underlyings=list(alert.underlyings),
        horizon_days=alert.horizon_days,
        regime=alert.regime,
        null_comparison=alert.null_comparison,
        changed_since=alert.changed_since.isoformat() if alert.changed_since else None,
        evidence=alert.evidence,
        safeguard=alert.safeguard,
        auto_downgraded=alert.auto_downgraded,
        downgraded_text=(
            None if alert.auto_downgraded is None else ("ja" if alert.auto_downgraded else "nein")
        ),
        limitations=list(alert.limitations),
        run_id=alert.run_id,
        run_url=alert.run_url,
        status=alert.status,
        shadow_only=alert.shadow_only,
    )
    return subject, body


def alpha_alert_hash(alert: AlphaAlert) -> str:
    """Dedup key: the kind, the hypothesis and the rounded headline numbers.

    Reuses `notification_hash` so alpha mails share one dedup table and one
    rounding convention with every other notification. The prose is
    excluded on purpose -- editing a template must not resurrect an alert
    that was already sent.
    """
    key_values: dict[str, float | str | None] = {
        "effect": alert.effect,
        "ci_low": alert.ci_low,
        "ci_high": alert.ci_high,
        "effective_sample": alert.effective_sample,
        "status": alert.status,
        "research_kind": str(alert.research_kind),
        "is_test": "yes" if alert.is_test else "no",
        **dict(alert.extra),
    }
    return notification_hash(
        candidate_id=alert.hypothesis_id,
        category=str(alert.kind),
        key_values=key_values,
    )


@dataclass(frozen=True, slots=True)
class AlphaDispatchResult:
    """What happened to one alert."""

    sent: bool
    suppressed_as_duplicate: bool
    attempts: int
    hash_value: str
    error: str | None = None


def send_alpha_alert(
    notifier: GmailNotifier | None,
    deduplicator: NotificationDeduplicator | None,
    alert: AlphaAlert,
    *,
    recipients: Sequence[str],
    now: datetime,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    backoff_seconds: float = 2.0,
) -> AlphaDispatchResult:
    """Send one alert, once, with a bounded retry.

    Order matters. The duplicate check comes first, so a repeated status
    costs nothing. ``mark_sent`` comes only after a confirmed send, so a
    delivery that failed on every attempt stays eligible for the next run
    rather than being silently recorded as delivered -- which is the
    failure mode that would make the operator believe they had been told.

    ``notifier=None`` is the dry-run path and reports ``sent=False`` with no
    error: nothing was delivered and nothing is pretended.

    Raises:
        ValueError: if ``max_attempts < 1`` or ``recipients`` is empty. An
            alert with nowhere to go is a configuration fault, and
            reporting it as a successful send would hide that.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}")
    if not recipients:
        raise ValueError("recipients must not be empty")

    hash_value = alpha_alert_hash(alert)
    if deduplicator is not None and not deduplicator.should_send(hash_value):
        logger.info(
            "alpha_alert_suppressed_duplicate",
            kind=str(alert.kind),
            hypothesis_id=alert.hypothesis_id,
            hash=hash_value,
        )
        return AlphaDispatchResult(
            sent=False, suppressed_as_duplicate=True, attempts=0, hash_value=hash_value
        )

    subject, body = render_alpha_alert(alert)
    if notifier is None:
        logger.info(
            "alpha_alert_dry_run",
            kind=str(alert.kind),
            hypothesis_id=alert.hypothesis_id,
            subject=subject,
        )
        return AlphaDispatchResult(
            sent=False, suppressed_as_duplicate=False, attempts=0, hash_value=hash_value
        )

    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result: SendResult = notifier.send(
                EmailMessageSpec(subject=subject, body_text=body, to=list(recipients))
            )
        except Exception as exc:  # a transport failure is expected and logged
            last_error = f"{type(exc).__name__}: {exc}"
            result = None  # type: ignore[assignment]
        else:
            last_error = None if result.sent else (result.message or "send reported not sent")

        if last_error is None:
            if deduplicator is not None:
                deduplicator.mark_sent(
                    hash_value,
                    candidate_id=alert.hypothesis_id,
                    category=str(alert.kind),
                    subject=subject,
                    sent_at=now,
                )
            logger.info(
                "alpha_alert_sent",
                kind=str(alert.kind),
                hypothesis_id=alert.hypothesis_id,
                attempts=attempt,
                hash=hash_value,
            )
            return AlphaDispatchResult(
                sent=True,
                suppressed_as_duplicate=False,
                attempts=attempt,
                hash_value=hash_value,
            )

        logger.warning(
            "alpha_alert_send_failed",
            kind=str(alert.kind),
            hypothesis_id=alert.hypothesis_id,
            attempt=attempt,
            max_attempts=max_attempts,
            error=last_error,
        )
        if attempt < max_attempts:
            sleep(backoff_seconds * attempt)

    logger.error(
        "alpha_alert_undelivered",
        kind=str(alert.kind),
        hypothesis_id=alert.hypothesis_id,
        attempts=max_attempts,
        error=last_error,
    )
    return AlphaDispatchResult(
        sent=False,
        suppressed_as_duplicate=False,
        attempts=max_attempts,
        hash_value=hash_value,
        error=last_error,
    )


def delivery_test_alerts() -> Sequence[AlphaAlert]:
    """Three isolated delivery checks, one per class.

    Every field is visibly synthetic: the hypothesis id is prefixed
    ``RO-DELIVERY-TEST``, ``is_test`` is set, and the body opens with a
    banner saying no discovery is being described. The operator asked for
    the delivery path to be proven "ohne eine tatsächliche
    Alpha-Entdeckung vorzutäuschen", and the only safe way to do that is
    for the test mail to say what it is.
    """
    common = {
        "research_kind": ResearchKind.ALPHA_HYPOTHESIS,
        "underlyings": ("DAX",),
        "horizon_days": 7,
        "regime": "mid_vol",
        "run_id": "0",
        "run_url": None,
        "is_test": True,
    }
    return (
        AlphaAlert(
            kind=AlphaAlertKind.CANDIDATE,
            hypothesis_id="RO-DELIVERY-TEST-CANDIDATE",
            signal_description="Synthetisches Beispielsignal ohne Bezug zu Marktdaten.",
            mechanism="Keiner. Dies ist ein Zustelltest.",
            effect=0.0,
            effect_unit="(Testwert, keine Einheit)",
            ci_low=-1.0,
            ci_high=1.0,
            effective_sample=0.0,
            observations=0,
            null_comparison="Kein Vergleich durchgeführt.",
            limitations=("Dies ist ein Zustelltest und kein Messergebnis.",),
            **common,  # type: ignore[arg-type]
        ),
        AlphaAlert(
            kind=AlphaAlertKind.CONFIRMED,
            hypothesis_id="RO-DELIVERY-TEST-CONFIRMED",
            signal_description="Synthetisches Beispielsignal ohne Bezug zu Marktdaten.",
            mechanism="Keiner. Dies ist ein Zustelltest.",
            effect=0.0,
            effect_unit="(Testwert, keine Einheit)",
            ci_low=-1.0,
            ci_high=1.0,
            effective_sample=0.0,
            observations=0,
            costs_included=("Testeintrag — keine echten Kosten gerechnet",),
            null_comparison="Kein Vergleich durchgeführt.",
            limitations=("Dies ist ein Zustelltest und kein Messergebnis.",),
            **common,  # type: ignore[arg-type]
        ),
        AlphaAlert(
            kind=AlphaAlertKind.DEGRADED,
            hypothesis_id="RO-DELIVERY-TEST-DEGRADED",
            signal_description="Synthetisches Beispielsignal ohne Bezug zu Marktdaten.",
            mechanism="Keiner. Dies ist ein Zustelltest.",
            effect=0.0,
            effect_unit="(Testwert, keine Einheit)",
            ci_low=-1.0,
            ci_high=1.0,
            effective_sample=0.0,
            observations=0,
            previous_effect=0.0,
            changed_since=date(2026, 10, 26),
            evidence="Kein Beleg — Zustelltest.",
            safeguard="Keine.",
            auto_downgraded=False,
            limitations=("Dies ist ein Zustelltest und kein Messergebnis.",),
            **common,  # type: ignore[arg-type]
        ),
    )


__all__ = [
    "AlphaAlert",
    "AlphaAlertKind",
    "AlphaDispatchResult",
    "ResearchKind",
    "alpha_alert_hash",
    "delivery_test_alerts",
    "render_alpha_alert",
    "send_alpha_alert",
]


def alert_for_screen(
    *,
    hypothesis_id: str,
    research_kind: ResearchKind,
    verdict: str,
    supported: bool,
    description: str,
    mechanism: str,
    underlyings: Sequence[str],
    horizon_days: int | None,
    metrics: Mapping[str, object],
    screen_id: str,
    run_url: str | None,
    run_id: str | None,
) -> AlphaAlert | None:
    """Turn a completed exploratory screen into an alert, or into nothing.

    `None` is the common and correct answer. A screen that found nothing is
    not news, and the operator asked for no notification flood; only a
    supported verdict produces a message. `supported` comes from the
    harness and is already false for every verdict outside
    `SUPPORTING_VERDICTS`, including `LOOKAHEAD_SUSPECT`, so a result that
    only survived through a look-ahead artefact cannot be announced.

    The alert is always `CANDIDATE`. A screen runs on DEVELOPMENT data,
    which `research/partitions.py` defines as never being evidence, so no
    screen can produce a confirmation however strong it looks. That is a
    property of where it ran, not of how large the number was.

    Numbers are taken from the harness's own metrics and left absent when
    the harness did not report them: the alert renderer prints "n/a" rather
    than letting a missing interval pass for a tight one.
    """
    if not supported:
        return None

    def _f(*names: str) -> float | None:
        for name in names:
            value = metrics.get(name)
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
        return None

    effect = _f("median_economic_size_bps", "mean_economic_size_bps", "rule_mean")
    unit = (
        "bps Round-Trip-Kosten über die Barrierespanne"
        if "median_economic_size_bps" in metrics
        else "(Einheit vom Harness nicht benannt)"
    )
    return AlphaAlert(
        kind=AlphaAlertKind.CANDIDATE,
        research_kind=research_kind,
        hypothesis_id=hypothesis_id,
        signal_description=description,
        mechanism=mechanism,
        underlyings=tuple(underlyings),
        horizon_days=horizon_days,
        regime=None,
        effect=effect,
        effect_unit=unit,
        ci_low=_f("spread_ci_low", "ci_low"),
        ci_high=_f("spread_ci_high", "ci_high"),
        effective_sample=_f("cells_used", "effective_sample", "n_oos"),
        observations=int(obs) if (obs := _f("products_used", "n_oos")) is not None else None,
        costs_included=(
            ("Spread (Ask/Bid), aus tatsächlich gestellten Quotes",)
            if "mean_spread_slope" in metrics
            else ()
        ),
        null_comparison=f"Harness-Urteil: {verdict}",
        status="EXPLORATORY_ONLY",
        limitations=(
            "Gemessen auf DEVELOPMENT-Daten, die ausdrücklich nicht als Nachweis gelten.",
            "Keine unabhängige Bestätigung; dafür ist eine menschliche Freigabe nötig.",
        ),
        run_id=run_id,
        run_url=run_url,
        screen_id=screen_id,
    )
