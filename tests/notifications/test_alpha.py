"""Tests for the three alpha-status mails.

The load-bearing tests are the refusals, not the renders. A mail that says
"wissenschaftlich bestätigt" is the most consequential thing this system
can emit, so:

* it cannot be rendered without an effect, an interval, an effective
  sample and a cost basis;
* a delivery that failed on every attempt must not be recorded as sent,
  because that is the one failure mode that leaves the operator believing
  they were told;
* a test mail must say it is a test, in the subject and in the first line.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from typing import ClassVar

import pytest

from turboedge.meta.research_opportunity import ResearchKind
from turboedge.notifications.alpha import (
    AlphaAlert,
    AlphaAlertKind,
    alert_for_screen,
    alpha_alert_hash,
    delivery_test_alerts,
    render_alpha_alert,
    send_alpha_alert,
)

NOW = datetime(2026, 10, 26, 20, 0, tzinfo=UTC)


def _confirmed(**over: object) -> AlphaAlert:
    base = AlphaAlert(
        kind=AlphaAlertKind.CONFIRMED,
        research_kind=ResearchKind.ALPHA_HYPOTHESIS,
        hypothesis_id="RO-EXAMPLE",
        signal_description="Beispiel.",
        mechanism="Beispielmechanismus.",
        underlyings=("DAX",),
        horizon_days=7,
        regime="mid_vol",
        effect=0.0031,
        effect_unit="Netto-EV pro Trade",
        ci_low=0.0009,
        ci_high=0.0054,
        effective_sample=212.0,
        observations=3104,
        costs_included=("Spread (Ask/Bid)", "Finanzierung", "Emittentenmarge"),
        null_comparison="Nullmodell NO_TRADE: 0.0000",
        p_ko=0.041,
        status="FORWARD_SHADOW",
        limitations=("Nur ein Basiswert.",),
        run_id="40377514500",
        run_url="https://github.com/x/y/actions/runs/40377514500",
        screen_id="2026Q4_SCREEN_abc",
    )
    return replace(base, **over)  # type: ignore[arg-type]


class _Result:
    def __init__(self, sent: bool, message: str = "") -> None:
        self.sent = sent
        self.dry_run = False
        self.recipients = ["a@b.c"]
        self.message = message


class _Notifier:
    """Records attempts and can be told to fail a given number of times."""

    def __init__(self, *, fail_times: int = 0, raise_times: int = 0) -> None:
        self.fail_times = fail_times
        self.raise_times = raise_times
        self.calls: list[tuple[str, str]] = []

    def send(self, spec: object) -> _Result:
        self.calls.append((spec.subject, spec.body_text))  # type: ignore[attr-defined]
        if self.raise_times > 0:
            self.raise_times -= 1
            raise TimeoutError("smtp timeout")
        if self.fail_times > 0:
            self.fail_times -= 1
            return _Result(False, "temporary failure")
        return _Result(True, "ok")


class _Dedup:
    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.marked: list[str] = []

    def should_send(self, hash_value: str) -> bool:
        return hash_value not in self.seen

    def mark_sent(self, hash_value: str, **_: object) -> None:
        self.seen.add(hash_value)
        self.marked.append(hash_value)


class TestRenderRefusals:
    @pytest.mark.parametrize("field", ["effect", "ci_low", "ci_high", "effective_sample"])
    def test_a_confirmation_without_its_evidence_is_refused(self, field: str) -> None:
        with pytest.raises(ValueError, match="refusing to render a confirmation"):
            render_alpha_alert(_confirmed(**{field: None}))

    def test_a_confirmation_naming_no_costs_is_refused(self) -> None:
        """An unadjusted figure is not a tradeable advantage."""
        with pytest.raises(ValueError, match="must name the costs"):
            render_alpha_alert(_confirmed(costs_included=()))

    def test_a_candidate_may_have_a_thin_interval(self) -> None:
        """Exempt by design: a candidate's uncertainty is the point."""
        alert = _confirmed(kind=AlphaAlertKind.CANDIDATE, ci_low=None, ci_high=None)
        subject, body = render_alpha_alert(alert)

        assert "Alpha-Kandidat" in subject
        assert "nicht bestimmbar" in body

    def test_a_degradation_is_sendable_when_the_numbers_have_gone_bad(self) -> None:
        alert = _confirmed(
            kind=AlphaAlertKind.DEGRADED,
            effect=None,
            ci_low=None,
            ci_high=None,
            previous_effect=0.0031,
            changed_since=date(2026, 10, 20),
            evidence="48 neue Forward-Labels",
            safeguard="Gewicht auf w_min reduziert",
            auto_downgraded=True,
        )
        _subject, body = render_alpha_alert(alert)

        assert "ZURÜCKGESTUFT" in body
        assert "2026-10-20" in body
        assert "Automatisch zurückgestuft: ja" in body

    def test_an_alert_with_no_underlying_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one underlying"):
            render_alpha_alert(_confirmed(underlyings=()))


class TestRenderContent:
    def test_a_candidate_says_it_is_not_confirmed(self) -> None:
        """The operator asked for no overstated success messages."""
        _subject, body = render_alpha_alert(_confirmed(kind=AlphaAlertKind.CANDIDATE))

        assert "Noch NICHT bestätigt" in body
        assert "nicht als Nachweis gelten" in body

    def test_the_confirmed_mail_carries_the_required_items(self) -> None:
        _subject, body = render_alpha_alert(_confirmed())

        for needle in (
            "RO-EXAMPLE",
            "2026Q4_SCREEN_abc",
            "DAX",
            "7 Handelstage",
            "mid_vol",
            "212.0000",
            "3104",
            "Spread (Ask/Bid)",
            "Finanzierung",
            "Nullmodell",
            "4.100 %",
            "40377514500",
            "Nur ein Basiswert.",
        ):
            assert needle in body, needle

    def test_shadow_only_status_is_stated_plainly(self) -> None:
        _subject, body = render_alpha_alert(_confirmed())
        assert "ausschließlich im Shadow-Modus" in body
        assert "keine Handelsfreigabe" in body

    def test_a_status_outside_shadow_mode_is_not_hidden(self) -> None:
        _subject, body = render_alpha_alert(_confirmed(status="LIVE_CANDIDATE"))
        assert "Status außerhalb des Shadow-Modus: LIVE_CANDIDATE" in body

    def test_missing_limitations_are_themselves_reported(self) -> None:
        _subject, body = render_alpha_alert(_confirmed(limitations=()))
        assert "Keine benannt. Das ist selbst eine Einschränkung." in body

    def test_every_test_alert_announces_itself(self) -> None:
        for alert in delivery_test_alerts():
            subject, body = render_alpha_alert(alert)
            assert subject.startswith("[TEST] ")
            assert "keine Alpha-Entdeckung" in body
            assert alert.hypothesis_id.startswith("RO-DELIVERY-TEST")


class TestDedupHash:
    def test_the_same_status_hashes_the_same(self) -> None:
        assert alpha_alert_hash(_confirmed()) == alpha_alert_hash(_confirmed())

    def test_rewording_the_prose_does_not_change_the_hash(self) -> None:
        """Editing a template must not resurrect an already-sent alert."""
        assert alpha_alert_hash(_confirmed()) == alpha_alert_hash(
            _confirmed(signal_description="Anders formuliert.", mechanism="Auch anders.")
        )

    def test_a_changed_effect_does_change_the_hash(self) -> None:
        """The number moving is the news."""
        assert alpha_alert_hash(_confirmed()) != alpha_alert_hash(_confirmed(effect=0.0074))

    def test_a_test_alert_cannot_suppress_a_real_one(self) -> None:
        assert alpha_alert_hash(_confirmed(is_test=True)) != alpha_alert_hash(
            _confirmed(is_test=False)
        )

    def test_the_kind_is_part_of_the_hash(self) -> None:
        assert alpha_alert_hash(_confirmed()) != alpha_alert_hash(
            _confirmed(kind=AlphaAlertKind.DEGRADED)
        )


class TestDispatch:
    def test_a_first_send_goes_out_and_is_recorded(self) -> None:
        notifier, dedup = _Notifier(), _Dedup()
        result = send_alpha_alert(notifier, dedup, _confirmed(), recipients=["a@b.c"], now=NOW)

        assert result.sent
        assert result.attempts == 1
        assert dedup.marked == [result.hash_value]
        assert len(notifier.calls) == 1

    def test_the_second_identical_alert_is_suppressed(self) -> None:
        notifier, dedup = _Notifier(), _Dedup()
        alert = _confirmed()
        send_alpha_alert(notifier, dedup, alert, recipients=["a@b.c"], now=NOW)
        second = send_alpha_alert(notifier, dedup, alert, recipients=["a@b.c"], now=NOW)

        assert second.suppressed_as_duplicate
        assert not second.sent
        assert len(notifier.calls) == 1  # not attempted again

    def test_a_transient_failure_is_retried_then_succeeds(self) -> None:
        notifier, dedup = _Notifier(fail_times=2), _Dedup()
        result = send_alpha_alert(
            notifier,
            dedup,
            _confirmed(),
            recipients=["a@b.c"],
            now=NOW,
            sleep=lambda _s: None,
        )

        assert result.sent
        assert result.attempts == 3
        assert len(notifier.calls) == 3

    def test_an_exception_counts_as_a_failed_attempt(self) -> None:
        notifier, dedup = _Notifier(raise_times=1), _Dedup()
        result = send_alpha_alert(
            notifier,
            dedup,
            _confirmed(),
            recipients=["a@b.c"],
            now=NOW,
            sleep=lambda _s: None,
        )
        assert result.sent
        assert result.attempts == 2

    def test_an_undelivered_alert_is_not_marked_sent(self) -> None:
        """The failure mode that would leave the operator wrongly reassured.

        If a never-delivered alert were recorded as sent, the dedup check
        would suppress every future attempt and the operator would never
        learn of the result.
        """
        notifier, dedup = _Notifier(fail_times=99), _Dedup()
        result = send_alpha_alert(
            notifier,
            dedup,
            _confirmed(),
            recipients=["a@b.c"],
            now=NOW,
            max_attempts=3,
            sleep=lambda _s: None,
        )

        assert not result.sent
        assert result.error
        assert result.attempts == 3
        assert dedup.marked == []
        # And the next run may try again.
        assert dedup.should_send(result.hash_value)

    def test_a_dry_run_delivers_nothing_and_claims_nothing(self) -> None:
        dedup = _Dedup()
        result = send_alpha_alert(None, dedup, _confirmed(), recipients=["a@b.c"], now=NOW)

        assert not result.sent
        assert not result.suppressed_as_duplicate
        assert result.error is None
        assert dedup.marked == []

    def test_backoff_grows_between_attempts(self) -> None:
        slept: list[float] = []
        send_alpha_alert(
            _Notifier(fail_times=2),
            _Dedup(),
            _confirmed(),
            recipients=["a@b.c"],
            now=NOW,
            sleep=slept.append,
            backoff_seconds=1.5,
        )
        assert slept == [1.5, 3.0]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_attempts": 0}, "max_attempts must be >= 1"),
            ({"recipients": []}, "recipients must not be empty"),
        ],
    )
    def test_invalid_dispatch_inputs_are_refused(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        base: dict[str, object] = {"recipients": ["a@b.c"], "now": NOW}
        with pytest.raises(ValueError, match=match):
            send_alpha_alert(_Notifier(), _Dedup(), _confirmed(), **{**base, **kwargs})  # type: ignore[arg-type]


class TestKindAwareWording:
    """The condition the 2026-10-26 authorisation was granted under.

    A cost finding must never reach the operator worded as an alpha
    discovery. The subject uses a different noun and the body opens with
    an explicit denial, so neither a skim of the inbox nor a skim of the
    first lines can mislead.
    """

    def test_a_cost_edge_subject_says_kostenvorteil_not_alpha_kandidat(self) -> None:
        subject, _body = render_alpha_alert(
            _confirmed(kind=AlphaAlertKind.CANDIDATE, research_kind=ResearchKind.COST_EDGE)
        )

        assert "Kostenvorteil-Kandidat" in subject
        assert "Alpha-Kandidat" not in subject

    def test_a_cost_edge_body_opens_by_denying_it_is_alpha(self) -> None:
        _subject, body = render_alpha_alert(
            _confirmed(kind=AlphaAlertKind.CANDIDATE, research_kind=ResearchKind.COST_EDGE)
        )

        assert body.index("DIES IST KEIN ALPHA") < body.index("Status:")
        assert "sagt nichts über die" in body
        assert "kann keinen Trade profitabel machen" in body

    def test_a_risk_improvement_is_also_denied_alpha_status(self) -> None:
        _subject, body = render_alpha_alert(
            _confirmed(kind=AlphaAlertKind.CANDIDATE, research_kind=ResearchKind.RISK_IMPROVEMENT)
        )
        assert "DIES IST KEIN ALPHA" in body
        assert "erzeugt aber selbst keinen Ertrag" in body

    def test_an_alpha_hypothesis_carries_no_denial_banner(self) -> None:
        _subject, body = render_alpha_alert(
            _confirmed(kind=AlphaAlertKind.CANDIDATE, research_kind=ResearchKind.ALPHA_HYPOTHESIS)
        )
        assert "DIES IST KEIN ALPHA" not in body

    def test_the_kind_is_part_of_the_dedup_hash(self) -> None:
        """Reclassifying a question is news, so it must not be suppressed."""
        assert alpha_alert_hash(
            _confirmed(research_kind=ResearchKind.COST_EDGE)
        ) != alpha_alert_hash(_confirmed(research_kind=ResearchKind.ALPHA_HYPOTHESIS))


class TestAlertForScreen:
    """The automatic path: a completed screen becomes a message, or nothing."""

    _METRICS: ClassVar[dict[str, object]] = {
        "supported": True,
        "cells_used": 107.0,
        "products_used": 179248.0,
        "mean_spread_slope": -0.4125,
        "spread_ci_low": -0.6519,
        "spread_ci_high": -0.2113,
        "median_economic_size_bps": 19.6,
    }

    def _call(self, **over: object) -> AlphaAlert | None:
        kwargs: dict[str, object] = dict(
            hypothesis_id="RO-LOTTERY-PRICING",
            research_kind=ResearchKind.COST_EDGE,
            verdict="CANDIDATE",
            supported=True,
            description="Beispielbeschreibung.",
            mechanism="Beispielmechanismus.",
            underlyings=("DAX", "NDX"),
            horizon_days=None,
            metrics=self._METRICS,
            screen_id="2026Q4_SCREEN_x",
            run_url="https://example.invalid/run/1",
            run_id="1",
        )
        kwargs.update(over)
        return alert_for_screen(**kwargs)  # type: ignore[arg-type]

    def test_an_unsupported_screen_produces_no_message(self) -> None:
        """A screen that found nothing is not news."""
        assert self._call(supported=False, verdict="NO_IMPROVEMENT") is None

    def test_a_lookahead_suspect_cannot_be_announced(self) -> None:
        """`supported` is already false for it, and that is the guard."""
        assert self._call(supported=False, verdict="LOOKAHEAD_SUSPECT") is None

    def test_a_supported_screen_is_always_only_a_candidate(self) -> None:
        """A screen runs on DEVELOPMENT data, which is never evidence.

        No number, however large, can turn a screen into a confirmation --
        that is a property of where it ran.
        """
        alert = self._call()
        assert alert is not None
        assert alert.kind is AlphaAlertKind.CANDIDATE
        assert alert.status == "EXPLORATORY_ONLY"
        assert any("DEVELOPMENT" in limitation for limitation in alert.limitations)

    def test_the_harness_numbers_are_carried_through(self) -> None:
        alert = self._call()
        assert alert is not None
        assert alert.effect == pytest.approx(19.6)
        assert alert.ci_low == pytest.approx(-0.6519)
        assert alert.effective_sample == pytest.approx(107.0)
        assert alert.observations == 179248
        assert alert.screen_id == "2026Q4_SCREEN_x"

    def test_a_missing_interval_is_absent_rather_than_invented(self) -> None:
        alert = self._call(metrics={"supported": True, "cells_used": 50.0})
        assert alert is not None
        assert alert.ci_low is None and alert.ci_high is None
        _subject, body = render_alpha_alert(alert)
        assert "nicht bestimmbar" in body

    def test_a_cost_edge_screen_renders_without_claiming_alpha(self) -> None:
        alert = self._call()
        assert alert is not None
        subject, body = render_alpha_alert(alert)
        assert "Kostenvorteil-Kandidat" in subject
        assert "DIES IST KEIN ALPHA" in body
