"""Tests for notifications/templates.py."""

from __future__ import annotations

from datetime import UTC, date, datetime

from turboedge.notifications.templates import (
    DailyResearchProtocolEntry,
    NoActionableDigestContext,
    ScanReportContext,
    ScanReportRow,
    render_no_actionable_digest,
    render_scan_report,
    render_test_email,
)


class TestRenderTestEmail:
    """Test render_test_email()."""

    def test_render_basic(self) -> None:
        """Render a test email with basic fields."""
        now = datetime(2026, 9, 11, 10, 30, tzinfo=UTC)
        version = "0.1.0"

        subject, body = render_test_email(now, version)

        assert "Test" in subject
        assert "2026-09-11" in subject
        assert version in body
        assert "Research system — manual execution only." in body

    def test_footer_present(self) -> None:
        """Test email ends with correct footer."""
        now = datetime(2026, 9, 11, 10, 30, tzinfo=UTC)
        _, body = render_test_email(now, "0.1.0")

        assert body.endswith("Research system — manual execution only.")

    def test_subject_includes_timestamp(self) -> None:
        """Subject includes formatted timestamp."""
        now = datetime(2026, 9, 11, 15, 45, tzinfo=UTC)
        subject, _ = render_test_email(now, "0.1.0")

        assert "2026-09-11" in subject
        assert "15:45" in subject


class TestRenderScanReport:
    """Test render_scan_report()."""

    def test_render_empty_report(self) -> None:
        """Render a report with no candidates."""
        now = datetime(2026, 9, 11, 10, 30, tzinfo=UTC)
        context = ScanReportContext(
            run_id="run_001",
            underlying_id="DAX",
            generated_at=now,
            signal_score=0.65,
            direction_hint="long",
            counts={"WATCH": 5, "REJECT": 20, "DATA_QUALITY": 2},
            rows=[],
            warnings=[],
        )

        subject, body = render_scan_report(context)

        assert "DAX" in subject
        assert "5 WATCH" in subject
        assert "20 REJECT" in subject
        assert "2 DATA_QUALITY" in subject
        assert "run_001" in body
        assert "Research system — manual execution only." in body
        assert "No ACTIONABLE candidates produced" in body

    def test_render_with_candidates(self) -> None:
        """Render a report with candidates."""
        now = datetime(2026, 9, 11, 10, 30, tzinfo=UTC)
        row = ScanReportRow(
            rank=1,
            isin="DE000ABC1234",
            wkn="ABC123",
            issuer="TestBank",
            direction="long",
            category="WATCH",
            leverage=5.5,
            spread_pct=0.015,
            distance_to_barrier_pct=0.10,
            issuer_margin_pct=0.005,
            financing_cost_7d_pct=0.0007,
            liquidity_factor=0.85,
            cost_rank_score=0.002,
            reasons=["strong signal", "good liquidity"],
        )

        context = ScanReportContext(
            run_id="run_002",
            underlying_id="DAX",
            generated_at=now,
            signal_score=0.72,
            direction_hint="long",
            counts={"WATCH": 1},
            rows=[row],
            warnings=[],
        )

        _, body = render_scan_report(context)

        assert "DE000ABC1234" in body
        assert "ABC123" in body
        assert "TestBank" in body
        assert "1.50%" in body  # spread_pct formatted
        assert "10.00%" in body  # distance_to_barrier_pct formatted
        assert "strong signal" in body
        assert "good liquidity" in body

    def test_percentage_formatting(self) -> None:
        """Test percentage filter formats decimals correctly."""
        row = ScanReportRow(
            rank=1,
            isin="DE000ABC1234",
            wkn="ABC123",
            issuer="TestBank",
            direction="long",
            category="WATCH",
            leverage=5.0,
            spread_pct=0.0123,
            distance_to_barrier_pct=0.50,
            issuer_margin_pct=0.001,
            financing_cost_7d_pct=0.0,
            liquidity_factor=0.95,
            cost_rank_score=0.0015,
            reasons=[],
        )

        context = ScanReportContext(
            run_id="run_003",
            underlying_id="DAX",
            generated_at=datetime.now(UTC),
            signal_score=None,
            direction_hint=None,
            counts={},
            rows=[row],
            warnings=[],
        )

        _, body = render_scan_report(context)

        assert "1.23%" in body  # spread_pct
        assert "50.00%" in body  # distance_to_barrier_pct
        assert "0.10%" in body  # issuer_margin_pct
        assert "0.00%" in body  # financing_cost_7d_pct
        assert "0.15%" in body  # cost_rank_score

    def test_none_formatting(self) -> None:
        """Test None values render as 'n/a'."""
        row = ScanReportRow(
            rank=1,
            isin="DE000ABC1234",
            wkn=None,
            issuer="TestBank",
            direction="short",
            category="REJECT",
            leverage=None,
            spread_pct=None,
            distance_to_barrier_pct=None,
            issuer_margin_pct=None,
            financing_cost_7d_pct=None,
            liquidity_factor=None,
            cost_rank_score=None,
            reasons=["no wkn"],
        )

        context = ScanReportContext(
            run_id="run_004",
            underlying_id="DAX",
            generated_at=datetime.now(UTC),
            signal_score=None,
            direction_hint=None,
            counts={},
            rows=[row],
            warnings=[],
        )

        _, body = render_scan_report(context)

        # Count 'n/a' occurrences (should be many): leverage, spread,
        # distance, margin, financing, cost_rank_score, liquidity
        assert body.count("n/a") >= 7

    def test_footer_and_actionable_message(self) -> None:
        """Test footer and ACTIONABLE message appear."""
        context = ScanReportContext(
            run_id="run_005",
            underlying_id="DAX",
            generated_at=datetime.now(UTC),
            signal_score=None,
            direction_hint=None,
            counts={},
            rows=[],
            warnings=[],
        )

        _, body = render_scan_report(context)

        assert "Research system — manual execution only." in body
        assert "No ACTIONABLE candidates produced in practice" in body
        assert "measured edge" in body

    def test_warnings_rendered(self) -> None:
        """Test warnings are included in the report."""
        context = ScanReportContext(
            run_id="run_006",
            underlying_id="DAX",
            generated_at=datetime.now(UTC),
            signal_score=None,
            direction_hint=None,
            counts={},
            rows=[],
            warnings=["source health degraded", "low signal confidence"],
        )

        _, body = render_scan_report(context)

        assert "source health degraded" in body
        assert "low signal confidence" in body

    def test_subject_summary_counts(self) -> None:
        """Subject includes category counts in order."""
        context = ScanReportContext(
            run_id="run_007",
            underlying_id="ESTX50",
            generated_at=datetime.now(UTC),
            signal_score=None,
            direction_hint=None,
            counts={"WATCH": 3, "REJECT": 15, "DATA_QUALITY": 1},
            rows=[],
            warnings=[],
        )

        subject, _ = render_scan_report(context)

        assert "ESTX50" in subject
        assert "3 WATCH" in subject
        assert "15 REJECT" in subject
        assert "1 DATA_QUALITY" in subject


class TestScanReportRow:
    """Test ScanReportRow dataclass."""

    def test_create_with_all_fields(self) -> None:
        """Create a row with all fields."""
        row = ScanReportRow(
            rank=1,
            isin="DE000ABC1234",
            wkn="ABC123",
            issuer="TestBank",
            direction="long",
            category="WATCH",
            leverage=5.5,
            spread_pct=0.015,
            distance_to_barrier_pct=0.10,
            issuer_margin_pct=0.005,
            financing_cost_7d_pct=0.0007,
            liquidity_factor=0.85,
            cost_rank_score=0.002,
            reasons=["signal", "liquidity"],
        )

        assert row.rank == 1
        assert row.wkn == "ABC123"
        assert row.leverage == 5.5

    def test_create_with_none_values(self) -> None:
        """Create a row with None values."""
        row = ScanReportRow(
            rank=2,
            isin="DE000XYZ9999",
            wkn=None,
            issuer="Bank2",
            direction="short",
            category="REJECT",
            leverage=None,
            spread_pct=None,
            distance_to_barrier_pct=None,
            issuer_margin_pct=None,
            financing_cost_7d_pct=None,
            liquidity_factor=None,
            cost_rank_score=None,
            reasons=[],
        )

        assert row.wkn is None
        assert row.leverage is None


class TestRenderNoActionableDigest:
    """Test render_no_actionable_digest() -- the daily research protocol
    digest (Konzept 1.2 Abschnitt 16)."""

    def _entry(self, **overrides: object) -> DailyResearchProtocolEntry:
        kwargs: dict[str, object] = dict(
            underlying_id="DAX",
            status="KEIN TRADE",
            direction="long",
            signal_score=0.83,
            best_lcb_ev=-0.012,
            median_product_ev=-0.02,
            best_p_ko=0.31,
            best_distance_to_barrier_pct=0.18,
            decisive_reason="lcb_ev_not_positive",
            reject_reason_counts={"lcb_ev_not_positive": 5, "spread_too_high": 2},
            source_health_summary="bnp=PASS, citi=PASS",
            quote_age_summary="median 12.3s, max 45.0s (n=7)",
        )
        kwargs.update(overrides)
        return DailyResearchProtocolEntry(**kwargs)  # type: ignore[arg-type]

    def test_header_date_and_trial_id(self) -> None:
        ctx = NoActionableDigestContext(
            run_date=date(2026, 9, 19),
            trial_id="TR-ROUTINE-SCAN",
            entries=[self._entry()],
            no_model_beats_null_disclosure="DISCLOSURE TEXT.",
        )
        subject, body = render_no_actionable_digest(ctx)

        assert "2026-09-19" in subject
        assert body.startswith("FORSCHUNGSPROTOKOLL - KEINE HANDELSEMPFEHLUNG")
        assert "Datum: 2026-09-19" in body
        assert "Trial-ID: TR-ROUTINE-SCAN" in body

    def test_required_fields_present_per_underlying(self) -> None:
        ctx = NoActionableDigestContext(
            run_date=date(2026, 9, 19),
            trial_id="TR-ROUTINE-SCAN",
            entries=[self._entry()],
            no_model_beats_null_disclosure="DISCLOSURE TEXT.",
        )
        _, body = render_no_actionable_digest(ctx)

        assert "DAX -- Status: KEIN TRADE" in body
        assert "Richtung: long" in body
        assert "Signalstaerke s: 0.83" in body
        assert "LCB(EV) -1.20%" in body
        assert "Median-Produkt-EV: -2.00%" in body
        assert "P(KO): 31.00%" in body
        assert "Barriereabstand: 18.00%" in body
        # the ABLEHNENDE BEDINGUNG: the deciding gate reason, plus counts
        # per reason (`reject_reason_counts`, from `scan_diagnostics`).
        assert "Ablehnende Bedingung: lcb_ev_not_positive" in body
        assert "lcb_ev_not_positive: 5" in body
        assert "spread_too_high: 2" in body
        # data quality: source health + quote age.
        assert "Datenqualitaet: Quellen [bnp=PASS, citi=PASS]" in body
        assert "Kursalter: median 12.3s, max 45.0s (n=7)" in body
        assert "DISCLOSURE TEXT." in body
        assert body.endswith("Research system — manual execution only.")

    def test_multiple_underlyings_each_get_own_block(self) -> None:
        ctx = NoActionableDigestContext(
            run_date=date(2026, 9, 19),
            trial_id="TR-ROUTINE-SCAN",
            entries=[
                self._entry(underlying_id="DAX"),
                self._entry(underlying_id="XAU", status="DATENQUALITAET"),
            ],
            no_model_beats_null_disclosure="DISCLOSURE TEXT.",
        )
        _, body = render_no_actionable_digest(ctx)

        assert "DAX -- Status: KEIN TRADE" in body
        assert "XAU -- Status: DATENQUALITAET" in body

    def test_vorschlag_status_rendered(self) -> None:
        ctx = NoActionableDigestContext(
            run_date=date(2026, 9, 19),
            trial_id="TR-ROUTINE-SCAN",
            entries=[self._entry(status="VORSCHLAG", decisive_reason="all_gates_passed")],
            no_model_beats_null_disclosure="DISCLOSURE TEXT.",
        )
        _, body = render_no_actionable_digest(ctx)

        assert "DAX -- Status: VORSCHLAG" in body

    def test_none_fields_render_as_na(self) -> None:
        ctx = NoActionableDigestContext(
            run_date=date(2026, 9, 19),
            trial_id="TR-ROUTINE-SCAN",
            entries=[
                self._entry(
                    status="DATENQUALITAET",
                    direction=None,
                    signal_score=None,
                    best_lcb_ev=None,
                    median_product_ev=None,
                    best_p_ko=None,
                    best_distance_to_barrier_pct=None,
                    decisive_reason=None,
                    reject_reason_counts={},
                    source_health_summary="n/a",
                    quote_age_summary="n/a",
                )
            ],
            no_model_beats_null_disclosure="DISCLOSURE TEXT.",
        )
        _, body = render_no_actionable_digest(ctx)

        assert "Richtung: n/a" in body
        assert "Ablehnende Bedingung: n/a" in body
        assert "keine" in body  # no reject_reason_counts at all
        assert body.count("n/a") >= 5

    def test_no_underlyings_still_renders_header_and_disclosure(self) -> None:
        """A digest with zero entries (e.g. every underlying failed before
        contributing one) must still be a well-formed, sendable mail -- the
        whole point is that the user always gets exactly one mail/day."""
        ctx = NoActionableDigestContext(
            run_date=date(2026, 9, 19),
            trial_id="TR-ROUTINE-SCAN",
            entries=[],
            no_model_beats_null_disclosure="DISCLOSURE TEXT.",
        )
        _, body = render_no_actionable_digest(ctx)

        assert "FORSCHUNGSPROTOKOLL - KEINE HANDELSEMPFEHLUNG" in body
        assert "DISCLOSURE TEXT." in body


class TestScanReportContext:
    """Test ScanReportContext dataclass."""

    def test_create_context(self) -> None:
        """Create a context."""
        now = datetime(2026, 9, 11, 10, 30, tzinfo=UTC)
        context = ScanReportContext(
            run_id="run_001",
            underlying_id="DAX",
            generated_at=now,
            signal_score=0.65,
            direction_hint="long",
            counts={"WATCH": 5},
            rows=[],
            warnings=[],
        )

        assert context.run_id == "run_001"
        assert context.underlying_id == "DAX"
        assert context.signal_score == 0.65
