"""Tests for reporting/console.py: rich rendering, JSON export, plain-text report."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from turboedge.notifications.gmail import SendResult
from turboedge.pipeline.scan import ScanResult
from turboedge.pipeline.universe import UniverseResult
from turboedge.reporting.console import (
    render_scan,
    render_universe,
    scan_report_text,
    scan_result_to_json,
)
from turboedge.storage.schemas import (
    CandidateEvaluation,
    Category,
    CostDecomposition,
    Direction,
    HealthStatus,
    SignalSnapshot,
    SourceHealthRecord,
)
from turboedge.storage.snapshots import SnapshotResult

_NOW = datetime(2026, 9, 10, 15, 30, tzinfo=UTC)
_FOOTER = "Research system — manual execution only."


def _console() -> Console:
    return Console(file=io.StringIO(), width=200, no_color=True)


def _costs(
    *, ask: float = 40.10, bid: float = 40.00, issuer_margin_pct: float = 0.01
) -> CostDecomposition:
    mid = (ask + bid) / 2.0
    return CostDecomposition(
        ask=ask,
        bid=bid,
        mid=mid,
        intrinsic=mid - 0.20,
        trading_spread_component=ask - mid,
        fair_gap_premium=0.08,
        financing_drag=0.02,
        issuer_margin=0.15,
        spread_pct=0.0012,
        gap_premium_pct=0.0020,
        financing_drag_pct=0.0005,
        issuer_margin_pct=issuer_margin_pct,
    )


def _candidate(
    *,
    isin: str = "DE000WATCH01",
    wkn: str | None = None,
    direction: Direction = Direction.LONG,
    category: Category = Category.WATCH,
    reasons: list[str] | None = None,
    cost_rank_score: float | None = 0.05,
) -> CandidateEvaluation:
    costs = _costs() if category != Category.DATA_QUALITY else None
    return CandidateEvaluation(
        run_id="run1",
        candidate_id=f"cand-{isin}",
        isin=isin,
        wkn=wkn if wkn is not None else isin[-6:],
        issuer="BankA",
        underlying_id="DAX",
        direction=direction,
        category=category,
        reasons=reasons if reasons is not None else ["watch_default"],
        leverage=5.9 if costs is not None else None,
        leverage_bucket="5-6" if costs is not None else None,
        distance_to_barrier_pct=0.167 if costs is not None else None,
        distance_to_barrier_sigma=12.5 if costs is not None else None,
        costs=costs,
        realized_financing_spread=0.02 if costs is not None else None,
        financing_cost_horizon_pct=(
            {"3d": 0.002, "5d": 0.003, "7d": 0.004, "10d": 0.006, "14d": 0.008}
            if costs is not None
            else {}
        ),
        cross_issuer_residual_zscore=0.1,
        issuer_markup_score=0.0,
        quote_dislocation_score=0.1,
        wrapper_edge=0.001,
        liquidity_factor=0.8 if costs is not None else None,
        integrity_passed=category != Category.DATA_QUALITY,
        lcb_ev=None,
        cost_rank_score=cost_rank_score,
    )


def _signal(direction_hint: Direction | None = Direction.LONG) -> SignalSnapshot:
    return SignalSnapshot(
        signal_id="sig1",
        signal_version_hash="sig-hash",
        underlying_id="DAX",
        prediction_time=_NOW,
        frozen_at=_NOW,
        score=0.83,
        components={"z_21": 1.2, "z_63": 0.9, "z_126": 0.1},
        direction_hint=direction_hint,
        threshold=0.5,
        config_hash="cfg-hash",
        git_commit="abc123",
        data_snapshot_hash="data-hash",
    )


def _health(
    source: str = "source_a", status: HealthStatus = HealthStatus.PASS
) -> SourceHealthRecord:
    return SourceHealthRecord(
        source=source,
        checked_at=_NOW,
        availability=1.0,
        freshness=1.0,
        missingness=0.0,
        schema_consistency=1.0,
        cross_source_agreement=None,
        score=1.0,
        status=status,
        message="ok",
    )


def _counts(candidates: Sequence[CandidateEvaluation]) -> dict[Category, int]:
    counts: dict[Category, int] = dict.fromkeys(Category, 0)
    for c in candidates:
        counts[c.category] += 1
    return counts


_UNSET = object()  # distinguishes "use the default signal" from an explicit signal=None


def _scan_result(
    *,
    candidates: list[CandidateEvaluation] | None = None,
    signal: SignalSnapshot | object | None = _UNSET,
    warnings: list[str] | None = None,
    notification: SendResult | None = None,
) -> ScanResult:
    cands = candidates if candidates is not None else [_candidate()]
    resolved_signal = _signal() if signal is _UNSET else signal
    return ScanResult(
        run_id="run1",
        signal=resolved_signal,  # type: ignore[arg-type]
        candidates=cands,
        counts=_counts(cands),
        warnings=warnings or [],
        health=[_health()],
        notification=notification,
    )


# --------------------------------------------------------------------------
# render_scan / render_universe (console)
# --------------------------------------------------------------------------


def test_render_scan_smoke_shows_signal_counts_and_footer() -> None:
    result = _scan_result(
        candidates=[
            _candidate(isin="DE000LONG001", category=Category.WATCH, cost_rank_score=0.03),
            _candidate(
                isin="DE000SHRT001",
                direction=Direction.SHORT,
                category=Category.REJECT,
                reasons=["spread_too_high"],
                cost_rank_score=None,
            ),
            _candidate(
                isin="DE000BADR001", category=Category.DATA_QUALITY, reasons=["fx_unavailable"]
            ),
        ],
        warnings=["fx_daily_close_approximation"],
    )
    console = _console()
    render_scan(result, console, top=10)
    output = console.file.getvalue()

    assert "DAX" in output
    assert "score=0.8300" in output or "0.83" in output
    assert "WATCH=1" in output
    assert "REJECT=1" in output
    assert "DATA_QUALITY=1" in output
    assert "DE000LONG001"[-6:] in output  # WKN/ISIN column shows the derived WKN
    assert "fx_daily_close_approximation" in output
    assert _FOOTER in output


def test_render_scan_signal_none_shows_unavailable_message() -> None:
    result = _scan_result(signal=None)
    console = _console()
    render_scan(result, console, top=5)
    output = console.file.getvalue()
    assert "Signal unavailable" in output
    assert _FOOTER in output


def test_render_scan_shows_notification_status() -> None:
    notification = SendResult(
        sent=True, dry_run=False, recipients=["a@example.com"], message="sent"
    )
    result = _scan_result(notification=notification)
    console = _console()
    render_scan(result, console, top=5)
    output = console.file.getvalue()
    assert "Notification: sent" in output


def test_render_universe_smoke_shows_counts_conflicts_and_footer(tmp_path: Path) -> None:
    from turboedge.universe.discover import FieldConflict

    snapshot = SnapshotResult(
        path=tmp_path / "snap.parquet",
        table="product_snapshots",
        run_id="run1",
        row_count=2,
        data_snapshot_hash="a" * 64,
    )
    result = UniverseResult(
        run_id="run1",
        products=[],  # rendering only reads len(), not per-product content
        conflicts=[
            FieldConflict(
                isin="DE000CONF001",
                field="financing_level",
                values=(20000.0, 20500.0),
                sources=("a", "b"),
            )
        ],
        source_errors={"source_broken": "timeout"},
        counts_by_source={"source_a": 3, "source_b": 2},
        snapshot=snapshot,
    )
    console = _console()
    render_universe(result, console)
    output = console.file.getvalue()

    assert "source_a" in output and "source_b" in output
    assert "source_broken" in output and "timeout" in output
    assert "DE000CONF001" in output
    assert "Snapshot:" in output
    assert _FOOTER in output


# --------------------------------------------------------------------------
# scan_result_to_json
# --------------------------------------------------------------------------


def test_scan_result_to_json_is_serializable_and_never_actionable() -> None:
    result = _scan_result(
        candidates=[
            _candidate(isin="DE000LONG001", category=Category.WATCH),
            _candidate(isin="DE000BADR001", category=Category.DATA_QUALITY),
        ]
    )
    payload = scan_result_to_json(result)

    # round-trips through the standard library JSON encoder without error
    text = json.dumps(payload)
    reloaded = json.loads(text)

    assert reloaded["run_id"] == "run1"
    assert reloaded["footer"] == _FOOTER
    assert reloaded["counts"]["ACTIONABLE"] == 0
    categories = {c["category"] for c in reloaded["candidates"]}
    assert "ACTIONABLE" not in categories
    assert categories == {"WATCH", "DATA_QUALITY"}
    assert reloaded["signal"]["underlying_id"] == "DAX"


def test_scan_result_to_json_handles_missing_signal_and_notification() -> None:
    result = _scan_result(signal=None, notification=None)
    payload = scan_result_to_json(result)
    assert payload["signal"] is None
    assert payload["notification"] is None
    json.dumps(payload)  # still serializable


# --------------------------------------------------------------------------
# scan_report_text -- contract test (j): footer present, never ACTIONABLE
# --------------------------------------------------------------------------


def test_scan_report_text_has_footer_and_no_actionable_candidates() -> None:
    result = _scan_result(
        candidates=[
            _candidate(isin="DE000LONG001", category=Category.WATCH),
            _candidate(isin="DE000SHRT001", direction=Direction.SHORT, category=Category.REJECT),
            _candidate(isin="DE000BADR001", category=Category.DATA_QUALITY),
        ]
    )
    text = scan_report_text(result, top=10)

    assert text.endswith(_FOOTER)
    assert "Category: ACTIONABLE" not in text
    assert "No ACTIONABLE category in this milestone" in text
    assert "DE000LONG001" in text
    assert "Run ID: run1" in text
    assert "Underlying: DAX" in text


def test_scan_report_text_falls_back_to_candidate_underlying_id_when_no_signal() -> None:
    result = _scan_result(signal=None, candidates=[_candidate(isin="DE000LONG001")])
    text = scan_report_text(result, top=5)
    assert "Underlying: DAX" in text
    assert text.endswith(_FOOTER)


def test_scan_report_text_empty_candidates_still_has_footer() -> None:
    result = _scan_result(candidates=[], signal=_signal())
    # empty candidates list forces the underlying_id fallback to "unknown"
    # only when signal is also None; here signal still carries it
    text = scan_report_text(result, top=5)
    assert "No candidates in this milestone." in text
    assert text.endswith(_FOOTER)
