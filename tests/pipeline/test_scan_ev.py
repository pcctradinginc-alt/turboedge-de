"""Tests for the Contract v3 EV pipeline extension in ``pipeline/scan.py``.

The EV pipeline (forecast -> paths -> EV -> gates -> ledger + ACTIONABLE
mail) runs by default (``run_scan(..., run_ev=True)``, matching production
-- both ``turboedge scan`` and ``turboedge scan-all`` since Befund 1,
2026-09-14). Tests here pass an explicit seeded ``rng`` to control which
random stream drives the path simulation, or ``run_ev=False`` to prove the
one remaining, test-only escape hatch (isolating the hard gates from this
slower, stochastic layer) still works. No network; all adapters are
in-memory fakes (see ``tests/pipeline/conftest.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import structlog.testing

from turboedge.config import TurboEdgeConfig
from turboedge.pipeline.scan import ScanOptions, run_scan
from turboedge.provenance import new_run_id
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Category,
    Direction,
    FieldReliability,
    LedgerEntryStatus,
    ProductSnapshot,
)

_EVAL_TIME = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)


def _clock(when: datetime = _EVAL_TIME) -> Callable[[], datetime]:
    return lambda: when


def _base_kwargs(
    *,
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    product_adapters: list[Any],
    price_adapter: Any,
    estr_adapter: Any,
    notifier: Any | None = None,
    rng: np.random.Generator | None = None,
    run_ev: bool = True,
) -> dict[str, Any]:
    return dict(
        cfg=cfg,
        store=store,
        state_dir=tmp_path,
        product_adapters=product_adapters,
        price_adapter=price_adapter,
        estr_adapter=estr_adapter,
        reference_healthchecks=[],
        notifier=notifier,
        run_id=new_run_id(),
        clock=_clock(),
        rng=rng,
        run_ev=run_ev,
    )


def _cheap_favorable_long(
    dax_product_factory: Callable[..., ProductSnapshot], **overrides: Any
) -> ProductSnapshot:
    """A long open-end turbo priced almost exactly at intrinsic (negligible
    issuer markup/spread) with a barrier far below spot -- economically the
    best possible case for a strong, low-noise uptrend to clear every EV
    gate (LCB(EV) > 0, P(KO) known, cluster risk in range)."""
    kwargs: dict[str, Any] = dict(
        isin="DE000FAVLNG1",
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=18000.0,  # far below spot (24000) -> tiny P(KO)
        ratio=0.01,
        quote_timestamp=_EVAL_TIME,
    )
    kwargs.update(overrides)
    product = dax_product_factory(**kwargs)
    # intrinsic = (24000-18000)*0.01 = 60.0; tight, cheap quote around it.
    return product.model_copy(update={"ask": 60.05, "bid": 60.03})


@pytest.fixture
def strong_uptrend_bars(bars_factory: Callable[..., Any]) -> Any:
    """400 trading days of low-noise, strongly positive-drift synthetic DAX
    bars -- deterministic (seeded), enough history for every default
    forecast model (Tsmom/Logistic/Null) to fit successfully."""
    return bars_factory("DAX", count=400, daily_drift=0.0025, noise_std=0.0015, seed=7)


def test_run_scan_with_run_ev_false_never_runs_ev_pipeline(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
    strong_uptrend_bars: Any,
) -> None:
    """``run_ev=False`` must never write forecasts/ledger entries, regardless
    of how favorable the candidate is -- the one remaining, explicit,
    test-only escape hatch from the EV pipeline (Befund 1, 2026-09-14: this
    used to be ``rng=None``, which was also every pre-existing caller's
    silent default -- including ``turboedge scan``'s, which is exactly the
    bug that measurement session found and this test now guards against
    regressing back to)."""
    adapter = make_product_adapter(
        "source_a", products=[_cheap_favorable_long(dax_product_factory)]
    )
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": strong_uptrend_bars})

    result = run_scan(
        **_base_kwargs(
            cfg=cfg,
            store=store,
            tmp_path=tmp_path,
            product_adapters=[adapter],
            price_adapter=price_adapter,
            estr_adapter=make_estr_adapter(),
            run_ev=False,
        ),
        options=ScanOptions(underlying_id="DAX"),
    )

    assert result.forecasts_written == 0
    assert result.ledger_entries_written == 0
    assert store.table_counts()["forecasts"] == 0
    assert store.table_counts()["forward_ledger"] == 0
    assert all(c.category != Category.ACTIONABLE for c in result.candidates)


def test_run_scan_default_auto_derives_rng_and_runs_ev_pipeline(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
    strong_uptrend_bars: Any,
) -> None:
    """Befund 1 (2026-09-14 measurement session): with neither ``rng`` nor
    ``run_ev`` passed -- i.e. exactly what ``turboedge scan``'s CLI command
    does -- ``run_scan`` must still auto-derive a seeded generator from
    ``cfg.simulation.seed`` and run the full EV pipeline, identically to how
    ``turboedge scan-all`` has always behaved. This is the unification fix:
    before it, this call silently produced the old WATCH-only Phase-1
    result instead."""
    adapter = make_product_adapter(
        "source_a", products=[_cheap_favorable_long(dax_product_factory)]
    )
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": strong_uptrend_bars})

    result = run_scan(
        **_base_kwargs(
            cfg=cfg,
            store=store,
            tmp_path=tmp_path,
            product_adapters=[adapter],
            price_adapter=price_adapter,
            estr_adapter=make_estr_adapter(),
        ),
        options=ScanOptions(underlying_id="DAX"),
    )

    assert result.forecasts_written > 0
    assert store.table_counts()["forecasts"] > 0
    assert "ev_pool_empty" not in result.warnings
    assert "forecast_unavailable_insufficient_history" not in result.warnings
    candidate = next(c for c in result.candidates if c.isin == "DE000FAVLNG1")
    assert any(r.startswith("ev_horizon=") for r in candidate.reasons)


def test_run_scan_with_rng_writes_forecasts_and_ledger_and_can_reach_actionable(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
    make_notifier: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
    strong_uptrend_bars: Any,
) -> None:
    """With a seeded ``rng`` and a strong, low-noise uptrend feeding a
    near-zero-cost long candidate, the EV pipeline runs to completion:
    forecasts and forward-ledger entries are persisted, and the candidate
    that clears every gate is promoted to ACTIONABLE with a §34 trade
    proposal email carrying both mandatory disclosures.
    """
    adapter = make_product_adapter(
        "source_a", products=[_cheap_favorable_long(dax_product_factory)]
    )
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": strong_uptrend_bars})
    notifier = make_notifier()

    result = run_scan(
        **_base_kwargs(
            cfg=cfg,
            store=store,
            tmp_path=tmp_path,
            product_adapters=[adapter],
            price_adapter=price_adapter,
            estr_adapter=make_estr_adapter(),
            notifier=notifier,
            rng=np.random.default_rng(1234),
        ),
        options=ScanOptions(underlying_id="DAX", email=True),
    )

    assert result.forecasts_written > 0
    assert store.table_counts()["forecasts"] > 0
    assert "ev_pool_empty" not in result.warnings
    assert "forecast_unavailable_insufficient_history" not in result.warnings

    ledger_rows = store.list_ledger_entries()
    assert len(ledger_rows) >= 1
    for entry, _label in ledger_rows:
        assert entry.status == LedgerEntryStatus.OPEN
        assert entry.trial_id
        assert entry.config_hash
        assert entry.feature_hash
        assert entry.model_hash

    candidate = next(c for c in result.candidates if c.isin == "DE000FAVLNG1")
    if candidate.category == Category.ACTIONABLE:
        assert candidate.lcb_ev is not None and candidate.lcb_ev > 0
        assert result.actionable_notifications, "ACTIONABLE candidate should trigger a mail"
        sent_spec = notifier.sent_specs[0]
        assert "P(KO)" in sent_spec.body_text
        assert "Nullmodell" in sent_spec.body_text
        assert "manual execution only" in sent_spec.body_text
    else:
        # Even if this seed does not clear every gate, the reasons must show
        # exactly which EV gate stopped it (never a silent WATCH).
        assert any(r.startswith("ev_horizon=") for r in candidate.reasons)


def test_ratio_unverified_prevents_actionable_end_to_end(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
    make_notifier: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
    strong_uptrend_bars: Any,
) -> None:
    """Phase B ("Produktstammdaten haerten"): the exact same near-zero-cost,
    strong-uptrend setup that reaches ACTIONABLE in
    ``test_run_scan_with_rng_writes_forecasts_and_ledger_and_can_reach_actionable``
    must never do so once the candidate's Bezugsverhaeltnis is UNVERIFIED --
    it still runs the full EV pipeline and lands in the ledger/shadow sample,
    just never as ACTIONABLE.
    """
    product = _cheap_favorable_long(dax_product_factory).model_copy(
        update={"ratio_reliability": FieldReliability.UNVERIFIED}
    )
    assert product.ratio_reliability == FieldReliability.UNVERIFIED
    adapter = make_product_adapter("source_a", products=[product])
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": strong_uptrend_bars})
    notifier = make_notifier()

    result = run_scan(
        **_base_kwargs(
            cfg=cfg,
            store=store,
            tmp_path=tmp_path,
            product_adapters=[adapter],
            price_adapter=price_adapter,
            estr_adapter=make_estr_adapter(),
            notifier=notifier,
            rng=np.random.default_rng(1234),  # same seed as the sibling ACTIONABLE-reaching test
        ),
        options=ScanOptions(underlying_id="DAX", email=True),
    )

    assert result.forecasts_written > 0
    candidate = next(c for c in result.candidates if c.isin == "DE000FAVLNG1")
    assert candidate.category != Category.ACTIONABLE
    if candidate.category == Category.WATCH:
        assert "ratio_unverified" in candidate.reasons

    # Still fully present in the forward ledger (Master Spec Sec25/46: never
    # dropped or hidden just because it can't be ACTIONABLE) -- we need it
    # to measure future improvements to master-data verification.
    ledger_rows = store.list_ledger_entries()
    assert any(entry.selected_isin == "DE000FAVLNG1" for entry, _label in ledger_rows)


def test_run_scan_with_rng_and_too_little_history_degrades_gracefully(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
    bars_factory: Callable[..., Any],
) -> None:
    """Too few underlying bars for any forecast model to fit must never
    crash the scan -- it degrades to the WATCH-only behavior with a
    documented warning (Contract v3 Verbindliche Entscheidung 5)."""
    # Even the NullModel (the lowest bar, min_train_samples=20) cannot fit
    # from this little history at every horizon: n - h < 20 for every h in
    # (3, 5, 7, 10, 14) when n=15.
    short_bars = bars_factory("DAX", count=15, seed=3)
    adapter = make_product_adapter(
        "source_a", products=[_cheap_favorable_long(dax_product_factory)]
    )
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": short_bars})

    result = run_scan(
        **_base_kwargs(
            cfg=cfg,
            store=store,
            tmp_path=tmp_path,
            product_adapters=[adapter],
            price_adapter=price_adapter,
            estr_adapter=make_estr_adapter(),
            rng=np.random.default_rng(99),
        ),
        options=ScanOptions(underlying_id="DAX"),
    )

    assert "forecast_unavailable_insufficient_history" in result.warnings
    assert result.forecasts_written == 0
    assert result.ledger_entries_written == 0
    assert all(c.category != Category.ACTIONABLE for c in result.candidates)


def test_shadow_sample_drawn_from_full_pool_includes_non_watch_categories(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
    strong_uptrend_bars: Any,
) -> None:
    """The stratified shadow sample must be able to include a REJECTed
    candidate too (Spec Sec 25 selection-bias protection: not just WATCH/
    ACTIONABLE), as long as it has pricing (ask) at all."""
    good = _cheap_favorable_long(dax_product_factory)
    # A REJECT candidate: spread far above configs/risk.yaml's max_spread_pct.
    wide_spread = dax_product_factory(
        isin="DE000WIDESP1",
        issuer="BankC",
        direction=Direction.LONG,
        financing_level=18000.0,
        quote_timestamp=_EVAL_TIME,
    ).model_copy(update={"ask": 100.0, "bid": 70.0})

    adapter = make_product_adapter("source_a", products=[good, wide_spread])
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": strong_uptrend_bars})

    result = run_scan(
        **_base_kwargs(
            cfg=cfg,
            store=store,
            tmp_path=tmp_path,
            product_adapters=[adapter],
            price_adapter=price_adapter,
            estr_adapter=make_estr_adapter(),
            rng=np.random.default_rng(42),
        ),
        options=ScanOptions(underlying_id="DAX"),
    )
    reject = next(c for c in result.candidates if c.isin == "DE000WIDESP1")
    assert reject.category in (Category.REJECT, Category.DATA_QUALITY)
    original_category = reject.category

    ledger_rows = store.list_ledger_entries()
    isins_in_ledger = {e.selected_isin for e, _l in ledger_rows}
    # The discarded candidate may or may not be drawn into the (small,
    # per-stratum-capped) shadow sample given the seed -- but if it IS
    # simulated, its ledger entry must be marked is_shadow and its category
    # must stay whatever the hard gates already decided (EV can never
    # resurrect a hard-gate REJECT/DATA_QUALITY into ACTIONABLE).
    if "DE000WIDESP1" in isins_in_ledger:
        entry = next(e for e, _l in ledger_rows if e.selected_isin == "DE000WIDESP1")
        assert entry.is_shadow is True
        assert entry.category == original_category


def test_prefiltered_out_watch_candidates_are_marked_not_evaluated(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
    strong_uptrend_bars: Any,
) -> None:
    """Befund 4 (2026-09-13 measurement session): a WATCH candidate that
    passed every pre-EV gate but lost out to configs/ranking.yaml's
    ``max_candidates_per_bucket`` (25) cap must stay WATCH -- never REJECT --
    with an explicit ``not_evaluated_prefilter`` reason, so it is not
    indistinguishable from a candidate that was actively rejected (stale,
    bid_only, leverage out of range, barrier too close)."""
    # 40 near-identical, gate-passing long candidates in the same
    # (direction, leverage_bucket) -- comfortably more than the 25 cap, so
    # at least a few are guaranteed to be left out of the EV pool even after
    # the (3-per-stratum) shadow sample on top of it.
    products = [
        _cheap_favorable_long(
            dax_product_factory,
            isin=f"DE000PFILT{i:02d}",
            issuer="BankA",
            financing_level=18000.0 - i * 2.0,
        )
        for i in range(40)
    ]
    adapter = make_product_adapter("source_a", products=products)
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": strong_uptrend_bars})

    result = run_scan(
        **_base_kwargs(
            cfg=cfg,
            store=store,
            tmp_path=tmp_path,
            product_adapters=[adapter],
            price_adapter=price_adapter,
            estr_adapter=make_estr_adapter(),
            rng=np.random.default_rng(7),
        ),
        options=ScanOptions(underlying_id="DAX"),
    )

    # Every gate-passing candidate stays WATCH or is promoted to ACTIONABLE
    # by the EV pipeline -- none of these 40 has a genuine reject reason
    # (stale/bid_only/leverage/barrier), so none may end up REJECT.
    non_reject = [
        c for c in result.candidates if c.category in (Category.WATCH, Category.ACTIONABLE)
    ]
    assert len(non_reject) == 40

    not_evaluated = [c for c in non_reject if "not_evaluated_prefilter" in c.reasons]
    evaluated = [c for c in non_reject if any(r.startswith("ev_horizon=") for r in c.reasons)]
    assert not_evaluated, "at least some candidates must have lost out to the bucket cap"
    assert evaluated, "at least some candidates must have actually been simulated"
    # The two groups are mutually exclusive and exhaustive: a candidate is
    # either genuinely simulated (carries the EV pipeline's own per-horizon
    # reason) or explicitly marked as skipped by the prefilter -- never
    # neither, and never both.
    assert set(c.candidate_id for c in not_evaluated).isdisjoint(c.candidate_id for c in evaluated)
    assert len(not_evaluated) + len(evaluated) == len(non_reject)


def test_scan_diagnostics_counts_watch_reasons_not_only_reject_reasons(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
    strong_uptrend_bars: Any,
) -> None:
    """``scan_diagnostics`` must tally WATCH reasons, not only REJECT ones.

    Until 2026-09-20 it counted ``Category.REJECT`` exclusively, so a scan
    whose candidates all stopped one step short of ACTIONABLE logged *that*
    they did but never *which* precondition was missing --
    ``evaluate_gates`` computes exactly that and it was discarded one line
    later. The distinction decides how the result must be read: an honest
    measured "expected value is not positive" (``lcb_ev_not_positive``) and
    a structural "expected value was never computed for this candidate"
    (``lcb_ev_not_evaluated``) look identical from outside the process
    otherwise, and only the second one would be a defect.
    """
    products = [
        _cheap_favorable_long(
            dax_product_factory,
            isin=f"DE000WRSNS{i:02d}",
            issuer="BankA",
            financing_level=18000.0 - i * 2.0,
        )
        for i in range(40)
    ]
    adapter = make_product_adapter("source_a", products=products)
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": strong_uptrend_bars})

    with structlog.testing.capture_logs() as logs:
        result = run_scan(
            **_base_kwargs(
                cfg=cfg,
                store=store,
                tmp_path=tmp_path,
                product_adapters=[adapter],
                price_adapter=price_adapter,
                estr_adapter=make_estr_adapter(),
                rng=np.random.default_rng(7),
            ),
            options=ScanOptions(underlying_id="DAX"),
        )

    diagnostics = [entry for entry in logs if entry.get("event") == "scan_diagnostics"]
    assert len(diagnostics) == 1, "exactly one scan_diagnostics line per scanned underlying"
    watch_counts = diagnostics[0]["watch_reason_counts"]

    watch_candidates = [c for c in result.candidates if c.category == Category.WATCH]
    assert watch_candidates, "this fixture is built to leave WATCH candidates behind"

    # The tally must reproduce the candidates' own reasons exactly -- it is a
    # view onto them, never an independent (and therefore driftable) second
    # derivation.
    expected: dict[str, int] = {}
    for candidate in watch_candidates:
        for reason in candidate.reasons:
            expected[reason] = expected.get(reason, 0) + 1
    assert watch_counts == expected

    # REJECT counting stays exactly as it was: the two tallies are disjoint
    # views, and adding one must not silently fold the other into it.
    reject_counts = diagnostics[0]["reject_reason_counts"]
    reject_candidates = [c for c in result.candidates if c.category == Category.REJECT]
    assert sum(reject_counts.values()) == sum(len(c.reasons) for c in reject_candidates)


def test_ev_evaluated_candidates_persist_raw_ko_probability(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
    strong_uptrend_bars: Any,
) -> None:
    """Every EV-evaluated candidate keeps its raw path-simulation P(KO).

    ``ranking/gates.py`` already gates on this number, and the simulation
    already produces it -- but until 2026-09-20 ``pipeline/scan.py`` dropped
    it instead of writing it to the candidate, leaving ``p_ko_raw`` NULL in
    all 65,095 rows of the local state database. A turbo's knock-out
    probability is its central risk figure; discarding it after every scan
    makes the persisted candidate unreproducible in the sense of rule 33.

    ``p_ko_calibrated``/``ko_calibrator_version`` must stay ``None``: no
    calibrator has been promoted, and there is no runtime-applicable
    calibrator artifact -- writing the raw value into a field named
    *calibrated* would misrepresent it.
    """
    adapter = make_product_adapter(
        "source_a", products=[_cheap_favorable_long(dax_product_factory)]
    )
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": strong_uptrend_bars})

    result = run_scan(
        **_base_kwargs(
            cfg=cfg,
            store=store,
            tmp_path=tmp_path,
            product_adapters=[adapter],
            price_adapter=price_adapter,
            estr_adapter=make_estr_adapter(),
            rng=np.random.default_rng(1234),
        ),
        options=ScanOptions(underlying_id="DAX"),
    )

    evaluated = [
        c for c in result.candidates if any(r.startswith("ev_horizon=") for r in c.reasons)
    ]
    assert evaluated, "this fixture is built so the EV pipeline actually runs"
    for candidate in evaluated:
        assert candidate.p_ko_raw is not None, f"{candidate.isin} lost its simulated P(KO)"
        assert 0.0 <= candidate.p_ko_raw <= 1.0
        assert candidate.p_ko_calibrated is None
        assert candidate.ko_calibrator_version is None

    # It must survive the round trip through DuckDB, not just live on the
    # in-memory result object -- persistence is the entire point.
    persisted = {c.isin: c for c in store.list_candidates()}
    for candidate in evaluated:
        assert persisted[candidate.isin].p_ko_raw == pytest.approx(candidate.p_ko_raw)
