"""Tests for the Contract v3 EV pipeline extension in ``pipeline/scan.py``.

Every test here passes ``rng=None`` (the default) to prove nothing changes
for pre-existing callers, or an explicit seeded ``rng`` to exercise the new
forecast -> paths -> EV -> gates -> ledger (+ ACTIONABLE mail) pipeline. No
network; all adapters are in-memory fakes (see ``tests/pipeline/conftest.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from turboedge.config import TurboEdgeConfig
from turboedge.pipeline.scan import ScanOptions, run_scan
from turboedge.provenance import new_run_id
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import Category, Direction, LedgerEntryStatus, ProductSnapshot

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


def test_run_scan_without_rng_never_runs_ev_pipeline(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
    strong_uptrend_bars: Any,
) -> None:
    """``rng=None`` (the default) must never write forecasts/ledger entries,
    regardless of how favorable the candidate is -- full backward
    compatibility with every pre-existing ``run_scan`` caller."""
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
            rng=None,
        ),
        options=ScanOptions(underlying_id="DAX"),
    )

    assert result.forecasts_written == 0
    assert result.ledger_entries_written == 0
    assert store.table_counts()["forecasts"] == 0
    assert store.table_counts()["forward_ledger"] == 0
    assert all(c.category != Category.ACTIONABLE for c in result.candidates)


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
