"""End-to-end wiring test for Phase F's shadow portfolio (Master Spec §46):
`run_scan` -> `_build_shadow_positions` -> `store.append_shadow_positions`.

`test_scan_shadow_positions.py` already covers each arm's selection logic
exhaustively at the unit level (deterministic, hand-built pools); this file
only proves the real pipeline actually persists rows end-to-end, through
`shadow_portfolio` (previously always empty -- 0 rows, never written),
using the same stochastic forecast/path-simulation machinery a real
`turboedge scan-all` run uses.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from turboedge.config import TurboEdgeConfig
from turboedge.pipeline.scan import ScanOptions, run_scan
from turboedge.provenance import new_run_id
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import Direction, ProductSnapshot, ShadowPortfolioKind

_EVAL_TIME = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)


def _clock(when: datetime = _EVAL_TIME) -> Callable[[], datetime]:
    return lambda: when


def _diverse_long_products(
    dax_product_factory: Callable[..., ProductSnapshot],
) -> list[ProductSnapshot]:
    """10 LONG open-end DAX turbos, cheap relative to intrinsic (so all
    clear the hard pre-EV gates), with distinct financing levels
    (-> distinct leverage) and distinct ask/bid offsets (-> distinct
    spread) across a few issuers -- enough diversity that TOP1/TOP3/TOP5,
    RANDOM_VALID_TURBO, the extremum arms and MEDIAN_PRODUCT are not all
    forced to coincide on the same single candidate."""
    products = []
    issuers = ["BankA", "BankB", "BankC"]
    for i in range(10):
        financing_level = 19500.0 - i * 300.0  # spot=24000 -> leverage increases with i
        product = dax_product_factory(
            isin=f"DE000DIVLNG{i}",
            issuer=issuers[i % len(issuers)],
            direction=Direction.LONG,
            financing_level=financing_level,
            ratio=0.01,
            quote_timestamp=_EVAL_TIME,
        )
        intrinsic = (24000.0 - financing_level) * 0.01
        ask = round(intrinsic + 0.05 + 0.01 * i, 2)  # distinct spread per product
        bid = round(ask - 0.05 - 0.005 * i, 2)
        products.append(product.model_copy(update={"ask": ask, "bid": bid}))
    return products


def test_shadow_positions_persisted_end_to_end_with_consistent_direction_time_horizon(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
    dax_product_factory: Callable[..., ProductSnapshot],
    bars_factory: Callable[..., Any],
) -> None:
    assert store.table_counts()["shadow_portfolio"] == 0, "shadow_portfolio starts empty"

    products = _diverse_long_products(dax_product_factory)
    bars = bars_factory("DAX", count=400, daily_drift=0.0025, noise_std=0.0015, seed=7)
    adapter = make_product_adapter("source_a", products=products)
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": bars})

    result = run_scan(
        cfg=cfg,
        store=store,
        state_dir=tmp_path,
        product_adapters=[adapter],
        price_adapter=price_adapter,
        estr_adapter=make_estr_adapter(),
        reference_healthchecks=[],
        notifier=None,
        run_id=new_run_id(),
        clock=_clock(),
        rng=np.random.default_rng(2026),
        run_ev=True,
        options=ScanOptions(underlying_id="DAX"),
    )

    assert "ev_pool_empty" not in result.warnings
    assert result.shadow_positions_written > 0, "shadow_portfolio must no longer be empty"
    assert store.table_counts()["shadow_portfolio"] == result.shadow_positions_written

    shadow_rows = store.list_shadow_positions(run_id=result.run_id)
    assert shadow_rows, "expected at least one shadow row for this run"

    by_kind: dict[str, list[Any]] = {}
    for row in shadow_rows:
        by_kind.setdefault(row.portfolio.value, []).append(row)

    # -- every single-pick arm: at most one row; TOP3/TOP5: the documented
    # one-row-per-constituent exception (see _build_shadow_positions).
    single_pick = {
        ShadowPortfolioKind.TOP1,
        ShadowPortfolioKind.RANDOM_VALID_TURBO,
        ShadowPortfolioKind.LOWEST_SPREAD,
        ShadowPortfolioKind.LOWEST_FINANCING_COST,
        ShadowPortfolioKind.HIGHEST_LEVERAGE,
        ShadowPortfolioKind.LOWEST_LEVERAGE,
        ShadowPortfolioKind.MEDIAN_PRODUCT,
    }
    for kind in single_pick:
        assert len(by_kind.get(kind.value, [])) <= 1, f"{kind.value} wrote more than one row"
    assert len(by_kind.get(ShadowPortfolioKind.TOP3.value, [])) <= 3
    assert len(by_kind.get(ShadowPortfolioKind.TOP5.value, [])) <= 5

    # -- identical direction/horizon/exit_due across every arm (the entire
    # point of the experiment -- see _build_shadow_positions's docstring).
    top1_row = by_kind[ShadowPortfolioKind.TOP1.value][0]
    expected_horizon = top1_row.horizon_days
    expected_exit_due = top1_row.exit_due
    expected_direction = store.get_instrument(top1_row.isin).direction
    assert expected_direction == Direction.LONG

    for row in shadow_rows:
        assert row.horizon_days == expected_horizon
        assert row.exit_due == expected_exit_due
        instrument = store.get_instrument(row.isin)
        assert instrument is not None
        assert instrument.direction == expected_direction

    # -- cross-check against the real ledger: the shadow TOP1 row must be
    # exactly the same (isin, horizon, direction) as the real, highest-
    # ranked forward-ledger entry this scan wrote (the actual TurboEdge
    # selection the whole experiment is benchmarked against).
    ledger_rows = store.list_ledger_entries(run_id=result.run_id, is_shadow=False)
    assert ledger_rows, "expected at least one non-shadow ledger entry"
    real_top = next((e for e, _label in ledger_rows if e.selected_isin == top1_row.isin), None)
    assert real_top is not None, "shadow TOP1's isin must also be a real ledger entry this scan"
    assert real_top.horizon_days == expected_horizon
    assert real_top.direction == expected_direction
    assert real_top.exit_due == expected_exit_due

    # -- printed for the human report (isins/entry_asks per arm).
    print(f"\nShadow portfolio rows (run_id={result.run_id}):")
    for kind_value, rows in sorted(by_kind.items()):
        summary = ", ".join(f"{r.isin}@{r.entry_ask:.2f}" for r in rows)
        print(f"  {kind_value}: n={len(rows)} [{summary}]")
    print(
        f"  horizon_days={expected_horizon} exit_due={expected_exit_due} "
        f"direction={expected_direction.value}"
    )
