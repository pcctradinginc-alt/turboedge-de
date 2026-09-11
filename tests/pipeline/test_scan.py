"""Tests for pipeline/scan.py. No network; all adapters are in-memory fakes.

Covers the mandatory scenarios from the build contract:
(a) the signal is persisted before any product source is queried
(b) ACTIONABLE is never assigned
(c) a ratio-factor-100 error is downgraded to DATA_QUALITY
(d) bid_only -> REJECT; a stale quote -> REJECT ("quote_stale"); a missing
    ask -> REJECT ("no_ask_quote", ask-dependent pricing steps skipped); a
    missing financing_level -> DATA_QUALITY
(e) every product source failing raises NoProductsError and the run's
    status is recorded as "error"
(f) one source failing while another succeeds still completes the scan,
    surfacing the failure as a warning
(g) a second scan on a later day, with an updated financing level, recovers
    a realized financing spread from history instead of the config default
(h) underlying bars whose available_at is after prediction_time are ignored
(i) a second, identical scan does not re-send the email report (dedup)
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from turboedge.adapters.base import AdapterError
from turboedge.config import TurboEdgeConfig
from turboedge.pipeline.scan import ScanOptions, run_scan
from turboedge.pipeline.universe import NoProductsError
from turboedge.provenance import new_run_id
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import Category, Direction, ProductSnapshot, UnderlyingBar

_EVAL_TIME = datetime(2025, 10, 8, 9, 0, tzinfo=UTC)  # Wed, after the last synthetic bar


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
    run_id: str | None = None,
    clock: Callable[[], datetime] | None = None,
    notifier: Any | None = None,
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
        run_id=run_id or new_run_id(),
        clock=clock or _clock(),
    )


def _run_status(store: Store, run_id: str) -> tuple[str, str | None]:
    row = store._conn.execute(  # test-only introspection of the private connection
        "SELECT status, error FROM runs WHERE run_id = ?", [run_id]
    ).fetchone()
    assert row is not None, f"no runs row for {run_id!r}"
    return (row[0], row[1])


def _good_long(
    dax_product_factory: Callable[..., ProductSnapshot], **overrides: Any
) -> ProductSnapshot:
    kwargs: dict[str, Any] = dict(
        isin="DE000LONG001",
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=20000.0,
        quote_timestamp=_EVAL_TIME,
    )
    kwargs.update(overrides)
    return dax_product_factory(**kwargs)


def _good_short(
    dax_product_factory: Callable[..., ProductSnapshot], **overrides: Any
) -> ProductSnapshot:
    kwargs: dict[str, Any] = dict(
        isin="DE000SHRT001",
        issuer="BankB",
        direction=Direction.SHORT,
        financing_level=28000.0,
        quote_timestamp=_EVAL_TIME,
    )
    kwargs.update(overrides)
    return dax_product_factory(**kwargs)


# --------------------------------------------------------------------------
# (a) signal persisted before fetch_products
# --------------------------------------------------------------------------


def test_signal_persisted_before_fetch_products(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    dax_bars: list[UnderlyingBar],
    dax_product_factory: Callable[..., ProductSnapshot],
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
) -> None:
    signal_rows_at_fetch_time: list[int] = []

    def _check_signal_already_persisted() -> None:
        signal_rows_at_fetch_time.append(store.table_counts()["signals"])

    adapter = make_product_adapter(
        "source_a",
        products=[_good_long(dax_product_factory)],
        on_fetch=_check_signal_already_persisted,
    )
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": dax_bars})

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

    assert adapter.fetch_calls == 1
    assert signal_rows_at_fetch_time == [1]  # signal row existed at the moment of fetch_products
    assert result.signal is not None
    assert store.table_counts()["signals"] == 1


# --------------------------------------------------------------------------
# (b) never ACTIONABLE
# --------------------------------------------------------------------------


def test_never_actionable(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    dax_bars: list[UnderlyingBar],
    dax_product_factory: Callable[..., ProductSnapshot],
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
) -> None:
    products = [_good_long(dax_product_factory), _good_short(dax_product_factory)]
    adapter = make_product_adapter("source_a", products=products)
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": dax_bars})

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

    assert len(result.candidates) == 2
    assert result.counts[Category.ACTIONABLE] == 0
    for candidate in result.candidates:
        assert candidate.category != Category.ACTIONABLE
        assert candidate.lcb_ev is None

    # the uptrending synthetic bars produce a LONG direction_hint; the SHORT
    # candidate should be tagged accordingly without changing its category
    assert result.signal is not None
    assert result.signal.direction_hint == Direction.LONG
    short_candidate = next(c for c in result.candidates if c.direction == Direction.SHORT)
    assert "counter_baseline_signal" in short_candidate.reasons
    assert short_candidate.category in (Category.WATCH, Category.REJECT, Category.DATA_QUALITY)


# --------------------------------------------------------------------------
# (c) ratio factor 100 error -> DATA_QUALITY
# --------------------------------------------------------------------------


def test_ratio_factor_100_error_marks_data_quality(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    dax_bars: list[UnderlyingBar],
    dax_product_factory: Callable[..., ProductSnapshot],
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
) -> None:
    # Two well-priced products establish a ~24000 consensus; a third has its
    # ratio corrupted (0.0001 instead of 0.01) with a mid-price consistent
    # with that wrong ratio, landing implied_underlying ~100x consensus --
    # verified in isolation against pricing/integrity.check_product.
    good_long = _good_long(dax_product_factory)
    good_short = _good_short(dax_product_factory)
    ratio_bad = dax_product_factory(
        isin="DE000BADR001",
        issuer="RatioBank",
        direction=Direction.LONG,
        financing_level=20000.0,
        ratio=0.0001,
        bid=237.95,
        ask=238.05,
        quote_timestamp=_EVAL_TIME,
    )

    adapter = make_product_adapter("source_a", products=[good_long, good_short, ratio_bad])
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": dax_bars})

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

    bad_candidate = next(c for c in result.candidates if c.isin == "DE000BADR001")
    assert bad_candidate.category == Category.DATA_QUALITY
    assert any(r.startswith("ratio_factor_error_10e") for r in bad_candidate.reasons)


# --------------------------------------------------------------------------
# (d) bid_only -> REJECT; stale -> REJECT; no ask -> REJECT; missing F -> DATA_QUALITY
# --------------------------------------------------------------------------


def test_bid_only_product_is_rejected(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    dax_bars: list[UnderlyingBar],
    dax_product_factory: Callable[..., ProductSnapshot],
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
) -> None:
    bid_only_product = _good_long(dax_product_factory, isin="DE000BIDO001", bid_only=True)
    adapter = make_product_adapter("source_a", products=[bid_only_product])
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": dax_bars})

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

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.category == Category.REJECT
    assert "bid_only" in candidate.reasons


def test_stale_quote_is_rejected(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    dax_bars: list[UnderlyingBar],
    dax_product_factory: Callable[..., ProductSnapshot],
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
) -> None:
    """A stale quote is a tradability gate, not a data-integrity failure
    (Build Contract Task 2 review finding): pricing/integrity.check_product
    only warns about it, so ranking/gates.py's REJECT branch (reason
    "quote_stale") is what actually excludes it -- not DATA_QUALITY.
    """
    stale_product = _good_long(
        dax_product_factory,
        isin="DE000STAL001",
        quote_timestamp=_EVAL_TIME - timedelta(hours=1),
    )
    adapter = make_product_adapter("source_a", products=[stale_product])
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": dax_bars})

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

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.category == Category.REJECT
    assert "quote_stale" in candidate.reasons


def test_no_ask_quote_is_rejected_and_skips_ask_dependent_pricing(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    dax_bars: list[UnderlyingBar],
    dax_product_factory: Callable[..., ProductSnapshot],
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
) -> None:
    """A product with no ask at all (issuer only quoting bid, e.g. outside
    trading hours) is a REJECT ("no_ask_quote"), not DATA_QUALITY -- and
    every ask-dependent pricing step (leverage, spread, decompose_ask) is
    skipped for it rather than raising or silently imputing an ask.
    """
    base = _good_long(dax_product_factory, isin="DE000NOASK01")
    no_ask_product = base.model_copy(update={"ask": None})
    adapter = make_product_adapter("source_a", products=[no_ask_product])
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": dax_bars})

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

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.category == Category.REJECT
    assert "no_ask_quote" in candidate.reasons
    assert candidate.leverage is None
    assert candidate.costs is None
    assert candidate.integrity_passed is True


def test_missing_financing_level_marks_data_quality(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    dax_bars: list[UnderlyingBar],
    dax_product_factory: Callable[..., ProductSnapshot],
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
) -> None:
    """A missing financing_level remains a genuine data-integrity failure
    (price-critical master data, never imputed) -> DATA_QUALITY, unlike the
    tradability gates (stale quote, no ask) above.
    """
    # underlying_price_ref supplied directly so spot resolution does not
    # itself depend on the cross-issuer consensus (which this lone product,
    # missing financing_level, could not contribute to either).
    base = _good_long(dax_product_factory, isin="DE000NOFIN01", underlying_price_ref=24000.0)
    no_financing_product = base.model_copy(update={"financing_level": None})
    adapter = make_product_adapter("source_a", products=[no_financing_product])
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": dax_bars})

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

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.category == Category.DATA_QUALITY
    assert "missing_financing_level" in candidate.reasons


# --------------------------------------------------------------------------
# (e) all sources fail -> NoProductsError, run status error
# --------------------------------------------------------------------------


def test_all_product_sources_fail_raises_and_records_error_status(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    dax_bars: list[UnderlyingBar],
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
) -> None:
    adapter = make_product_adapter("source_a", exception=AdapterError("upstream down"))
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": dax_bars})
    run_id = new_run_id()

    with pytest.raises(NoProductsError):
        run_scan(
            **_base_kwargs(
                cfg=cfg,
                store=store,
                tmp_path=tmp_path,
                product_adapters=[adapter],
                price_adapter=price_adapter,
                estr_adapter=make_estr_adapter(),
                run_id=run_id,
            ),
            options=ScanOptions(underlying_id="DAX"),
        )

    status, error = _run_status(store, run_id)
    assert status == "error"
    assert error is not None and "upstream down" in error
    # the signal was still frozen and persisted before the failed fetch
    assert store.table_counts()["signals"] == 1
    assert store.table_counts()["candidate_sets"] == 0


# --------------------------------------------------------------------------
# (f) one source fails, another succeeds -> scan ok + warning
# --------------------------------------------------------------------------


def test_one_source_fails_scan_still_completes_with_warning(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    dax_bars: list[UnderlyingBar],
    dax_product_factory: Callable[..., ProductSnapshot],
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
) -> None:
    good_adapter = make_product_adapter("source_ok", products=[_good_long(dax_product_factory)])
    broken_adapter = make_product_adapter("source_broken", exception=AdapterError("timeout"))
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": dax_bars})

    result = run_scan(
        **_base_kwargs(
            cfg=cfg,
            store=store,
            tmp_path=tmp_path,
            product_adapters=[good_adapter, broken_adapter],
            price_adapter=price_adapter,
            estr_adapter=make_estr_adapter(),
        ),
        options=ScanOptions(underlying_id="DAX"),
    )

    assert len(result.candidates) == 1
    assert any(w.startswith("product_source_failed:source_broken:") for w in result.warnings)


# --------------------------------------------------------------------------
# (g) second scan, later day, realized financing spread from history
# --------------------------------------------------------------------------


def test_second_scan_later_day_uses_realized_financing_spread(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    dax_bars: list[UnderlyingBar],
    dax_product_factory: Callable[..., ProductSnapshot],
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
) -> None:
    r = 0.03
    true_spread = 0.02
    f0 = 20000.0
    day1 = _EVAL_TIME
    day2 = _EVAL_TIME + timedelta(days=1)
    f1 = f0 * (1.0 + (r + true_spread) * 1.0 / 360.0)

    price_adapter = make_price_adapter(bars_by_underlying={"DAX": dax_bars})
    estr_adapter = make_estr_adapter(rate=r)

    # -- day 1: only one historical financing-level observation -> default spread
    day1_product = _good_long(dax_product_factory, isin="DE000ROLL001", quote_timestamp=day1)
    adapter_day1 = make_product_adapter("source_a", products=[day1_product])
    result_day1 = run_scan(
        **_base_kwargs(
            cfg=cfg,
            store=store,
            tmp_path=tmp_path,
            product_adapters=[adapter_day1],
            price_adapter=price_adapter,
            estr_adapter=estr_adapter,
            clock=_clock(day1),
        ),
        options=ScanOptions(underlying_id="DAX"),
    )
    candidate_day1 = next(c for c in result_day1.candidates if c.isin == "DE000ROLL001")
    assert "financing_spread_default" in candidate_day1.reasons
    assert candidate_day1.realized_financing_spread == pytest.approx(
        cfg.risk.default_financing_spread
    )

    # -- day 2: financing level rolled forward by the true spread -> realized spread recovered
    day2_product = _good_long(
        dax_product_factory,
        isin="DE000ROLL001",
        financing_level=f1,
        knockout_barrier=f1,
        quote_timestamp=day2,
    )
    adapter_day2 = make_product_adapter("source_a", products=[day2_product])
    result_day2 = run_scan(
        **_base_kwargs(
            cfg=cfg,
            store=store,
            tmp_path=tmp_path,
            product_adapters=[adapter_day2],
            price_adapter=price_adapter,
            estr_adapter=estr_adapter,
            clock=_clock(day2),
        ),
        options=ScanOptions(underlying_id="DAX"),
    )
    candidate_day2 = next(c for c in result_day2.candidates if c.isin == "DE000ROLL001")
    assert "financing_spread_default" not in candidate_day2.reasons
    assert candidate_day2.realized_financing_spread == pytest.approx(true_spread, abs=1e-4)


# --------------------------------------------------------------------------
# (h) bars with available_at > prediction_time are ignored
# --------------------------------------------------------------------------


def test_future_bars_are_excluded_from_signal_computation(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    dax_bars: list[UnderlyingBar],
    dax_product_factory: Callable[..., ProductSnapshot],
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
) -> None:
    # A bar dated (and available) after prediction_time, with a wildly
    # different close -- if it leaked into the TSMOM computation the score
    # would change substantially.
    future_bar = dax_bars[-1].model_copy(
        update={
            "ts": dax_bars[-1].ts + timedelta(days=30),
            "available_at": dax_bars[-1].available_at + timedelta(days=30),
            "observation_time": dax_bars[-1].observation_time + timedelta(days=30),
            "close": dax_bars[-1].close * 5,
            "open": dax_bars[-1].close * 5,
            "high": dax_bars[-1].close * 5.01,
            "low": dax_bars[-1].close * 4.99,
        }
    )
    assert future_bar.available_at > _EVAL_TIME

    price_adapter_clean = make_price_adapter(bars_by_underlying={"DAX": dax_bars})
    price_adapter_with_future = make_price_adapter(
        bars_by_underlying={"DAX": [*dax_bars, future_bar]}
    )

    def run_once(price_adapter: Any, store_instance: Store) -> Any:
        adapter = make_product_adapter("source_a", products=[_good_long(dax_product_factory)])
        return run_scan(
            **_base_kwargs(
                cfg=cfg,
                store=store_instance,
                tmp_path=tmp_path,
                product_adapters=[adapter],
                price_adapter=price_adapter,
                estr_adapter=make_estr_adapter(),
            ),
            options=ScanOptions(underlying_id="DAX"),
        )

    result_clean = run_once(price_adapter_clean, store)

    # fresh store for the second run so persisted state from the first run
    # (e.g. the financing-level history) cannot influence the comparison
    with Store(tmp_path / "state2" / "turboedge.duckdb") as store2:
        store2.init_schema()
        result_with_future = run_once(price_adapter_with_future, store2)

    assert result_clean.signal is not None
    assert result_with_future.signal is not None
    assert result_with_future.signal.score == pytest.approx(result_clean.signal.score)
    assert result_with_future.signal.data_snapshot_hash == result_clean.signal.data_snapshot_hash


# --------------------------------------------------------------------------
# (i) email dedup: a second identical scan does not resend
# --------------------------------------------------------------------------


def test_email_dedup_second_identical_scan_does_not_resend(
    cfg: TurboEdgeConfig,
    store: Store,
    tmp_path: Path,
    dax_bars: list[UnderlyingBar],
    dax_product_factory: Callable[..., ProductSnapshot],
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
    make_notifier: Callable[..., Any],
) -> None:
    price_adapter = make_price_adapter(bars_by_underlying={"DAX": dax_bars})
    notifier = make_notifier()

    def run_once() -> Any:
        adapter = make_product_adapter("source_a", products=[_good_long(dax_product_factory)])
        return run_scan(
            **_base_kwargs(
                cfg=cfg,
                store=store,
                tmp_path=tmp_path,
                product_adapters=[adapter],
                price_adapter=price_adapter,
                estr_adapter=make_estr_adapter(),
                notifier=notifier,
            ),
            options=ScanOptions(underlying_id="DAX", email=True),
        )

    result_1 = run_once()
    result_2 = run_once()

    assert result_1.notification is not None and result_1.notification.sent
    assert notifier.send_calls == 1
    assert result_2.notification is not None
    assert result_2.notification.sent is False
    assert result_2.notification.message == "skipped_duplicate"
