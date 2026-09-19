"""Tests for pricing/instrument_compare.py (Systemkonzept 1.2 §12.1
"Instrumentenneutralitaet").

Three properties are load-bearing for this module and get dedicated tests:

1. A higher (measured) financing spread strictly increases a wrapped
   instrument's total cost, holding everything else fixed.
2. At P(KO) = 0, with trading spread and gap premium held at exactly zero,
   the Turbo-vs-Future cost difference equals EXACTLY the financing
   surcharge (the Turbo's own spread over the risk-free rate) plus the
   issuer margin already embedded in the ask -- no other term is silently
   folded in.
3. A missing required input (e.g. no reference rate) makes the affected
   :class:`~turboedge.pricing.instrument_compare.InstrumentCostResult`
   report ``total_cost_eur=None`` with a populated ``missing_reason``,
   never a silent zero (CLAUDE.md rule 29).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from turboedge.pricing.financing import financing_cost_over_horizon
from turboedge.pricing.gap_premium import GapDistribution
from turboedge.pricing.instrument_compare import (
    InputValue,
    InstrumentsConfig,
    SimulationParams,
    ValueSource,
    compare_instruments,
    eurex_future_cost,
    load_instruments_config,
    wrapped_instrument_cost,
)
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import Direction, ProductSnapshot, ProductType, UnderlyingBar

_AS_OF = datetime(2026, 9, 10, 8, 0, tzinfo=UTC)  # a Thursday -- not the Friday special case


def _make_product(
    *,
    isin: str = "DE000ABCDEF1",
    issuer: str = "TestBank",
    direction: Direction = Direction.LONG,
    product_type: ProductType = ProductType.TURBO_OPEN_END,
    financing_level: float,
    knockout_barrier: float | None = None,
    ratio: float = 0.01,
    bid: float,
    ask: float,
    currency: str = "EUR",
    underlying_currency: str | None = "EUR",
) -> ProductSnapshot:
    barrier = knockout_barrier if knockout_barrier is not None else financing_level
    ts = _AS_OF
    return ProductSnapshot(
        isin=isin,
        wkn=isin[-6:],
        issuer=issuer,
        venue="stuttgart",
        underlying_raw="DAX",
        underlying_id="DAX",
        direction=direction,
        product_type=product_type,
        financing_level=financing_level,
        knockout_barrier=barrier,
        ratio=ratio,
        currency=currency,
        underlying_currency=underlying_currency,
        quanto=False,
        open_end=True,
        maturity=None,
        first_trading_day=None,
        bid=bid,
        ask=ask,
        bid_size=5000.0,
        ask_size=5000.0,
        quote_timestamp=ts,
        quote_presence=True,
        bid_only=False,
        knocked_out=False,
        trading_hours="09:00-22:00",
        product_age_days=100,
        underlying_price_ref=None,
        underlying_price_ref_timestamp=None,
        raw_hash=hashlib.sha256(isin.encode()).hexdigest(),
        observation_time=ts,
        available_at=ts,
        retrieved_at=ts,
        source_timestamp=ts,
        source="test",
        parser_version="1",
        is_stale=False,
        quality_score=1.0,
    )


def _zero_gap_dist() -> GapDistribution:
    """A GapDistribution whose expected gap loss is exactly 0.0 for a LONG
    product whose financing_level == spot: every sampled gap is >= 0, so
    ``spot * exp(g) >= spot == financing_level`` and max(F - projected, 0) is
    0 for every path."""
    return GapDistribution(
        weeknight_gaps=np.array([0.0, 0.001, 0.002]),
        weekend_gaps=np.array([0.0, 0.0015]),
    )


def _nonzero_gap_dist() -> GapDistribution:
    return GapDistribution(
        weeknight_gaps=np.array([-0.01, -0.004, 0.0, 0.003, 0.012]),
        weekend_gaps=np.array([-0.018, -0.006, 0.0, 0.005, 0.02]),
    )


# --------------------------------------------------------------------------
# InputValue
# --------------------------------------------------------------------------


def test_input_value_rejects_value_without_source() -> None:
    with pytest.raises(ValueError, match="both None or both set"):
        InputValue(value=1.0, source=None, note="broken")


def test_input_value_rejects_source_without_value() -> None:
    with pytest.raises(ValueError, match="both None or both set"):
        InputValue(value=None, source=ValueSource.MEASURED, note="broken")


def test_input_value_missing_has_no_value_and_no_source() -> None:
    iv = InputValue.missing("no data")
    assert iv.value is None
    assert iv.source is None
    assert iv.note == "no data"


# --------------------------------------------------------------------------
# load_instruments_config
# --------------------------------------------------------------------------


def test_load_instruments_config_reads_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "instruments.yaml"
    path.write_text(
        "eurex_fee_bps_of_notional: 0.02\n"
        "eurex_margin_pct_of_notional: 0.08\n"
        "mini_future_financing_spread_assumed: 0.035\n"
    )
    cfg = load_instruments_config(path)
    assert cfg.eurex_fee_bps_of_notional == pytest.approx(0.02)
    assert cfg.eurex_margin_pct_of_notional == pytest.approx(0.08)
    assert cfg.mini_future_financing_spread_assumed == pytest.approx(0.035)
    assert "eurex_fee_bps_of_notional" in cfg.eurex_fee_note


def test_load_instruments_config_missing_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "instruments.yaml"
    path.write_text("eurex_fee_bps_of_notional: 0.02\n")
    with pytest.raises(ValueError, match="eurex_margin_pct_of_notional"):
        load_instruments_config(path)


def test_load_instruments_config_real_repo_file_is_valid() -> None:
    """The actual configs/instruments.yaml this module ships with must
    itself be loadable -- a broken shipped config would defeat the whole
    feature silently at runtime."""
    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_instruments_config(repo_root / "configs" / "instruments.yaml")
    assert cfg.eurex_fee_bps_of_notional > 0
    assert 0.0 < cfg.eurex_margin_pct_of_notional < 1.0
    assert cfg.mini_future_financing_spread_assumed > 0


# --------------------------------------------------------------------------
# property 1: higher financing spread -> strictly higher turbo total cost
# --------------------------------------------------------------------------


def test_higher_financing_spread_increases_turbo_cost_monotonically() -> None:
    product = _make_product(financing_level=22000.0, bid=4.80, ask=4.86)
    spot = InputValue.measured(24000.0, "test spot")
    fx = InputValue.measured(1.0, "same currency")
    ref_rate = InputValue.measured(0.03, "test ref rate")
    gap_dist = _nonzero_gap_dist()
    p_ko = InputValue.measured(0.0, "test p_ko")

    totals: list[float] = []
    for spread in (0.01, 0.025, 0.05, 0.08):
        financing_spread = InputValue.measured(spread, f"test spread {spread}")
        result = wrapped_instrument_cost(
            product=product,
            spot=spot,
            fx=fx,
            notional_eur=24000.0,
            horizon_days=10,
            ref_rate=ref_rate,
            financing_spread=financing_spread,
            gap_dist=gap_dist,
            next_night_is_weekend=False,
            p_ko=p_ko,
            as_of=_AS_OF,
        )
        assert result.total_cost_eur is not None, result.missing_reason
        totals.append(result.total_cost_eur)

    assert totals == sorted(totals)
    # Strictly increasing, not just non-decreasing.
    assert all(b > a for a, b in pairwise(totals))


# --------------------------------------------------------------------------
# property 2: P(KO)=0, spread/gap held at exactly 0 -> Turbo-Future diff ==
# financing surcharge + issuer margin, exactly.
# --------------------------------------------------------------------------


def test_turbo_vs_future_diff_equals_financing_surcharge_plus_margin_at_p_ko_zero() -> None:
    spot_value = 24000.0
    ratio = 0.01
    fx_value = 1.0
    horizon_days = 10
    notional_eur = 24000.0
    ref_rate_value = 0.03
    financing_spread_value = 0.02
    ask = 6.00  # == bid: zero trading spread, well above the tiny 1-day financing drag

    # financing_level == spot: makes intrinsic (and thus fair_value) exactly
    # 0, so the "units" scaling used internally cancels exactly against the
    # future's own plain-notional financing formula (see module docstring's
    # derivation in eurex_future_cost and CostLine notes below).
    product = _make_product(financing_level=spot_value, bid=ask, ask=ask)

    spot = InputValue.measured(spot_value, "test spot")
    fx = InputValue.measured(fx_value, "same currency")
    ref_rate = InputValue.measured(ref_rate_value, "test ref rate")
    financing_spread = InputValue.measured(financing_spread_value, "test spread")
    gap_dist = _zero_gap_dist()
    p_ko = InputValue.measured(0.0, "test p_ko")

    turbo = wrapped_instrument_cost(
        product=product,
        spot=spot,
        fx=fx,
        notional_eur=notional_eur,
        horizon_days=horizon_days,
        ref_rate=ref_rate,
        financing_spread=financing_spread,
        gap_dist=gap_dist,
        next_night_is_weekend=False,
        p_ko=p_ko,
        as_of=_AS_OF,
    )
    assert turbo.total_cost_eur is not None, turbo.missing_reason

    by_label = {line.label: line for line in turbo.lines}
    assert by_label["Handelsspread (Geld/Brief, Round-Trip)"].amount_eur == pytest.approx(0.0)
    assert by_label["Gap-Praemie (Overnight/Weekend, Horizont)"].amount_eur == pytest.approx(0.0)
    assert by_label["Erwarteter KO-Verlust"].amount_eur == pytest.approx(0.0)
    turbo_financing = by_label["Finanzierungskosten"].amount_eur
    turbo_margin = by_label["Emittentenmarge (im Ask enthalten)"].amount_eur
    assert turbo_financing is not None
    assert turbo_margin is not None
    assert turbo_margin > 0.0  # ask (6.00) comfortably exceeds the 1-day financing drag

    instruments_cfg = InstrumentsConfig(
        eurex_fee_bps_of_notional=0.0,  # isolate the identity from the exchange fee
        eurex_fee_note="test",
        eurex_margin_pct_of_notional=0.0,
        eurex_margin_note="test",
        mini_future_financing_spread_assumed=0.0,
        mini_future_financing_spread_note="test",
    )
    future = eurex_future_cost(
        notional_eur=notional_eur,
        horizon_days=horizon_days,
        ref_rate=ref_rate,
        instruments_cfg=instruments_cfg,
    )
    assert future.total_cost_eur is not None, future.missing_reason
    future_financing = {line.label: line for line in future.lines}[
        "Finanzierung (impliziter Future-Basis, Marktzins)"
    ].amount_eur
    assert future_financing is not None

    # The financing surcharge is exactly what pricing/financing.py's own
    # formula attributes to the Turbo's spread over the risk-free rate --
    # recomputed independently here (not copied from the module under test).
    calendar_days = horizon_days * 7.0 / 5.0
    units = notional_eur / (spot_value * ratio / fx_value)
    financing_with_spread = (
        financing_cost_over_horizon(
            spot_value,
            financing_spread_value,
            ref_rate_value,
            horizon_days,
            ratio,
            Direction.LONG,
            fx_value,
        )
        * units
    )
    financing_without_spread = (
        financing_cost_over_horizon(
            spot_value, 0.0, ref_rate_value, horizon_days, ratio, Direction.LONG, fx_value
        )
        * units
    )
    financing_surcharge = financing_with_spread - financing_without_spread
    assert financing_surcharge == pytest.approx(
        notional_eur * financing_spread_value * calendar_days / 360.0
    )
    assert turbo_financing == pytest.approx(financing_with_spread)
    assert future_financing == pytest.approx(notional_eur * ref_rate_value * calendar_days / 360.0)

    diff = turbo.total_cost_eur - future.total_cost_eur
    assert diff == pytest.approx(financing_surcharge + turbo_margin, abs=1e-6)


# --------------------------------------------------------------------------
# property 3: a missing required input -> None with a reason, never a
# silent zero.
# --------------------------------------------------------------------------


def test_missing_ref_rate_gives_none_total_not_silent_zero() -> None:
    product = _make_product(financing_level=22000.0, bid=4.80, ask=4.86)
    result = wrapped_instrument_cost(
        product=product,
        spot=InputValue.measured(24000.0, "test spot"),
        fx=InputValue.measured(1.0, "same currency"),
        notional_eur=24000.0,
        horizon_days=10,
        ref_rate=InputValue.missing("ECB nicht erreichbar in diesem Test."),
        financing_spread=InputValue.measured(0.025, "test spread"),
        gap_dist=_nonzero_gap_dist(),
        next_night_is_weekend=False,
        p_ko=InputValue.measured(0.02, "test p_ko"),
        as_of=_AS_OF,
    )
    assert result.total_cost_eur is None
    assert result.total_cost_pct_of_notional is None
    assert result.missing_reason is not None
    assert "ECB nicht erreichbar" in result.missing_reason

    by_label = {line.label: line for line in result.lines}
    financing_line = by_label["Finanzierungskosten"]
    assert financing_line.amount_eur is None  # never a silent 0.0
    assert financing_line.source is None
    margin_line = by_label["Emittentenmarge (im Ask enthalten)"]
    assert margin_line.amount_eur is None  # depends on ref_rate too
    # Lines that do NOT depend on the missing ref_rate are still computed.
    assert by_label["Handelsspread (Geld/Brief, Round-Trip)"].amount_eur is not None
    assert by_label["Gap-Praemie (Overnight/Weekend, Horizont)"].amount_eur is not None


def test_missing_spot_gives_none_total_with_empty_lines() -> None:
    product = _make_product(financing_level=22000.0, bid=4.80, ask=4.86)
    result = wrapped_instrument_cost(
        product=product,
        spot=InputValue.missing("keine historischen Kursbars."),
        fx=InputValue.measured(1.0, "same currency"),
        notional_eur=24000.0,
        horizon_days=10,
        ref_rate=InputValue.measured(0.03, "test ref rate"),
        financing_spread=InputValue.measured(0.025, "test spread"),
        gap_dist=_nonzero_gap_dist(),
        next_night_is_weekend=False,
        p_ko=InputValue.measured(0.02, "test p_ko"),
        as_of=_AS_OF,
    )
    assert result.total_cost_eur is None
    assert result.missing_reason is not None
    assert "keine historischen Kursbars" in result.missing_reason


def test_missing_p_ko_gives_none_total() -> None:
    product = _make_product(financing_level=22000.0, bid=4.80, ask=4.86)
    result = wrapped_instrument_cost(
        product=product,
        spot=InputValue.measured(24000.0, "test spot"),
        fx=InputValue.measured(1.0, "same currency"),
        notional_eur=24000.0,
        horizon_days=10,
        ref_rate=InputValue.measured(0.03, "test ref rate"),
        financing_spread=InputValue.measured(0.025, "test spread"),
        gap_dist=_nonzero_gap_dist(),
        next_night_is_weekend=False,
        p_ko=InputValue.missing("Pfadsimulation fehlgeschlagen: keine Bars."),
        as_of=_AS_OF,
    )
    assert result.total_cost_eur is None
    by_label = {line.label: line for line in result.lines}
    assert by_label["Erwarteter KO-Verlust"].amount_eur is None
    assert by_label["Erwarteter KO-Verlust"].source is None


def test_eurex_future_cost_missing_ref_rate_gives_none() -> None:
    instruments_cfg = InstrumentsConfig(
        eurex_fee_bps_of_notional=0.02,
        eurex_fee_note="test",
        eurex_margin_pct_of_notional=0.08,
        eurex_margin_note="test",
        mini_future_financing_spread_assumed=0.035,
        mini_future_financing_spread_note="test",
    )
    result = eurex_future_cost(
        notional_eur=24000.0,
        horizon_days=15,
        ref_rate=InputValue.missing("kein Netzwerkzugriff im Test."),
        instruments_cfg=instruments_cfg,
    )
    assert result.total_cost_eur is None
    assert result.missing_reason is not None
    by_label = {line.label: line for line in result.lines}
    assert by_label["Finanzierung (impliziter Future-Basis, Marktzins)"].amount_eur is None


def test_eurex_future_cost_every_line_tagged_assumed_or_ref_rate_source() -> None:
    """Documents the honesty mandate at the unit level: the Eurex side never
    claims 'measured' except by inheriting the caller-supplied ref_rate's
    own tag; every constant-derived line is 'assumed'."""
    instruments_cfg = InstrumentsConfig(
        eurex_fee_bps_of_notional=0.02,
        eurex_fee_note="test note",
        eurex_margin_pct_of_notional=0.08,
        eurex_margin_note="test note",
        mini_future_financing_spread_assumed=0.035,
        mini_future_financing_spread_note="test",
    )
    ref_rate = InputValue.measured(0.03, "ECB test observation")
    result = eurex_future_cost(
        notional_eur=24000.0, horizon_days=15, ref_rate=ref_rate, instruments_cfg=instruments_cfg
    )
    assert result.total_cost_eur is not None
    by_label = {line.label: line for line in result.lines}
    financing_source = by_label["Finanzierung (impliziter Future-Basis, Marktzins)"].source
    assert financing_source == ValueSource.MEASURED
    for label in ("Boersengebuehr (Eurex, angenommen)", "Emittentenmarge", "Erwarteter KO-Verlust"):
        assert by_label[label].source == ValueSource.ASSUMED
    margin_line = by_label["Margin-Anforderung (informativ, NICHT in Summe)"]
    assert margin_line.source == ValueSource.ASSUMED
    assert margin_line.included_in_total is False


# --------------------------------------------------------------------------
# orchestration: compare_instruments against a real (synthetic) Store
# --------------------------------------------------------------------------


def _underlying_bars(n: int = 60) -> list[UnderlyingBar]:
    bars = []
    start = _AS_OF - timedelta(days=n + 5)
    price = 24000.0
    rng = np.random.default_rng(42)
    for i in range(n):
        ts = start + timedelta(days=i)
        if ts.weekday() >= 5:
            continue
        open_ = price
        close = price * float(np.exp(rng.normal(0.0, 0.01)))
        bars.append(
            UnderlyingBar(
                underlying_id="DAX",
                ts=ts,
                interval="1d",
                open=open_,
                high=max(open_, close) * 1.002,
                low=min(open_, close) * 0.998,
                close=close,
                volume=1000.0,
                observation_time=ts,
                available_at=ts,
                retrieved_at=ts,
                source="test",
                parser_version="1",
                quality_score=1.0,
            )
        )
        price = close
    return bars


def _default_sim() -> SimulationParams:
    return SimulationParams(
        n_paths=200, method="vol_scaled_bootstrap", block_size=5, lookback_days=750, seed=1234
    )


def _default_instruments_cfg() -> InstrumentsConfig:
    return InstrumentsConfig(
        eurex_fee_bps_of_notional=0.02,
        eurex_fee_note="test",
        eurex_margin_pct_of_notional=0.08,
        eurex_margin_note="test",
        mini_future_financing_spread_assumed=0.035,
        mini_future_financing_spread_note="test",
    )


def test_compare_instruments_empty_store_still_prices_future(tmp_path: Path) -> None:
    """No product/underlying data at all: Turbo and Mini-Future results
    explain why they are missing; the Eurex-Future side (independent of
    Store data) still prices, since it only needs the caller-supplied
    reference rate and configs/instruments.yaml's constants."""
    with Store(tmp_path / "turboedge.duckdb") as store:
        store.init_schema()
        result = compare_instruments(
            store,
            underlying_id="DAX",
            notional_eur=24000.0,
            horizon_days=15,
            direction=Direction.LONG,
            ref_rate=InputValue.assumed(0.03, "test fallback"),
            financing_spread_fallback=InputValue.assumed(0.025, "test fallback"),
            financing_adjustment_jump_threshold_pct=0.08,
            instruments_cfg=_default_instruments_cfg(),
            sim=_default_sim(),
            as_of=_AS_OF,
        )

    by_class = {r.instrument_class: r for r in result.results if r.issuer is None}
    turbo_placeholder = next(
        r for r in result.results if r.instrument_class == "turbo_open_end" and r.issuer is None
    )
    mini_placeholder = next(
        r for r in result.results if r.instrument_class == "mini_future" and r.issuer is None
    )
    assert turbo_placeholder.total_cost_eur is None
    assert turbo_placeholder.missing_reason is not None
    assert mini_placeholder.total_cost_eur is None
    assert mini_placeholder.missing_reason is not None

    future = by_class["eurex_future"]
    assert future.total_cost_eur is not None
    assert result.cheapest_wrapped is None
    assert result.factor_vs_future is None
    assert "nicht berechenbar" in result.factor_note


def test_compare_instruments_prices_synthetic_representative_products(tmp_path: Path) -> None:
    with Store(tmp_path / "turboedge.duckdb") as store:
        store.init_schema()
        store.append_underlying_bars(_underlying_bars())
        turbo = _make_product(
            isin="DE000TURBO01",
            issuer="BankA",
            financing_level=20000.0,
            bid=39.90,
            ask=40.00,
        )
        mini = _make_product(
            isin="DE000MINI001",
            issuer="BankA",
            product_type=ProductType.MINI_FUTURE,
            financing_level=20000.0,
            knockout_barrier=20500.0,
            bid=39.90,
            ask=40.00,
        )
        store.append_product_snapshots([turbo, mini])

        result = compare_instruments(
            store,
            underlying_id="DAX",
            notional_eur=24000.0,
            horizon_days=10,
            direction=Direction.LONG,
            ref_rate=InputValue.assumed(0.03, "test fallback"),
            financing_spread_fallback=InputValue.assumed(0.025, "test fallback"),
            financing_adjustment_jump_threshold_pct=0.08,
            instruments_cfg=_default_instruments_cfg(),
            sim=_default_sim(),
            as_of=_AS_OF,
        )

    turbo_result = next(
        r for r in result.results if r.instrument_class == "turbo_open_end" and r.issuer == "BankA"
    )
    mini_result = next(
        r for r in result.results if r.instrument_class == "mini_future" and r.issuer == "BankA"
    )
    future_result = next(r for r in result.results if r.instrument_class == "eurex_future")

    assert turbo_result.isin == "DE000TURBO01"
    assert turbo_result.total_cost_eur is not None, turbo_result.missing_reason
    assert mini_result.isin == "DE000MINI001"
    assert mini_result.total_cost_eur is not None, mini_result.missing_reason
    assert future_result.total_cost_eur is not None

    # Only one financing-level observation was ever inserted for either
    # ISIN -- financing spread must fall back to the caller-supplied
    # fallback, and that must be reflected honestly as 'assumed'.
    turbo_financing = {line.label: line for line in turbo_result.lines}["Finanzierungskosten"]
    assert turbo_financing.source == ValueSource.ASSUMED

    assert result.cheapest_wrapped is not None
    assert result.factor_vs_future is not None
    assert result.factor_vs_future > 0.0
