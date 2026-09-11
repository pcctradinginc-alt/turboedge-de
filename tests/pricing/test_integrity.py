from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from turboedge.pricing.integrity import IntegrityReport, check_product
from turboedge.storage.schemas import Direction, ProductSnapshot, ProductType

_NOW = datetime(2026, 9, 10, 16, 0, tzinfo=UTC)


def test_check_product_passes_for_a_clean_snapshot(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    product = make_product_snapshot(
        financing_level=18000.0,
        knockout_barrier=18000.0,
        product_type=ProductType.TURBO_OPEN_END,
        direction=Direction.LONG,
        bid=5.20,
        ask=5.30,
        quote_timestamp=_NOW - timedelta(seconds=10),
        underlying_price_ref=18500.0,
    )
    report = check_product(
        product,
        consensus=18500.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=frozenset({"TestBank"}),
        margin_warn_pct=0.5,
    )
    assert isinstance(report, IntegrityReport)
    assert report.passed is True
    assert report.failures == []


def test_check_product_flags_bid_above_ask(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    product = make_product_snapshot(bid=5.0, ask=4.9)
    report = check_product(
        product,
        consensus=18500.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=None,
        margin_warn_pct=0.5,
    )
    assert report.passed is False
    assert "bid_greater_than_ask" in report.failures


def test_check_product_flags_wrong_barrier_side(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    # Long, barrier == financing_level (turbo_open_end), but spot (consensus)
    # is below the barrier -> should have knocked out already.
    product = make_product_snapshot(
        direction=Direction.LONG,
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=18000.0,
        knockout_barrier=18000.0,
        knocked_out=False,
    )
    report = check_product(
        product,
        consensus=17000.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=None,
        margin_warn_pct=0.5,
    )
    assert report.passed is False
    assert "knocked_out_or_wrong_side" in report.failures


def test_check_product_flags_ratio_factor_error(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    financing_level, ratio = 22000.0, 0.01
    # mid implies an underlying of 24000, but the "true" consensus is 100x
    # that -- a classic Bezugsverhaltnis/ratio parsing error.
    mid = (24000.0 - financing_level) * ratio
    product = make_product_snapshot(
        direction=Direction.LONG,
        financing_level=financing_level,
        ratio=ratio,
        bid=mid - 0.01,
        ask=mid + 0.01,
    )
    report = check_product(
        product,
        consensus=2_400_000.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=None,
        margin_warn_pct=0.5,
    )
    assert report.passed is False
    assert "ratio_factor_error_10e-2" in report.failures
    assert "implied_spot_deviation" not in report.failures


def test_check_product_flags_implied_spot_deviation_without_factor_error(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    financing_level, ratio = 22000.0, 0.01
    mid = (24000.0 - financing_level) * ratio
    product = make_product_snapshot(
        direction=Direction.LONG,
        financing_level=financing_level,
        ratio=ratio,
        bid=mid - 0.01,
        ask=mid + 0.01,
    )
    # 10% off consensus -- large enough to fail, but nowhere near a 10^k factor.
    report = check_product(
        product,
        consensus=24000.0 * 1.10,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=None,
        margin_warn_pct=0.5,
    )
    assert report.passed is False
    assert "implied_spot_deviation" in report.failures


def test_check_product_warns_stale_quote_but_still_passes(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    """A stale quote is a tradability gate (ranking/gates.py REJECT
    "quote_stale"), not a data-integrity failure -- Build Contract Task 2
    review finding. check_product warns, it does not fail.
    """
    product = make_product_snapshot(quote_timestamp=_NOW - timedelta(seconds=600))
    report = check_product(
        product,
        consensus=18500.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=None,
        margin_warn_pct=0.5,
    )
    assert report.passed is True
    assert report.failures == []
    assert "quote_stale" in report.warnings


def test_check_product_warns_missing_quote_timestamp_but_still_passes(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    """A missing quote timestamp is likewise a tradability gate
    (ranking/gates.py REJECT "quote_timestamp_missing"), not a data-integrity
    failure.
    """
    product = make_product_snapshot(quote_timestamp=None)
    report = check_product(
        product,
        consensus=18500.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=None,
        margin_warn_pct=0.5,
    )
    assert report.passed is True
    assert report.failures == []
    assert "quote_timestamp_missing" in report.warnings


def test_check_product_does_not_fail_on_missing_ask(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    """A missing ask (issuer quoting only bid, e.g. outside trading hours) is
    a tradability gate (ranking/gates.py REJECT "no_ask_quote"), not a data
    integrity failure, provided bid/financing_level/ratio are otherwise
    plausible -- Build Contract Task 2 review finding.
    """
    product = make_product_snapshot(ask=None)
    report = check_product(
        product,
        consensus=18500.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=None,
        margin_warn_pct=0.5,
    )
    assert report.passed is True
    assert report.failures == []


def test_check_product_does_not_fail_on_no_live_quote(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    """Bid AND ask both missing, with the source explicitly reporting no live
    quote (``quote_presence=False``, e.g. Citi's ``referencePriceMethod ==
    "Closing Price"`` rows -- adapters/issuer_feeds.py), is "no tradable
    quote", not a data-integrity failure, provided master data (financing
    level, barrier, underlying mapping, ...) is otherwise plausible --
    ranking/gates.py rejects it explicitly ("no_live_quote").
    """
    product = make_product_snapshot(bid=None, ask=None, quote_presence=False)
    report = check_product(
        product,
        consensus=18500.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=None,
        margin_warn_pct=0.5,
    )
    assert report.passed is True
    assert "missing_bid" not in report.failures
    assert report.failures == []


def test_check_product_flags_missing_bid_when_quote_presence_true(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    """A missing bid with ``quote_presence=True`` (unexpected: the source
    claims a live quote exists but the bid field is empty) stays a genuine
    data-integrity failure -- the no-live-quote exemption only applies when
    the source explicitly says there is no live quote at all.
    """
    product = make_product_snapshot(bid=None, ask=None, quote_presence=True)
    report = check_product(
        product,
        consensus=18500.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=None,
        margin_warn_pct=0.5,
    )
    assert report.passed is False
    assert "missing_bid" in report.failures


def test_check_product_flags_missing_bid_when_quote_presence_none(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    """A missing bid with ``quote_presence=None`` (source did not say either
    way) also stays a genuine data-integrity failure -- only an explicit
    ``quote_presence=False`` (plus a missing ask) counts as "source says no
    live quote".
    """
    product = make_product_snapshot(bid=None, ask=None, quote_presence=None)
    report = check_product(
        product,
        consensus=18500.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=None,
        margin_warn_pct=0.5,
    )
    assert report.passed is False
    assert "missing_bid" in report.failures


def test_check_product_flags_missing_bid_when_ask_present(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    """A missing bid with an ask still present is never "no live quote" (an
    ask exists) -- stays a genuine data-integrity failure regardless of
    ``quote_presence``.
    """
    product = make_product_snapshot(bid=None, quote_presence=False)
    report = check_product(
        product,
        consensus=18500.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=None,
        margin_warn_pct=0.5,
    )
    assert report.passed is False
    assert "missing_bid" in report.failures


def test_check_product_flags_missing_financing_level(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    product = make_product_snapshot(financing_level=None)
    report = check_product(
        product,
        consensus=18500.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=None,
        margin_warn_pct=0.5,
    )
    assert report.passed is False
    assert "missing_financing_level" in report.failures


def test_check_product_flags_unknown_issuer(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    product = make_product_snapshot(issuer="ShadyBank")
    report = check_product(
        product,
        consensus=18500.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=frozenset({"TestBank"}),
        margin_warn_pct=0.5,
    )
    assert report.passed is False
    assert "unknown_issuer" in report.failures


def test_check_product_flags_missing_underlying_mapping(
    make_product_snapshot: Callable[..., ProductSnapshot],
) -> None:
    product = make_product_snapshot(underlying_id=None)
    report = check_product(
        product,
        consensus=18500.0,
        now=_NOW,
        max_quote_age_s=120.0,
        known_issuers=None,
        margin_warn_pct=0.5,
    )
    assert report.passed is False
    assert "missing_underlying_mapping" in report.failures
