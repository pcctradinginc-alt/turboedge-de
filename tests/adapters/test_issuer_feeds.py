"""Contract tests for the BNP Paribas / Citi issuer-feed adapters.

No real network access (respx mocks every HTTP call). Fixtures under
``tests/fixtures/issuer_feeds/{bnp_paribas,citi}`` are real captures from the
Round 2 data-source research session -- see ``docs/data_sources.md`` and
``src/turboedge/adapters/issuer_feeds.py``'s module docstring for the
research context every test below is grounded in.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import httpx
import pytest
import respx
import structlog.testing

from turboedge.adapters.base import AdapterHttpError, HttpClient
from turboedge.adapters.issuer_feeds import (
    _BNP_DEFAULT_BASE_URL,
    _BNP_DERIVATIVE_TYPE_CATALOG,
    _BNP_DERIVATIVE_TYPE_IDS,
    _CITI_DEFAULT_BASE_URL,
    _CITI_UNDERLYING_ISINS,
    BnpParibasTurboAdapter,
    CitiFirstTurboAdapter,
    IssuerRowError,
    _bnp_factory,
    _citi_factory,
    _parse_berlin_naive_to_utc,
    _register,
    _resolve_underlying_currency,
)
from turboedge.adapters.registry import PRODUCT_ADAPTER_FACTORIES, ProductSourceAdapter
from turboedge.config import SourceConfig
from turboedge.storage.schemas import Direction, HealthStatus, ProductSnapshot, ProductType

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "issuer_feeds"

BNP_INDEXES_URL = f"{_BNP_DEFAULT_BASE_URL}/underlying/indexes"
BNP_LEVERAGE_URL = f"{_BNP_DEFAULT_BASE_URL}/productlist/leverage"
CITI_SEARCH_URL = f"{_CITI_DEFAULT_BASE_URL}/ProductSearch/de-DE/Search"

DAX_ISIN = "DE0008469008"


def _load_fixture(*parts: str) -> Any:
    path = FIXTURES_DIR.joinpath(*parts)
    return json.loads(path.read_text())


def _fixed_clock(when: datetime) -> Callable[[], datetime]:
    return lambda: when


# -- test-only synthetic-record builders (for pagination/error-path tests) ----


def _bnp_product(
    isin: str,
    *,
    direction: str = "long",
    bid: float | None = 10.0,
    ask: float | None = 10.5,
    include_ask_key: bool = True,
    strike: float = 20000.0,
    ko: float = 20000.0,
    ratio: float = 0.01,
    maturity_ts: int = -1,
    is_knocked_out: bool = False,
    is_bid_only: bool = False,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "isin": isin,
        "wkn": isin[-6:],
        "currency": {"isoCode": "EUR"},
        "bid": bid,
        "bidDate": "2026-09-10T21:59:54.042",
        "bidSize": 1000,
        "first": {
            "underlyingOfficialName": "DAX®",
            "underlyingISIN": DAX_ISIN,
            "strikeAbsolute": strike,
            "knockOutAbsolute": ko,
            "ratio": ratio,
            "price": 25333.9,
            "barrierMonitoringTime": "09:00:00-17:30:00",
            "barrierMonitoringTimeZone": "GMT+02:00",
        },
        "config": {
            "derivativeDirectionName": direction,
            "isBidOnly": is_bid_only,
            "isKnockedOut": is_knocked_out,
        },
        "keyFigures": {"maturityDateTimestamp": maturity_ts},
    }
    if include_ask_key:
        record["ask"] = ask
        record["askDate"] = "2026-09-10T21:59:54.042"
        record["askSize"] = 1000
    return record


def _bnp_page_response(
    items: list[dict[str, Any]], *, total: int, response_date: str = "2026-09-10T23:06:28.9558001Z"
) -> dict[str, Any]:
    return {
        "productListResponse": {
            "limit": len(items),
            "total": total,
            "responseDate": response_date,
            "result": items,
        }
    }


def _citi_product(
    isin: str,
    *,
    underlying_isin: str = DAX_ISIN,
    underlying_name: str = "DAX",
    sub_type: str = "Long",
    bid_amount: float | None = 200.0,
    ask_amount: float | None = 0.0,
    strike: float = 3300.0,
    ko: float = 3390.0,
    ratio: float = 0.01,
    maturity_date: str | None = None,
    reference_price_method: str = "Closing Price",
    product_type: str = "MiniFuture",
) -> dict[str, Any]:
    return {
        "isin": isin,
        "wkn": isin[-6:],
        "currencyCode": "EUR",
        "productType": product_type,
        "subTypeTranslation": sub_type,
        "referencePriceMethod": reference_price_method,
        "maturityDate": maturity_date,
        "isQuanto": False,
        "underlyings": [
            {
                "isin": underlying_isin,
                "name": underlying_name,
                "currencyCode": "Pkt",
                "tradingTimes": {"productTime": {"from": "09:00", "to": "22:00"}},
            }
        ],
        "strike": {"amount": strike, "currencyCode": "Pkt"},
        "koBarrier": {"amount": ko, "currencyCode": "Pkt"},
        "ratio": ratio,
        "barrierBreached": False,
        "price": {
            "timeStamp": "2026-09-10T21:58:44",
            "bid": {"amount": bid_amount, "currencyCode": "EUR"},
            "ask": {"amount": ask_amount, "currencyCode": "EUR"},
            "bidSize": 0,
            "askSize": 0,
        },
    }


def _citi_search_response(
    items: list[dict[str, Any]], *, total_elements_count: int, items_count: int | None = None
) -> dict[str, Any]:
    return {
        "availableValues": {},
        "totalElementsCount": total_elements_count,
        "items": items,
        "itemsCount": items_count if items_count is not None else len(items),
    }


# ===========================================================================
# BNP Paribas
# ===========================================================================


def _bnp_adapter(**kwargs: Any) -> BnpParibasTurboAdapter:
    http = HttpClient(user_agent="test-agent/1.0", max_retries=kwargs.pop("max_retries", 3))
    return BnpParibasTurboAdapter(http, **kwargs)


def test_bnp_metadata() -> None:
    adapter = _bnp_adapter()
    meta = adapter.metadata()
    assert meta.name == "bnp_paribas"
    assert meta.kind == "product"
    assert adapter.name == "bnp_paribas"


@respx.mock
def test_bnp_fetch_products_parses_real_fixture() -> None:
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_load_fixture(
                "bnp_paribas", "productlist_leverage_dax_market_hours_ask_present.json"
            ),
        )
    )
    adapter = _bnp_adapter(clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert len(snapshots) >= 20
    assert all(isinstance(s, ProductSnapshot) for s in snapshots)
    assert all(len(s.isin) == 12 and s.isin.isalnum() for s in snapshots)
    assert all(s.ratio > 0 for s in snapshots)
    assert all(s.financing_level is not None and s.financing_level > 0 for s in snapshots)
    directions = {s.direction for s in snapshots}
    assert directions == {Direction.LONG, Direction.SHORT}
    assert not adapter.last_errors
    # Befund 1 (2026-09-13 measurement session): underlying_currency must be
    # populated from the canonical underlying's own static currency, not
    # left at None -- a None here made pricing/integrity.check_product's
    # same-currency guard silently assume fx=1 for a genuinely cross-
    # currency product (see adapters/issuer_feeds._resolve_underlying_currency).
    assert all(s.underlying_currency == "EUR" for s in snapshots)


@respx.mock
def test_bnp_ask_absent_treated_as_none_and_quote_presence_false() -> None:
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "productlist_leverage_dax_probe_output.json")
        )
    )
    adapter = _bnp_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 4, 0, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert len(snapshots) >= 15
    assert all(s.ask is None for s in snapshots)
    assert all(s.quote_presence is False for s in snapshots)
    # quality_score reflects "no usable ask" (0.3), never the "fresh" (1.0) case.
    assert all(s.quality_score == pytest.approx(0.3) for s in snapshots)


@respx.mock
def test_bnp_dated_maturity_timestamp_still_never_guess_parsed() -> None:
    """Build Contract W1 measurement (2026-09-11/12): a live, full-coverage
    pull of BNP's entire DAX book (4304/4304 rows, every derivativeTypeId
    this adapter requests) found `maturityDateTimestamp == -1` on every
    single row -- BNP is not currently issuing any dated leverage product on
    DAX (or, per a separate probe, on any of its 14 index underlyings). No
    real dated-maturity fixture could therefore be captured from a live pull
    (there is currently nothing live to capture). This test instead uses the
    existing synthetic-record builder (`_bnp_product`, already used
    throughout this file for pagination/error-path coverage that a real
    fixture cannot exercise) to pin down the still-necessary defensive
    behavior for the day a dated row does appear: the unit/epoch format
    remains genuinely unconfirmed, so it must stay un-guess-parsed
    (CLAUDE.md rule 29) -- `maturity` stays `None`, `open_end` is `False`,
    and (with no name-based signal either) `product_type` falls back to
    `UNKNOWN` rather than being silently treated as `TURBO_OPEN_END`.
    """
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_bnp_page_response(
                [_bnp_product("DE00CLASS001", maturity_ts=1234567890)], total=1
            ),
        )
    )
    adapter = _bnp_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 22, 5, tzinfo=UTC)))

    with structlog.testing.capture_logs() as logs:
        snapshots = adapter.fetch_products(["DAX"])

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.maturity is None
    assert snapshot.open_end is False
    assert snapshot.product_type == ProductType.UNKNOWN

    warnings = [e for e in logs if e.get("event") == "bnp_maturity_timestamp_format_unconfirmed"]
    assert len(warnings) == 1
    assert warnings[0]["isin"] == "DE00CLASS001"
    assert warnings[0]["maturity_timestamp"] == 1234567890


@respx.mock
def test_bnp_mini_future_classified_by_live_naming_convention_even_with_no_buffer() -> None:
    """BNP's real live naming convention (confirmed 2026-09-11 research
    session) is "Mini Long auf den DAX(R)" / "Mini Short auf den DAX(R)" --
    it never contains the word "future", so the pre-existing Mini Future
    keyword list never actually matched it and the adapter never even passed
    a name to `classify_product_type` at all. Both gaps are fixed together
    here: the adapter now passes `productName`, and
    `universe/classify.py`'s keyword list now recognizes "mini long"/"mini
    short". A record with `strike == ko` (would otherwise structurally look
    like a bufferless TURBO_OPEN_END) must still classify as MINI_FUTURE
    once its real name says so -- the documented "name wins even when the
    buffer is temporarily zero" behavior actually firing end-to-end.
    """
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    record = _bnp_product("DE00MINIL001", strike=20000.0, ko=20000.0)
    record["productName"] = "Mini Long auf den DAX®"
    respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(200, json=_bnp_page_response([record], total=1))
    )
    adapter = _bnp_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 22, 5, tzinfo=UTC)))

    snapshots = adapter.fetch_products(["DAX"])

    assert len(snapshots) == 1
    assert snapshots[0].product_type == ProductType.MINI_FUTURE


@respx.mock
def test_bnp_ask_anomalies_logged_as_one_aggregated_summary_not_per_row() -> None:
    """20 rows, all missing `ask` -- previously 20 INFO lines (one per row,
    `bnp_ask_key_absent`), now exactly one `bnp_quotes_summary` INFO line
    with the aggregated counts, capped at 5 example ISINs; per-row detail
    still logged, but only at DEBUG."""
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "productlist_leverage_dax_probe_output.json")
        )
    )
    adapter = _bnp_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 4, 0, tzinfo=UTC)))

    with structlog.testing.capture_logs() as logs:
        snapshots = adapter.fetch_products(["DAX"])

    summary_events = [e for e in logs if e.get("event") == "bnp_quotes_summary"]
    assert len(summary_events) == 1
    summary = summary_events[0]
    assert summary["count_total"] == len(snapshots)
    assert summary["count_ask_missing"] == len(snapshots)
    assert summary["count_bid_missing"] == 0
    assert len(summary["example_isins"]) == 5

    # Per-row detail must not appear as INFO-level individual lines anymore.
    per_row_info_events = [
        e for e in logs if e.get("event") == "bnp_ask_key_absent" and e.get("log_level") == "info"
    ]
    assert per_row_info_events == []


@respx.mock
def test_bnp_quotes_summary_not_logged_when_no_anomalies() -> None:
    """A healthy fetch (every row has a usable bid+ask) stays silent."""
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_load_fixture(
                "bnp_paribas", "productlist_leverage_dax_market_hours_ask_present.json"
            ),
        )
    )
    adapter = _bnp_adapter(clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC)))

    with structlog.testing.capture_logs() as logs:
        adapter.fetch_products(["DAX"])

    assert [e for e in logs if e.get("event") == "bnp_quotes_summary"] == []


def test_bnp_timezone_conversion_summer_and_winter() -> None:
    # CEST (summer, UTC+2): 2026-09-10 is well before the late-October DST
    # switch-back.
    summer = _parse_berlin_naive_to_utc("2026-09-10T21:59:54.042")
    assert summer is not None
    assert summer == datetime(2026, 9, 10, 19, 59, 54, 42000, tzinfo=UTC)

    # CET (winter, UTC+1): 2026-01-15 is well within standard time.
    winter = _parse_berlin_naive_to_utc("2026-01-15T21:59:54.042")
    assert winter is not None
    assert winter == datetime(2026, 1, 15, 20, 59, 54, 42000, tzinfo=UTC)

    assert _parse_berlin_naive_to_utc(None) is None
    assert _parse_berlin_naive_to_utc("") is None
    assert _parse_berlin_naive_to_utc("not-a-timestamp") is None


def test_resolve_underlying_currency() -> None:
    """Befund 1: the canonical underlying's own currency, never guessed --
    an unknown underlying_id resolves to None rather than defaulting to EUR."""
    assert _resolve_underlying_currency("DAX") == "EUR"
    assert _resolve_underlying_currency("NDX") == "USD"
    assert _resolve_underlying_currency("SPX") == "USD"
    assert _resolve_underlying_currency("NOT_A_REAL_UNDERLYING") is None


@respx.mock
def test_bnp_implied_underlying_within_band_of_median() -> None:
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_load_fixture(
                "bnp_paribas", "productlist_leverage_dax_market_hours_ask_present.json"
            ),
        )
    )
    adapter = _bnp_adapter(clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    with_quotes = [s for s in snapshots if s.quote_presence]
    assert len(with_quotes) >= 15

    implied: list[float] = []
    for s in with_quotes:
        assert s.bid is not None and s.ask is not None and s.financing_level is not None
        mid = (s.bid + s.ask) / 2
        value = (
            s.financing_level + mid / s.ratio
            if s.direction == Direction.LONG
            else s.financing_level - mid / s.ratio
        )
        implied.append(value)

    med = median(implied)
    for value in implied:
        assert abs(value - med) / med <= 0.03


@respx.mock
def test_bnp_pagination_across_multiple_pages_and_max_pages_limit() -> None:
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    page1 = [_bnp_product("DE000AAAAAA1"), _bnp_product("DE000AAAAAA2", direction="short")]
    page2 = [_bnp_product("DE000AAAAAA3"), _bnp_product("DE000AAAAAA4", direction="short")]
    respx.post(BNP_LEVERAGE_URL).mock(
        side_effect=[
            httpx.Response(200, json=_bnp_page_response(page1, total=10)),
            httpx.Response(200, json=_bnp_page_response(page2, total=10)),
        ]
    )
    adapter = _bnp_adapter(
        max_pages=2, page_size=2, clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC))
    )
    snapshots = adapter.fetch_products(["DAX"])

    assert {s.isin for s in snapshots} == {
        "DE000AAAAAA1",
        "DE000AAAAAA2",
        "DE000AAAAAA3",
        "DE000AAAAAA4",
    }
    # total=10, only 2 pages * page_size=2 = 4 fetched -> partial_universe.
    assert adapter._partial_universe["DAX"] == (4, 10)


@respx.mock
def test_bnp_pagination_stops_early_when_a_short_page_is_returned() -> None:
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    only_page = [_bnp_product("DE000BBBBBB1")]
    route = respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(200, json=_bnp_page_response(only_page, total=1))
    )
    adapter = _bnp_adapter(
        max_pages=20, page_size=50, clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC))
    )
    snapshots = adapter.fetch_products(["DAX"])

    assert route.call_count == 1  # a short page (< page_size) stops pagination immediately
    assert len(snapshots) == 1
    assert "DAX" not in adapter._partial_universe


@respx.mock
def test_bnp_offset_cap_on_later_page_returns_partial_not_raise() -> None:
    """Befund 1 (e) live finding: BNP's backend hard-errors any request past
    an undocumented offset ceiling (confirmed live at offset=10000). A page
    AFTER the first hitting this (or any other transient HTTP failure) must
    not discard already-successfully-fetched pages -- pagination stops and
    the partial result is returned, surfaced via the existing
    bnp_partial_universe WARN, exactly as a max_pages cutoff already is."""
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    page1 = [_bnp_product("DE000OFFCAP1"), _bnp_product("DE000OFFCAP2", direction="short")]
    respx.post(BNP_LEVERAGE_URL).mock(
        side_effect=[
            httpx.Response(200, json=_bnp_page_response(page1, total=100)),
            httpx.Response(500),  # simulates the offset-ceiling 500 on page 2
        ]
    )
    adapter = _bnp_adapter(
        max_pages=5,
        page_size=2,
        max_retries=1,
        clock=_fixed_clock(datetime(2026, 9, 13, 12, 0, tzinfo=UTC)),
    )

    with structlog.testing.capture_logs() as logs:
        snapshots = adapter.fetch_products(["DAX"])

    assert {s.isin for s in snapshots} == {"DE000OFFCAP1", "DE000OFFCAP2"}
    assert adapter._partial_universe["DAX"] == (2, 100)
    offset_error_events = [e for e in logs if e.get("event") == "bnp_pagination_offset_error"]
    assert len(offset_error_events) == 1
    assert offset_error_events[0]["offset"] == 2  # page_index=1 * page_size=2


@respx.mock
def test_bnp_first_page_http_error_still_raises() -> None:
    """Unlike a later page (see the offset-cap test above), a failure on the
    very first page is a genuine reachability failure with nothing to
    salvage -- must still raise, exactly as before this fix."""
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    respx.post(BNP_LEVERAGE_URL).mock(return_value=httpx.Response(500))
    adapter = _bnp_adapter(max_retries=1)
    with pytest.raises(AdapterHttpError):
        adapter.fetch_products(["DAX"])


@respx.mock
def test_bnp_unresolved_underlying_is_skipped_cleanly() -> None:
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    leverage_route = respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(200, json=_bnp_page_response([], total=0))
    )
    adapter = _bnp_adapter()
    # XAU (gold) is not among BNP's /underlying/indexes entries (index-only
    # endpoint) -- must be skipped, never guessed.
    snapshots = adapter.fetch_products(["XAU"])

    assert snapshots == []
    assert leverage_route.call_count == 0


@respx.mock
def test_bnp_schema_drift_missing_field_goes_to_last_errors_not_raised() -> None:
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    good = _bnp_product("DE000CCCCCC1")
    bad = _bnp_product("DE000CCCCCC2")
    del bad["first"]["ratio"]  # required field missing
    respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(200, json=_bnp_page_response([good, bad], total=2))
    )
    adapter = _bnp_adapter(clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert len(snapshots) == 1
    assert snapshots[0].isin == "DE000CCCCCC1"
    assert len(adapter.last_errors) == 1
    assert isinstance(adapter.last_errors[0], IssuerRowError)
    assert adapter.last_errors[0].isin == "DE000CCCCCC2"


@respx.mock
def test_bnp_http_500_retries_then_raises_adapter_http_error() -> None:
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    respx.post(BNP_LEVERAGE_URL).mock(return_value=httpx.Response(500))
    adapter = _bnp_adapter(max_retries=2)
    with pytest.raises(AdapterHttpError):
        adapter.fetch_products(["DAX"])


@respx.mock
def test_bnp_healthcheck_pass() -> None:
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    adapter = _bnp_adapter()
    result = adapter.healthcheck()
    assert result.status == HealthStatus.PASS
    assert result.ok is True


@respx.mock
def test_bnp_healthcheck_fail_on_http_error() -> None:
    respx.get(BNP_INDEXES_URL).mock(return_value=httpx.Response(500))
    adapter = _bnp_adapter(max_retries=1)
    result = adapter.healthcheck()
    assert result.status == HealthStatus.FAIL
    assert result.ok is False


@respx.mock
def test_bnp_healthcheck_fail_on_schema_drift() -> None:
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json={"result": [{"instrumentId": 1}]}
        )  # missing name/isin
    )
    adapter = _bnp_adapter()
    result = adapter.healthcheck()
    assert result.status == HealthStatus.FAIL


@respx.mock
def test_bnp_healthcheck_warn_on_empty_index_list() -> None:
    respx.get(BNP_INDEXES_URL).mock(return_value=httpx.Response(200, json={"result": []}))
    adapter = _bnp_adapter()
    result = adapter.healthcheck()
    assert result.status == HealthStatus.WARN


@respx.mock
def test_bnp_healthcheck_warn_reflects_partial_universe_from_prior_fetch() -> None:
    indexes_route = respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    page = [_bnp_product("DE000DDDDDD1")]
    respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(200, json=_bnp_page_response(page, total=999))
    )
    adapter = _bnp_adapter(
        max_pages=1, page_size=1, clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC))
    )
    adapter.fetch_products(["DAX"])
    assert adapter._partial_universe

    result = adapter.healthcheck()
    assert result.status == HealthStatus.WARN
    assert "partial_universe" in result.message
    assert indexes_route.call_count == 2  # once in fetch_products, once in healthcheck


@respx.mock
def test_bnp_factory_builds_correct_api_base_url_from_site_root() -> None:
    src = SourceConfig(
        enabled=True,
        base_url="https://derivate.bnpparibas.com",
        timeout_s=20.0,
        min_interval_s=0.0,
        max_pages=5,
        user_agent="test-agent/1.0",
    )
    http = HttpClient(user_agent=src.user_agent, min_interval_s=src.min_interval_s)
    adapter = _bnp_factory(src, http)
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    result = adapter.healthcheck()
    assert result.status == HealthStatus.PASS


# ===========================================================================
# Citi / CitiFirst
# ===========================================================================


def _citi_adapter(**kwargs: Any) -> CitiFirstTurboAdapter:
    http = HttpClient(user_agent="test-agent/1.0", max_retries=kwargs.pop("max_retries", 3))
    return CitiFirstTurboAdapter(http, **kwargs)


def test_citi_metadata() -> None:
    adapter = _citi_adapter()
    meta = adapter.metadata()
    assert meta.name == "citi"
    assert meta.kind == "product"
    assert adapter.name == "citi"


@respx.mock
def test_citi_fetch_products_parses_real_fixture() -> None:
    respx.post(CITI_SEARCH_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("citi", "productsearch_search_dax_probe_output.json")
        )
    )
    adapter = _citi_adapter(clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert len(snapshots) >= 20
    assert all(isinstance(s, ProductSnapshot) for s in snapshots)
    assert all(len(s.isin) == 12 and s.isin.isalnum() for s in snapshots)
    assert all(s.ratio > 0 for s in snapshots)
    assert all(s.financing_level is not None and s.financing_level > 0 for s in snapshots)
    assert not adapter.last_errors


@respx.mock
def test_citi_product_type_field_used_for_classification() -> None:
    """Citi exposes an explicit, unambiguous `productType` field (observed
    values "OpenEndTurbo"/"MiniFuture" -- live probe, 2026-09-11 research
    session) that the adapter now passes to `classify_product_type` as its
    name-based signal, in addition to the structural barrier comparison.
    A `strike == koBarrier` row explicitly labelled "OpenEndTurbo" must
    classify as TURBO_OPEN_END (name and structure agree); a `MiniFuture`
    row with a real buffer must stay MINI_FUTURE (unchanged from before this
    fix, since structure alone already got it right for that case).
    """
    items = [
        _citi_product("DE00OPENE001", product_type="OpenEndTurbo", strike=20000.0, ko=20000.0),
        _citi_product("DE00MINIL002", product_type="MiniFuture", strike=20000.0, ko=20300.0),
    ]
    respx.post(CITI_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_citi_search_response(items, total_elements_count=2))
    )
    adapter = _citi_adapter(clock=_fixed_clock(datetime(2026, 9, 11, 22, 5, tzinfo=UTC)))

    snapshots = adapter.fetch_products(["DAX"])
    by_isin = {s.isin: s for s in snapshots}

    assert by_isin["DE00OPENE001"].product_type == ProductType.TURBO_OPEN_END
    assert by_isin["DE00MINIL002"].product_type == ProductType.MINI_FUTURE


@respx.mock
def test_citi_closing_price_reference_treated_as_no_live_quote() -> None:
    """Real fixture: every row is referencePriceMethod == "Closing Price".

    Market-hours disambiguation (2026-09-11, ~08:51 UTC): 25/25 DAX rows
    were closing-price snapshots, not live two-way quotes -- bid AND ask
    (not just the zero-sentinel ask) must be discarded, the row forced
    stale, and quality low. Master data must remain intact regardless.
    """
    respx.post(CITI_SEARCH_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("citi", "productsearch_search_dax_probe_output.json")
        )
    )
    adapter = _citi_adapter(clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert snapshots  # fixture is non-empty
    assert all(s.ask is None for s in snapshots)
    assert all(s.bid is None for s in snapshots)
    assert all(s.bid_size is None and s.ask_size is None for s in snapshots)
    assert all(s.quote_presence is False for s in snapshots)
    assert all(s.is_stale is True for s in snapshots)
    assert all(s.quality_score == pytest.approx(0.3) for s in snapshots)
    # Master data must survive even without a live quote.
    assert all(s.isin and len(s.isin) == 12 for s in snapshots)
    assert all(s.financing_level is not None and s.financing_level > 0 for s in snapshots)
    assert all(s.knockout_barrier is not None and s.knockout_barrier > 0 for s in snapshots)
    assert all(s.ratio > 0 for s in snapshots)


@respx.mock
def test_citi_non_live_reference_forces_none_even_with_nonzero_amounts() -> None:
    """A nonzero closing-price bid/ask is just as unusable as a zero one.

    referencePriceMethod alone -- not the numeric value -- decides whether
    this is a live quote.
    """
    product = _citi_product(
        "DE000IIIIII1",
        reference_price_method="Closing Price",
        bid_amount=200.5,
        ask_amount=201.0,
    )
    respx.post(CITI_SEARCH_URL).mock(
        return_value=httpx.Response(
            200, json=_citi_search_response([product], total_elements_count=1)
        )
    )
    adapter = _citi_adapter(clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.bid is None
    assert snapshot.ask is None
    assert snapshot.quote_presence is False
    assert snapshot.is_stale is True
    assert snapshot.quality_score == pytest.approx(0.3)


@respx.mock
def test_citi_live_reference_price_method_keeps_zero_sentinel_fallback() -> None:
    """A referencePriceMethod not in the non-live set uses the plain
    zero-sentinel path: ask=0.0 -> None, but a real nonzero bid survives."""
    product = _citi_product(
        "DE000JJJJJJ1",
        reference_price_method="Live",
        bid_amount=200.0,
        ask_amount=0.0,
    )
    respx.post(CITI_SEARCH_URL).mock(
        return_value=httpx.Response(
            200, json=_citi_search_response([product], total_elements_count=1)
        )
    )
    adapter = _citi_adapter(clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.ask is None
    assert snapshot.bid == pytest.approx(200.0)
    assert snapshot.quote_presence is False


@respx.mock
def test_citi_quotes_summary_logged_as_one_aggregated_line_not_per_row() -> None:
    """25 rows, all `referencePriceMethod == "Closing Price"` -- previously
    25 INFO lines (`citi_ask_zero_sentinel`), now exactly one
    `citi_quotes_summary` INFO line with aggregated counts."""
    respx.post(CITI_SEARCH_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("citi", "productsearch_search_dax_probe_output.json")
        )
    )
    adapter = _citi_adapter(clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC)))

    with structlog.testing.capture_logs() as logs:
        snapshots = adapter.fetch_products(["DAX"])

    summary_events = [e for e in logs if e.get("event") == "citi_quotes_summary"]
    assert len(summary_events) == 1
    summary = summary_events[0]
    assert summary["count_total"] == len(snapshots)
    assert summary["count_ask_missing"] == len(snapshots)
    assert summary["count_bid_missing"] == len(snapshots)
    assert len(summary["example_isins"]) == 5


@respx.mock
def test_citi_partial_universe_flagged_and_surfaced_in_healthcheck() -> None:
    fixture = _load_fixture("citi", "productsearch_search_dax_probe_output.json")
    assert (
        fixture["totalElementsCount"] > fixture["itemsCount"]
    )  # sanity: fixture is the 25-of-33 case

    respx.post(CITI_SEARCH_URL).mock(return_value=httpx.Response(200, json=fixture))
    adapter = _citi_adapter(clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC)))
    adapter.fetch_products(["DAX"])

    assert adapter._partial_universe["DAX"] == (
        fixture["itemsCount"],
        fixture["totalElementsCount"],
    )

    result = adapter.healthcheck()
    assert result.status == HealthStatus.WARN
    assert "partial_universe" in result.message


@respx.mock
def test_citi_underlying_mismatch_defended_against_silent_wrong_result() -> None:
    good = _citi_product("DE000EEEEEE1")
    mismatched = _citi_product("DE000EEEEEE2", underlying_isin="US0378331005")  # wrong underlying
    respx.post(CITI_SEARCH_URL).mock(
        return_value=httpx.Response(
            200, json=_citi_search_response([good, mismatched], total_elements_count=2)
        )
    )
    adapter = _citi_adapter(clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert [s.isin for s in snapshots] == ["DE000EEEEEE1"]
    assert any(e.isin == "DE000EEEEEE2" for e in adapter.last_errors)


@respx.mock
def test_citi_unresolved_underlying_is_skipped_cleanly() -> None:
    search_route = respx.post(CITI_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_citi_search_response([], total_elements_count=0))
    )
    adapter = _citi_adapter()
    # NDX has no independently-verified Citi ISIN at research time -- must be
    # skipped, never guessed.
    assert "NDX" not in _CITI_UNDERLYING_ISINS
    snapshots = adapter.fetch_products(["NDX"])

    assert snapshots == []
    assert search_route.call_count == 0


@respx.mock
def test_citi_schema_drift_missing_field_goes_to_last_errors_not_raised() -> None:
    good = _citi_product("DE000FFFFFF1")
    bad = _citi_product("DE000FFFFFF2")
    del bad["strike"]  # required field missing
    respx.post(CITI_SEARCH_URL).mock(
        return_value=httpx.Response(
            200, json=_citi_search_response([good, bad], total_elements_count=2)
        )
    )
    adapter = _citi_adapter(clock=_fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert len(snapshots) == 1
    assert snapshots[0].isin == "DE000FFFFFF1"
    assert any(e.isin == "DE000FFFFFF2" for e in adapter.last_errors)


@respx.mock
def test_citi_http_500_retries_then_raises_adapter_http_error() -> None:
    respx.post(CITI_SEARCH_URL).mock(return_value=httpx.Response(500))
    adapter = _citi_adapter(max_retries=2)
    with pytest.raises(AdapterHttpError):
        adapter.fetch_products(["DAX"])


@respx.mock
def test_citi_healthcheck_pass() -> None:
    items = [_citi_product("DE000GGGGGG1"), _citi_product("DE000GGGGGG2")]
    respx.post(CITI_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_citi_search_response(items, total_elements_count=2))
    )
    adapter = _citi_adapter()
    result = adapter.healthcheck()
    assert result.status == HealthStatus.PASS
    assert result.ok is True


@respx.mock
def test_citi_healthcheck_fail_on_http_error() -> None:
    respx.post(CITI_SEARCH_URL).mock(return_value=httpx.Response(500))
    adapter = _citi_adapter(max_retries=1)
    result = adapter.healthcheck()
    assert result.status == HealthStatus.FAIL


@respx.mock
def test_citi_healthcheck_warn_on_empty_items() -> None:
    respx.post(CITI_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_citi_search_response([], total_elements_count=0))
    )
    adapter = _citi_adapter()
    result = adapter.healthcheck()
    assert result.status == HealthStatus.WARN


@respx.mock
def test_citi_factory_builds_correct_api_base_url_from_site_root() -> None:
    src = SourceConfig(
        enabled=True,
        base_url="https://de.citifirst.com",
        timeout_s=20.0,
        min_interval_s=0.0,
        max_pages=5,
        user_agent="test-agent/1.0",
    )
    http = HttpClient(user_agent=src.user_agent, min_interval_s=src.min_interval_s)
    adapter = _citi_factory(src, http)
    items = [_citi_product("DE000HHHHHH1")]
    respx.post(CITI_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_citi_search_response(items, total_elements_count=1))
    )
    result = adapter.healthcheck()
    assert result.status == HealthStatus.PASS


# ===========================================================================
# Cross-cutting: protocol conformance, registration, combined coverage
# ===========================================================================


def test_bnp_and_citi_satisfy_product_source_adapter_protocol() -> None:
    assert isinstance(_bnp_adapter(), ProductSourceAdapter)
    assert isinstance(_citi_adapter(), ProductSourceAdapter)


def test_registration_is_idempotent() -> None:
    assert "bnp_paribas" in PRODUCT_ADAPTER_FACTORIES
    assert "citi" in PRODUCT_ADAPTER_FACTORIES
    # Calling _register() again must not raise (each factory module's own
    # registration guard short-circuits registering an already-known name).
    _register()
    _register()
    assert "bnp_paribas" in PRODUCT_ADAPTER_FACTORIES
    assert "citi" in PRODUCT_ADAPTER_FACTORIES


@respx.mock
def test_combined_bnp_and_citi_yield_at_least_20_snapshots_with_both_directions() -> None:
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_load_fixture(
                "bnp_paribas", "productlist_leverage_dax_market_hours_ask_present.json"
            ),
        )
    )
    respx.post(CITI_SEARCH_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("citi", "productsearch_search_dax_probe_output.json")
        )
    )
    clock = _fixed_clock(datetime(2026, 9, 10, 22, 5, tzinfo=UTC))
    bnp = _bnp_adapter(clock=clock)
    citi = _citi_adapter(clock=clock)

    combined = bnp.fetch_products(["DAX"]) + citi.fetch_products(["DAX"])

    assert len(combined) >= 20
    assert all(s.isin.isalnum() and len(s.isin) == 12 for s in combined)
    assert all(s.ratio > 0 for s in combined)
    assert all(s.financing_level is not None and s.financing_level > 0 for s in combined)
    assert {s.direction for s in combined} == {Direction.LONG, Direction.SHORT}


def test_bnp_fixture_and_synthetic_builders_produce_valid_json() -> None:
    # Guards the test-only synthetic builders themselves: a deep-copy + a
    # round trip through json must not raise (catches accidental non-JSON
    # values such as datetimes creeping into a synthetic record).
    record = copy.deepcopy(_bnp_product("DE000ZZZZZZ1"))
    json.dumps(record)
    citi_record = copy.deepcopy(_citi_product("DE000ZZZZZZ2"))
    json.dumps(citi_record)


# ===========================================================================
# Befund 1 (2026-09-13): BNP derivativeTypeId coverage -- "Unlimited Turbo"
# (ids 67/68) added, every discovered id documented include/exclude
# ===========================================================================


def test_bnp_derivative_type_catalog_includes_unlimited_turbo() -> None:
    """The single largest coverage gap this session found: BNP's 'Unlimited
    Turbo' product line (ids 67/68) is structurally an open-end turbo
    (strike==knockOut, maturity=-1, live-verified) but was never requested
    before this fix."""
    assert 67 in _BNP_DERIVATIVE_TYPE_IDS
    assert 68 in _BNP_DERIVATIVE_TYPE_IDS
    assert _BNP_DERIVATIVE_TYPE_CATALOG[67].name == "Unlimited Long"
    assert _BNP_DERIVATIVE_TYPE_CATALOG[67].included is True
    assert _BNP_DERIVATIVE_TYPE_CATALOG[68].name == "Unlimited Short"
    assert _BNP_DERIVATIVE_TYPE_CATALOG[68].included is True


def test_bnp_derivative_type_catalog_excludes_non_turbo_families_with_reasons() -> None:
    """Every id this session's live census found for BNP's DAX book that is
    NOT a turbo/mini-future/open-end-KO product must be explicitly excluded
    (never requested) with a documented, non-empty reason -- Optionsscheine,
    Faktor-Zertifikate, and Discount/Bonus/Express families per Befund 1's
    explicit scope decision."""
    excluded_by_name = {
        entry.name: (tid, entry)
        for tid, entry in _BNP_DERIVATIVE_TYPE_CATALOG.items()
        if not entry.included
    }
    expected_excluded_names = {
        "Discount",
        "Call",
        "Put",
        "Bonus",
        "Discount Call",
        "Reverse Bonus",
        "Discount Put",
        "Capped Bonus",
        "Capped Reverse Bonus",
        "Memory Express Zertifikat",
        "Discount Call Plus",
        "Discount Put Plus",
        "Bonus Call",
        "Strukturierte Anleihe",
        "Andere Zinsanleihe",
        "Fix Kupon Express",
        "Express-Zertifikat",
        "Faktor Long",
        "Faktor Short",
        "Inline",
    }
    assert expected_excluded_names <= excluded_by_name.keys()
    for name in expected_excluded_names:
        tid, entry = excluded_by_name[name]
        assert entry.reason.strip(), f"id {tid} ({name}) has no documented exclusion reason"
        assert tid not in _BNP_DERIVATIVE_TYPE_IDS


def test_bnp_derivative_type_catalog_every_entry_has_a_reason() -> None:
    for tid, entry in _BNP_DERIVATIVE_TYPE_CATALOG.items():
        assert entry.name.strip(), f"id {tid} has no name"
        assert entry.reason.strip(), f"id {tid} has no reason"


@respx.mock
def test_bnp_unlimited_turbo_real_fixture_classified_as_turbo_open_end() -> None:
    """Real-captured (2026-09-13) BNP DAX rows under the new ids 67/68 --
    structurally identical to ids 7/9's open-end turbo (strike==knockOut,
    maturity=-1) -- must classify as TURBO_OPEN_END, not UNKNOWN, and price
    exactly like any other open-end turbo (positive ratio, financing_level,
    both directions present in the fixture)."""
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_load_fixture(
                "bnp_paribas", "productlist_leverage_dax_unlimited_turbo_sample.json"
            ),
        )
    )
    adapter = _bnp_adapter(clock=_fixed_clock(datetime(2026, 9, 13, 12, 0, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    assert snapshots
    assert not adapter.last_errors
    assert all(s.product_type == ProductType.TURBO_OPEN_END for s in snapshots)
    assert all(s.open_end is True for s in snapshots)
    assert all(s.ratio > 0 for s in snapshots)
    assert all(s.financing_level is not None and s.financing_level > 0 for s in snapshots)
    directions = {s.direction for s in snapshots}
    assert directions == {Direction.LONG, Direction.SHORT}


@respx.mock
def test_bnp_excluded_product_type_rows_are_rejected_not_mispriced() -> None:
    """Defense in depth (Befund 1 (c)/(d)): even though Call/Bonus/Faktor ids
    are never requested by this adapter (see the catalog tests above), a
    real-captured (2026-09-13) sample of exactly those rows -- as if they
    had somehow reached this adapter -- must be cleanly rejected via the
    normal schema-drift error path (missing strike/knockOut/ratio, or an
    unmapped direction like 'call'), never silently priced as if they were
    turbo products."""
    respx.get(BNP_INDEXES_URL).mock(
        return_value=httpx.Response(
            200, json=_load_fixture("bnp_paribas", "underlying_indexes.json")
        )
    )
    respx.post(BNP_LEVERAGE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_load_fixture(
                "bnp_paribas", "productlist_leverage_dax_excluded_product_types_sample.json"
            ),
        )
    )
    adapter = _bnp_adapter(clock=_fixed_clock(datetime(2026, 9, 13, 12, 0, tzinfo=UTC)))
    snapshots = adapter.fetch_products(["DAX"])

    # Not one Call/Bonus row produced a (necessarily wrong) snapshot -- every
    # single row in this fixture is missing a required turbo field.
    assert snapshots == []
    assert len(adapter.last_errors) > 0
    assert all(isinstance(e, IssuerRowError) for e in adapter.last_errors)
