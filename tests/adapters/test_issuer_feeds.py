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
    _CITI_DEFAULT_BASE_URL,
    _CITI_UNDERLYING_ISINS,
    BnpParibasTurboAdapter,
    CitiFirstTurboAdapter,
    IssuerRowError,
    _bnp_factory,
    _citi_factory,
    _parse_berlin_naive_to_utc,
    _register,
)
from turboedge.adapters.registry import PRODUCT_ADAPTER_FACTORIES, ProductSourceAdapter
from turboedge.config import SourceConfig
from turboedge.storage.schemas import Direction, HealthStatus, ProductSnapshot

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
) -> dict[str, Any]:
    return {
        "isin": isin,
        "wkn": isin[-6:],
        "currencyCode": "EUR",
        "productType": "MiniFuture",
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
