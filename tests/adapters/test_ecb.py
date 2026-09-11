from __future__ import annotations

import httpx
import pytest
import respx

from turboedge.adapters.base import HttpClient
from turboedge.adapters.ecb import EcbEstrAdapter
from turboedge.storage.schemas import HealthStatus

BASE_URL = "https://data-api.ecb.europa.eu"
SERIES_URL = f"{BASE_URL}/service/data/EST/B.EU000A2X2A25.WT"

SAMPLE_RESPONSE = {
    "dataSets": [
        {
            "series": {
                "0:0:0:0:0": {
                    "observations": {
                        "0": [3.15],
                        "1": [3.14],
                        "2": [3.16],
                        "3": [3.15],
                        "4": [3.17],
                    }
                }
            }
        }
    ],
    "structure": {
        "dimensions": {
            "observation": [
                {
                    "values": [
                        {"id": "2026-09-04"},
                        {"id": "2026-09-05"},
                        {"id": "2026-09-08"},
                        {"id": "2026-09-09"},
                        {"id": "2026-09-10"},
                    ]
                }
            ]
        }
    },
}


def _adapter() -> EcbEstrAdapter:
    http = HttpClient(user_agent="test-agent/1.0")
    return EcbEstrAdapter(http, base_url=BASE_URL, fallback_rate=0.03)


@respx.mock
def test_fetch_latest_parses_and_converts_percent_to_decimal() -> None:
    respx.get(SERIES_URL).mock(return_value=httpx.Response(200, json=SAMPLE_RESPONSE))
    adapter = _adapter()
    latest = adapter.fetch_latest()
    assert latest is not None
    assert latest.period.isoformat() == "2026-09-10"
    assert latest.value == pytest.approx(0.0317)


@respx.mock
def test_get_estr_returns_latest_value() -> None:
    respx.get(SERIES_URL).mock(return_value=httpx.Response(200, json=SAMPLE_RESPONSE))
    adapter = _adapter()
    assert adapter.get_estr() == pytest.approx(0.0317)


@respx.mock
def test_get_estr_falls_back_on_http_failure() -> None:
    respx.get(SERIES_URL).mock(return_value=httpx.Response(500))
    adapter = EcbEstrAdapter(
        HttpClient(user_agent="test-agent/1.0", max_retries=1),
        base_url=BASE_URL,
        fallback_rate=0.03,
    )
    assert adapter.get_estr() == pytest.approx(0.03)


@respx.mock
def test_fetch_latest_returns_none_on_malformed_response() -> None:
    respx.get(SERIES_URL).mock(return_value=httpx.Response(200, json={"unexpected": True}))
    adapter = _adapter()
    assert adapter.fetch_latest() is None


@respx.mock
def test_fetch_latest_returns_none_on_empty_series() -> None:
    respx.get(SERIES_URL).mock(
        return_value=httpx.Response(200, json={"dataSets": [{"series": {}}], "structure": {}})
    )
    adapter = _adapter()
    assert adapter.fetch_latest() is None


@respx.mock
def test_healthcheck_pass() -> None:
    respx.get(SERIES_URL).mock(return_value=httpx.Response(200, json=SAMPLE_RESPONSE))
    adapter = _adapter()
    result = adapter.healthcheck()
    assert result.ok is True
    assert result.status == HealthStatus.PASS
    assert "3.17" in result.message


@respx.mock
def test_healthcheck_fail_on_http_error() -> None:
    respx.get(SERIES_URL).mock(return_value=httpx.Response(500))
    adapter = EcbEstrAdapter(
        HttpClient(user_agent="test-agent/1.0", max_retries=1),
        base_url=BASE_URL,
        fallback_rate=0.03,
    )
    result = adapter.healthcheck()
    assert result.ok is False
    assert result.status == HealthStatus.FAIL


def test_metadata() -> None:
    adapter = _adapter()
    meta = adapter.metadata()
    assert meta.name == "ecb_estr"
    assert meta.kind == "reference_rate"
