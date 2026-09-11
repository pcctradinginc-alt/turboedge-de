from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from turboedge.adapters.base import AdapterHttpError, HttpClient


@respx.mock
def test_get_json_success() -> None:
    route = respx.get("https://example.com/data").mock(
        return_value=httpx.Response(200, json={"a": 1})
    )
    with HttpClient(user_agent="test-agent/1.0") as client:
        data = client.get_json("https://example.com/data")
    assert data == {"a": 1}
    assert route.called
    assert client.stats.request_count == 1
    assert client.stats.error_count == 0


@respx.mock
def test_post_json_success() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["content_type"] = request.headers.get("content-type")
        return httpx.Response(200, json={"ok": True})

    respx.post("https://example.com/search").mock(side_effect=handler)
    with HttpClient(user_agent="test-agent/1.0") as client:
        data = client.post_json(
            "https://example.com/search",
            json={"underlyingIsins": ["DE0008469008"]},
            headers={"Content-Type": "application/json"},
        )
    assert data == {"ok": True}
    assert captured["body"] == {"underlyingIsins": ["DE0008469008"]}
    assert captured["content_type"] == "application/json"


@respx.mock
def test_post_json_retries_on_503_then_succeeds() -> None:
    route = respx.post("https://example.com/flaky-post").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    with HttpClient(user_agent="test-agent/1.0", max_retries=5) as client:
        data = client.post_json("https://example.com/flaky-post", json={"a": 1})
    assert data == {"ok": True}
    assert route.call_count == 2
    assert client.stats.retry_count == 1


@respx.mock
def test_post_json_raises_adapter_http_error_after_exhausting_retries() -> None:
    respx.post("https://example.com/always-down-post").mock(return_value=httpx.Response(500))
    with (
        HttpClient(user_agent="test-agent/1.0", max_retries=2) as client,
        pytest.raises(AdapterHttpError),
    ):
        client.post_json("https://example.com/always-down-post", json={})


@respx.mock
def test_get_text_success() -> None:
    respx.get("https://example.com/text").mock(return_value=httpx.Response(200, text="hello"))
    with HttpClient(user_agent="test-agent/1.0") as client:
        text = client.get_text("https://example.com/text")
    assert text == "hello"


@respx.mock
def test_retries_on_503_then_succeeds() -> None:
    route = respx.get("https://example.com/flaky").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    with HttpClient(user_agent="test-agent/1.0", max_retries=5) as client:
        data = client.get_json("https://example.com/flaky")
    assert data == {"ok": True}
    assert route.call_count == 3
    assert client.stats.error_count == 2
    assert client.stats.retry_count == 2


@respx.mock
def test_does_not_retry_on_404() -> None:
    route = respx.get("https://example.com/missing").mock(return_value=httpx.Response(404))
    with (
        HttpClient(user_agent="test-agent/1.0", max_retries=5) as client,
        pytest.raises(AdapterHttpError),
    ):
        client.get_json("https://example.com/missing")
    assert route.call_count == 1  # non-retryable, no backoff attempts


@respx.mock
def test_raises_adapter_http_error_after_exhausting_retries() -> None:
    respx.get("https://example.com/always-down").mock(return_value=httpx.Response(500))
    with (
        HttpClient(user_agent="test-agent/1.0", max_retries=2) as client,
        pytest.raises(AdapterHttpError),
    ):
        client.get_json("https://example.com/always-down")
    assert client.stats.request_count == 2


@respx.mock
def test_user_agent_header_is_sent() -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, json={})

    respx.get("https://example.com/ua").mock(side_effect=handler)
    with HttpClient(user_agent="MyAgent/9.9") as client:
        client.get_json("https://example.com/ua")
    assert captured["ua"] == "MyAgent/9.9"


@respx.mock
def test_rate_limiter_sleeps_between_requests_to_same_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("turboedge.adapters.base.time.sleep", lambda s: sleep_calls.append(s))

    respx.get("https://example.com/a").mock(return_value=httpx.Response(200, json={}))
    respx.get("https://example.com/b").mock(return_value=httpx.Response(200, json={}))
    with HttpClient(user_agent="test-agent", min_interval_s=1.0) as client:
        client.get_json("https://example.com/a")
        client.get_json("https://example.com/b")
    assert len(sleep_calls) == 1
    assert sleep_calls[0] > 0


@respx.mock
def test_rate_limiter_independent_per_host(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("turboedge.adapters.base.time.sleep", lambda s: sleep_calls.append(s))

    respx.get("https://a.example.com/x").mock(return_value=httpx.Response(200, json={}))
    respx.get("https://b.example.com/x").mock(return_value=httpx.Response(200, json={}))
    with HttpClient(user_agent="test-agent", min_interval_s=5.0) as client:
        client.get_json("https://a.example.com/x")
        client.get_json("https://b.example.com/x")
    assert sleep_calls == []  # different hosts never throttle each other
