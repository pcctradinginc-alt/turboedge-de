"""Shared adapter contract and HTTP client used by every data source adapter.

Every concrete adapter (ECB EST, the yfinance underlying-price fallback, and
- in later milestones - Deutsche Boerse, Boerse Stuttgart, issuer feeds)
implements the :class:`DataSourceAdapter` protocol and talks to the network
only through :class:`HttpClient`, which centralizes retry policy, per-host
rate limiting and the User-Agent so behavior is consistent and testable
(contract tests via ``respx``) across every source.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from turboedge.storage.schemas import HealthStatus

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class AdapterError(Exception):
    """Base class for all adapter-level failures."""


class AdapterHttpError(AdapterError):
    """A request ultimately failed (after retries, if any) with an HTTP/transport error."""


class AdapterMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: str  # "product" | "price" | "reference_rate"
    version: str
    homepage: str | None = None


class HealthCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    status: HealthStatus
    ok: bool
    latency_ms: float | None
    checked_at: datetime
    message: str


@runtime_checkable
class DataSourceAdapter(Protocol):
    """Structural contract every source adapter satisfies.

    Deliberately generic over both product-quote adapters and price /
    reference-rate adapters: ``fetch`` returns whatever raw payload the
    source hands back, ``normalize`` turns it into validated, typed records
    (e.g. a list of ``ProductSnapshot``, a list of ``UnderlyingBar``, or a
    single reference rate), independent of what those raw/normalized types
    concretely are for a given adapter.
    """

    def fetch(self, **kwargs: Any) -> Any:
        """Retrieve the raw payload from the source. All network I/O happens here."""
        ...

    def normalize(self, raw: Any, **kwargs: Any) -> Any:
        """Convert a raw payload (as returned by ``fetch``) into typed records."""
        ...

    def healthcheck(self) -> HealthCheckResult:
        """Cheap liveness/quality probe used by ``turboedge sources health``."""
        ...

    def metadata(self) -> AdapterMetadata:
        """Static description of this adapter (name, kind, version)."""
        ...


@dataclass
class HttpClientStats:
    """Running counters exposed for adapters' ``healthcheck()`` implementations."""

    request_count: int = 0
    error_count: int = 0
    retry_count: int = 0


class _HostRateLimiter:
    """Serializes requests to one host to be >= ``min_interval_s`` apart."""

    def __init__(self, min_interval_s: float) -> None:
        self._min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last_request_at: float | None = None

    def wait(self) -> None:
        if self._min_interval_s <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if self._last_request_at is not None:
                remaining = self._min_interval_s - (now - self._last_request_at)
                if remaining > 0:
                    time.sleep(remaining)
            self._last_request_at = time.monotonic()


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    return False


class HttpClient:
    """A shared, retrying, per-host rate-limited HTTP client wrapper for adapters.

    - Retries transient failures (timeouts, connection errors, 429/5xx) with
      exponential backoff + jitter, up to ``max_retries`` attempts.
    - Enforces a minimum interval between requests to the same host, tracked
      independently per host so one slow source never throttles another.
    - Tracks request/error/retry counts (``.stats``) for adapters'
      ``healthcheck()`` implementations.
    - Every failure (after retries are exhausted, or immediately for a
      non-retryable error) is re-raised as :class:`AdapterHttpError`, so
      callers only need to catch one exception type.
    """

    def __init__(
        self,
        *,
        user_agent: str,
        timeout_s: float = 10.0,
        min_interval_s: float = 0.0,
        max_retries: int = 3,
    ) -> None:
        self._user_agent = user_agent
        self._client = httpx.Client(timeout=timeout_s, headers={"User-Agent": user_agent})
        self._min_interval_s = min_interval_s
        self._max_retries = max_retries
        self._limiters: dict[str, _HostRateLimiter] = {}
        self._limiters_lock = threading.Lock()
        self.stats = HttpClientStats()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @property
    def user_agent(self) -> str:
        return self._user_agent

    def _limiter_for(self, host: str) -> _HostRateLimiter:
        with self._limiters_lock:
            limiter = self._limiters.get(host)
            if limiter is None:
                limiter = _HostRateLimiter(self._min_interval_s)
                self._limiters[host] = limiter
            return limiter

    def _on_retry(self, retry_state: RetryCallState) -> None:
        self.stats.retry_count += 1

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_body: Any = None,
    ) -> httpx.Response:
        host = httpx.URL(url).host

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=8.0),
            retry=retry_if_exception(_is_retryable_error),
            before_sleep=self._on_retry,
        )
        def _do() -> httpx.Response:
            self._limiter_for(host).wait()
            self.stats.request_count += 1
            try:
                response = self._client.request(
                    method, url, params=params, headers=headers, json=json_body
                )
                response.raise_for_status()
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError):
                self.stats.error_count += 1
                raise
            return response

        try:
            return _do()
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            raise AdapterHttpError(f"{method} {url} failed: {exc}") from exc

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return self._request("GET", url, params=params, headers=headers).json()

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        return self._request("GET", url, params=params, headers=headers).text

    def post_json(
        self,
        url: str,
        *,
        json: Any = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """POST a JSON body and parse the JSON response.

        Shares the exact same retry policy, per-host rate limiting and
        ``AdapterHttpError`` wrapping as :meth:`get_json` -- used by issuer
        feed adapters (BNP Paribas, Citi) whose product-search endpoints are
        POST-only.
        """
        return self._request("POST", url, headers=headers, json_body=json).json()
