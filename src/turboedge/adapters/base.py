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
from collections.abc import Mapping
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


@dataclass(frozen=True, slots=True)
class ProductFetchContext:
    """Optional, additive per-run context passed to every product adapter's
    ``fetch_products`` (Befund 2, 2026-09-13 measurement session).

    ``pipeline/universe.py`` previously called every adapter's
    ``fetch_products(underlying_ids)`` with nothing else -- fine for
    BNP/Citi/CSV (which need nothing more), but the gettex adapter accepts an
    optional ``reference_spot``/``daily_close_reference`` sanity cross-check
    for its internally-derived S_ref (see ``adapters/gettex.py`` module
    docstring) that the generic ``ProductSourceAdapter`` call path simply had
    no channel to pass through at all -- ``run_universe`` only ever called
    ``adapter.fetch_products(underlying_ids)``, so gettex's richer
    single-adapter contract (``reference_spot=``/``daily_close_reference=``)
    was reachable from a direct test/script but not from the real pipeline.
    Once threaded through, a same-run daily close lets gettex's ``S_ref``
    (otherwise unchecked against anything, see that module's Pitfall notes)
    be sanity-cross-checked even when every issuer-quote source in the same
    scan is itself down or stale.

    Extending the shared :class:`~turboedge.adapters.registry.
    ProductSourceAdapter` Protocol with one optional, typed field (rather
    than reaching into each adapter's own kwargs via ``inspect.signature`` to
    decide what to pass) keeps the contract statically checkable by mypy:
    every adapter's ``fetch_products`` signature is verified against the
    Protocol at class-definition time, a caller cannot typo a kwarg name that
    silently never reaches the adapter it was meant for, and adding a second
    context field later is a one-place, compiler-checked change instead of
    an ad-hoc ``getattr``/``inspect`` guess repeated at every call site. The
    field is optional and every adapter that has no use for it (BNP, Citi,
    CSV import) simply accepts and ignores it -- no adapter is forced to act
    on context it doesn't need, so this is purely additive.
    """

    # Same-day daily-close price per canonical underlying_id (e.g. from
    # ``PriceSource.fetch_daily_bars``) -- a same-issuer-outage-proof sanity
    # cross-check for an adapter's own internally-derived reference spot.
    daily_close_reference: Mapping[str, float] | None = None
    # A genuinely independent *live* reference spot per underlying_id, when
    # the caller has one from a source other than the adapter itself (tight
    # tolerance cross-check, preferred over ``daily_close_reference`` when
    # both are available -- see ``adapters/gettex.py``'s
    # ``fetch_products`` docstring for the priority order).
    reference_spot: Mapping[str, float] | None = None


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

    def get_bytes(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        """Raw response body, for sources that serve archives rather than text.

        Added for W12-D: the CFTC publishes Commitments of Traders history as
        annual ZIP files, and routing those through `get_text` would decode
        binary data as UTF-8 and corrupt it. Shares the same per-host rate
        limiting, retry policy and stats counters as the other accessors --
        the point is only that the body is not decoded.
        """
        return self._request("GET", url, params=params, headers=headers).content

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
