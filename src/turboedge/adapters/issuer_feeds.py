"""Issuer-feed adapters: BNP Paribas and Citi (CitiFirst) turbo/KO products.

Both APIs were reverse-engineered and confirmed live (twice, hours apart) by
the Round 2 data-source research session -- see
``docs/data_sources.md`` for the summary table and the full research log for
every probe, pitfall and fixture referenced below. Both are plain,
unauthenticated JSON REST APIs with no anti-bot signature of any kind
(unlike Boerse Stuttgart/Cloudflare or Boerse Frankfurt/salted-hash-CORS,
which remain interface-only per that research).

BNP Paribas (``derivate.bnpparibas.com``)
------------------------------------------
- ``GET  {base}/underlying/indexes`` resolves the underlying-name -> BNP
  ``instrumentId`` mapping (cached after the first call). Only *index*
  underlyings are covered by this endpoint (DAX, Euro Stoxx 50, Dow Jones,
  Nasdaq 100, TecDAX, MDAX, SDAX, SMI, ATX, CAC 40, FTSE MIB, IBEX 35, AEX,
  OMX Stockholm 30 at the time of research) -- an ``underlying_id`` that
  cannot be resolved this way (e.g. FX pairs, metals, energy -- no commodity/
  FX index endpoint was researched) is skipped cleanly with a log line, never
  guessed.
- ``POST {base}/productlist/leverage`` is the actual product search. No auth
  header is required (confirmed live with a bare, cookie-less client).
  ``productSetIds`` **must** be ``null`` -- a guessed non-null default
  silently returns ``total: 0`` for every query (no HTTP error at all).
- Pagination: ``offset``/``limit``. A dedicated live probe in this session
  confirmed ``limit`` up to (at least) 1000 is honored by the server (the
  DAX book currently totals ~4340 across all derivative-type variants), so a
  page size of 1000 comfortably covers the full book within a handful of
  pages -- well under ``max_pages`` from ``SourceConfig``. If ``total``
  still exceeds ``max_pages * _BNP_PAGE_SIZE`` for some underlying, this is
  logged as ``bnp_partial_universe`` (WARN) with covered/total counts and
  surfaced via :meth:`BnpParibasTurboAdapter.healthcheck`, never silently
  truncated. The same live probe confirmed no server-side leverage-range or
  ``isKnockedOut`` filter exists under any of the plausible ``filterSelections``
  field-key guesses tried (``keyFigures.leverage``/``leverage``/
  ``first.leverage`` had zero effect on ``total``; ``config.isKnockedOut``
  had zero effect; a bare ``isKnockedOut`` key returned ``total: 0`` for
  *every* row -- inconsistent/unreliable, not a working filter) -- so no
  server-side volume reduction is attempted; leverage-band and
  knocked-out/bid-only filtering is the documented job of
  ``universe/filters.py`` and ``ranking/gates.py`` downstream, not this
  adapter, which normalizes every row honestly (including knocked-out ones,
  via the ``knocked_out``/``bid_only`` fields) and lets those later stages
  decide.
- Pitfall 1 (session-critical): outside BNP's active quoting session, the
  ``ask`` key is dropped from the JSON object **entirely** (not ``null`` --
  absent). Code that does ``p["ask"]`` crashes; this adapter always uses
  ``p.get("ask")`` and treats a missing key identically to
  ``config.isBidOnly=True`` (no live two-way quote).
- Pitfall 2 (session-critical): ``bidDate``/``askDate`` carry **no UTC
  offset** (e.g. ``"2026-09-10T21:48:00.034"``), unlike the top-level
  ``responseDate``/``keyFigures.lastUpdate`` (both explicit ``"...Z"`` UTC).
  These naive-looking timestamps are Europe/Berlin local time (evidenced by
  clustering just before the German OTC 22:00 CET/CEST closing auction) --
  naive UTC parsing would silently shift every BNP quote timestamp by 1-2
  hours. Always localized via real ``Europe/Berlin`` tzdata (correct
  CET/CEST per calendar date) before conversion to UTC.
- Pitfall 3 (session-critical, Build Contract BEFUND 1): ``first.price`` (the
  underlying reference price, -> ``ProductSnapshot.underlying_price_ref``)
  carries its own, separate timestamp ``first.priceDate`` (same naive
  Europe/Berlin convention as ``bidDate``/``askDate`` -- see Pitfall 2) plus a
  same-day flag ``first.isPriceToday``. This timestamp is **not** the same as
  the product's own ``bidDate``/``askDate``: live probing (BEFUND 1
  measurement, see ``docs/data_sources.md``/scan review) found ``first.price``
  batched/throttled to only 1-2 distinct values across an entire ~4300-row DAX
  page, while individual products' bid/ask update continuously -- so a fresh
  ``bidDate``/``askDate`` does **not** imply a fresh ``first.price``. This
  adapter therefore parses ``first.priceDate`` into
  ``ProductSnapshot.underlying_price_ref_timestamp`` as an independent field;
  ``pipeline/scan.py._resolve_spot`` uses *that* timestamp (never
  ``quote_timestamp``) to decide whether ``underlying_price_ref`` is fresh
  enough to use in place of the cross-issuer consensus spot.

Citi / CitiFirst (``de.citifirst.com``)
-----------------------------------------
- ``POST {base}/ProductSearch/de-DE/Search`` with body
  ``{"underlyingIsins": [<isin>]}`` -- the **bare, unwrapped** filter object,
  not ``{"filter": {...}, "minPushItem": ..., "maxPushItem": ...}`` (an
  earlier, wrong guess based on a misread Vuex action signature). Two wrong
  body shapes were confirmed live to return HTTP 200 with a **silently
  unrelated** default result set (not an error) -- so this adapter always
  asserts every returned row's ``underlyings[].isin`` actually matches the
  requested ISIN, and treats a mismatch as a row-level error rather than
  trusting the response shape alone.
- Underlying coverage: only ``DAX`` (``DE0008469008``) has an
  independently-verified ISIN from this research; no other
  ``configs/universe.yaml``-enabled underlying (``NDX``, ``EURUSD``, ``XAU``)
  had a verified Citi ``underlyingIsins`` value at research time (FX/metals
  underlyings likely are not ISIN-addressable in this API at all -- never
  researched, never guessed). Requesting any other ``underlying_id`` is
  skipped cleanly with a log line (CLAUDE.md rule 29: never guess
  pricing-critical identifiers).
- Pagination: **unresolved, structural**. The response caps ``items`` at 25
  (``itemsCount``) regardless of ``totalElementsCount`` (33 for DAX,
  confirmed identical across two separate live pulls); a ``?skip=25``
  query-string guess had no effect. No working way to retrieve further rows
  was found. This adapter therefore fetches a single page and, whenever
  ``totalElementsCount > itemsCount``, logs ``citi_partial_universe`` (WARN)
  with covered/total counts and surfaces it via
  :meth:`CitiFirstTurboAdapter.healthcheck` -- never silently truncated.
- ``ask == 0.0`` was observed on every sampled row in both live pulls,
  including rows with ``referencePriceMethod`` suggesting auction-only
  pricing. A dedicated market-hours pull (2026-09-11, ~08:51 UTC, well
  inside Xetra trading hours) disambiguated this: every one of the 25
  returned DAX products had ``ask == 0.0`` *and*
  ``referencePriceMethod == "Closing Price"`` -- i.e. Citi's search endpoint
  is returning end-of-day/reference pricing, not a live two-way market, for
  this underlying at this time. Per CLAUDE.md rule 29 (never silently
  impute pricing-critical data), any row whose ``referencePriceMethod`` is
  in ``_CITI_NON_LIVE_REFERENCE_PRICE_METHODS`` (currently just
  ``"Closing Price"`` -- the only value observed across every fixture and
  live pull to date) has ``bid``/``ask`` forced to ``None`` and
  ``quote_presence=False``/``is_stale=True`` *regardless of the numeric
  value* -- a nonzero closing price is just as unusable as a live ask as a
  zero one would be, so this is checked before, and independently of, the
  zero-sentinel handling below. Master data (ISIN, WKN, financing_level,
  knockout_barrier, ratio, maturity, ...) is unaffected and still populated
  -- still useful for cross-issuer universe/master-data purposes even
  without a live quote. Rows whose ``referencePriceMethod`` is *not* one of
  the known non-live values fall through to the plain zero-sentinel
  handling: ``ask == 0.0`` (or ``bid == 0.0``) is always treated as "no live
  ask/bid" (``=None``, ``quote_presence=False``), with
  ``referencePriceMethod`` surfaced in the (DEBUG-level, see
  ``_QuoteAnomalyStats``) per-row log line so a future, currently-unseen
  ``referencePriceMethod`` value can be triaged and, if warranted, added to
  ``_CITI_NON_LIVE_REFERENCE_PRICE_METHODS``.
- ``price.timeStamp`` carries no UTC offset either -- same Europe/Berlin
  local-time treatment as BNP's ``bidDate``/``askDate`` (evidenced by
  ``underlyings[].origin.timeZone == "CET"`` and values clustering just
  before the ~22:00 German close).

Neither adapter imputes a missing/zero bid or ask (CLAUDE.md rule 29): a
missing or zero-sentinel price is always represented as ``None`` with
``quote_presence=False``, never guessed from the other side of the market or
from ``underlying_price_ref``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog
from pydantic import ValidationError

from turboedge.adapters.base import (
    AdapterError,
    AdapterHttpError,
    AdapterMetadata,
    HealthCheckResult,
    HttpClient,
)
from turboedge.config import SourceConfig
from turboedge.storage.schemas import Direction, HealthStatus, ProductSnapshot
from turboedge.universe.classify import classify_product_type
from turboedge.universe.underlying_map import resolve_underlying_id

logger = structlog.get_logger(__name__)

_DEFAULT_STALE_AFTER_S = 900.0


@dataclass(frozen=True)
class IssuerRowError:
    """Error record for one product row that failed parsing/normalization."""

    source: str
    isin: str | None
    error: str


# -- shared helpers -----------------------------------------------------------


def _berlin_tz() -> ZoneInfo | None:
    """Return the Europe/Berlin IANA zone, or ``None`` if tzdata isn't loadable.

    Deliberately never falls back to a guessed fixed offset -- CLAUDE.md rule
    29 forbids silently imputing pricing-critical data, and a fixed-offset
    guess is wrong for roughly half the year (CET vs. CEST). Callers must
    treat ``None`` as a hard parse failure, same convention as
    ``adapters/csv_import.py``.
    """
    try:
        return ZoneInfo("Europe/Berlin")
    except ZoneInfoNotFoundError:
        logger.error("issuer_feeds_zoneinfo_load_failed", tz="Europe/Berlin")
        return None


def _parse_berlin_naive_to_utc(raw: str | None) -> datetime | None:
    """Parse a naive (no-UTC-offset) ISO timestamp as Europe/Berlin -> UTC.

    Both BNP (``bidDate``/``askDate``) and Citi (``price.timeStamp``) emit
    quote timestamps with no offset suffix that are, per this session's
    research, Europe/Berlin local time -- never UTC. Returns ``None`` for a
    missing/empty/unparseable value or if tzdata cannot be loaded, rather
    than guessing.
    """
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        # Already offset-aware (not expected for these two fields, but
        # handled defensively rather than mis-localizing an already-correct
        # timestamp).
        return dt.astimezone(UTC)
    tz = _berlin_tz()
    if tz is None:
        return None
    return dt.replace(tzinfo=tz).astimezone(UTC)


def _parse_utc_z(raw: str | None) -> datetime | None:
    """Parse an explicit UTC (``...Z`` suffixed) ISO timestamp."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).astimezone(UTC)
    except ValueError:
        return None


def _raw_hash(record: dict[str, Any]) -> str:
    """sha256 of the canonical JSON encoding of a raw source record."""
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _quality_score(*, quote_presence: bool, is_stale: bool) -> float:
    """1.0 fresh with bid+ask; 0.6 stale; 0.3 without a usable ask."""
    if not quote_presence:
        return 0.3
    if is_stale:
        return 0.6
    return 1.0


def _resolve_zero_as_missing(value: float | None) -> tuple[float | None, bool]:
    """Treat ``0.0`` as "missing" for a bid/ask price (documented sentinel).

    Returns ``(value_or_none, was_sentinel)``.
    """
    if value is None:
        return None, False
    if value == 0.0:
        return None, True
    return value, False


@dataclass
class _QuoteAnomalyStats:
    """Aggregates per-row "no live ask/bid" anomalies across one fetch.

    Previously every affected row logged its own INFO-level line
    (``bnp_ask_key_absent``/``bnp_ask_zero_sentinel``/``bnp_bid_zero_sentinel``,
    ``citi_ask_zero_sentinel``/``citi_bid_zero_sentinel``) -- thousands of
    near-identical lines per run whenever a source is broadly returning
    stale/closing-price/after-hours data (e.g. Citi's ``"Closing Price"``
    rows, see the module docstring). The per-row detail is still logged, but
    only at DEBUG; :func:`_log_quote_anomaly_summary` emits one INFO-level
    summary line per :meth:`fetch_products` call instead.
    """

    total: int = 0
    ask_missing: int = 0
    bid_missing: int = 0
    example_isins: list[str] = field(default_factory=list)

    def note(self, isin: str | None, *, ask_missing: bool, bid_missing: bool) -> None:
        self.total += 1
        if ask_missing:
            self.ask_missing += 1
        if bid_missing:
            self.bid_missing += 1
        if (ask_missing or bid_missing) and isin and len(self.example_isins) < 5:
            self.example_isins.append(isin)


def _log_quote_anomaly_summary(event: str, stats: _QuoteAnomalyStats) -> None:
    """Emit one aggregated INFO line for a fetch_products() call, if needed.

    A no-op when nothing was anomalous (every row had a usable bid and ask)
    -- a healthy fetch stays silent at INFO level.
    """
    if stats.ask_missing == 0 and stats.bid_missing == 0:
        return
    logger.info(
        event,
        count_total=stats.total,
        count_ask_missing=stats.ask_missing,
        count_bid_missing=stats.bid_missing,
        example_isins=stats.example_isins,
    )


# -- BNP Paribas ----------------------------------------------------------------

_BNP_PARSER_VERSION = "bnp_paribas/1"
_BNP_SOURCE_NAME = "bnp_paribas"
_BNP_VENUE = "issuer_quote_bnp"
_BNP_ISSUER = "BNP Paribas"
_BNP_DEFAULT_BASE_URL = "https://derivate.bnpparibas.com/apiv2/api/v1"
_BNP_UNDERLYING_INDEXES_PATH = "/underlying/indexes"
_BNP_PRODUCTLIST_LEVERAGE_PATH = "/productlist/leverage"
# Default derivative-type filter covering turbo/mini/KO variants -- the exact
# numeric -> label mapping was never decoded from the bundle; each row
# self-describes via `derivativeTypeName` (e.g. "Turbo Long", "MINI Short").
_BNP_DERIVATIVE_TYPE_IDS: tuple[int, ...] = (7, 9, 23, 24, 238, 239, 580, 669, 670, 581)
# Confirmed live (this session's coordinator-requested pagination probe):
# limit up to (at least) 1000 is honored by the server unchanged. The DAX
# book currently totals ~4340 across all derivative-type variants, so 1000
# keeps full coverage within ~5 pages, well under `max_pages` from config.
_BNP_PAGE_SIZE = 1000

_BNP_DIRECTION_MAP: dict[str, Direction] = {"long": Direction.LONG, "short": Direction.SHORT}


def _bnp_headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Content-Type": "application/json",
        "clientid": "0",
        "countryid": "",
        "languageid": "de",
    }


class BnpParibasTurboAdapter:
    """BNP Paribas ``derivate.bnpparibas.com`` leverage-product feed adapter."""

    def __init__(
        self,
        http: HttpClient,
        *,
        base_url: str = _BNP_DEFAULT_BASE_URL,
        max_pages: int = 20,
        page_size: int = _BNP_PAGE_SIZE,
        stale_after_s: float = _DEFAULT_STALE_AFTER_S,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._max_pages = max_pages
        self._page_size = page_size
        self._stale_after_s = stale_after_s
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self.last_errors: list[IssuerRowError] = []
        self._partial_universe: dict[str, tuple[int, int]] = {}
        self._index_cache: dict[str, int] | None = None

    @property
    def name(self) -> str:
        return _BNP_SOURCE_NAME

    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=_BNP_SOURCE_NAME,
            kind="product",
            version=_BNP_PARSER_VERSION,
            homepage="https://derivate.bnpparibas.com/knockouts/",
        )

    # -- fetch/normalize (generic DataSourceAdapter contract) ---------------

    def fetch(self, **kwargs: Any) -> Any:
        """Resolve the underlying index list. All network I/O lives here or
        in :meth:`_fetch_all_pages`; this method exists to satisfy the
        generic ``DataSourceAdapter.fetch`` half of the contract."""
        del kwargs
        return self._resolve_underlying_index()

    def normalize(self, raw: Any, **kwargs: Any) -> Any:
        """Not used directly -- product normalization happens per-page in
        :meth:`fetch_products` (each page needs its own ``underlying_id``
        context). Present to satisfy ``DataSourceAdapter.normalize``."""
        del raw, kwargs
        return []

    # -- underlying index resolution -----------------------------------------

    def _resolve_underlying_index(self) -> dict[str, int]:
        """Fetch (once, cached) canonical underlying_id -> BNP instrumentId.

        Only underlyings BNP's ``/underlying/indexes`` endpoint actually
        lists (equity/index underlyings) can ever appear here -- FX/metals/
        energy underlyings are never guessed a mapping.
        """
        if self._index_cache is not None:
            return self._index_cache

        url = f"{self._base_url}{_BNP_UNDERLYING_INDEXES_PATH}"
        raw = self._http.get_json(url, headers=_bnp_headers(self._http.user_agent))
        if not isinstance(raw, dict) or "result" not in raw:
            raise AdapterError(f"unexpected BNP underlying/indexes response shape: {raw!r}")

        entries = raw["result"]
        if not isinstance(entries, list):
            raise AdapterError("BNP underlying/indexes 'result' is not a list")

        index_map: dict[str, int] = {}
        for entry in entries:
            name = entry.get("name")
            instrument_id = entry.get("instrumentId")
            if not isinstance(name, str) or not isinstance(instrument_id, int):
                continue
            canonical_id = resolve_underlying_id(name)
            if canonical_id is None:
                logger.debug("bnp_underlying_index_unmapped", raw_name=name)
                continue
            index_map[canonical_id] = instrument_id

        self._index_cache = index_map
        return index_map

    # -- product fetching -----------------------------------------------------

    def fetch_products(self, underlying_ids: Sequence[str]) -> list[ProductSnapshot]:
        self.last_errors = []
        self._partial_universe = {}
        now = self._clock()
        quote_stats = _QuoteAnomalyStats()

        index_map = self._resolve_underlying_index()

        snapshots: list[ProductSnapshot] = []
        for underlying_id in underlying_ids:
            instrument_id = index_map.get(underlying_id)
            if instrument_id is None:
                logger.warning("bnp_underlying_unresolved", underlying_id=underlying_id)
                continue

            raw_items, total, response_date = self._fetch_all_pages(instrument_id)
            covered = len(raw_items)
            if total is not None and covered < total:
                self._partial_universe[underlying_id] = (covered, total)
                logger.warning(
                    "bnp_partial_universe",
                    underlying_id=underlying_id,
                    covered=covered,
                    total=total,
                )

            snapshots.extend(
                self._normalize_products(
                    raw_items,
                    underlying_id=underlying_id,
                    now=now,
                    response_date=response_date,
                    quote_stats=quote_stats,
                )
            )
        _log_quote_anomaly_summary("bnp_quotes_summary", quote_stats)
        return snapshots

    def _fetch_all_pages(
        self, instrument_id: int
    ) -> tuple[list[dict[str, Any]], int | None, datetime | None]:
        url = f"{self._base_url}{_BNP_PRODUCTLIST_LEVERAGE_PATH}"
        headers = _bnp_headers(self._http.user_agent)

        collected: list[dict[str, Any]] = []
        total: int | None = None
        response_date: datetime | None = None

        for page_index in range(self._max_pages):
            offset = page_index * self._page_size
            body = {
                "clientId": 0,
                "languageId": "de",
                "countryId": "",
                "derivativeTypeIds": list(_BNP_DERIVATIVE_TYPE_IDS),
                "productSubGroupIds": None,
                "productGroupIds": None,
                # Must stay null: a guessed non-null default silently
                # returns total=0 for every query (no HTTP error).
                "productSetIds": None,
                "filterSelections": [
                    {
                        "fieldKey": "first.underlyingId",
                        "filterType": "DropDown",
                        "selectedValues": [str(instrument_id)],
                    }
                ],
                "offset": offset,
                "limit": self._page_size,
                "doNotTrimFilters": True,
                "sortPreference": [
                    {"sortField": "leverage", "sortDirection": "Ascending", "sortIndex": 0}
                ],
                "allowLeverageGrouping": False,
            }
            raw = self._http.post_json(url, json=body, headers=headers)
            if not isinstance(raw, dict) or "productListResponse" not in raw:
                raise AdapterError(f"unexpected BNP productlist/leverage response shape: {raw!r}")
            result = raw["productListResponse"]
            total = result.get("total")
            response_date = _parse_utc_z(result.get("responseDate")) or response_date
            page_items = result.get("result", [])
            if not isinstance(page_items, list):
                raise AdapterError("BNP productlist/leverage 'result' is not a list")

            collected.extend(page_items)

            if len(page_items) < self._page_size:
                break
            if total is not None and offset + len(page_items) >= total:
                break

        return collected, total, response_date

    def _normalize_products(
        self,
        raw_items: list[dict[str, Any]],
        *,
        underlying_id: str,
        now: datetime,
        response_date: datetime | None,
        quote_stats: _QuoteAnomalyStats,
    ) -> list[ProductSnapshot]:
        snapshots: list[ProductSnapshot] = []
        for raw_product in raw_items:
            isin_for_error = raw_product.get("isin")
            try:
                snapshot = self._build_snapshot(
                    raw_product,
                    underlying_id=underlying_id,
                    now=now,
                    response_date=response_date,
                    quote_stats=quote_stats,
                )
                snapshots.append(snapshot)
            except (KeyError, ValidationError, ValueError, TypeError) as exc:
                self.last_errors.append(
                    IssuerRowError(source=_BNP_SOURCE_NAME, isin=isin_for_error, error=str(exc))
                )
                logger.error("bnp_row_normalization_error", isin=isin_for_error, error=str(exc))
        return snapshots

    def _build_snapshot(
        self,
        p: dict[str, Any],
        *,
        underlying_id: str,
        now: datetime,
        response_date: datetime | None,
        quote_stats: _QuoteAnomalyStats,
    ) -> ProductSnapshot:
        isin = p["isin"]
        wkn = p.get("wkn")

        currency = (p.get("currency") or {}).get("isoCode") or "EUR"

        first = p.get("first") or {}
        underlying_raw = first.get("underlyingOfficialName") or first.get("underlyingISIN")
        if not underlying_raw:
            raise ValueError("missing first.underlyingOfficialName/underlyingISIN")

        financing_level = first.get("strikeAbsolute")
        if financing_level is None:
            raise ValueError("missing first.strikeAbsolute (financing_level)")
        knockout_barrier = first.get("knockOutAbsolute")
        if knockout_barrier is None:
            raise ValueError("missing first.knockOutAbsolute (knockout_barrier)")
        ratio = first.get("ratio")
        if ratio is None:
            raise ValueError("missing first.ratio")

        cfg = p.get("config") or {}
        direction_raw = cfg.get("derivativeDirectionName")
        direction = (
            _BNP_DIRECTION_MAP.get(direction_raw) if isinstance(direction_raw, str) else None
        )
        if direction is None:
            raise ValueError(f"unmapped config.derivativeDirectionName: {direction_raw!r}")

        key_figures = p.get("keyFigures") or {}
        maturity_ts = key_figures.get("maturityDateTimestamp")
        if maturity_ts is None:
            raise ValueError("missing keyFigures.maturityDateTimestamp")
        open_end = maturity_ts == -1
        # Dated-maturity timestamp unit/epoch format was never observed/
        # confirmed by research -- never guess-parsed (CLAUDE.md rule 29).
        maturity: date | None = None
        if not open_end:
            logger.warning(
                "bnp_maturity_timestamp_format_unconfirmed",
                isin=isin,
                maturity_timestamp=maturity_ts,
            )

        product_type = classify_product_type(
            financing_level=financing_level,
            knockout_barrier=knockout_barrier,
            open_end=open_end,
            maturity=maturity,
        )

        bid_raw = p.get("bid")
        bid, bid_was_zero = _resolve_zero_as_missing(
            float(bid_raw) if bid_raw is not None else None
        )
        if bid_was_zero:
            logger.debug("bnp_bid_zero_sentinel", isin=isin)

        ask_present = "ask" in p
        ask_raw = p.get("ask") if ask_present else None
        ask, ask_was_zero = _resolve_zero_as_missing(
            float(ask_raw) if ask_raw is not None else None
        )
        if not ask_present:
            logger.debug("bnp_ask_key_absent", isin=isin)
        elif ask_was_zero:
            logger.debug("bnp_ask_zero_sentinel", isin=isin)

        quote_stats.note(isin, ask_missing=ask is None, bid_missing=bid_was_zero)

        quote_presence = bid is not None and ask is not None

        bid_size = p.get("bidSize")
        ask_size = p.get("askSize") if ask_present else None

        bid_ts = _parse_berlin_naive_to_utc(p.get("bidDate"))
        ask_ts = _parse_berlin_naive_to_utc(p.get("askDate")) if ask_present else None
        candidates = [ts for ts in (bid_ts, ask_ts) if ts is not None]
        quote_timestamp = max(candidates) if candidates else None

        is_stale = (
            True
            if quote_timestamp is None
            else (now - quote_timestamp).total_seconds() > self._stale_after_s
        )
        quality_score = _quality_score(quote_presence=quote_presence, is_stale=is_stale)

        bid_only = bool(cfg.get("isBidOnly", False)) or (bid is not None and ask is None)
        knocked_out = bool(cfg.get("isKnockedOut", False))

        barrier_monitoring_time = first.get("barrierMonitoringTime")
        barrier_monitoring_tz = first.get("barrierMonitoringTimeZone")
        trading_hours = (
            f"{barrier_monitoring_time} {barrier_monitoring_tz}"
            if barrier_monitoring_time
            else None
        )

        underlying_price_ref = first.get("price")
        # Own, independent timestamp for `first.price` -- see module
        # docstring Pitfall 3. Deliberately NOT the product's own
        # bidDate/askDate-derived `quote_timestamp`: the two update at
        # different cadences, and conflating them was the BEFUND 1 bug.
        underlying_price_ref_timestamp = _parse_berlin_naive_to_utc(first.get("priceDate"))

        observation_time = quote_timestamp if quote_timestamp is not None else now
        source_timestamp = quote_timestamp if quote_timestamp is not None else response_date

        return ProductSnapshot(
            isin=isin,
            wkn=wkn,
            issuer=_BNP_ISSUER,
            venue=_BNP_VENUE,
            underlying_raw=underlying_raw,
            underlying_id=underlying_id,
            direction=direction,
            product_type=product_type,
            financing_level=float(financing_level),
            knockout_barrier=float(knockout_barrier),
            ratio=float(ratio),
            currency=currency,
            underlying_currency=None,
            quanto=None,
            open_end=open_end,
            maturity=maturity,
            first_trading_day=None,
            bid=bid,
            ask=ask,
            bid_size=float(bid_size) if bid_size is not None else None,
            ask_size=float(ask_size) if ask_size is not None else None,
            quote_timestamp=quote_timestamp,
            quote_presence=quote_presence,
            bid_only=bid_only,
            knocked_out=knocked_out,
            trading_hours=trading_hours,
            product_age_days=None,
            underlying_price_ref=(
                float(underlying_price_ref) if underlying_price_ref is not None else None
            ),
            underlying_price_ref_timestamp=underlying_price_ref_timestamp,
            raw_hash=_raw_hash(p),
            observation_time=observation_time,
            available_at=now,
            retrieved_at=now,
            source_timestamp=source_timestamp,
            source=_BNP_SOURCE_NAME,
            parser_version=_BNP_PARSER_VERSION,
            is_stale=is_stale,
            quality_score=quality_score,
        )

    # -- healthcheck ------------------------------------------------------------

    def healthcheck(self) -> HealthCheckResult:
        checked_at = datetime.now(UTC)
        start = time.monotonic()
        try:
            url = f"{self._base_url}{_BNP_UNDERLYING_INDEXES_PATH}"
            raw = self._http.get_json(url, headers=_bnp_headers(self._http.user_agent))
        except AdapterHttpError as exc:
            return HealthCheckResult(
                source=_BNP_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=None,
                checked_at=checked_at,
                message=f"HTTP error: {exc}",
            )
        except Exception as exc:  # defensive: healthcheck must never raise
            return HealthCheckResult(
                source=_BNP_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=None,
                checked_at=checked_at,
                message=f"unexpected error: {exc}",
            )
        latency_ms = (time.monotonic() - start) * 1000

        if not isinstance(raw, dict) or "result" not in raw or not isinstance(raw["result"], list):
            return HealthCheckResult(
                source=_BNP_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=latency_ms,
                checked_at=checked_at,
                message="unexpected underlying/indexes response schema",
            )
        entries = raw["result"]
        if not entries:
            return HealthCheckResult(
                source=_BNP_SOURCE_NAME,
                status=HealthStatus.WARN,
                ok=True,
                latency_ms=latency_ms,
                checked_at=checked_at,
                message="underlying/indexes returned an empty list",
            )
        required_fields = {"instrumentId", "name", "isin"}
        for entry in entries:
            if not isinstance(entry, dict) or not required_fields.issubset(entry.keys()):
                return HealthCheckResult(
                    source=_BNP_SOURCE_NAME,
                    status=HealthStatus.FAIL,
                    ok=False,
                    latency_ms=latency_ms,
                    checked_at=checked_at,
                    message=f"underlying/indexes entry missing required fields: {entry!r}",
                )

        if self._partial_universe:
            summary = ", ".join(
                f"{uid}: {covered}/{total}"
                for uid, (covered, total) in self._partial_universe.items()
            )
            return HealthCheckResult(
                source=_BNP_SOURCE_NAME,
                status=HealthStatus.WARN,
                ok=True,
                latency_ms=latency_ms,
                checked_at=checked_at,
                message=f"partial_universe (from last fetch_products): {summary}",
            )

        return HealthCheckResult(
            source=_BNP_SOURCE_NAME,
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=latency_ms,
            checked_at=checked_at,
            message=f"{len(entries)} underlying indexes resolved",
        )


def _bnp_factory(src: SourceConfig, http: HttpClient) -> BnpParibasTurboAdapter:
    # configs/sources.yaml's base_url is the site root; the confirmed-working
    # API base additionally requires the /apiv2/api/v1 prefix (see module
    # docstring / docs/data_sources.md for how that was resolved).
    api_base_url = f"{src.base_url.rstrip('/')}/apiv2/api/v1"
    return BnpParibasTurboAdapter(http, base_url=api_base_url, max_pages=src.max_pages)


# -- Citi / CitiFirst -------------------------------------------------------------

_CITI_PARSER_VERSION = "citi/1"
_CITI_SOURCE_NAME = "citi"
_CITI_VENUE = "issuer_quote_citi"
_CITI_ISSUER = "Citigroup"
_CITI_DEFAULT_BASE_URL = "https://de.citifirst.com/citi/v1/theq/api"
_CITI_SEARCH_PATH = "/ProductSearch/de-DE/Search"

# Only underlyings with an independently research-verified ISIN. Per
# CLAUDE.md rule 29, an underlying enabled in configs/universe.yaml but not
# present here (NDX, EURUSD, XAU at the time of writing -- FX/metals are
# likely not ISIN-addressable in this API at all, never researched) is
# skipped cleanly with a log line rather than guessed.
_CITI_UNDERLYING_ISINS: dict[str, str] = {
    "DAX": "DE0008469008",
}

# `referencePriceMethod` values that mean "this bid/ask is a closing/
# reference-price snapshot, not a live tradable quote" -- see module
# docstring for the disambiguating market-hours pull (2026-09-11, ~08:51 UTC,
# 25/25 DAX rows: ask == 0.0 with referencePriceMethod == "Closing Price").
# Currently the only value ever observed, across every fixture and both
# research-session live pulls plus this disambiguation pull -- deliberately
# a closed allowlist (not e.g. "contains 'Closing'") per CLAUDE.md rule 29:
# an unrecognized future value falls through to the plain zero-sentinel
# handling and is surfaced via the per-row DEBUG log / `_QuoteAnomalyStats`
# summary rather than silently guessed either way.
_CITI_NON_LIVE_REFERENCE_PRICE_METHODS: frozenset[str] = frozenset({"Closing Price"})

_CITI_DIRECTION_MAP: dict[str, Direction] = {
    "Bull": Direction.LONG,
    "Long": Direction.LONG,
    "Bear": Direction.SHORT,
    "Short": Direction.SHORT,
}


def _citi_headers(user_agent: str) -> dict[str, str]:
    return {"User-Agent": user_agent, "Content-Type": "application/json"}


class CitiFirstTurboAdapter:
    """Citi (CitiFirst) ``de.citifirst.com`` TheQ product-search adapter."""

    def __init__(
        self,
        http: HttpClient,
        *,
        base_url: str = _CITI_DEFAULT_BASE_URL,
        max_pages: int = 10,
        stale_after_s: float = _DEFAULT_STALE_AFTER_S,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._max_pages = max_pages  # respected structurally; see module docstring
        self._stale_after_s = stale_after_s
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self.last_errors: list[IssuerRowError] = []
        self._partial_universe: dict[str, tuple[int, int]] = {}

    @property
    def name(self) -> str:
        return _CITI_SOURCE_NAME

    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=_CITI_SOURCE_NAME,
            kind="product",
            version=_CITI_PARSER_VERSION,
            homepage="https://de.citifirst.com/open-end-turbos/",
        )

    # -- fetch/normalize (generic DataSourceAdapter contract) ------------------

    def fetch(self, **kwargs: Any) -> Any:
        isin = kwargs.get("isin")
        if not isinstance(isin, str):
            raise AdapterError("CitiFirstTurboAdapter.fetch requires isin=<underlying ISIN>")
        return self._search(isin)

    def normalize(self, raw: Any, **kwargs: Any) -> Any:
        del kwargs
        if not isinstance(raw, dict):
            return []
        return raw.get("items", [])

    # -- product fetching ---------------------------------------------------

    def _search(self, isin: str) -> dict[str, Any]:
        url = f"{self._base_url}{_CITI_SEARCH_PATH}"
        body = {"underlyingIsins": [isin]}
        raw = self._http.post_json(url, json=body, headers=_citi_headers(self._http.user_agent))
        if not isinstance(raw, dict) or "items" not in raw:
            raise AdapterError(f"unexpected Citi ProductSearch response shape: {raw!r}")
        return raw

    def fetch_products(self, underlying_ids: Sequence[str]) -> list[ProductSnapshot]:
        self.last_errors = []
        self._partial_universe = {}
        now = self._clock()
        quote_stats = _QuoteAnomalyStats()

        snapshots: list[ProductSnapshot] = []
        for underlying_id in underlying_ids:
            isin = _CITI_UNDERLYING_ISINS.get(underlying_id)
            if isin is None:
                logger.warning("citi_underlying_unresolved", underlying_id=underlying_id)
                continue

            data = self._search(isin)
            total = data.get("totalElementsCount")
            items = data.get("items", [])
            items_count = data.get("itemsCount", len(items))

            matched_items: list[dict[str, Any]] = []
            for item in items:
                underlyings = item.get("underlyings") or []
                if any(u.get("isin") == isin for u in underlyings):
                    matched_items.append(item)
                else:
                    # Defends against the documented "silent wrong result"
                    # pitfall: a body-shape bug here returns HTTP 200 with
                    # an unrelated default result set rather than erroring.
                    self.last_errors.append(
                        IssuerRowError(
                            source=_CITI_SOURCE_NAME,
                            isin=item.get("isin"),
                            error=(
                                f"row's underlyings do not include requested "
                                f"underlyingIsins={isin!r} (possible silent-wrong-result)"
                            ),
                        )
                    )
                    logger.error(
                        "citi_underlying_mismatch", requested_isin=isin, item_isin=item.get("isin")
                    )

            if total is not None and total > items_count:
                self._partial_universe[underlying_id] = (len(matched_items), total)
                logger.warning(
                    "citi_partial_universe",
                    underlying_id=underlying_id,
                    covered=len(matched_items),
                    total=total,
                )

            snapshots.extend(
                self._normalize_products(
                    matched_items, underlying_id=underlying_id, now=now, quote_stats=quote_stats
                )
            )
        _log_quote_anomaly_summary("citi_quotes_summary", quote_stats)
        return snapshots

    def _normalize_products(
        self,
        raw_items: list[dict[str, Any]],
        *,
        underlying_id: str,
        now: datetime,
        quote_stats: _QuoteAnomalyStats,
    ) -> list[ProductSnapshot]:
        snapshots: list[ProductSnapshot] = []
        for raw_product in raw_items:
            isin_for_error = raw_product.get("isin")
            try:
                snapshot = self._build_snapshot(
                    raw_product, underlying_id=underlying_id, now=now, quote_stats=quote_stats
                )
                snapshots.append(snapshot)
            except (KeyError, ValidationError, ValueError, TypeError) as exc:
                self.last_errors.append(
                    IssuerRowError(source=_CITI_SOURCE_NAME, isin=isin_for_error, error=str(exc))
                )
                logger.error("citi_row_normalization_error", isin=isin_for_error, error=str(exc))
        return snapshots

    def _build_snapshot(
        self,
        it: dict[str, Any],
        *,
        underlying_id: str,
        now: datetime,
        quote_stats: _QuoteAnomalyStats,
    ) -> ProductSnapshot:
        isin = it["isin"]
        wkn = it.get("wkn")

        currency = (
            it.get("currencyCode")
            or ((it.get("price") or {}).get("bid") or {}).get("currencyCode")
            or "EUR"
        )

        underlyings = it.get("underlyings") or [{}]
        underlying0 = underlyings[0] if underlyings else {}
        underlying_raw = underlying0.get("name") or underlying0.get("isin")
        if not underlying_raw:
            raise ValueError("missing underlyings[0].name/isin")

        strike = (it.get("strike") or {}).get("amount")
        if strike is None:
            raise ValueError("missing strike.amount (financing_level)")
        ko_barrier = (it.get("koBarrier") or {}).get("amount")
        if ko_barrier is None:
            raise ValueError("missing koBarrier.amount (knockout_barrier)")
        ratio = it.get("ratio")
        if ratio is None:
            raise ValueError("missing ratio")

        sub_type = it.get("subTypeTranslation")
        direction = _CITI_DIRECTION_MAP.get(sub_type) if isinstance(sub_type, str) else None
        if direction is None:
            raise ValueError(f"unmapped subTypeTranslation: {sub_type!r}")

        maturity_raw = it.get("maturityDate")
        maturity: date | None = None
        if maturity_raw:
            # Observed as a plain ISO date, sometimes with a midnight
            # time-of-day component (matches issueDate's shape in the same
            # payload) -- take just the date portion in either case.
            try:
                maturity = date.fromisoformat(maturity_raw[:10])
            except ValueError as exc:
                raise ValueError(f"unparseable maturityDate: {maturity_raw!r}") from exc
        open_end = maturity is None

        product_type = classify_product_type(
            financing_level=strike,
            knockout_barrier=ko_barrier,
            open_end=open_end,
            maturity=maturity,
        )

        price = it.get("price") or {}
        reference_price_method = it.get("referencePriceMethod")
        is_non_live_reference = reference_price_method in _CITI_NON_LIVE_REFERENCE_PRICE_METHODS

        if is_non_live_reference:
            # referencePriceMethod says this is a closing/reference-price
            # snapshot, not a live two-way market -- see module docstring
            # and _CITI_NON_LIVE_REFERENCE_PRICE_METHODS. bid/ask/sizes are
            # discarded regardless of their numeric value (a nonzero closing
            # price is just as unusable as a live ask as a zero one), and
            # the row is forced stale: it is never a fresh tradable quote no
            # matter how recent price.timeStamp is. Master data (financing
            # level, knockout barrier, ratio, ISIN, ...) is untouched.
            bid = None
            ask = None
            bid_was_zero = False
            ask_was_zero = False
            bid_size = None
            ask_size = None
            logger.debug(
                "citi_non_live_reference_price",
                isin=isin,
                reference_price_method=reference_price_method,
            )
        else:
            bid_amount = (price.get("bid") or {}).get("amount")
            bid, bid_was_zero = _resolve_zero_as_missing(
                float(bid_amount) if bid_amount is not None else None
            )
            if bid_was_zero:
                logger.debug(
                    "citi_bid_zero_sentinel",
                    isin=isin,
                    reference_price_method=reference_price_method,
                )

            ask_amount = (price.get("ask") or {}).get("amount")
            ask, ask_was_zero = _resolve_zero_as_missing(
                float(ask_amount) if ask_amount is not None else None
            )
            if ask_was_zero:
                logger.debug(
                    "citi_ask_zero_sentinel",
                    isin=isin,
                    reference_price_method=reference_price_method,
                )

            bid_size = price.get("bidSize")
            ask_size = price.get("askSize")

        quote_stats.note(
            isin,
            ask_missing=ask is None,
            bid_missing=bid is None,
        )

        quote_presence = bid is not None and ask is not None

        quote_timestamp = _parse_berlin_naive_to_utc(price.get("timeStamp"))

        is_stale = (
            True
            if quote_timestamp is None or is_non_live_reference
            else (now - quote_timestamp).total_seconds() > self._stale_after_s
        )
        quality_score = _quality_score(quote_presence=quote_presence, is_stale=is_stale)

        bid_only = bid is not None and ask is None
        # Citi has no dedicated "isKnockedOut" config flag like BNP does;
        # `barrierBreached` is the closest available field for the same
        # concept (observed `False` on every sampled row).
        knocked_out = bool(it.get("barrierBreached", False))

        trading_times = (underlying0.get("tradingTimes") or {}).get("productTime") or {}
        from_time = trading_times.get("from")
        to_time = trading_times.get("to")
        trading_hours = f"{from_time}-{to_time}" if from_time and to_time else None

        observation_time = quote_timestamp if quote_timestamp is not None else now
        source_timestamp = quote_timestamp

        return ProductSnapshot(
            isin=isin,
            wkn=wkn,
            issuer=_CITI_ISSUER,
            venue=_CITI_VENUE,
            underlying_raw=underlying_raw,
            underlying_id=underlying_id,
            direction=direction,
            product_type=product_type,
            financing_level=float(strike),
            knockout_barrier=float(ko_barrier),
            ratio=float(ratio),
            currency=currency,
            underlying_currency=None,
            quanto=it.get("isQuanto"),
            open_end=open_end,
            maturity=maturity,
            first_trading_day=None,
            bid=bid,
            ask=ask,
            bid_size=float(bid_size) if bid_size is not None else None,
            ask_size=float(ask_size) if ask_size is not None else None,
            quote_timestamp=quote_timestamp,
            quote_presence=quote_presence,
            bid_only=bid_only,
            knocked_out=knocked_out,
            trading_hours=trading_hours,
            product_age_days=None,
            underlying_price_ref=None,
            underlying_price_ref_timestamp=None,
            raw_hash=_raw_hash(it),
            observation_time=observation_time,
            available_at=now,
            retrieved_at=now,
            source_timestamp=source_timestamp,
            source=_CITI_SOURCE_NAME,
            parser_version=_CITI_PARSER_VERSION,
            is_stale=is_stale,
            quality_score=quality_score,
        )

    # -- healthcheck ------------------------------------------------------------

    def healthcheck(self) -> HealthCheckResult:
        checked_at = datetime.now(UTC)
        probe_underlying_id = "DAX"
        isin = _CITI_UNDERLYING_ISINS.get(probe_underlying_id)
        if isin is None:
            return HealthCheckResult(
                source=_CITI_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=None,
                checked_at=checked_at,
                message="no verified underlying ISIN configured for healthcheck probe",
            )

        start = time.monotonic()
        try:
            data = self._search(isin)
        except AdapterHttpError as exc:
            return HealthCheckResult(
                source=_CITI_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=None,
                checked_at=checked_at,
                message=f"HTTP error: {exc}",
            )
        except Exception as exc:  # defensive: healthcheck must never raise
            return HealthCheckResult(
                source=_CITI_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=None,
                checked_at=checked_at,
                message=f"unexpected error: {exc}",
            )
        latency_ms = (time.monotonic() - start) * 1000

        items = data.get("items", [])
        total = data.get("totalElementsCount")
        items_count = data.get("itemsCount", len(items))

        if not items:
            return HealthCheckResult(
                source=_CITI_SOURCE_NAME,
                status=HealthStatus.WARN,
                ok=True,
                latency_ms=latency_ms,
                checked_at=checked_at,
                message=f"ProductSearch returned no items for {probe_underlying_id}",
            )

        required_fields = {"isin", "wkn", "price"}
        for item in items:
            if not isinstance(item, dict) or not required_fields.issubset(item.keys()):
                return HealthCheckResult(
                    source=_CITI_SOURCE_NAME,
                    status=HealthStatus.FAIL,
                    ok=False,
                    latency_ms=latency_ms,
                    checked_at=checked_at,
                    message=f"ProductSearch item missing required fields: {item!r}",
                )

        if total is not None and total > items_count:
            return HealthCheckResult(
                source=_CITI_SOURCE_NAME,
                status=HealthStatus.WARN,
                ok=True,
                latency_ms=latency_ms,
                checked_at=checked_at,
                message=(
                    f"partial_universe: {items_count}/{total} rows for "
                    f"{probe_underlying_id} (Citi's 25-row-per-call cap is structural, "
                    "not resolved)"
                ),
            )

        return HealthCheckResult(
            source=_CITI_SOURCE_NAME,
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=latency_ms,
            checked_at=checked_at,
            message=f"{len(items)} rows returned for {probe_underlying_id}",
        )


def _citi_factory(src: SourceConfig, http: HttpClient) -> CitiFirstTurboAdapter:
    # configs/sources.yaml's base_url is the site root; the confirmed-working
    # API base additionally requires the /citi/v1/theq/api prefix (see module
    # docstring / docs/data_sources.md for how that was resolved).
    api_base_url = f"{src.base_url.rstrip('/')}/citi/v1/theq/api"
    return CitiFirstTurboAdapter(http, base_url=api_base_url, max_pages=src.max_pages)


# -- idempotent registration --------------------------------------------------


def _register() -> None:
    from turboedge.adapters.registry import PRODUCT_ADAPTER_FACTORIES

    if _BNP_SOURCE_NAME not in PRODUCT_ADAPTER_FACTORIES:
        from turboedge.adapters.registry import register_product_adapter

        register_product_adapter(_BNP_SOURCE_NAME, _bnp_factory)
    if _CITI_SOURCE_NAME not in PRODUCT_ADAPTER_FACTORIES:
        from turboedge.adapters.registry import register_product_adapter

        register_product_adapter(_CITI_SOURCE_NAME, _citi_factory)


_register()
