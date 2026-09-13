"""CSV import adapter for user-provided product lists.

Allows users to import manually-downloaded product lists from exchange websites
(Deutsche Börse, Börse Stuttgart, broker exports) in CSV format.

Format support:
- Delimiters: auto-detected (semicolon or comma) via csv.Sniffer
- Encoding: UTF-8, with or without BOM
- Number formats: German (1.234,56) and English (1234.56) auto-detected
- Timestamps: ISO-8601; naive datetimes interpreted as Europe/Berlin -> UTC

Required columns: isin, issuer, underlying, direction, financing_level,
knockout_barrier, ratio, bid, ask, quote_timestamp.

Optional columns: wkn, product_type, currency, underlying_currency, quanto,
open_end, maturity, first_trading_day, bid_size, ask_size, bid_only,
knocked_out, underlying_price_ref, venue.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog
from pydantic import ValidationError

from turboedge.adapters.base import AdapterMetadata, HealthCheckResult, ProductFetchContext
from turboedge.config import SourceConfig
from turboedge.storage.schemas import (
    Direction,
    HealthStatus,
    ProductSnapshot,
)
from turboedge.universe.classify import classify_direction, classify_product_type
from turboedge.universe.underlying_map import resolve_underlying_id

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RowError:
    """Error record for a malformed CSV row."""

    file: str
    line: int
    isin: str | None
    error: str


_BOOLEAN_TRUE = frozenset(("true", "1", "yes", "ja", "wahr"))
_BOOLEAN_FALSE = frozenset(("false", "0", "no", "nein", "falsch"))

# Ratio patterns: "0,01", "100:1", "10 : 1", etc.
_RATIO_COLON_RE = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*:\s*(\d+(?:[.,]\d+)?)\s*$")


def _parse_boolean(value: str | None) -> bool | None:
    """Parse a boolean from string, returning None if not a boolean value."""
    if value is None or value.strip() == "":
        return None
    normalized = value.strip().lower()
    if normalized in _BOOLEAN_TRUE:
        return True
    if normalized in _BOOLEAN_FALSE:
        return False
    return None


def _parse_number(value: str | None) -> float | None:
    """Parse German or English number format.

    Heuristic: if the value contains both comma and period, the rightmost
    separator is the decimal separator. If only one separator, use it as
    decimal if it's the last character or separated from the end by at most
    3 digits (German style: 1234,56 or 1.234,56). Otherwise treat as thousands
    separator.
    """
    if value is None or value.strip() == "":
        return None

    clean = value.strip()

    # Remove spaces
    clean = clean.replace(" ", "")

    comma_idx = clean.rfind(",")
    period_idx = clean.rfind(".")

    if comma_idx == -1 and period_idx == -1:
        try:
            return float(clean)
        except ValueError:
            return None

    # Both present: rightmost is decimal
    if comma_idx > -1 and period_idx > -1:
        decimal_idx = max(comma_idx, period_idx)
        if decimal_idx == comma_idx:
            # Rightmost is comma: comma is decimal
            clean = clean.replace(".", "").replace(",", ".")
        else:
            # Rightmost is period: period is decimal
            clean = clean.replace(",", "")
    elif comma_idx > -1:
        # Only comma: is it thousands or decimal?
        # If fewer than 4 digits after, it's likely decimal (German)
        rest = clean[comma_idx + 1 :]
        if len(rest) <= 3 and rest.isdigit():
            clean = clean.replace(".", "").replace(",", ".")
        else:
            clean = clean.replace(",", "")
    else:
        # Only period: is it thousands or decimal?
        rest = clean[period_idx + 1 :]
        if len(rest) <= 3 and rest.isdigit():
            # Decimal (English)
            pass
        else:
            # Thousands separator
            clean = clean.replace(".", "")

    try:
        return float(clean)
    except ValueError:
        return None


def _parse_ratio(value: str | None) -> float | None:
    """Parse ratio in forms: 0,01, 0.01, 100:1, 10:1, etc.

    Converts X:Y to Y/X (so 100:1 becomes 1/100 = 0.01).
    In financial context, X:Y (e.g., 100:1) means X units of product
    per Y units of underlying, so the ratio is Y/X.
    """
    if value is None or value.strip() == "":
        return None

    clean = value.strip()

    # Check for X:Y format
    match = _RATIO_COLON_RE.match(clean)
    if match:
        numerator_str = match.group(1).replace(",", ".")
        denominator_str = match.group(2).replace(",", ".")
        try:
            numerator = float(numerator_str)
            denominator = float(denominator_str)
            if numerator == 0:
                return None
            return denominator / numerator
        except ValueError:
            return None

    # Plain decimal
    return _parse_number(clean)


def _berlin_tz() -> ZoneInfo | None:
    """Return the Europe/Berlin IANA zone, or ``None`` if tzdata isn't loadable.

    Deliberately never falls back to a guessed fixed offset (e.g. "assume
    UTC+2") -- CLAUDE.md rule 29 forbids silently imputing pricing-critical
    data, and a fixed-offset guess is wrong for roughly half the year
    (CET vs. CEST). Callers must treat ``None`` as a hard parse failure.
    """
    try:
        return ZoneInfo("Europe/Berlin")
    except ZoneInfoNotFoundError:
        logger.error("zoneinfo_load_failed", tz="Europe/Berlin")
        return None


def _parse_timestamp(value: str | None) -> datetime | None:
    """Parse ISO-8601 timestamp, interpreting naive times as Europe/Berlin -> UTC.

    Naive (timezone-unaware) values are always localized via the real
    Europe/Berlin tzdata (correctly resolving CET/CEST per date, including
    DST transition days) before conversion to UTC. If the zone cannot be
    loaded, the value is treated as unparseable rather than guessed.
    """
    if value is None or value.strip() == "":
        return None

    clean = value.strip()

    try:
        dt = datetime.fromisoformat(clean)
    except (ValueError, TypeError):
        return None

    if dt.tzinfo is None:
        berlin_tz = _berlin_tz()
        if berlin_tz is None:
            return None
        dt = dt.replace(tzinfo=berlin_tz)

    return dt.astimezone(UTC)


def _parse_date(value: str | None) -> date | None:
    """Parse ISO-8601 date."""
    if value is None or value.strip() == "":
        return None

    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _calculate_raw_hash(row_dict: dict[str, str]) -> str:
    """SHA256 hash of the canonical JSON representation of a raw row."""
    canonical = json.dumps(row_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CsvProductImportAdapter:
    """Adapter for user-provided CSV product lists.

    Implements the ``ProductSourceAdapter`` (and generic ``DataSourceAdapter``)
    protocols structurally, like every other adapter in this codebase --
    deliberately not inheriting from the Protocol class itself, since these
    contracts are duck-typed (`typing.Protocol` + `@runtime_checkable`) and
    explicit inheritance is neither required nor idiomatic here.
    """

    def __init__(
        self,
        import_dir: Path,
        *,
        stale_after_s: float = 900.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._import_dir = Path(import_dir)
        self._stale_after_s = stale_after_s
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self.last_errors: list[RowError] = []
        self.last_file_errors: list[str] = []

    @property
    def name(self) -> str:
        return "csv_import"

    def fetch(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Read every CSV file in the import directory into raw parsed row dicts.

        This is the ``DataSourceAdapter.fetch`` half of the contract: all
        I/O (here, filesystem reads) happens exclusively in this method.
        Malformed rows never raise -- they are skipped and recorded in
        ``self.last_errors`` (reset at the start of this call) so one bad
        row never aborts ingestion of the rest of a file.

        Each returned dict carries its provenance under the
        ``_source_file`` / ``_source_line`` / ``_source_mtime`` keys, which
        ``normalize()`` uses to attribute any downstream snapshot-
        construction failure back to its originating CSV row.
        """
        del kwargs  # no fetch-time filtering; fetch_products() filters after normalize()
        self.last_errors = []
        self.last_file_errors = []

        if not self._import_dir.exists():
            return []

        csv_files = sorted(self._import_dir.glob("*.csv"))
        if not csv_files:
            return []

        raw_rows: list[dict[str, Any]] = []

        for csv_path in csv_files:
            try:
                mtime_dt: datetime | None = datetime.fromtimestamp(csv_path.stat().st_mtime, tz=UTC)
            except OSError:
                mtime_dt = None

            try:
                with open(csv_path, encoding="utf-8-sig") as f:
                    # Detect delimiter
                    sample = f.read(8192)
                    f.seek(0)

                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
                    except csv.Error:
                        # Fallback to semicolon
                        dialect = csv.excel
                        dialect.delimiter = ";"

                    reader = csv.DictReader(f, dialect=dialect)
                    if reader.fieldnames is None:
                        continue

                    fieldnames_lower = {fn.lower().strip(): fn for fn in reader.fieldnames}

                    for line_num, row in enumerate(reader, start=2):  # start=2 (header is line 1)
                        # Normalize row keys to lowercase
                        row_lower = {k.lower().strip(): v for k, v in row.items()}

                        try:
                            raw_row = self._parse_row(row_lower, fieldnames_lower)
                            raw_row["_source_file"] = csv_path.name
                            raw_row["_source_line"] = line_num
                            raw_row["_source_mtime"] = mtime_dt
                            raw_rows.append(raw_row)
                        except ValueError as exc:
                            isin_raw = row_lower.get("isin")
                            error_rec = RowError(
                                file=csv_path.name,
                                line=line_num,
                                isin=isin_raw,
                                error=str(exc),
                            )
                            self.last_errors.append(error_rec)
                            logger.warning(
                                "csv_row_error",
                                file=csv_path.name,
                                line=line_num,
                                error=str(exc),
                            )

            except (OSError, UnicodeDecodeError) as exc:
                # A file that cannot even be opened/decoded is a genuinely
                # broken source (as opposed to "no CSV files present" --
                # see healthcheck()), so it is tracked separately from
                # per-row parse errors and surfaced there as FAIL.
                self.last_file_errors.append(f"{csv_path.name}: {exc}")
                logger.error("csv_file_read_error", file=csv_path.name, error=str(exc))

        return raw_rows

    def fetch_products(
        self,
        underlying_ids: Sequence[str],
        *,
        context: ProductFetchContext | None = None,
    ) -> list[ProductSnapshot]:
        """Fetch, validate and normalize CSV rows into ``ProductSnapshot`` objects.

        Implements the ``ProductSourceAdapter`` contract exactly: the scan
        pipeline calls only this method and expects finished, validated
        ``ProductSnapshot`` instances back -- never raw dicts. Internally
        this is ``normalize(fetch())``, filtered to ``underlying_ids``
        (already resolved per row via ``resolve_underlying_id`` while
        parsing). An empty ``underlying_ids`` sequence means "no filter",
        matching how ``healthcheck()`` uses this method to inspect the
        whole import directory regardless of universe.

        ``context`` (Befund 2's optional per-run cross-check data) is
        accepted for ``ProductSourceAdapter`` contract compliance but unused:
        a manually-curated CSV import has no external reference to sanity
        check against and is trusted as entered.
        """
        del context
        raw_rows = self.fetch()
        snapshots = self.normalize(raw_rows)

        if not underlying_ids:
            return snapshots

        wanted = set(underlying_ids)
        return [snapshot for snapshot in snapshots if snapshot.underlying_id in wanted]

    def _parse_row(
        self, row_lower: dict[str, str], fieldnames_lower: dict[str, str]
    ) -> dict[str, Any]:
        """Parse a single CSV row into a typed dict.

        Raises ValueError if required fields are missing or invalid. Every
        pricing-critical field -- ``ratio``, ``bid``, ``ask``,
        ``financing_level``, ``knockout_barrier`` -- must be present *and*
        parseable; CLAUDE.md rule 29 forbids silently imputing missing
        pricing data, so none of these is ever left as a guessed/default
        value.
        """
        required_fields = {
            "isin",
            "issuer",
            "underlying",
            "ratio",
            "bid",
            "ask",
            "financing_level",
            "knockout_barrier",
            "quote_timestamp",
        }

        missing = required_fields - set(row_lower.keys())
        if missing:
            raise ValueError(f"missing required fields: {', '.join(sorted(missing))}")

        isin_raw = row_lower["isin"].strip().upper()
        if not isin_raw or len(isin_raw) != 12:
            raise ValueError(f"invalid ISIN: {isin_raw!r}")

        issuer = row_lower["issuer"].strip()
        if not issuer:
            raise ValueError("issuer cannot be empty")

        underlying_raw = row_lower["underlying"].strip()
        if not underlying_raw:
            raise ValueError("underlying cannot be empty")

        underlying_id = resolve_underlying_id(underlying_raw)

        # Direction
        direction_raw = row_lower.get("direction", "").strip()
        direction: Direction | None = None
        if direction_raw:
            direction = classify_direction(direction_raw)
        if direction is None and "direction" in row_lower:
            raise ValueError(f"could not classify direction from: {direction_raw!r}")

        # Financing level and knockout barrier are pricing-critical: required
        # and never imputed (rule 29). Both are always present as CSV
        # columns for a knock-out product; an unparseable value is a row
        # error, not a silent None.
        financing_level_str = row_lower["financing_level"].strip()
        financing_level = _parse_number(financing_level_str)
        if financing_level is None:
            raise ValueError(f"invalid financing_level: {financing_level_str!r}")

        knockout_barrier_str = row_lower["knockout_barrier"].strip()
        knockout_barrier = _parse_number(knockout_barrier_str)
        if knockout_barrier is None:
            raise ValueError(f"invalid knockout_barrier: {knockout_barrier_str!r}")

        # Ratio
        ratio_str = row_lower["ratio"].strip()
        ratio = _parse_ratio(ratio_str)
        if ratio is None or ratio <= 0:
            raise ValueError(f"invalid ratio: {ratio_str!r}")

        # Prices: bid and ask are both pricing-critical and independently
        # required (rules 11/12 -- entry is always ask, exit always bid, and
        # the spread between them must always be accounted for), so neither
        # is ever imputed from the other or left silently missing.
        bid_str = row_lower["bid"].strip()
        bid = _parse_number(bid_str)
        if bid is None:
            raise ValueError(f"invalid bid: {bid_str!r}")

        ask_str = row_lower["ask"].strip()
        ask = _parse_number(ask_str)
        if ask is None:
            raise ValueError(f"invalid ask: {ask_str!r}")

        # Quote timestamp
        quote_timestamp_str = row_lower["quote_timestamp"].strip()
        quote_timestamp = _parse_timestamp(quote_timestamp_str)
        if quote_timestamp is None:
            raise ValueError(f"invalid quote_timestamp: {quote_timestamp_str!r}")

        # Optional fields
        wkn = row_lower.get("wkn", "").strip() or None
        venue = row_lower.get("venue", "").strip() or "unknown"
        currency = row_lower.get("currency", "EUR").strip() or "EUR"
        underlying_currency = row_lower.get("underlying_currency", "").strip() or None

        # Booleans
        quanto_str = row_lower.get("quanto", "").strip()
        quanto = _parse_boolean(quanto_str)

        open_end_str = row_lower.get("open_end", "").strip()
        open_end = _parse_boolean(open_end_str)
        # If not specified, default to True (open-end is the typical variant)
        if open_end is None:
            open_end = True

        bid_only_str = row_lower.get("bid_only", "").strip()
        bid_only = _parse_boolean(bid_only_str) or False

        knocked_out_str = row_lower.get("knocked_out", "").strip()
        knocked_out = _parse_boolean(knocked_out_str) or False

        # Dates
        maturity_str = row_lower.get("maturity", "").strip()
        maturity = _parse_date(maturity_str) if maturity_str else None

        first_trading_day_str = row_lower.get("first_trading_day", "").strip()
        first_trading_day = _parse_date(first_trading_day_str) if first_trading_day_str else None

        # Numbers
        bid_size = _parse_number(row_lower.get("bid_size", "").strip())
        ask_size = _parse_number(row_lower.get("ask_size", "").strip())
        underlying_price_ref = _parse_number(row_lower.get("underlying_price_ref", "").strip())

        # Product type
        product_type_raw = row_lower.get("product_type", "").strip()
        product_type = classify_product_type(
            type_text=product_type_raw,
            financing_level=financing_level,
            knockout_barrier=knockout_barrier,
            open_end=open_end,
            maturity=maturity,
        )

        # Build the raw row dict for hashing
        raw_row_dict = {k: v for k, v in row_lower.items() if v and v.strip()}
        raw_hash = _calculate_raw_hash(raw_row_dict)

        return {
            "isin": isin_raw,
            "wkn": wkn,
            "issuer": issuer,
            "venue": venue,
            "underlying_raw": underlying_raw,
            "underlying_id": underlying_id,
            "direction": direction,
            "product_type": product_type,
            "financing_level": financing_level,
            "knockout_barrier": knockout_barrier,
            "ratio": ratio,
            "currency": currency,
            "underlying_currency": underlying_currency,
            "quanto": quanto,
            "open_end": open_end,
            "maturity": maturity,
            "first_trading_day": first_trading_day,
            "bid": bid,
            "ask": ask,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "quote_timestamp": quote_timestamp,
            "bid_only": bid_only,
            "knocked_out": knocked_out,
            "underlying_price_ref": underlying_price_ref,
            "raw_hash": raw_hash,
        }

    def normalize(self, raw: Any, **kwargs: Any) -> list[ProductSnapshot]:
        """Convert raw product records into ``ProductSnapshot`` instances.

        ``raw`` is expected to be a list of dicts returned by ``fetch()``.
        Any row that fails snapshot construction (e.g. a pydantic
        ``ValidationError`` for a field the schema requires, such as
        ``direction``) is recorded in ``self.last_errors`` -- attributed
        back to its file/line via the ``_source_file``/``_source_line``
        provenance keys ``fetch()`` attaches to each raw row -- rather than
        only being logged and silently dropped. This method *appends* to
        ``self.last_errors`` instead of resetting it, so parse-time errors
        collected by a preceding ``fetch()`` call are preserved.
        """
        del kwargs
        if not isinstance(raw, list):
            return []

        snapshots: list[ProductSnapshot] = []
        now = self._clock()

        for raw_product in raw:
            source_file = raw_product.get("_source_file", "<unknown>")
            source_line = raw_product.get("_source_line", 0)
            isin_for_error = raw_product.get("isin")
            try:
                quote_timestamp = raw_product["quote_timestamp"]
                # File was available since mtime or quote_timestamp
                available_at = max(now, quote_timestamp)
                is_stale = (now - quote_timestamp).total_seconds() > self._stale_after_s
                quality_score = 1.0 if not is_stale else 0.5

                snapshot = ProductSnapshot(
                    isin=raw_product["isin"],
                    wkn=raw_product["wkn"],
                    issuer=raw_product["issuer"],
                    venue=raw_product["venue"],
                    underlying_raw=raw_product["underlying_raw"],
                    underlying_id=raw_product["underlying_id"],
                    direction=raw_product["direction"],
                    product_type=raw_product["product_type"],
                    financing_level=raw_product["financing_level"],
                    knockout_barrier=raw_product["knockout_barrier"],
                    ratio=raw_product["ratio"],
                    currency=raw_product["currency"],
                    underlying_currency=raw_product["underlying_currency"],
                    quanto=raw_product["quanto"],
                    open_end=raw_product["open_end"],
                    maturity=raw_product["maturity"],
                    first_trading_day=raw_product["first_trading_day"],
                    bid=raw_product["bid"],
                    ask=raw_product["ask"],
                    bid_size=raw_product["bid_size"],
                    ask_size=raw_product["ask_size"],
                    quote_timestamp=quote_timestamp,
                    quote_presence=True,
                    bid_only=raw_product["bid_only"],
                    knocked_out=raw_product["knocked_out"],
                    underlying_price_ref=raw_product["underlying_price_ref"],
                    raw_hash=raw_product["raw_hash"],
                    # Provenance
                    observation_time=quote_timestamp,
                    available_at=available_at,
                    retrieved_at=now,
                    source_timestamp=quote_timestamp,
                    source="csv_import",
                    parser_version="csv_import/1",
                    is_stale=is_stale,
                    quality_score=quality_score,
                )
                snapshots.append(snapshot)
            except (KeyError, ValidationError, ValueError) as exc:
                self.last_errors.append(
                    RowError(
                        file=str(source_file),
                        line=int(source_line) if isinstance(source_line, int) else 0,
                        isin=isin_for_error,
                        error=str(exc),
                    )
                )
                logger.error(
                    "snapshot_normalization_error",
                    file=source_file,
                    line=source_line,
                    error=str(exc),
                )

        return snapshots

    def healthcheck(self) -> HealthCheckResult:
        """Check the health of the CSV import directory.

        CSV import is an OPTIONAL, user-curated fallback source (see
        ``docs/data_sources.md``): the common/default state is an absent or
        empty ``state/imports/products/`` directory, e.g. every CI run where
        nobody has dropped a manual export there. That is not an
        operational failure and must never be reported as FAIL -- it is
        WARN ("nothing to import"), so that ``sources health
        --email-on-fail``/``--fail-on-error`` (which key off *critical*
        sources only, see ``monitoring/source_health.OPTIONAL_SOURCES`` /
        ``critical_failures``) do not fire a daily false alarm for it.

        FAIL is reserved for content that is actually defective once CSV
        files are present: every file failed to even open/decode, or every
        row that was read failed to parse.
        """
        now = self._clock()

        if not self._import_dir.exists():
            return HealthCheckResult(
                source="csv_import",
                status=HealthStatus.WARN,
                ok=True,
                latency_ms=None,
                checked_at=now,
                message=f"optional source: import directory does not exist: {self._import_dir}",
            )

        csv_files = list(self._import_dir.glob("*.csv"))
        if not csv_files:
            return HealthCheckResult(
                source="csv_import",
                status=HealthStatus.WARN,
                ok=True,
                latency_ms=0.0,
                checked_at=now,
                message=(
                    f"optional source: no CSV files in {self._import_dir} "
                    "— export product lists there if desired"
                ),
            )

        # Try to read and parse all files
        snapshots = self.fetch_products([])
        total_rows = len(snapshots)
        error_rows = len(self.last_errors)
        unreadable_files = len(self.last_file_errors)

        if total_rows == 0:
            if error_rows > 0 or unreadable_files > 0:
                # Files are present but nothing usable came out of them --
                # either every row failed to parse or a file could not even
                # be opened/decoded. That is an actually broken source, not
                # "nobody uploaded anything today": FAIL, not WARN.
                parts = []
                if error_rows:
                    parts.append(f"{error_rows} row(s) failed to parse")
                if unreadable_files:
                    parts.append(f"{unreadable_files} file(s) unreadable: {self.last_file_errors}")
                return HealthCheckResult(
                    source="csv_import",
                    status=HealthStatus.FAIL,
                    ok=False,
                    latency_ms=0.0,
                    checked_at=now,
                    message=f"CSV files present but defective: {'; '.join(parts)}",
                )
            return HealthCheckResult(
                source="csv_import",
                status=HealthStatus.WARN,
                ok=True,
                latency_ms=0.0,
                checked_at=now,
                message="CSV files present but no valid rows parsed",
            )

        total = total_rows + error_rows
        error_rate = error_rows / total if total > 0 else 0.0

        if error_rate > 0.5:
            return HealthCheckResult(
                source="csv_import",
                status=HealthStatus.WARN,
                ok=True,
                latency_ms=0.0,
                checked_at=now,
                message=f"high error rate: {error_rows}/{total_rows + error_rows} rows failed",
            )

        # Check if all quotes are stale (is_stale is already computed by normalize())
        if snapshots:
            stale_count = sum(1 for snapshot in snapshots if snapshot.is_stale)
            if stale_count == len(snapshots):
                msg = f"all {len(snapshots)} quotes are stale (older than {self._stale_after_s}s)"
                return HealthCheckResult(
                    source="csv_import",
                    status=HealthStatus.WARN,
                    ok=True,
                    latency_ms=0.0,
                    checked_at=now,
                    message=msg,
                )

        return HealthCheckResult(
            source="csv_import",
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=0.0,
            checked_at=now,
            message=f"{len(snapshots)} valid rows parsed from {len(csv_files)} files",
        )

    def metadata(self) -> AdapterMetadata:
        """Return adapter metadata."""
        return AdapterMetadata(
            name="csv_import",
            kind="products",
            version="1",
        )


def _factory(src: SourceConfig, http: Any) -> CsvProductImportAdapter:
    """Factory function for creating a CsvProductImportAdapter from configuration."""
    import_dir = Path(src.base_url)
    if not import_dir.is_absolute():
        import_dir = Path.cwd() / import_dir
    return CsvProductImportAdapter(import_dir)


# Idempotent registration
def _register() -> None:
    from turboedge.adapters.registry import PRODUCT_ADAPTER_FACTORIES

    if "csv_import" not in PRODUCT_ADAPTER_FACTORIES:
        from turboedge.adapters.registry import register_product_adapter

        register_product_adapter("csv_import", _factory)


_register()
