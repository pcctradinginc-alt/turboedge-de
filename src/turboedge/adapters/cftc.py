"""CFTC Commitments of Traders positioning adapter (Workstream W12-D).

Traders in Financial Futures (TFF), the report that splits reportable
positions into dealer / asset manager / leveraged money / other -- the
breakdown that makes "who is crowded" answerable at all. Annual ZIP files
from the CFTC's own historical archive
(`cftc.gov/files/dea/history/fut_fin_txt_<year>.zip`), TFF history runs from
2010-07-20.

`robots.txt` check (2026-09-25): `/files/dea/history/` is not disallowed and
there is no `Crawl-delay` directive. The endpoint answers 200 to an honest
user agent. Nothing is bypassed.

AVAILABILITY MODEL -- the decision this module lives or dies by. A COT report
carries a Tuesday as-of date and is published the following **Friday at
15:30 ET**: three days of processing. Using the as-of date as `available_at`
would hand a model three days of future knowledge every single week, which
is by far the largest leak available in this data source.

Worse, that Friday is not guaranteed: the CFTC states plainly that holidays
change the release schedule. Rather than carry a US federal holiday calendar
(and be silently wrong whenever it drifts), `available_at` is set to the
as-of date **+ 6 days at 00:00 UTC** -- the Monday after. This gives up two
days of freshness in an ordinary week and cannot leak in a shifted one. If a
later session wants those two days back, the way to earn them is to *measure*
actual publication timestamps over a quarter, not to assume them.

The features built on this are weekly by construction, so the cost is small:
between two Tuesdays the value does not change anyway, and a positioning
series is a regime signal rather than a timing one.
"""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import UTC, datetime, timedelta

import structlog

from turboedge.adapters.base import (
    AdapterError,
    AdapterMetadata,
    HealthCheckResult,
    HttpClient,
)
from turboedge.storage.schemas import ExternalObservation, HealthStatus

logger = structlog.get_logger(__name__)

_SOURCE_NAME = "cftc"
_PARSER_VERSION = "1"
_SOURCE_VERSION = "cftc_tff_annual_txt"
_BASE_URL = "https://www.cftc.gov/files/dea/history"
_UNIT = "contracts"
_FREQUENCY = "weekly"

#: Publication lag in days from the Tuesday as-of date. Six, not three: the
#: regular Friday 15:30 ET release is as-of + 3, but holidays shift it and
#: the CFTC says so explicitly. See the module docstring.
_PUBLICATION_LAG_DAYS = 6

#: Market name (exact, as it appears in `Market_and_Exchange_Names`) -> the
#: short series prefix used in `series_id`. Deliberately a fixed map rather
#: than a substring match: "E-MINI S&P 500" and "MICRO E-MINI S&P 500 INDEX"
#: are different contracts, and a loose match would silently blend them.
DEFAULT_MARKETS: dict[str, str] = {
    "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE": "SP500",
    "NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE": "NASDAQ",
    "EURO FX - CHICAGO MERCANTILE EXCHANGE": "EURFX",
}
# Why the *Consolidated* equity-index series rather than the headline
# contracts: the individual contract names are not stable across this
# archive's 15-year span. In 2015 the same exposure was filed as "E-MINI S&P
# 500 STOCK INDEX" and "NASDAQ-100 STOCK INDEX (MINI)"; today it is "E-MINI
# S&P 500" and "NASDAQ MINI". An exact-name map against today's spelling
# silently returned ONE market instead of three for every year before ~2022 --
# measured, and the reason this map was changed. The Consolidated series keep
# one name throughout (verified present with 52 weekly rows in 2011, 2015,
# 2020 and 2025) and additionally aggregate across contract sizes, so a shift
# of open interest from E-mini into Micro does not register as a change in
# positioning when none occurred.

#: Trader groups extracted per market. `net` is long minus short; spread
#: positions are deliberately NOT netted in -- a spread position is
#: directionally flat by construction, and folding it into a net figure
#: would dilute exactly the directional signal this report exists to expose.
_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("dealer", "Dealer_Positions_Long_All", "Dealer_Positions_Short_All"),
    ("asset_mgr", "Asset_Mgr_Positions_Long_All", "Asset_Mgr_Positions_Short_All"),
    ("lev_money", "Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All"),
)

_OPEN_INTEREST_COLUMN = "Open_Interest_All"
#: The report-date COLUMN was renamed inside this archive -- files up to 2012
#: call it "Report_Date_as_MM_DD_YYYY", 2013 onward "Report_Date_as_YYYY-MM-DD"
#: -- while the VALUES stayed ISO in both ("2011-12-27" in a column whose name
#: promises MM_DD_YYYY). Measured 2026-09-25: the 2010-2012 files first failed
#: the contract check, then parsed zero rows when the name was taken at face
#: value and US dates were assumed. Hence both names, and both formats tried
#: per value rather than inferred from the header.
_REPORT_DATE_COLUMNS: tuple[str, ...] = (
    "Report_Date_as_YYYY-MM-DD",
    "Report_Date_as_MM_DD_YYYY",
)
_REPORT_DATE_FORMATS: tuple[str, ...] = ("%Y-%m-%d", "%m/%d/%Y")
_MARKET_COLUMN = "Market_and_Exchange_Names"

_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError)

TFF_FIRST_YEAR = 2010


def series_id_for(market_key: str, field: str) -> str:
    """Canonical series id, e.g. ``SP500.LEV_MONEY_NET`` or ``SP500.OPEN_INTEREST``."""
    return f"{market_key.upper()}.{field.upper()}"


def _available_at(report_date: datetime) -> datetime:
    """Conservative publication time: as-of date + 6 days at 00:00 UTC."""
    return (report_date + timedelta(days=_PUBLICATION_LAG_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def _to_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = raw.strip().replace(",", "")
    if not text or text == ".":
        return None
    try:
        return int(float(text))
    except _PARSE_ERROR_TYPES:
        return None


def _parse_report_date(raw: str) -> datetime | None:
    """Parse a report date, trying each known format.

    Tried per value rather than chosen from the column name, because the name
    is not reliable here: 2011's "Report_Date_as_MM_DD_YYYY" column contains
    ISO values.
    """
    for fmt in _REPORT_DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
        except _PARSE_ERROR_TYPES:
            continue
    return None


def _observation(
    market_key: str,
    field: str,
    value: float,
    *,
    report_date: datetime,
    available_at: datetime,
    retrieved_at: datetime,
) -> ExternalObservation:
    """One observation row. A module-level function rather than a closure
    over the parse loop: a closure would capture the loop variables by
    reference, which is correct only as long as every call happens inside
    the same iteration -- a constraint nothing enforces and a later refactor
    could quietly break (ruff B023)."""
    return ExternalObservation(
        series_id=series_id_for(market_key, field),
        value=value,
        unit=_UNIT,
        frequency=_FREQUENCY,
        source_version=_SOURCE_VERSION,
        observation_time=report_date,
        available_at=available_at,
        retrieved_at=retrieved_at,
        source=_SOURCE_NAME,
        parser_version=_PARSER_VERSION,
        quality_score=1.0,
    )


def parse_tff_csv(
    text: str,
    *,
    retrieved_at: datetime,
    markets: dict[str, str] | None = None,
) -> list[ExternalObservation]:
    """Parse one annual TFF text file into ExternalObservations.

    Emits, per (market, report date): one net position per trader group plus
    open interest. Rows for markets outside ``markets`` are skipped -- the
    file carries 107 markets and only a handful are in scope.

    Rows with an unparseable date or a missing position leg are skipped and
    counted, never imputed (CLAUDE.md rule 29).

    Raises:
        AdapterError: if the header lacks the columns this report is defined
            by -- a contract change, which must fail loudly rather than
            quietly yield nothing.
    """
    market_map = DEFAULT_MARKETS if markets is None else markets
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = [f.strip().strip('"') for f in (reader.fieldnames or [])]
    date_column = next((c for c in _REPORT_DATE_COLUMNS if c in fieldnames), "")
    required = {_MARKET_COLUMN, _OPEN_INTEREST_COLUMN}
    missing = required - set(fieldnames)
    if missing or not date_column:
        if not date_column:
            missing = missing | set(_REPORT_DATE_COLUMNS)
        raise AdapterError(
            f"CFTC TFF file missing required column(s) {sorted(missing)} "
            f"-- upstream contract changed (header={fieldnames[:6]}...)"
        )

    observations: list[ExternalObservation] = []
    skipped = 0
    for row in reader:
        clean = {(k or "").strip().strip('"'): v for k, v in row.items()}
        market_key = market_map.get((clean.get(_MARKET_COLUMN) or "").strip())
        if market_key is None:
            continue
        raw_date = (clean.get(date_column) or "").strip()
        report_date = _parse_report_date(raw_date)
        if report_date is None:
            skipped += 1
            continue
        available_at = _available_at(report_date)

        open_interest = _to_int(clean.get(_OPEN_INTEREST_COLUMN))
        if open_interest is None:
            skipped += 1
            continue
        observations.append(
            _observation(
                market_key,
                "open_interest",
                float(open_interest),
                report_date=report_date,
                available_at=available_at,
                retrieved_at=retrieved_at,
            )
        )
        for group, long_col, short_col in _GROUPS:
            long_pos, short_pos = _to_int(clean.get(long_col)), _to_int(clean.get(short_col))
            if long_pos is None or short_pos is None:
                skipped += 1
                continue
            observations.append(
                _observation(
                    market_key,
                    f"{group}_net",
                    float(long_pos - short_pos),
                    report_date=report_date,
                    available_at=available_at,
                    retrieved_at=retrieved_at,
                )
            )
    if skipped:
        logger.info("cftc_rows_skipped", skipped=skipped)
    return observations


class CftcPositioningAdapter:
    """Fetches CFTC Traders-in-Financial-Futures annual history files."""

    def __init__(
        self,
        http_client: HttpClient,
        *,
        markets: dict[str, str] | None = None,
        base_url: str = _BASE_URL,
    ) -> None:
        self._http = http_client
        self._markets = DEFAULT_MARKETS if markets is None else markets
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return _SOURCE_NAME

    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=_SOURCE_NAME,
            kind="external_series",
            version=_PARSER_VERSION,
            homepage="https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
        )

    def _url(self, year: int) -> str:
        return f"{self._base_url}/fut_fin_txt_{year}.zip"

    @staticmethod
    def extract_single_text_member(payload: bytes) -> str:
        """The one .txt member of a CFTC annual ZIP.

        Raises:
            AdapterError: if the archive is unreadable or does not contain
                exactly one text member -- an archive whose shape changed is
                not something to guess at.
        """
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
                if len(members) != 1:
                    raise AdapterError(f"expected exactly one .txt member, found {members!r}")
                return zf.read(members[0]).decode("utf-8", errors="replace")
        except zipfile.BadZipFile as exc:
            raise AdapterError(f"CFTC payload is not a readable ZIP: {exc}") from exc

    def fetch_year(self, year: int) -> list[ExternalObservation]:
        raw = self._http.get_bytes(self._url(year))
        text = self.extract_single_text_member(raw)
        return parse_tff_csv(text, retrieved_at=datetime.now(UTC), markets=self._markets)

    def fetch_observations(self, *, years: tuple[int, ...]) -> list[ExternalObservation]:
        """Every requested year, concatenated.

        One year failing does not discard the others: a partial history is
        still usable (the feature builder simply starts later), whereas
        raising would lose a decade because one file was briefly unavailable.
        """
        out: list[ExternalObservation] = []
        for year in years:
            try:
                parsed = self.fetch_year(year)
            except Exception as exc:
                logger.warning("cftc_year_fetch_failed", year=year, error=str(exc))
                continue
            logger.info("cftc_year_fetched", year=year, observations=len(parsed))
            out.extend(parsed)
        return out

    def healthcheck(self) -> HealthCheckResult:
        started = datetime.now(UTC)
        year = started.year
        try:
            parsed = self.fetch_year(year)
        except Exception as exc:
            return HealthCheckResult(
                source=_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=(datetime.now(UTC) - started).total_seconds() * 1000.0,
                checked_at=started,
                message=f"{year} fetch/parse failed: {exc}",
            )
        latency_ms = (datetime.now(UTC) - started).total_seconds() * 1000.0
        if not parsed:
            return HealthCheckResult(
                source=_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=latency_ms,
                checked_at=started,
                message=f"{year} returned no usable observations for the configured markets",
            )
        latest = max(o.observation_time for o in parsed)
        age_days = (started - latest).days
        # Weekly data with a 6-day availability lag: ~14 days is normal, a
        # month means publication actually stopped.
        status = HealthStatus.PASS if age_days <= 30 else HealthStatus.WARN
        return HealthCheckResult(
            source=_SOURCE_NAME,
            status=status,
            ok=status is HealthStatus.PASS,
            latency_ms=latency_ms,
            checked_at=started,
            message=(
                f"{len(parsed)} observation(s) for {year}, "
                f"latest as-of {latest.date().isoformat()} ({age_days}d old)"
            ),
        )


__all__ = [
    "DEFAULT_MARKETS",
    "TFF_FIRST_YEAR",
    "CftcPositioningAdapter",
    "parse_tff_csv",
    "series_id_for",
]
