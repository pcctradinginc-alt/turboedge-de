"""Tests for the CSV import adapter (no network access required)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from turboedge.adapters.csv_import import (
    CsvProductImportAdapter,
    _parse_boolean,
    _parse_date,
    _parse_number,
    _parse_ratio,
    _parse_timestamp,
)
from turboedge.adapters.registry import PRODUCT_ADAPTER_FACTORIES, ProductSourceAdapter
from turboedge.config import SourceConfig
from turboedge.storage.schemas import (
    Direction,
    HealthStatus,
    ProductSnapshot,
    ProductType,
)

# Test reference time (2026-09-11 12:00 UTC)
TEST_NOW = datetime(2026, 9, 11, 12, 0, 0, tzinfo=UTC)


class TestParseNumber:
    """Test number parsing in German and English formats."""

    def test_parse_number_english_no_sep(self) -> None:
        assert _parse_number("1234") == 1234.0

    def test_parse_number_english_decimal(self) -> None:
        assert _parse_number("1234.56") == 1234.56

    def test_parse_number_english_thousands(self) -> None:
        assert _parse_number("1,234.56") == 1234.56

    def test_parse_number_german_decimal(self) -> None:
        assert _parse_number("1234,56") == 1234.56

    def test_parse_number_german_thousands(self) -> None:
        assert _parse_number("1.234,56") == 1234.56

    def test_parse_number_ambiguous_rightmost_comma(self) -> None:
        assert _parse_number("1.234,56") == 1234.56

    def test_parse_number_ambiguous_rightmost_period(self) -> None:
        assert _parse_number("1,234.56") == 1234.56

    def test_parse_number_with_spaces(self) -> None:
        assert _parse_number("1 234,56") == 1234.56

    def test_parse_number_empty(self) -> None:
        assert _parse_number("") is None
        assert _parse_number(None) is None

    def test_parse_number_invalid(self) -> None:
        assert _parse_number("abc") is None


class TestParseRatio:
    """Test ratio parsing (decimal and colon notation)."""

    def test_parse_ratio_decimal_english(self) -> None:
        assert _parse_ratio("0.01") == 0.01

    def test_parse_ratio_decimal_german(self) -> None:
        assert _parse_ratio("0,01") == 0.01

    def test_parse_ratio_colon_notation(self) -> None:
        assert _parse_ratio("100:1") == 0.01

    def test_parse_ratio_colon_with_spaces(self) -> None:
        assert _parse_ratio("10 : 1") == 0.1

    def test_parse_ratio_empty(self) -> None:
        assert _parse_ratio("") is None
        assert _parse_ratio(None) is None

    def test_parse_ratio_invalid(self) -> None:
        assert _parse_ratio("abc") is None
        assert _parse_ratio("0:0") is None  # Division by zero


class TestParseBoolean:
    """Test boolean parsing."""

    def test_parse_boolean_true_variants(self) -> None:
        assert _parse_boolean("true") is True
        assert _parse_boolean("1") is True
        assert _parse_boolean("yes") is True
        assert _parse_boolean("ja") is True
        assert _parse_boolean("wahr") is True
        assert _parse_boolean("TRUE") is True

    def test_parse_boolean_false_variants(self) -> None:
        assert _parse_boolean("false") is False
        assert _parse_boolean("0") is False
        assert _parse_boolean("no") is False
        assert _parse_boolean("nein") is False
        assert _parse_boolean("falsch") is False
        assert _parse_boolean("FALSE") is False

    def test_parse_boolean_empty(self) -> None:
        assert _parse_boolean("") is None
        assert _parse_boolean(None) is None

    def test_parse_boolean_invalid(self) -> None:
        assert _parse_boolean("maybe") is None


class TestParseTimestamp:
    """Test ISO-8601 timestamp parsing with timezone handling.

    Naive timestamps must always be localized via the real Europe/Berlin
    tzdata (zoneinfo), never a hardcoded fixed offset -- the DST boundary
    (late March - late October) means a fixed "assume UTC+2" guess is wrong
    for roughly half the year.
    """

    def test_parse_timestamp_iso_date_only(self) -> None:
        # 2026-09-10 is within CEST (DST): midnight Europe/Berlin = 2026-09-09T22:00:00Z.
        result = _parse_timestamp("2026-09-10")
        assert result == datetime(2026, 9, 9, 22, 0, 0, tzinfo=UTC)

    def test_parse_timestamp_iso_datetime_naive(self) -> None:
        # 2026-09-11 is within CEST (UTC+2): 12:00 Europe/Berlin = 10:00Z.
        result = _parse_timestamp("2026-09-11T12:00:00")
        assert result == datetime(2026, 9, 11, 10, 0, 0, tzinfo=UTC)

    def test_parse_timestamp_iso_datetime_with_tz(self) -> None:
        result = _parse_timestamp("2026-09-11T12:00:00+02:00")
        assert result == datetime(2026, 9, 11, 10, 0, 0, tzinfo=UTC)

    def test_parse_timestamp_winter_time_cet(self) -> None:
        """Winter (CET, UTC+1): 2026-01-15T10:00 (naive) -> 2026-01-15T09:00Z."""
        result = _parse_timestamp("2026-01-15T10:00:00")
        assert result == datetime(2026, 1, 15, 9, 0, 0, tzinfo=UTC)

    def test_parse_timestamp_summer_time_cest(self) -> None:
        """Summer (CEST, UTC+2): 2026-07-15T10:00 (naive) -> 2026-07-15T08:00Z."""
        result = _parse_timestamp("2026-07-15T10:00:00")
        assert result == datetime(2026, 7, 15, 8, 0, 0, tzinfo=UTC)

    def test_parse_timestamp_empty(self) -> None:
        assert _parse_timestamp("") is None
        assert _parse_timestamp(None) is None

    def test_parse_timestamp_invalid(self) -> None:
        assert _parse_timestamp("not a date") is None
        assert _parse_timestamp("2026-13-01") is None


class TestParseDate:
    """Test date parsing."""

    def test_parse_date_valid(self) -> None:
        result = _parse_date("2026-09-10")
        assert result == date(2026, 9, 10)

    def test_parse_date_empty(self) -> None:
        assert _parse_date("") is None
        assert _parse_date(None) is None

    def test_parse_date_invalid(self) -> None:
        assert _parse_date("2026-13-01") is None
        assert _parse_date("not a date") is None


class TestCsvImportAdapter:
    """Test the CsvProductImportAdapter."""

    def test_adapter_name(self) -> None:
        adapter = CsvProductImportAdapter(Path("/tmp"))
        assert adapter.name == "csv_import"

    def test_adapter_metadata(self) -> None:
        adapter = CsvProductImportAdapter(Path("/tmp"))
        metadata = adapter.metadata()
        assert metadata.name == "csv_import"
        assert metadata.kind == "products"
        assert metadata.version == "1"

    def test_healthcheck_missing_directory(self) -> None:
        # csv_import is an OPTIONAL, user-curated fallback source (see
        # monitoring/source_health.OPTIONAL_SOURCES): a missing import
        # directory is the everyday default state, not an operational
        # failure -- WARN, never FAIL, so it can never trigger
        # `sources health --email-on-fail`/`--fail-on-error` on its own.
        adapter = CsvProductImportAdapter(Path("/nonexistent/path/12345"))
        result = adapter.healthcheck()
        assert result.status == HealthStatus.WARN
        assert result.ok
        assert "optional source" in result.message

    def test_healthcheck_empty_directory(self, tmp_path: Path) -> None:
        # No CSV files present is likewise the everyday default (nobody
        # uploaded a manual export today) -- WARN, not FAIL.
        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        result = adapter.healthcheck()
        assert result.status == HealthStatus.WARN
        assert result.ok
        assert "optional source" in result.message

    def test_healthcheck_fail_when_every_row_fails_to_parse(self, tmp_path: Path) -> None:
        # CSV files ARE present but every row is defective -- this is an
        # actually broken source, unlike "nothing uploaded" above: FAIL.
        csv_file = tmp_path / "broken.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "INVALID;Bank1;DAX;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00\n"
        )
        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        result = adapter.healthcheck()
        assert result.status == HealthStatus.FAIL
        assert not result.ok
        assert "defective" in result.message

    def test_healthcheck_fail_when_file_unreadable(self, tmp_path: Path) -> None:
        # A CSV file that cannot even be decoded is likewise an actually
        # broken source: FAIL, with the failure tracked separately from
        # per-row parse errors via `last_file_errors`.
        csv_file = tmp_path / "bad_encoding.csv"
        csv_file.write_bytes(b"\xff\xfe\x00\x01not-valid-utf8\x80\x81")
        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        result = adapter.healthcheck()
        assert result.status == HealthStatus.FAIL
        assert not result.ok
        assert adapter.last_file_errors

    def test_healthcheck_valid_data(self, tmp_path: Path) -> None:
        # Create a minimal valid CSV
        csv_file = tmp_path / "test.csv"
        # Use UTC timestamp (with +00:00 offset) to match TEST_NOW
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0001;TestBank;DAX;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00+00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        result = adapter.healthcheck()
        assert result.status == HealthStatus.PASS
        assert result.ok

    def test_fetch_and_normalize_semicolon_german_format(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_content = (
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            f"DE000ABC0001;Bank1;DAX;long;18000,00;18000,00;0,01;4,80;4,86;{TEST_NOW.isoformat()}\n"
        )
        csv_file.write_text(csv_content)

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 1
        assert raw[0]["isin"] == "DE000ABC0001"
        assert raw[0]["issuer"] == "Bank1"
        assert raw[0]["underlying_id"] == "DAX"
        assert raw[0]["direction"] == Direction.LONG
        assert raw[0]["ratio"] == 0.01
        assert raw[0]["bid"] == 4.80
        assert raw[0]["ask"] == 4.86
        # fetch() attaches provenance used by normalize() for error attribution.
        assert raw[0]["_source_file"] == "products.csv"
        assert raw[0]["_source_line"] == 2

        snapshots = adapter.normalize(raw)
        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert isinstance(snapshot, ProductSnapshot)
        assert snapshot.isin == "DE000ABC0001"
        assert snapshot.quality_score == 1.0

    def test_fetch_and_normalize_comma_english_format(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin,issuer,underlying,direction,financing_level,knockout_barrier,ratio,bid,ask,quote_timestamp\n"
            "DE000ABC0002,Bank2,S&P 500,short,4200.00,4100.00,0.01,3.50,3.55,2026-09-11T12:00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 1
        assert raw[0]["isin"] == "DE000ABC0002"
        assert raw[0]["underlying_id"] == "SPX"
        assert raw[0]["direction"] == Direction.SHORT
        assert raw[0]["financing_level"] == 4200.0
        assert raw[0]["knockout_barrier"] == 4100.0
        assert raw[0]["ratio"] == 0.01

    def test_fetch_handles_bom(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        # Write with UTF-8 BOM
        content = (
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0003;Bank3;EUR/USD;long;1,1000;1,0800;0,001;1,20;1,21;2026-09-11\n"
        )
        csv_file.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 1
        assert raw[0]["isin"] == "DE000ABC0003"

    def test_optional_fields(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp;wkn;venue;currency;open_end\n"
            "DE000ABC0004;Bank4;DAX;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00;TEST04;Stuttgart;EUR;true\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()
        snapshots = adapter.normalize(raw)

        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert snapshot.wkn == "TEST04"
        assert snapshot.venue == "Stuttgart"
        assert snapshot.currency == "EUR"
        assert snapshot.open_end is True

    def test_product_type_classification(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp;open_end;maturity\n"
            "DE000ABC0005;Bank5;DAX;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00;true;\n"
            "DE000ABC0006;Bank6;DAX;short;18000,00;17500,00;0,01;3,50;3,55;2026-09-11T12:00:00;false;2026-12-31\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()
        snapshots = adapter.normalize(raw)

        assert len(snapshots) == 2
        assert snapshots[0].product_type == ProductType.TURBO_OPEN_END
        assert snapshots[1].product_type == ProductType.TURBO_CLASSIC

    def test_ratio_colon_notation(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0007;Bank7;DAX;long;18000,00;18000,00;100:1;4,80;4,86;2026-09-11T12:00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 1
        assert raw[0]["ratio"] == 0.01

    def test_error_handling_missing_required_field(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier\n"
            "DE000ABC0008;Bank8;DAX;long;18000,00;18000,00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 0
        assert len(adapter.last_errors) == 1
        assert "ratio" in adapter.last_errors[0].error.lower()

    def test_error_handling_invalid_isin(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "INVALID;Bank9;DAX;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 0
        assert len(adapter.last_errors) == 1

    def test_error_handling_invalid_ratio(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0009;Bank9;DAX;long;18000,00;18000,00;invalid;4,80;4,86;2026-09-11T12:00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 0
        assert len(adapter.last_errors) == 1

    def test_error_handling_invalid_timestamp(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0010;Bank10;DAX;long;18000,00;18000,00;0,01;4,80;4,86;not-a-date\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 0
        assert len(adapter.last_errors) == 1

    def test_missing_bid_column_is_row_error(self, tmp_path: Path) -> None:
        """bid missing entirely -> RowError; never silently dropped/imputed (rule 29)."""
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;ask;quote_timestamp\n"
            "DE000ABC0030;Bank30;DAX;long;18000,00;18000,00;0,01;4,86;2026-09-11T12:00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 0
        assert len(adapter.last_errors) == 1
        assert "bid" in adapter.last_errors[0].error.lower()

    def test_unparseable_bid_is_row_error_not_imputed(self, tmp_path: Path) -> None:
        """bid present but not a valid number -> RowError, never coerced to None/ask."""
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0031;Bank31;DAX;long;18000,00;18000,00;0,01;n/a;4,86;2026-09-11T12:00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 0
        assert len(adapter.last_errors) == 1
        assert "bid" in adapter.last_errors[0].error.lower()

    def test_missing_ask_column_is_row_error(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;quote_timestamp\n"
            "DE000ABC0032;Bank32;DAX;long;18000,00;18000,00;0,01;4,80;2026-09-11T12:00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 0
        assert len(adapter.last_errors) == 1
        assert "ask" in adapter.last_errors[0].error.lower()

    def test_missing_financing_level_is_row_error(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0033;Bank33;DAX;long;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 0
        assert len(adapter.last_errors) == 1
        assert "financing_level" in adapter.last_errors[0].error.lower()

    def test_missing_knockout_barrier_is_row_error(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0034;Bank34;DAX;long;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 0
        assert len(adapter.last_errors) == 1
        assert "knockout_barrier" in adapter.last_errors[0].error.lower()

    def test_multiple_csv_files(self, tmp_path: Path) -> None:
        csv1 = tmp_path / "file1.csv"
        csv1.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0011;Bank11;DAX;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00\n"
        )

        csv2 = tmp_path / "file2.csv"
        csv2.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0012;Bank12;S&P 500;short;4200,00;4100,00;0,01;3,50;3,55;2026-09-11T12:00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 2
        assert raw[0]["isin"] == "DE000ABC0011"
        assert raw[1]["isin"] == "DE000ABC0012"

    def test_stale_quote_detection(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        # Use a timestamp 20 minutes ago (1200 seconds, beyond default 900s threshold)
        old_time = (datetime.now(UTC) - timedelta(seconds=1200)).isoformat()
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            f"DE000ABC0013;Bank13;DAX;long;18000,00;18000,00;0,01;4,80;4,86;{old_time}\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()
        snapshots = adapter.normalize(raw)

        assert len(snapshots) == 1
        assert snapshots[0].is_stale is True
        assert snapshots[0].quality_score == 0.5

    def test_fresh_quote_detection(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        # Use a recent timestamp
        now = TEST_NOW.isoformat()
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            f"DE000ABC0014;Bank14;DAX;long;18000,00;18000,00;0,01;4,80;4,86;{now}\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()
        snapshots = adapter.normalize(raw)

        assert len(snapshots) == 1
        assert snapshots[0].is_stale is False
        assert snapshots[0].quality_score == 1.0

    def test_unresolvable_underlying(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0015;Bank15;UnknownUnderlying;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()
        snapshots = adapter.normalize(raw)

        assert len(snapshots) == 1
        assert snapshots[0].underlying_id is None

    def test_raw_hash_calculation(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0016;Bank16;DAX;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()

        assert len(raw) == 1
        # raw_hash should be a SHA256 hex string
        assert len(raw[0]["raw_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in raw[0]["raw_hash"])

    def test_factory_function_relative_path(self, tmp_path: Path) -> None:
        """Test that factory resolves relative paths correctly."""
        # Create a temporary directory structure
        rel_import_dir = tmp_path / "relative" / "import"
        rel_import_dir.mkdir(parents=True)

        csv_file = rel_import_dir / "test.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0017;Bank17;DAX;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00\n"
        )

        # Create a mock SourceConfig
        src_cfg = SourceConfig(
            enabled=True,
            base_url=str(rel_import_dir),
            timeout_s=5.0,
            min_interval_s=0.0,
            max_pages=1,
            user_agent="test",
        )

        # Import factory at call site to test registration
        from turboedge.adapters.csv_import import _factory

        adapter = _factory(src_cfg, None)  # type: ignore
        assert adapter.name == "csv_import"

    def test_registry_idempotent_registration(self) -> None:
        """Test that the adapter can be registered/imported multiple times without error."""
        # The module-level _register() call should be idempotent
        initial_count = len(PRODUCT_ADAPTER_FACTORIES)

        # Import the module again (mock scenario)
        import importlib

        import turboedge.adapters.csv_import

        importlib.reload(turboedge.adapters.csv_import)

        # Should still be the same count, not doubled
        assert len(PRODUCT_ADAPTER_FACTORIES) <= initial_count + 1


class TestFetchProductsContract:
    """fetch_products() must return finished, validated ProductSnapshot objects
    -- the ProductSourceAdapter contract the scan pipeline relies on -- never
    raw dicts.
    """

    def test_fetch_products_returns_product_snapshots(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0040;Bank40;DAX;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00+00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        products = adapter.fetch_products([])

        assert len(products) == 1
        assert all(isinstance(p, ProductSnapshot) for p in products)
        assert products[0].isin == "DE000ABC0040"

    def test_fetch_products_filters_by_underlying_id(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0041;Bank41;DAX;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00+00:00\n"
            "DE000ABC0042;Bank42;S&P 500;short;4200,00;4100,00;0,01;3,50;3,55;2026-09-11T12:00:00+00:00\n"  # noqa: E501
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        products = adapter.fetch_products(["DAX"])

        assert len(products) == 1
        assert products[0].isin == "DE000ABC0041"
        assert products[0].underlying_id == "DAX"

    def test_fetch_products_empty_underlying_ids_means_unfiltered(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0043;Bank43;DAX;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00+00:00\n"
            "DE000ABC0044;Bank44;S&P 500;short;4200,00;4100,00;0,01;3,50;3,55;2026-09-11T12:00:00+00:00\n"  # noqa: E501
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        products = adapter.fetch_products([])

        assert len(products) == 2

    def test_fetch_products_resets_last_errors_each_call(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "INVALID;Bank1;DAX;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00\n"
        )
        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)

        adapter.fetch_products([])
        assert len(adapter.last_errors) == 1

        # Fix the file and re-fetch: stale errors from the previous call must not linger.
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp\n"
            "DE000ABC0045;Bank1;DAX;long;18000,00;18000,00;0,01;4,80;4,86;2026-09-11T12:00:00\n"
        )
        products = adapter.fetch_products([])
        assert len(adapter.last_errors) == 0
        assert len(products) == 1

    def test_missing_bid_excluded_from_fetch_products(self, tmp_path: Path) -> None:
        """A row with a missing pricing-critical field never reaches fetch_products()'s
        output -- it is a RowError, not a ProductSnapshot with an imputed bid.
        """
        csv_file = tmp_path / "products.csv"
        csv_file.write_text(
            "isin;issuer;underlying;direction;financing_level;knockout_barrier;ratio;ask;quote_timestamp\n"
            "DE000ABC0046;Bank46;DAX;long;18000,00;18000,00;0,01;4,86;2026-09-11T12:00:00\n"
        )

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        products = adapter.fetch_products([])

        assert products == []
        assert len(adapter.last_errors) == 1
        assert "bid" in adapter.last_errors[0].error.lower()


class TestProtocolConformance:
    """CsvProductImportAdapter must satisfy ProductSourceAdapter structurally
    (it deliberately does not inherit the Protocol class -- see the docstring
    on CsvProductImportAdapter).
    """

    def test_isinstance_check(self, tmp_path: Path) -> None:
        adapter = CsvProductImportAdapter(tmp_path)
        assert isinstance(adapter, ProductSourceAdapter)

    def test_mypy_compatible_assignment(self, tmp_path: Path) -> None:
        # The assignment itself is the assertion that matters: mypy must
        # accept a CsvProductImportAdapter wherever a ProductSourceAdapter is
        # expected, purely via structural typing (checked for real by
        # `uv run mypy src/turboedge/adapters`, not just at runtime here).
        adapter: ProductSourceAdapter = CsvProductImportAdapter(tmp_path)
        assert adapter.name == "csv_import"
        assert adapter.fetch_products([]) == []


class TestCsvImportIntegration:
    """Integration tests combining parsing and snapshot creation."""

    def test_full_pipeline_mixed_format(self, tmp_path: Path) -> None:
        """Test reading, parsing, and normalizing a realistic mixed CSV."""
        csv_file = tmp_path / "products.csv"
        csv_content = (
            "isin;wkn;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp;venue;currency;open_end;maturity\n"
            "DE000ABC0020;TST20;Bank20;DAX;long;18000,50;18000,50;0,01;4,80;4,86;2026-09-11T12:00:00;Stuttgart;EUR;true;\n"
            "DE000ABC0021;TST21;Bank21;Euro Stoxx 50;short;4750,00;4700,00;0,01;5,20;5,25;2026-09-11T12:00:00;Frankfurt;EUR;false;2026-12-31\n"  # noqa: E501
            "DE000ABC0022;TST22;Bank22;EURUSD;long;1,1000;1,0800;100:1;0,45;0,47;2026-09-11T12:00:00;Stuttgart;EUR;true;\n"
        )
        csv_file.write_text(csv_content)

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        raw = adapter.fetch()
        snapshots = adapter.normalize(raw)

        assert len(snapshots) == 3

        s0 = snapshots[0]
        assert s0.isin == "DE000ABC0020"
        assert s0.wkn == "TST20"
        assert s0.underlying_id == "DAX"
        assert s0.direction == Direction.LONG
        assert s0.product_type == ProductType.TURBO_OPEN_END
        assert s0.source == "csv_import"
        assert s0.parser_version == "csv_import/1"

        s1 = snapshots[1]
        assert s1.underlying_id == "ESTX50"
        assert s1.direction == Direction.SHORT
        assert s1.product_type == ProductType.TURBO_CLASSIC
        assert s1.maturity == date(2026, 12, 31)

        s2 = snapshots[2]
        assert s2.underlying_id == "EURUSD"
        assert s2.ratio == 0.01

    def test_full_pipeline_via_fetch_products(self, tmp_path: Path) -> None:
        """The same mixed CSV, exercised through the public fetch_products()
        contract the scan pipeline actually calls.
        """
        csv_file = tmp_path / "products.csv"
        csv_content = (
            "isin;wkn;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp;venue;currency;open_end;maturity\n"
            "DE000ABC0023;TST23;Bank23;DAX;long;18000,50;18000,50;0,01;4,80;4,86;2026-09-11T12:00:00;Stuttgart;EUR;true;\n"
            "DE000ABC0024;TST24;Bank24;Euro Stoxx 50;short;4750,00;4700,00;0,01;5,20;5,25;2026-09-11T12:00:00;Frankfurt;EUR;false;2026-12-31\n"  # noqa: E501
        )
        csv_file.write_text(csv_content)

        adapter = CsvProductImportAdapter(tmp_path, clock=lambda: TEST_NOW)
        products = adapter.fetch_products(["DAX", "ESTX50"])

        assert len(products) == 2
        assert all(isinstance(p, ProductSnapshot) for p in products)
        assert {p.underlying_id for p in products} == {"DAX", "ESTX50"}
