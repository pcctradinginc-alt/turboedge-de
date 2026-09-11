from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from turboedge.adapters.base import AdapterError
from turboedge.adapters.fallback_prices import (
    YFinancePriceAdapter,
    _normalize_history,
    _to_utc_datetime,
)
from turboedge.storage.schemas import HealthStatus


class _FakeTicker:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def history(self, **_kwargs: Any) -> pd.DataFrame:
        return self._frame


def _make_history(dates: list[Any], *, tz: str | None = None) -> pd.DataFrame:
    """Build a synthetic yfinance-shaped OHLCV frame.

    ``dates`` may already be tz-aware (mixed origin, e.g. from different
    zones) - if ``tz`` is given, the resulting DatetimeIndex is localized to
    it (requires tz-naive inputs).
    """
    index = pd.DatetimeIndex(dates)
    if tz is not None:
        index = index.tz_localize(tz)
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [18000.0 + i for i in range(n)],
            "High": [18100.0 + i for i in range(n)],
            "Low": [17950.0 + i for i in range(n)],
            "Close": [18050.0 + i for i in range(n)],
            "Volume": [1_000_000.0 + i for i in range(n)],
        },
        index=index,
    )


# -- _normalize_history: trading-date / session-close logic ----------------


def test_dax_excludes_running_session_bar() -> None:
    # ^GDAXI daily bars are indexed at local midnight, Europe/Berlin.
    frame = _make_history(["2026-09-09", "2026-09-10"], tz="Europe/Berlin")
    retrieved_at = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)  # before DAX 22:00 CEST close

    bars = _normalize_history(frame, underlying_id="DAX", retrieved_at=retrieved_at)

    assert len(bars) == 1
    bar = bars[0]
    assert bar.ts == datetime(2026, 9, 9, tzinfo=UTC)
    assert bar.available_at == datetime(2026, 9, 9, 20, 0, tzinfo=UTC)  # 22:00 CEST = 20:00 UTC
    assert bar.observation_time == bar.available_at
    assert bar.available_at <= retrieved_at


def test_dax_includes_bar_once_session_has_closed() -> None:
    frame = _make_history(["2026-09-09", "2026-09-10"], tz="Europe/Berlin")
    retrieved_at = datetime(2026, 9, 10, 21, 0, tzinfo=UTC)  # after DAX 22:00 CEST close

    bars = _normalize_history(frame, underlying_id="DAX", retrieved_at=retrieved_at)

    assert len(bars) == 2
    assert bars[1].ts == datetime(2026, 9, 10, tzinfo=UTC)
    assert bars[1].available_at == datetime(2026, 9, 10, 20, 0, tzinfo=UTC)


def test_dax_bug_regression_utc_shift_no_longer_misdates_bars() -> None:
    # Regression for the original bug: converting the Europe/Berlin midnight
    # index to UTC *before* taking .date() used to shift the trading date
    # back by one day (2026-09-10 00:00+02:00 -> 2026-09-09 22:00 UTC).
    frame = _make_history(["2026-09-10"], tz="Europe/Berlin")
    retrieved_at = datetime(2026, 9, 11, 0, 0, tzinfo=UTC)  # well after the 10th's close

    bars = _normalize_history(frame, underlying_id="DAX", retrieved_at=retrieved_at)

    assert len(bars) == 1
    assert bars[0].ts == datetime(2026, 9, 10, tzinfo=UTC)
    assert bars[0].source_timestamp == datetime(2026, 9, 9, 22, 0, tzinfo=UTC)


def test_ndx_uses_new_york_session_close() -> None:
    # ^NDX daily bars are indexed at local midnight, America/New_York.
    frame = _make_history(["2026-09-09", "2026-09-10"], tz="America/New_York")

    before_close = datetime(2026, 9, 10, 20, 0, tzinfo=UTC)  # before 17:00 EDT close
    bars_before = _normalize_history(frame, underlying_id="NDX", retrieved_at=before_close)
    assert len(bars_before) == 1
    assert bars_before[0].ts == datetime(2026, 9, 9, tzinfo=UTC)
    assert bars_before[0].available_at == datetime(2026, 9, 9, 21, 0, tzinfo=UTC)

    after_close = datetime(2026, 9, 10, 21, 0, tzinfo=UTC)  # 17:00 EDT = 21:00 UTC
    bars_after = _normalize_history(frame, underlying_id="NDX", retrieved_at=after_close)
    assert len(bars_after) == 2
    assert bars_after[1].ts == datetime(2026, 9, 10, tzinfo=UTC)
    assert bars_after[1].available_at == datetime(2026, 9, 10, 21, 0, tzinfo=UTC)


def test_fx_24h_market_uses_next_day_in_index_tz() -> None:
    # EURUSD=X: 24h market, no real session close - conservative rule is
    # "usable only from the start of the next calendar day, in the index's
    # own timezone".
    frame = _make_history(["2026-09-09", "2026-09-10"], tz="Europe/London")

    just_after_midnight_09 = datetime(2026, 9, 9, 23, 30, tzinfo=UTC)
    bars = _normalize_history(frame, underlying_id="EURUSD", retrieved_at=just_after_midnight_09)
    # 09-09's bar becomes available at 2026-09-10T00:00 Europe/London (BST,
    # UTC+1) = 2026-09-09T23:00 UTC, which has already passed.
    assert len(bars) == 1
    assert bars[0].ts == datetime(2026, 9, 9, tzinfo=UTC)
    assert bars[0].available_at == datetime(2026, 9, 9, 23, 0, tzinfo=UTC)

    just_before = datetime(2026, 9, 9, 22, 0, tzinfo=UTC)
    bars_before = _normalize_history(frame, underlying_id="EURUSD", retrieved_at=just_before)
    assert bars_before == []


def test_unknown_underlying_id_falls_back_to_conservative_next_day_rule() -> None:
    frame = _make_history(["2026-09-09"], tz="UTC")
    retrieved_at = datetime(2026, 9, 9, 23, 59, tzinfo=UTC)

    bars = _normalize_history(frame, underlying_id="SOME_UNKNOWN_ID", retrieved_at=retrieved_at)

    # Not yet the start of 2026-09-10 UTC -> still excluded.
    assert bars == []


def test_tz_naive_index_is_treated_as_utc() -> None:
    frame = _make_history([datetime(2026, 9, 9), datetime(2026, 9, 10)])  # tz-naive
    retrieved_at = datetime(2026, 9, 10, 23, 59, tzinfo=UTC)

    bars = _normalize_history(frame, underlying_id="EURUSD", retrieved_at=retrieved_at)

    # Treated as already-UTC: trade date 09-09, next-day-UTC close 09-10T00:00Z.
    assert len(bars) == 1
    assert bars[0].ts == datetime(2026, 9, 9, tzinfo=UTC)
    assert bars[0].available_at == datetime(2026, 9, 10, 0, 0, tzinfo=UTC)
    assert bars[0].source_timestamp == datetime(2026, 9, 9, tzinfo=UTC)


def test_duplicate_trade_dates_keep_last() -> None:
    dates = [
        datetime(2026, 9, 8, tzinfo=ZoneInfo("Europe/Berlin")),
        datetime(2026, 9, 9, tzinfo=ZoneInfo("Europe/Berlin")),
        datetime(2026, 9, 9, tzinfo=ZoneInfo("Europe/Berlin")),  # duplicate trade date
    ]
    frame = _make_history(dates)  # already tz-aware, no further localization
    retrieved_at = datetime(2026, 9, 11, tzinfo=UTC)

    bars = _normalize_history(frame, underlying_id="DAX", retrieved_at=retrieved_at)

    assert [b.ts for b in bars] == [
        datetime(2026, 9, 8, tzinfo=UTC),
        datetime(2026, 9, 9, tzinfo=UTC),
    ]
    # Last occurrence (index 2, Close=18052.0) wins over the first (index 1, Close=18051.0).
    assert bars[-1].close == 18052.0


def test_nan_ohlc_row_is_skipped() -> None:
    frame = _make_history(["2026-09-08", "2026-09-09"], tz="UTC")
    frame.loc[frame.index[0], "Close"] = float("nan")
    retrieved_at = datetime(2026, 9, 11, tzinfo=UTC)

    bars = _normalize_history(frame, underlying_id="EURUSD", retrieved_at=retrieved_at)

    assert len(bars) == 1
    assert bars[0].ts == datetime(2026, 9, 9, tzinfo=UTC)


def test_inconsistent_ohlc_row_is_skipped() -> None:
    frame = _make_history(["2026-09-08", "2026-09-09"], tz="UTC")
    # High below both Open and Close -> physically inconsistent bar.
    frame.loc[frame.index[0], "High"] = 1.0
    retrieved_at = datetime(2026, 9, 11, tzinfo=UTC)

    bars = _normalize_history(frame, underlying_id="EURUSD", retrieved_at=retrieved_at)

    assert len(bars) == 1
    assert bars[0].ts == datetime(2026, 9, 9, tzinfo=UTC)


def test_normalize_history_empty_frame_returns_empty_list() -> None:
    empty = pd.DataFrame()
    assert _normalize_history(empty, underlying_id="DAX", retrieved_at=datetime.now(UTC)) == []


def test_normalize_history_raises_adapter_error_on_missing_columns() -> None:
    frame = pd.DataFrame(
        {"Open": [1.0]}, index=pd.DatetimeIndex([datetime(2020, 1, 6, tzinfo=UTC)])
    )
    with pytest.raises(AdapterError):
        _normalize_history(frame, underlying_id="DAX", retrieved_at=datetime.now(UTC))


def test_normalize_history_bars_not_marked_stale() -> None:
    frame = _make_history(["2026-09-08", "2026-09-09"], tz="UTC")
    retrieved_at = datetime(2026, 9, 11, tzinfo=UTC)

    bars = _normalize_history(frame, underlying_id="EURUSD", retrieved_at=retrieved_at)

    assert len(bars) == 2
    assert all(not b.is_stale for b in bars)
    assert bars[0].ts < bars[1].ts


# -- invariant: no look-ahead, always UTC -----------------------------------


@given(
    underlying_id=st.sampled_from(
        ["DAX", "ESTX50", "SPX", "NDX", "NKY", "UKX", "SMI", "EURUSD", "XAU", "UNKNOWN_ID"]
    ),
    start_days_ago=st.integers(min_value=0, max_value=10),
    n_days=st.integers(min_value=1, max_value=5),
    retrieved_offset_hours=st.integers(min_value=-48, max_value=48),
)
def test_invariant_no_lookahead_and_utc_ts(
    underlying_id: str, start_days_ago: int, n_days: int, retrieved_offset_hours: int
) -> None:
    base = datetime(2026, 9, 1, tzinfo=UTC) - timedelta(days=start_days_ago)
    dates = [base + timedelta(days=i) for i in range(n_days)]
    frame = _make_history(dates)  # tz-naive -> treated as UTC
    retrieved_at = dates[-1] + timedelta(hours=retrieved_offset_hours)

    bars = _normalize_history(frame, underlying_id=underlying_id, retrieved_at=retrieved_at)

    for bar in bars:
        assert bar.available_at <= retrieved_at
        assert bar.ts.tzinfo is UTC
        assert bar.available_at.tzinfo is not None


# -- adapter wiring: monkeypatched yfinance.Ticker --------------------------


def test_fetch_daily_bars_uses_ticker_from_underlying_map(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _make_history(["2026-09-08", "2026-09-09"], tz="Europe/Berlin")
    seen_symbols: list[str] = []

    def _fake_ticker(symbol: str) -> _FakeTicker:
        seen_symbols.append(symbol)
        return _FakeTicker(frame)

    monkeypatch.setattr("turboedge.adapters.fallback_prices.yf.Ticker", _fake_ticker)

    adapter = YFinancePriceAdapter()
    bars = adapter.fetch_daily_bars("DAX")

    assert seen_symbols == ["^GDAXI"]
    assert all(b.underlying_id == "DAX" for b in bars)
    assert all(b.source == "yfinance" for b in bars)


def test_fetch_raises_adapter_error_when_ticker_throws(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(symbol: str) -> Any:
        raise RuntimeError("network down")

    monkeypatch.setattr("turboedge.adapters.fallback_prices.yf.Ticker", _raise)
    adapter = YFinancePriceAdapter()
    with pytest.raises(AdapterError):
        adapter.fetch(underlying_id="DAX")


def test_healthcheck_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _make_history(["2020-01-06", "2020-01-07"], tz="Europe/Berlin")
    monkeypatch.setattr(
        "turboedge.adapters.fallback_prices.yf.Ticker", lambda symbol: _FakeTicker(frame)
    )
    adapter = YFinancePriceAdapter()
    result = adapter.healthcheck()
    assert result.ok is True
    assert result.status == HealthStatus.PASS


def test_healthcheck_fail_when_fetch_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(symbol: str) -> Any:
        raise RuntimeError("network down")

    monkeypatch.setattr("turboedge.adapters.fallback_prices.yf.Ticker", _raise)
    adapter = YFinancePriceAdapter()
    result = adapter.healthcheck()
    assert result.ok is False
    assert result.status == HealthStatus.FAIL


def test_metadata() -> None:
    adapter = YFinancePriceAdapter()
    meta = adapter.metadata()
    assert meta.name == "yfinance"
    assert meta.kind == "price"


def test_to_utc_datetime_treats_naive_as_utc() -> None:
    assert _to_utc_datetime(datetime(2026, 1, 1, 12, 0)) == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def test_to_utc_datetime_rejects_unexpected_type() -> None:
    with pytest.raises(AdapterError):
        _to_utc_datetime("2026-01-01")
