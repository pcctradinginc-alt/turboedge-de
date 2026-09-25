"""Contract tests for the CFTC positioning adapter (W12-D).

Fixture is a real slice of `fut_fin_txt_2025.zip`'s FinFutYY.txt captured
2026-09-25 (header + 12 rows across the three configured markets).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from turboedge.adapters.base import AdapterError
from turboedge.adapters.cftc import parse_tff_csv, series_id_for

FIXTURE = Path(__file__).parent / "fixtures" / "cftc_tff_sample.txt"
RETRIEVED_AT = datetime(2026, 9, 25, 17, 0, tzinfo=UTC)


def load() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_real_fixture_parses_into_the_expected_series() -> None:
    obs = parse_tff_csv(load(), retrieved_at=RETRIEVED_AT)
    assert obs
    ids = {o.series_id for o in obs}
    for market in ("SP500", "NASDAQ", "EURFX"):
        for field in ("DEALER_NET", "ASSET_MGR_NET", "LEV_MONEY_NET", "OPEN_INTEREST"):
            assert f"{market}.{field}" in ids


def test_publication_lag_cannot_leak_even_on_a_shifted_week() -> None:
    """The single most important property of this adapter.

    A COT report is published the Friday after its as-of date, at 15:30 ET.
    Using the as-of date as `available_at` would hand every model three days
    of future knowledge, every week.

    The as-of date is USUALLY Tuesday but not always: the fixture contains
    2025-11-10, a Monday, because 2025-11-11 was Veterans Day -- measured,
    not assumed (2025 had 51 Tuesdays and that one Monday). So the test
    asserts the property that actually matters -- `available_at` lands
    strictly after that week's Friday publication -- rather than the weekday
    itself, which is exactly the assumption that made an earlier version of
    this test fail against real data.
    """
    obs = parse_tff_csv(load(), retrieved_at=RETRIEVED_AT)
    assert obs
    saw_shifted_week = False
    for o in obs:
        assert (o.available_at - o.observation_time).days == 6
        assert o.available_at.hour == 0 and o.available_at.minute == 0
        # Friday of the as-of date's own week, 15:30 ET ~= 20:30 UTC.
        days_to_friday = (4 - o.observation_time.weekday()) % 7
        publication = o.observation_time + timedelta(days=days_to_friday, hours=21)
        assert o.available_at > publication, (
            f"{o.series_id} as-of {o.observation_time.date()} "
            f"({o.observation_time.strftime('%A')}) would be usable before publication"
        )
        if o.observation_time.strftime("%A") != "Tuesday":
            saw_shifted_week = True
    assert saw_shifted_week, "fixture must retain the holiday-shifted week (2025-11-10)"


def test_net_is_long_minus_short_and_excludes_spread() -> None:
    """Spread positions are directionally flat by construction.

    Folding them into a net figure would dilute the very directional signal
    this report exists to expose.
    """
    text = (
        '"Market_and_Exchange_Names","As_of_Date_In_Form_YYMMDD",'
        '"Report_Date_as_YYYY-MM-DD","Open_Interest_All",'
        '"Dealer_Positions_Long_All","Dealer_Positions_Short_All",'
        '"Dealer_Positions_Spread_All","Asset_Mgr_Positions_Long_All",'
        '"Asset_Mgr_Positions_Short_All","Asset_Mgr_Positions_Spread_All",'
        '"Lev_Money_Positions_Long_All","Lev_Money_Positions_Short_All",'
        '"Lev_Money_Positions_Spread_All"\n'
        '"S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE",251230,2025-12-30,'
        "1000,300,100,999,50,20,888,10,40,777\n"
    )
    obs = {o.series_id: o.value for o in parse_tff_csv(text, retrieved_at=RETRIEVED_AT)}
    assert obs["SP500.DEALER_NET"] == 200.0  # 300 - 100, spread 999 ignored
    assert obs["SP500.ASSET_MGR_NET"] == 30.0  # 50 - 20
    assert obs["SP500.LEV_MONEY_NET"] == -30.0  # 10 - 40
    assert obs["SP500.OPEN_INTEREST"] == 1000.0


def test_markets_outside_the_configured_map_are_skipped() -> None:
    """The file carries 107 markets; only the configured handful is in scope.

    An exact-name map rather than a substring match: "E-MINI S&P 500" and
    "MICRO E-MINI S&P 500 INDEX" are different contracts and a loose match
    would silently blend them.
    """
    text = (
        '"Market_and_Exchange_Names","As_of_Date_In_Form_YYMMDD",'
        '"Report_Date_as_YYYY-MM-DD","Open_Interest_All",'
        '"Dealer_Positions_Long_All","Dealer_Positions_Short_All",'
        '"Dealer_Positions_Spread_All","Asset_Mgr_Positions_Long_All",'
        '"Asset_Mgr_Positions_Short_All","Asset_Mgr_Positions_Spread_All",'
        '"Lev_Money_Positions_Long_All","Lev_Money_Positions_Short_All",'
        '"Lev_Money_Positions_Spread_All"\n'
        '"MICRO E-MINI S&P 500 INDEX - CHICAGO MERCANTILE EXCHANGE",251230,2025-12-30,'
        "1,1,1,1,1,1,1,1,1,1\n"
    )
    assert parse_tff_csv(text, retrieved_at=RETRIEVED_AT) == []


def test_missing_required_column_raises() -> None:
    """A changed upstream contract must fail loudly, not return nothing."""
    with pytest.raises(AdapterError, match="upstream contract changed"):
        parse_tff_csv(
            '"Market_and_Exchange_Names","Something"\n"x","y"\n', retrieved_at=RETRIEVED_AT
        )


def test_unparseable_position_is_skipped_not_imputed() -> None:
    """Rule 29: an absent position leg is missing information, not zero."""
    text = (
        '"Market_and_Exchange_Names","As_of_Date_In_Form_YYMMDD",'
        '"Report_Date_as_YYYY-MM-DD","Open_Interest_All",'
        '"Dealer_Positions_Long_All","Dealer_Positions_Short_All",'
        '"Dealer_Positions_Spread_All","Asset_Mgr_Positions_Long_All",'
        '"Asset_Mgr_Positions_Short_All","Asset_Mgr_Positions_Spread_All",'
        '"Lev_Money_Positions_Long_All","Lev_Money_Positions_Short_All",'
        '"Lev_Money_Positions_Spread_All"\n'
        '"S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE",251230,2025-12-30,'
        "1000,300,,0,50,20,0,10,40,0\n"
    )
    ids = {o.series_id for o in parse_tff_csv(text, retrieved_at=RETRIEVED_AT)}
    assert "SP500.DEALER_NET" not in ids, "a blank short leg must not become a zero net"
    assert "SP500.ASSET_MGR_NET" in ids
    assert "SP500.OPEN_INTEREST" in ids


def test_provenance_and_frequency_are_populated() -> None:
    o = parse_tff_csv(load(), retrieved_at=RETRIEVED_AT)[0]
    assert o.source == "cftc"
    assert o.source_version == "cftc_tff_annual_txt"
    assert o.unit == "contracts"
    assert o.frequency == "weekly"
    assert o.retrieved_at == RETRIEVED_AT


def test_series_id_helper_is_stable() -> None:
    assert series_id_for("sp500", "lev_money_net") == "SP500.LEV_MONEY_NET"
    assert series_id_for("EURFX", "open_interest") == "EURFX.OPEN_INTEREST"
