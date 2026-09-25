"""Contract tests for the Cboe volatility adapter (W12-A).

Fixtures are real Cboe payloads captured on 2026-09-25 (tail of each
`<INDEX>_History.csv`), covering both shapes the endpoint actually serves:
`DATE,OPEN,HIGH,LOW,CLOSE` for the S&P 500 term points and `DATE,<INDEX>`
for the close-only indices. CLAUDE.md requires contract tests with real
fixtures for every adapter -- a parser validated only against hand-written
input silently stops matching the source the first time the source changes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from turboedge.adapters.base import AdapterError
from turboedge.adapters.cboe import parse_history_csv, series_id_for

FIXTURES = Path(__file__).parent / "fixtures"
RETRIEVED_AT = datetime(2026, 9, 25, 17, 0, tzinfo=UTC)


def load(index: str) -> str:
    return (FIXTURES / f"cboe_{index}_History_sample.csv").read_text(encoding="utf-8")


def test_ohlc_index_yields_four_series_per_day() -> None:
    obs = parse_history_csv(load("VIX"), "VIX", retrieved_at=RETRIEVED_AT)
    assert obs, "real fixture must parse"
    ids = {o.series_id for o in obs}
    assert ids == {"VIX.OPEN", "VIX.HIGH", "VIX.LOW", "VIX.CLOSE"}
    dates = {o.observation_time.date() for o in obs}
    assert len(obs) == len(dates) * 4, "one row per (date, column)"


def test_close_only_index_normalises_to_dot_close() -> None:
    """`DATE,VVIX` must become `VVIX.CLOSE`, not `VVIX.VVIX`.

    Without this a feature builder would have to know which upstream header
    shape each index uses -- exactly the per-source special-casing the
    adapter layer exists to absorb.
    """
    obs = parse_history_csv(load("VVIX"), "VVIX", retrieved_at=RETRIEVED_AT)
    assert obs
    assert {o.series_id for o in obs} == {"VVIX.CLOSE"}


def test_gvz_close_only_parses_too() -> None:
    obs = parse_history_csv(load("GVZ"), "GVZ", retrieved_at=RETRIEVED_AT)
    assert obs
    assert {o.series_id for o in obs} == {"GVZ.CLOSE"}
    assert all(o.value > 0.0 for o in obs), "a volatility index is strictly positive"


def test_available_at_is_the_day_after_observation() -> None:
    """The availability model, pinned.

    A trading day's close is not published while that day still trades, so
    `available_at` must be strictly later than `observation_time` and land
    on the following calendar day at 00:00 UTC. This is what stops the
    feature layer from seeing a close that had not happened yet.
    """
    obs = parse_history_csv(load("VIX"), "VIX", retrieved_at=RETRIEVED_AT)
    for o in obs:
        assert o.available_at > o.observation_time
        assert o.available_at.hour == 0 and o.available_at.minute == 0
        assert (o.available_at.date() - o.observation_time.date()).days == 1


def test_provenance_fields_are_populated() -> None:
    obs = parse_history_csv(load("VIX"), "VIX", retrieved_at=RETRIEVED_AT)
    o = obs[0]
    assert o.source == "cboe"
    assert o.source_version == "cboe_daily_prices_csv"
    assert o.unit == "index_points"
    assert o.frequency == "daily"
    assert o.retrieved_at == RETRIEVED_AT
    assert o.parser_version


def test_us_date_format_is_parsed_not_guessed() -> None:
    """Cboe writes MM/DD/YYYY. Reading it as DD/MM/YYYY would silently
    scramble every date that is valid under both (the first 12 of a month),
    which is the kind of error that never raises and quietly ruins a
    walk-forward split."""
    text = "DATE,OPEN,HIGH,LOW,CLOSE\n03/11/2026,10.0,11.0,9.0,10.5\n"
    obs = parse_history_csv(text, "VIX", retrieved_at=RETRIEVED_AT)
    assert obs[0].observation_time.date().isoformat() == "2026-03-11"


def test_empty_and_unparseable_values_are_skipped_not_imputed() -> None:
    """Rule 29: a missing volatility print is missing information.

    Forward-filling it would manufacture a feature value on a day the index
    did not publish one.
    """
    text = (
        "DATE,OPEN,HIGH,LOW,CLOSE\n"
        "09/21/2026,10.0,11.0,9.0,10.5\n"
        "09/22/2026,,,,\n"
        "09/23/2026,10.0,11.0,9.0,n/a\n"
    )
    obs = parse_history_csv(text, "VIX", retrieved_at=RETRIEVED_AT)
    by_date = {o.observation_time.date().isoformat() for o in obs}
    assert "2026-09-21" in by_date
    assert "2026-09-22" not in by_date, "a blank row must not become a zero"
    closes = [o for o in obs if o.series_id == "VIX.CLOSE"]
    assert {o.observation_time.date().isoformat() for o in closes} == {"2026-09-21"}


def test_missing_date_column_raises() -> None:
    with pytest.raises(AdapterError, match="no DATE column"):
        parse_history_csv("FOO,BAR\n1,2\n", "VIX", retrieved_at=RETRIEVED_AT)


def test_unexpected_header_raises_rather_than_returning_nothing() -> None:
    """A changed upstream contract must fail loudly.

    Returning an empty list would look identical to a quiet market day and
    would let the feature family silently degrade to no data at all.
    """
    with pytest.raises(AdapterError, match="upstream contract changed"):
        parse_history_csv("DATE,SOMETHING_ELSE\n09/21/2026,1.0\n", "VIX", retrieved_at=RETRIEVED_AT)


def test_series_id_helper_is_stable() -> None:
    assert series_id_for("vix", "close") == "VIX.CLOSE"
    assert series_id_for("VVIX", "VVIX") == "VVIX.CLOSE"
    assert series_id_for("VIX", "HIGH") == "VIX.HIGH"
