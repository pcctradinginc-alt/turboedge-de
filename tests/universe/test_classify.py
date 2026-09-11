from __future__ import annotations

from datetime import date

import pytest

from turboedge.storage.schemas import Direction, ProductType
from turboedge.universe.classify import classify_direction, classify_product_type


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("Turbo Long auf DAX", Direction.LONG),
        ("Call Optionsschein", Direction.LONG),
        ("Bullish Zertifikat", Direction.LONG),
        ("Turbo Short auf DAX", Direction.SHORT),
        ("Put Optionsschein", Direction.SHORT),
        ("Bearish Zertifikat", Direction.SHORT),
        ("long", Direction.LONG),
        ("SHORT", Direction.SHORT),
    ],
)
def test_classify_direction_known_cases(raw_text: str, expected: Direction) -> None:
    assert classify_direction(raw_text) == expected


@pytest.mark.parametrize(
    "raw_text",
    [
        "",
        "Bull/Bear Zertifikat",  # both keywords present -> ambiguous
        "Discount Zertifikat",  # neither keyword
    ],
)
def test_classify_direction_ambiguous_or_unknown_returns_none(raw_text: str) -> None:
    assert classify_direction(raw_text) is None


def test_classify_product_type_open_end_turbo() -> None:
    result = classify_product_type(
        type_text="Turbo Open End",
        financing_level=18000.0,
        knockout_barrier=18000.0,
        open_end=True,
        maturity=None,
    )
    assert result == ProductType.TURBO_OPEN_END


def test_classify_product_type_classic_turbo_with_maturity() -> None:
    result = classify_product_type(
        type_text="Turbo Classic",
        financing_level=18000.0,
        knockout_barrier=18000.0,
        open_end=False,
        maturity=date(2026, 12, 19),
    )
    assert result == ProductType.TURBO_CLASSIC


def test_classify_product_type_mini_future_by_barrier_buffer() -> None:
    result = classify_product_type(
        type_text="Turbo",
        financing_level=18000.0,
        knockout_barrier=18200.0,  # buffer above financing level
        open_end=True,
        maturity=None,
    )
    assert result == ProductType.MINI_FUTURE


@pytest.mark.parametrize(
    "label",
    ["Mini Future", "Smart Turbo", "Turbo Pro", "mini-future"],
)
def test_classify_product_type_mini_future_by_name(label: str) -> None:
    result = classify_product_type(
        type_text=label,
        financing_level=18000.0,
        knockout_barrier=18000.0,  # would otherwise look like open-end
        open_end=True,
        maturity=None,
    )
    assert result == ProductType.MINI_FUTURE


def test_classify_product_type_unknown_when_no_signal() -> None:
    result = classify_product_type(
        type_text=None,
        financing_level=None,
        knockout_barrier=None,
        open_end=None,
        maturity=None,
    )
    assert result == ProductType.UNKNOWN


def test_classify_product_type_infers_classic_from_maturity_when_open_end_unknown() -> None:
    result = classify_product_type(
        type_text="Zertifikat",
        financing_level=18000.0,
        knockout_barrier=18200.0,
        open_end=None,
        maturity=date(2026, 12, 19),
    )
    assert result == ProductType.TURBO_CLASSIC


def test_classify_product_type_infers_open_end_from_barrier_equality_unknown_flag() -> None:
    result = classify_product_type(
        type_text="Zertifikat",
        financing_level=18000.0,
        knockout_barrier=18000.0,
        open_end=None,
        maturity=None,
    )
    assert result == ProductType.TURBO_OPEN_END
