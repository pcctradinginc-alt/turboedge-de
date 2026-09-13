from __future__ import annotations

import os

import pytest
import structlog

from turboedge.logging import configure_logging, redact_enabled, redact_public_fields


def test_configure_logging_console_does_not_raise() -> None:
    configure_logging("console")
    logger = structlog.get_logger("test")
    logger.info("hello", key="value")


def test_configure_logging_json_does_not_raise() -> None:
    configure_logging("json")
    logger = structlog.get_logger("test")
    logger.info("hello", key="value")


def test_configure_logging_env_resolution() -> None:
    os.environ["TURBOEDGE_LOG_FORMAT"] = "json"
    try:
        configure_logging()  # should not raise, resolves from env
    finally:
        del os.environ["TURBOEDGE_LOG_FORMAT"]


# -- redact_enabled / redact_public_fields (Build Contract W3) --------------


@pytest.fixture(autouse=True)
def _clean_public_logs_env() -> None:
    """Every test in this module starts and ends with TURBOEDGE_PUBLIC_LOGS
    unset, regardless of what an individual test does to it."""
    os.environ.pop("TURBOEDGE_PUBLIC_LOGS", None)
    yield
    os.environ.pop("TURBOEDGE_PUBLIC_LOGS", None)


@pytest.mark.parametrize("value", ["1", "true", "True", "YES", "on"])
def test_redact_enabled_truthy_values(value: str) -> None:
    os.environ["TURBOEDGE_PUBLIC_LOGS"] = value
    assert redact_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "garbage"])
def test_redact_enabled_falsy_values(value: str) -> None:
    os.environ["TURBOEDGE_PUBLIC_LOGS"] = value
    assert redact_enabled() is False


def test_redact_enabled_unset_is_false() -> None:
    assert redact_enabled() is False


def test_redact_public_fields_noop_when_disabled() -> None:
    event = {"isin": "DE000ABC1234", "bid": 4.80, "msg": "scan complete"}
    result = redact_public_fields(None, "info", dict(event))
    assert result == event


def test_redact_public_fields_masks_configured_keys_when_enabled() -> None:
    os.environ["TURBOEDGE_PUBLIC_LOGS"] = "1"
    event = {
        "isin": "DE000ABC1234",
        "wkn": "ABC123",
        "bid": 4.80,
        "ask": 4.86,
        "entry_ask": 4.90,
        "entry_bid": 4.70,
        "price": 100.5,
        "underlying_price_ref": 18500.0,
        "example_isins": ["DE0001", "DE0002", "DE0003"],
        "msg": "scan complete",
        "run_id": "20260911T120000Z-abc123",
    }
    result = redact_public_fields(None, "info", dict(event))

    for key in (
        "isin",
        "wkn",
        "bid",
        "ask",
        "entry_ask",
        "entry_bid",
        "price",
        "underlying_price_ref",
    ):
        assert result[key] == "***redacted***"

    # A list-valued field is redacted to a count, not the fixed scalar.
    assert result["example_isins"] == "***redacted:3 item(s)***"

    # Fields not in the redaction set survive untouched.
    assert result["msg"] == "scan complete"
    assert result["run_id"] == "20260911T120000Z-abc123"


def test_redact_public_fields_missing_keys_do_not_error() -> None:
    os.environ["TURBOEDGE_PUBLIC_LOGS"] = "1"
    event = {"msg": "no candidate fields here"}
    result = redact_public_fields(None, "info", dict(event))
    assert result == event


def test_configure_logging_with_redaction_enabled_does_not_raise() -> None:
    """The processor is wired into the real pipeline, not just unit-tested
    standalone -- a log call with candidate fields must not raise when
    TURBOEDGE_PUBLIC_LOGS=1 (JSON or console renderer)."""
    os.environ["TURBOEDGE_PUBLIC_LOGS"] = "1"
    configure_logging("json")
    logger = structlog.get_logger("test")
    logger.info("scan_candidate", isin="DE000ABC1234", bid=4.80, ask=4.86)
