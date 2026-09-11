from __future__ import annotations

import structlog

from turboedge.logging import configure_logging


def test_configure_logging_console_does_not_raise() -> None:
    configure_logging("console")
    logger = structlog.get_logger("test")
    logger.info("hello", key="value")


def test_configure_logging_json_does_not_raise() -> None:
    configure_logging("json")
    logger = structlog.get_logger("test")
    logger.info("hello", key="value")


def test_configure_logging_env_resolution() -> None:
    import os

    os.environ["TURBOEDGE_LOG_FORMAT"] = "json"
    try:
        configure_logging()  # should not raise, resolves from env
    finally:
        del os.environ["TURBOEDGE_LOG_FORMAT"]
