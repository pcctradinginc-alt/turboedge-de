"""structlog configuration shared by the CLI and every module.

Two renderers are supported: a human-readable console renderer for
interactive use, and a JSON renderer for machine aggregation (CI logs, cron
jobs, log shippers). All log records carry an ISO-8601 UTC timestamp.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal

import structlog

LogFormat = Literal["console", "json"]

_VALID_FORMATS: frozenset[str] = frozenset({"console", "json"})


def _resolve_format(fmt: LogFormat | None) -> LogFormat:
    if fmt is not None:
        return fmt
    env_fmt = os.environ.get("TURBOEDGE_LOG_FORMAT")
    if env_fmt in _VALID_FORMATS:
        return "json" if env_fmt == "json" else "console"
    if not sys.stderr.isatty():
        return "json"
    return "console"


def configure_logging(fmt: LogFormat | None = None, *, level: str | None = None) -> None:
    """Configure structlog (and stdlib logging as its backend) for the process.

    Args:
        fmt: Explicit output format. When ``None``, resolution falls back to
            the ``TURBOEDGE_LOG_FORMAT`` environment variable, then to "json"
            when stderr is not a TTY (piped output, CI, cron), else "console".
        level: Minimum log level name (e.g. "DEBUG", "INFO"). Falls back to
            ``TURBOEDGE_LOG_LEVEL``, then "INFO".

    Safe to call multiple times (e.g. once per CLI invocation); each call
    fully replaces the previous structlog configuration.
    """
    resolved = _resolve_format(fmt)
    level_name = (level or os.environ.get("TURBOEDGE_LOG_LEVEL") or "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)
    if not isinstance(log_level, int):
        log_level = logging.INFO

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.types.Processor
    if resolved == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        # `cache_logger_on_first_use=True` would freeze each module-level
        # logger's underlying PrintLogger -- and the `sys.stderr` object
        # reference baked into it -- at whatever it was the FIRST time that
        # module ever logged anything, for the rest of the process. A
        # single CLI invocation never notices (its `sys.stderr` never
        # changes mid-run), but two invocations of the Typer app object in
        # the same process (typer.testing.CliRunner across tests, or any
        # other embedding) do: `configure_logging()` runs again per
        # invocation (AppContext.__init__), but an already-cached logger
        # would keep writing to the FIRST invocation's (by then closed)
        # stream, raising "I/O operation on closed file." False re-resolves
        # the live config (and thus the live `sys.stderr`) on every call --
        # a negligible cost for a CLI tool's log volume.
        cache_logger_on_first_use=False,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=log_level, force=True)
