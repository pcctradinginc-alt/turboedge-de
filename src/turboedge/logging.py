"""structlog configuration shared by the CLI and every module.

Two renderers are supported: a human-readable console renderer for
interactive use, and a JSON renderer for machine aggregation (CI logs, cron
jobs, log shippers). All log records carry an ISO-8601 UTC timestamp.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import MutableMapping
from typing import Any, Literal

import structlog

LogFormat = Literal["console", "json"]

_VALID_FORMATS: frozenset[str] = frozenset({"console", "json"})

# Public-repo log hygiene (CLAUDE.md rule 18 / Build Contract W3): every
# GitHub Actions log, job summary and artifact on a public repo is visible
# to anyone. When TURBOEDGE_PUBLIC_LOGS is truthy, `redact_public_fields`
# (registered below) masks these keys wherever they appear in a structlog
# event -- ISIN/WKN identify a specific candidate, the rest are prices/
# levels that could reveal an ACTIONABLE trade suggestion before the user
# reads it by email.
_PUBLIC_LOG_ENV = "TURBOEDGE_PUBLIC_LOGS"
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_REDACTED_FIELDS: frozenset[str] = frozenset(
    {
        "isin",
        "wkn",
        "bid",
        "ask",
        "entry_ask",
        "entry_bid",
        "price",
        "example_isins",
        "underlying_price_ref",
    }
)
_REDACTED_SCALAR = "***redacted***"


def redact_enabled() -> bool:
    """``True`` when ``TURBOEDGE_PUBLIC_LOGS`` is set to a truthy value.

    Convention shared across the CLI (``turboedge.reporting.redaction.
    redact_console_enabled`` delegates to this) and every scheduled
    ``pipeline.yml`` job, which sets ``TURBOEDGE_PUBLIC_LOGS=1``: candidate
    ISIN/WKN/prices must never appear in a public log or console table, only
    in the (private) email report. Absent or any other value -> ``False``
    (local/interactive use keeps full detail).
    """
    return os.environ.get(_PUBLIC_LOG_ENV, "").strip().lower() in _TRUTHY


def redact_public_fields(
    logger: object, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: mask :data:`_REDACTED_FIELDS` when
    :func:`redact_enabled` is true. Checked per log call (not cached at
    ``configure_logging()`` time), so toggling the env var at runtime works.
    A list/tuple/set value (e.g. ``example_isins``) is replaced by a count
    rather than the fixed scalar placeholder, since "how many" is itself
    useful diagnostic information that carries no product detail.
    """
    if not redact_enabled():
        return event_dict
    for key in _REDACTED_FIELDS:
        if key not in event_dict:
            continue
        value = event_dict[key]
        if isinstance(value, list | tuple | set):
            event_dict[key] = f"***redacted:{len(value)} item(s)***"
        else:
            event_dict[key] = _REDACTED_SCALAR
    return event_dict


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
        redact_public_fields,
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
