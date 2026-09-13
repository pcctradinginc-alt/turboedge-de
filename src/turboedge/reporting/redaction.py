"""CLI console-output redaction gate for public-repo log hygiene.

Deliberately a one-function module rather than a change to
``reporting/console.py`` (owned by another workstream, outside this
assignment): ``cli.py`` guards its own render call sites with
:func:`redact_console_enabled`, printing only counts when it is true instead
of the full candidate table -- see ``scan_cmd``/``universe_cmd`` in
``cli.py``. The full, unredacted candidate detail still reaches the user via
the (private) Gmail report either way.
"""

from __future__ import annotations

from turboedge.logging import redact_enabled

__all__ = ["redact_console_enabled"]


def redact_console_enabled() -> bool:
    """``True`` when candidate/report console tables must be suppressed to
    counts-only (``TURBOEDGE_PUBLIC_LOGS`` is truthy). Delegates to
    :func:`turboedge.logging.redact_enabled` -- one convention, one place
    reading the environment variable."""
    return redact_enabled()
