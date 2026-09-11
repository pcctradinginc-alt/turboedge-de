"""TurboEdge-DE - research-only cost/edge analysis for German turbo certificates.

This package never places orders, never talks to a broker, and never executes
trades. Every entry point produces research output (console, DuckDB/Parquet,
Gmail reports) for a human to act on manually.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
