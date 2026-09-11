"""Scan and universe pipeline orchestration.

``pipeline.universe`` fetches, merges and persists the product universe;
``pipeline.scan`` runs the full signal -> quotes -> pricing -> gates scan
(Master Spec §43, Build Contract "Scan-Pipeline"). Re-exported here for
convenience so callers (the CLI, tests) can do
``from turboedge.pipeline import run_scan, run_universe`` instead of
reaching into the submodules directly.
"""

from __future__ import annotations

from turboedge.pipeline.scan import EstrSource, PriceSource, ScanOptions, ScanResult, run_scan
from turboedge.pipeline.universe import NoProductsError, UniverseResult, run_universe

__all__ = [
    "EstrSource",
    "NoProductsError",
    "PriceSource",
    "ScanOptions",
    "ScanResult",
    "UniverseResult",
    "run_scan",
    "run_universe",
]
