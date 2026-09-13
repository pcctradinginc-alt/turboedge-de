"""Encrypted state persistence for TurboEdge-DE's public GitHub repository.

The repo (and therefore its GitHub Actions logs, job summaries and
artifacts) is public. Two consequences this package exists to address:

1. The state directory (DuckDB + snapshot/registry/ledger/trial files) must
   survive between scheduled runs without ever landing on disk, in an
   artifact, or in a log as plaintext that could reveal an ACTIONABLE trade
   suggestion before the user reads their email. :mod:`turboedge.state.crypto`
   provides authenticated encryption (scrypt + AES-256-GCM) for that;
   :mod:`turboedge.state.archive` packs/unpacks the state directory as an
   encrypted, integrity-checked tar.gz using it.
2. The GitHub Actions cache used previously (``scan-report.yml``) expires
   after 7 days of inactivity; the encrypted archive is instead carried as a
   90-day-retention workflow artifact, restored by ``pipeline.yml`` at the
   start of every run (see ``turboedge state pack``/``unpack`` in
   :mod:`turboedge.cli_state`).

:mod:`turboedge.state.retention` implements ``turboedge db compact``, which
keeps ``product_snapshots`` from growing unbounded over the life of the
public repo's always-on scheduler.
"""

from __future__ import annotations
