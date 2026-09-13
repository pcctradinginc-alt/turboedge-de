"""Optional live tests for the gettex (Boerse Muenchen) adapter.

Real network access, real (polite, rate-limited) HTTP requests to
``gettex.wsd.com``. Marked ``@pytest.mark.live`` and deselected by default
(``pyproject.toml``: ``addopts = "-m 'not live'"``).

Run explicitly with::

    uv run pytest -q -m live tests/adapters/test_gettex_live.py

These are deliberately light (one or two requests, honoring
``min_interval_s`` pacing) -- see ``docs/data_sources.md`` and
``src/turboedge/adapters/gettex.py`` for the research/derivation context.
"""

from __future__ import annotations

import pytest

from turboedge.adapters.base import HttpClient
from turboedge.adapters.gettex import _GETTEX_DEFAULT_BASE_URL, GettexAdapter
from turboedge.storage.schemas import Direction, HealthStatus

_HONEST_UA = "TurboEdge-DE-Research/0.1 (+https://github.com/pcctradinginc-alt/turboedge-de)"


@pytest.mark.live
def test_gettex_healthcheck_live() -> None:
    http = HttpClient(user_agent=_HONEST_UA, min_interval_s=1.5, timeout_s=20.0)
    adapter = GettexAdapter(http, base_url=_GETTEX_DEFAULT_BASE_URL)
    result = adapter.healthcheck()
    assert result.status in (HealthStatus.PASS, HealthStatus.WARN)
    assert result.ok is True


@pytest.mark.live
def test_gettex_fetch_products_live_dax_derives_ratio() -> None:
    http = HttpClient(user_agent=_HONEST_UA, min_interval_s=1.5, timeout_s=20.0)
    adapter = GettexAdapter(http, base_url=_GETTEX_DEFAULT_BASE_URL, max_pages=3, rows_per_page=100)
    snapshots = adapter.fetch_products(["DAX"])

    # A live pull realistically derives a ratio for only a subset of rows
    # (most are bid-only or fail the strict verification gate) -- assert the
    # pipeline actually produces *some* usable output, not an exact count.
    assert len(snapshots) > 0
    assert all(s.isin for s in snapshots)
    assert all(s.ratio > 0 for s in snapshots)
    assert all(s.currency == "EUR" for s in snapshots)
    directions = {s.direction for s in snapshots}
    assert directions.issubset({Direction.LONG, Direction.SHORT})
