"""Optional live tests for the BNP Paribas / Citi issuer-feed adapters.

Real network access, real (polite, rate-limited) HTTP requests to
``derivate.bnpparibas.com`` and ``de.citifirst.com``. Marked ``@pytest.mark.live``
and deselected by default (``pyproject.toml``: ``addopts = "-m 'not live'"``).

Run explicitly with::

    uv run pytest -q -m live tests/adapters/test_issuer_feeds_live.py

These are deliberately light (one or two requests per adapter, honoring
``min_interval_s`` pacing) -- see ``docs/data_sources.md`` and
``src/turboedge/adapters/issuer_feeds.py`` for the research/pitfalls context.
"""

from __future__ import annotations

import pytest

from turboedge.adapters.base import HttpClient
from turboedge.adapters.issuer_feeds import (
    _BNP_DEFAULT_BASE_URL,
    _CITI_DEFAULT_BASE_URL,
    BnpParibasTurboAdapter,
    CitiFirstTurboAdapter,
)
from turboedge.storage.schemas import HealthStatus

_HONEST_UA = "TurboEdge-DE-Research/0.1 (+https://github.com/pcctradinginc-alt/turboedge-de)"


@pytest.mark.live
def test_bnp_healthcheck_live() -> None:
    http = HttpClient(user_agent=_HONEST_UA, min_interval_s=1.5, timeout_s=20.0)
    adapter = BnpParibasTurboAdapter(http, base_url=_BNP_DEFAULT_BASE_URL)
    result = adapter.healthcheck()
    assert result.status in (HealthStatus.PASS, HealthStatus.WARN)
    assert result.ok is True


@pytest.mark.live
def test_bnp_fetch_products_live_dax() -> None:
    http = HttpClient(user_agent=_HONEST_UA, min_interval_s=1.5, timeout_s=20.0)
    adapter = BnpParibasTurboAdapter(
        http, base_url=_BNP_DEFAULT_BASE_URL, max_pages=1, page_size=50
    )
    snapshots = adapter.fetch_products(["DAX"])
    assert len(snapshots) > 0
    assert all(s.isin for s in snapshots)
    assert all(s.ratio > 0 for s in snapshots)


@pytest.mark.live
def test_citi_healthcheck_live() -> None:
    http = HttpClient(user_agent=_HONEST_UA, min_interval_s=1.5, timeout_s=20.0)
    adapter = CitiFirstTurboAdapter(http, base_url=_CITI_DEFAULT_BASE_URL)
    result = adapter.healthcheck()
    assert result.status in (HealthStatus.PASS, HealthStatus.WARN)
    assert result.ok is True


@pytest.mark.live
def test_citi_fetch_products_live_dax() -> None:
    http = HttpClient(user_agent=_HONEST_UA, min_interval_s=1.5, timeout_s=20.0)
    adapter = CitiFirstTurboAdapter(http, base_url=_CITI_DEFAULT_BASE_URL)
    snapshots = adapter.fetch_products(["DAX"])
    assert len(snapshots) > 0
    assert all(s.isin for s in snapshots)
    assert all(s.ratio > 0 for s in snapshots)
