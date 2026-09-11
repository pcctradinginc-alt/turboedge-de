"""Tests for the adapter registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from turboedge.adapters.base import AdapterMetadata, HealthCheckResult, HttpClient
from turboedge.adapters.registry import (
    PRODUCT_ADAPTER_FACTORIES,
    build_http_client,
    build_product_adapters,
    build_reference_healthchecks,
    register_product_adapter,
)
from turboedge.config import load_config
from turboedge.storage.schemas import HealthStatus


@pytest.fixture
def config_dir() -> Path:
    """Return the real config directory from the project."""
    return Path(__file__).parent.parent.parent / "configs"


@pytest.fixture
def base_config(config_dir: Path):
    """Load the actual project config."""
    return load_config(config_dir)


@pytest.fixture
def clean_registry() -> None:
    """Clear the registry before and after tests."""
    original = PRODUCT_ADAPTER_FACTORIES.copy()
    PRODUCT_ADAPTER_FACTORIES.clear()
    yield
    PRODUCT_ADAPTER_FACTORIES.clear()
    PRODUCT_ADAPTER_FACTORIES.update(original)


def test_register_product_adapter(clean_registry: None) -> None:
    """Test registering a product adapter."""

    class FakeAdapter:
        @property
        def name(self) -> str:
            return "fake"

        def fetch_products(self, underlying_ids):
            return []

        def healthcheck(self):
            return HealthCheckResult(
                source="fake",
                status=HealthStatus.PASS,
                ok=True,
                latency_ms=10.0,
                checked_at=None,
                message="ok",
            )

        def metadata(self):
            return AdapterMetadata(name="fake", kind="product", version="1.0")

    def fake_factory(src, http):
        return FakeAdapter()

    register_product_adapter("fake", fake_factory)
    assert "fake" in PRODUCT_ADAPTER_FACTORIES
    assert PRODUCT_ADAPTER_FACTORIES["fake"] is fake_factory


def test_register_product_adapter_duplicate_error(clean_registry: None) -> None:
    """Test that duplicate registration raises ValueError."""

    def factory1(src, http):
        pass

    def factory2(src, http):
        pass

    register_product_adapter("test", factory1)
    with pytest.raises(ValueError, match="already registered"):
        register_product_adapter("test", factory2)


def test_build_http_client(base_config) -> None:
    """Test building an HTTP client from config."""
    src_cfg = base_config.sources["ecb"]
    client = build_http_client(src_cfg)

    assert isinstance(client, HttpClient)
    assert client.user_agent == src_cfg.user_agent
    client.close()


def test_build_product_adapters_empty_registry(clean_registry: None, base_config) -> None:
    """Test build_product_adapters with empty registry."""
    adapters = build_product_adapters(base_config)
    assert adapters == []


def test_build_product_adapters_with_registered_adapter(clean_registry: None, base_config) -> None:
    """Test build_product_adapters with a registered adapter."""

    class FakeAdapter:
        def __init__(self, src, http):
            self._name = src.user_agent

        @property
        def name(self) -> str:
            return "test_adapter"

        def fetch_products(self, underlying_ids):
            return []

        def healthcheck(self):
            return HealthCheckResult(
                source="test",
                status=HealthStatus.PASS,
                ok=True,
                latency_ms=1.0,
                checked_at=None,
                message="ok",
            )

        def metadata(self):
            return AdapterMetadata(name="test", kind="product", version="1.0")

    def factory(src, http):
        return FakeAdapter(src, http)

    register_product_adapter("ecb", factory)

    adapters = build_product_adapters(base_config)
    # ecb should be instantiated if enabled
    if base_config.sources["ecb"].enabled:
        assert len(adapters) >= 1


def test_build_product_adapters_only_filter(clean_registry: None, base_config) -> None:
    """Test build_product_adapters with 'only' filter."""

    class FakeAdapter:
        def __init__(self, src, http):
            pass

        @property
        def name(self) -> str:
            return "test"

        def fetch_products(self, underlying_ids):
            return []

        def healthcheck(self):
            return HealthCheckResult(
                source="test",
                status=HealthStatus.PASS,
                ok=True,
                latency_ms=1.0,
                checked_at=None,
                message="ok",
            )

        def metadata(self):
            return AdapterMetadata(name="test", kind="product", version="1.0")

    def factory(src, http):
        return FakeAdapter(src, http)

    register_product_adapter("deutsche_boerse", factory)

    # With only filter
    adapters = build_product_adapters(base_config, only="deutsche_boerse")
    # Result depends on whether deutsche_boerse is enabled
    for adapter in adapters:
        assert adapter.name == "test"


def test_build_product_adapters_only_unknown_raises(clean_registry: None, base_config) -> None:
    """Test that unknown 'only' value raises ValueError."""
    with pytest.raises(ValueError, match="unknown product adapter"):
        build_product_adapters(base_config, only="nonexistent")


def test_build_reference_healthchecks(base_config) -> None:
    """Test building reference rate healthchecks."""
    checks = build_reference_healthchecks(base_config)

    # Should include ECB and yfinance if enabled
    assert isinstance(checks, list)
    # Each check should be callable
    for check in checks:
        assert callable(check)


def test_build_reference_healthchecks_ecb_enabled(base_config) -> None:
    """Test that ECB healthcheck is included if enabled."""
    if base_config.sources.get("ecb", None) and base_config.sources["ecb"].enabled:
        checks = build_reference_healthchecks(base_config)
        assert len(checks) > 0  # At least ECB or yfinance should be present


def test_build_reference_healthchecks_yfinance_enabled(base_config) -> None:
    """Test that yfinance healthcheck is included if enabled."""
    if base_config.sources.get("yfinance", None) and base_config.sources["yfinance"].enabled:
        checks = build_reference_healthchecks(base_config)
        assert len(checks) > 0  # At least yfinance should be present
