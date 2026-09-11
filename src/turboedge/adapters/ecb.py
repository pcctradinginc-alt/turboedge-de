"""ECB Data API adapter for the euro short-term rate (EUR Short-Term Rate, EST).

Used as the risk-free reference rate ``r`` in financing spread inference
(``pricing/financing.py``: ``s = (F1/F0 - 1) * 360/dt - r`` for longs). The
ECB publishes EST as a percentage (e.g. ``3.15``); this adapter converts to a
decimal (``0.0315``) so downstream pricing code never has to remember which
convention a given number uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from turboedge.adapters.base import (
    AdapterError,
    AdapterMetadata,
    HealthCheckResult,
    HttpClient,
)
from turboedge.storage.schemas import HealthStatus

_SERIES_PATH = "/service/data/EST/B.EU000A2X2A25.WT"
_PARSER_VERSION = "1"
_SOURCE_NAME = "ecb_estr"

# Errors expected from a malformed/unexpected upstream payload; anything else
# is a programming error and should propagate.
_PARSE_ERROR_TYPES = (ValueError, KeyError, TypeError, IndexError)


@dataclass(frozen=True)
class EstrObservation:
    """One EST observation, already converted from percent to decimal."""

    period: date
    value: float  # decimal, e.g. 0.0315 for 3.15%
    retrieved_at: datetime


def _parse_sdmx_json(raw: Any, *, retrieved_at: datetime) -> list[EstrObservation]:
    """Parse an ECB Data API SDMX-JSON response into EstrObservation records.

    Expected shape (simplified)::

        {
          "dataSets": [{"series": {"0:0:0:0:0": {"observations": {"0": [3.15], "1": [3.14]}}}}],
          "structure": {"dimensions": {"observation": [{"values": [{"id": "2026-09-08"}, ...]}]}}
        }

    The observation index (the "0", "1", ... keys) maps positionally into
    ``structure.dimensions.observation[0].values`` to recover each
    observation's calendar date.
    """
    if not isinstance(raw, dict):
        raise ValueError("ECB response is not a JSON object")

    try:
        data_sets = raw["dataSets"]
        obs_dimension_values = raw["structure"]["dimensions"]["observation"][0]["values"]
    except _PARSE_ERROR_TYPES as exc:
        raise ValueError(f"unexpected ECB SDMX-JSON structure: missing {exc}") from exc

    if not data_sets:
        return []
    series_map = data_sets[0].get("series") if isinstance(data_sets[0], dict) else None
    if not series_map:
        return []
    series = next(iter(series_map.values()))
    observations = series.get("observations", {}) if isinstance(series, dict) else {}

    results: list[EstrObservation] = []
    for index_str, values in observations.items():
        try:
            index = int(index_str)
            period_label = obs_dimension_values[index]["id"]
            period = date.fromisoformat(period_label)
            raw_value = values[0]
        except _PARSE_ERROR_TYPES as exc:
            raise ValueError(f"malformed ECB observation at index {index_str!r}: {exc}") from exc
        if raw_value is None:
            continue
        results.append(
            EstrObservation(
                period=period, value=float(raw_value) / 100.0, retrieved_at=retrieved_at
            )
        )
    return results


class EcbEstrAdapter:
    """Fetches the latest EST (euro short-term rate) from the ECB Data API.

    Falls back to a configured constant (``fallback_rate``, typically
    ``risk.yaml: reference_rate_fallback``) whenever the API is unreachable or
    returns something unparsable: the reference rate feeds financing-spread
    inference, which must never crash a scan just because the ECB endpoint
    happens to be down (CLAUDE.md rule 29: never silently impute critical
    data - here we degrade to an explicit, configured fallback instead).
    """

    def __init__(
        self,
        http: HttpClient,
        *,
        base_url: str = "https://data-api.ecb.europa.eu",
        fallback_rate: float = 0.0,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._fallback_rate = fallback_rate

    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=_SOURCE_NAME,
            kind="reference_rate",
            version=_PARSER_VERSION,
            homepage="https://data.ecb.europa.eu/",
        )

    def fetch(self, *, last_n_observations: int = 5) -> Any:
        url = f"{self._base_url}{_SERIES_PATH}"
        return self._http.get_json(
            url,
            params={"lastNObservations": last_n_observations, "format": "jsondata"},
            headers={"Accept": "application/json"},
        )

    def normalize(self, raw: Any, **_kwargs: Any) -> list[EstrObservation]:
        return _parse_sdmx_json(raw, retrieved_at=datetime.now(UTC))

    def fetch_latest(self) -> EstrObservation | None:
        """Fetch and return the most recent EST observation, or ``None`` on failure.

        Never raises: any network or parse failure is treated as "unknown"
        so callers (including :meth:`get_estr`) can fall back cleanly.
        """
        try:
            raw = self.fetch(last_n_observations=5)
            observations = self.normalize(raw)
        except (AdapterError, *_PARSE_ERROR_TYPES):
            return None
        if not observations:
            return None
        return max(observations, key=lambda obs: obs.period)

    def get_estr(self) -> float:
        """Latest EST as a decimal, or ``fallback_rate`` if it cannot be fetched."""
        latest = self.fetch_latest()
        return latest.value if latest is not None else self._fallback_rate

    def healthcheck(self) -> HealthCheckResult:
        checked_at = datetime.now(UTC)
        start = time.monotonic()
        try:
            latest = self.fetch_latest()
        except Exception as exc:  # defensive: healthcheck must never raise
            return HealthCheckResult(
                source=_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=None,
                checked_at=checked_at,
                message=f"unexpected error: {exc}",
            )
        latency_ms = (time.monotonic() - start) * 1000
        if latest is None:
            return HealthCheckResult(
                source=_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=latency_ms,
                checked_at=checked_at,
                message="no EST observation could be fetched or parsed",
            )
        return HealthCheckResult(
            source=_SOURCE_NAME,
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=latency_ms,
            checked_at=checked_at,
            message=f"latest EST {latest.value:.4%} as of {latest.period.isoformat()}",
        )
