"""Protected TSMOM baseline signal: ``tsmom_horizon_norm_v1``.

Formula reference: Master Spec §7 ("Protected Baseline Signal").

CLAUDE.md rule 10: this baseline is never deleted and remains the permanent
reference challengers are benchmarked against. Its threshold is versioned
(``TsmomConfig.version``, hashed into ``signal_version_hash``) and must never
be silently re-optimized -- any change to the formula or its parameters is a
new version, tracked through the research governance ledger, not a
mutation of this one.

For lookback ``k``::

    z_k = ln(P_t / P_t-k) / (sigma_t * sqrt(k))

with ``sigma_t`` the EWMA daily volatility of log returns up to and
including day ``t`` (see ``turboedge.features.product.ewma_volatility`` --
using only returns up to ``t`` is what keeps this look-ahead free, CLAUDE.md
rule 4). Each ``z_k`` is clipped to ``[-clip, +clip]`` and the score is their
mean::

    s = mean(z_21, z_63, z_126)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from turboedge.features.product import ewma_volatility
from turboedge.storage.schemas import Direction

_DEFAULT_LOOKBACKS: tuple[int, ...] = (21, 63, 126)


@dataclass(frozen=True, slots=True)
class TsmomConfig:
    """Parameters of the protected ``tsmom_horizon_norm_v1`` baseline."""

    signal_id: str = "tsmom_horizon_norm_v1"
    lookbacks: tuple[int, ...] = field(default=_DEFAULT_LOOKBACKS)
    ewma_lambda: float = 0.94
    clip: float = 3.0
    threshold: float = 0.5
    version: str = "1"


def signal_version_hash(cfg: TsmomConfig) -> str:
    """SHA-256 hex digest over the canonical JSON encoding of ``cfg``.

    Stable across processes/machines for identical config content; persisted
    on every ``SignalSnapshot`` (``storage/schemas.py``) so a prediction can
    always be traced back to the exact signal version that produced it
    (CLAUDE.md rule 33).
    """
    payload = {
        "signal_id": cfg.signal_id,
        "lookbacks": list(cfg.lookbacks),
        "ewma_lambda": cfg.ewma_lambda,
        "clip": cfg.clip,
        "threshold": cfg.threshold,
        "version": cfg.version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TsmomResult:
    """Output of :func:`compute_tsmom`."""

    score: float
    components: dict[str, float]
    direction_hint: Direction | None
    sigma: float


def compute_tsmom(closes: npt.NDArray[np.float64], cfg: TsmomConfig) -> TsmomResult:
    """Compute the protected TSMOM score as of the *last* element of ``closes``.

    ``closes`` must be ordered oldest-to-newest; the score is always
    evaluated "as of today" (``closes[-1]``), which is what makes this
    function inherently look-ahead safe: it has no visibility into any price
    beyond the last one it was given, so appending future prices to
    ``closes`` and re-slicing back to the original length can never change
    the score computed at that earlier point in time.

    Raises:
        ValueError: if ``closes`` has fewer than ``max(cfg.lookbacks) + 1``
            observations, contains NaN or non-positive prices, or the EWMA
            volatility as of the last observation is zero.
    """
    arr = np.asarray(closes, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"closes must be 1-dimensional, got shape {arr.shape!r}")
    if np.any(np.isnan(arr)):
        raise ValueError("closes must not contain NaN")
    if np.any(arr <= 0):
        raise ValueError("closes must be strictly positive")

    max_lookback = max(cfg.lookbacks)
    min_length = max_lookback + 1
    if arr.size < min_length:
        raise ValueError(
            f"need at least {min_length} closes for lookbacks {cfg.lookbacks}, got {arr.size}"
        )

    log_returns = np.diff(np.log(arr))
    sigma_series = ewma_volatility(log_returns, lam=cfg.ewma_lambda)
    sigma_t = float(sigma_series[-1])
    if not (sigma_t > 0):
        raise ValueError("EWMA volatility as of the last observation is zero; cannot normalize")

    components: dict[str, float] = {}
    clipped_zs: list[float] = []
    for k in cfg.lookbacks:
        raw_z = float(np.log(arr[-1] / arr[-1 - k]) / (sigma_t * np.sqrt(k)))
        clipped_z = float(np.clip(raw_z, -cfg.clip, cfg.clip))
        components[f"z_{k}"] = clipped_z
        clipped_zs.append(clipped_z)

    score = float(np.mean(clipped_zs))
    if score >= cfg.threshold:
        direction_hint: Direction | None = Direction.LONG
    elif score <= -cfg.threshold:
        direction_hint = Direction.SHORT
    else:
        direction_hint = None

    return TsmomResult(
        score=score, components=components, direction_hint=direction_hint, sigma=sigma_t
    )
