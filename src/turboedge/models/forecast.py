"""Forecast Engine top-level types: ``HorizonForecast``, ``ForecastModel``, ``HORIZONS``.

Build Contract v2, W4 interface. ``HorizonForecast`` is the canonical output
of every ``ForecastModel`` for one ``(underlying_id, horizon_days)`` pair, as
of a fixed ``prediction_time``; ``ForecastModel`` is the protocol every
concrete model (``models/directional.py``, ``models/ensemble.py``)
implements. ``HORIZONS`` is the primary horizon ladder used everywhere in
this milestone (Master Spec §3.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from turboedge.storage.schemas import UnderlyingBar

HORIZONS: tuple[int, ...] = (3, 5, 7, 10, 14)

_QUANTILE_KEYS = frozenset({"q05", "q25", "q50", "q75", "q95"})


@dataclass(frozen=True, slots=True)
class HorizonForecast:
    """A full predictive distribution for one underlying/horizon, as of ``prediction_time``."""

    underlying_id: str
    horizon_days: int
    prediction_time: datetime
    p_up: float  # calibrated P(h-day log return > 0)
    mean: float  # E[h-day log return]
    sigma: float  # std of h-day log return
    quantiles: dict[str, float]  # keys "q05","q25","q50","q75","q95"
    expected_shortfall_05: float  # E[r | r <= q05]
    uncertainty: float  # standard error of `mean` (model/estimation uncertainty), >= 0
    model_id: str
    model_hash: str
    signal_family: str  # e.g. "tsmom", "trend_vol", "logit"
    n_train: int
    n_effective: float  # average-uniqueness adjusted sample size

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_up <= 1.0):
            raise ValueError(f"p_up must be within [0, 1], got {self.p_up!r}")
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma!r}")
        if self.uncertainty < 0.0:
            raise ValueError(f"uncertainty must be >= 0, got {self.uncertainty!r}")
        if self.horizon_days <= 0:
            raise ValueError(f"horizon_days must be > 0, got {self.horizon_days!r}")
        if self.n_train < 0:
            raise ValueError(f"n_train must be >= 0, got {self.n_train!r}")
        if self.n_effective < 0.0:
            raise ValueError(f"n_effective must be >= 0, got {self.n_effective!r}")
        if set(self.quantiles) != _QUANTILE_KEYS:
            raise ValueError(
                f"quantiles must have exactly keys {sorted(_QUANTILE_KEYS)}, "
                f"got {sorted(self.quantiles)}"
            )


@runtime_checkable
class ForecastModel(Protocol):
    model_id: str
    signal_family: str

    def fit(self, bars: Sequence[UnderlyingBar], as_of: datetime) -> None: ...

    def predict(
        self,
        bars: Sequence[UnderlyingBar],
        as_of: datetime,
        horizons: Sequence[int] = HORIZONS,
    ) -> list[HorizonForecast]: ...

    def model_hash(self) -> str: ...


def build_default_models() -> list[ForecastModel]:
    """The protected TSMOM-mapped, logistic and null baseline models (Master Spec §3.5, §9.1).

    Imported lazily: ``models.directional`` needs ``HORIZONS``/``HorizonForecast``
    from this module, so importing it eagerly at module load time would be
    circular.
    """
    from turboedge.models.directional import (
        LogisticDirectionModel,
        NullModel,
        TsmomForecastModel,
    )

    models: list[ForecastModel] = [TsmomForecastModel(), LogisticDirectionModel(), NullModel()]
    return models
