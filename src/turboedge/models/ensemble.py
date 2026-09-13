"""Combine several models' ``HorizonForecast``s into one ensemble forecast.

Build Contract v2 item 3: weighted mixture of the component predictive
distributions, one ``HorizonForecast`` in, one out, ``model_id="ensemble"``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np

from turboedge.models.forecast import HorizonForecast
from turboedge.models.quantile import QUANTILE_LEVELS

_MONTE_CARLO_SEED = 20260101
_MONTE_CARLO_SAMPLES = 100_000


def combine_forecasts(
    forecasts: Sequence[HorizonForecast], weights: Mapping[str, float]
) -> HorizonForecast:
    """Weighted mixture of ``forecasts`` (all for the same underlying/horizon/prediction_time).

    - ``mean``/``p_up``: weight-normalized average.
    - ``sigma``: mixture standard deviation via the law of total variance,
      ``Var = sum(w_i*sigma_i^2) + sum(w_i*(mean_i - mean)^2)`` (within- plus
      between-component variance).
    - ``quantiles``/``expected_shortfall_05``: Monte Carlo mixture -- draw
      ``100_000`` samples from ``Normal(mean_i, sigma_i)`` with probability
      ``w_i`` each, using a fixed seed for full determinism, then take the
      empirical quantiles/tail mean of the pooled sample (documented
      approximation rather than an analytic mixture-of-normals quantile,
      which has no closed form).
    - ``uncertainty``: ``sqrt(sum(w_i^2 * u_i^2)) + disagreement``, where
      ``disagreement = sqrt(sum(w_i*(mean_i - mean)^2))`` is the
      between-component spread of the point estimates (models disagreeing
      about the mean is itself a form of uncertainty on top of each model's
      own estimation error).

    Raises:
        ValueError: if ``forecasts`` is empty, the forecasts do not all
            share one ``(underlying_id, horizon_days, prediction_time)``, or
            ``weights`` has no positive-weight entry matching any forecast's
            ``model_id``.
    """
    if not forecasts:
        raise ValueError("forecasts must not be empty")
    underlying_ids = {f.underlying_id for f in forecasts}
    horizons = {f.horizon_days for f in forecasts}
    prediction_times = {f.prediction_time for f in forecasts}
    if len(underlying_ids) > 1 or len(horizons) > 1 or len(prediction_times) > 1:
        raise ValueError(
            "forecasts must all share one underlying_id, horizon_days and prediction_time"
        )

    w = np.array([float(weights.get(f.model_id, 0.0)) for f in forecasts], dtype=np.float64)
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        raise ValueError("weights must include at least one positive weight for these forecasts")
    w_norm = w / w_sum

    means = np.array([f.mean for f in forecasts], dtype=np.float64)
    sigmas = np.array([f.sigma for f in forecasts], dtype=np.float64)
    p_ups = np.array([f.p_up for f in forecasts], dtype=np.float64)
    uncertainties = np.array([f.uncertainty for f in forecasts], dtype=np.float64)

    mean = float(np.sum(w_norm * means))
    p_up = float(np.sum(w_norm * p_ups))
    between_var = float(np.sum(w_norm * (means - mean) ** 2))
    within_var = float(np.sum(w_norm * sigmas**2))
    sigma = float(np.sqrt(max(within_var + between_var, 0.0)))
    disagreement = float(np.sqrt(max(between_var, 0.0)))
    uncertainty = float(np.sqrt(np.sum((w_norm * uncertainties) ** 2))) + disagreement

    rng = np.random.default_rng(_MONTE_CARLO_SEED)
    component_idx = rng.choice(len(forecasts), size=_MONTE_CARLO_SAMPLES, p=w_norm)
    draws = rng.normal(loc=means[component_idx], scale=np.maximum(sigmas[component_idx], 1e-12))
    quantiles = {key: float(np.quantile(draws, q)) for key, q in QUANTILE_LEVELS.items()}
    q05 = quantiles["q05"]
    tail = draws[draws <= q05]
    expected_shortfall_05 = float(np.mean(tail)) if tail.size > 0 else q05

    n_train = int(sum(f.n_train for f in forecasts))
    n_effective = float(np.sum(w_norm * np.array([f.n_effective for f in forecasts])))

    component_payload = sorted(
        (
            {"model_id": f.model_id, "model_hash": f.model_hash, "weight": round(wn, 10)}
            for f, wn in zip(forecasts, w_norm.tolist(), strict=True)
        ),
        key=lambda entry: str(entry["model_id"]),
    )
    canonical = json.dumps(component_payload, sort_keys=True, separators=(",", ":"))
    model_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    first = forecasts[0]
    return HorizonForecast(
        underlying_id=first.underlying_id,
        horizon_days=first.horizon_days,
        prediction_time=first.prediction_time,
        p_up=float(np.clip(p_up, 0.0, 1.0)),
        mean=mean,
        sigma=sigma,
        quantiles=quantiles,
        expected_shortfall_05=expected_shortfall_05,
        uncertainty=uncertainty,
        model_id="ensemble",
        model_hash=model_hash,
        signal_family="ensemble",
        n_train=n_train,
        n_effective=n_effective,
    )
