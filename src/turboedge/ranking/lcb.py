"""Lower confidence bound on a product/horizon's simulated net return.

Formula reference: Master Spec §15.2 ("Lower Confidence Bound"): select
``argmax LCB_95(EV_product)`` instead of ``argmax point_estimate(EV_product)``.

IMPORTANT (Master Spec §10, Build Contract v2 W7 requirement 2): the value
returned here is **not** a formal ``(1 - alpha)`` confidence interval bound
with any statistical coverage guarantee. It is a heuristic, deliberately
conservative combination of three separate sources of optimism this system
must correct for:

1. Winner's-curse / selection bias (``shrunk_mean``, ``ranking/shrinkage.py``
   -- Master Spec §15.1): correcting for the fact that among many candidate
   products, the raw top point estimate is biased upward.
2. Model/forecast drift uncertainty (``pessimistic_mean`` vs ``central_mean``
   -- the pessimistic-drift-scenario simulation from ``ranking/ev.py``):
   correcting for the possibility that the underlying's true drift is worse
   than the forecast's central estimate.
3. Monte Carlo estimation noise (``mc_standard_error``): correcting for the
   fact that ``mean_net_return`` is itself estimated from a finite number of
   simulated paths.

None of these is a formally derived confidence interval on its own, and
their combination below is a documented engineering choice, not a
statistically calibrated quantity -- no coverage probability is claimed for
it. It is monotonically conservative by construction (see
:func:`lower_confidence_bound`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LcbConfig(BaseModel):
    """Own config for this module (wired into ``config.py``/YAML by the
    integration wave).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Standard-normal one-sided ~95th-percentile multiplier (default
    #: matches the pessimistic-drift-scenario z used in ``ranking/ev.py``,
    #: 1.645), applied to the Monte Carlo standard error.
    z: float = Field(default=1.645, gt=0.0)


def lower_confidence_bound(
    central_mean: float,
    pessimistic_mean: float,
    mc_standard_error: float,
    shrunk_mean: float,
    *,
    z: float = 1.645,
) -> float:
    """A conservative, downside-only combination of the winner's-curse-shrunk
    mean, the pessimistic-drift-scenario mean, and the Monte Carlo standard
    error.

    ``base = min(central_mean, shrunk_mean)`` -- starts from whichever of the
    raw central-scenario mean or its shrinkage-corrected counterpart is
    already lower, so shrinkage can never accidentally *raise* the bound.

    ``downside = min(0.0, pessimistic_mean - central_mean)`` -- the drop from
    central to pessimistic scenario, clamped to be non-positive so a
    pessimistic-scenario simulation that (due to Monte Carlo noise) happens
    to land above the central-scenario mean never raises the bound either.

    ``result = base + downside - z * mc_standard_error``.

    By construction ``result <= min(central_mean, shrunk_mean) <=
    central_mean`` always (``mc_standard_error >= 0``, ``z > 0``).

    Raises:
        ValueError: if ``mc_standard_error < 0`` or ``z <= 0``.
    """
    if mc_standard_error < 0.0:
        raise ValueError(f"mc_standard_error must be >= 0, got {mc_standard_error!r}")
    if z <= 0.0:
        raise ValueError(f"z must be > 0, got {z!r}")

    base = min(central_mean, shrunk_mean)
    downside = min(0.0, pessimistic_mean - central_mean)
    return base + downside - z * mc_standard_error


__all__ = ["LcbConfig", "lower_confidence_bound"]
