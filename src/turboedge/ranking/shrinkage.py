"""Winner's-curse shrinkage of simulated ``mean_net_return`` toward a group mean.

Formula reference: Master Spec §15.1 ("Shrinkage"): ``EV_product ->
cluster_mean_EV``, with shrinkage intensity depending on uncertainty and
sample size.

This is a pragmatic, empirical-Bayes-flavoured shrinkage, not textbook
James-Stein (which requires cleanly separating between-unit "signal"
variance from within-unit "noise" variance -- not available here across
heterogeneous products). Instead, a single per-group shrinkage intensity
``B in [0, 1)`` is computed from:

- ``group_dispersion``: the standard deviation of the group's raw point
  estimates (a *conservative* proxy for how noisy the whole batch is --
  Build Contract v2 W7 requirement 3 explicitly asks for shrinkage to
  *increase* with this quantity, not decrease with it as a "pure signal"
  read would suggest; see the module-level note below),
- ``avg_mc_standard_error``: the average Monte Carlo standard error of the
  group's simulated means,
- ``avg_model_uncertainty``: the average forecast/model uncertainty
  (``HorizonForecast.uncertainty``) feeding the group's drift scenarios,
- ``n``: the group size, entering as a prior pseudo-count denominator so
  small groups (< ``cfg.prior_pseudo_count``, default 5) are always
  strongly shrunk even when the noise terms above happen to be small.

A single, group-level ``B`` (rather than one per member) is used
deliberately: applying the *same* ``B`` to every member of a group is an
affine, strictly increasing transform of each member's raw estimate
(for ``B < 1``), which trivially preserves the within-group ordering of
point estimates -- a explicit test requirement (order preservation) that a
per-member ``B_i`` could violate if members' noise estimates differ.

Note on "increases with dispersion": the Build Contract explicitly asks
for shrinkage intensity to rise with *within-group* dispersion of the raw
estimates. This is the opposite of a textbook empirical-Bayes rule (where
higher *between-unit* dispersion of true means should reduce shrinkage,
since pooling genuinely different products is then less appropriate) --
but is a defensible, more conservative choice here: this module cannot
distinguish "these products really do have different true EVs" from "our
simulation is just noisy for this batch", so a noisier-looking group is
treated as *less* trustworthy overall and pulled harder toward its own
mean. This is documented as an explicit, intentional simplification (see
Kurzbericht "Annahmen").
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from turboedge.storage.schemas import Direction

#: Master Spec §17 leverage buckets, used as one of the three shrinkage
#: group keys (underlying_id, direction, leverage_bucket).
LEVERAGE_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (2.0, 3.0, "2-3"),
    (3.0, 4.0, "3-4"),
    (4.0, 5.0, "4-5"),
    (5.0, 6.0, "5-6"),
    (6.0, 8.0, "6-8"),
    (8.0, 10.0, "8-10"),
    (10.0, 15.0, "10-15"),
    (15.0, float("inf"), ">15"),
)


class ShrinkageConfig(BaseModel):
    """Own config for this module (Build Contract v2: every module owns its
    ``XxxConfig``; wired into ``config.py``/YAML by the integration wave).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Prior pseudo-count ("virtual group members"). At group size
    #: ``n == prior_pseudo_count`` with zero measured dispersion/noise, B ==
    #: 0.5 (half shrunk). Groups with n < 5 are strongly shrunk (Build
    #: Contract requirement 3) because 5.0 is comfortably above typical
    #: sub-5 group sizes even before any dispersion/noise term is added.
    prior_pseudo_count: float = Field(default=5.0, gt=0.0)
    #: Weight on within-group dispersion of raw point estimates in the
    #: shrinkage numerator (see module docstring for why higher dispersion
    #: means *more*, not less, shrinkage here).
    weight_dispersion: float = Field(default=1.0, ge=0.0)
    #: Weight on the group's average Monte Carlo standard error.
    weight_mc_standard_error: float = Field(default=1.0, ge=0.0)
    #: Weight on the group's average model/forecast uncertainty.
    weight_model_uncertainty: float = Field(default=1.0, ge=0.0)


def leverage_bucket_for(leverage: float) -> str:
    """Master Spec §17 leverage bucket label for one leverage value.

    Raises:
        ValueError: if ``leverage`` is not positive.
    """
    if not (leverage > 0):
        raise ValueError(f"leverage must be > 0, got {leverage!r}")
    for lo, hi, label in LEVERAGE_BUCKETS:
        if lo <= leverage < hi:
            return label
    return LEVERAGE_BUCKETS[-1][2]  # unreachable given the open-ended last bucket, kept for mypy


def shrinkage_group_key(underlying_id: str, direction: Direction, leverage_bucket: str) -> str:
    """Group key for shrinkage clustering (Master Spec §15.1).

    Build Contract v2 lists exactly ``(underlying_id, direction,
    leverage_bucket)``; this module's caller (``ranking/ev.py``) further
    stratifies by ``horizon_days`` (appended by the caller to the returned
    key) since pooling a product's 3-day and 14-day evaluations together
    would conflate distributions with materially different shapes -- an
    explicit, documented extension, not a deviation from the three listed
    keys themselves.
    """
    return f"{underlying_id}|{direction.value}|{leverage_bucket}"


def shrinkage_intensity(
    group_dispersion: float,
    avg_mc_standard_error: float,
    avg_model_uncertainty: float,
    n: int,
    *,
    cfg: ShrinkageConfig | None = None,
) -> float:
    """Group-level shrinkage intensity ``B in [0, 1)`` (Master Spec §15.1).

    ``B = numerator / (numerator + n)`` with ``numerator = prior_pseudo_count
    + weight_dispersion * group_dispersion + weight_mc_standard_error *
    avg_mc_standard_error + weight_model_uncertainty * avg_model_uncertainty``.
    Monotonically increases with each noise/dispersion term and decreases
    with ``n`` (group size), matching Build Contract v2 W7 requirement 3.

    Raises:
        ValueError: if any of ``group_dispersion``, ``avg_mc_standard_error``,
            ``avg_model_uncertainty`` is negative, or ``n <= 0``.
    """
    if group_dispersion < 0.0:
        raise ValueError(f"group_dispersion must be >= 0, got {group_dispersion!r}")
    if avg_mc_standard_error < 0.0:
        raise ValueError(f"avg_mc_standard_error must be >= 0, got {avg_mc_standard_error!r}")
    if avg_model_uncertainty < 0.0:
        raise ValueError(f"avg_model_uncertainty must be >= 0, got {avg_model_uncertainty!r}")
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n!r}")

    c = cfg if cfg is not None else ShrinkageConfig()
    numerator = (
        c.prior_pseudo_count
        + c.weight_dispersion * group_dispersion
        + c.weight_mc_standard_error * avg_mc_standard_error
        + c.weight_model_uncertainty * avg_model_uncertainty
    )
    b = numerator / (numerator + n)
    return float(min(1.0, max(0.0, b)))


def shrink_group_means(
    raw_means: Sequence[float],
    mc_standard_errors: Sequence[float],
    model_uncertainties: Sequence[float],
    *,
    cfg: ShrinkageConfig | None = None,
) -> tuple[list[float], float]:
    """Shrink one group's raw point estimates toward their own group mean.

    Returns ``(shrunk_means, intensity)`` -- ``shrunk_means[i] = (1 -
    intensity) * raw_means[i] + intensity * group_mean``, in the same order
    as ``raw_means``. A single group-level ``intensity`` is used for every
    member (see module docstring: this guarantees order preservation).

    Raises:
        ValueError: if the three input sequences have different lengths, or
            are empty.
    """
    n = len(raw_means)
    if n == 0:
        raise ValueError("raw_means must not be empty")
    if len(mc_standard_errors) != n or len(model_uncertainties) != n:
        raise ValueError(
            "raw_means, mc_standard_errors and model_uncertainties must have the same length, "
            f"got {n}, {len(mc_standard_errors)}, {len(model_uncertainties)}"
        )

    means_arr: npt.NDArray[np.float64] = np.asarray(raw_means, dtype=np.float64)
    group_mean = float(means_arr.mean())
    group_dispersion = float(means_arr.std(ddof=0))
    avg_mc_se = float(np.mean(np.asarray(mc_standard_errors, dtype=np.float64)))
    avg_uncertainty = float(np.mean(np.asarray(model_uncertainties, dtype=np.float64)))

    intensity = shrinkage_intensity(group_dispersion, avg_mc_se, avg_uncertainty, n, cfg=cfg)
    shrunk = [(1.0 - intensity) * float(m) + intensity * group_mean for m in raw_means]
    return shrunk, intensity


__all__ = [
    "LEVERAGE_BUCKETS",
    "ShrinkageConfig",
    "leverage_bucket_for",
    "shrink_group_means",
    "shrinkage_group_key",
    "shrinkage_intensity",
]
