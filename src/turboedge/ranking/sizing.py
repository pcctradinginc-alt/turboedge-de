"""Suggested position sizing: fractional Kelly with hard caps (Master Spec §32).

```text
fractional Kelly = 0.25 * Kelly
```

Kelly is computed by maximizing expected log utility over the *empirical*
simulated net-return distribution (``sum_i log(1 + f * r_i)``), not a Normal
approximation (Build Contract v2 W7 requirement 5) -- the simulated
distribution is typically strongly skewed/fat-tailed (KO absorbs at a
bounded loss, upside is open-ended), which a Normal (mean/variance) Kelly
approximation would misprice.

Caps applied on top of fractional Kelly (Master Spec §32):

- ``max_position_fraction``
- ``max_cluster_fraction`` (net of the cluster's already-used fraction)
- ``max_total_risk`` (net of the portfolio's already-used fraction)
- ``max_KO_loss_contribution``: ``P_KO * position_fraction <= limit``

Uncertainty (``HorizonForecast.uncertainty``) scales the suggested size
down. The result is never negative and never exceeds any cap. Master Spec
§32: "TurboEdge darf eine vorgeschlagene Positionsgröße berechnen, aber
niemals ausführen" -- this module only ever returns a suggested fraction,
never places or sizes a live order.

Note on ``p_ko``: ``simulation/paths.py``'s own calibration study found
simulated P(KO) remains conservatively biased (over-predicted) at the
trading-relevant barrier distances even under its current default method.
``max_ko_loss_contribution`` is therefore itself conservative (it caps
against an already-inflated KO probability, not a calibrated one) -- this
module applies no separate correction for that bias.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy import optimize

_KELLY_ROOT_BRACKET_SHRINK = 1e-9


class SizingConfig(BaseModel):
    """Own config for this module (wired into ``config.py``/YAML by the
    integration wave).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Fraction of full Kelly actually suggested (Master Spec §32: 0.25).
    kelly_multiplier: float = Field(default=0.25, gt=0.0, le=1.0)
    #: Hard cap on any single position's capital fraction.
    max_position_fraction: float = Field(default=0.05, gt=0.0)
    #: Hard cap on total capital fraction concurrently deployed in one
    #: correlation cluster (``ranking/cluster.py``).
    max_cluster_fraction: float = Field(default=0.15, gt=0.0)
    #: Hard cap on total capital fraction deployed across the whole
    #: portfolio at once.
    max_total_risk: float = Field(default=0.30, gt=0.0)
    #: Hard cap on ``P_KO * position_fraction`` -- the position's expected
    #: contribution to "capital lost to a knockout", as a fraction of total
    #: capital.
    max_ko_loss_contribution: float = Field(default=0.02, gt=0.0)
    #: Sensitivity of the uncertainty haircut: suggested size is scaled by
    #: ``1 / (1 + uncertainty_sensitivity * uncertainty)``.
    uncertainty_sensitivity: float = Field(default=2.0, ge=0.0)
    #: Upper search bound for the empirical Kelly root-finder (in units of
    #: "multiples of capital"); full Kelly on a bounded-loss, open-upside
    #: distribution is rarely anywhere near this.
    kelly_search_upper_bound: float = Field(default=4.0, gt=0.0)


def kelly_fraction_empirical(
    net_returns: Sequence[float] | npt.NDArray[np.float64],
    *,
    upper_bound: float = 4.0,
) -> float:
    """Full Kelly fraction maximizing ``E[log(1 + f * r)]`` over the
    empirical distribution of simulated net returns ``r``.

    Solved by finding the root of the derivative ``g(f) = mean(r / (1 + f *
    r)) == 0`` via bisection (``g`` is continuous and strictly decreasing on
    the feasible domain for any non-degenerate distribution with some
    positive-return mass). Returns ``0.0`` (no fractional bet) whenever
    there is no positive edge (``g(0) = E[r] <= 0``) or the return sample is
    empty / entirely non-positive.

    ``f`` is searched within ``(0, min(upper_bound, feasible_max))``, where
    ``feasible_max`` keeps ``1 + f * r_i > 0`` for every simulated path
    (otherwise ``log`` is undefined for that path) -- i.e. never suggests
    leveraging the "capital at risk" beyond what the worst simulated outcome
    can sustain.

    Raises:
        ValueError: if ``upper_bound <= 0``.
    """
    if not (upper_bound > 0.0):
        raise ValueError(f"upper_bound must be > 0, got {upper_bound!r}")

    r = np.asarray(net_returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    if np.all(r <= 0.0):
        return 0.0

    min_r = float(r.min())
    feasible_max = (1.0 - _KELLY_ROOT_BRACKET_SHRINK) / (-min_r) if min_r < 0.0 else upper_bound
    f_hi = min(upper_bound, feasible_max)
    if f_hi <= 0.0:
        return 0.0

    def g(f: float) -> float:
        return float(np.mean(r / (1.0 + f * r)))

    if g(0.0) <= 0.0:
        return 0.0
    if g(f_hi) > 0.0:
        # Positive edge persists across the whole feasible/search range;
        # cap at f_hi rather than extrapolating past it.
        return float(f_hi)

    f_star = optimize.brentq(g, 0.0, f_hi)
    return float(max(0.0, f_star))


def suggested_position_fraction(
    net_returns: Sequence[float] | npt.NDArray[np.float64],
    p_ko: float,
    uncertainty: float,
    *,
    cluster_fraction_used: float = 0.0,
    total_risk_used: float = 0.0,
    cfg: SizingConfig | None = None,
) -> float:
    """Suggested position size as a fraction of capital (Master Spec §32).

    Never negative, never exceeds ``cfg.max_position_fraction`` nor any of
    the other caps (see module docstring). Degenerate inputs (e.g. every
    simulated return negative) yield ``0.0``.

    Raises:
        ValueError: if ``p_ko`` is not within ``[0, 1]``, ``uncertainty`` is
            negative, or ``cluster_fraction_used``/``total_risk_used`` is
            negative.
    """
    if not (0.0 <= p_ko <= 1.0):
        raise ValueError(f"p_ko must be within [0, 1], got {p_ko!r}")
    if uncertainty < 0.0:
        raise ValueError(f"uncertainty must be >= 0, got {uncertainty!r}")
    if cluster_fraction_used < 0.0:
        raise ValueError(f"cluster_fraction_used must be >= 0, got {cluster_fraction_used!r}")
    if total_risk_used < 0.0:
        raise ValueError(f"total_risk_used must be >= 0, got {total_risk_used!r}")

    c = cfg if cfg is not None else SizingConfig()

    f_kelly = kelly_fraction_empirical(net_returns, upper_bound=c.kelly_search_upper_bound)
    fraction = c.kelly_multiplier * f_kelly
    if fraction <= 0.0:
        return 0.0

    fraction /= 1.0 + c.uncertainty_sensitivity * uncertainty

    fraction = min(fraction, c.max_position_fraction)
    fraction = min(fraction, max(0.0, c.max_cluster_fraction - cluster_fraction_used))
    fraction = min(fraction, max(0.0, c.max_total_risk - total_risk_used))
    if p_ko > 0.0:
        fraction = min(fraction, c.max_ko_loss_contribution / p_ko)

    return float(max(0.0, fraction))


__all__ = ["SizingConfig", "kelly_fraction_empirical", "suggested_position_fraction"]
