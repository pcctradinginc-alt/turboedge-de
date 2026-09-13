"""Utility function and final ranking score (Master Spec §18).

```text
U_jh =
    E(NetReturn_jh)
  - lambda_es * abs(ExpectedShortfall95_jh)
  - lambda_ko * P_KO_jh
  - lambda_uncertainty * uncertainty_jh
  - lambda_cluster * cluster_risk

score_jh =
    LCB(U_jh)
  * liq
  * calibration_factor
  * strategy_posterior_factor
  * positive_memory_factor
```

``LCB(U_jh)`` (Master Spec §18 "Final") is implemented here by substituting
the already lower-confidence-bounded net return (``ranking/lcb.py``,
``lcb_net_return``) for the raw simulated ``E(NetReturn_jh)`` inside the
utility formula, rather than deriving a second, separate confidence
interval around the scalar utility value itself. This is a deliberate,
documented simplification: the dominant source of winner's-curse /
estimation-noise bias this system corrects for is the mean-return term
(products are selected via ``argmax`` over many candidates), while
``ExpectedShortfall95``/``P_KO`` come from the same simulation run (so are
not separately re-biased by product *selection*) and ``uncertainty``/
``cluster_risk`` are comparatively low-variance inputs. Flagged as an open
point in the Kurzbericht.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Floor for the combined multiplier product ``f`` (see :func:`score`) when
# dividing a negative utility by it -- guards against division by exactly
# zero (a factor of 0.0 is a legal, documented input: e.g.
# ``positive_memory_factor=0.0`` for "no positive-memory support at all").
# Deliberately tiny relative to any realistic factor product so it only ever
# bites at the true f=0 edge case, not real (however illiquid/uncalibrated)
# inputs.
_MIN_FACTOR_PRODUCT = 1e-9


class UtilityConfig(BaseModel):
    """Own config for this module (wired into ``config.py``/YAML by the
    integration wave).

    Default lambdas (Build Contract v2 W7 "Annahmen", versioned here and
    intended to be revisited empirically -- Master Spec §53 guiding
    question):

    - ``lambda_es = 0.5``: ``ExpectedShortfall95`` is already expressed in
      net-return units directly comparable to ``E(NetReturn)``. Weighting it
      at 0.5 means a 10-percentage-point ES95 subtracts 5 points of utility
      -- moderate tail-risk aversion. Full weight (1.0) was rejected as
      double-counting: the LCB stage (``ranking/lcb.py``) already pulls the
      return estimate down via a pessimistic drift scenario and Monte Carlo
      noise margin, both of which are correlated with tail risk.
    - ``lambda_ko = 0.5``: ``P_KO`` is *not* a calibrated probability --
      ``simulation/paths.py``'s own calibration study found it remains
      conservatively biased (over-predicted) at the trading-relevant barrier
      distances even under its current default method; no correction is
      applied here, so this term already leans conservative (fewer, not
      riskier, proposals) on top of its stated 0.5 weight. ``P_KO`` is a
      probability in ``[0, 1]``; the
      near-total-loss KO outcome is already reflected inside
      ``E(NetReturn)`` (the simulated net-return distribution includes KO
      paths at their settlement value). ``lambda_ko`` is therefore an
      *additional*, moderate penalty on top of that -- representing the
      non-pecuniary cost of being stopped out (forced exit removes any
      chance of the position recovering, unlike a merely adverse but still
      open position) -- kept at 0.5 rather than 1.0 to avoid double
      counting the same loss twice.
    - ``lambda_uncertainty = 1.0``: forecast/model uncertainty
      (``HorizonForecast.uncertainty``, a standard error on the underlying's
      predicted drift) is not otherwise represented anywhere else in this
      formula (unlike ES/P_KO, which come from the simulation itself), so it
      is fully (1:1) charged against expected return -- a risk-neutral-to-
      model-error stance: a wider forecast standard error is treated as
      exactly as costly as an equivalent reduction in expected return.
    - ``lambda_cluster = 0.3``: ``cluster_risk`` (portfolio concentration in
      the candidate's correlation cluster, ``ranking/cluster.py``) is kept
      deliberately soft here (0.3) because the *primary* defense against
      cluster/ruin risk is the hard caps in ``ranking/sizing.py``
      (``max_cluster_fraction``) and the ``cluster_risk_pass`` gate
      (``ranking/gates.py`` via ``ranking/ev.py.to_candidate_gate_input``) --
      this utility term is a softer tie-breaker among otherwise-similar
      candidates, not the main risk control.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    lambda_es: float = Field(default=0.5, ge=0.0)
    lambda_ko: float = Field(default=0.5, ge=0.0)
    lambda_uncertainty: float = Field(default=1.0, ge=0.0)
    lambda_cluster: float = Field(default=0.3, ge=0.0)


def expected_utility(
    mean_net_return: float,
    expected_shortfall_95: float,
    p_ko: float,
    uncertainty: float,
    cluster_risk: float,
    *,
    cfg: UtilityConfig | None = None,
) -> float:
    """``U_jh`` (Master Spec §18). ``mean_net_return`` may be the raw
    simulated mean or an already-adjusted (e.g. LCB) net return -- this
    function is agnostic to which; :func:`score` passes the LCB-adjusted
    value (see module docstring).

    Raises:
        ValueError: if ``p_ko`` is not in ``[0, 1]``, or ``uncertainty`` or
            ``cluster_risk`` is negative.
    """
    if not (0.0 <= p_ko <= 1.0):
        raise ValueError(f"p_ko must be within [0, 1], got {p_ko!r}")
    if uncertainty < 0.0:
        raise ValueError(f"uncertainty must be >= 0, got {uncertainty!r}")
    if cluster_risk < 0.0:
        raise ValueError(f"cluster_risk must be >= 0, got {cluster_risk!r}")

    c = cfg if cfg is not None else UtilityConfig()
    return (
        mean_net_return
        - c.lambda_es * abs(expected_shortfall_95)
        - c.lambda_ko * p_ko
        - c.lambda_uncertainty * uncertainty
        - c.lambda_cluster * cluster_risk
    )


def score(
    lcb_net_return: float,
    expected_shortfall_95: float,
    p_ko: float,
    uncertainty: float,
    cluster_risk: float,
    liquidity_factor: float,
    *,
    cfg: UtilityConfig | None = None,
    calibration_factor: float = 1.0,
    strategy_posterior_factor: float = 1.0,
    positive_memory_factor: float = 1.0,
) -> tuple[float, float]:
    """Final ranking ``score_jh`` (Master Spec §18 "Final").

    Returns ``(utility, score)`` where ``utility = LCB(U_jh)`` (see module
    docstring).

    **Deviation from the literal Master Spec §18 formula, documented here
    per that section's own reasoning:** the spec writes
    ``score = LCB(U) * liq * calibration_factor * strategy_posterior_factor
    * positive_memory_factor`` unconditionally. Each of the four
    multipliers is a quality/confidence factor in ``(0, 1]`` (1.0 = neutral;
    lower = worse -- e.g. ``liquidity_factor`` penalizes thin order books).
    Taken literally, that formula is only order-preserving in the intended
    direction (a *better* multiplier -> a *better*, i.e. higher, score) when
    ``LCB(U) >= 0``: multiplying a *non-negative* number by a smaller
    positive factor makes it smaller, correctly ranking it worse. But when
    ``LCB(U) < 0`` (a candidate whose lower-confidence-bounded utility is
    already negative -- not a rare case, since the whole point of ``LCB`` is
    to be pessimistic), multiplying a *negative* number by a smaller
    positive factor moves it *towards zero*, i.e. makes it *larger* --
    exactly backwards. Concretely: two candidates with identical
    ``LCB(U) = -0.145``, one liquid (``liquidity_factor = 1.0``, score
    ``-0.145``) and one illiquid (``liquidity_factor = 0.3``, score
    ``-0.0435``) -- the illiquid, otherwise-identical candidate would rank
    *above* the liquid one under the literal formula, an order violation a
    ranking function must never produce (a strictly worse attribute must
    never improve rank).

    The fix applied here keeps every multiplier monotonic in the *intended*
    direction regardless of ``LCB(U)``'s sign, by combining the four
    multipliers into a single ``f = liquidity_factor * calibration_factor *
    strategy_posterior_factor * positive_memory_factor in (0, 1]`` and
    switching the operator on the sign of the utility::

        score = LCB(U) * f          if LCB(U) >= 0   (unchanged: matches spec)
        score = LCB(U) / max(f, eps) if LCB(U) < 0    (deviation: divide, not multiply)

    For ``LCB(U) < 0``, dividing by a smaller positive ``f`` produces a
    *more negative* (worse) score -- restoring "worse multiplier -> worse
    score" in both branches, so the four multipliers are monotonic in the
    intended direction for every candidate regardless of its utility's
    sign, and two candidates with equal ``LCB(U)`` are ordered purely by
    their multipliers in the same direction either side of zero (order
    preserved: the previous cross-over at ``LCB(U) = 0`` is gone). ``eps``
    (``_MIN_FACTOR_PRODUCT``) only guards the true ``f == 0`` edge (a
    multiplier of exactly ``0.0`` is a legal input); the two branches agree
    at ``LCB(U) == 0`` (both give ``0.0``), so the switch introduces no
    discontinuity there. Flagged as an open point in the Kurzbericht per
    CLAUDE.md rule 23 (every research change gets a Trial-ID) --
    this is a correctness fix to the ranking order, not a new research
    hypothesis, so no separate trial is opened, but the deviation from the
    literal spec text is recorded here for anyone reconciling this code
    against Master Spec §18.

    ``calibration_factor``, ``strategy_posterior_factor`` and
    ``positive_memory_factor`` default to ``1.0`` (neutral) -- Build
    Contract v2 W7: these are wired to real values (forecast calibration
    quality, ``StrategyPosterior.strategy_posterior_factor``,
    positive-memory similarity) by a later integration wave; this module
    only needs to combine them in correctly.

    Raises:
        ValueError: if ``liquidity_factor`` is not positive, or any of
            ``calibration_factor``/``strategy_posterior_factor``/
            ``positive_memory_factor`` is negative.
    """
    if not (liquidity_factor > 0.0):
        raise ValueError(f"liquidity_factor must be > 0, got {liquidity_factor!r}")
    for name, value in (
        ("calibration_factor", calibration_factor),
        ("strategy_posterior_factor", strategy_posterior_factor),
        ("positive_memory_factor", positive_memory_factor),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value!r}")

    utility = expected_utility(
        lcb_net_return, expected_shortfall_95, p_ko, uncertainty, cluster_risk, cfg=cfg
    )
    multiplier = (
        liquidity_factor * calibration_factor * strategy_posterior_factor * positive_memory_factor
    )
    if utility >= 0.0:
        final = utility * multiplier
    else:
        final = utility / max(multiplier, _MIN_FACTOR_PRODUCT)
    return utility, final


__all__ = ["UtilityConfig", "expected_utility", "score"]
