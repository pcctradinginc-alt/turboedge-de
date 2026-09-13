"""Cross-issuer consensus spot and dislocation scoring.

Formula reference: Master Spec §13.5 ("Cross-Issuer Dislocation") and the
Build Contract's "Formeln (verbindlich)" section.

Nothing in this module should be read as a risk-free arbitrage signal
(Master Spec §13.5: "Nicht als risikolose Arbitrage interpretieren"). Two
certificates on the same underlying from different issuers are not
fungible: financing terms, knockout mechanics, quote liquidity and issuer
credit risk all differ. The scores here flag *relative* cost dislocation
for a human to investigate, not a mechanically exploitable spread.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from turboedge.pricing.intrinsic import implied_underlying
from turboedge.storage.schemas import Direction, ProductSnapshot

_MAD_TO_STD = 1.4826
_MIN_GROUP_SIZE = 3

# Build Contract Task 2: below this many full (bid+ask) quotes, bid-only
# products (an issuer quoting only bid, e.g. outside trading hours) are
# allowed to contribute to the consensus as a fallback rather than being
# dropped entirely -- see the "bid-only fallback" section of
# :func:`consensus_spot`'s docstring.
_MIN_FULL_QUOTES_TO_EXCLUDE_BID_ONLY = 5


@dataclass(frozen=True, slots=True)
class ConsensusSpot:
    """Robust cross-issuer consensus estimate of an underlying's spot price."""

    value: float
    n_used: int
    n_rejected: int
    dispersion: float  # robust (MAD-scaled) spread of the accepted sample
    # True iff at least one bid-only product (no ask) contributed to this
    # consensus under the fallback described in :func:`consensus_spot`.
    # Callers (pipeline/scan.py) surface this as a "consensus_from_bid_only"
    # warning.
    used_bid_only_fallback: bool = False
    # True iff both LONG and SHORT accepted quotes contributed, so `value`
    # is the bias-cancelling average of the two directions' medians (see
    # "Long/Short wrapper-margin bias" in :func:`consensus_spot`'s
    # docstring). False means only one direction had accepted quotes and
    # `value` is that direction's plain median instead -- a degraded but
    # still best-available estimate, not bias-corrected.
    direction_balanced: bool = False


def consensus_spot(
    products: Sequence[ProductSnapshot],
    fx: float = 1.0,
    mad_k: float = 5.0,
    fx_by_isin: Mapping[str, float] | None = None,
) -> ConsensusSpot:
    """Robust median consensus spot implied by a set of products on one underlying.

    Only products with a valid, quotable price (``bid``, ``financing_level``,
    ``ratio`` all present, ``ratio > 0``, and -- when ``ask`` is present --
    ``bid <= ask`` and ``ask > 0``) contribute -- missing critical pricing
    data is never imputed (CLAUDE.md rule 29), it simply does not vote. Each
    full (bid+ask) contributing product's implied spot is computed from its
    mid price via :func:`turboedge.pricing.intrinsic.implied_underlying`.

    ``fx_by_isin`` (Befund 1, 2026-09-13 measurement session): when given,
    each product's own fx is looked up by ISIN instead of applying the single
    ``fx`` uniformly to every product -- needed once fx is resolved
    per-(issuer, underlying) rather than assumed identical across issuers
    (see ``pricing/fx_resolution.py``). A product whose ISIN is *not* in
    ``fx_by_isin`` is excluded from the consensus entirely (its fx could not
    be resolved -- CLAUDE.md rule 29 forbids falling back to a guessed fx
    just to let it vote). When ``fx_by_isin`` is ``None`` (the default), the
    single ``fx`` argument is used for every product, unchanged from before.

    Bid-only fallback (Build Contract Task 2): a product with no ``ask`` at
    all (e.g. an issuer quoting only bid outside trading hours) is a
    tradability gate elsewhere (``ranking/gates.py``: REJECT
    "no_ask_quote"), not a reason to silently drop it from the consensus
    entirely when full quotes are scarce.

    - When at least :data:`_MIN_FULL_QUOTES_TO_EXCLUDE_BID_ONLY` products
      with a full bid/ask quote are available, bid-only products are
      excluded from the consensus -- the full-quote sample is large enough
      on its own.
    - Otherwise, bid-only products DO contribute, using ``bid`` in place of
      ``mid``. This is a documented, one-sided approximation:
      ``implied_underlying`` from a bid instead of a mid is biased slightly
      *low* for LONG products (Long adds ``mid * fx / ratio``, and
      ``bid <= mid``) and biased slightly *high* for SHORT products (Short
      subtracts it). ``ConsensusSpot.used_bid_only_fallback`` is set
      whenever this fallback actually contributed a bid-only product.

    Outliers (e.g. a Bezugsverhaltnis/ratio factor-of-10 data error) are
    removed via a median-absolute-deviation (MAD) filter: any implied spot
    further than ``mad_k`` scaled-MADs from the sample median is rejected
    before the final median/dispersion are computed. When the initial MAD is
    exactly zero (a degenerate/near-identical sample), a small relative
    epsilon is used instead so a single far-off outlier is still excluded
    rather than accidentally accepted by a zero-width band.

    Long/Short wrapper-margin bias (Build Contract BEFUND 1): ``mid = fair
    value + issuer margin`` to first order (``pricing/issuer_margin.py``), so
    a typically-positive margin makes ``implied_underlying`` computed from a
    LONG product's mid *overstate* the true spot (``implied_S = S +
    margin/ratio``) and from a SHORT product's mid *understate* it
    (``implied_S = S - margin/ratio``). Measured on a live BNP+Citi DAX scan
    (2026-09-11, ``docs/data_sources.md``): median implied spot from LONG
    quotes was ~25487.6 vs. ~25478.7 from SHORT quotes (~8.9 points / ~0.03%
    apart, both directions with >2000 contributing quotes) -- a small but
    real, systematically-signed gap, not sampling noise. Averaging
    ``median(implied | LONG)`` and ``median(implied | SHORT)`` cancels this
    bias to first order (the same live sample: ~25483.1, vs. ~25485.7 for the
    plain combined median, which is pulled toward the LONG side). This
    function therefore computes the MAD outlier filter on the full combined
    sample (outlier rejection is direction-agnostic -- a ratio/factor error
    looks the same from either side), then -- when the accepted sample has
    at least one LONG *and* one SHORT quote -- takes the average of the two
    per-direction medians as the final ``value``
    (``ConsensusSpot.direction_balanced=True``). When only one direction
    survives (thin/one-sided sample), the plain median of the accepted
    sample is used instead (``direction_balanced=False``): a real, if
    uncorrected, estimate is still preferable to refusing to price the
    underlying at all (CLAUDE.md rule 29 forbids imputing *missing* data, not
    reporting an unbalanced-but-genuine estimate).

    Raises:
        ValueError: if no product contributes a valid implied spot.
    """
    full_quotes: list[ProductSnapshot] = []
    bid_only_quotes: list[ProductSnapshot] = []
    for product in products:
        if product.bid is None or product.financing_level is None or product.ratio is None:
            continue
        if not (product.ratio > 0):
            continue
        if product.ask is not None:
            if not (product.bid <= product.ask) or not (product.ask > 0):
                continue
            full_quotes.append(product)
        else:
            bid_only_quotes.append(product)

    use_bid_only_fallback = (
        len(full_quotes) < _MIN_FULL_QUOTES_TO_EXCLUDE_BID_ONLY and len(bid_only_quotes) > 0
    )
    contributing = [*full_quotes, *(bid_only_quotes if use_bid_only_fallback else [])]

    implied_values: list[float] = []
    directions: list[Direction] = []
    for product in contributing:
        # bid/financing_level/ratio were already checked non-None while
        # building full_quotes/bid_only_quotes above; re-asserted here since
        # that narrowing does not carry across the two separate loops/lists.
        assert product.bid is not None
        assert product.financing_level is not None
        assert product.ratio is not None
        if fx_by_isin is not None:
            product_fx = fx_by_isin.get(product.isin)
            if product_fx is None:
                continue  # fx not resolved for this product -- never guessed
        else:
            product_fx = fx
        price = (product.bid + product.ask) / 2.0 if product.ask is not None else product.bid
        try:
            implied = implied_underlying(
                price, product.financing_level, product.ratio, product.direction, product_fx
            )
        except ValueError:
            continue
        implied_values.append(implied)
        directions.append(product.direction)

    n_total = len(implied_values)
    if n_total == 0:
        raise ValueError("no product supplied a valid implied spot to build a consensus from")

    arr = np.asarray(implied_values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scaled_mad = mad * _MAD_TO_STD
    threshold = max(abs(median) * 1e-6, 1e-9) if scaled_mad == 0.0 else mad_k * scaled_mad
    mask = np.abs(arr - median) <= threshold
    accepted = arr[mask]
    accepted_directions = [d for d, keep in zip(directions, mask, strict=True) if keep]
    n_rejected = int(n_total - accepted.size)

    paired = list(zip(accepted, accepted_directions, strict=True))
    long_values = [v for v, d in paired if d == Direction.LONG]
    short_values = [v for v, d in paired if d == Direction.SHORT]
    direction_balanced = bool(long_values) and bool(short_values)
    if direction_balanced:
        value = (statistics.median(long_values) + statistics.median(short_values)) / 2.0
    else:
        value = float(np.median(accepted))

    accepted_mad = float(np.median(np.abs(accepted - value))) if accepted.size > 0 else 0.0
    dispersion = accepted_mad * _MAD_TO_STD

    return ConsensusSpot(
        value=value,
        n_used=int(accepted.size),
        n_rejected=n_rejected,
        dispersion=dispersion,
        used_bid_only_fallback=use_bid_only_fallback,
        direction_balanced=direction_balanced,
    )


def robust_zscore(values: Sequence[float]) -> npt.NDArray[np.float64]:
    """Median/MAD-based robust z-score, resilient to a handful of outliers.

    ``z_i = (x_i - median(x)) / (1.4826 * MAD(x))``. Returns an all-zero
    array when ``MAD(x) == 0`` (degenerate/constant sample) rather than
    dividing by zero.
    """
    arr = np.asarray(values, dtype=np.float64)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    if mad == 0.0:
        return np.zeros_like(arr)
    return (arr - median) / (_MAD_TO_STD * mad)


@dataclass(frozen=True, slots=True)
class CrossIssuerInput:
    """One product's inputs to the cross-issuer scoring pass."""

    isin: str
    issuer: str
    underlying_id: str
    direction: Direction
    leverage_bucket: str
    normalized_spread: float
    issuer_margin_pct: float
    financing_spread: float | None
    quote_age_s: float | None


@dataclass(frozen=True, slots=True)
class CrossIssuerScores:
    """Cross-issuer dislocation scores for one product.

    All fields are ``None`` when the product's peer group (same
    ``underlying_id``, ``direction``, ``leverage_bucket``) has fewer than 3
    members -- too small a sample for a robust cross-sectional statistic.
    """

    cross_issuer_residual_zscore: float | None
    issuer_markup_score: float | None
    quote_dislocation_score: float | None
    wrapper_edge: float | None


def _group_key(row: CrossIssuerInput) -> tuple[str, Direction, str]:
    return (row.underlying_id, row.direction, row.leverage_bucket)


def cross_issuer_scores(rows: Sequence[CrossIssuerInput]) -> dict[str, CrossIssuerScores]:
    """Compute cross-issuer dislocation scores, keyed by ISIN.

    Products are grouped by ``(underlying_id, direction, leverage_bucket)``
    -- the smallest unit within which two products are genuinely comparable
    (Master Spec §13.5). Within each group of at least 3 members:

    - ``cross_issuer_residual_zscore``: robust z-score of ``issuer_margin_pct``.
    - ``quote_dislocation_score``: combines the standardized ``normalized_spread``
      anomaly and the margin residual z-score into one dislocation magnitude
      (Euclidean norm of the two z-scores), so a product can be flagged for
      either an unusually wide spread, an unusually high margin, or both.
    - ``wrapper_edge``: ``median(total_cost_pct in group) - own total_cost_pct``,
      where ``total_cost_pct = normalized_spread + issuer_margin_pct`` is the
      portion of total product cost this scoring pass has visibility into.
      Positive means this product is cheaper than its peers.

    ``issuer_markup_score`` is computed at issuer level, not per-group: for
    each issuer, it is the median of that issuer's ``cross_issuer_residual_zscore``
    values across every group (of size >= 3) it appears in within ``rows``.
    This captures whether an issuer systematically charges an above-peer
    margin across its whole product shelf, not just in one bucket. An issuer
    with no qualifying group memberships gets ``None``.

    Groups smaller than 3 members yield all-``None`` scores for their
    members.
    """
    groups: dict[tuple[str, Direction, str], list[CrossIssuerInput]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    residual_z_by_isin: dict[str, float] = {}
    dislocation_by_isin: dict[str, float] = {}
    wrapper_edge_by_isin: dict[str, float] = {}

    for members in groups.values():
        if len(members) < _MIN_GROUP_SIZE:
            continue
        margins = [m.issuer_margin_pct for m in members]
        spreads = [m.normalized_spread for m in members]
        margin_z = robust_zscore(margins)
        spread_z = robust_zscore(spreads)
        total_costs = [m.normalized_spread + m.issuer_margin_pct for m in members]
        group_median_total_cost = statistics.median(total_costs)

        for member, mz, sz, total_cost in zip(
            members, margin_z, spread_z, total_costs, strict=True
        ):
            residual_z_by_isin[member.isin] = float(mz)
            dislocation_by_isin[member.isin] = float(np.hypot(float(mz), float(sz)))
            wrapper_edge_by_isin[member.isin] = group_median_total_cost - total_cost

    issuer_zs: dict[str, list[float]] = {}
    for row in rows:
        z = residual_z_by_isin.get(row.isin)
        if z is not None:
            issuer_zs.setdefault(row.issuer, []).append(z)
    issuer_markup: dict[str, float] = {
        issuer: statistics.median(zs) for issuer, zs in issuer_zs.items()
    }

    result: dict[str, CrossIssuerScores] = {}
    for row in rows:
        result[row.isin] = CrossIssuerScores(
            cross_issuer_residual_zscore=residual_z_by_isin.get(row.isin),
            issuer_markup_score=issuer_markup.get(row.issuer),
            quote_dislocation_score=dislocation_by_isin.get(row.isin),
            wrapper_edge=wrapper_edge_by_isin.get(row.isin),
        )
    return result
