"""Product-level integrity checks.

Formula reference: Master Spec §6 ("Source Health und Cross-Source
Validation") and the Build Contract's "Formeln (verbindlich)" section
(``ratio_factor_error``).

CLAUDE.md rule 29: missing critical data is never silently imputed. Every
check below either passes, or records a ``failures``/``warnings`` entry --
it never substitutes a guessed value for a missing one.

Two structural limitations, both a consequence of this function's fixed,
contract-specified signature (it receives a single ``ProductSnapshot`` plus
a scalar ``consensus``, not an FX-rate table or a full cost decomposition):

- FX conversion: the fair-value plausibility check only runs when the
  product's ``underlying_currency`` matches its ``currency`` (i.e. an
  implicit fx=1 is safe). A genuinely cross-currency product (e.g. a EUR
  certificate on SPX) gets a warning instead of a possibly-wrong fair-value
  comparison, since no FX rate is available here.
- Margin plausibility: without ``fair_gap_premium``/``financing_spread``
  inputs this layer cannot compute the full issuer-margin decomposition
  from ``pricing/issuer_margin.py``. ``margin_warn_pct`` is instead applied
  to the coarser ``(mid - fair_value) / ask`` "raw premium" ratio (Build
  Contract W1: ``pricing/fair_value.theoretical_fair_value``, not plain
  intrinsic value, so a ``turbo_classic``'s real carry is not mistaken for
  an anomaly) as an early warning signal; the precise ``issuer_margin_pct``
  check belongs to the scan pipeline once ``decompose_ask`` has run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from turboedge.pricing.fair_value import dividend_yield_for_underlying, theoretical_fair_value
from turboedge.pricing.intrinsic import implied_underlying
from turboedge.storage.schemas import Direction, ProductSnapshot, ProductType

# Barrier == financing_level is expected for turbo_open_end / turbo_classic;
# allow a small relative tolerance for rounding/venue quirks before flagging.
_BARRIER_EQUALITY_TOLERANCE_PCT = 0.005

# Relative + absolute tolerance band for "bid not massively below intrinsic".
_INTRINSIC_PLAUSIBILITY_TOLERANCE_PCT = 0.05
_INTRINSIC_PLAUSIBILITY_ABS_FLOOR = 0.01

# Contract: ratio_factor_error tolerance band around 10^k is 15%.
_RATIO_FACTOR_TOLERANCE_PCT = 0.15
_RATIO_FACTOR_EXPONENTS = (-3, -2, -1, 1, 2, 3)

# Contract: >3% implied-spot deviation without a factor error is a failure.
_IMPLIED_SPOT_DEVIATION_THRESHOLD_PCT = 0.03


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Result of :func:`check_product`. ``passed`` is True iff ``failures`` is empty."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _barrier_side_ok(direction: Direction, spot: float, barrier: float) -> bool:
    if direction == Direction.LONG:
        return spot > barrier
    return spot < barrier


def check_product(
    p: ProductSnapshot,
    consensus: float | None,
    now: datetime,
    max_quote_age_s: float,
    known_issuers: frozenset[str] | None,
    margin_warn_pct: float,
    *,
    ref_rate: float = 0.0,
) -> IntegrityReport:
    """Run every integrity check from Master Spec §6 against one product snapshot.

    ``ref_rate`` (annual, decimal) is only used by the fair-value/premium
    plausibility check below, and only matters for a ``turbo_classic``
    (Build Contract W1: it discounts the fixed strike to ``now.date()``);
    it has no effect on ``turbo_open_end``/``mini_future`` (fair value ==
    intrinsic, no discounting), so its ``0.0`` default is safe for every
    caller/test that only ever prices those product types.
    """
    failures: list[str] = []
    warnings: list[str] = []

    # -- critical pricing data presence (never imputed) ----------------
    # Note: a missing ``ask`` is deliberately NOT flagged here. An issuer
    # quoting only a bid (e.g. outside trading hours) is a tradability
    # concern, not a data-integrity error -- CLAUDE.md rule 29 ("never
    # silently impute") is still honored because ``ask`` is genuinely left
    # ``None``, never guessed. ranking/gates.py rejects such products
    # explicitly (reason "no_ask_quote") once bid/financing_level/ratio are
    # otherwise plausible; pipeline/scan.py skips every ask-dependent
    # pricing step (decompose_ask, leverage, spread) for them.
    #
    # Symmetrically, a missing ``bid`` is NOT flagged here either when the
    # source has explicitly told us there is no live two-way market at all
    # (``bid`` AND ``ask`` both ``None`` and ``quote_presence is False`` --
    # e.g. Citi's ``referencePriceMethod == "Closing Price"`` rows, see
    # adapters/issuer_feeds.py). That is "source has no tradable quote for
    # this product", not a data-quality violation -- master data (financing
    # level, barrier, ISIN, underlying mapping, ...) is untouched and still
    # validated by every other check in this function. ranking/gates.py
    # rejects such products explicitly (reason "no_live_quote") once those
    # other checks pass. A missing ``bid`` with ``quote_presence`` True or
    # None (i.e. the source did NOT explicitly say "no live quote") remains
    # an unexpected, genuine data-integrity failure.
    no_live_quote = p.bid is None and p.ask is None and p.quote_presence is False
    if p.bid is None and not no_live_quote:
        failures.append("missing_bid")
    if p.financing_level is None:
        failures.append("missing_financing_level")
    # p.ratio is a PositiveFloat at the schema level, so ratio > 0 always
    # holds for a valid ProductSnapshot instance; no separate check needed.

    if p.bid is not None and p.ask is not None and p.bid > p.ask:
        failures.append("bid_greater_than_ask")

    # -- barrier validity -------------------------------------------------
    if p.knockout_barrier is None:
        failures.append("missing_barrier")
    elif p.financing_level is not None:
        barrier = p.knockout_barrier
        f = p.financing_level
        if p.product_type in (ProductType.TURBO_OPEN_END, ProductType.TURBO_CLASSIC):
            if f == 0 or abs(barrier - f) / abs(f) > _BARRIER_EQUALITY_TOLERANCE_PCT:
                failures.append("barrier_deviates_from_financing_level")
        else:
            if p.direction == Direction.LONG and barrier < f:
                failures.append("barrier_invalid_long")
            elif p.direction == Direction.SHORT and barrier > f:
                failures.append("barrier_invalid_short")

        # -- barrier on the correct side of the spot -----------------------
        spot_ref = consensus if consensus is not None else p.underlying_price_ref
        if spot_ref is None:
            warnings.append("no_spot_reference_for_barrier_side_check")
        else:
            side_ok = _barrier_side_ok(p.direction, spot_ref, barrier)
            if not side_ok and not p.knocked_out:
                failures.append("knocked_out_or_wrong_side")
            elif side_ok and p.knocked_out:
                warnings.append("marked_knocked_out_but_spot_still_active")

    # -- fair value / premium plausibility ---------------------------
    # Build Contract W1: plausibility is checked against the same
    # theoretical fair value pricing/issuer_margin.decompose_ask uses, not
    # plain intrinsic value -- for turbo_open_end/mini_future (everything
    # observed live so far) the two are identical, so this is a no-op on
    # today's data; for a turbo_classic (fixed strike/maturity), a real,
    # legitimate carry/present-value gap between the two means an
    # intrinsic-only check would misfire "bid_below_intrinsic" /
    # "premium_pct_high" on an honestly-priced product (exactly the failure
    # mode this milestone's measurement found -- see the Build Contract
    # report).
    spot_ref = consensus if consensus is not None else p.underlying_price_ref
    same_currency = p.underlying_currency is None or p.underlying_currency == p.currency
    if (
        p.bid is not None
        and p.ask is not None
        and p.financing_level is not None
        and spot_ref is not None
    ):
        if same_currency:
            fair_value: float | None
            try:
                fair_value = theoretical_fair_value(
                    direction=p.direction,
                    product_type=p.product_type,
                    spot=spot_ref,
                    financing_level=p.financing_level,
                    knockout_barrier=(
                        p.knockout_barrier if p.knockout_barrier is not None else p.financing_level
                    ),
                    ratio=p.ratio,
                    fx=1.0,
                    ref_rate=ref_rate,
                    financing_spread=0.0,
                    as_of=now.date(),
                    maturity=p.maturity,
                    dividend_yield=dividend_yield_for_underlying(p.underlying_id),
                )
            except ValueError:
                # turbo_classic with an unresolvable maturity (e.g. the BNP
                # timestamp-format gap, adapters/issuer_feeds.py) -- never
                # guess a fair value; skip this plausibility pass rather
                # than silently falling back to (possibly wrong) intrinsic.
                fair_value = None
                warnings.append("fair_value_unavailable_for_plausibility_check")
            if fair_value is not None:
                tolerance = max(
                    fair_value * _INTRINSIC_PLAUSIBILITY_TOLERANCE_PCT,
                    _INTRINSIC_PLAUSIBILITY_ABS_FLOOR,
                )
                if p.bid < fair_value - tolerance:
                    failures.append("bid_below_intrinsic")
                mid = (p.bid + p.ask) / 2.0
                if p.ask > 0:
                    premium_pct = (mid - fair_value) / p.ask
                    if premium_pct > margin_warn_pct:
                        warnings.append("premium_pct_high")
        else:
            warnings.append("fx_conversion_unavailable_for_intrinsic_check")

    # -- issuer known -------------------------------------------------------
    if known_issuers is not None and p.issuer not in known_issuers:
        failures.append("unknown_issuer")

    # -- underlying mapping valid --------------------------------------------
    if p.underlying_id is None:
        failures.append("missing_underlying_mapping")

    # -- quote freshness --------------------------------------------------
    # A missing or stale quote timestamp is a tradability gate, not a data
    # integrity error (Build Contract Task 2 review finding): the quote data
    # itself may be perfectly well-formed, it is simply too old (or entirely
    # absent) to trade on right now. Both are therefore ``warnings`` here,
    # not ``failures`` -- ranking/gates.py is what actually rejects such
    # candidates, via reasons "quote_timestamp_missing"/"quote_stale". A
    # quote timestamp in the future, however, remains a genuine integrity
    # failure (a corrupted/implausible timestamp, not merely an old one).
    if p.quote_timestamp is None:
        warnings.append("quote_timestamp_missing")
    else:
        age_s = (now - p.quote_timestamp).total_seconds()
        if age_s < 0:
            failures.append("quote_timestamp_in_future")
        elif age_s > max_quote_age_s:
            warnings.append("quote_stale")

    # -- ratio factor error / implied spot deviation -----------------------
    if (
        consensus is not None
        and consensus > 0
        and p.bid is not None
        and p.ask is not None
        and p.financing_level is not None
    ):
        mid = (p.bid + p.ask) / 2.0
        try:
            implied = implied_underlying(mid, p.financing_level, p.ratio, p.direction, fx=1.0)
        except ValueError:
            implied = None
        if implied is not None:
            ratio_to_consensus = implied / consensus
            factor_error_exponent = None
            for k in _RATIO_FACTOR_EXPONENTS:
                factor = 10.0**k
                if abs(ratio_to_consensus / factor - 1.0) <= _RATIO_FACTOR_TOLERANCE_PCT:
                    factor_error_exponent = k
                    break
            if factor_error_exponent is not None:
                failures.append(f"ratio_factor_error_10e{factor_error_exponent}")
            else:
                deviation = abs(ratio_to_consensus - 1.0)
                if deviation > _IMPLIED_SPOT_DEVIATION_THRESHOLD_PCT:
                    failures.append("implied_spot_deviation")

    return IntegrityReport(passed=len(failures) == 0, failures=failures, warnings=warnings)
