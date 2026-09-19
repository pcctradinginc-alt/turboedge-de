"""Instrument-class cost comparison: Turbo/KO-Zertifikat vs. Mini-Future vs.
Eurex-Future, for identical underlying exposure.

Formula reference: Systemkonzept 1.2 §12.1 ("Instrumentenneutralitaet"). The
Master Spec requires the execution layer to compare instrument CLASSES, not
only turbos against each other -- and states, verbatim, that a clear result
in the future's favor is "kein Scheitern, sondern das wertvollste Ergebnis
des Projekts" (not a failure but the project's most valuable result),
estimating the order of magnitude at factor 5 to 10.

EHRLICHKEITSGEBOT (CLAUDE.md rule 29) -- the single most important property
of this module: this system ingests NO real Eurex quotes (no Eurex adapter
exists; see CLAUDE.md "Data sources"). The Turbo and Mini-Future sides are
priced from real, ingested product/underlying data wherever that data
supports it; the Eurex-Future side is ALWAYS a model on the risk-free rate
plus the placeholder constants in ``configs/instruments.yaml``, never a
measurement. Every :class:`CostLine` this module produces carries an
explicit :class:`ValueSource` (``measured`` or ``assumed``, never both, never
neither unless the amount itself is ``None``) plus a human-readable ``note``
explaining exactly where the number came from. A missing input NEVER becomes
a silent zero: :func:`wrapped_instrument_cost` and :func:`eurex_future_cost`
return ``total_cost_eur=None`` with a populated ``missing_reason`` whenever
any cost line they need cannot be determined, rather than summing only the
lines that happened to be available.

Cost blocks priced per instrument class (identical notional EUR exposure,
over one holding horizon in trading days):

- **Turbo/KO-Zertifikat** (``ProductType.TURBO_OPEN_END``): measured
  financing spread (``pricing/financing.py``'s realized-history inversion,
  act/360), measured trading (bid/ask) spread, measured issuer margin
  (``pricing/issuer_margin.py``'s ask decomposition), measured fair gap
  premium (``pricing/gap_premium.py``, from the underlying's own historical
  overnight/weekend gaps), and an expected KO loss from a simulated P(KO)
  (``simulation/paths.py`` + ``simulation/barrier.py::first_hit_index``,
  driftless -- this system claims no forecasting edge, CLAUDE.md "Current
  Milestone").
- **Mini-Future** (``ProductType.MINI_FUTURE``): identical structure, its
  own financing spread (measured the same way when the ISIN's own history
  supports it, else ``configs/instruments.yaml``'s
  ``mini_future_financing_spread_assumed``), and a real stop-loss buffer
  (``knockout_barrier != financing_level``) instead of an absorbing barrier
  --  reflected in a non-zero KO residual value.
- **Eurex-Future**: financing priced entirely at the risk-free reference
  rate (no issuer spread, no issuer margin -- an exchange-traded future has
  no bilateral issuer wrapper to price one into), no knock-out, and an
  exchange fee from ``configs/instruments.yaml`` (assumed, see that file's
  own module-level honesty comment).

This module never touches the network; it reads only an already-open
:class:`~turboedge.storage.duckdb.Store` (whose schema must already be
initialized, i.e. ``store.init_schema()`` was already called by the caller)
plus the two small dataclasses (:class:`InstrumentsConfig`,
:class:`SimulationParams`) and the reference-rate :class:`InputValue` the
caller resolves (live via ``adapters/ecb.py`` for the CLI, an explicit
offline fallback for the monthly report -- see ``reporting/monthly.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from turboedge.pricing.financing import (
    financing_cost_over_horizon,
    financing_spread_history,
    realized_financing_spread,
)
from turboedge.pricing.gap_premium import (
    GapDistribution,
    fair_gap_premium,
    gap_distribution_from_bars,
    gap_premium_over_horizon,
)
from turboedge.pricing.issuer_margin import decompose_ask
from turboedge.simulation.barrier import first_hit_index
from turboedge.simulation.paths import PathSet, simulate_paths
from turboedge.storage.duckdb import (
    _PRODUCT_SNAPSHOT_COLUMNS,
    Store,
    _row_to_product_snapshot,
)
from turboedge.storage.schemas import Direction, ProductSnapshot, ProductType, UnderlyingBar

_DAY_COUNT_BASIS = 360.0
# Matches pricing/financing.py / simulation/payoff.py's own conversion --
# see eurex_future_cost's docstring for why this module needs it too.
_CALENDAR_DAYS_PER_TRADING_DAY = 7.0 / 5.0
# Minimum distinct calendar days of financing_level history required before
# pricing/financing.py's realized_financing_spread is trusted at all (its own
# minimum is >=1 clean observation, i.e. >=2 levels -- see financing.py).
_MIN_FINANCING_LEVELS_FOR_MEASUREMENT = 2
# Per issuer, how many of the cheapest currently-quotable products to probe
# for financing-level history before picking a representative (bounds the
# number of Store.financing_level_history() calls -- a real DAX scan can
# have thousands of ISINs per issuer, most of them one-off/rarely-repeated
# quotes never worth a per-ISIN history lookup).
_MAX_REP_CANDIDATES_PER_ISSUER = 20
_BPS = 10_000.0


class ValueSource(StrEnum):
    """Whether a number is derived from real, ingested data, or is a
    documented modeling assumption. Never both; a :class:`CostLine`/
    :class:`InputValue` with ``amount``/``value`` ``None`` has
    ``source=None`` too -- the honesty mandate (CLAUDE.md rule 29) is that a
    missing input is ``None`` with a reason, never a guessed source tag on a
    guessed number."""

    MEASURED = "measured"
    ASSUMED = "assumed"


@dataclass(frozen=True, slots=True)
class InputValue:
    """One scalar input, tagged with its provenance. ``value`` is ``None``
    exactly when ``source`` is ``None`` -- the data could not be determined
    and ``note`` says why (CLAUDE.md rule 29: never silently impute)."""

    value: float | None
    source: ValueSource | None
    note: str

    def __post_init__(self) -> None:
        if (self.value is None) != (self.source is None):
            raise ValueError(
                "InputValue.value and .source must be both None or both set "
                f"(got value={self.value!r}, source={self.source!r})"
            )

    @staticmethod
    def measured(value: float, note: str) -> InputValue:
        return InputValue(value=value, source=ValueSource.MEASURED, note=note)

    @staticmethod
    def assumed(value: float, note: str) -> InputValue:
        return InputValue(value=value, source=ValueSource.ASSUMED, note=note)

    @staticmethod
    def missing(note: str) -> InputValue:
        return InputValue(value=None, source=None, note=note)


@dataclass(frozen=True, slots=True)
class CostLine:
    """One row of an instrument's cost breakdown."""

    label: str
    amount_eur: float | None
    source: ValueSource | None
    note: str
    # False for informational-only lines (e.g. a future's margin
    # requirement) that must never be summed into total_cost_eur.
    included_in_total: bool = True


@dataclass(frozen=True, slots=True)
class InstrumentCostResult:
    """Full cost breakdown for one representative instrument."""

    instrument_class: str  # "turbo_open_end" | "mini_future" | "eurex_future"
    issuer: str | None  # None for eurex_future (no issuer wrapper) or "no candidate found"
    isin: str | None
    lines: tuple[CostLine, ...]
    total_cost_eur: float | None
    total_cost_pct_of_notional: float | None
    # Populated (non-None) exactly when total_cost_eur is None -- names every
    # missing input, never silently drops a line from the sum.
    missing_reason: str | None


@dataclass(frozen=True, slots=True)
class InstrumentComparisonResult:
    """Full instrument-class comparison for one (underlying, notional,
    horizon, direction)."""

    underlying_id: str
    direction: Direction
    notional_eur: float
    horizon_days: int
    as_of: datetime
    results: tuple[InstrumentCostResult, ...]
    # Cheapest of the wrapped-instrument (turbo/mini-future) results with a
    # computable total_cost_eur, across every issuer -- None if none priced.
    cheapest_wrapped: InstrumentCostResult | None
    factor_vs_future: float | None
    factor_note: str


@dataclass(frozen=True, slots=True)
class InstrumentsConfig:
    """``configs/instruments.yaml``'s assumed constants -- see that file's
    own module-level comment for the honesty mandate and each value's
    (unverified) derivation."""

    eurex_fee_bps_of_notional: float
    eurex_fee_note: str
    eurex_margin_pct_of_notional: float
    eurex_margin_note: str
    mini_future_financing_spread_assumed: float
    mini_future_financing_spread_note: str


def load_instruments_config(path: Path) -> InstrumentsConfig:
    """Load :class:`InstrumentsConfig` from ``configs/instruments.yaml``.

    Raises:
        ValueError: if the file is missing a required key.
        OSError: if ``path`` cannot be read.
    """
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return InstrumentsConfig(
            eurex_fee_bps_of_notional=float(raw["eurex_fee_bps_of_notional"]),
            eurex_fee_note=(
                f"configs/instruments.yaml: eurex_fee_bps_of_notional="
                f"{raw['eurex_fee_bps_of_notional']} (angenommen, unverifiziert -- "
                "siehe Datei-Kommentar fuer Herleitung)."
            ),
            eurex_margin_pct_of_notional=float(raw["eurex_margin_pct_of_notional"]),
            eurex_margin_note=(
                f"configs/instruments.yaml: eurex_margin_pct_of_notional="
                f"{raw['eurex_margin_pct_of_notional']} (angenommen, unverifiziert)."
            ),
            mini_future_financing_spread_assumed=float(raw["mini_future_financing_spread_assumed"]),
            mini_future_financing_spread_note=(
                f"configs/instruments.yaml: mini_future_financing_spread_assumed="
                f"{raw['mini_future_financing_spread_assumed']} (angenommen, unverifiziert)."
            ),
        )
    except KeyError as exc:
        raise ValueError(f"{path}: fehlender Pflichtschluessel {exc}") from exc


@dataclass(frozen=True, slots=True)
class SimulationParams:
    """Path-simulation parameters for the KO-probability estimate (mirrors
    ``configs/simulation.yaml`` / ``ranking.ev.EvConfig`` -- callers pass
    their own already-loaded values through rather than this module reading
    ``configs/simulation.yaml`` itself, so it stays consistent with whatever
    the rest of a given run already used)."""

    n_paths: int
    method: str
    block_size: int
    lookback_days: int
    seed: int


# --------------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------------


def _sum_included(lines: Sequence[CostLine]) -> float:
    return sum(
        line.amount_eur for line in lines if line.included_in_total and line.amount_eur is not None
    )


def _missing_result(
    instrument_class: str, issuer: str | None, isin: str | None, reason: str
) -> InstrumentCostResult:
    return InstrumentCostResult(
        instrument_class=instrument_class,
        issuer=issuer,
        isin=isin,
        lines=(),
        total_cost_eur=None,
        total_cost_pct_of_notional=None,
        missing_reason=reason,
    )


def _resolve_fx(product: ProductSnapshot) -> InputValue:
    """Resolve the underlying-to-product FX rate. Never guesses a
    cross-currency rate (CLAUDE.md rule 29) -- see ``pricing/fx_resolution.py``
    for the full quanto-detection machinery this module deliberately does
    not replicate; it only ever prices the trivial same-currency case."""
    if product.underlying_currency is not None:
        if product.currency == product.underlying_currency:
            return InputValue.measured(
                1.0,
                f"Produktwaehrung {product.currency} == Basiswertwaehrung "
                f"{product.underlying_currency}.",
            )
        return InputValue.missing(
            f"Kreuzwaehrung {product.currency}/{product.underlying_currency} fuer ISIN "
            f"{product.isin} wird von instrument_compare.py nicht aufgeloest (siehe "
            "pricing/fx_resolution.py)."
        )
    if product.currency == "EUR":
        return InputValue.assumed(
            1.0,
            f"underlying_currency nicht vom Source gemeldet fuer ISIN {product.isin}; da "
            "Produktwaehrung EUR ist, wird Basiswaehrung EUR angenommen (nicht ueber "
            "pricing/fx_resolution.py verifiziert).",
        )
    return InputValue.missing(
        f"underlying_currency nicht gemeldet und Produktwaehrung {product.currency} != EUR "
        f"fuer ISIN {product.isin}; FX kann nicht sicher angenommen werden."
    )


def _resolve_financing_spread(
    store: Store,
    product: ProductSnapshot,
    ref_rate: InputValue,
    fallback: InputValue,
    adjustment_jump_threshold_pct: float,
) -> InputValue:
    """Prefer the realized financing spread measured from this ISIN's own
    financing-level history (``pricing/financing.py``, CLAUDE.md rule 13),
    fall back to ``fallback`` (already tagged measured/assumed by the
    caller) when there is no clean history."""
    levels = store.financing_level_history(product.isin)
    if len(levels) >= _MIN_FINANCING_LEVELS_FOR_MEASUREMENT and ref_rate.value is not None:
        observations = financing_spread_history(
            levels, ref_rate.value, product.direction, adjustment_jump_threshold_pct
        )
        spread = realized_financing_spread(observations)
        if spread is not None:
            return InputValue.measured(
                spread,
                f"realisierter Finanzierungsspread (pricing/financing.py) aus "
                f"{len(observations)} Beobachtung(en) ueber {len(levels)} "
                f"Finanzierungslevel-Tage, ISIN {product.isin}.",
            )
    return InputValue(
        value=fallback.value,
        source=fallback.source,
        note=(
            f"keine verwertbare Finanzierungslevel-Historie fuer ISIN {product.isin} "
            f"({len(levels)} Tage, benoetigt >= {_MIN_FINANCING_LEVELS_FOR_MEASUREMENT}) "
            f"-- Fallback: {fallback.note}"
        ),
    )


def _resolve_p_ko(
    paths: PathSet | None, paths_note: str, product: ProductSnapshot, sim: SimulationParams
) -> InputValue:
    if paths is None or product.knockout_barrier is None:
        reason = paths_note if paths is None else f"kein knockout_barrier fuer ISIN {product.isin}."
        return InputValue.missing(reason)
    first_idx = first_hit_index(paths, product.knockout_barrier, product.direction)
    p_ko = float(np.mean(first_idx >= 0))
    return InputValue.measured(
        p_ko,
        f"P(KO) aus {sim.n_paths} simulierten Pfaden (simulation/paths.py Methode "
        f"{sim.method!r}, simulation/barrier.py::first_hit_index), driftfrei -- keine "
        "Forecast-Praemisse unterstellt (CLAUDE.md 'Current Milestone': kein gemessener "
        "Forecast-Edge).",
    )


def _latest_snapshots_for_underlying(
    store: Store, underlying_id: str, product_type: ProductType, direction: Direction
) -> list[ProductSnapshot]:
    """The most recent ``product_snapshots`` row per ISIN for
    ``(underlying_id, product_type, direction)``.

    No existing :class:`~turboedge.storage.duckdb.Store` method lists
    snapshots by underlying (only by ISIN, which is not yet known here) --
    adding one is out of this module's file scope for this change, so this
    reaches into ``Store``'s own row-parsing helpers
    (``_PRODUCT_SNAPSHOT_COLUMNS``/``_row_to_product_snapshot``) directly
    rather than duplicating the column mapping.
    """
    columns = ", ".join(_PRODUCT_SNAPSHOT_COLUMNS)
    query = f"""
        WITH ranked AS (
            SELECT {columns},
                   ROW_NUMBER() OVER (
                       PARTITION BY isin
                       ORDER BY COALESCE(quote_timestamp, observation_time) DESC
                   ) AS rn
            FROM product_snapshots
            WHERE underlying_id = ? AND product_type = ? AND direction = ?
        )
        SELECT {columns} FROM ranked WHERE rn = 1
    """
    rows = store._conn.execute(
        query, [underlying_id, product_type.value, direction.value]
    ).fetchall()
    return [_row_to_product_snapshot(row) for row in rows]


def _select_representatives(
    products: Sequence[ProductSnapshot], store: Store
) -> dict[str, ProductSnapshot]:
    """Per issuer, pick one representative currently-tradable product.

    Among the ``_MAX_REP_CANDIDATES_PER_ISSUER`` cheapest (tightest
    bid/ask spread) quotable products of that issuer, prefer one with
    ``>= _MIN_FINANCING_LEVELS_FOR_MEASUREMENT`` days of financing-level
    history (so its financing spread CAN be measured, not just assumed),
    breaking ties by tightest spread; falls back to the plain cheapest when
    no candidate of that issuer has such history at all.
    """
    by_issuer: dict[str, list[ProductSnapshot]] = {}
    for p in products:
        if p.bid is None or p.ask is None or not (p.ask > 0) or not (p.bid <= p.ask):
            continue
        if p.financing_level is None or p.knockout_barrier is None or not (p.ratio > 0):
            continue
        by_issuer.setdefault(p.issuer, []).append(p)

    result: dict[str, ProductSnapshot] = {}
    for issuer, candidates in by_issuer.items():
        candidates.sort(key=lambda p: (p.ask - p.bid) / p.ask)  # type: ignore[operator]
        shortlist = candidates[:_MAX_REP_CANDIDATES_PER_ISSUER]

        def _rank(p: ProductSnapshot) -> tuple[bool, float]:
            n_days = len(store.financing_level_history(p.isin))
            spread = (p.ask - p.bid) / p.ask  # type: ignore[operator]
            return (n_days < _MIN_FINANCING_LEVELS_FOR_MEASUREMENT, spread)

        result[issuer] = min(shortlist, key=_rank)
    return result


# --------------------------------------------------------------------------
# per-class cost functions
# --------------------------------------------------------------------------


def wrapped_instrument_cost(
    *,
    product: ProductSnapshot,
    spot: InputValue,
    fx: InputValue,
    notional_eur: float,
    horizon_days: int,
    ref_rate: InputValue,
    financing_spread: InputValue,
    gap_dist: GapDistribution | None,
    next_night_is_weekend: bool,
    p_ko: InputValue,
    as_of: datetime,
) -> InstrumentCostResult:
    """Total cost of ``notional_eur`` identical underlying exposure held for
    ``horizon_days`` in ``product`` (a Turbo/KO-Zertifikat or Mini-Future).

    Five cost lines, each independently tagged measured/assumed/missing:
    trading (round-trip bid/ask) spread, financing cost over the horizon,
    the issuer margin already embedded in the current ask (one-time, not
    horizon-scaled -- it is priced into the ask at the moment of purchase),
    the fair gap premium over the horizon, and the expected KO loss implied
    by a simulated P(KO). ``total_cost_eur`` is ``None`` (with
    ``missing_reason`` naming every missing line) whenever ANY of the five
    cannot be computed -- never a silent partial sum (CLAUDE.md rule 29).
    """
    instrument_class = product.product_type.value

    if spot.value is None:
        return _missing_result(instrument_class, product.issuer, product.isin, f"Spot: {spot.note}")
    if fx.value is None:
        return _missing_result(instrument_class, product.issuer, product.isin, f"FX: {fx.note}")
    if product.bid is None or product.ask is None:
        return _missing_result(
            instrument_class,
            product.issuer,
            product.isin,
            f"kein bid/ask fuer ISIN {product.isin}.",
        )
    assert product.financing_level is not None  # guaranteed by _select_representatives
    assert product.knockout_barrier is not None

    units = notional_eur / (spot.value * product.ratio / fx.value)

    lines: list[CostLine] = []
    missing: list[str] = []

    # 1) trading spread: entry always ask, exit always bid (CLAUDE.md rule 11).
    lines.append(
        CostLine(
            label="Handelsspread (Geld/Brief, Round-Trip)",
            amount_eur=(product.ask - product.bid) * units,
            source=ValueSource.MEASURED,
            note=(
                f"gemessen aus aktuellem bid={product.bid:.6g}/ask={product.ask:.6g}, "
                f"ISIN {product.isin}."
            ),
        )
    )

    # 2) financing cost over the horizon.
    if financing_spread.value is None or ref_rate.value is None:
        reason = financing_spread.note if financing_spread.value is None else ref_rate.note
        lines.append(CostLine("Finanzierungskosten", None, None, reason))
        missing.append(f"Finanzierungskosten: {reason}")
    else:
        amount = (
            financing_cost_over_horizon(
                product.financing_level,
                financing_spread.value,
                ref_rate.value,
                float(horizon_days),
                product.ratio,
                product.direction,
                fx.value,
            )
            * units
        )
        source = (
            ValueSource.MEASURED
            if financing_spread.source == ValueSource.MEASURED
            and ref_rate.source == ValueSource.MEASURED
            else ValueSource.ASSUMED
        )
        lines.append(
            CostLine(
                "Finanzierungskosten",
                amount,
                source,
                f"Spread: {financing_spread.note} | Referenzzins: {ref_rate.note}",
            )
        )

    # 3) issuer margin already embedded in the current ask (pricing/issuer_margin.py).
    if gap_dist is None:
        reason = "keine Gap-Verteilung verfuegbar (siehe Gap-Praemie-Zeile)."
        lines.append(CostLine("Emittentenmarge (im Ask enthalten)", None, None, reason))
        missing.append(f"Emittentenmarge: {reason}")
    elif financing_spread.value is None or ref_rate.value is None:
        reason = financing_spread.note if financing_spread.value is None else ref_rate.note
        lines.append(CostLine("Emittentenmarge (im Ask enthalten)", None, None, reason))
        missing.append(f"Emittentenmarge: {reason}")
    else:
        try:
            fgp_one_night = fair_gap_premium(
                spot.value,
                product.financing_level,
                product.ratio,
                product.direction,
                gap_dist,
                next_night_is_weekend,
                fx.value,
            )
        except ValueError as exc:
            reason = f"Gap-Verteilung unzureichend fuer Emittentenmarge-Isolierung: {exc}"
            lines.append(CostLine("Emittentenmarge (im Ask enthalten)", None, None, reason))
            missing.append(f"Emittentenmarge: {reason}")
        else:
            costs = decompose_ask(
                product.bid,
                product.ask,
                spot.value,
                product.financing_level,
                product.ratio,
                product.direction,
                fgp_one_night,
                financing_spread.value,
                ref_rate.value,
                fx.value,
                product_type=product.product_type,
                knockout_barrier=product.knockout_barrier,
                as_of=as_of.date(),
            )
            margin_amount = max(costs.issuer_margin, 0.0) * units
            source = (
                ValueSource.MEASURED
                if financing_spread.source == ValueSource.MEASURED
                and ref_rate.source == ValueSource.MEASURED
                else ValueSource.ASSUMED
            )
            lines.append(
                CostLine(
                    "Emittentenmarge (im Ask enthalten)",
                    margin_amount,
                    source,
                    "Ask-Zerlegung (pricing/issuer_margin.py): rohe Marge "
                    f"{costs.issuer_margin:.6g} EUR/Stk; negative Marge nicht als Kosten "
                    "gezaehlt (max(..., 0)).",
                )
            )

    # 4) fair gap premium over the full horizon (pricing/gap_premium.py).
    if gap_dist is None:
        reason = "keine Gap-Verteilung verfuegbar (zu wenige historische Kursbars des Basiswerts)."
        lines.append(CostLine("Gap-Praemie (Overnight/Weekend, Horizont)", None, None, reason))
        missing.append(f"Gap-Praemie: {reason}")
    else:
        try:
            gap_amount = (
                gap_premium_over_horizon(
                    spot.value,
                    product.financing_level,
                    product.ratio,
                    product.direction,
                    gap_dist,
                    horizon_days,
                    fx.value,
                )
                * units
            )
        except ValueError as exc:
            reason = f"Gap-Verteilung unzureichend: {exc}"
            lines.append(CostLine("Gap-Praemie (Overnight/Weekend, Horizont)", None, None, reason))
            missing.append(f"Gap-Praemie: {reason}")
        else:
            lines.append(
                CostLine(
                    "Gap-Praemie (Overnight/Weekend, Horizont)",
                    gap_amount,
                    ValueSource.MEASURED,
                    f"gemessen aus historischen Overnight/Weekend-Gaps des Basiswerts ueber "
                    f"{horizon_days} Handelstage (pricing/gap_premium.py).",
                )
            )

    # 5) expected KO loss implied by the simulated P(KO).
    if p_ko.value is None:
        lines.append(CostLine("Erwarteter KO-Verlust", None, None, p_ko.note))
        missing.append(f"Erwarteter KO-Verlust: {p_ko.note}")
    else:
        if product.product_type == ProductType.MINI_FUTURE:
            residual_per_cert = (
                abs(product.knockout_barrier - product.financing_level) * product.ratio / fx.value
            )
            residual_note = (
                "Stop-Puffer (|Barriere - Finanzierungslevel| * Bezugsverhaeltnis), "
                "heutiger Stand -- nicht auf den (zukuenftigen) KO-Tag vorgerollt, "
                "Vereinfachung."
            )
        else:
            residual_per_cert = 0.0
            residual_note = (
                "strukturell 0: turbo_open_end hat Barriere == Finanzierungslevel per Definition."
            )
        loss_given_ko = max(product.ask - residual_per_cert, 0.0)
        ko_amount = p_ko.value * loss_given_ko * units
        lines.append(
            CostLine(
                "Erwarteter KO-Verlust",
                ko_amount,
                p_ko.source,
                f"P(KO in {horizon_days}d)={p_ko.value:.4%} ({p_ko.note}) * Restverlust "
                f"{loss_given_ko:.6g} EUR/Stk (Restwert bei KO: {residual_per_cert:.6g} EUR/Stk, "
                f"{residual_note}).",
            )
        )

    if missing:
        return InstrumentCostResult(
            instrument_class=instrument_class,
            issuer=product.issuer,
            isin=product.isin,
            lines=tuple(lines),
            total_cost_eur=None,
            total_cost_pct_of_notional=None,
            missing_reason="; ".join(missing),
        )

    total = _sum_included(lines)
    return InstrumentCostResult(
        instrument_class=instrument_class,
        issuer=product.issuer,
        isin=product.isin,
        lines=tuple(lines),
        total_cost_eur=total,
        total_cost_pct_of_notional=total / notional_eur,
        missing_reason=None,
    )


def eurex_future_cost(
    *,
    notional_eur: float,
    horizon_days: int,
    ref_rate: InputValue,
    instruments_cfg: InstrumentsConfig,
) -> InstrumentCostResult:
    """Total modeled cost of ``notional_eur`` exposure via an Eurex DAX
    future, held ``horizon_days``. See module docstring's EHRLICHKEITSGEBOT:
    every line here is either the reference rate (whatever the caller
    tagged it, typically ``measured`` via ``adapters/ecb.py`` or ``assumed``
    when unavailable offline) or an ``assumed`` constant from
    ``configs/instruments.yaml`` -- NEVER a real Eurex quote, none exist in
    this system.

    Calendar days are approximated from trading days as ``horizon_days *
    7/5``, exactly like ``pricing/financing.py``'s
    ``financing_cost_over_horizon`` -- both sides of the comparison must use
    the identical day-count convention for the same nominal ``horizon_days``
    or the comparison silently stops being apples-to-apples.
    """
    lines: list[CostLine] = []
    missing: list[str] = []
    calendar_days = horizon_days * _CALENDAR_DAYS_PER_TRADING_DAY

    if ref_rate.value is None:
        lines.append(
            CostLine("Finanzierung (impliziter Future-Basis, Marktzins)", None, None, ref_rate.note)
        )
        missing.append(f"Finanzierung: {ref_rate.note}")
    else:
        financing_amount = notional_eur * ref_rate.value * calendar_days / _DAY_COUNT_BASIS
        lines.append(
            CostLine(
                "Finanzierung (impliziter Future-Basis, Marktzins)",
                financing_amount,
                ref_rate.source,
                "Modell notional * r * calendar_days/360 (act/360, calendar_days = "
                "horizon_days * 7/5 wie pricing/financing.py), "
                f"r: {ref_rate.note} KEIN gemessener Future-Preis -- kein Eurex-Adapter in diesem "
                "System; Annahme 'Finanzierung == Marktzins, keine zusaetzliche Emittentenmarge' "
                "(boersengehandelter Future hat keinen bilateralen Emittenten-Wrapper).",
            )
        )

    lines.append(
        CostLine(
            "Boersengebuehr (Eurex, angenommen)",
            notional_eur * instruments_cfg.eurex_fee_bps_of_notional / _BPS,
            ValueSource.ASSUMED,
            instruments_cfg.eurex_fee_note,
        )
    )
    lines.append(
        CostLine(
            "Emittentenmarge",
            0.0,
            ValueSource.ASSUMED,
            "strukturelle Annahme: boersengehandelter Future hat keinen bilateralen "
            "Emittenten-Wrapper, der eine Marge einpreist -- nicht gegen echte Eurex-Quotes "
            "verifiziert (keine vorhanden).",
        )
    )
    lines.append(
        CostLine(
            "Erwarteter KO-Verlust",
            0.0,
            ValueSource.ASSUMED,
            "strukturelle Annahme: ein Future hat keinen Knock-out-Mechanismus.",
        )
    )
    lines.append(
        CostLine(
            "Margin-Anforderung (informativ, NICHT in Summe)",
            notional_eur * instruments_cfg.eurex_margin_pct_of_notional,
            ValueSource.ASSUMED,
            instruments_cfg.eurex_margin_note,
            included_in_total=False,
        )
    )

    if missing:
        return InstrumentCostResult(
            instrument_class="eurex_future",
            issuer=None,
            isin=None,
            lines=tuple(lines),
            total_cost_eur=None,
            total_cost_pct_of_notional=None,
            missing_reason="; ".join(missing),
        )

    total = _sum_included(lines)
    return InstrumentCostResult(
        instrument_class="eurex_future",
        issuer=None,
        isin=None,
        lines=tuple(lines),
        total_cost_eur=total,
        total_cost_pct_of_notional=total / notional_eur,
        missing_reason=None,
    )


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def compare_instruments(
    store: Store,
    *,
    underlying_id: str,
    notional_eur: float,
    horizon_days: int,
    direction: Direction,
    ref_rate: InputValue,
    financing_spread_fallback: InputValue,
    financing_adjustment_jump_threshold_pct: float,
    instruments_cfg: InstrumentsConfig,
    sim: SimulationParams,
    as_of: datetime | None = None,
) -> InstrumentComparisonResult:
    """Build the full instrument-class comparison against existing (already
    ingested) local state -- never fetches from the network, never mutates
    the store. ``store.init_schema()`` must already have been called.

    Never raises for missing/insufficient DATA (CLAUDE.md rule 29: a data
    gap becomes an explicit ``missing_reason`` on the affected
    :class:`InstrumentCostResult`, not an exception) -- only for genuinely
    invalid arguments (``notional_eur <= 0``, ``horizon_days <= 0``).
    """
    if not (notional_eur > 0):
        raise ValueError(f"notional_eur must be > 0, got {notional_eur!r}")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be > 0, got {horizon_days!r}")

    resolved_as_of = as_of if as_of is not None else _utc_now()

    bars: list[UnderlyingBar] = store.latest_underlying_bars(underlying_id, sim.lookback_days + 5)
    spot: InputValue
    gap_dist: GapDistribution | None
    paths: PathSet | None
    paths_note: str
    if len(bars) < 2:
        spot = InputValue.missing(
            f"keine ausreichenden historischen Kursbars fuer {underlying_id} in vorhandenen Daten "
            f"({len(bars)} Bar(s))."
        )
        gap_dist = None
        paths = None
        paths_note = "Pfadsimulation nicht moeglich: keine ausreichenden historischen Kursbars."
    else:
        last_bar = bars[-1]
        spot = InputValue.measured(
            last_bar.close,
            f"letzter Schlusskurs {underlying_id} vom {last_bar.ts.date().isoformat()}.",
        )
        gap_dist = gap_distribution_from_bars(bars)
        try:
            rng = np.random.default_rng(sim.seed)
            paths = simulate_paths(
                bars,
                spot0=last_bar.close,
                start=last_bar.ts + timedelta(days=1),
                horizon_days=horizon_days,
                n_paths=sim.n_paths,
                rng=rng,
                drift_log_return=None,  # driftfrei -- kein gemessener Forecast-Edge (CLAUDE.md)
                method=sim.method,  # type: ignore[arg-type]
                block_size=sim.block_size,
                lookback_days=sim.lookback_days,
            )
            paths_note = ""
        except ValueError as exc:
            paths = None
            paths_note = f"Pfadsimulation fehlgeschlagen: {exc}"

    next_night_is_weekend = resolved_as_of.weekday() == 4  # Friday (UTC-day approximation)

    results: list[InstrumentCostResult] = []

    for product_type, class_fallback, class_label in (
        (
            ProductType.TURBO_OPEN_END,
            financing_spread_fallback,
            "turbo_open_end",
        ),
        (
            ProductType.MINI_FUTURE,
            InputValue.assumed(
                instruments_cfg.mini_future_financing_spread_assumed,
                instruments_cfg.mini_future_financing_spread_note,
            ),
            "mini_future",
        ),
    ):
        snapshots = _latest_snapshots_for_underlying(store, underlying_id, product_type, direction)
        reps = _select_representatives(snapshots, store)
        if not reps:
            results.append(
                _missing_result(
                    class_label,
                    None,
                    None,
                    f"keine handelbaren {class_label}-Produkte fuer "
                    f"{underlying_id}/{direction.value} in den vorhandenen Daten gefunden.",
                )
            )
            continue
        for _issuer, product in sorted(reps.items()):
            fx = _resolve_fx(product)
            financing_spread = _resolve_financing_spread(
                store, product, ref_rate, class_fallback, financing_adjustment_jump_threshold_pct
            )
            p_ko = _resolve_p_ko(paths, paths_note, product, sim)
            results.append(
                wrapped_instrument_cost(
                    product=product,
                    spot=spot,
                    fx=fx,
                    notional_eur=notional_eur,
                    horizon_days=horizon_days,
                    ref_rate=ref_rate,
                    financing_spread=financing_spread,
                    gap_dist=gap_dist,
                    next_night_is_weekend=next_night_is_weekend,
                    p_ko=p_ko,
                    as_of=resolved_as_of,
                )
            )

    future_result = eurex_future_cost(
        notional_eur=notional_eur,
        horizon_days=horizon_days,
        ref_rate=ref_rate,
        instruments_cfg=instruments_cfg,
    )
    results.append(future_result)

    wrapped_priced = [
        r for r in results if r.instrument_class != "eurex_future" and r.total_cost_eur is not None
    ]
    cheapest_wrapped: InstrumentCostResult | None = (
        min(wrapped_priced, key=lambda r: r.total_cost_eur or 0.0) if wrapped_priced else None
    )

    factor_vs_future: float | None = None
    if cheapest_wrapped is None:
        factor_note = (
            "Faktor nicht berechenbar: kein Turbo/Mini-Future mit vollstaendigen Kosten gepreist."
        )
    elif future_result.total_cost_eur is None:
        factor_note = f"Faktor nicht berechenbar: {future_result.missing_reason}"
    elif future_result.total_cost_eur == 0.0:
        factor_note = "Faktor nicht berechenbar: modellierte Future-Gesamtkosten sind 0."
    else:
        cheapest_total = cheapest_wrapped.total_cost_eur
        future_total = future_result.total_cost_eur
        assert cheapest_total is not None  # guaranteed by the wrapped_priced filter above
        factor_vs_future = cheapest_total / future_total
        factor_note = (
            "Faktor = guenstigste Turbo/Mini-Future-Gesamtkosten "
            f"({cheapest_wrapped.instrument_class}, {cheapest_wrapped.issuer}, "
            f"{cheapest_total:.2f} EUR) / modellierte Future-Gesamtkosten "
            f"({future_total:.2f} EUR) = {factor_vs_future:.2f}x. "
            "Future-Seite ist ein Modell (siehe eurex_future_cost), keine Messung."
        )

    return InstrumentComparisonResult(
        underlying_id=underlying_id,
        direction=direction,
        notional_eur=notional_eur,
        horizon_days=horizon_days,
        as_of=resolved_as_of,
        results=tuple(results),
        cheapest_wrapped=cheapest_wrapped,
        factor_vs_future=factor_vs_future,
        factor_note=factor_note,
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "CostLine",
    "InputValue",
    "InstrumentComparisonResult",
    "InstrumentCostResult",
    "InstrumentsConfig",
    "SimulationParams",
    "ValueSource",
    "compare_instruments",
    "eurex_future_cost",
    "load_instruments_config",
    "wrapped_instrument_cost",
]
