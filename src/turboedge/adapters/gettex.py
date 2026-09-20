"""gettex (Boerse Muenchen) leverage-product feed adapter.

Confirmed live and reverse-engineered by the Round 3 data-source research
session -- see ``docs/data_sources.md`` (section 9 and its follow-up 9a) for
the full research log, every probe, pitfall and fixture referenced below.
gettex is a **multi-issuer marketplace aggregator**, not a single issuer's own
site: BNP Paribas, Goldman Sachs, HSBC and UniCredit market-maker quotes were
all observed live in the same feed, across DAX/Nasdaq-100/S&P 500/Euro Stoxx
50. This is the single biggest cross-issuer-coverage unblock available to
this milestone (BNP+Citi gave 2 issuer perspectives; BNP+Citi+gettex gives 5,
from 3 independently-operated sources).

Endpoint
--------
gettex's own site (``www.gettex.de``) embeds a third-party, white-label
finder widget (vendor "Solvians"/wsd.com) via an iframe. The widget's REST
API lives on its own origin, ``gettex.wsd.com``, is not gated behind the
iframe (confirmed live with a bare, cookie-less client, no ``Origin``/
``Referer`` needed), and has no anti-bot signature, no Cloudflare challenge,
no auth of any kind::

    GET https://gettex.wsd.com/page-api/products/DE/leverageProducts
        ?underlying=<underlyingId>      (int, static per canonical underlying
                                          -- see _GETTEX_UNDERLYING_IDS)
        &productType=<German label>     ("Turbos endlos" -- open-end turbos)
        &rowsPerPage=<10|20|50|100>
        &page=<1-based>

Response shape (``data.groups.products[]``): flat objects, one key per
display column, several with a literal dot in the key name (NOT nested
JSON) -- ``wkn``, ``isin``, ``underlying`` (name), ``issuer``,
``structure.direction`` ("long"/"short"), ``financingLevelRefCurAbsolute``,
``koLevelRefCurAbsolute`` (== financing level for every open-end turbo/mini
sampled -- barrier=strike design), ``leverage``, ``bid``/``ask`` (each a
``{valueTuple: {value, size, timestamp}}`` wrapper, ``timestamp`` a plain
epoch-millisecond UTC integer -- the cleanest timestamp field of any source
integrated in this project so far, no naive-local-time handling needed
unlike BNP/Citi), ``underlyings.price`` (gettex's own live reference price
for the underlying, same ``valueTuple`` shape).

The genuine, structural gap (Round 3 follow-up 9a): **no endpoint reachable
from gettex, for any issuer, returns ``ratio`` (Bezugsverhaeltnis),
``currency``, ``maturity`` or ``quanto``.** Four separate hypotheses were
tested live and disproven (single-ISIN detail endpoint, the certificate
detail page's third-party LSEG/financial.com widget stack -- a different,
unrelated vendor with zero certificate-ISIN coverage --, an undocumented
``fields=`` expand parameter, and the sibling ``investmentProducts`` list
id). This is not a "didn't look hard enough" gap: gettex's own UI never
needs to *display* per-unit ratio (its comparison table shows absolute price
levels), so the field is simply never wired to any endpoint gettex exposes.

Rather than silently imputing ``ratio=1`` or any other default (CLAUDE.md
rule 29: "Fehlende Daten niemals still imputieren, wenn sie fuer Pricing
kritisch sind"), this adapter **derives and then verifies** the ratio from
data gettex *does* expose, using a relationship that is structurally
independent of ratio:

1. **Reference spot (S_ref)**: a turbo/KO certificate's leverage, by
   construction, is ``leverage = S / |S - F|`` (identical to
   ``pricing/intrinsic.leverage`` in the in-the-money region where price ==
   intrinsic value, with no ratio/fx term at all -- turbos are priced at
   ~intrinsic value by design, unlike vanilla options). Inverting for S:

   - Long:  ``S = F * leverage / (leverage - 1)``
   - Short: ``S = F * leverage / (leverage + 1)``

   Every row in a fetch (regardless of bid/ask availability) yields one S
   candidate this way. A robust, MAD-outlier-filtered, direction-balanced
   median (the exact pattern already used by
   ``pricing/cross_issuer.consensus_spot`` for the analogous cross-issuer
   spot estimate -- duplicated here in miniature rather than imported, since
   this runs *before* any ratio is known and ``consensus_spot`` requires one)
   over the *fresh* (non-stale), *moderate-leverage* (<=
   ``max_leverage_for_reference_spot``, default 200) rows gives ``S_ref``.
   Long/Short direction balancing matters here for the same documented
   reason as ``cross_issuer.py``: gettex's own ``leverage`` field is
   evidently computed from an actual quoted price (bid, when ask is the "no
   live ask" sentinel -- see Pitfall 2 below), not the pure geometric
   ``S/|S-F|`` identity, so a per-row S estimate carries the same small,
   systematically-signed wrapper-margin bias documented there; averaging the
   LONG-median and SHORT-median S estimates cancels it to first order. The
   leverage cap is a separate, numerical-conditioning concern (see
   ``_DEFAULT_MAX_LEVERAGE_FOR_REFERENCE_SPOT``'s own docstring): moneyness
   ``|S-F| = S/leverage`` shrinks as leverage grows, so any fixed absolute
   noise in a high-leverage row's own S estimate is a proportionally huge
   fraction of its true (tiny) moneyness -- confirmed live (2026-09 DAX
   validation) to swing individual high-leverage candidates by hundreds of
   index points, degrading the median far more than a MAD filter alone
   catches. The cap only gates *S_ref construction*; high-leverage rows are
   still fully eligible for ratio derivation once a clean ``S_ref`` exists.

2. Callers may additionally supply ``reference_spot`` (a genuinely
   independent external value, e.g. same-session yfinance) to
   :meth:`GettexAdapter.fetch_products`/``normalize``; when given, ``S_ref``
   is checked against it (default tolerance 0.5%) and *all* products for that
   underlying are dropped (never priced from an unverified S_ref) if it
   deviates beyond tolerance -- surfaced via ``last_errors`` and
   :meth:`GettexAdapter.healthcheck`. This is deliberately optional: this
   adapter never reaches into another adapter's territory (CLAUDE.md rule 27,
   "Jede Source hinter Adapter kapseln") to fetch yfinance data itself --
   cross-validating one source's derived numbers against another live source
   is the scan pipeline's job (see ``pipeline/scan.py._resolve_spot`` for the
   established pattern with BNP's ``underlying_price_ref``), not this
   adapter's.

3. **Ratio**: with ``S_ref`` established, ``ratio_raw = price * fx /
   |S_ref - F|`` (``price`` = mid when both bid/ask are live, else bid alone
   -- the same documented, one-sided bid-only approximation
   ``cross_issuer.consensus_spot`` already uses). ``ratio_raw`` is snapped to
   the canonical Bezugsverhaeltnis grid (``_RATIO_GRID``) only when the
   relative deviation to the nearest grid value is under
   ``ratio_snap_tolerance_pct`` (default 3%); otherwise the product is
   dropped as master-data-incomplete (``ratio_rejected``).

4. **Verification**: the snapped ratio must reproduce ``S_ref`` via the exact
   algebraic inverse (``pricing.intrinsic.implied_underlying``) within
   ``implied_spot_tolerance_pct`` (default 0.3%); otherwise dropped
   (``verification_failed``). This also catches a product whose price
   silently includes time value from a fixed maturity (a dated,
   non-open-end structure) -- its bid/ask would not reproduce S_ref from a
   pure-intrinsic ratio guess, so it is rejected rather than mispriced.

5. **Currency/quanto**: gettex trades exclusively on Boerse Muenchen in EUR,
   so ``currency="EUR"`` always. For a EUR-denominated underlying (DAX, Euro
   Stoxx 50) ``fx=1`` is the only hypothesis and ``quanto`` stays ``None``
   (the concept does not apply -- no FX conversion is involved either way).
   For a non-EUR underlying (Nasdaq 100, S&P 500 -- both USD), two
   hypotheses are tried -- quanto (``fx=1``) and non-quanto (``fx=fx_hint``)
   -- and the product is only accepted (with ``quanto`` set accordingly)
   when *exactly one* hypothesis passes step 4's verification; zero passing
   is ``verification_failed``, more than one is ``quanto_ambiguous`` (tracked
   as a distinct, separately-logged outcome from a plain verification
   failure) -- either way, genuine ambiguity is never guessed.

   Befund 2 (2026-09-13 measurement session): the generic
   ``ProductSourceAdapter.fetch_products(underlying_ids)`` contract (see
   ``adapters/registry.py``) never actually passes this ``fx_hint`` kwarg --
   in production it is always ``None``, live-measured to leave a non-EUR
   underlying limited to the quanto-only hypothesis and a ~5% (Nasdaq 100)
   derivation rate. When the caller doesn't supply ``fx_hint``,
   :meth:`GettexAdapter._learn_fx_by_issuer` learns a per-(issuer,
   underlying_id) non-quanto fx candidate from this *same* fetch's own data
   instead (grid-ratio inversion + a robust MAD-filtered median across each
   issuer's fx=1-unresolved rows, validated by requiring a literal majority
   of those rows to actually verify under the learned value before it is
   used at all -- see that method's docstring and the
   ``_DEFAULT_FX_LEARNING_*`` module constants for the full rationale,
   live-measured numbers and the safety gate that discriminates a
   good learned estimate from a bad one). An explicit caller-supplied
   ``fx_hint`` always takes precedence over the learned value when both
   exist for the same underlying.

6. **Maturity**: never present, never derivable, never invented.
   ``maturity=None`` and ``first_trading_day=None`` always.
   ``open_end=True``/``product_type=TURBO_OPEN_END`` only when the barrier
   equals the financing level (the open-end turbo/mini design, true in every
   sample seen) *and* the row survived verification above; otherwise
   ``product_type=UNKNOWN``/``open_end=False`` with a ``last_errors`` entry
   -- never assumed open-ended just because the request filtered on
   gettex's own "Turbos endlos" product-type label (the same "never trust a
   filter's effect without checking the actual row" lesson Citi's
   silently-ignored ISIN filter taught in Round 2).

Pitfall 1 (staleness): a ``valueTuple`` key can be present with a real but
almost-year-old timestamp on an illiquid/dead product (observed:
``bid.value=0.001`` with a ~1-year-old timestamp while the same fetch's other
rows were minutes old). Per-row freshness (age <= ``stale_after_s``, default
900s) is checked before a row is allowed to *contribute to the S_ref median*
(a stale leverage/price pair would bias the reference spot toward an old
underlying level) -- but a stale row can still independently earn its own
ratio and appear in the output (``is_stale=True``, lower ``quality_score``),
via the same verification gate as any other row: if its stale price no
longer reproduces the *current* S_ref within tolerance, it is naturally
rejected exactly as any bad ratio guess would be, with no separate
staleness-specific exclusion rule needed.

Pitfall 2 (ask==0.0 sentinel, same family as BNP/Citi): a meaningful minority
of rows (concentrated among very-high-leverage/near-knockout products) have
``ask.valueTuple.value == 0.0`` with ``size == 0.0`` while ``bid`` is a
plausible live value -- treated identically to BNP's absent-``ask``-key and
Citi's ``ask==0.0`` sentinels: ``bid_only=True``, not a
``bid <= ask`` integrity violation.

Neither this adapter nor any upstream step imputes a missing/zero bid or ask
(CLAUDE.md rule 29): a missing or zero-sentinel price is always represented
as ``None`` with ``quote_presence=False``, never guessed from the other side
of the market.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog

from turboedge.adapters.base import (
    AdapterError,
    AdapterHttpError,
    AdapterMetadata,
    HealthCheckResult,
    HttpClient,
    ProductFetchContext,
)
from turboedge.config import SourceConfig
from turboedge.pricing.intrinsic import implied_underlying
from turboedge.storage.schemas import (
    Direction,
    FieldReliability,
    HealthStatus,
    ProductSnapshot,
    ProductType,
    RatioDerivationOutcome,
    RejectedRatioDerivation,
)
from turboedge.universe.underlying_map import get_underlying_meta, resolve_underlying_id

logger = structlog.get_logger(__name__)

_GETTEX_PARSER_VERSION = "gettex/1"
_GETTEX_SOURCE_NAME = "gettex"
_GETTEX_VENUE = "gettex"
_GETTEX_DEFAULT_BASE_URL = "https://gettex.wsd.com"
_GETTEX_PRODUCTS_PATH = "/page-api/products/DE/leverageProducts"
_GETTEX_PRODUCT_TYPE_LABEL = "Turbos endlos"
# Confirmed live (Round 3 research): 10/20/50/100 are the only accepted
# rowsPerPage values; 100 is the largest.
_GETTEX_ROWS_PER_PAGE = 100

# Canonical underlying_id -> gettex's own numeric `underlying` query param,
# confirmed live this research session (docs/data_sources.md section 9).
# An underlying_id not in this table is skipped cleanly with a log line,
# never guessed (CLAUDE.md rule 29) -- gettex's site covers many more
# underlyings than this project's canonical vocabulary, and mapping every one
# of them was out of scope for this research.
_GETTEX_UNDERLYING_IDS: dict[str, int] = {
    "DAX": 1,
    "NDX": 526295,
    "SPX": 451413,
    "ESTX50": 450495,
}

_GETTEX_DIRECTION_MAP: dict[str, Direction] = {"long": Direction.LONG, "short": Direction.SHORT}

# Canonical Bezugsverhaeltnis (ratio) grid. A derived ratio_raw is only
# accepted when it lands within `ratio_snap_tolerance_pct` of one of these --
# never accepted "as-is" (an un-snapped float would just be re-encoding
# derivation noise as a fake-precise ratio).
_RATIO_GRID: tuple[float, ...] = (
    10.0,
    5.0,
    2.0,
    1.0,
    0.5,
    0.2,
    0.1,
    0.05,
    0.02,
    0.01,
    0.005,
    0.002,
    0.001,
    0.0001,
)

_DEFAULT_STALE_AFTER_S = 900.0
_DEFAULT_REFERENCE_SPOT_TOLERANCE_PCT = 0.005  # 0.5%, two live quotes close in time
# 2026-09-13 integration finding: the scan pipeline's only `reference_spot`
# source for gettex was another issuer's (BNP's) own live spot -- when BNP's
# API returned HTTP 500, gettex's cross-check received a stale/unavailable
# value and rejected every DAX/Nasdaq 100 product this adapter had otherwise
# correctly derived and verified on its own, purely because its one external
# validation source shared BNP's outage (a single point of failure this
# adapter's S_ref never actually needed -- S_ref is always derived
# internally from gettex's own leverage identity, module docstring step 1).
# `daily_close_reference` (a same-day yfinance-style close, a source with no
# dependency on any single issuer's uptime) is the fallback sanity check when
# `reference_spot` isn't supplied. Its tolerance must be WIDER than the
# live-vs-live 0.5% above: a prior close can legitimately be a full session
# away from the current live level. 2% comfortably covers ordinary
# single-day moves for the equity indices this project covers (DAX/Nasdaq
# 100 daily realized vol is typically ~1-1.5%, so 2% is roughly a 1.3-2
# standard-deviation band -- generous enough not to reject a normal trading
# day's drift) while still catching a genuinely broken/stale reference (an
# order-of-magnitude-off value, a wrong underlying, a unit mismatch) exactly
# as the tight cross-check does for a live quote.
_DEFAULT_DAILY_CLOSE_REFERENCE_TOLERANCE_PCT = 0.02  # 2%
_DEFAULT_RATIO_SNAP_TOLERANCE_PCT = 0.03  # 3%, base value for a low-leverage row
_DEFAULT_IMPLIED_SPOT_TOLERANCE_PCT = 0.003  # 0.3%
_DEFAULT_MAD_K = 5.0

# Befund 2 (2026-09-13 measurement session) fix 1 -- leverage-scaled
# ratio-snap tolerance. A live diagnostic pull (2000-row DAX sample) found
# the FIXED 3% `ratio_snap_tolerance_pct` alone explains most of the "43.5%
# DAX rows fail for no quanto-related reason" gap flagged in Befund 2: rows
# with leverage 20-35 passed the fixed 3% snap 84.3% of the time, but rows
# with leverage 50-75 passed only 0.2% of the time, and >=75 essentially
# 0%(!) -- yet when the snap step was relaxed to simply take the NEAREST
# canonical grid ratio (no cutoff at all) and let the existing, unchanged,
# strict 0.3% `implied_spot_tolerance_pct` verification gate decide
# correctness, 97-99.7% of those same high-leverage rows verified correctly.
# This is the exact same moneyness-shrinks-with-leverage effect already
# documented on `_DEFAULT_MAX_LEVERAGE_FOR_REFERENCE_SPOT` above (moneyness
# = S/leverage, so a fixed absolute bid/ask noise/rounding in `price`
# produces a `ratio_raw = price/moneyness` deviation that grows
# proportionally with leverage) -- it was previously only compensated for in
# the S_ref *construction* step, not in each row's own ratio-snap step.
# Scaling the snap tolerance with leverage (capped, see
# `_DEFAULT_RATIO_SNAP_TOLERANCE_CAP_PCT`) recovers these rows safely
# because the snap step only proposes a *candidate* ratio -- the unchanged,
# tight 0.3% implied-spot verification is what actually decides correctness
# (CLAUDE.md rule 29: this never accepts an unverified guess); live-measured
# false-accept rate (candidate snaps within the relaxed tolerance but then
# ALSO fails verification, i.e. would have been silently wrong without the
# verification gate) was 4 rows out of 2000 (0.2%) on the DAX sample -- all
# 4 caught and rejected by verification as designed, zero reaching output.
_DEFAULT_RATIO_SNAP_TOLERANCE_LEVERAGE_REF = 20.0
_DEFAULT_RATIO_SNAP_TOLERANCE_CAP_PCT = 0.50  # 50% -- outer sanity bound only

# Befund 2 fix 2 -- per-(issuer, underlying) quanto/fx majority learning.
# The generic `ProductSourceAdapter.fetch_products(underlying_ids)` contract
# (see `adapters/registry.py`) never passes this adapter's optional
# `fx_hint` kwarg, so in production `fx_hint` is always `None` -- for a
# non-EUR underlying (Nasdaq 100, S&P 500) only the quanto (fx=1) hypothesis
# was ever actually tested, live-measured (2026-09-13, 2000-row NDX sample)
# at just 5.0% ratio_derived (matching Befund 2's reported 3.3% within
# session-to-session data variance). Per CLAUDE.md rule 29, silently
# defaulting to "assume quanto" or inventing a hardcoded EURUSD constant are
# both forbidden -- but not deriving a candidate fx from the SAME fetch's own
# data is not similarly forbidden, provided it is validated before use, never
# applied blind, logged, and every row still passes through the unchanged
# implied-spot verification gate. See `GettexAdapter._learn_fx_by_issuer`
# for the two-phase method: (1) rows already unambiguous under fx=1 alone
# are left alone; (2) for each issuer's remaining (fx=1-unresolved) rows, a
# candidate fx is proposed via grid-ratio inversion
# (`fx_candidate = grid_ratio / (price/moneyness)`, i.e. "which fx would
# make THIS row's price consistent with a canonical ratio") and a robust
# (MAD-filtered) median taken across all such candidates in that issuer's
# group -- then the candidate is validated by literally re-running the exact
# same ratio-snap+verify pipeline on the group's own remaining rows and
# requiring at least a MAJORITY (`_DEFAULT_FX_LEARNING_MIN_VERIFIED_FRACTION`
# = 0.5, directly the "Mehrheit" required by Befund 2) to verify before the
# learned fx is used at all; live-measured on the NDX sample, this lifted
# BNP Paribas/HSBC/Goldman Sachs's groups to 95-99.7% internal verification
# (learned fx ~1.135-1.14, close to the live EURUSD ~1.16 used only for this
# session's own validation, never hardcoded into the adapter) while
# UniCredit's group converged on a wrong candidate (0.4% verified) and was
# correctly REJECTED by the majority gate -- that group's rows are left
# exactly as conservative as before this fix (never guessed), demonstrating
# the safety net actually discriminates good from bad learned estimates
# rather than just rubber-stamping whatever the grid search proposes.
_DEFAULT_FX_LEARNING_SEARCH_MIN = 0.5
_DEFAULT_FX_LEARNING_SEARCH_MAX = 2.0
_DEFAULT_FX_LEARNING_MIN_CANDIDATES = 5
_DEFAULT_FX_LEARNING_MIN_VERIFIED_FRACTION = 0.5  # "Mehrheit" (majority)

# Befund 2 fix 3 -- leverage-INVERSE-scaled verification, two-hypothesis
# path only. Combining fix 1 (wider snap tolerance) with a second tested
# hypothesis (fix 2's learned fx, or a caller fx_hint) reopened a genuine
# edge found while building the contract test for this change: at very high
# leverage, `implied_spot_tolerance_pct` measured *relative to S_ref*
# (unchanged since it existed) loses discriminating power exactly as fast as
# the snap tolerance needs to widen -- both are driven by the same
# moneyness-shrinks-with-leverage relationship, but in the two-hypothesis
# case a WRONG fx hypothesis can, at high enough leverage, produce an
# implied-spot deviation small enough to slip under even the *original*
# unwidened 0.3% -- a systematic (not random-noise) mispricing, e.g. a
# uniformly-8%-off non-quanto guess at leverage 50-200, was confirmed live
# (this session's own contract-test construction) to verify incorrectly
# under fx=1 once fix 1's snap tolerance was wide enough to let it reach the
# verification step at all. The single-hypothesis (EUR/DAX) path is NOT
# affected -- there is only ever one fx candidate there, so there is no
# "wrong-but-plausible" alternative to mistake for correct, and the DAX
# live-measurement above (4 false-accepts / 2000, all caught) used the
# original, unmodified 0.3% throughout.
# Fix: for the two-hypothesis path only, tighten
# (`_leverage_scaled_verification_tolerance`) the SAME 0.3%
# `implied_spot_tolerance_pct` proportionally to leverage (dividing where
# fix 1 multiplies, same `leverage_ref`) before either hypothesis is
# checked. Live-verified (2026-09-13, same NDX sample used for fix 2): the
# adversarial 8%-offset construction is rejected at every leverage from 20
# to 200 with this fix (was silently accepted for leverage>=50 without it),
# while overall NDX derivation drops only marginally, from 87.0% to 85.6%
# (BNP Paribas/HSBC/Goldman Sachs's learned-fx groups still verify at
# 92-99.6% internally) -- i.e. the extra safety costs little of fix 2's gain
# while closing a real, demonstrated false-accept path.
_DEFAULT_QUANTO_VERIFICATION_TOLERANCE_LEVERAGE_REF = _DEFAULT_RATIO_SNAP_TOLERANCE_LEVERAGE_REF

# Numerical-conditioning cap on which rows are allowed to CONTRIBUTE to the
# S_ref median (see _robust_reference_spot call site): moneyness = S/leverage,
# so a fixed relative uncertainty in a candidate's own S estimate translates
# into a relative error in |S_ref - F| that scales with leverage (near-the-
# money-by-definition high-leverage rows have a tiny true moneyness, so the
# same absolute S_ref noise is a much larger fraction of it). Confirmed live
# (2026-09-12/13 validation session, DAX, WKN-sorted default page): observed
# leverage floor for that page was ~60 (median ~78, max ~244) -- an
# aggressively low cap (e.g. 50) excludes the ENTIRE default page and yields
# no S_ref at all, while 200 changes essentially nothing in-range (S_ref
# moved <0.1 index points across every cap tried from 100 to 1000) yet still
# guards against the genuinely pathological tail (leverage >500, observed
# elsewhere in this project's research on the same DAX universe) that would
# otherwise be numerically unstable to invert. The resulting S_ref, with this
# cap and the direction-balancing above, was independently verified against
# gettex's OWN reported `underlyings.price` for the same fetch (25552.47 vs.
# 25555.5 -- 0.012% deviation) -- i.e. the estimator itself is sound; a low
# per-fetch ratio-derivation yield in a given slice reflects genuine bid/ask
# wrapper-margin/spread noise on individual high-leverage products exceeding
# the strict ratio-snap/verification tolerances, not a mis-tuned cap.
_DEFAULT_MAX_LEVERAGE_FOR_REFERENCE_SPOT = 200.0
_MAD_TO_STD = 1.4826  # same constant/convention as pricing/cross_issuer.py

# Issuer-string normalization: gettex's `issuer.value` is a plain, sometimes
# issuer's-own-legal-entity-name string ("HSBC Trinkaus & Burkhardt GmbH")
# rather than the short form used elsewhere in this project ("HSBC"). Mapped
# by lowercased exact match; any issuer string not in this table (a real
# gap -- gettex covers issuers this project has not otherwise researched)
# passes through unchanged rather than being guessed.
_ISSUER_NORMALIZATION: dict[str, str] = {
    "bnp paribas": "BNP Paribas",
    "bnp paribas emissions- und handelsgesellschaft mbh": "BNP Paribas",
    "goldman sachs bank europe se": "Goldman Sachs",
    "goldman sachs": "Goldman Sachs",
    "hsbc trinkaus & burkhardt gmbh": "HSBC",
    "hsbc trinkaus & burkhardt ag": "HSBC",
    "hsbc": "HSBC",
    "unicredit bank gmbh": "UniCredit",
    "unicredit bank ag": "UniCredit",
    "unicredit onemarkets": "UniCredit",
    "unicredit": "UniCredit",
}


@dataclass(frozen=True)
class GettexRowError:
    """Error record for one product row that failed parsing/derivation."""

    source: str
    isin: str | None
    error: str


# -- small shared helpers ------------------------------------------------


def _raw_hash(record: dict[str, Any]) -> str:
    """sha256 of the canonical JSON encoding of a raw source record."""
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _quality_score(*, quote_presence: bool, is_stale: bool) -> float:
    """1.0 fresh with bid+ask; 0.6 stale; 0.3 without a usable ask."""
    if not quote_presence:
        return 0.3
    if is_stale:
        return 0.6
    return 1.0


def _normalize_issuer(raw: str) -> str:
    return _ISSUER_NORMALIZATION.get(raw.strip().lower(), raw.strip())


def _resolve_scalar_or_mapping(value: Mapping[str, float] | float | None, key: str) -> float | None:
    """Resolve a `reference_spot`/`fx_hint`-style argument for one underlying_id.

    Accepts either a single float (applied to every requested underlying --
    the common single-underlying-call case) or a per-underlying_id mapping.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        found = value.get(key)
        return float(found) if found is not None else None
    return float(value)


def _resolve_underlying_currency(underlying_id: str) -> str | None:
    try:
        return get_underlying_meta(underlying_id).currency
    except KeyError:
        return None


def _parse_epoch_ms(raw: Any) -> datetime | None:
    """Parse gettex's plain epoch-millisecond UTC timestamp field.

    Unlike BNP (naive Europe/Berlin) and Citi (naive, no offset), gettex's
    `valueTuple.timestamp` is an unambiguous epoch-ms integer -- no
    locale/timezone parsing needed at all.
    """
    if not isinstance(raw, int | float) or isinstance(raw, bool):
        return None
    try:
        return datetime.fromtimestamp(raw / 1000.0, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _extract_quote(field_obj: Any) -> tuple[float | None, float | None, datetime | None]:
    """Extract (value, size, timestamp) from a gettex bid/ask field.

    Handles Pitfall 1 (`valueTuple` can be absent entirely on an illiquid/
    dead row) and Pitfall 2 (`value == 0.0` + `size == 0.0` is the documented
    "no live quote" sentinel, same family as BNP/Citi) -- both collapse to
    `(None, size, timestamp)` rather than a fake zero price. `size`/
    `timestamp` are still returned even when `value` is nulled, for
    diagnostics, mirroring how ``adapters/issuer_feeds.py`` keeps BNP's
    ``askSize`` regardless of the ask-key-absent case.
    """
    if not isinstance(field_obj, dict):
        return None, None, None
    value_tuple = field_obj.get("valueTuple")
    if not isinstance(value_tuple, dict):
        return None, None, None
    raw_value = value_tuple.get("value")
    raw_size = value_tuple.get("size")
    timestamp = _parse_epoch_ms(value_tuple.get("timestamp"))
    value = (
        float(raw_value)
        if isinstance(raw_value, int | float) and not isinstance(raw_value, bool)
        else None
    )
    size = (
        float(raw_size)
        if isinstance(raw_size, int | float) and not isinstance(raw_size, bool)
        else None
    )
    if value == 0.0:
        return None, size, timestamp
    return value, size, timestamp


def _extract_plain(field_obj: Any) -> tuple[float | None, datetime | None]:
    """Extract (value, timestamp) from a gettex `underlyings.price`-style field.

    No zero-sentinel treatment (unlike bid/ask): a non-positive underlying
    reference price is simply invalid, not a documented "no quote" marker.
    """
    if not isinstance(field_obj, dict):
        return None, None
    value_tuple = field_obj.get("valueTuple")
    if not isinstance(value_tuple, dict):
        return None, None
    raw_value = value_tuple.get("value")
    if not isinstance(raw_value, int | float) or isinstance(raw_value, bool) or raw_value <= 0:
        return None, None
    timestamp = _parse_epoch_ms(value_tuple.get("timestamp"))
    return float(raw_value), timestamp


def _get_field_obj(raw: dict[str, Any], key: str) -> dict[str, Any]:
    obj = raw.get(key)
    if not isinstance(obj, dict):
        raise ValueError(f"missing/invalid field {key!r}")
    return obj


def _get_str(raw: dict[str, Any], key: str) -> str:
    value = _get_field_obj(raw, key).get("value")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing/empty {key}.value")
    return value


def _get_optional_str(raw: dict[str, Any], key: str) -> str | None:
    obj = raw.get(key)
    if not isinstance(obj, dict):
        return None
    value = obj.get("value")
    return value if isinstance(value, str) and value.strip() else None


def _get_number(raw: dict[str, Any], key: str) -> float:
    value = _get_field_obj(raw, key).get("value")
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"missing/invalid numeric {key}.value")
    return float(value)


# -- row parsing -----------------------------------------------------------


@dataclass
class _ParsedRow:
    """One gettex product row, schema-parsed but pre-ratio-derivation."""

    raw: dict[str, Any]
    isin: str
    wkn: str | None
    issuer_raw: str
    underlying_raw: str
    direction: Direction
    financing_level: float
    knockout_barrier: float
    leverage: float
    bid: float | None
    bid_size: float | None
    ask: float | None
    ask_size: float | None
    quote_timestamp: datetime | None
    quote_presence: bool
    is_stale: bool
    underlying_price_ref: float | None
    underlying_price_ref_timestamp: datetime | None

    @property
    def mid_price(self) -> float | None:
        """(bid+ask)/2 when both are live, else bid alone (documented,
        one-sided bid-only approximation -- see module docstring point 3 /
        `pricing/cross_issuer.consensus_spot`'s identical fallback)."""
        if self.bid is None:
            return None
        if self.ask is None:
            return self.bid
        return (self.bid + self.ask) / 2.0


def _parse_row(raw: dict[str, Any], *, now: datetime, stale_after_s: float) -> _ParsedRow:
    """Parse one raw gettex product dict into a :class:`_ParsedRow`.

    Raises ``ValueError`` on any missing/malformed required field (schema
    drift) -- callers catch this per-row and record it in ``last_errors``
    rather than letting one bad row abort the whole fetch.
    """
    isin = _get_str(raw, "isin")
    wkn = _get_optional_str(raw, "wkn")
    issuer_raw = _get_str(raw, "issuer")
    underlying_raw = _get_str(raw, "underlying")

    direction_raw = _get_str(raw, "structure.direction")
    direction = _GETTEX_DIRECTION_MAP.get(direction_raw)
    if direction is None:
        raise ValueError(f"unmapped structure.direction.value: {direction_raw!r}")

    financing_level = _get_number(raw, "financingLevelRefCurAbsolute")
    knockout_barrier = _get_number(raw, "koLevelRefCurAbsolute")
    leverage = _get_number(raw, "leverage")
    if leverage <= 0:
        raise ValueError(f"non-positive leverage.value: {leverage!r}")

    bid, bid_size, bid_ts = _extract_quote(raw.get("bid"))
    ask, ask_size, ask_ts = _extract_quote(raw.get("ask"))
    underlying_price_ref, underlying_price_ref_ts = _extract_plain(raw.get("underlyings.price"))

    candidates = []
    if bid is not None and bid_ts is not None:
        candidates.append(bid_ts)
    if ask is not None and ask_ts is not None:
        candidates.append(ask_ts)
    quote_timestamp = max(candidates) if candidates else None
    quote_presence = bid is not None and ask is not None
    is_stale = (
        True if quote_timestamp is None else (now - quote_timestamp).total_seconds() > stale_after_s
    )

    return _ParsedRow(
        raw=raw,
        isin=isin,
        wkn=wkn,
        issuer_raw=issuer_raw,
        underlying_raw=underlying_raw,
        direction=direction,
        financing_level=financing_level,
        knockout_barrier=knockout_barrier,
        leverage=leverage,
        bid=bid,
        bid_size=bid_size,
        ask=ask,
        ask_size=ask_size,
        quote_timestamp=quote_timestamp,
        quote_presence=quote_presence,
        is_stale=is_stale,
        underlying_price_ref=underlying_price_ref,
        underlying_price_ref_timestamp=underlying_price_ref_ts,
    )


# -- reference-spot derivation (leverage identity, ratio-independent) ------


def _spot_from_leverage(
    direction: Direction, financing_level: float, leverage: float
) -> float | None:
    """Invert ``leverage = S / |S - F|`` for S, given direction and F.

    Long:  ``S = F * leverage / (leverage - 1)`` (requires ``leverage > 1``).
    Short: ``S = F * leverage / (leverage + 1)``.

    Returns ``None`` for a degenerate/non-physical input (e.g. ``leverage``
    at or below the long-side singularity) rather than raising -- callers
    simply exclude that row's candidate from the median.
    """
    if leverage <= 0 or financing_level <= 0:
        return None
    if direction == Direction.LONG:
        denom = leverage - 1.0
        if denom <= 0:
            return None
        return financing_level * leverage / denom
    denom = leverage + 1.0
    if denom <= 0:
        return None
    return financing_level * leverage / denom


@dataclass(frozen=True, slots=True)
class _SpotEstimate:
    value: float
    n_used: int
    n_rejected: int
    direction_balanced: bool


def _robust_reference_spot(
    candidates: Sequence[tuple[float, Direction]], mad_k: float
) -> _SpotEstimate | None:
    """Robust median of leverage-implied spot candidates, MAD-outlier-filtered
    and long/short direction-balanced.

    Deliberately mirrors ``pricing/cross_issuer.consensus_spot`` (same MAD
    constant, same direction-balancing rationale -- see module docstring)
    rather than importing it: that function operates on already-priced
    ``ProductSnapshot``s (it needs a known ``ratio``), while this runs
    *before* any ratio is known, on raw ``(S_candidate, direction)`` pairs
    derived from ``leverage``/``financing_level`` alone.

    Returns ``None`` when ``candidates`` is empty.
    """
    if not candidates:
        return None
    values = np.asarray([c[0] for c in candidates], dtype=np.float64)
    directions = [c[1] for c in candidates]

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scaled_mad = mad * _MAD_TO_STD
    threshold = max(abs(median) * 1e-6, 1e-9) if scaled_mad == 0.0 else mad_k * scaled_mad
    mask = np.abs(values - median) <= threshold
    accepted = values[mask]
    accepted_directions = [d for d, keep in zip(directions, mask, strict=True) if keep]
    n_rejected = int(values.size - accepted.size)

    paired = list(zip(accepted, accepted_directions, strict=True))
    long_values = [v for v, d in paired if d == Direction.LONG]
    short_values = [v for v, d in paired if d == Direction.SHORT]
    direction_balanced = bool(long_values) and bool(short_values)
    if direction_balanced:
        value = (statistics.median(long_values) + statistics.median(short_values)) / 2.0
    else:
        value = float(np.median(accepted)) if accepted.size > 0 else median

    return _SpotEstimate(
        value=value,
        n_used=int(accepted.size),
        n_rejected=n_rejected,
        direction_balanced=direction_balanced,
    )


# -- ratio derivation + verification ---------------------------------------


def _snap_ratio(raw_ratio: float, tolerance_pct: float) -> float | None:
    """Snap ``raw_ratio`` to the nearest canonical grid value, only if the
    relative deviation to that grid value is within ``tolerance_pct``."""
    if raw_ratio <= 0:
        return None
    best: float | None = None
    best_dev: float | None = None
    for grid_value in _RATIO_GRID:
        dev = abs(raw_ratio - grid_value) / grid_value
        if dev <= tolerance_pct and (best_dev is None or dev < best_dev):
            best = grid_value
            best_dev = dev
    return best


@dataclass(frozen=True, slots=True)
class _RatioAttempt:
    ratio: float | None
    verified: bool
    deviation_pct: float | None


def _attempt_ratio(
    *,
    direction: Direction,
    financing_level: float,
    price: float,
    s_ref: float,
    fx: float,
    ratio_snap_tolerance_pct: float,
    implied_spot_tolerance_pct: float,
) -> _RatioAttempt:
    moneyness = abs(s_ref - financing_level)
    if moneyness <= 0:
        return _RatioAttempt(ratio=None, verified=False, deviation_pct=None)
    ratio_raw = price * fx / moneyness
    ratio = _snap_ratio(ratio_raw, ratio_snap_tolerance_pct)
    if ratio is None:
        return _RatioAttempt(ratio=None, verified=False, deviation_pct=None)
    implied_spot = implied_underlying(price, financing_level, ratio, direction, fx)
    deviation_pct = abs(implied_spot - s_ref) / s_ref
    verified = deviation_pct <= implied_spot_tolerance_pct
    return _RatioAttempt(ratio=ratio, verified=verified, deviation_pct=deviation_pct)


def _leverage_scaled_ratio_snap_tolerance(
    leverage: float, base_tolerance_pct: float, leverage_ref: float, cap_pct: float
) -> float:
    """Widen the ratio-snap tolerance for high-leverage rows (Befund 2 fix 1).

    ``base_tolerance_pct`` applies unscaled at/below ``leverage_ref``; above
    it, the tolerance grows proportionally with leverage (mirroring the same
    moneyness-shrinks-with-leverage relationship documented on
    ``_DEFAULT_MAX_LEVERAGE_FOR_REFERENCE_SPOT``), capped at ``cap_pct`` so a
    pathologically high leverage never gets an unbounded tolerance. This only
    controls which grid value is *proposed* as a candidate ratio -- the
    separate ``implied_spot_tolerance_pct`` verification step (see
    :func:`_attempt_ratio`) is what actually decides correctness, so
    widening this alone cannot let an unverified guess through (for a
    non-EUR underlying's two-hypothesis path specifically, that verification
    tolerance is itself tightened at high leverage -- see
    :func:`_leverage_scaled_verification_tolerance`, Befund 2 fix 3 -- so the
    two adjustments together keep verification meaningful at every leverage
    rather than just widening one side of the check).
    """
    if leverage_ref <= 0:
        return base_tolerance_pct
    return min(cap_pct, base_tolerance_pct * max(1.0, leverage / leverage_ref))


def _leverage_scaled_verification_tolerance(
    leverage: float, base_tolerance_pct: float, leverage_ref: float
) -> float:
    """Tighten the implied-spot verification tolerance for high-leverage rows
    when a second (non-quanto) fx hypothesis is in play (Befund 2 fix 3).

    Inverse of :func:`_leverage_scaled_ratio_snap_tolerance` (divides instead
    of multiplies, same ``leverage_ref``): a wrong ratio/fx guess produces an
    implied-spot deviation that shrinks proportionally to ``1/leverage`` (the
    same moneyness-shrinks-with-leverage relationship, from the other side),
    so without this, a wide enough snap tolerance (needed to recover genuine
    high-leverage rows, see fix 1) could let a systematically-wrong fx
    hypothesis slip under even the *unscaled* 0.3% verification bar at high
    enough leverage -- confirmed live via this change's own contract test
    (a uniform 8%-off fx construction verified incorrectly at leverage
    50-200 without this tightening). Only used for the two-hypothesis
    (non-EUR underlying) path; the single-hypothesis EUR path has no
    "wrong-but-plausible" alternative fx to mistake for correct and keeps
    the unscaled ``implied_spot_tolerance_pct`` throughout.
    """
    if leverage_ref <= 0:
        return base_tolerance_pct
    return base_tolerance_pct / max(1.0, leverage / leverage_ref)


def _fx_candidates_from_ratio_raw_at_1(
    ratio_raw_at_1: float, fx_search_min: float, fx_search_max: float
) -> list[float]:
    """Grid-inversion fx candidates for one row (Befund 2 fix 2, step 1).

    ``ratio_raw_at_1 = price / moneyness`` is the fx=1 ratio estimate (cheap,
    always computable). For each canonical grid ratio ``r``, the fx value
    that would make THIS row's price consistent with exactly that ratio is
    ``r / ratio_raw_at_1`` (inverting ``ratio_raw = price*fx/moneyness =
    r``). Only candidates within a plausible major-currency-pair band
    (``[fx_search_min, fx_search_max]``, default ``[0.5, 2.0]``) are kept --
    this is a search-space bound, not a guessed answer: a real EURUSD-,
    EURGBP- or EURCHF-style rate has stayed well within that band for the
    entire history relevant to this project, while a wrong-grid-ratio
    candidate (the grid steps by factors of 2/2.5/5/10) typically lands
    outside it and is discarded here, before any robust averaging happens.
    """
    if ratio_raw_at_1 <= 0:
        return []
    candidates: list[float] = []
    for grid_value in _RATIO_GRID:
        fx_candidate = grid_value / ratio_raw_at_1
        if fx_search_min <= fx_candidate <= fx_search_max:
            candidates.append(fx_candidate)
    return candidates


def _robust_fx_median(candidates: Sequence[float], mad_k: float) -> float | None:
    """MAD-filtered median of fx candidates (same convention as
    :func:`_robust_reference_spot`, no direction-balancing -- fx candidates
    have no long/short analogue). Returns ``None`` for an empty input."""
    if not candidates:
        return None
    values = np.asarray(candidates, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scaled_mad = mad * _MAD_TO_STD
    if scaled_mad == 0.0:
        return median
    mask = np.abs(values - median) <= mad_k * scaled_mad
    accepted = values[mask]
    return float(np.median(accepted)) if accepted.size > 0 else median


# -- aggregated per-fetch logging -------------------------------------------


@dataclass
class _FetchStats:
    total: int = 0
    ratio_derived: int = 0
    ratio_rejected: int = 0
    verification_failed: int = 0
    # Distinct from `verification_failed` (Befund 2): this counts rows where
    # BOTH the quanto (fx=1) and non-quanto (fx=fx_hint/learned) hypotheses
    # independently snapped a ratio AND verified -- genuine ambiguity, never
    # guessed away, vs. `verification_failed` where zero hypotheses verified.
    quanto_ambiguous: int = 0
    ask_missing: int = 0
    example_isins: list[str] = field(default_factory=list)

    def note_example(self, isin: str | None) -> None:
        if isin and len(self.example_isins) < 5:
            self.example_isins.append(isin)


def _log_fetch_summary(stats: _FetchStats) -> None:
    logger.info(
        "gettex_fetch_summary",
        total=stats.total,
        ratio_derived=stats.ratio_derived,
        ratio_rejected=stats.ratio_rejected,
        verification_failed=stats.verification_failed,
        quanto_ambiguous=stats.quanto_ambiguous,
        ask_missing=stats.ask_missing,
    )
    if stats.example_isins:
        logger.debug("gettex_fetch_summary_examples", example_isins=stats.example_isins)


def _headers(user_agent: str) -> dict[str, str]:
    return {"User-Agent": user_agent, "Accept": "application/json"}


def _ratio_reliability_for_outcome(outcome: str) -> FieldReliability:
    """Map :meth:`GettexAdapter._derive_ratio`'s outcome string to the
    field-level reliability tier a resulting product should carry (Phase B,
    "Produktstammdaten haerten").

    Only ``"ok"`` -- a ratio that was both derived AND independently
    verified against the leverage-implied reference spot (module docstring
    steps 1-4) -- earns ``DERIVED_VERIFIED``; every other outcome
    (``"ratio_rejected"``, ``"verification_failed"``, ``"quanto_ambiguous"``)
    maps to ``UNVERIFIED``, the honest default (CLAUDE.md rule 29).

    In :meth:`fetch_products` today, a non-``"ok"`` outcome never reaches a
    ``ProductSnapshot`` at all (``ratio`` is a required, pricing-critical
    field on that model -- there is nothing to attach an ``UNVERIFIED``
    ratio *to*), so every snapshot this adapter actually emits currently
    gets ``DERIVED_VERIFIED``. This function is exercised directly by
    contract tests so the mapping's "otherwise UNVERIFIED" half stays
    correct and testable independent of that (today-incidental) fact.
    """
    return FieldReliability.DERIVED_VERIFIED if outcome == "ok" else FieldReliability.UNVERIFIED


# -- adapter -----------------------------------------------------------------


class GettexAdapter:
    """gettex (Boerse Muenchen) ``gettex.wsd.com`` leverage-product feed adapter."""

    def __init__(
        self,
        http: HttpClient,
        *,
        base_url: str = _GETTEX_DEFAULT_BASE_URL,
        max_pages: int = 20,
        rows_per_page: int = _GETTEX_ROWS_PER_PAGE,
        stale_after_s: float = _DEFAULT_STALE_AFTER_S,
        reference_spot_tolerance_pct: float = _DEFAULT_REFERENCE_SPOT_TOLERANCE_PCT,
        daily_close_reference_tolerance_pct: float = _DEFAULT_DAILY_CLOSE_REFERENCE_TOLERANCE_PCT,
        ratio_snap_tolerance_pct: float = _DEFAULT_RATIO_SNAP_TOLERANCE_PCT,
        implied_spot_tolerance_pct: float = _DEFAULT_IMPLIED_SPOT_TOLERANCE_PCT,
        mad_k: float = _DEFAULT_MAD_K,
        max_leverage_for_reference_spot: float = _DEFAULT_MAX_LEVERAGE_FOR_REFERENCE_SPOT,
        ratio_snap_tolerance_leverage_ref: float = _DEFAULT_RATIO_SNAP_TOLERANCE_LEVERAGE_REF,
        ratio_snap_tolerance_cap_pct: float = _DEFAULT_RATIO_SNAP_TOLERANCE_CAP_PCT,
        fx_learning_search_min: float = _DEFAULT_FX_LEARNING_SEARCH_MIN,
        fx_learning_search_max: float = _DEFAULT_FX_LEARNING_SEARCH_MAX,
        fx_learning_min_candidates: int = _DEFAULT_FX_LEARNING_MIN_CANDIDATES,
        fx_learning_min_verified_fraction: float = _DEFAULT_FX_LEARNING_MIN_VERIFIED_FRACTION,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._max_pages = max_pages
        self._rows_per_page = rows_per_page
        self._stale_after_s = stale_after_s
        self._reference_spot_tolerance_pct = reference_spot_tolerance_pct
        self._daily_close_reference_tolerance_pct = daily_close_reference_tolerance_pct
        self._ratio_snap_tolerance_pct = ratio_snap_tolerance_pct
        self._implied_spot_tolerance_pct = implied_spot_tolerance_pct
        self._mad_k = mad_k
        self._max_leverage_for_reference_spot = max_leverage_for_reference_spot
        self._ratio_snap_tolerance_leverage_ref = ratio_snap_tolerance_leverage_ref
        self._ratio_snap_tolerance_cap_pct = ratio_snap_tolerance_cap_pct
        self._fx_learning_search_min = fx_learning_search_min
        self._fx_learning_search_max = fx_learning_search_max
        self._fx_learning_min_candidates = fx_learning_min_candidates
        self._fx_learning_min_verified_fraction = fx_learning_min_verified_fraction
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self.last_errors: list[GettexRowError] = []
        self._partial_universe: dict[str, tuple[int, int]] = {}
        self._reference_spot_mismatches: dict[str, str] = {}
        # Per-(issuer, underlying_id) fx learned by :meth:`_learn_fx_by_issuer`
        # during the most recent :meth:`fetch_products` call -- exposed for
        # diagnostics/tests, keyed ``f"{underlying_id}:{issuer}"``.
        self.last_learned_fx: dict[str, float] = {}
        # Discarded ratio-derivation attempts from the most recent
        # `fetch_products` call, drained by `pipeline/universe.py` via
        # `RejectedRatioDerivationSource`. Before this existed, a row that
        # failed derivation only incremented a `_FetchStats` counter and
        # appended a transient `last_errors` line that nothing persists --
        # which is exactly why docs/product_data_quality.md can only report
        # 0% UNVERIFIED ratio reliability for gettex (an artifact of what was
        # thrown away, not a measurement). See `RejectedRatioDerivation`'s
        # docstring and CLAUDE.md rule 31 ("negative Muster nicht loeschen").
        self._rejected_ratio_derivations: list[RejectedRatioDerivation] = []

    @property
    def name(self) -> str:
        return _GETTEX_SOURCE_NAME

    def metadata(self) -> AdapterMetadata:
        return AdapterMetadata(
            name=_GETTEX_SOURCE_NAME,
            kind="product",
            version=_GETTEX_PARSER_VERSION,
            homepage="https://www.gettex.de/maerkte/zertifikate-finder/",
        )

    # -- fetch/normalize (generic DataSourceAdapter contract) ---------------

    def fetch(self, **kwargs: Any) -> Any:
        """Lightweight liveness probe (one small DAX page). All the real
        network I/O for product retrieval lives in :meth:`_fetch_all_pages`;
        this method exists to satisfy the generic
        ``DataSourceAdapter.fetch`` half of the contract, mirroring
        ``BnpParibasTurboAdapter.fetch``."""
        del kwargs
        url = f"{self._base_url}{_GETTEX_PRODUCTS_PATH}"
        params = {
            "underlying": _GETTEX_UNDERLYING_IDS["DAX"],
            "productType": _GETTEX_PRODUCT_TYPE_LABEL,
            "rowsPerPage": 10,
            "page": 1,
        }
        return self._http.get_json(url, params=params, headers=_headers(self._http.user_agent))

    def normalize(self, raw: Any, **kwargs: Any) -> Any:
        """Not used directly -- product normalization happens in
        :meth:`fetch_products`, which needs each row's ``underlying_id``/
        reference-spot context. Present to satisfy
        ``DataSourceAdapter.normalize``."""
        del raw, kwargs
        return []

    # -- product fetching -----------------------------------------------------

    def fetch_products(
        self,
        underlying_ids: Sequence[str],
        *,
        context: ProductFetchContext | None = None,
        reference_spot: Mapping[str, float] | float | None = None,
        daily_close_reference: Mapping[str, float] | float | None = None,
        fx_hint: Mapping[str, float] | float | None = None,
    ) -> list[ProductSnapshot]:
        """Fetch, derive-and-verify ratio for, and normalize gettex products.

        ``context`` (Befund 2, :class:`~turboedge.adapters.base.
        ProductFetchContext`) is how the generic ``ProductSourceAdapter``
        call path (``pipeline/universe.py``) supplies the two cross-check
        arguments below -- ``reference_spot``/``daily_close_reference`` stay
        as direct keyword arguments too (unchanged) for callers/tests that
        already use this adapter's own richer signature directly. When both
        a direct argument and ``context`` are given, the explicit direct
        argument wins (it is the more specific request); ``context`` only
        fills in whichever of the two is otherwise left at its default
        ``None``.

        ``S_ref`` (the reference spot) is *always* derived internally from
        gettex's own leverage identity (module docstring step 1) -- it never
        depends on either of the two arguments below existing. They are
        optional, additive sanity cross-checks only, tried in this priority
        order per underlying: ``reference_spot`` (tight tolerance) first,
        else ``daily_close_reference`` (wide tolerance) if that one is given
        instead, else no cross-check at all (S_ref stands on its own, as
        module docstring step 1 already documents it validating well against
        gettex's own reported ``underlyings.price``).

        Args:
            underlying_ids: canonical underlying ids to fetch (unresolvable
                ids, i.e. not in ``_GETTEX_UNDERLYING_IDS``, are skipped with
                a log line).
            reference_spot: optional external **live-quote** cross-check for
                the internally derived ``S_ref`` (see module docstring point
                2), as a single float (applied to every requested underlying)
                or a ``{underlying_id: spot}`` mapping. When given and
                ``S_ref`` deviates beyond ``reference_spot_tolerance_pct``
                (tight, default 0.5% -- appropriate for two live quotes taken
                close together in time), *no* products are returned for that
                underlying this fetch.
            daily_close_reference: optional **daily-close** (e.g. a same-day
                ``yfinance`` close) cross-check, same single-float-or-mapping
                shape as ``reference_spot``, used only when ``reference_spot``
                is not supplied for a given underlying (2026-09-13
                integration finding: a single issuer-sourced ``reference_spot``
                turns that issuer's own outage into a single point of failure
                for gettex too, even though gettex's ``S_ref`` never actually
                needed that issuer to begin with -- see
                ``_DEFAULT_DAILY_CLOSE_REFERENCE_TOLERANCE_PCT``). Checked
                against the wider ``daily_close_reference_tolerance_pct``
                (default 2%), since a prior session's close can legitimately
                sit further from the current live level than two simultaneous
                live quotes would.
            fx_hint: optional EURUSD-style FX rate (or ``{underlying_id: fx}``
                mapping) used as the "non-quanto" hypothesis for underlyings
                whose currency is not EUR (see module docstring point 5).
                When omitted for such an underlying (notably: always, via
                the generic ``ProductSourceAdapter`` contract -- see Befund 2
                in module docstring point 5),
                :meth:`_learn_fx_by_issuer` learns a per-issuer non-quanto fx
                candidate from this fetch's own data instead of falling back
                to the quanto-only hypothesis for every row.
        """
        if context is not None:
            if reference_spot is None:
                reference_spot = context.reference_spot
            if daily_close_reference is None:
                daily_close_reference = context.daily_close_reference

        self.last_errors = []
        self._partial_universe = {}
        self._reference_spot_mismatches = {}
        self.last_learned_fx = {}
        self._rejected_ratio_derivations = []
        now = self._clock()
        stats = _FetchStats()

        snapshots: list[ProductSnapshot] = []
        for underlying_id in underlying_ids:
            gettex_id = _GETTEX_UNDERLYING_IDS.get(underlying_id)
            if gettex_id is None:
                logger.warning("gettex_underlying_unresolved", underlying_id=underlying_id)
                continue

            raw_items, filtered_count = self._fetch_all_pages(gettex_id)
            covered = len(raw_items)
            if filtered_count is not None and covered < filtered_count:
                self._partial_universe[underlying_id] = (covered, filtered_count)
                logger.warning(
                    "gettex_partial_universe",
                    underlying_id=underlying_id,
                    covered=covered,
                    total=filtered_count,
                )

            parsed_rows: list[_ParsedRow] = []
            for raw_product in raw_items:
                # Best-effort ISIN for error reporting, read directly from
                # the raw dict (never raises) -- `_parse_row` can fail before
                # it reaches the point of returning a `_ParsedRow.isin`, so
                # this must not depend on a successful parse.
                isin_obj = raw_product.get("isin")
                isin_for_error = isin_obj.get("value") if isinstance(isin_obj, dict) else None
                try:
                    row = _parse_row(raw_product, now=now, stale_after_s=self._stale_after_s)
                    isin_for_error = row.isin
                    resolved = resolve_underlying_id(row.underlying_raw)
                    if resolved != underlying_id:
                        raise ValueError(
                            f"underlying label {row.underlying_raw!r} resolved to "
                            f"{resolved!r}, expected requested {underlying_id!r} "
                            "(gettex_underlying_filter_mismatch -- never trust a "
                            "server-side filter without checking the row)"
                        )
                    parsed_rows.append(row)
                except (KeyError, ValueError, TypeError) as exc:
                    stats.total += 1
                    stats.note_example(isin_for_error)
                    self.last_errors.append(
                        GettexRowError(
                            source=_GETTEX_SOURCE_NAME, isin=isin_for_error, error=str(exc)
                        )
                    )
                    logger.debug("gettex_row_parse_error", isin=isin_for_error, error=str(exc))

            # -- Step 1: robust, ratio-independent reference spot ------------
            # Only fresh, moderate-leverage rows contribute (see
            # _DEFAULT_MAX_LEVERAGE_FOR_REFERENCE_SPOT's docstring for why
            # very-high-leverage candidates are numerically ill-conditioned
            # for this specific estimation, independent of staleness) -- this
            # does not affect which rows are later eligible for ratio
            # derivation itself, only which rows help build S_ref.
            fresh_candidates: list[tuple[float, Direction]] = []
            for row in parsed_rows:
                if row.is_stale:
                    continue
                if row.leverage > self._max_leverage_for_reference_spot:
                    continue
                estimate = _spot_from_leverage(row.direction, row.financing_level, row.leverage)
                if estimate is not None:
                    fresh_candidates.append((estimate, row.direction))

            spot_estimate = _robust_reference_spot(fresh_candidates, self._mad_k)
            if spot_estimate is None:
                message = "no fresh row yielded a leverage-implied reference spot"
                self._reference_spot_mismatches[underlying_id] = message
                logger.warning(
                    "gettex_no_reference_spot", underlying_id=underlying_id, n_rows=len(parsed_rows)
                )
                continue
            s_ref = spot_estimate.value

            expected = _resolve_scalar_or_mapping(reference_spot, underlying_id)
            if expected is not None:
                deviation_pct = abs(s_ref - expected) / expected
                if deviation_pct > self._reference_spot_tolerance_pct:
                    message = (
                        f"leverage-derived S_ref={s_ref:.4f} deviates {deviation_pct:.4%} "
                        f"from supplied reference_spot={expected:.4f} "
                        f"(tolerance {self._reference_spot_tolerance_pct:.4%})"
                    )
                    self._reference_spot_mismatches[underlying_id] = message
                    logger.warning(
                        "gettex_reference_spot_mismatch",
                        underlying_id=underlying_id,
                        s_ref=s_ref,
                        reference_spot=expected,
                        deviation_pct=deviation_pct,
                    )
                    continue
                logger.debug(
                    "gettex_reference_spot_confirmed",
                    underlying_id=underlying_id,
                    s_ref=s_ref,
                    reference_spot=expected,
                    deviation_pct=deviation_pct,
                )
            else:
                # No live external reference_spot -- either the caller never
                # had one (single-source scan) or its supplying source failed
                # this run (2026-09-13 integration finding: a BNP outage
                # silently zeroed gettex entirely, because the pipeline's
                # only cross-check source shared gettex's fate). S_ref never
                # depended on `reference_spot` to exist in the first place
                # (it is always derived internally from gettex's own
                # leverage identity, step 1 above) -- but when a `yfinance`
                # -style daily close is available, it is a genuinely
                # independent, any-issuer-outage-proof source, so it is used
                # as a (wider-tolerance) sanity check here instead of leaving
                # S_ref completely unchecked. See
                # `_DEFAULT_DAILY_CLOSE_REFERENCE_TOLERANCE_PCT` for why the
                # tolerance is wider than the live-quote cross-check above.
                daily_close = _resolve_scalar_or_mapping(daily_close_reference, underlying_id)
                if daily_close is not None:
                    deviation_pct = abs(s_ref - daily_close) / daily_close
                    if deviation_pct > self._daily_close_reference_tolerance_pct:
                        message = (
                            f"leverage-derived S_ref={s_ref:.4f} deviates {deviation_pct:.4%} "
                            f"from supplied daily_close_reference={daily_close:.4f} "
                            f"(tolerance {self._daily_close_reference_tolerance_pct:.4%})"
                        )
                        self._reference_spot_mismatches[underlying_id] = message
                        logger.warning(
                            "gettex_daily_close_reference_mismatch",
                            underlying_id=underlying_id,
                            s_ref=s_ref,
                            daily_close_reference=daily_close,
                            deviation_pct=deviation_pct,
                        )
                        continue
                    logger.debug(
                        "gettex_daily_close_reference_confirmed",
                        underlying_id=underlying_id,
                        s_ref=s_ref,
                        daily_close_reference=daily_close,
                        deviation_pct=deviation_pct,
                    )

            underlying_currency = _resolve_underlying_currency(underlying_id)
            fx_for_underlying = _resolve_scalar_or_mapping(fx_hint, underlying_id)
            needs_fx_hypothesis = underlying_currency is not None and underlying_currency != "EUR"

            # Befund 2 fix 2: the generic ProductSourceAdapter contract never
            # supplies fx_hint (see module docstring) -- when the caller
            # didn't either, learn a per-issuer fx candidate from this same
            # fetch's own data rather than being limited to the quanto-only
            # hypothesis for every row of a non-EUR underlying.
            learned_fx_by_issuer: dict[str, float] = {}
            if needs_fx_hypothesis and fx_for_underlying is None:
                learned_fx_by_issuer = self._learn_fx_by_issuer(
                    parsed_rows, s_ref, underlying_id=underlying_id
                )
                for issuer, fx in learned_fx_by_issuer.items():
                    self.last_learned_fx[f"{underlying_id}:{issuer}"] = fx

            # -- Steps 2-6: per-row ratio derivation + verification ----------
            for row in parsed_rows:
                stats.total += 1
                price = row.mid_price
                if price is None:
                    stats.ask_missing += 1
                    stats.note_example(row.isin)
                    self.last_errors.append(
                        GettexRowError(
                            source=_GETTEX_SOURCE_NAME,
                            isin=row.isin,
                            error="no usable bid or ask price",
                        )
                    )
                    continue
                if row.ask is None:
                    stats.ask_missing += 1

                row_fx_hint = fx_for_underlying
                if row_fx_hint is None and needs_fx_hypothesis:
                    row_fx_hint = learned_fx_by_issuer.get(_normalize_issuer(row.issuer_raw))

                outcome, ratio, quanto = self._derive_ratio(
                    row, s_ref, underlying_currency, row_fx_hint
                )
                if outcome == "ratio_rejected":
                    stats.ratio_rejected += 1
                    stats.note_example(row.isin)
                    detail = "ratio_raw did not snap to the canonical grid within tolerance"
                    self.last_errors.append(
                        GettexRowError(
                            source=_GETTEX_SOURCE_NAME,
                            isin=row.isin,
                            error=detail,
                        )
                    )
                    self._record_rejected_derivation(
                        row,
                        outcome=RatioDerivationOutcome.RATIO_REJECTED,
                        detail=detail,
                        underlying_id=underlying_id,
                        reference_spot=s_ref,
                        now=now,
                    )
                    continue
                if outcome == "verification_failed":
                    stats.verification_failed += 1
                    stats.note_example(row.isin)
                    detail = "implied-spot verification failed for every hypothesis tried"
                    self.last_errors.append(
                        GettexRowError(
                            source=_GETTEX_SOURCE_NAME,
                            isin=row.isin,
                            error=detail,
                        )
                    )
                    self._record_rejected_derivation(
                        row,
                        outcome=RatioDerivationOutcome.VERIFICATION_FAILED,
                        detail=detail,
                        underlying_id=underlying_id,
                        reference_spot=s_ref,
                        now=now,
                    )
                    continue
                if outcome == "quanto_ambiguous":
                    stats.quanto_ambiguous += 1
                    stats.note_example(row.isin)
                    detail = (
                        "quanto ambiguous: both the fx=1 and the non-quanto "
                        "hypothesis verified -- genuine ambiguity, never guessed"
                    )
                    self.last_errors.append(
                        GettexRowError(
                            source=_GETTEX_SOURCE_NAME,
                            isin=row.isin,
                            error=detail,
                        )
                    )
                    self._record_rejected_derivation(
                        row,
                        outcome=RatioDerivationOutcome.QUANTO_AMBIGUOUS,
                        detail=detail,
                        underlying_id=underlying_id,
                        reference_spot=s_ref,
                        now=now,
                    )
                    continue

                assert ratio is not None  # outcome == "ok" guarantees this
                stats.ratio_derived += 1

                barrier_eq_financing = math.isclose(
                    row.financing_level, row.knockout_barrier, rel_tol=1e-6, abs_tol=1e-6
                )
                if barrier_eq_financing:
                    open_end = True
                    product_type = ProductType.TURBO_OPEN_END
                else:
                    open_end = False
                    product_type = ProductType.UNKNOWN
                    self.last_errors.append(
                        GettexRowError(
                            source=_GETTEX_SOURCE_NAME,
                            isin=row.isin,
                            error=(
                                "barrier != financing_level: open_end/maturity left "
                                "unknown, product_type=UNKNOWN"
                            ),
                        )
                    )

                quality_score = _quality_score(
                    quote_presence=row.quote_presence, is_stale=row.is_stale
                )
                observation_time = row.quote_timestamp if row.quote_timestamp is not None else now

                snapshots.append(
                    ProductSnapshot(
                        isin=row.isin,
                        wkn=row.wkn,
                        issuer=_normalize_issuer(row.issuer_raw),
                        venue=_GETTEX_VENUE,
                        underlying_raw=row.underlying_raw,
                        underlying_id=underlying_id,
                        direction=row.direction,
                        product_type=product_type,
                        financing_level=row.financing_level,
                        knockout_barrier=row.knockout_barrier,
                        ratio=ratio,
                        currency="EUR",  # gettex trades exclusively on Boerse Muenchen in EUR
                        underlying_currency=underlying_currency,
                        quanto=quanto,
                        open_end=open_end,
                        # never present/derivable from gettex -- see module docstring
                        maturity=None,
                        first_trading_day=None,
                        bid=row.bid,
                        ask=row.ask,
                        bid_size=row.bid_size,
                        ask_size=row.ask_size,
                        quote_timestamp=row.quote_timestamp,
                        quote_presence=row.quote_presence,
                        bid_only=row.bid is not None and row.ask is None,
                        # Not exposed by gettex as an explicit flag. A
                        # genuinely knocked-out product's moneyness collapses
                        # to ~0, which the ratio-snap/verification gate above
                        # naturally rejects (division by a near-zero
                        # |S_ref - F|) rather than mispricing it -- so this is
                        # never guessed True, only ever left at the honest
                        # default.
                        knocked_out=False,
                        trading_hours=None,
                        product_age_days=None,
                        underlying_price_ref=row.underlying_price_ref,
                        underlying_price_ref_timestamp=row.underlying_price_ref_timestamp,
                        # ratio: derived-and-verified (see
                        # `_ratio_reliability_for_outcome`; `outcome == "ok"`
                        # is guaranteed here). barrier/financing_level: gettex
                        # reports `koLevelRefCurAbsolute`/
                        # `financingLevelRefCurAbsolute` directly as raw feed
                        # fields (no derivation, unlike ratio) -- SOURCE_REPORTED,
                        # same tier BNP/Citi use for their own directly-reported
                        # master data.
                        ratio_reliability=_ratio_reliability_for_outcome(outcome),
                        barrier_reliability=FieldReliability.SOURCE_REPORTED,
                        financing_level_reliability=FieldReliability.SOURCE_REPORTED,
                        raw_hash=_raw_hash(row.raw),
                        observation_time=observation_time,
                        available_at=now,
                        retrieved_at=now,
                        source_timestamp=row.quote_timestamp,
                        source=_GETTEX_SOURCE_NAME,
                        parser_version=_GETTEX_PARSER_VERSION,
                        is_stale=row.is_stale,
                        quality_score=quality_score,
                    )
                )

        _log_fetch_summary(stats)
        return snapshots

    def drain_rejected_ratio_derivations(self) -> list[RejectedRatioDerivation]:
        """Hand over this fetch's discarded derivation attempts exactly once.

        Satisfies ``adapters/registry.RejectedRatioDerivationSource``. The
        list is cleared as it is handed over, so a caller that persists these
        (``pipeline/universe.py``) cannot write the same attempt twice if it
        is ever called more than once per fetch -- `fetch_products` also
        resets it at the start of every call, so the two together bound
        duplicates from both ends.
        """
        drained = self._rejected_ratio_derivations
        self._rejected_ratio_derivations = []
        return drained

    def _record_rejected_derivation(
        self,
        row: _ParsedRow,
        *,
        outcome: RatioDerivationOutcome,
        detail: str,
        underlying_id: str,
        reference_spot: float,
        now: datetime,
    ) -> None:
        """Keep one discarded ratio-derivation attempt as research material.

        Every field comes straight off the parsed row, the reference spot in
        effect for it, or this fetch's own clock -- nothing is computed,
        defaulted or invented (CLAUDE.md rule 29). ``bid``/``ask`` stay
        ``None`` when gettex did not quote them. There is deliberately no
        ratio field: for ``ratio_rejected`` no candidate ever snapped to the
        grid, so there is no such quantity to store (see
        :class:`RejectedRatioDerivation`).
        """
        self._rejected_ratio_derivations.append(
            RejectedRatioDerivation(
                isin=row.isin,
                wkn=row.wkn,
                issuer=_normalize_issuer(row.issuer_raw),
                underlying_id=underlying_id,
                underlying_raw=row.underlying_raw,
                direction=row.direction,
                outcome=outcome,
                detail=detail,
                leverage=row.leverage,
                financing_level=row.financing_level,
                knockout_barrier=row.knockout_barrier,
                bid=row.bid,
                ask=row.ask,
                reference_spot=reference_spot,
                quote_timestamp=row.quote_timestamp,
                observed_at=now,
                source=_GETTEX_SOURCE_NAME,
                parser_version=_GETTEX_PARSER_VERSION,
                raw_hash=_raw_hash(row.raw),
            )
        )

    def _derive_ratio(
        self,
        row: _ParsedRow,
        s_ref: float,
        underlying_currency: str | None,
        fx_value: float | None,
    ) -> tuple[str, float | None, bool | None]:
        """Try to derive-and-verify this row's ratio.

        Returns ``(outcome, ratio, quanto)`` where ``outcome`` is one of
        ``"ok"``, ``"ratio_rejected"`` (no hypothesis's ratio_raw snapped to
        the canonical grid), ``"verification_failed"`` (a ratio snapped for
        at least one hypothesis, but zero passed the implied-spot check) or
        ``"quanto_ambiguous"`` (Befund 2: more than one currency hypothesis
        independently snapped AND verified -- genuine ambiguity, never
        guessed away, tracked separately from a plain verification failure
        for diagnostics). ``fx_value`` is either the caller-supplied
        ``fx_hint`` or a per-issuer value learned by
        :meth:`_learn_fx_by_issuer` -- this method does not care which.
        """
        price = row.mid_price
        assert price is not None  # caller already filtered rows with no usable price

        # Befund 2 fix 1: the snap tolerance widens for high-leverage rows
        # (same moneyness-shrinks-with-leverage effect as the S_ref
        # numerical-conditioning cap); the separate implied_spot_tolerance_pct
        # verification below is what actually decides correctness, so
        # widening this alone cannot admit an unverified row.
        effective_ratio_tol = _leverage_scaled_ratio_snap_tolerance(
            row.leverage,
            self._ratio_snap_tolerance_pct,
            self._ratio_snap_tolerance_leverage_ref,
            self._ratio_snap_tolerance_cap_pct,
        )

        needs_fx_hypothesis = underlying_currency is not None and underlying_currency != "EUR"
        # Befund 2 fix 3: only the two-hypothesis (non-EUR) path tightens
        # verification at high leverage -- see
        # _leverage_scaled_verification_tolerance's docstring for why a
        # second, potentially-wrong fx hypothesis needs this and the
        # single-hypothesis EUR path does not.
        effective_verify_tol = (
            _leverage_scaled_verification_tolerance(
                row.leverage,
                self._implied_spot_tolerance_pct,
                self._ratio_snap_tolerance_leverage_ref,
            )
            if needs_fx_hypothesis
            else self._implied_spot_tolerance_pct
        )

        def attempt(fx: float) -> _RatioAttempt:
            return _attempt_ratio(
                direction=row.direction,
                financing_level=row.financing_level,
                price=price,
                s_ref=s_ref,
                fx=fx,
                ratio_snap_tolerance_pct=effective_ratio_tol,
                implied_spot_tolerance_pct=effective_verify_tol,
            )

        if not needs_fx_hypothesis:
            result = attempt(1.0)
            if result.ratio is None:
                return "ratio_rejected", None, None
            if not result.verified:
                return "verification_failed", None, None
            return "ok", result.ratio, None

        attempts: list[tuple[_RatioAttempt, bool]] = [(attempt(1.0), True)]
        if fx_value is not None:
            attempts.append((attempt(fx_value), False))

        if not any(a.ratio is not None for a, _ in attempts):
            return "ratio_rejected", None, None

        passing = [(a, q) for a, q in attempts if a.ratio is not None and a.verified]
        if len(passing) == 0:
            return "verification_failed", None, None
        if len(passing) > 1:
            return "quanto_ambiguous", None, None
        ratio_attempt, quanto = passing[0]
        return "ok", ratio_attempt.ratio, quanto

    def _learn_fx_by_issuer(
        self, rows: Sequence[_ParsedRow], s_ref: float, *, underlying_id: str
    ) -> dict[str, float]:
        """Learn a per-issuer non-quanto fx candidate from this fetch's own
        data (Befund 2 fix 2) -- see the module-level constants' docstring
        block above for the full method rationale.

        Two-phase, per (normalized) issuer:

        1. Rows already unambiguous under the plain fx=1 hypothesis (using
           the same leverage-scaled tolerance every row gets) are set aside
           -- they need no learned fx at all.
        2. From the remaining (fx=1-unresolved) rows, a candidate fx is
           proposed via grid-ratio inversion
           (:func:`_fx_candidates_from_ratio_raw_at_1`) and a robust,
           MAD-filtered median taken (:func:`_robust_fx_median`). The
           candidate is then validated by literally re-running the same
           ratio-snap+verify pipeline on the group's own remaining rows: it
           is only accepted (and returned) if at least
           ``fx_learning_min_verified_fraction`` (default 0.5, a literal
           majority) of them verify under it -- an issuer group whose
           proposed fx fails this check is left exactly as conservative as
           before this fix (no learned fx returned for it, its rows fall
           through to the plain single-hypothesis path and are correctly
           rejected rather than mispriced).

        Every acceptance/rejection is logged (``gettex_fx_learned`` /
        ``gettex_fx_learning_rejected_low_support`` /
        ``gettex_fx_learning_insufficient_candidates``) for observability.
        """
        groups: dict[str, list[_ParsedRow]] = {}
        for row in rows:
            if row.mid_price is None:
                continue
            groups.setdefault(_normalize_issuer(row.issuer_raw), []).append(row)

        def try_fx(row: _ParsedRow, fx: float) -> _RatioAttempt:
            # Mirrors _derive_ratio's two-hypothesis tolerances exactly
            # (fix 1 widened snap + fix 3 tightened verify) -- this method
            # only ever runs in the two-hypothesis (non-EUR) context.
            price = row.mid_price
            assert price is not None
            snap_tol = _leverage_scaled_ratio_snap_tolerance(
                row.leverage,
                self._ratio_snap_tolerance_pct,
                self._ratio_snap_tolerance_leverage_ref,
                self._ratio_snap_tolerance_cap_pct,
            )
            verify_tol = _leverage_scaled_verification_tolerance(
                row.leverage,
                self._implied_spot_tolerance_pct,
                self._ratio_snap_tolerance_leverage_ref,
            )
            return _attempt_ratio(
                direction=row.direction,
                financing_level=row.financing_level,
                price=price,
                s_ref=s_ref,
                fx=fx,
                ratio_snap_tolerance_pct=snap_tol,
                implied_spot_tolerance_pct=verify_tol,
            )

        learned: dict[str, float] = {}
        for issuer, issuer_rows in groups.items():
            remaining: list[_ParsedRow] = []
            for row in issuer_rows:
                if abs(s_ref - row.financing_level) <= 0:
                    continue
                fx1_attempt = try_fx(row, 1.0)
                if fx1_attempt.ratio is None or not fx1_attempt.verified:
                    remaining.append(row)
            if len(remaining) < self._fx_learning_min_candidates:
                continue

            candidates: list[float] = []
            for row in remaining:
                price = row.mid_price
                assert price is not None
                moneyness = abs(s_ref - row.financing_level)
                ratio_raw_at_1 = price / moneyness
                candidates.extend(
                    _fx_candidates_from_ratio_raw_at_1(
                        ratio_raw_at_1, self._fx_learning_search_min, self._fx_learning_search_max
                    )
                )

            if len(candidates) < self._fx_learning_min_candidates:
                logger.debug(
                    "gettex_fx_learning_insufficient_candidates",
                    underlying_id=underlying_id,
                    issuer=issuer,
                    n_remaining=len(remaining),
                    n_candidates=len(candidates),
                )
                continue

            fx_proposed = _robust_fx_median(candidates, self._mad_k)
            if fx_proposed is None:
                continue

            n_verified = sum(1 for row in remaining if try_fx(row, fx_proposed).verified)
            verified_fraction = n_verified / len(remaining)
            if verified_fraction < self._fx_learning_min_verified_fraction:
                logger.info(
                    "gettex_fx_learning_rejected_low_support",
                    underlying_id=underlying_id,
                    issuer=issuer,
                    fx_proposed=fx_proposed,
                    n_remaining=len(remaining),
                    n_verified=n_verified,
                    verified_fraction=verified_fraction,
                )
                continue

            learned[issuer] = fx_proposed
            logger.info(
                "gettex_fx_learned",
                underlying_id=underlying_id,
                issuer=issuer,
                fx=fx_proposed,
                n_candidates=len(candidates),
                n_remaining=len(remaining),
                n_verified=n_verified,
                verified_fraction=verified_fraction,
            )

        return learned

    def _fetch_all_pages(
        self, gettex_underlying_id: int
    ) -> tuple[list[dict[str, Any]], int | None]:
        url = f"{self._base_url}{_GETTEX_PRODUCTS_PATH}"
        headers = _headers(self._http.user_agent)

        collected: list[dict[str, Any]] = []
        filtered_count: int | None = None

        for page in range(1, self._max_pages + 1):
            params = {
                "underlying": gettex_underlying_id,
                "productType": _GETTEX_PRODUCT_TYPE_LABEL,
                "rowsPerPage": self._rows_per_page,
                "page": page,
            }
            raw = self._http.get_json(url, params=params, headers=headers)
            if not isinstance(raw, dict) or "data" not in raw:
                raise AdapterError(f"unexpected gettex leverageProducts response shape: {raw!r}")
            data = raw["data"]
            pagination = data.get("pagination") if isinstance(data, dict) else None
            if isinstance(pagination, dict) and isinstance(pagination.get("filteredCount"), int):
                filtered_count = pagination["filteredCount"]

            groups = data.get("groups") if isinstance(data, dict) else None
            page_items = groups.get("products") if isinstance(groups, dict) else None
            if not isinstance(page_items, list):
                raise AdapterError("gettex leverageProducts 'data.groups.products' is not a list")

            collected.extend(page_items)

            if len(page_items) < self._rows_per_page:
                break
            if filtered_count is not None and len(collected) >= filtered_count:
                break

        return collected, filtered_count

    # -- healthcheck ------------------------------------------------------------

    def healthcheck(self) -> HealthCheckResult:
        checked_at = datetime.now(UTC)
        start = time.monotonic()
        url = f"{self._base_url}{_GETTEX_PRODUCTS_PATH}"
        params = {
            "underlying": _GETTEX_UNDERLYING_IDS["DAX"],
            "productType": _GETTEX_PRODUCT_TYPE_LABEL,
            "rowsPerPage": 10,
            "page": 1,
        }
        try:
            raw = self._http.get_json(url, params=params, headers=_headers(self._http.user_agent))
        except AdapterHttpError as exc:
            return HealthCheckResult(
                source=_GETTEX_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=None,
                checked_at=checked_at,
                message=f"HTTP error: {exc}",
            )
        except Exception as exc:  # defensive: healthcheck must never raise
            return HealthCheckResult(
                source=_GETTEX_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=None,
                checked_at=checked_at,
                message=f"unexpected error: {exc}",
            )
        latency_ms = (time.monotonic() - start) * 1000

        if not isinstance(raw, dict) or "data" not in raw:
            return HealthCheckResult(
                source=_GETTEX_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=latency_ms,
                checked_at=checked_at,
                message="unexpected leverageProducts response schema",
            )
        data = raw["data"]
        groups = data.get("groups") if isinstance(data, dict) else None
        products = groups.get("products") if isinstance(groups, dict) else None
        if not isinstance(products, list) or not products:
            return HealthCheckResult(
                source=_GETTEX_SOURCE_NAME,
                status=HealthStatus.WARN,
                ok=True,
                latency_ms=latency_ms,
                checked_at=checked_at,
                message="leverageProducts returned zero rows for the DAX probe",
            )

        required_keys = {
            "isin",
            "wkn",
            "underlying",
            "issuer",
            "structure.direction",
            "financingLevelRefCurAbsolute",
            "koLevelRefCurAbsolute",
            "leverage",
            "bid",
            "ask",
        }
        missing = required_keys - products[0].keys()
        if missing:
            return HealthCheckResult(
                source=_GETTEX_SOURCE_NAME,
                status=HealthStatus.FAIL,
                ok=False,
                latency_ms=latency_ms,
                checked_at=checked_at,
                message=f"leverageProducts row missing expected fields: {sorted(missing)}",
            )

        problems: list[str] = []
        if self._partial_universe:
            summary = ", ".join(
                f"{uid}: {covered}/{total}"
                for uid, (covered, total) in self._partial_universe.items()
            )
            problems.append(f"gettex_partial_universe (from last fetch_products): {summary}")
        if self._reference_spot_mismatches:
            summary = "; ".join(
                f"{uid}: {msg}" for uid, msg in self._reference_spot_mismatches.items()
            )
            problems.append(f"reference_spot_mismatch (from last fetch_products): {summary}")
        if problems:
            return HealthCheckResult(
                source=_GETTEX_SOURCE_NAME,
                status=HealthStatus.WARN,
                ok=True,
                latency_ms=latency_ms,
                checked_at=checked_at,
                message="; ".join(problems),
            )

        return HealthCheckResult(
            source=_GETTEX_SOURCE_NAME,
            status=HealthStatus.PASS,
            ok=True,
            latency_ms=latency_ms,
            checked_at=checked_at,
            message=f"leverageProducts reachable, {len(products)} sample row(s) OK",
        )


def _gettex_factory(src: SourceConfig, http: HttpClient) -> GettexAdapter:
    return GettexAdapter(http, base_url=src.base_url, max_pages=src.max_pages)


# -- idempotent registration --------------------------------------------------


def _register() -> None:
    from turboedge.adapters.registry import PRODUCT_ADAPTER_FACTORIES

    if _GETTEX_SOURCE_NAME not in PRODUCT_ADAPTER_FACTORIES:
        from turboedge.adapters.registry import register_product_adapter

        register_product_adapter(_GETTEX_SOURCE_NAME, _gettex_factory)


_register()
