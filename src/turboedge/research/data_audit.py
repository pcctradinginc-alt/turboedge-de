"""Is the forward dataset representative, and are its quotes executable?

Two questions that have to be answered before any forward result means
anything, and both are about data rather than about models.

MISSING LABELS (Phase 1)
------------------------
`learning/labeler.py` already refuses to invent an exit: a missing exit
quote becomes the last known bid discounted by the entry spread, flagged
``ambiguous_path=True`` with ``exit_reason=NO_EXIT_QUOTE_CONSERVATIVE``,
and when not even that exists the P&L fields stay ``None``
(``EXPIRED_NO_DATA``). Nothing is silently imputed, which is the thing
CLAUDE.md rule 29 is about.

That fallback is conservative **relative to the last quote**, and that is
exactly where the statistical problem starts. If an issuer stops quoting
*because* the product is about to knock out, the last known bid is not a
neutral anchor -- it is the last value from before the loss, so the
fallback is optimistic in precisely the cases that matter. Missingness
that depends on the unobserved outcome is MNAR, and MNAR is not
identifiable from the observed data alone; no reweighting fixes it without
an assumption that cannot be checked here.

What rescues this product class is a structural accident worth stating
plainly: **a turbo's exit value is mostly a function of something that is
never missing.** The underlying is observable whether or not the issuer
quotes, and intrinsic value is ``(spot - barrier) / ratio`` for a long.
So instead of a corrected point estimate -- which would be
scheinpräzise -- :func:`exit_value_bounds` brackets the exit with explicit,
stated assumptions, and :func:`pnl_bounds` turns that into a return
interval. The width of that interval *is* the answer to "how much could
the missing labels be biasing this": if a conclusion survives both ends,
missingness cannot overturn it.

EXECUTABLE QUOTES (Phase 2)
---------------------------
A certificate can look cheap because its issuer has not repriced it. The
apparent advantage then belongs to the stale quote, not to the product,
and it is gone before anyone can trade it. That is adverse selection on
quote age, and selecting the cheapest product selects for it: the
*stalest* quote is disproportionately likely to be the one that looks
best.

:func:`quote_age_seconds` measures the age the decision was actually made
on (``prediction_time - entry_quote_timestamp``), and
:func:`staleness_advantage` tests whether apparent cheapness and quote age
move together. A positive association is evidence that the cheapness is
partly staleness. Neither function claims a fill probability: without
execution data there is no measured fill, and modelling one would dress a
guess as a measurement.

This module reads no database and writes nothing. It is pure functions over
``(LedgerEntry, LedgerLabel | None)`` tuples so it can be tested without a
DuckDB fixture, and nothing in ``src/turboedge`` outside ``research/``
imports it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import numpy as np

from turboedge.pricing.intrinsic import leverage as effective_leverage
from turboedge.research.product_pairs import (
    ComparabilityCriteria,
    _relative_bucket,
)
from turboedge.storage.schemas import (
    Direction,
    ExitReason,
    FieldReliability,
    LedgerEntry,
    LedgerEntryStatus,
    LedgerLabel,
    ProductSnapshot,
)

#: One entry, with its label when it has one.
LedgerRow = tuple[LedgerEntry, "LedgerLabel | None"]


class ObservationState(StrEnum):
    """The brief's eight classes, mapped onto fields that actually exist.

    Deliberately not a new taxonomy: each value is decidable from
    ``LedgerEntry.status``, ``LedgerLabel.exit_reason`` and
    ``LedgerLabel.ambiguous_path``, so this classifies the ledger rather
    than reinterpreting it.
    """

    #: Still open and its exit is not yet due -- not missing, just young.
    NOT_YET_DUE = "not_yet_due"
    #: Open, past its ``exit_due``, and still unlabeled. This is the class
    #: that silently eats a dataset, so it is never merged with the above.
    OVERDUE_UNLABELED = "overdue_unlabeled"
    #: Labeled from a real exit quote at the horizon.
    LABELED_HORIZON = "labeled_horizon"
    #: Labeled as a knock-out, from the feed or from the underlying's bars.
    LABELED_KO = "labeled_ko"
    #: Labeled, but with the conservative last-known-bid fallback. Carries a
    #: value, and that value is an assumption.
    LABELED_NO_EXIT_QUOTE = "labeled_no_exit_quote"
    #: Labeled but flagged ``ambiguous_path`` for any other reason.
    LABELED_AMBIGUOUS = "labeled_ambiguous"
    #: Resolved to "no data": no P&L at all, correctly refusing to guess.
    EXPIRED_NO_DATA = "expired_no_data"
    #: Labeled, but the label carries no realized P&L -- a state that should
    #: not occur, so it is counted rather than ignored.
    LABELED_WITHOUT_PNL = "labeled_without_pnl"


def classify(row: LedgerRow, *, now: datetime) -> ObservationState:
    """Assign one ledger row to exactly one :class:`ObservationState`.

    ``now`` decides only whether an unlabeled open entry is young or
    overdue; it never reclassifies a labeled row.

    Raises:
        ValueError: if ``now`` is not timezone-aware. A naive ``now``
            compared against a tz-aware ``exit_due`` would raise deep
            inside the comparison, or worse, silently compare wall-clock
            times from different zones.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware (CLAUDE.md: all timestamps tz-aware UTC)")
    entry, label = row
    if label is None:
        if entry.status is LedgerEntryStatus.EXPIRED_NO_DATA:
            return ObservationState.EXPIRED_NO_DATA
        # `exit_due` is a calendar date, so the comparison is made in UTC
        # days. A date is "due" once that day has passed, which is the same
        # convention `learning/labeler.py` resolves exits on.
        return (
            ObservationState.OVERDUE_UNLABELED
            if entry.exit_due <= now.astimezone(UTC).date()
            else ObservationState.NOT_YET_DUE
        )
    if label.exit_reason is ExitReason.EXPIRED_NO_DATA:
        return ObservationState.EXPIRED_NO_DATA
    if label.realized_selected_pnl is None:
        return ObservationState.LABELED_WITHOUT_PNL
    if label.exit_reason is ExitReason.NO_EXIT_QUOTE_CONSERVATIVE:
        return ObservationState.LABELED_NO_EXIT_QUOTE
    if label.exit_reason is ExitReason.KO:
        return ObservationState.LABELED_KO
    if label.ambiguous_path:
        return ObservationState.LABELED_AMBIGUOUS
    return ObservationState.LABELED_HORIZON


#: States whose outcome is a real observation rather than an assumption.
TRUSTWORTHY_STATES: frozenset[ObservationState] = frozenset(
    {ObservationState.LABELED_HORIZON, ObservationState.LABELED_KO}
)


@dataclass(frozen=True, slots=True)
class CovariateContrast:
    """One covariate's mean among trustworthy vs assumption-carrying rows.

    The point is not a p-value. It is whether the two groups look alike: if
    entries whose outcome had to be assumed sit at systematically tighter
    barriers or wider spreads than entries with a real exit quote, then the
    labeled sample is not a random subset and MCAR is off the table.
    """

    name: str
    n_trustworthy: int
    n_assumed: int
    mean_trustworthy: float
    mean_assumed: float

    @property
    def difference(self) -> float:
        return self.mean_assumed - self.mean_trustworthy

    @property
    def comparable(self) -> bool:
        """Whether both groups have anything in them at all.

        With one side empty the contrast is undefined, and reporting a
        difference against an empty group is how "no evidence of bias" gets
        manufactured out of no data.
        """
        return self.n_trustworthy > 0 and self.n_assumed > 0


#: Entry covariates the brief asks the missingness to be analysed against,
#: restricted to the ones a `LedgerEntry` actually carries point-in-time.
_COVARIATES: tuple[str, ...] = (
    "entry_spread",
    "p_ko",
    "predicted_return",
    "uncertainty",
    "horizon_days",
    "lcb_ev",
)


@dataclass(frozen=True, slots=True)
class LedgerCensus:
    """The population of forward observations, by state and by covariate."""

    total: int
    by_state: Mapping[str, int]
    by_underlying: Mapping[str, Mapping[str, int]]
    by_issuer: Mapping[str, Mapping[str, int]]
    contrasts: tuple[CovariateContrast, ...]

    @property
    def trustworthy(self) -> int:
        return sum(self.by_state.get(s.value, 0) for s in TRUSTWORTHY_STATES)

    @property
    def assumed(self) -> int:
        """Rows that carry an outcome which is partly an assumption."""
        return self.by_state.get(
            ObservationState.LABELED_NO_EXIT_QUOTE.value, 0
        ) + self.by_state.get(ObservationState.LABELED_AMBIGUOUS.value, 0)

    @property
    def missing(self) -> int:
        """Rows that are due and have no usable outcome."""
        return self.by_state.get(ObservationState.OVERDUE_UNLABELED.value, 0) + self.by_state.get(
            ObservationState.EXPIRED_NO_DATA.value, 0
        )

    def summary(self) -> str:
        resolved = self.trustworthy + self.assumed + self.missing
        if resolved == 0:
            return (
                f"{self.total} forward observation(s), none resolved yet "
                f"({self.by_state.get(ObservationState.NOT_YET_DUE.value, 0)} not yet due) "
                "-- representativeness is not measurable"
            )
        return (
            f"{self.total} forward observation(s): {self.trustworthy} with a real exit, "
            f"{self.assumed} carrying an assumed value, {self.missing} due without one "
            f"({100.0 * self.missing / resolved:.1f}% of resolved)"
        )


def ledger_census(rows: Sequence[LedgerRow], *, now: datetime) -> LedgerCensus:
    """Classify every row and contrast covariates between the two groups.

    Shadow entries are included: they are forward observations too, and
    excluding them here would hide a missingness pattern that affects the
    shadow sample. Callers that want only live entries filter before
    calling.
    """
    states = [classify(row, now=now) for row in rows]
    by_underlying: dict[str, Counter[str]] = {}
    by_issuer: dict[str, Counter[str]] = {}
    for (entry, _), state in zip(rows, states, strict=True):
        by_underlying.setdefault(entry.underlying, Counter())[state.value] += 1
        by_issuer.setdefault(entry.issuer, Counter())[state.value] += 1

    trust_idx = [i for i, s in enumerate(states) if s in TRUSTWORTHY_STATES]
    assumed_idx = [
        i
        for i, s in enumerate(states)
        if s in (ObservationState.LABELED_NO_EXIT_QUOTE, ObservationState.LABELED_AMBIGUOUS)
        or s in (ObservationState.OVERDUE_UNLABELED, ObservationState.EXPIRED_NO_DATA)
    ]

    contrasts: list[CovariateContrast] = []
    for name in _COVARIATES:
        trust_vals = [
            v for i in trust_idx if (v := _numeric(getattr(rows[i][0], name, None))) is not None
        ]
        assumed_vals = [
            v for i in assumed_idx if (v := _numeric(getattr(rows[i][0], name, None))) is not None
        ]
        contrasts.append(
            CovariateContrast(
                name=name,
                n_trustworthy=len(trust_vals),
                n_assumed=len(assumed_vals),
                mean_trustworthy=float(np.mean(trust_vals)) if trust_vals else float("nan"),
                mean_assumed=float(np.mean(assumed_vals)) if assumed_vals else float("nan"),
            )
        )

    return LedgerCensus(
        total=len(rows),
        by_state=dict(Counter(s.value for s in states)),
        by_underlying={k: dict(v) for k, v in by_underlying.items()},
        by_issuer={k: dict(v) for k, v in by_issuer.items()},
        contrasts=tuple(contrasts),
    )


def _numeric(value: object) -> float | None:
    """Coerce a covariate to a float, or ``None`` if it is not usable.

    ``None`` and ``NaN`` are both dropped rather than treated as zero --
    a missing covariate in a missingness analysis must not become a value.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        f = float(value)
        return None if math.isnan(f) else f
    return None


@dataclass(frozen=True, slots=True)
class ExitValueBounds:
    """An interval for a product's exit value, from observable inputs only.

    ``lower``/``upper`` are per-certificate prices in the product's own
    currency. ``ko`` records whether the underlying's own path already
    settles the question, in which case the interval collapses.
    """

    lower: float
    upper: float
    ko: bool

    @property
    def width(self) -> float:
        return self.upper - self.lower


def exit_value_bounds(
    *,
    spot_at_exit: float,
    barrier: float,
    ratio: float,
    direction: Direction,
    ko_occurred: bool,
    ko_residual: float = 0.0,
    max_premium_fraction: float = 0.10,
    max_spread_fraction: float = 0.05,
) -> ExitValueBounds:
    """Bracket the exit value without using any issuer quote.

    The identity this rests on is that intrinsic value depends only on the
    underlying, the barrier and the ratio -- none of which go missing when
    an issuer stops quoting:

        intrinsic = (spot - barrier) / ratio      (long)
        intrinsic = (barrier - spot) / ratio      (short)

    The bounds then add two *stated* assumptions, and they are assumptions
    rather than measurements:

    - ``max_spread_fraction``: the bid can sit below intrinsic by at most
      this fraction, which is the worst execution haircut considered
      plausible. It sets the lower bound.
    - ``max_premium_fraction``: the issuer's time/gap premium above
      intrinsic is at most this fraction. It sets the upper bound.

    Both default to values that are deliberately wide rather than fitted,
    because a bound calibrated to the data it is used on is no longer a
    bound. Replace them with measured per-issuer quantiles once the
    archive supports it, and say so where the number is used.

    A knock-out collapses the interval to ``ko_residual`` (0 for
    ``turbo_open_end``/``turbo_classic``), matching ``learning/labeler.py``.
    ``ko_occurred`` is the caller's determination from the underlying's own
    high/low path -- which is exactly the input that is never missing.

    Raises:
        ValueError: on a non-positive ratio, a negative spot or barrier, or
            a fraction outside ``[0, 1)``.
    """
    if not ratio > 0:
        raise ValueError(f"ratio must be > 0, got {ratio!r}")
    if spot_at_exit < 0 or barrier < 0:
        raise ValueError(f"spot and barrier must be >= 0, got {spot_at_exit!r}, {barrier!r}")
    for name, frac in (
        ("max_premium_fraction", max_premium_fraction),
        ("max_spread_fraction", max_spread_fraction),
    ):
        if not 0.0 <= frac < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {frac!r}")
    if ko_residual < 0:
        raise ValueError(f"ko_residual must be >= 0, got {ko_residual!r}")

    if ko_occurred:
        return ExitValueBounds(lower=ko_residual, upper=ko_residual, ko=True)

    raw = (
        (spot_at_exit - barrier) / ratio
        if direction is Direction.LONG
        else (barrier - spot_at_exit) / ratio
    )
    # A negative intrinsic without a recorded KO means the barrier was
    # crossed but the caller did not see it; clamping to 0 is the
    # conservative reading and never raises the lower bound.
    intrinsic = max(raw, 0.0)
    return ExitValueBounds(
        lower=intrinsic * (1.0 - max_spread_fraction),
        upper=intrinsic * (1.0 + max_premium_fraction),
        ko=False,
    )


def pnl_bounds(*, entry_ask: float, bounds: ExitValueBounds) -> tuple[float, float]:
    """Turn an exit-value interval into a return interval on the entry ask.

    Entry is at Ask (CLAUDE.md rule 11), so the denominator is the price
    actually paid. ``(-1.0, -1.0)`` for a knocked-out product with no
    residual is a total loss, which is the correct answer and not a
    sentinel.

    Raises:
        ValueError: if ``entry_ask`` is not positive -- there is no return
            to compute against a price that was never paid.
    """
    if not entry_ask > 0:
        raise ValueError(f"entry_ask must be > 0, got {entry_ask!r}")
    return (bounds.lower / entry_ask - 1.0, bounds.upper / entry_ask - 1.0)


@dataclass(frozen=True, slots=True)
class QuoteAgeStats:
    """Distribution of the quote age each decision was actually made on."""

    n: int
    n_negative: int
    median_seconds: float
    p90_seconds: float
    max_seconds: float

    def summary(self) -> str:
        if self.n == 0:
            return "no entry carried a quote timestamp"
        return (
            f"{self.n} decision(s): median quote age {self.median_seconds:.0f}s, "
            f"p90 {self.p90_seconds:.0f}s, max {self.max_seconds:.0f}s"
            + (
                f", {self.n_negative} with a quote newer than the decision"
                if self.n_negative
                else ""
            )
        )


def quote_age_seconds(entry: LedgerEntry) -> float | None:
    """``prediction_time - entry_quote_timestamp``, in seconds.

    NOT the quote's age at the moment of the decision, and the difference
    matters. ``prediction_time`` is taken on the first line of
    ``run_scan``, before any product is fetched, so the issuer's quote
    timestamps normally land *after* it and this quantity is normally
    negative. See :class:`TimestampOrdering` for why that is a permissible
    ordering rather than a leak, and why quote age at the decision is not
    recoverable from the ledger at all.

    ``None`` when the entry carries no quote timestamp, which is itself a
    finding and must not be read as age zero.

    A *negative* age is returned as-is rather than clamped: it means the
    quote is stamped after the decision, which is a provenance defect worth
    surfacing, and clamping it to zero would hide it.
    """
    if entry.entry_quote_timestamp is None:
        return None
    return (entry.prediction_time - entry.entry_quote_timestamp).total_seconds()


def quote_freshness(entries: Sequence[LedgerEntry]) -> QuoteAgeStats:
    """Summarise quote age across decisions.

    Needs no labels, so unlike everything in Phase 1 this is measurable on
    an entirely unlabeled ledger.
    """
    ages = [a for e in entries if (a := quote_age_seconds(e)) is not None]
    if not ages:
        return QuoteAgeStats(
            n=0,
            n_negative=0,
            median_seconds=float("nan"),
            p90_seconds=float("nan"),
            max_seconds=float("nan"),
        )
    arr = np.asarray(ages, dtype=np.float64)
    return QuoteAgeStats(
        n=arr.size,
        n_negative=int(np.sum(arr < 0)),
        median_seconds=float(np.median(arr)),
        p90_seconds=float(np.quantile(arr, 0.9)),
        max_seconds=float(np.max(arr)),
    )


@dataclass(frozen=True, slots=True)
class StalenessAssociation:
    """Whether apparent cheapness and quote age move together.

    ``spearman`` is a rank correlation between quote age and the apparent
    advantage. Positive means the cheaper-looking quotes are the older
    ones, which is the adverse-selection pattern. It is an association on
    observational data and names no cause.

    ``distinct_ages`` and ``distinct_advantages`` exist because a ``nan``
    correlation on a large sample is not a result, it is a question: it
    means one of the two inputs has no variation at all, and which one
    decides what is actually wrong with the data. Reporting the
    correlation without them makes a degenerate input look like "no
    association found".
    """

    n: int
    spearman: float
    distinct_ages: int = 0
    distinct_advantages: int = 0

    @property
    def interpretable(self) -> bool:
        """Whether this is worth reading at all.

        Needs a sample (30 is a convention, not a test; below it the rank
        correlation of a handful of points is noise) *and* variation on
        both sides. Without the second condition a constant input reads as
        an interpretable null.
        """
        return self.n >= 30 and self.distinct_ages > 1 and self.distinct_advantages > 1

    @property
    def degenerate_side(self) -> str | None:
        """Which input has no variation, when one of them has none."""
        flat_age = self.distinct_ages <= 1
        flat_adv = self.distinct_advantages <= 1
        if flat_age and flat_adv:
            return "both"
        if flat_age:
            return "quote_age"
        if flat_adv:
            return "cheapness"
        return None


def staleness_advantage(
    *, quote_ages: Sequence[float], advantages: Sequence[float]
) -> StalenessAssociation:
    """Rank-correlate quote age against apparent price advantage.

    Spearman rather than Pearson because neither input is remotely normal
    (quote ages are heavy-tailed and bounded below) and only the ordering
    carries the claim. Implemented on ranks directly rather than pulling in
    another dependency.

    Raises:
        ValueError: if the two sequences differ in length.
    """
    if len(quote_ages) != len(advantages):
        raise ValueError(
            "quote_ages and advantages must have equal length, got "
            f"{len(quote_ages)} and {len(advantages)}"
        )
    n = len(quote_ages)
    ages = np.asarray(quote_ages, dtype=np.float64)
    advs = np.asarray(advantages, dtype=np.float64)
    # Rounded before counting: float noise far below any meaningful
    # difference would otherwise report variation that is not there.
    n_ages = int(np.unique(np.round(ages, 6)).size) if n else 0
    n_advs = int(np.unique(np.round(advs, 12)).size) if n else 0
    if n < 2:
        return StalenessAssociation(
            n=n, spearman=float("nan"), distinct_ages=n_ages, distinct_advantages=n_advs
        )
    a = _ranks(ages)
    b = _ranks(advs)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return StalenessAssociation(
            n=n, spearman=float("nan"), distinct_ages=n_ages, distinct_advantages=n_advs
        )
    return StalenessAssociation(
        n=n,
        spearman=float(np.corrcoef(a, b)[0, 1]),
        distinct_ages=n_ages,
        distinct_advantages=n_advs,
    )


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not invent an ordering."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # Average the ranks within each group of equal values.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(unique.size, dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


__all__ = [
    "PLAUSIBLE_SCAN_SECONDS",
    "TRUSTWORTHY_STATES",
    "CovariateContrast",
    "ExitValueBounds",
    "LedgerCensus",
    "LedgerRow",
    "ObservationState",
    "PointInTimeAudit",
    "QuoteAgeStats",
    "SnapshotStalenessReport",
    "StalenessAssociation",
    "TimestampOrdering",
    "WithinGroupAssociation",
    "classify",
    "classify_ordering",
    "exit_value_bounds",
    "ledger_census",
    "pnl_bounds",
    "point_in_time_audit",
    "quote_age_seconds",
    "quote_freshness",
    "snapshot_staleness_advantage",
    "staleness_advantage",
    "within_group_staleness_association",
]


@dataclass(frozen=True, slots=True)
class SnapshotStalenessReport:
    """Does apparent cheapness track quote age across comparable products?

    ``association`` pools, over every comparison group, each product's
    relative cheapness against its peers and its own quote age. A positive
    Spearman value means the cheaper-looking quotes in a group tend to be
    the older ones -- the adverse-selection pattern: picking the cheapest
    product partly picks the least recently repriced one.

    ``groups`` is the number of comparison groups with at least two
    different issuers, which is the sample size that matters; a large
    ``association.n`` built from very few groups is one market moment
    repeated, not independent evidence.
    """

    groups: int
    association: StalenessAssociation
    within_group: WithinGroupAssociation
    dropped: Mapping[str, int]


def _observation_window(retrieved: datetime, window_seconds: int) -> datetime:
    """Floor a retrieval time to its window, in UTC.

    Deliberately not ``product_pairs._window_start``: that one prefers the
    quote timestamp, which is the dimension under measurement here -- it
    would give every quote age its own window, so no stale quote could ever
    be compared against a fresh one. Grouping on the retrieval time asks
    the right question: of the products this process saw at the same
    moment, were the cheaper-looking ones the ones already more out of
    date?
    """
    if window_seconds < 1:
        raise ValueError(f"window_seconds must be >= 1, got {window_seconds!r}")
    epoch = int(retrieved.astimezone(UTC).timestamp())
    return datetime.fromtimestamp(epoch - epoch % window_seconds, tz=UTC)


def snapshot_staleness_advantage(
    snapshots: Sequence[ProductSnapshot],
    *,
    criteria: ComparabilityCriteria | None = None,
) -> SnapshotStalenessReport:
    """Correlate within-group cheapness against quote age.

    Comparison groups use
    :class:`~turboedge.research.product_pairs.ComparabilityCriteria` for the
    underlying, direction, time window and barrier-distance tolerance --
    but **deliberately not** its leverage dimension, and the reason is the
    whole validity of the measurement.

    ``pricing/intrinsic.leverage`` is computed from the ask. Bucketing on it
    would condition on the very quantity whose variation is being measured:
    a cheaper ask means a higher leverage means a different bucket, so the
    price differences would be partitioned away before they could be
    compared. Measured on a planted association, including the leverage
    bucket split 40 comparable products into 7 groups and collapsed a
    rank correlation of ~0.9 to 0.14.

    Barrier distance is safe, because it depends only on the barrier and
    the underlying's price, neither of which is the product's price. For
    products at the same barrier distance, leverage differences *are* the
    price differences -- which is what we want to look at, not control for.

    The time window is taken from ``observation_time`` -- when the archive
    looked -- and not from ``quote_timestamp``, for the same reason.
    ``product_pairs._window_start`` prefers the quote timestamp, which is
    the right choice when counting products quoted at the same moment, but
    here it would put every quote age in its own window and no stale quote
    could ever be compared against a fresh one. In the planted-association
    test it alone split 40 comparable products into 7 groups.

    Cheapness is measured as cost per unit of underlying exposure,
    ``ask / ratio``, relative to the group median:

        cheapness = (median_cost - own_cost) / median_cost

    so a positive value means cheaper than its peers. Entry is at Ask
    (CLAUDE.md rule 11), which is why the ask and not the mid is the cost.
    Only groups spanning at least two issuers are used: within one issuer
    the comparison is about that issuer's own curve, not about choosing
    between issuers.

    Quote age is ``retrieved_at - quote_timestamp``: how old the issuer's
    own quote already was when this process fetched it. It is deliberately
    **not** measured against ``observation_time``, and getting that wrong
    once is worth recording -- the schema defines ``observation_time`` as
    "when the underlying fact became true (e.g. a quote's own timestamp
    when known)", and ``adapters/issuer_feeds.py`` duly sets it *equal to*
    ``quote_timestamp``. The difference is therefore identically zero by
    construction. Measured against the real archive it produced exactly one
    distinct age across 4,752 comparable snapshots (run 40230382857), which
    looked like a dead field in the data and was a subtraction of a field
    from itself. ``retrieved_at`` is documented as "purely when this
    process fetched it", which is the only one of the three that carries
    the latency.

    A snapshot without a quote timestamp is dropped and counted rather than
    treated as fresh.
    """
    crit = criteria or ComparabilityCriteria()
    dropped: Counter[str] = Counter()
    groups: dict[tuple[str, Direction, datetime, int], list[tuple[float, float, str]]] = {}

    for snap in snapshots:
        if snap.underlying_id is None:
            dropped["no_underlying_id"] += 1
            continue
        if snap.knocked_out:
            dropped["knocked_out"] += 1
            continue
        if snap.ask is None or not snap.ask > 0 or snap.bid_only:
            dropped["no_ask"] += 1
            continue
        if snap.quote_timestamp is None:
            dropped["no_quote_timestamp"] += 1
            continue
        if snap.underlying_price_ref is None or not snap.underlying_price_ref > 0:
            dropped["no_underlying_reference"] += 1
            continue
        if snap.knockout_barrier is None:
            dropped["no_barrier"] += 1
            continue
        if crit.require_verified_ratio and snap.ratio_reliability is FieldReliability.UNVERIFIED:
            dropped["ratio_unverified"] += 1
            continue
        if not snap.ratio > 0:
            dropped["no_ratio"] += 1
            continue

        spot = float(snap.underlying_price_ref)
        barrier = float(snap.knockout_barrier)
        absolute = spot - barrier if snap.direction is Direction.LONG else barrier - spot
        barrier_pct = absolute / spot
        if not barrier_pct > 0.0:
            dropped["barrier_touched"] += 1
            continue
        # Leverage is no longer a grouping key (see the docstring), but a
        # product whose leverage cannot be computed at all has an
        # inconsistent spot/ask/ratio triple and is not comparable.
        try:
            lev = effective_leverage(spot, float(snap.ask), float(snap.ratio))
        except ValueError:
            dropped["leverage_not_computable"] += 1
            continue
        if not lev > 0:
            dropped["leverage_not_positive"] += 1
            continue

        age = (snap.retrieved_at - snap.quote_timestamp).total_seconds()
        cost = float(snap.ask) / float(snap.ratio)
        key = (
            snap.underlying_id,
            snap.direction,
            _observation_window(snap.retrieved_at, crit.window_seconds),
            _relative_bucket(barrier_pct, crit.barrier_pct_tolerance),
        )
        groups.setdefault(key, []).append((cost, age, snap.issuer))

    ages: list[float] = []
    cheapness: list[float] = []
    per_group: list[tuple[list[float], list[float]]] = []
    usable_groups = 0
    for members in groups.values():
        if len({issuer for _, _, issuer in members}) < 2:
            dropped["single_issuer_group"] += 1
            continue
        costs = np.asarray([c for c, _, _ in members], dtype=np.float64)
        median = float(np.median(costs))
        if not median > 0:
            dropped["degenerate_group_cost"] += 1
            continue
        usable_groups += 1
        group_ages: list[float] = []
        group_cheapness: list[float] = []
        for cost, age, _ in members:
            group_cheapness.append((median - cost) / median)
            group_ages.append(age)
        per_group.append((group_ages, group_cheapness))
        cheapness.extend(group_cheapness)
        ages.extend(group_ages)

    return SnapshotStalenessReport(
        groups=usable_groups,
        association=staleness_advantage(quote_ages=ages, advantages=cheapness),
        within_group=within_group_staleness_association(per_group),
        dropped=dict(dropped),
    )


@dataclass(frozen=True, slots=True)
class WithinGroupAssociation:
    """The staleness/cheapness association, computed the honest way.

    The pooled correlation in :class:`StalenessAssociation` mixes two
    different comparisons. Cheapness is defined *relative to a group's
    median*, so it is already within-group; quote age is absolute. Any
    systematic difference *between* groups -- products fetched later in a
    scan carrying both older quotes and a different price dispersion --
    therefore enters the pooled number as if it were the within-group
    effect under test. On the real archive the pooled value is +0.2952 and
    the within-group mean is what this reports instead.

    One group is one market moment, so one group is **one** observation,
    not one per member. ``mean`` averages the per-group rank correlations
    with equal weight and the interval resamples *groups*, which is the
    only resampling that respects that.
    """

    groups_used: int
    members: int
    mean: float
    ci_low: float
    ci_high: float

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval is entirely on one side of zero.

        Deliberately not called "significant": this is an observational
        association over one archive, with no pre-registered hypothesis
        behind it.
        """
        if math.isnan(self.ci_low) or math.isnan(self.ci_high):
            return False
        return (self.ci_low > 0.0) or (self.ci_high < 0.0)


def within_group_staleness_association(
    groups: Sequence[tuple[Sequence[float], Sequence[float]]],
    *,
    min_members: int = 4,
    reps: int = 4000,
    alpha: float = 0.05,
    seed: int = 23,
) -> WithinGroupAssociation:
    """Average the per-group rank correlation, with a group bootstrap.

    ``groups`` is one ``(quote_ages, cheapness)`` pair per comparison
    group. A group with fewer than ``min_members`` usable points, or with
    no variation on either side, contributes nothing: a rank correlation of
    two or three points is noise, and averaging noise into the mean would
    bias it toward whatever the small groups happen to show.

    Raises:
        ValueError: if ``min_members < 3``, ``reps < 1``, or ``alpha`` is
            outside ``(0, 1)``.
    """
    if min_members < 3:
        raise ValueError(f"min_members must be >= 3, got {min_members!r}")
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    per_group: list[float] = []
    members = 0
    for ages, advs in groups:
        if len(ages) != len(advs):
            raise ValueError("each group's ages and cheapness must have equal length")
        if len(ages) < min_members:
            continue
        assoc = staleness_advantage(quote_ages=ages, advantages=advs)
        if math.isnan(assoc.spearman):
            continue
        per_group.append(assoc.spearman)
        members += len(ages)

    if not per_group:
        return WithinGroupAssociation(
            groups_used=0,
            members=0,
            mean=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
        )

    arr = np.asarray(per_group, dtype=np.float64)
    mean = float(np.mean(arr))
    if arr.size < 3:
        # Too few groups for a bootstrap to mean anything; report the mean
        # with an explicitly undefined interval rather than a fake one.
        return WithinGroupAssociation(
            groups_used=arr.size,
            members=members,
            mean=mean,
            ci_low=float("nan"),
            ci_high=float("nan"),
        )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(reps, arr.size))
    means = arr[draws].mean(axis=1)
    return WithinGroupAssociation(
        groups_used=arr.size,
        members=members,
        mean=mean,
        ci_low=float(np.quantile(means, alpha / 2.0)),
        ci_high=float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


class TimestampOrdering(StrEnum):
    """Where a decision's quote sits relative to the decision's own stamp.

    THE DISTINCTION THIS EXISTS TO MAKE. ``LedgerEntry.prediction_time`` is
    taken at the very first line of ``pipeline/scan.run_scan``
    (``prediction_time = clock()``), *before* any product is fetched. The
    issuer's quote timestamps therefore land after it as a matter of
    course, and a negative "quote age" is the normal case rather than a
    defect.

    That is a permissible ordering, not a leak. A leak would mean using
    information from the future *of the decision*; here the quote is the
    price the decision was actually made on and transacted at. The feature
    check ``available_at <= prediction_time`` is if anything conservative,
    because it holds features to a moment earlier than the one the
    selection really happened at.

    What it does mean is that **quote age at the decision is not
    measurable from the ledger**: the ledger records the scan's start, not
    the instant the candidate was chosen. The archive-side measure,
    ``retrieved_at - quote_timestamp`` in
    :func:`snapshot_staleness_advantage`, is the one that answers the
    freshness question, and A02 uses that.

    A quote stamped *far* beyond any plausible scan duration is a different
    matter and is separated out, because that cannot be explained by
    ordering alone.
    """

    #: Quote predates the recorded decision stamp. Unambiguous.
    QUOTE_BEFORE_DECISION = "quote_before_decision"
    #: Quote follows it, but within one plausible scan. Expected.
    QUOTE_WITHIN_SCAN_WINDOW = "quote_within_scan_window"
    #: Quote follows it by more than any scan could take. Needs a cause.
    QUOTE_BEYOND_SCAN_WINDOW = "quote_beyond_scan_window"
    #: No quote timestamp at all -- a finding, never read as age zero.
    NO_QUOTE_TIMESTAMP = "no_quote_timestamp"


#: How long one scan may plausibly take, in seconds. Measured against real
#: pipeline runs, a `scan` job completes in about 10 minutes including
#: every underlying, so 15 minutes is a generous ceiling for the gap
#: between one `run_scan` call's first line and the quotes it then fetches.
#: Anything past it is reported separately rather than absorbed.
PLAUSIBLE_SCAN_SECONDS = 900.0


@dataclass(frozen=True, slots=True)
class PointInTimeAudit:
    """The full ordering picture for a set of decisions."""

    n: int
    by_ordering: Mapping[str, int]
    by_issuer: Mapping[str, Mapping[str, int]]
    #: Quantiles of ``prediction_time - entry_quote_timestamp`` in seconds.
    #: Negative means the quote is stamped after the decision's stamp.
    age_quantiles: Mapping[str, float]
    min_age_seconds: float
    max_age_seconds: float

    @property
    def beyond_window(self) -> int:
        return self.by_ordering.get(TimestampOrdering.QUOTE_BEYOND_SCAN_WINDOW.value, 0)

    @property
    def leak_suspected(self) -> bool:
        """Whether any case cannot be explained by scan ordering.

        Deliberately conservative: it is True when a single decision's
        quote is stamped beyond a plausible scan duration. Such a case may
        still turn out to be a provider clock, but it cannot be waved
        through as ordering.
        """
        return self.beyond_window > 0

    def summary(self) -> str:
        if self.n == 0:
            return "no decision carried a quote timestamp"
        before = self.by_ordering.get(TimestampOrdering.QUOTE_BEFORE_DECISION.value, 0)
        within = self.by_ordering.get(TimestampOrdering.QUOTE_WITHIN_SCAN_WINDOW.value, 0)
        return (
            f"{self.n} decision(s): {before} quote-first, {within} within the "
            f"{PLAUSIBLE_SCAN_SECONDS:.0f}s scan window, {self.beyond_window} beyond it"
        )


def classify_ordering(
    entry: LedgerEntry, *, scan_seconds: float = PLAUSIBLE_SCAN_SECONDS
) -> TimestampOrdering:
    """Classify one decision's quote ordering.

    Raises:
        ValueError: if ``scan_seconds`` is negative.
    """
    if scan_seconds < 0:
        raise ValueError(f"scan_seconds must be >= 0, got {scan_seconds!r}")
    age = quote_age_seconds(entry)
    if age is None:
        return TimestampOrdering.NO_QUOTE_TIMESTAMP
    if age >= 0:
        return TimestampOrdering.QUOTE_BEFORE_DECISION
    if -age <= scan_seconds:
        return TimestampOrdering.QUOTE_WITHIN_SCAN_WINDOW
    return TimestampOrdering.QUOTE_BEYOND_SCAN_WINDOW


def point_in_time_audit(
    entries: Sequence[LedgerEntry], *, scan_seconds: float = PLAUSIBLE_SCAN_SECONDS
) -> PointInTimeAudit:
    """Audit the quote/decision ordering across every decision.

    Reports the distribution rather than a single statistic: a median of
    -20 seconds and a maximum of -5,587 are two different situations, and
    only the second one needs explaining.
    """
    orderings = [classify_ordering(e, scan_seconds=scan_seconds) for e in entries]
    by_issuer: dict[str, Counter[str]] = {}
    for entry, ordering in zip(entries, orderings, strict=True):
        by_issuer.setdefault(entry.issuer, Counter())[ordering.value] += 1

    ages = [a for e in entries if (a := quote_age_seconds(e)) is not None]
    if ages:
        arr = np.asarray(ages, dtype=np.float64)
        quantiles = {
            f"p{int(q * 100)}": float(np.quantile(arr, q))
            for q in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
        }
        lo, hi = float(arr.min()), float(arr.max())
    else:
        quantiles, lo, hi = {}, float("nan"), float("nan")

    return PointInTimeAudit(
        n=len(ages),
        by_ordering=dict(Counter(o.value for o in orderings)),
        by_issuer={k: dict(v) for k, v in by_issuer.items()},
        age_quantiles=quantiles,
        min_age_seconds=lo,
        max_age_seconds=hi,
    )
