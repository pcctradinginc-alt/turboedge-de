"""Shared statistics helpers for ``reporting/monthly.py`` and
``reporting/weekly.py`` (kept private -- not part of the reporting package's
public API, imported only by sibling report builders).

Centralizes the honesty-critical small-sample machinery so both reports
apply it identically (Master Spec §9.3, §27.4; CLAUDE.md rules 8/25/26):

- Wilson score confidence intervals for a success rate.
- Average-uniqueness sample weights for overlapping forward-ledger labels
  (Lopez de Prado-style concurrency discount), grouped by underlying.
- Empirical Expected Shortfall.
- Signal-family resolution for a ``LedgerEntry`` (the entry itself only
  carries ``signal_id``/``model_hash``; the family is looked up via the
  model registry, falling back to stripping a ``_v<digits>`` suffix off
  ``signal_id`` when the model was never registered).
- Leverage bucketing from the entry's frozen ``feature_snapshot``.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import numpy.typing as npt
from scipy import stats

from turboedge.storage.schemas import LedgerEntry, LedgerLabel, ModelRegistryEntry

_SIGNAL_ID_VERSION_RE = re.compile(r"^(?P<family>.+)_v\d+$")

#: Bucket for ledger entries whose signal family cannot be resolved.
#: Grouping them together is deliberate: see `signal_family_for`.
UNRESOLVED_SIGNAL_FAMILY = "unresolved"


@dataclass(frozen=True)
class WilsonInterval:
    """Wilson score confidence interval for a binomial success rate."""

    point: float
    lower: float
    upper: float
    n: int
    confidence: float


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> WilsonInterval:
    """Wilson score interval for ``successes / n`` at the given confidence
    level (Wilson 1927) -- much better calibrated than the naive normal
    approximation at the small trade counts a research system like this
    actually sees.
    """
    if n <= 0:
        return WilsonInterval(point=float("nan"), lower=0.0, upper=1.0, n=0, confidence=confidence)
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    phat = successes / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2.0 * n)) / denom
    half = (z * math.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * n)) / n)) / denom
    return WilsonInterval(
        point=phat,
        lower=max(0.0, center - half),
        upper=min(1.0, center + half),
        n=n,
        confidence=confidence,
    )


def _daterange(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def average_uniqueness_weights(
    pairs: Sequence[tuple[LedgerEntry, LedgerLabel]],
) -> dict[str, float]:
    """Average-uniqueness sample weight per ``entry_id`` (Master Spec §9.3 /
    §28, Lopez de Prado "Advances in Financial Machine Learning" ch. 4):
    overlapping forward-ledger labels are not independent observations.

    For each ``entry_id``, ``weight = mean over days in
    [prediction_time.date(), exit_due] of 1 / (number of *other entries in
    the same underlying* whose own interval also covers that day)`` -- a
    daily-grid approximation of the continuous-time average-uniqueness
    formula, grouped by ``underlying`` (predictions on the same underlying
    at overlapping times share information; different underlyings are
    treated as independent bets). ``sum(weights.values())`` over a set of
    entries is that set's *effective sample size*.
    """
    by_underlying: dict[str, list[tuple[str, date, date]]] = defaultdict(list)
    for entry, _label in pairs:
        by_underlying[entry.underlying].append(
            (entry.entry_id, entry.prediction_time.date(), entry.exit_due)
        )

    weights: dict[str, float] = {}
    for spans in by_underlying.values():
        concurrency: dict[date, int] = defaultdict(int)
        for _entry_id, start, end in spans:
            for d in _daterange(start, end):
                concurrency[d] += 1
        for entry_id, start, end in spans:
            days = list(_daterange(start, end))
            if not days:
                weights[entry_id] = 1.0
                continue
            weights[entry_id] = float(np.mean([1.0 / concurrency[d] for d in days]))
    return weights


def expected_shortfall(returns: npt.NDArray[np.float64], alpha: float = 0.05) -> float | None:
    """Empirical Expected Shortfall at level ``alpha``: the mean of the
    worst ``ceil(alpha * n)`` observations (at least 1). ``None`` if
    ``returns`` is empty.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return None
    k = max(1, math.ceil(alpha * r.size))
    tail = np.sort(r)[:k]
    return float(np.mean(tail))


def signal_family_for(entry: LedgerEntry, registry_by_hash: dict[str, str]) -> str:
    """Best-effort signal family for one ledger entry.

    ``LedgerEntry`` itself does not carry ``signal_family`` (only
    ``signal_id``/``model_hash`` -- Master Spec §25); the authoritative
    mapping is the model registry (``model_hash -> signal_family``,
    ``ModelRegistryEntry``). When the entry's ``model_hash`` was never
    registered (e.g. an old/retired model, or registry data not loaded in a
    test), falls back to stripping a trailing ``_v<digits>`` version suffix
    off ``signal_id`` (matching the ``<family>_v<n>`` naming convention used
    throughout this codebase, e.g. ``tsmom_horizon_norm_v1``); if that
    pattern does not match either, the raw ``signal_id`` is used verbatim.
    """
    family = registry_by_hash.get(entry.model_hash)
    if family is not None:
        return family
    m = _SIGNAL_ID_VERSION_RE.match(entry.signal_id)
    if m is not None:
        return m.group("family")
    # Falling back to the raw `signal_id` was wrong, and wrong in a way that
    # manufactured statistical significance. `scan-all` writes a per-run
    # `signal_id` (`<timestamp>-<hash>-<underlying>`) and an *ensemble*
    # `model_hash` that is not in the registry, so the lookup above always
    # missed and every scan run became its own "signal family": the weekly
    # tournament of 2026-09-26 listed ~84 of them instead of three, showed
    # tsmom/logit/null at n=0, ran Benjamini-Hochberg across 84 restatements
    # of the same strategy, and reported three families as significant
    # (`bh_rejected=True`) whose entries were all same-day positions from one
    # run.
    #
    # One visibly unresolved bucket is the honest representation: it is one
    # hypothesis rather than eighty-four, it cannot be promoted (nothing in
    # the registry matches it), and the name says the mapping is missing
    # instead of inventing a family per run.
    return UNRESOLVED_SIGNAL_FAMILY


def registry_hash_map(entries: Sequence[ModelRegistryEntry]) -> dict[str, str]:
    """``model_hash -> signal_family`` lookup built from the model registry."""
    return {e.model_hash: e.signal_family for e in entries}


def leverage_bucket(entry: LedgerEntry, edges: Sequence[float]) -> str:
    """Leverage bucket label for one entry.

    ``LedgerEntry`` has no dedicated leverage field (leverage is a
    ``CandidateEvaluation``-time concept -- Master Spec §25's ledger schema
    only freezes ``ratio``/``fx``/prices, not a derived leverage number);
    when the pipeline chose to freeze a ``"leverage"`` key into
    ``feature_snapshot`` (Master Spec §22/§48), that value is bucketed
    against ``edges`` (ascending); otherwise the bucket is ``"unknown"``
    rather than a fabricated guess (CLAUDE.md rule 29).
    """
    lev = entry.feature_snapshot.get("leverage")
    if lev is None:
        return "unknown"
    lo = 0.0
    for edge in edges:
        if lev < edge:
            return f"{lo:g}-{edge:g}x"
        lo = edge
    return f">{lo:g}x"


def holding_days(entry: LedgerEntry, label: LedgerLabel) -> int:
    """Actual trading days the position was held: ``time_to_ko_days`` when
    knocked out and known, else the full ``horizon_days`` (a normal
    horizon exit, or a KO whose exact day is unknown -- never guessed)."""
    if label.ko_hit and label.time_to_ko_days is not None:
        return label.time_to_ko_days
    return entry.horizon_days


def financing_drag_pct(entry: LedgerEntry, label: LedgerLabel) -> float | None:
    """Indicative financing drag over the holding period, as a fraction of
    ``entry_ask``, from the realized financing-level path.

    This is a coarse proxy from the raw financing-level change (``None``
    unless both ``financing_level_entry`` and ``label.financing_level_exit``
    are present) -- it is *not* the authoritative cost decomposition (that
    lives in ``pricing/fair_value.py`` / ``CostDecomposition``, W1); it only
    reflects how much the financing level itself moved, converted to
    product-currency terms via ``ratio``/``fx`` and expressed relative to
    ``entry_ask``. Sign convention: for LONG, a rising financing level is a
    cost (positive drag); for SHORT, a falling one is (so the raw change's
    sign is flipped).
    """
    if entry.financing_level_entry is None or label.financing_level_exit is None:
        return None
    if entry.entry_ask <= 0:
        return None
    raw = (
        (label.financing_level_exit - entry.financing_level_entry)
        * entry.ratio
        / entry.fx
        / entry.entry_ask
    )
    from turboedge.storage.schemas import Direction

    return raw if entry.direction is Direction.LONG else -raw


def quarter_label(dt: datetime) -> str:
    """``"2026Q3"``-style quarter label, matching
    ``learning.trials._quarter_label``'s convention."""
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{q}"


def last_day_of_month(d: date) -> date:
    """Last calendar day of ``d``'s month."""
    if d.month == 12:
        next_month = d.replace(year=d.year + 1, month=1, day=1)
    else:
        next_month = d.replace(month=d.month + 1, day=1)
    return next_month - timedelta(days=1)


__all__ = [
    "WilsonInterval",
    "average_uniqueness_weights",
    "expected_shortfall",
    "financing_drag_pct",
    "holding_days",
    "last_day_of_month",
    "leverage_bucket",
    "quarter_label",
    "registry_hash_map",
    "signal_family_for",
    "wilson_interval",
]
