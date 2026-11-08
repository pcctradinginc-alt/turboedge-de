"""Autonomous Alpha Discovery Engine, Phase 1: bounded and DEVELOPMENT-only.

The agent does not execute generated code. It proposes previously unlisted
two-feature quote-selection rules from *entry-time observations only*, then
runs one frozen rule against a later chronological validation segment using
real historical Ask/Bid quotes, an explicit no-trade baseline and an issuer
baseline. This is hypothesis *discovery*, not an economic alpha confirmation.

All history belongs to the existing research_opportunities, research_trials,
and research_screens tables. There is no parallel research database; there
are no changes to model weights, gates, portfolio state or trading orders.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from itertools import pairwise, permutations
from math import isfinite
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from turboedge.meta.research_opportunity import (
    Estimate,
    InformationFamily,
    ResearchKind,
    ResearchOpportunity,
)
from turboedge.research.controller import EXPLORATORY_SCREENS_PER_QUARTER
from turboedge.research.product_pair_net_ev import (
    EXECUTION_BUFFER_BPS,
    MAX_QUOTE_AGE_S,
    _fresh_exit,
    _valid_quote,
)
from turboedge.research.product_pairs import ComparabilityCriteria, census
from turboedge.research.step import EXECUTED_SCREEN_STATUSES, SCREEN_KIND
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Direction,
    ProductSnapshot,
    ResearchScreen,
    ResearchTrial,
    ScreenStatus,
    TrialStatus,
)

DSL_VERSION = "1"
HARNESS = "reviewed_feature_pair_dsl"
UNDERLYING = "DAX"
DIRECTION = Direction.LONG
MIN_DEVELOPMENT_DAYS = 12
MIN_VALIDATION_DAYS = 48
MIN_DAY_SPACING = 3
EMBARGO_DAYS = 2
MAX_PROPOSALS_PER_CYCLE = 3
MIN_FEATURE_CONTRAST = 0.02
MAX_EXPLORATORY_TESTS = 6
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261108
MIN_ECONOMIC_LCB = 0.001
EXIT_EARLIEST = timedelta(hours=24)
EXIT_LATEST = timedelta(hours=48)


class Feature(StrEnum):
    RELATIVE_SPREAD = "relative_spread"
    QUOTE_AGE = "quote_age"
    BID_DEPTH = "bid_depth"


class Rule(BaseModel):
    """Fixed, reviewed DSL: exactly two known features, no expressions/code.

    Values are relative, dimensionless entry-time ranking penalties. Smaller
    score is better. New rules come from measured combinations; there is no
    ability to modify source code or execute free-form research instructions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    underlying: Literal["DAX"] = UNDERLYING
    direction: Literal["LONG"] = "LONG"
    primary: Feature
    secondary: Feature
    primary_weight: Literal[0.65] = 0.65
    holding_hours: Literal[24] = 24
    version: Literal["1"] = DSL_VERSION

    @model_validator(mode="after")
    def distinct_features(self) -> Self:
        if self.primary == self.secondary:
            raise ValueError("a novel rule needs two distinct observable features")
        return self

    @property
    def hypothesis_id(self) -> str:
        fingerprint = hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()[:16]
        return f"RO-GEN-{fingerprint.upper()}"


@dataclass(frozen=True, slots=True)
class PairDay:
    day: date
    left: ProductSnapshot
    right: ProductSnapshot


@dataclass(frozen=True, slots=True)
class Candidate:
    rule: Rule
    observable_contrast: float
    priority: float
    overlap_with_failures: int


@dataclass(frozen=True, slots=True)
class Measurement:
    verdict: str
    validation_days: int
    completed_days: int
    missing_exit_days: int
    missing_feature_days: int
    mean_net: float | None
    lcb_net: float | None
    mean_excess_naive: float | None
    lcb_excess_naive: float | None


@dataclass(frozen=True, slots=True)
class DiscoveryOutcome:
    status: str
    eligible_days: int
    development_days: int
    validation_days: int
    candidate_ids: tuple[str, ...]
    selected_hypothesis_id: str | None
    screen_id: str | None
    verdict: str | None
    reason: str
    measurement: Measurement | None = None


def _feature(snapshot: ProductSnapshot, name: Feature) -> float | None:
    """Only as-of-entry features. A missing size must not become a free fill."""
    if name is Feature.RELATIVE_SPREAD:
        if snapshot.bid is None or snapshot.ask is None or snapshot.ask <= 0:
            return None
        return max(0.0, (snapshot.ask - snapshot.bid) / snapshot.ask)
    if name is Feature.QUOTE_AGE:
        if snapshot.quote_timestamp is None:
            return None
        return (snapshot.retrieved_at - snapshot.quote_timestamp).total_seconds() / MAX_QUOTE_AGE_S
    if snapshot.bid_size is None or snapshot.bid_size <= 0:
        return None
    return 1.0 / (1.0 + snapshot.bid_size / 100.0)


def _rule_score(snapshot: ProductSnapshot, rule: Rule) -> float | None:
    a, b = _feature(snapshot, rule.primary), _feature(snapshot, rule.secondary)
    if a is None or b is None:
        return None
    return rule.primary_weight * a + (1.0 - rule.primary_weight) * b


def _candidate_sessions(snapshots: list[ProductSnapshot], *, now: datetime) -> list[PairDay]:
    """PIT-valid first cross-issuer pair on each non-overlapping UTC day."""
    eligible = [
        s
        for s in snapshots
        if s.retrieved_at <= now
        and s.available_at <= s.retrieved_at
        and s.underlying_id == UNDERLYING
        and s.direction is DIRECTION
    ]
    by_quote: dict[tuple[str, datetime], list[ProductSnapshot]] = defaultdict(list)
    for s in eligible:
        if s.quote_timestamp is None:
            continue
        second = int(s.quote_timestamp.astimezone(UTC).timestamp())
        minute = datetime.fromtimestamp(second - second % 60, tz=UTC)
        by_quote[(s.isin, minute)].append(s)
    for quotes in by_quote.values():
        quotes.sort(key=lambda s: (s.retrieved_at, s.raw_hash))

    matched = census(
        eligible, criteria=ComparabilityCriteria(window_seconds=60, require_verified_ratio=True)
    ).pairs
    out: list[PairDay] = []
    last_day: date | None = None
    for pair in sorted(
        matched, key=lambda p: (p.window_start, p.issuer_a, p.isin_a, p.issuer_b, p.isin_b)
    ):
        day = pair.window_start.date()
        if last_day is not None and (day - last_day).days < MIN_DAY_SPACING:
            continue
        left = next(
            (q for q in by_quote.get((pair.isin_a, pair.window_start), []) if _valid_quote(q)),
            None,
        )
        right = next(
            (q for q in by_quote.get((pair.isin_b, pair.window_start), []) if _valid_quote(q)),
            None,
        )
        if left is None or right is None:
            continue
        if abs((left.retrieved_at - right.retrieved_at).total_seconds()) > 60:
            continue
        out.append(PairDay(day=day, left=left, right=right))
        last_day = day
    return out


def _split(sessions: list[PairDay]) -> tuple[list[PairDay], list[PairDay]]:
    if not sessions:
        return [], []
    middle = len(sessions) // 2
    development = sessions[:middle]
    cutoff = development[-1].day + timedelta(days=EMBARGO_DAYS)
    return development, [s for s in sessions[middle:] if s.day > cutoff]


def generate(development: list[PairDay], *, history: list[ResearchScreen]) -> list[Candidate]:
    """Learn *which observable differences exist*, never which exits won."""
    if len(development) < MIN_DEVELOPMENT_DAYS:
        return []
    contrast: dict[Feature, float] = {}
    for feature in Feature:
        differences = []
        for pair in development:
            a, b = _feature(pair.left, feature), _feature(pair.right, feature)
            if a is not None and b is not None:
                differences.append(abs(a - b))
        if len(differences) == len(development):
            value = float(np.mean(differences))
            if value >= MIN_FEATURE_CONTRAST:
                contrast[feature] = value

    rejected = [
        s
        for s in history
        if s.harness == HARNESS and s.verdict in {"REJECTED", "NO_DEMONSTRATED_NET_ALPHA"}
    ]
    proposals: list[Candidate] = []
    for primary, secondary in permutations(contrast, 2):
        rule = Rule(primary=primary, secondary=secondary)
        overlap = sum(
            primary.value in (s.spec.get("primary"), s.spec.get("secondary"))
            or secondary.value in (s.spec.get("primary"), s.spec.get("secondary"))
            for s in rejected
        )
        # An information/risk-ranking heuristic, NOT estimated economic value.
        # Penalises similarity to prior failures; deterministic ties.
        feature_score = contrast[primary] * 0.65 + contrast[secondary] * 0.35
        priority = feature_score / (1 + overlap)
        proposals.append(
            Candidate(
                rule=rule,
                observable_contrast=feature_score,
                priority=priority,
                overlap_with_failures=overlap,
            )
        )
    return sorted(proposals, key=lambda c: (-c.priority, c.rule.hypothesis_id))


def _exit_return(
    entry: ProductSnapshot,
    *,
    histories: dict[str, list[ProductSnapshot]],
) -> float | None:
    """Ask to actual later Bid, KO losses first; no invented exit prices."""
    if entry.ask is None:
        return None
    window = [
        s
        for s in histories.get(entry.isin, ())
        if entry.retrieved_at < s.retrieved_at <= entry.retrieved_at + EXIT_LATEST
        and s.available_at <= s.retrieved_at
    ]
    exits = [
        s for s in window if s.retrieved_at >= entry.retrieved_at + EXIT_EARLIEST and _fresh_exit(s)
    ]
    exit_snapshot = min(exits, key=lambda x: x.retrieved_at) if exits else None
    ko_before_exit = [
        s
        for s in window
        if s.knocked_out and (exit_snapshot is None or s.retrieved_at <= exit_snapshot.retrieved_at)
    ]
    if ko_before_exit:
        return -1.0 - EXECUTION_BUFFER_BPS / 10000.0
    if exit_snapshot is None or exit_snapshot.bid is None:
        return None
    return float(exit_snapshot.bid / entry.ask - 1.0 - EXECUTION_BUFFER_BPS / 10000.0)


def _lower_confidence_bound(values: list[float]) -> float | None:
    if len(values) < MIN_VALIDATION_DAYS or not all(isfinite(x) for x in values):
        return None
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.mean(data[rng.integers(0, len(data), size=(BOOTSTRAP_REPS, len(data)))], axis=1)
    # At most six prespecified DSL permutations: family-wise one-sided 5%.
    return float(np.quantile(means, 0.05 / MAX_EXPLORATORY_TESTS))


def evaluate(
    rule: Rule,
    validation: list[PairDay],
    *,
    snapshots: list[ProductSnapshot],
    now: datetime,
) -> Measurement:
    """Never transform incomplete validation into a profitable observation."""
    # Externally supplied validation rows must be truly separate trading
    # sessions; repeating one positive day must never inflate sample size.
    ordered_days = [pair.day for pair in validation]
    if len(set(ordered_days)) != len(ordered_days) or any(
        (b - a).days < MIN_DAY_SPACING for a, b in pairwise(ordered_days)
    ):
        return Measurement(
            verdict="INVALID_SAMPLE",
            validation_days=len(validation),
            completed_days=0,
            missing_exit_days=0,
            missing_feature_days=0,
            mean_net=None,
            lcb_net=None,
            mean_excess_naive=None,
            lcb_excess_naive=None,
        )
    histories: dict[str, list[ProductSnapshot]] = defaultdict(list)
    for s in snapshots:
        if s.retrieved_at <= now and s.available_at <= s.retrieved_at:
            histories[s.isin].append(s)
    for arr in histories.values():
        arr.sort(key=lambda s: (s.retrieved_at, s.raw_hash))

    net: list[float] = []
    excess: list[float] = []
    missing_exit = 0
    missing_feature = 0
    for pair in validation:
        if pair.left.retrieved_at > now or pair.right.retrieved_at > now:
            missing_feature += 1
            continue
        a, b = _rule_score(pair.left, rule), _rule_score(pair.right, rule)
        if a is None or b is None:
            missing_feature += 1
            continue
        selected = min(
            ((a, pair.left), (b, pair.right)),
            key=lambda x: (x[0], x[1].issuer, x[1].isin),
        )[1]
        naive = min((pair.left, pair.right), key=lambda s: (s.issuer, s.isin))
        selected_result = _exit_return(selected, histories=histories)
        naive_result = _exit_return(naive, histories=histories)
        if selected_result is None or naive_result is None:
            missing_exit += 1
            continue
        net.append(selected_result)
        excess.append(selected_result - naive_result)

    bound_net = _lower_confidence_bound(net)
    bound_excess = _lower_confidence_bound(excess)
    if missing_exit or missing_feature:
        verdict = "INCOMPLETE_EVIDENCE"
    elif len(validation) < MIN_VALIDATION_DAYS or bound_net is None or bound_excess is None:
        verdict = "AWAIT_DATA"
    elif bound_net > MIN_ECONOMIC_LCB and bound_excess > 0:
        verdict = "CANDIDATE_REQUIRES_INDEPENDENT_CONFIRMATION"
    else:
        verdict = "NO_DEMONSTRATED_NET_ALPHA"
    return Measurement(
        verdict=verdict,
        validation_days=len(validation),
        completed_days=len(net),
        missing_exit_days=missing_exit,
        missing_feature_days=missing_feature,
        mean_net=float(np.mean(net)) if net else None,
        lcb_net=bound_net,
        mean_excess_naive=float(np.mean(excess)) if excess else None,
        lcb_excess_naive=bound_excess,
    )


def _opportunity(candidate: Candidate, validation_days: int) -> ResearchOpportunity:
    rule = candidate.rule
    return ResearchOpportunity(
        hypothesis_id=rule.hypothesis_id,
        description=(
            f"Automatically generated {DSL_VERSION} DSL hypothesis: DAX LONG selection "
            f"by entry-known {rule.primary.value} + {rule.secondary.value}; "
            "does it beat NO_TRADE and issuer-alphabetical naive on real Ask/Bid? "
            "DEVELOPMENT/VALIDATION only; economic sign never used to generate the rule."
        ),
        information_family=InformationFamily.PRODUCT_SELECTION,
        research_kind=ResearchKind.ALPHA_HYPOTHESIS,
        affected_underlyings=[UNDERLYING],
        affected_horizons=["1"],
        expected_information_gain=Estimate.unknown(
            "feature dispersion is observable but information gain is not proven"
        ),
        expected_economic_value=Estimate.unknown("not measured before the trial"),
        probability_of_resolving_uncertainty=Estimate.declared(
            0.5, "bounded screened comparison is expected to be informative, not profitable"
        ),
        implementation_cost=Estimate.unknown("no measured computational cost before execution"),
        implementation_complexity=Estimate.declared(
            0.1, "fixed reviewed DSL operator set without new code"
        ),
        estimated_sample_size=Estimate.measured(
            float(validation_days), "distinct, non-overlapping validation calendar days"
        ),
        current_uncertainty=Estimate.unknown("not inferred from feature dispersion"),
        data_availability=Estimate.measured(
            float(validation_days >= MIN_VALIDATION_DAYS),
            "PIT-valid matched sessions in later chronological partition",
        ),
        leakage_risk=Estimate.unknown(
            "historical archive cannot prove contemporary write-time availability"
        ),
        overlap_with_existing_research=Estimate.unknown(
            "ID deduplicated, but novelty is not equivalent to independent economic mechanism"
        ),
    )


def _fingerprint(sessions: list[PairDay]) -> str:
    material = [(p.day.isoformat(), p.left.raw_hash, p.right.raw_hash) for p in sessions]
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


def run_discovery_step(
    store: Store,
    *,
    now: datetime,
    snapshots: list[ProductSnapshot] | None = None,
    dry_run: bool = False,
) -> DiscoveryOutcome:
    """Generate, register and test <=1 new DSL idea; shared quarterly cap.

    No code generation, no external connections, no approval or model/portfolio
    mutations. A replay never re-tests a known idea; its immutable historical
    verdict remains available as memory, including failures.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("discovery cutoff must be timezone aware")
    now = now.astimezone(UTC)
    all_quotes = (
        snapshots
        if snapshots is not None
        else store.list_product_snapshots(underlying_id=UNDERLYING, until=now)
    )
    sessions = _candidate_sessions(all_quotes, now=now)
    development, validation = _split(sessions)
    history = store.list_research_screens()
    candidates = generate(development, history=history)
    quarter = f"{now.year}Q{(now.month - 1) // 3 + 1}"
    consumed = sum(s.quarter == quarter and s.status in EXECUTED_SCREEN_STATUSES for s in history)
    already_screened = {s.hypothesis_id for s in history if s.harness == HARNESS}
    novel = [c for c in candidates if c.rule.hypothesis_id not in already_screened]
    candidate_ids = tuple(c.rule.hypothesis_id for c in novel[:MAX_PROPOSALS_PER_CYCLE])

    def outcome(status: str, reason: str) -> DiscoveryOutcome:
        return DiscoveryOutcome(
            status=status,
            eligible_days=len(sessions),
            development_days=len(development),
            validation_days=len(validation),
            candidate_ids=candidate_ids,
            selected_hypothesis_id=None,
            screen_id=None,
            verdict=None,
            reason=reason,
        )

    if not candidates:
        return outcome(
            "AWAIT_DATA",
            "too few independent development days or no two measurable, "
            "non-redundant entry features; no speculative alpha was invented",
        )
    if not dry_run:
        for candidate in novel[:MAX_PROPOSALS_PER_CYCLE]:
            if store.get_research_opportunity(candidate.rule.hypothesis_id) is None:
                store.upsert_research_opportunity(_opportunity(candidate, len(validation)))
    if len(validation) < MIN_VALIDATION_DAYS:
        return outcome("AWAIT_DATA", "validation requires 48 independent days after embargo")
    if consumed >= EXPLORATORY_SCREENS_PER_QUARTER:
        return outcome("BUDGET_EXHAUSTED", "shared quarterly exploratory screen cap reached")

    selected = novel[0] if novel else None
    if selected is None:
        return outcome(
            "NO_NEW_HYPOTHESIS",
            "all admissible rule fingerprints already have historical experiments",
        )
    spec = selected.rule
    screen_id = f"{quarter}_AUTO_DSL_{spec.hypothesis_id.removeprefix('RO-GEN-')}"
    if dry_run:
        return DiscoveryOutcome(
            status="DRY_RUN",
            eligible_days=len(sessions),
            development_days=len(development),
            validation_days=len(validation),
            candidate_ids=candidate_ids,
            selected_hypothesis_id=spec.hypothesis_id,
            screen_id=screen_id,
            verdict=None,
            reason="no writes or tests in dry-run",
        )

    screen = ResearchScreen(
        screen_id=screen_id,
        trial_id=screen_id,
        hypothesis_id=spec.hypothesis_id,
        family=str(InformationFamily.PRODUCT_SELECTION),
        harness=HARNESS,
        harness_version=DSL_VERSION,
        spec=spec.model_dump(mode="json"),
        data_version=_fingerprint(validation),
        quarter=quarter,
        status=ScreenStatus.RUNNING,
        started_at=now,
    )
    # Write RUNNING before executing; an interrupted invocation remains visible
    # and blocks an unbudgeted automatic re-test on restart.
    store.insert_research_trial(
        ResearchTrial(
            trial_id=screen_id,
            kind=SCREEN_KIND,
            description=f"Phase 1 DSL {spec.primary.value}+{spec.secondary.value}",
            created_at=now,
            quarter=quarter,
            status=TrialStatus.EXPERIMENTAL,
        )
    )
    store.upsert_research_screen(screen)
    try:
        measurement = evaluate(spec, validation, snapshots=all_quotes, now=now)
    except Exception as exc:
        store.upsert_research_screen(
            screen.model_copy(
                update={"status": ScreenStatus.FAILED, "error": str(exc), "finished_at": now}
            )
        )
        raise
    store.upsert_research_screen(
        screen.model_copy(
            update={
                "status": ScreenStatus.COMPLETED,
                "verdict": measurement.verdict,
                "metrics": {
                    **asdict(measurement),
                    "supported": False,
                    "confirmation_allowed": False,
                    "development_days": len(development),
                    "rule_priority": selected.priority,
                    "failure_overlap": selected.overlap_with_failures,
                    "comparison": "NO_TRADE_AND_NAIVE_ISSUER",
                    "bootstrap_familywise_alpha": 0.05 / MAX_EXPLORATORY_TESTS,
                },
                "finished_at": now,
            }
        )
    )
    return DiscoveryOutcome(
        status="SCREENED_EXPLORATORY",
        eligible_days=len(sessions),
        development_days=len(development),
        validation_days=len(validation),
        candidate_ids=candidate_ids,
        selected_hypothesis_id=spec.hypothesis_id,
        screen_id=screen_id,
        verdict=measurement.verdict,
        reason=(
            "historical research only; even a candidate requires untouched "
            "confirmation and human-governed promotion"
        ),
        measurement=measurement,
    )


__all__ = [
    "Candidate",
    "DiscoveryOutcome",
    "Feature",
    "Measurement",
    "Rule",
    "evaluate",
    "generate",
    "run_discovery_step",
]
