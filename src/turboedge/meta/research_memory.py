"""Failed-hypothesis and positive-pattern memory for the research queue (§9, §10).

`research_opportunity.py` defines the schema; this module is what turns
repository state into the three measurable `ResearchOpportunity` fields
(`current_uncertainty`, `data_availability`, `overlap_with_existing_research`)
plus three additional `Estimate`s (`prior_failure_similarity`,
`family_redundancy`, `pattern_support`) that `value_of_information.
score_opportunity` takes as keyword arguments. It is the counterweight to
the catalog, which is entirely declared judgement: every quantity produced
here is either `MEASURED` from stored records or `UNKNOWN`, never a
silently invented number.

Two independent memories are involved:

* Negative memory (§9, `learning.failed_hypotheses`) -- ideas already tried
  and found wanting. A new opportunity that closely resembles one of these
  must be penalised, because otherwise the queue would keep re-proposing
  the same failed idea under a new name. `similar_failures` computes how
  closely, and `prior_failure_similarity` turns that into one penalty
  input.
* Positive memory (§10, `SuccessfulResearchPattern`) -- things that
  actually worked. `pattern_support` turns a still-supported pattern in the
  same family into a bounded bonus, discounted by measured decay so an old
  winner is not automatically re-credited forever ("keine automatische
  Ueberbewertung alter Gewinner").

## Similarity is a lexical/structural proxy, not semantic understanding

`similar_failures` has no model of meaning. It compares information-family
membership (inferred lexically, since `FailedHypothesis` carries no family
field of its own), token overlap between the failed feature's name and the
opportunity's id/description/family, and horizon overlap. Two ideas that
use different words for the same underlying question (e.g. a failed
"realised-vol regime" feature and a new "historical volatility regime"
opportunity, if the vocabulary genuinely does not overlap and the family
mapping misses the connection) can therefore score as dissimilar when a
human would recognise them as the same attempt. This is a deliberate
choice: the function is biased toward **false negatives** (missing a real
match) rather than **false positives** (penalising an unrelated idea for
sharing a common word). A missed match costs a re-run of an idea that was
already tried, which is expensive but recoverable and self-correcting (it
fails again and gets recorded again); a false-positive penalty would
silently suppress a genuinely new, valuable question, which the queue
would then never surface for a human to see. Given the choice, this
module prefers to occasionally waste a re-test over silently starving a
good idea.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from turboedge.learning.failed_hypotheses import FailedHypothesis
from turboedge.meta.research_opportunity import (
    Estimate,
    InformationFamily,
    ResearchOpportunity,
    ResearchStatus,
    SuccessfulResearchPattern,
)
from turboedge.meta.schemas import MetaDecision

# --------------------------------------------------------------------------
# Similarity weights. Family match dominates because it is the strongest
# structural signal available (an explicit, declared field on the
# opportunity); token overlap is the weakest because feature names are
# short and idiosyncratic; horizon overlap is a secondary corroborating
# signal only (most opportunities and most failed hypotheses in this
# repository share the same standard horizon set, so it must not dominate
# on its own -- see the "always returns 1" anti-triviality test).
# --------------------------------------------------------------------------
_FAMILY_WEIGHT = 0.50
_TOKEN_WEIGHT = 0.35
_HORIZON_WEIGHT = 0.15

#: A documented §9 exception (regime change, new data source, new
#: methodology) does not make the resemblance disappear -- it makes the
#: resemblance less disqualifying. The similarity itself is left untouched
#: (it is still an honest structural measurement); the discount is applied
#: only when turning similarities into a single penalty input, so the
#: "reason a human accepted" is visible on the `FailureSimilarity` and the
#: reduced weight is visible in `prior_failure_similarity`'s note.
_RETEST_ALLOWED_DISCOUNT = 0.4

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "to",
        "in",
        "on",
        "for",
        "from",
        "with",
        "by",
        "is",
        "are",
        "as",
        "this",
        "that",
        "it",
    }
)

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Lower-case, normalise separators, drop stopwords and empties.

    Deliberately no stemming/lemmatisation and no fuzzy matching: the
    contract forbids new dependencies, and a simple, deterministic
    tokenizer is easier to audit than a clever one.
    """
    normalised = text.lower().replace("_", " ").replace("-", " ")
    return {tok for tok in _TOKEN_SPLIT_RE.split(normalised) if tok and tok not in _STOPWORDS}


#: Lexical proxy for "which information family does this word belong to".
#: This is a DECLARED judgement embedded in code, used only to infer a
#: failed hypothesis's family (which is not a field it carries) well
#: enough to compare it against an opportunity's *actual*, declared
#: `information_family`. It is intentionally small and literal -- no
#: attempt is made to be exhaustive, per the false-negative bias explained
#: in the module docstring.
_FAMILY_KEYWORDS: dict[InformationFamily, frozenset[str]] = {
    InformationFamily.VOLATILITY_SURFACE: frozenset(
        {
            "vix",
            "volatility",
            "vol",
            "iv",
            "implied",
            "skew",
            "surface",
            "term",
            "structure",
            "cboe",
            "garch",
        }
    ),
    InformationFamily.POSITIONING: frozenset(
        {"positioning", "cot", "commitment", "traders", "futures", "openinterest", "flow", "cftc"}
    ),
    InformationFamily.SENTIMENT: frozenset(
        {"sentiment", "news", "social", "survey", "putcall", "put", "call", "ratio"}
    ),
    InformationFamily.RATES_CREDIT: frozenset(
        {"rates", "rate", "credit", "yield", "spread", "curve", "bond", "cds", "estr"}
    ),
    InformationFamily.BREADTH_DISPERSION: frozenset(
        {"breadth", "dispersion", "advance", "decline", "participation"}
    ),
    InformationFamily.EVENT_RISK: frozenset(
        {"event", "earnings", "macro", "fomc", "cpi", "announcement", "calendar"}
    ),
    InformationFamily.PATH_STATISTICS: frozenset(
        {"path", "gap", "overnight", "weekend", "jump", "barrier", "ko", "knockout"}
    ),
    InformationFamily.CROSS_ASSET: frozenset(
        {"cross", "asset", "leadlag", "lead", "lag", "correlation", "intermarket"}
    ),
    InformationFamily.MICROSTRUCTURE: frozenset(
        {"microstructure", "liquidity", "orderbook", "bidask", "tick", "spread"}
    ),
    InformationFamily.PRODUCT_SELECTION: frozenset(
        {"product", "selection", "issuer", "wkn", "turbo", "universe"}
    ),
}

#: Which source this module expects to answer for a given family, so that
#: `measure_data_availability` can look it up in the caller's source-health
#: mapping. DECLARED judgement, not a measurement -- several families have
#: no integrated adapter yet, which is exactly why those entries correctly
#: come back UNKNOWN rather than guessing at a name that will never be in
#: the mapping. Real adapter source names (`adapters/*.py::_SOURCE_NAME`)
#: are used where one genuinely exists.
_FAMILY_PRIMARY_SOURCE: dict[InformationFamily, str] = {
    InformationFamily.VOLATILITY_SURFACE: "cboe",
    InformationFamily.POSITIONING: "cftc",
    InformationFamily.RATES_CREDIT: "ecb_estr",
    InformationFamily.CROSS_ASSET: "yfinance",
    InformationFamily.PATH_STATISTICS: "gettex",
    InformationFamily.MICROSTRUCTURE: "bnp_paribas",
    InformationFamily.PRODUCT_SELECTION: "gettex",
    InformationFamily.SENTIMENT: "sentiment_survey",
    InformationFamily.BREADTH_DISPERSION: "breadth",
    InformationFamily.EVENT_RISK: "macro_calendar",
}

#: Which `DecisionConfidence` axis (or axes) the research in a given family
#: would actually reduce, per Phase 1's uncertainty decomposition
#: (`meta/schemas.py::DecisionConfidence`). DECLARED judgement -- this is a
#: mapping choice, not a measurement, and is documented as such rather than
#: presented as derived fact.
_FAMILY_TO_UNCERTAINTY_AXES: dict[InformationFamily, tuple[str, ...]] = {
    InformationFamily.VOLATILITY_SURFACE: ("epistemic_uncertainty", "regime_uncertainty"),
    InformationFamily.POSITIONING: ("epistemic_uncertainty", "regime_uncertainty"),
    InformationFamily.SENTIMENT: ("epistemic_uncertainty",),
    InformationFamily.RATES_CREDIT: ("regime_uncertainty",),
    InformationFamily.BREADTH_DISPERSION: ("model_disagreement",),
    InformationFamily.EVENT_RISK: ("regime_uncertainty",),
    InformationFamily.PATH_STATISTICS: ("calibration_uncertainty",),
    InformationFamily.CROSS_ASSET: ("model_disagreement",),
    InformationFamily.MICROSTRUCTURE: ("product_data_uncertainty",),
    InformationFamily.PRODUCT_SELECTION: ("product_data_uncertainty",),
}

_HORIZON_RE = re.compile(r"^(\d+)d$")


def _parse_horizon_days(horizon: str) -> int | None:
    """``"5d" -> 5``. Anything else (unparseable horizon strings) is
    treated as not matchable, never as 0 or a guessed value."""
    m = _HORIZON_RE.match(horizon.strip().lower())
    return int(m.group(1)) if m else None


class FailureSimilarity(BaseModel):
    """How closely one failed hypothesis resembles one opportunity.

    `similarity` is the raw structural/lexical measurement -- it is left
    untouched by `retest_allowed`, because the resemblance itself did not
    change. Only the *penalty* derived from it (in
    `prior_failure_similarity`) is discounted when a human has documented
    a §9 exception.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    similarity: float = Field(ge=0.0, le=1.0)
    same_family: bool
    retest_allowed: bool
    reason: str


def _failure_similarity(
    opportunity: ResearchOpportunity, failed: FailedHypothesis
) -> FailureSimilarity:
    failed_tokens = _tokens(failed.feature)
    opp_tokens = (
        _tokens(opportunity.hypothesis_id)
        | _tokens(opportunity.description)
        | _tokens(opportunity.information_family.value)
    )

    family_keywords = _FAMILY_KEYWORDS.get(opportunity.information_family, frozenset())
    same_family = bool(failed_tokens & family_keywords)

    if failed_tokens and opp_tokens:
        token_overlap = len(failed_tokens & opp_tokens) / min(len(failed_tokens), len(opp_tokens))
    else:
        token_overlap = 0.0

    failed_horizons = set(failed.horizons)
    opp_horizons = set(opportunity.affected_horizons)
    if failed_horizons and opp_horizons:
        horizon_overlap = len(failed_horizons & opp_horizons) / min(
            len(failed_horizons), len(opp_horizons)
        )
    else:
        horizon_overlap = 0.0

    # Horizon overlap is a modifier on substantive evidence, never evidence on
    # its own. Two questions sharing "3d..14d" have the same horizons as
    # practically everything else in the catalog, so counting it unconditionally
    # gave every unrelated pair a fixed 0.15 similarity -- and, worse, penalised
    # a broad question more than a narrow one purely for covering more horizons,
    # which is the opposite of what breadth should cost. It only counts once the
    # family or the vocabulary already says the two are related.
    substantive = _FAMILY_WEIGHT * (1.0 if same_family else 0.0) + _TOKEN_WEIGHT * token_overlap
    similarity = substantive + (_HORIZON_WEIGHT * horizon_overlap if substantive > 0.0 else 0.0)
    similarity = max(0.0, min(1.0, similarity))

    retest_allowed = failed.status == "retest_allowed"

    parts: list[str] = []
    if same_family:
        parts.append(
            f"lexical family match: '{failed.feature}' shares a "
            f"{opportunity.information_family.value} keyword"
        )
    if token_overlap > 0:
        parts.append(f"token overlap {token_overlap:.2f}")
    if horizon_overlap > 0:
        parts.append(f"horizon overlap {horizon_overlap:.2f}")
    if not parts:
        parts.append("no family, token, or horizon overlap detected")
    if retest_allowed:
        parts.append(
            "retest_allowed: a human documented a §9 exception "
            f"({failed.regime_change_note!r}) -- penalty should be reduced, "
            "not the similarity itself"
        )
    reason = "; ".join(parts)

    return FailureSimilarity(
        feature=failed.feature,
        similarity=similarity,
        same_family=same_family,
        retest_allowed=retest_allowed,
        reason=reason,
    )


def similar_failures(
    opportunity: ResearchOpportunity,
    failed: Sequence[FailedHypothesis],
) -> list[FailureSimilarity]:
    """One `FailureSimilarity` per entry in ``failed``, most-similar first.

    Every entry is returned, not only "matches" above some threshold: a
    caller may want the full ranked picture, and thresholding here would
    hide the (informative) fact that nothing resembles the opportunity.
    """
    scored = [_failure_similarity(opportunity, f) for f in failed]
    return sorted(scored, key=lambda s: s.similarity, reverse=True)


def prior_failure_similarity(
    opportunity: ResearchOpportunity,
    failed: Sequence[FailedHypothesis],
) -> Estimate:
    """The single strongest failure-resemblance penalty input (§9).

    Takes the MAX across matches, not the average: one hypothesis that is
    basically a repeat of a known failure is the problem, and averaging it
    against several unrelated entries would dilute exactly the signal this
    function exists to surface.

    Returns ``Estimate.unknown(...)`` when ``failed`` is empty. This
    module only ever receives an already-loaded sequence (see
    `learning.failed_hypotheses.load`, which itself returns ``[]`` both
    when the registry file is genuinely absent -- a valid "nothing failed
    yet" state -- and would raise, not return silently, on a malformed
    file). Since both "nothing has failed yet" and "the registry could not
    be read" are indistinguishable once they reach this function as an
    empty sequence, this deliberately does not claim a confident
    ``measured(0.0)`` for that case; see the empty-input test.
    """
    if not failed:
        return Estimate.unknown(
            "no failed-hypothesis registry entries were supplied; an empty "
            "sequence here is indistinguishable from a registry that could "
            "not be read, so no measurement is claimed"
        )

    similarities = similar_failures(opportunity, failed)

    def _effective(s: FailureSimilarity) -> float:
        return s.similarity * (_RETEST_ALLOWED_DISCOUNT if s.retest_allowed else 1.0)

    best = max(similarities, key=_effective)
    effective = _effective(best)

    if effective <= 0.0:
        return Estimate.measured(
            0.0,
            note=(
                f"failed-hypothesis registry has {len(failed)} entries but none "
                "resemble this opportunity"
            ),
        )

    discount_note = " (retest_allowed discount applied)" if best.retest_allowed else ""
    return Estimate.measured(
        effective,
        note=(
            f"closest failed hypothesis: '{best.feature}' "
            f"(raw similarity {best.similarity:.2f}{discount_note})"
        ),
    )


def family_redundancy(
    opportunity: ResearchOpportunity,
    others: Sequence[ResearchOpportunity],
) -> Estimate:
    """Share of OTHER, non-REJECTED opportunities in the same family (§7).

    A family already crowded with queued work is worth less at the
    margin -- this is the computable proxy for that. Excludes the
    opportunity itself by `hypothesis_id` so an opportunity never competes
    with its own stored copy.
    """
    peers = [o for o in others if o.hypothesis_id != opportunity.hypothesis_id]
    if not peers:
        return Estimate.measured(
            0.0,
            note="no other opportunities in the catalog; this family cannot be crowded",
        )
    same_family_active = [
        o
        for o in peers
        if o.information_family == opportunity.information_family
        and o.status != ResearchStatus.REJECTED
    ]
    share = len(same_family_active) / len(peers)
    return Estimate.measured(
        share,
        note=(
            f"{len(same_family_active)}/{len(peers)} other non-REJECTED opportunities "
            f"share family {opportunity.information_family.value}"
        ),
    )


def pattern_support(
    opportunity: ResearchOpportunity,
    patterns: Sequence[SuccessfulResearchPattern],
    *,
    now: datetime,
) -> Estimate:
    """Bounded support from confirmed, non-decayed patterns in family (§10).

    Two "never invent a value" rules apply here, both required by the
    contract:

    * ``discovered_at > now`` is excluded (no look-ahead: a pattern from
      the future cannot inform a decision made at ``now``).
    * ``decay_since_discovery is None`` means decay has never been
      re-measured -- that pattern contributes NO support. Treating
      ``None`` as "no decay yet, so full strength" would automatically
      overweight old winners precisely the way the user's rule ("keine
      automatische Ueberbewertung alter Gewinner") forbids.

    Takes the MAX of ``stability * (1 - decay)`` across the remaining,
    decay-confirmed patterns -- not the sum -- so that several patterns in
    one family cannot manufacture more confidence than the single best one
    actually earned.
    """
    if not patterns:
        return Estimate.unknown(
            "no successful research patterns have been recorded in the repository yet"
        )

    family = opportunity.information_family
    matches = [p for p in patterns if p.information_family == family and p.discovered_at <= now]
    if not matches:
        return Estimate.measured(
            0.0,
            note=(
                f"no successful patterns recorded (as of {now.isoformat()}) "
                f"in family {family.value}"
            ),
        )

    contributions = []
    for p in matches:
        if p.decay_since_discovery is None:
            continue
        decay = max(0.0, min(1.0, p.decay_since_discovery))
        contributions.append(p.stability * (1.0 - decay))

    if not contributions:
        return Estimate.measured(
            0.0,
            note=(
                f"{len(matches)} pattern(s) in family {family.value} but none have a "
                "measured decay_since_discovery yet; an unconfirmed pattern contributes "
                "no support"
            ),
        )

    best = max(contributions)
    return Estimate.measured(
        best,
        note=(
            f"strongest decay-confirmed pattern in family {family.value}: "
            f"stability x (1 - decay) = {best:.2f} across {len(contributions)}/{len(matches)} "
            "decay-measured pattern(s)"
        ),
    )


def measure_current_uncertainty(
    opportunity: ResearchOpportunity,
    decisions: Sequence[MetaDecision],
) -> Estimate:
    """How uncertain the (Phase 1) meta layer currently is on this question.

    Maps `opportunity.information_family` onto the `DecisionConfidence`
    axis it would actually reduce -- see `_FAMILY_TO_UNCERTAINTY_AXES` --
    which is a DECLARED mapping choice, not a measurement, and is
    documented as such. Restricted to stored `MetaDecision`s matching the
    opportunity's underlyings/horizons where any exist; falls back to the
    full set (with a note saying so) when no decision matches, rather than
    claiming unknown over an unrestricted but real signal.

    Returns ``Estimate.unknown(...)`` when no `MetaDecision`s are stored at
    all -- this must never be defaulted to a neutral value (CLAUDE.md rule
    29 / the meta layer's own `trust.missing_factors` convention).
    """
    if not decisions:
        return Estimate.unknown(
            "no MetaDecision records are stored; current uncertainty on this "
            "question cannot be measured"
        )

    axes = _FAMILY_TO_UNCERTAINTY_AXES.get(opportunity.information_family)
    if axes is None:
        return Estimate.unknown(
            f"no uncertainty-axis mapping declared for family "
            f"{opportunity.information_family.value}"
        )

    underlyings = set(opportunity.affected_underlyings)
    horizons_days = {
        d for d in (_parse_horizon_days(h) for h in opportunity.affected_horizons) if d is not None
    }

    def _matches(d: MetaDecision) -> bool:
        if underlyings and d.underlying_id not in underlyings:
            return False
        return not (horizons_days and d.horizon_days not in horizons_days)

    filtered = [d for d in decisions if _matches(d)]
    restricted = bool(filtered)
    pool = filtered if restricted else decisions

    values = [sum(getattr(d.confidence, ax) for ax in axes) / len(axes) for d in pool]
    mean_value = sum(values) / len(values)

    note = f"averaged axes {axes} over {len(pool)} MetaDecision(s)"
    note += (
        " restricted to matching underlyings/horizons"
        if restricted
        else " (no decision matched the opportunity's underlyings/horizons; "
        "using all stored decisions as a fallback)"
    )
    return Estimate.measured(mean_value, note=note)


def measure_data_availability(
    opportunity: ResearchOpportunity,
    *,
    source_available: Mapping[str, bool],
) -> Estimate:
    """Whether the source this family would need is reachable.

    Pure dict lookup against the caller-supplied source-health mapping --
    no network calls. `_FAMILY_PRIMARY_SOURCE` names the source each
    family is expected to draw on; a family whose source is not (yet) in
    ``source_available`` -- because no adapter exists, or health simply
    was not checked this run -- correctly comes back UNKNOWN rather than
    guessing.
    """
    source = _FAMILY_PRIMARY_SOURCE.get(opportunity.information_family)
    if source is None or source not in source_available:
        return Estimate.unknown(
            f"no reachability recorded for the source family "
            f"{opportunity.information_family.value} depends on "
            f"(expected source {source!r})"
        )
    reachable = source_available[source]
    return Estimate.measured(
        1.0 if reachable else 0.0, note=f"source '{source}' reachable={reachable}"
    )


def enrich(
    opportunity: ResearchOpportunity,
    *,
    failed: Sequence[FailedHypothesis],
    others: Sequence[ResearchOpportunity],
    patterns: Sequence[SuccessfulResearchPattern],
    decisions: Sequence[MetaDecision],
    source_available: Mapping[str, bool],
    now: datetime,
) -> tuple[ResearchOpportunity, dict[str, Estimate]]:
    """Replace the measurable fields of ``opportunity`` with measurements.

    Returns a COPY (`model_copy(update=...)`) -- the input is never
    mutated -- with `current_uncertainty`, `data_availability` and
    `overlap_with_existing_research` replaced by MEASURED estimates where
    they can be computed, and left UNKNOWN where they cannot (each
    delegate function makes that call on its own evidence; this function
    does not override it either way).

    The returned dict carries the three additional estimates
    `value_of_information.score_opportunity` needs beyond what the
    opportunity schema itself holds.
    """
    redundancy = family_redundancy(opportunity, others)
    current_uncertainty = measure_current_uncertainty(opportunity, decisions)
    data_availability = measure_data_availability(opportunity, source_available=source_available)

    extra = {
        "prior_failure_similarity": prior_failure_similarity(opportunity, failed),
        "family_redundancy": redundancy,
        "pattern_support": pattern_support(opportunity, patterns, now=now),
    }

    updated = opportunity.model_copy(
        update={
            "current_uncertainty": current_uncertainty,
            "data_availability": data_availability,
            "overlap_with_existing_research": redundancy,
        }
    )
    return updated, extra


def record_successful_pattern(
    pattern_id: str,
    *,
    information_family: InformationFamily,
    feature: str,
    underlying_id: str,
    horizon: str,
    volatility_regime: str,
    trend_regime: str,
    oos_effect: float,
    effective_sample: int,
    stability: float,
    economic_value: float,
    discovered_at: datetime,
    trial_id: str | None = None,
    note: str = "",
) -> SuccessfulResearchPattern:
    """Convenience constructor for a new positive-memory entry (§10).

    Mirrors `learning.failed_hypotheses.new_hypothesis`, but for the
    opposite result: `last_confirmed_at` and `decay_since_discovery` are
    always left `None` at creation, because decay is -- by definition --
    unmeasured until a later re-measurement exists. Scoring
    (`pattern_support`) must and does treat that `None` as "no support
    yet", never as "no decay".
    """
    return SuccessfulResearchPattern(
        pattern_id=pattern_id,
        information_family=information_family,
        feature=feature,
        underlying_id=underlying_id,
        horizon=horizon,
        volatility_regime=volatility_regime,
        trend_regime=trend_regime,
        oos_effect=oos_effect,
        effective_sample=effective_sample,
        stability=stability,
        economic_value=economic_value,
        discovered_at=discovered_at,
        last_confirmed_at=None,
        decay_since_discovery=None,
        trial_id=trial_id,
        note=note,
    )


__all__ = [
    "FailureSimilarity",
    "enrich",
    "family_redundancy",
    "measure_current_uncertainty",
    "measure_data_availability",
    "pattern_support",
    "prior_failure_similarity",
    "record_successful_pattern",
    "similar_failures",
]
