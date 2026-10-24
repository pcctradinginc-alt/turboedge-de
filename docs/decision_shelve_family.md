# Decision required: classifying five historical failed hypotheses

**Status: ANALYSIS AND DECISION PROPOSAL. No historical failure was
reclassified, no threshold lowered, and no family shelved.**

---

## 1. The measured state

`state/registry/failed_hypotheses.json` holds seven entries, all `dormant`.
`meta/research_memory.family_for_feature` maps two of them:

| feature | mapped family |
|---|---|
| `cross_asset_leadlag` | `cross_asset` |
| `vix_term_structure` | `volatility_surface` |
| `lowvol_regime_trend` | — |
| `reversal_short_horizon` | — |
| `seasonality_turn_of_month` | — |
| `voltarget_tsmom` | — |
| `regime_conditional_economic_value` | — |

`SHELVE_FAMILY` requires three failures in one family. The highest count is
one, in two different families, so it is currently unreachable. The
production research run of 2026-10-24 12:54 logged
`unattributable_failures: 5`, confirming this in operation.

## 2. Technical mapping defect, or missing scientific decision?

Both, and they must be separated.

**Technical.** `family_for_feature` matches feature-name tokens against each
family's keyword list. It is a deliberately conservative heuristic:
unmatched returns `None`, which counts toward nothing, which leaves research
open rather than shelving a family on a guess. The five unmatched names fail
for a plain lexical reason — `lowvol`, `reversal`, `voltarget`,
`seasonality` are not tokens in any keyword list. Extending the lists would
map them. That is a one-line change and it is **not** what should happen,
for the reason in §4.

**Scientific.** Four of the five have a documented, pre-result definition in
`src/turboedge/models/challengers.py`, each carrying its own
`signal_family` attribute fixed before any result was seen:

| feature | pre-result definition (challengers.py) | derived from |
|---|---|---|
| `voltarget_tsmom` | time-series momentum, volatility-targeted | the underlying's own return path |
| `lowvol_regime_trend` | trend conditioned on a realised-volatility regime tercile | the underlying's own return path |
| `reversal_short_horizon` | 3-day return reversal | the underlying's own return path |
| `seasonality_turn_of_month` | calendar turn-of-month effect | the underlying's own return path + the calendar |

All four are directional return forecasts computed from the underlying's own
price history and nothing else. On the existing taxonomy
(`meta/research_opportunity.InformationFamily`) the only family that
describes that is `PATH_STATISTICS`. This is determinable from the
pre-registered definitions and does **not** require knowing their results.

The fifth, `regime_conditional_economic_value`, is not a feature family at
all: it is a meta-question about whether economic value is regime-dependent.
No `InformationFamily` describes it, and forcing one would be arbitrary.

## 3. What that classification would cause

This is why it is a decision and not a cleanup.

`PATH_STATISTICS` contains **exactly two** catalogue entries:
`RO-MAE-PREDICTION` and `RO-MFE-PREDICTION` — and those are the **only two
of fourteen** that have an approved harness (`research/harness.py`).

Classifying the four as `path_statistics` gives that family four failures,
crosses the threshold of three, and shelves it. The consequence is that the
autonomous loop would have **nothing left it can execute at all**: every
other family reports `NO_APPROVED_HARNESS`, and the one family it can
screen would be closed.

And the conflation is substantive, not just inconvenient. The four failures
are about **predicting the direction of returns** from the price path. The
two catalogue entries are about **predicting the magnitude of a position's
excursion** (maximum adverse / favourable) from entry-time features, for
position sizing and exit rules. Same raw data source, different question,
different decision it informs. That four directional forecasts failed is
weak evidence at best about whether excursion magnitude is predictable.

The real finding is therefore about the rule, not the five entries:
**`InformationFamily` is too coarse a grain for the shelving rule.** It was
designed to group research *opportunities* by information source; shelving
needs to group *hypotheses by the question they answer*. Using one for the
other makes a family of two unrelated questions get closed by evidence
about a third.

## 4. Options

**Option A — leave the five unmapped.** Status quo. `SHELVE_FAMILY` stays
unreachable. Honest, costs nothing today, and the negative memory still
gates retests through `failed_hypotheses.allow_retest`, which is a separate
and working mechanism. Cost: the shelving rule remains decorative, so the
loop cannot stop spending screens on a family that keeps failing — which it
has not yet had the chance to do, since no screen has completed.

**Option B — classify the four as `path_statistics`.** Defensible on
pre-result grounds. Immediately shelves the only executable family and
halts exploratory research entirely. Not recommended as a *consequence* of a
bookkeeping fix.

**Option C — split the taxonomy before classifying.** Separate directional
return forecasting from excursion/path-magnitude prediction, as two
families. The four failures then land in the directional family, which
contains no catalogue entry at all, so nothing is shelved that the evidence
does not speak to — and the shelving rule becomes meaningful rather than
unreachable. Cost: a taxonomy change touches `InformationFamily`, the
catalogue's family assignments, `PERMITTED_EXPLORATORY_FAMILIES` and the
keyword lists. It is a real change and it is a scientific judgement about
what counts as one line of inquiry.

**Option D — classify the four and raise the threshold.** Explicitly
excluded: the brief forbids lowering the threshold to make `SHELVE_FAMILY`
fire, and raising it to stop it firing is the same move in reverse. The
threshold should not be tuned to produce a desired status.

## 5. Recommendation

**Option C, and Option A until C is decided.**

Leave the five unmapped for now — that is the state that cannot cause
damage — and treat the taxonomy split as the actual question. The five
entries are not the problem; the rule's grain is. Classifying them under the
current taxonomy would produce a technically defensible assignment whose
effect is to shut down the only research the system can do, on the strength
of evidence about a different question.

`regime_conditional_economic_value` should stay unmapped under every option.
It is not a feature family, and the honest record is that it has none.

## 6. Process for future hypotheses

Independent of the five, and implementable without a taxonomy decision:

1. A hypothesis records its `InformationFamily` **when it is registered**,
   before any result exists — the way `AlphaSource.family` already works
   and `FailedHypothesis` does not.
2. `FailedHypothesis` gains an explicit `family` field, so classification is
   a recorded decision rather than a token-matching guess at failure time.
   `family_for_feature` stays as the fallback for the historical entries.
3. An unclassifiable hypothesis is recorded as unclassified and counts
   toward nothing. An absent classification must never shelve a family.
4. The monthly report states, per family, the failure count and whether it
   is at the shelving threshold — so the state is visible before it acts
   rather than discovered when something stops.

Step 2 changes a persisted schema and the number of recorded failures must
not change in the process; §6 of the brief requires the original failure
history and the count of tested hypotheses to stay exactly as they are.

## 7. Exactly what is needed

- A decision between options A, B and C above.
- If C: a ruling on where the boundary falls between directional
  forecasting and path-magnitude prediction, and whether
  `RO-MAE-PREDICTION`/`RO-MFE-PREDICTION` move with it.
- A ruling that `regime_conditional_economic_value` stays unclassified.
- Approval for the process in §6, which can proceed independently.

---

*Prepared 2026-10-24. Seven failed hypotheses, unchanged; no family
shelved.*
