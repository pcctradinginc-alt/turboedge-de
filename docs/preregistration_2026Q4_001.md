# Pre-registration: 2026Q4-001 — does a better return distribution carry economic value?

**Written 2026-09-26. Frozen on commit. No measurement had been run against
this specification when it was written, and none may be run before
2026-10-01.**

This is the first trial written under GOVERNANCE.md §11.2's one-primary-
hypothesis rule. It exists because `docs/measured_results.md` §6.11-§6.13
produced *robust retrospective evidence* for one forecast model and nothing
confirmatory: the moving-block bootstrap that made `regime_conditional`
survive deflation was chosen after its positive result was already known. The
number that matters has not been produced. This document fixes the method
before it is.

---

## 1. Primary hypothesis — exactly one

Let `NetEV(m)` be the aggregate simulated net expected return of a fixed,
standardised turbo universe, priced through the existing payoff / knock-out /
cost machine, using forecast model `m`.

    H0:  NetEV(regime_conditional) - NetEV(null)  <=  0
    H1:  NetEV(regime_conditional) - NetEV(null)  >   0

**One primary p-value.** One-sided, α = 0.10, moving-block bootstrap over
prediction dates, block length 21 trading days (≥ the longest horizon, as in
§6.12).

Everything else in §6 below is stability analysis. It is reported and it may
not be deflated as an independent primary test, and it may not be promoted to
primary after the fact.

## 2. Why this and not the other candidate

`regularized_linear_location` also survives BH at the model level (p = 0.0142
vs 0.0135). Only `regime_conditional` is pre-registered here, because
registering both would make this two primary hypotheses and halve the BH
threshold for each. If this test passes, the second model is a separate
trial against a separate budget unit.

## 3. What this test is NOT

This is the **historical synthetic** arm of the two-part design in §6.13. It
uses standardised turbos with stated terms and makes **no claim that any of
these products existed or were quotable at any historical date.** Master Spec
rule 18 forbids back-historizing current products, and the reason is
substantive: underlying history does not record which turbos an issuer
offered, what it quoted, or what its financing level was.

It therefore answers *"does the better distribution carry economic information
at all?"* and cannot answer *"is this tradeable?"* The tradeability question
is the forward real-product arm, which needs forward data accumulating since
2026-09-13 and is not part of this trial.

A pass here is a necessary, not sufficient, condition for promotion.

## 4. Frozen method

**Models.** `NullModel` and `RegimeConditionalEmpiricalModel` exactly as they
exist at the commit that adds this file. No configuration tuning, no variant
selection. If either class changes before the run, the trial is void and must
be re-registered.

**Data.** `adapters/fallback_prices.py::YFinancePriceAdapter`,
`lookback_days=4000`, underlyings DAX, NDX, EURUSD, XAU. The fetch date, the
per-underlying bar count after the OHLC-consistency filter, and the adapter's
skip summary are recorded with the result (the omission that made §6
irreproducible).

**Walk-forward.** `PurgedWalkForwardSplit`, `min_train=750`, `step=21`,
`embargo=horizon`, horizons 3/5/7/10/14 — identical to §6.12 so the forecast
side is not re-tuned by this trial.

**Standardised turbo universe.** At each prediction date, for each underlying
and horizon, a fixed grid constructed from that date's spot:

- barrier distances: 2%, 5%, 10%, 15%, 20% below spot (long) and above (short)
- `financing_level` = barrier (open-end convention, as `ProductTerms` uses)
- `ratio` = 0.01, `fx` = 1.0
- `entry_ask` = theoretical fair value; `entry_bid` = `entry_ask * (1 - spread)`
- `spread` = 0.005, `financing_spread` = 0.02, `ref_rate` = 0.02,
  `premium_over_fair` = 0.0, `exit_spread_pct` = 0.005

**These terms are identical across both arms at every date.** The only
difference between arms is the forecast handed to
`ranking/ev.py::evaluate_product_horizons`. Any result therefore attributes to
the forecast and to nothing else. The absolute level of `NetEV` is an artefact
of the cost assumptions above and is not interpretable on its own; only the
difference between arms is.

**Aggregation.** Per prediction date, the mean net expected return across the
whole grid, per arm. The primary statistic is the mean over dates of
(regime_conditional − null).

**Paths.** 2,000 per evaluation, seed 20261001, identical across arms.

## 5. Decision rule — stated before the result

| outcome | reading |
|---|---|
| p ≤ 0.10 **and** mean ΔNetEV > 0 | **PASS.** The distribution improvement carries economic information on standardised terms. Necessary condition met; forward real-product arm becomes the next trial. Still no promotion. |
| p > 0.10 | **FAIL.** Recorded in `failed_hypotheses.json`. The CRPS improvement is statistically interesting and economically inert on these terms. |
| mean ΔNetEV ≤ 0 with p ≤ 0.10 | **FAIL**, and reported as evidence the better distribution is actively *worse* economically — a more informative negative than a null result. |

**No threshold, gate or cost assumption may be adjusted after seeing the
result.** If the cost assumptions in §4 turn out to dominate the answer, that
is itself the finding, and changing them constitutes a new trial.

## 6. Stability analysis (secondary, never primary)

Reported with the result, deflated against nothing, promoted to primary never:
per-underlying ΔNetEV; per-horizon ΔNetEV; per-barrier-distance ΔNetEV; the
share of (date, product, horizon) cells with ΔNetEV > 0; sensitivity of the
sign of the result to `spread` at 0.0025 and 0.01.

The last of these is included deliberately: §6.10 measured that the EV stage
amplifies a 0.23pp forecast difference into 365 gate crossings, so a result
that flips sign under a halved spread assumption is not a robust result and
the reader must be able to see that without re-running anything.

## 7. Budget and denominator

Charged to **2026Q4**, trial id assigned on the run date (2026-10-01 or
later), not now — §11.1 records what charging a quarter's work to a different
quarter's budget did to this repository's books twice already.

**2026Q4 primary hypotheses pre-registered so far: 1 (this one).** The BH
denominator for this trial's primary p-value is the count of Q4 primary
hypotheses at the time of correction, not a cell count. Per §11.2, per-cell
results are stability analysis.

## 8. Pre-committed statement of ignorance

Written before the run: the honest prior is FAIL. Every economic test in this
repository has failed — 0/20 forecast cells, 0/80 challenger cells, Cboe,
CFTC. A 1.4% CRPS improvement is small against a spread of 50bp plus financing
and knock-out asymmetry. If this passes, the first question to ask is whether
the cost assumptions in §4 are too generous, not whether an edge has been
found.

---

## 9. Amendment A (2026-09-26, before any measurement)

Recorded as an amendment rather than an edit, because a frozen document that
is quietly revised is not frozen. **No measurement of the primary hypothesis
had been run when this was written.**

**`financing_level = barrier` is a simplification, and it is the conservative
one.** Measured against the 2,672 real ledger entries: the two are exactly
equal in 2,251 of them (84%), with a mean relative gap of 0.25% in the rest.
Real turbos commonly carry a small stop-loss buffer between the financing
level and the barrier, so a knock-out leaves a residual payout.

Setting them equal removes that buffer, which means a knock-out pays zero.
That is *worse* than a real turbo, so the simplification biases against
finding an edge rather than toward it. It stays as specified in §4.

It is recorded here because §3 already states this universe makes no claim
these products existed — and a reader should be able to see exactly which way
each idealisation cuts, rather than having to trust that they average out.

**`theoretical_fair_value` lives in `pricing/fair_value.py`, not
`pricing/intrinsic.py`** as §4 implies. No behavioural change; the named
function is the one intended.

---

## 10. Amendment B (2026-09-26, before any measurement) — §1 as written cannot be tested

Found while building the harness, before any measurement of the primary
hypothesis. Verified directly in the code, not taken on report.

### The finding

**No statistic the current EV pipeline produces is driven by the
distributional width that CRPS measures.**

- `ranking/ev.py::_drift_for_scenario` maps a forecast to a scenario drift
  using `forecast.mean` (central) and `forecast.uncertainty` (pessimistic /
  optimistic). `forecast.sigma` is passed only into a diagnostic record.
- `simulation/paths.py::simulate_paths` takes `drift_log_return` and has **no
  volatility parameter at all**. Path dispersion is derived entirely from
  `bars`, which are identical across both arms of this trial.

So the paths in both arms have the same dispersion, the same gap structure and
the same knock-out geometry; the only difference is a shifted mean drift. The
trial as specified therefore tests *"does `regime_conditional`'s **mean**
forecast differ from `NullModel`'s enough to move NetEV"* — not *"does the
improved distribution carry economic value"*, which is what §1 claims.

This matters because §6 of `docs/measured_results.md` attributes the CRPS
improvement primarily to the models widening and narrowing their intervals
with current volatility. If the gain lives in the width, this trial cannot see
it, and would return a null result for a reason that has nothing to do with
the hypothesis.

### Consequence for §1

**§1's primary hypothesis is withdrawn and replaced.** The new primary
statistic is `lcb_net_return` — the lower-confidence-bound net return — rather
than `mean_net_return`:

    H0:  LCB_NetEV(regime_conditional) - LCB_NetEV(null)  <=  0

Reasons, stated before the result:

1. It is the quantity the ACTIONABLE gate actually uses. Whatever else is
   true, this is the number that decides whether a trade happens.
2. It reads `forecast.uncertainty` as well as `forecast.mean`, so it feels
   more of the forecast than the mean alone.
3. It is honest about scope: `uncertainty` is the standard error of the mean
   estimate, **not** the predictive width CRPS scores. This test still cannot
   answer the width question.

**The width question requires extending `simulate_paths` to accept a forecast
volatility.** That is a change to the production path engine affecting every
existing measurement, so it is a separate, later trial and not a patch to this
one. Recorded as the open question it is.

Everything else in §4-§8 is unchanged, including the decision rule, the
stability analysis and the pre-committed FAIL prior.

### Entry-side cost: §4 was too generous

Second finding, same origin. `simulation/payoff.py` computes every net return
from `terms.entry_ask` and **never reads `terms.entry_bid`**. §4 set
`entry_ask` = theoretical fair value, so the synthetic universe bought every
product at fair value with **zero entry-side friction**, and no value of
`spread` could change any result — making §6's spread-sensitivity check a
provable no-op.

That cuts the opposite way from Amendment A's barrier simplification: it made
the test more generous, not less. Corrected, before any measurement:

    entry_ask = fair_value * (1 + spread / 2)
    entry_bid = fair_value * (1 - spread / 2)

which is how a real two-sided quote straddles fair value. `spread` now
genuinely affects NetEV and §6's sensitivity check becomes meaningful.

### Why this is an amendment and not a fresh document

No measurement of the primary hypothesis has been run. Both findings are about
what the machinery can and cannot feel, discovered by reading it rather than by
looking at a result. Fixing a test before it runs is the entire purpose of
writing the specification first; the failure mode this document guards against
is changing it *after*.


---

## 11. How it runs (added 2026-09-26)

`turboedge research synthetic-ev`, from the monthly pipeline job — the first
scheduled job on or after the opening date. **Automated deliberately.** What
separates this trial from the retrospective `p = 0.0135` in
`docs/measured_results.md` §6.12 is not the statistics; it is that nobody
chooses when it runs, and nobody gets to run it again. Automation removes a
researcher degree of freedom here rather than adding risk.

Two guards, both refusing rather than warning, each with its own exit code:

| guard | exit | behaviour |
|---|---|---|
| `--not-before 2026-10-01` | 2 | refuses in Q3, which is closed at 11 of 6 |
| `docs/results_2026Q4_001.json` exists | 3 | refuses once the trial has been run; `--force` overrides and says so in the output, and voids the trial |

Every monthly run after October will legitimately refuse, so the step is
`continue-on-error`. The result file is committed to version control beside
this document, not left in an expiring artifact, and it records the fetch
date, the per-underlying bar count and the git commit — the provenance whose
absence made §6 irreproducible.
