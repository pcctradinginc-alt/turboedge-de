# Alpha Factory

Why this repository is moving from model-centric to alpha-centric research, and
what that changes. Plain language; the numbers live in
`docs/measured_results.md` and the rules in `GOVERNANCE.md`.

## 1. Why the change

TurboEdge spent its first phase looking for a good forecast. It found one. In
September 2026 `regime_conditional` improved CRPS in 18 of 20 cells at
p ≈ 0.0135 under a bootstrap that respects both overlapping label windows and
cross-cell correlation — a real result by any forecasting standard.

Then it was pushed through the full payoff / knock-out / cost machine and
measured **economically worse than doing nothing**: mean ΔLCB-NetEV = −0.0205,
negative in every underlying, every horizon and every barrier distance
(§6.15).

That is the whole argument. A better forecast is not an edge. Searching harder
for one good model was searching in the wrong place.

## 2. Model, alpha, strategy — three different things

These get conflated, and conflating them is how a repository convinces itself
it has something.

A **model** is a mathematical predictor. It has a Brier score, a CRPS, a
calibration curve. `ModelRegistry` tracks models.

An **alpha source** is a *measurable economic effect*. It may come from a
model, but equally from a regime interaction, a product-selection rule, issuer
behaviour, a pricing anomaly, execution, or path modelling. Its unit is net EV
after costs, not accuracy. `AlphaRegistry` tracks alpha sources.

A **strategy** is a decision rule that combines alpha sources and allocates
risk across them. Neither registry tracks strategies; the portfolio layer will.

Model performance is not alpha. Alpha is not a strategy.

## 3. Data Factory

Point-in-time history is the one asset that cannot be bought later. Underlying
prices can be re-downloaded; a record of what BNP quoted for a specific ISIN at
14:32 on a specific day, and whether that quote vanished an hour later, cannot.

Until 2026-09-27 this repository deleted its own product snapshots after 90
days. The deletion was justified by a measured sweep that asked what the
*operational* consumers need — correct answer, wrong question. It now keeps
them indefinitely, at a measured ~0.36 GB/year.

The deep fetch that fills the archive is deliberately **separate from the
decision path**: `pipeline/archive.py` runs once daily, reads gettex at full
depth across four issuers, and writes only snapshots. Raising the live scan's
depth instead would have added ~120 s per underlying and pushed decision-time
quote age past `max_quote_age_at_decision_s`, rejecting products on staleness.
More data would have meant fewer candidates.

## 4. Alpha lifecycle

    IDEA -> EXPLORATORY -> VALIDATED -> CONFIRMATORY -> FORWARD_SHADOW
      -> CANARY_PRODUCTION -> LIMITED_PRODUCTION -> NORMAL_PRODUCTION

with `HEALTHY / WEAKENING / DECAYING / DORMANT` describing a promoted alpha's
condition, `REJECTED` terminal, and `DISABLED` reachable immediately from any
production state.

Promotion is **evidence-gated and automatic** under frozen criteria
(`GOVERNANCE.md` §1.1a). Demotion is also automatic. A historical winner is
never permanently trusted.

## 5. Research lifecycle, and why it is separate

An alpha's lifecycle is about an *effect*. A research hypothesis's lifecycle is
about a *question*: proposed, approved, running, measured, promoted or dormant
(`meta/research_opportunity.py`). The two are deliberately distinct enums. A
question can be answered "no" — that is a successful research outcome and a
failed alpha.

## 6. Exploration versus confirmation

The tension: a factory needs to try many things, and trying many things
inflates false positives. The resolution is not a looser threshold.

**Exploratory sandbox** — broad screening, ablation, development data. Every
idea logged. Exploratory evidence can never enter production and exploratory
p-values are never treated as confirmation. Its output is hypotheses.

**Confirmatory pipeline** — frozen hypothesis, frozen features, frozen method,
frozen parameters, frozen economics, frozen primary endpoint, untouched
confirmation data, one shot. No result-dependent adjustment.

Chronological partitions make this concrete: DEVELOPMENT, VALIDATION,
CONFIRMATION, FORWARD. A CONFIRMATION boundary cannot be moved after it is
frozen. `research/partitions.py` enforces what it can and documents what it
cannot — intent is not checkable by a type system, and pretending otherwise
would be worse than saying so.

This repository has already shown why the discipline matters. Its 2026Q3
adaptation budget of 6 stands at 13. Every overrun was individually justified
in writing. That is exactly how it happens.

## 7. Shadow versus production

Research experiments. Shadow evaluates on live data and records outcomes with
no trading authority. Production runs frozen, promoted components. There is no
direct research-to-production path.

Shadow results are never labelled real trades. The Phase 1 meta-controller has
run shadow-only since it was built and abstains on every horizon; that is not a
placeholder, it is the layer reporting honestly that it cannot yet judge.

## 8. Edge attribution

The question "did this trade make money" is much less useful than "which part
of the machine made it". `EdgeAttribution` decomposes each decision into
forecast, path, product-selection, cost-selection, issuer, timing and portfolio
components plus an interaction residual, and validates that they reconcile to
the total.

The residual is not a dumping ground — it is the measure of how much is *not*
understood. Hiding everything under "model alpha" is the failure mode this
object exists to prevent.

The first such measurement landed on 2026-09-27: product selection beat the
median alternative by +71 bp on 134 observations while random picks lost 350 bp.
Effective sample ≈ 1, so it is a first reading and not a result.

## 9. Alpha decay

An edge that stops working while still being sized is worse than no edge. Decay
monitoring — rolling forward net EV, LCB, posterior probability positive,
calibration, selection regret, drawdown, tail loss, regime stability,
disagreement, data quality — gets the same effort as discovery, and drives
automatic weight reduction within pre-approved bounds.

History is never deleted. A decayed alpha stays in the registry.

## 10. Counterfactual learning

Learning only from what was chosen throws away most of the information. Each
decision records realistic alternatives: no-trade, other models, other
horizons, other directions, other products, other issuers, other allocations.

That turns "did this make money" into "was this better than what else was
available" — a question with an answer even when the trade lost.

## 11. Research prioritisation

`ResearchPriority ≈ PlausibleEconomicImpact × ExpectedInformationGain ×
P(resolving uncertainty) × Reusability / ImplementationCost`, penalised for
leakage risk, tiny effective samples, poor sources, redundancy and
near-duplicates of failed hypotheses.

The controller's real question is *which unresolved uncertainty currently costs
the most money*. Every input carries its provenance — MEASURED, DECLARED or
UNKNOWN — because a priority score assembled from invented numbers looks
exactly as authoritative as one assembled from measured ones.

## 12. Meta-learning

Eventually the system should learn about its own research process: which
families historically produced incremental net EV, typical effect sizes,
typical effective samples, time-to-answer, duplication and decay rates.

One input into prioritisation, never the only one. Past success must not
suppress exploration, so an exploration component is retained.

## 13. Portfolio layer

Not "is this turbo good" but how much risk each simultaneous opportunity gets,
across underlyings, horizons, directions, alpha sources, products and issuers —
maximising expected net EV subject to expected shortfall, drawdown budget,
knock-out and issuer concentration, alpha correlation, model uncertainty,
liquidity and capacity.

Shadow only when it arrives. Existing Kelly sizing and cluster risk stay.

## 14. System generations

A frozen configuration gets a `generation_id`. A generation is **not better
because its backtest is better** — improvement requires forward evidence on
genuinely new data. `research/generations.py` makes that comparison explicit
precisely so "the system improved" cannot be claimed because weights moved.

## 15. Why most hypotheses are expected to fail

Because they are. 0/20 forecast cells, 0/80 challenger cells, Cboe (83,723
observations), CFTC (10,128 observations over 16.2 years), and the one
forecast-quality success that failed economically.

A research process with a high hit rate is either lucky or not correcting for
multiple testing. The factory's value is not that most ideas work; it is that
bad ones are rejected cheaply and the survivors combine.

Every failure stays in `failed_hypotheses.json`. No cosmetic rename bypasses
it: retesting requires a materially different condition — new data, new regime,
new mechanism, new methodology, a much larger independent forward sample, or
better data quality.

## 16. What self-improvement means here

Not more models, more features, more trades or more promotions. Those are
activity.

Improvement means **better forward economic outcomes on data that did not exist
when the change was made.** Everything else is a diagnostic — including every
number in this document.
