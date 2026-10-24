# Research plan (not yet pre-registered): is there an executable cost edge
in issuer selection?

**Status: PLAN ONLY. Not a pre-registration, not an opened trial, and it
consumes no quarterly budget. No experiment with a confirmatory claim is
opened by this document.**

Written because `docs/measured_results.md` records no directional edge from
any forecast model, and the question of whether *cost* can be systematically
reduced between comparable products is separable from whether returns can be
predicted. It is also the question this repository is best equipped to
answer: it already archives real bid/ask quotes with issuer, barrier,
financing level and timestamps, which is the only data the question needs.

---

## 1. The question, narrowed

> Among turbo certificates that are **comparable** — same underlying, same
> direction, barrier distance and leverage within a stated tolerance, same
> calendar moment — does a systematic selection rule achieve a lower
> **realised round-trip cost** than a naive rule, by enough to survive the
> spread it must cross to act on it?

Deliberately **not** asked: whether such a rule produces positive net
expected value, or alpha. A cheaper product is not an edge. It lowers the
hurdle a directional signal must clear; with no signal, a cost saving on a
trade with zero expected gross return is still zero.

## 2. What "cost" means here

Round-trip cost, per unit of underlying exposure, with every component the
archive can actually evidence:

- entry at **Ask**, exit at **Bid** — never a mid, never one side twice
- the spread at both ends, measured, not modelled
- financing cost over the holding period from the observed financing-level
  path (`pricing/financing.py`), not from a quoted rate
- the issuer's implied margin as the residual against theoretical fair value
  (`premium_over_fair`)
- the **ratio** (Bezugsverhältnis), only where `ratio_reliability` is not
  `UNVERIFIED`; an unverified ratio makes every per-unit figure unreliable
- the knock-out barrier as a path-dependent terminal cost, with ambiguous
  bars never resolved favourably

Excluded as not evidenced by the archive: taxes, order fees, any assumed
rebate.

## 3. Null hypothesis

Let `C(rule)` be the realised round-trip cost per unit of exposure, averaged
over matched comparable sets, for a selection rule.

    H0:  C(naive) - C(selection)  <=  s
    H1:  C(naive) - C(selection)  >   s

where `s` is the **executability margin**: the cost of actually obtaining the
selected product rather than the naive one, including the spread at the
moment of selection and a stated allowance for quote staleness. Setting
`s = 0` would test whether a cheaper product exists on paper, which is not
the question. `s` must be fixed before any measurement.

`naive` = the first comparable product by a fixed, non-cost criterion
(proposed: lowest WKN, to be arbitrary rather than plausible).
`selection` = lowest total modelled round-trip cost at entry time.

## 4. Data requirement

Unusual for this repository: **the data largely exists**. What is needed:

- matched comparable sets: ≥ 2 products from ≥ 2 issuers on one underlying,
  same direction, barrier distance within ±10 % and leverage within ±15 %,
  quoted within the same 60-second window
- a holding period with quotes at both ends, from the existing forward
  ledger and its labels
- **independent** observations: one set per underlying per day at most, and
  effective sample counted in distinct dates with average uniqueness, not in
  rows

The binding constraint is issuer diversity, not history. `measured_results.md`
§4 records that one issuer dominates coverage, and that a direct ISIN-level
cross-check found **zero overlap** between one issuer's own feed and the
same issuer's products as attributed by a multi-issuer venue. Until matched
sets from genuinely distinct issuers can be constructed, this question is
not measurable, and that is the first thing to check rather than assume.

**Required before pre-registration:** a count of how many matched comparable
sets the existing archive actually contains, per underlying and per issuer
pair. If that count is small, the honest outcome is that the question waits.

## 5. Falsification criteria

The hypothesis is rejected if any of these holds:

1. The cost difference does not exceed `s` at the stated confidence, using
   the lower confidence bound and not the point estimate.
2. It exceeds `s` only before financing costs or only before the margin
   residual — i.e. the saving is an artefact of an incomplete cost model.
3. It exceeds `s` only for products whose `ratio_reliability` is
   `UNVERIFIED`.
4. It does not survive excluding the dominant issuer, i.e. the "selection"
   is really "avoid one issuer", which is a single observation about one
   counterparty and not a rule.
5. It vanishes outside continuous-trading hours, or depends on quotes older
   than the decision-time freshness gate allows.
6. The effective sample is below the threshold stated in advance. An
   underpowered result is not a negative result and must be recorded as
   neither.

## 6. What a positive result would and would not permit

Would permit: a cost-ranking term in product selection *among already
comparable candidates*. It changes which product implements a decision, not
whether to act.

Would **not** permit: a trade proposal. ACTIONABLE requires a confirmed
alpha source in a production state (`ranking/gates.py`), and a cost
advantage is not an alpha source. Nothing in this plan changes that, and a
positive result here with no directional edge still means **NO TRADE**.

## 7. Why this is not yet a pre-registration

Three things must be settled first, and two of them are mine to produce:

1. **The matched-set count** (§4). Without it the data requirement is an
   assumption.
2. **The value of `s`** (§3), which must be derived from measured spread and
   staleness, not chosen.
3. **A quarterly budget slot.** A confirmatory trial consumes one of six
   primary hypotheses and requires `research approve` with a named human.
   This document opens nothing.

The first two are measurement work that can proceed without a governance
decision. The third is the operator's.

---

*Prepared 2026-10-24. No trial opened, no budget consumed, no measurement
run.*
