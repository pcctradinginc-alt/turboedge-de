# Autonomous Alpha Discovery Engine — Phase 1 (2026-11-08)

**State: reviewed bounded exploratory feature-composition engine. No alpha
validated, no strategy deployed, no risk gate relaxed.**

## Why the engine is autonomous, and what it is not

Before this change the controller chose from catalog hypotheses whose
harnesses were listed explicitly in source code. Phase 1 can generate
new, reproducible research questions NOT present in that catalog. A
deterministic generator reads entry-time issuer quotes and composes two
independently observed features from a finite approved DSL:

- relative bid/ask spread (smaller is cheaper);
- independently timestamped quote age (smaller is fresher);
- displayed bid depth (larger suggests more executable size).

It learns which features discriminate between otherwise comparable products
from development data (not from P&L), and automatically proposes a priority
order of combinations. Feature names/operators are fixed reviewed Python
code, not free-form input that can execute arbitrary Python. Parameters are
frozen: 65/35 blend, DAX LONG, 24-hour hold, 24–48h exit observation window.
This is bounded **compositional** novelty. It is *not* unrestricted algorithm
invention and cannot know that the resulting strategy is profitable.

The machine constructs hypotheses and runs one admissible exploration per
scheduled weekly research cycle without per-hypothesis approval. Hypotheses
are persisted as PROPOSED only; no machine-generated statement becomes an
approved research queue lifecycle status or a confirmed alpha.

## Immutable historical research evidence and budget

Reuse existing DB tables:
- research_opportunities: generated hypotheses prefixed RO-GEN- with stable
  hash of typed Rule specification; estimates carry MEASURED, DECLARED or
  UNKNOWN provenance (economic expectation stays UNKNOWN before a test).
- research_trials: each executed experiment uses existing exploratory_screen
  kind, never a confirmatory trial;
- research_screens: exact DSL spec, historical data fingerprint, verdict,
  error, sample counts and costs, all auditably preserved. Screen support
  flag is always FALSE because historical exploratory evidence is
  *ineligible for confirmation/promotion without a separate approval*.

A historical hypothesis fingerprint is never executed twice, even next
quarter. Failed experiments remain archived; overlap with failed feature
combinations lowers the next exploration's priority. No new research-state
store is introduced. The same 12-per-quarter exploratory screen cap applies
to both the prior controller and this engine; at most one experiment runs per
cycle. The 6-per-quarter confirmation budget is untouched.

## Data, causality and executable accounting

- Only quote records available at or before the evaluation cutoff; strict
  quote/reference timestamp checks, <=300s age, >=0.8 quality, non-stale,
  verified ratio, valid bid <= ask.
- Same DAX underlying/direction, barrier and leverage matching by the
  existing product_pairs census, genuine distinct issuers, <=60s quote
  window. First eligible pair per day and at least three calendar days
  between observations. Do not choose the better surviving product after
  learning exits.
- Chronological first-half DEVELOPMENT features; latter-half historical
  VALIDATION after a two-day embargo. The generator never accesses outcome
  prices to choose its feature rule. Validation is historical, **not a
  secretly untouched confirmatory partition**.
- Entry at *actual archived Ask*. Exit is the first valid, later *Bid*
  at 24–48 hours, documented KO means complete loss when it occurs before
  exit. An additional 10bp execution allowance is charged. A missing
  realised exit or feature causes INCOMPLETE_EVIDENCE (never profitability).
  Recorded Bid is an executable-price *proxy*, not a demonstrated real fill:
  market depth, trade size, fees/taxes and conditional execution remain
  uncertain. No extra theoretical financing cost is double-charged; the
  observed ask-to-bid path includes financing-level changes.
- Controls: NO_TRADE (zero return) and the same matched naive issuer-first
  counterfactual. A long DAX return alone is not alpha.
- At least **48 distinct, 3-day-spaced validation observations**, not
  nominal product rows. 2,000 deterministic bootstrap draws over those days;
  family-wise one-sided 5% is divided by 6 for the finite DSL candidate
  family. The lower bound on mean net return must exceed 10bp, and the
  lower bound on incremental return vs the naive issuer must exceed zero.
  The result is only CANDIDATE_REQUIRES_INDEPENDENT_CONFIRMATION.
- No evidence returns AWAIT_DATA or INCOMPLETE_EVIDENCE. Weak or negative
  economic results return NO_DEMONSTRATED_NET_ALPHA. A valid technical
  experiment is not proof of a discovered edge.

## Operational constraints

The existing five-day hot snapshot retention thins ordinary ISINs to one
quote/day, so cross-issuer intraminute observations may accumulate too
slowly for 48 independent validation days. Phase 1 reports AWAIT_DATA rather
than weakening the threshold. Preserving a carefully bounded independent
intraday research sample is the next **data engineering** workstream; this
PR does not increase state retention or state upload costs.

Commands:
- uv run turboedge research discover-alpha --dry-run
- uv run turboedge research discover-alpha
- uv run turboedge research discover-alpha --json-out reports/weekly/alpha_discovery.json

The weekly pipeline invokes the same command after the comparable-pair
census. It writes a structured report. It does not send orders, change
models, produce ACTIONABLE signals, create a confirmatory trial or promote
an alpha. Human-approved independent future confirmation and forward
shadow verification are separate Phase 2 work.

## Verification criteria

Unit/integration tests prove:
1. invalid arbitrary DSL code and duplicate features are rejected;
2. a new RO-GEN hypothesis is derived from entry quote data, not a catalog;
3. synthetic positive outcomes stay explicitly exploratory;
4. negative outcomes and missing exits are preserved as non-alpha;
5. chronological cutoff, distinct-day and freshness guards fail closed;
6. replay does not re-test the same hypothesis;
7. past failures penalise similar feature combinations;
8. shared quarterly screen cap and human approval barriers remain unchanged.

Synthetic wins in tests are intentionally constructed and **do not** count
as historical alpha.
