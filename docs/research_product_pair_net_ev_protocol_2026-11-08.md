# Fixed exploratory protocol: cross-issuer product selection vs NO_TRADE

Date fixed in source: 2026-11-08. **Not a confirmatory pre-registration,
not an approved primary trial, not a trading signal.** Historical snapshots
were accessible during method design; results from them are DEVELOPMENT
only. A real confirmatory experiment must be registered by a named human
and use a disjoint, unseen forward partition.

## Question

Does selecting the cheaper *executable spread* among two otherwise
comparable DAX LONG turbo certificates yield positive **net** forward return,
and improve on a naive issuer-alphabetical selection, rather than merely
reducing a displayed spread? The primary comparator is NO_TRADE = 0.

## Frozen measurement

- Fixed market: DAX LONG, not chosen after seeing P&L.
- Entry: first eligible comparable cross-issuer pair per UTC day, product
  matched within 60s, 10% barrier-distance tolerance, 15% leverage tolerance,
  verified ratio; minimum quote quality 0.8. All quotes and reference spot
  timestamps must be independently available and no older than 300 seconds.
- Rule: buy at observed Ask the member with smaller observed
  (Ask - Bid) / Ask; deterministic issuer/ISIN tie-break. Naive rule:
  issuer/ISIN alphabetically first. There is no short leg.
- Exit: first independently timestamped real Bid 24-48h after entry.
  Documented knockout before exit => complete loss. Otherwise missing exit
  => INCOMPLETE_EXITS, **not** a conveniently omitted winner/loser.
- Executability allowance: 10 basis points on top of the real bid/ask
  round trip. Financing and issuer margin are already reflected in actual
  observed prices; no second theoretical financing fee is subtracted.
  Broker-specific fees/taxes and actual fill probability remain unverified.
- Unit of independence: at most **one** DAX observation per UTC day.
  Require >=30 independent days; 3000 deterministic day-bootstrap replicates,
  one-sided 5% lower bound. No search over thresholds, instruments or horizons.
- Exploratory CANDIDATE requires both LCB(mean net return) > 10 bp and
  LCB(mean excess over naive) > 0. A lower result is
  NO_DEMONSTRATED_NET_ALPHA, not a reason to relax thresholds.
- This is the *only* exploratory economic probe in this change and consumes
  no confirmatory slot. It does not alter the production scan, weights,
  gate thresholds, risk limits, promotion state or broker integration.

## Interpretation and limitations

A long DAX exposure can profit because DAX rose, even if the product-selection
rule had no alpha. This is why a naive same-underlying paired comparator is
required in addition to NO_TRADE. Moreover, the matched-product census is
bucket-based and undercounts; source coverage and missing quote times can make
this unanswerable for weeks. A positive retrospective exploratory number is
not confirmation: the true test requires a fresh untouched forward partition,
independent sessions, issuer/liquidity plausibility, and a formal cost/risk
approval. The only allowed automatic output is a research report.

Run manually: `uv run turboedge research product-pair-net-ev`
