# Weekly Tournament Evidence Audit — 2026-11-14

Scope: The week-46 email reported `n=125 / n_effective=1247` for historical models and
`n=41094 / n_effective=9.45` for `unresolved`, with no champion or passing promotion.
This memo documents the *implementation changes*, not a newly discovered strategy.

## Fixed measurement and provenance

1. **OOS sample independence:** `backtest/walkforward.py` now counts average uniqueness
   of actual overlapping h-bar forward labels. `n_folds` remains the number of
   cross-validation folds; raw `len(oos_predictions)` is not labelled effective.
   Historical PSR receives the corrected `n_effective`.
2. **Correct directional return:** The historical mean is the unconditional,
   uniqueness-weighted forecast-direction strategy return (including neutral
   predictions), **not** the equal-weight average of conditional winning long
   and short subsets. This is still underlying *gross* return, not Turbo P&L.
3. **Identity:** Latest walk-forward results retain `(model_id, underlying_id,
   horizon_days)`. Each cell is printed separately in the weekly email; no
   "latest cell" is selected as the family's global performance.
4. **Forward labels:** The weekly performance calculation uses confirmed
   horizon-bid or KO labels only. Missing exit quotes and conservative proxy
   valuations are excluded from measured economics and counted explicitly.
5. **Proposed strategy vs research:** Only non-shadow `ACTIONABLE` decisions
   enter performance statistics and promotion comparisons. `WATCH`, `REJECT`
   and shadow observations are reported separately as research-only, not as
   hypothetical executed strategies. No actual brokerage trades exist.
6. **Ensemble identity:** Newly written ledger rows use `ensemble_v1` and
   the actual ensemble forecast hash. Old `unresolved` observations remain
   unassigned if historical provenance cannot be reconstructed; we do not
   re-label them as Logit or TSMOM without a matching record.
7. **Matched model comparison:** Challenger vs champion EV and hit-rate
   improvement use the intersection of identical (basiswert, direction,
   horizon, prediction timestamp, exit date) events, averaged by event and
   overlap-weighted. A first champion is compared to explicit NO_TRADE=0,
   without inventing a no-trade win-rate.
8. **Safeguards:** No PSR/DSR without 100 effective outcomes. Promotion also
   requires sufficient matched outcomes, a registered trial and plausible
   entry quotes. Significance and automatic release are prevented when data
   and independent confirmation are insufficient. Demotion by statistical
   metrics also needs an adequate sample.
9. **Research budgets:** The email reports quarterly registered exploratory,
   executed screen and nonexploratory trial counts against the existing
   12/6 ceilings. It does not authorize additional experiments.
10. **Email readability:** Explicit evidence source, historical per-cell
    metrics, research-only counts, excluded exits, budget and failed gates
    appear in HTML and text.

## Do not confuse these quantities

- **Gross directional OOS return**: underlying long/short return on historical
  price bars, no Turbo leverage, issuer spread, financing or KO execution.
- **Quote-based Turbo outcome**: observed entry Ask vs exit Bid (or measured
  KO event), which captures recorded price movements and quoted spreads.
  It is an *execution-price proxy*, not a real fill; actual fees, taxes,
  size-dependent slippage and issuer quote availability can still change P&L.
- **No-trade benchmark**: zero investment return, assessed on identical
  decision opportunities, not against a different model's unrelated trades.
- **Validated live alpha**: NOT established by either a single historical
  forecast cell or this weekly email.

## Remaining evidence-dependent blockers

- The seven undefined enterprise-level risk/promotion thresholds in
  `alpha/promotion.py` are **not made up** to allow promotion.
- Independent untouched confirmation, sufficiently long forward-shadow
  observation, issuer diversification and execution/fill plausibility must
  be established before any production alpha promotion. This is separate
  from software tests and synthetic positive examples.
- Old `unresolved` rows cannot safely be reconstructed without the original
  model/provenance state.
- Research quote capture runs under `research_capture` and
  `research product-pair-net-ev` and must accumulate real, market-hour,
  PIT-valid ask/exit-bid observations. The per-run outcome and data
  completeness should be interpreted from their own research reports; a
  successful CI job does not imply a profitable product-selection edge.
- Cross-asset correlation, nonstationarity and trial-budget violations can
  further reduce effective statistical power. No numeric sample size by
  itself proves market independence.

**Acceptance:** Ruff lint, formatting, mypy and full pytest suite green in
GitHub Actions. Tests validate implementation invariants, **not** trading
profitability. Keep broker/execution features absent and research-only
disclosures intact.
