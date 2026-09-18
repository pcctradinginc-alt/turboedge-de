# Measured Results

This document is the single source of truth for what has actually been measured
in TurboEdge-DE so far. It replaces impressions with numbers, dates, methods
and sample sizes. Sections 1–4 come from the three internal research
workstreams (W4, W5, W9) whose full write-ups live outside this repository,
on the machine this project was developed on
(`scratchpad/w4_walkforward_results.md`, `scratchpad/w9_challenger_results.md`,
`scratchpad/w5_simulation_validation.md`) — for those, this file is a
faithful, condensed summary of those runs for anyone reading the repository.
Section 5 is different in kind: it records what the scheduled pipeline
itself produced in CI, and its numbers come from the run logs and artifacts
named there, which anyone with repository access can re-read.

**Read this first if you read nothing else:** as of 2026-09-18, **no forecast
model and no challenger signal family in this codebase has a measured,
statistically significant out-of-sample advantage over doing nothing (the
unconditional/null model).** The correct behavior of the system today is to
output "no trade" on every scan. Thresholds are not lowered to manufacture
suggestions. This is now confirmed live as well as offline: across 22
scheduled CI runs (2026-09-14 to 2026-09-18), with up to 874 products fully
priced and path-simulated per run, **not one candidate reached a positive
lower-bound EV** — see section 5.

---

## 1. Forecast models (Workstream W4 — walk-forward measurement)

**Date of measurement:** 2026-09-12. **Data:** `yfinance` daily bars for
`^GDAXI` (DAX), `^NDX` (NDX), `^GSPC` (SPX), `^STOXX50E` (ESTX50),
2010-01-04 through 2026-09-11 (4187–4237 bars per index). **Method:**
`turboedge.backtest.walkforward.walk_forward_evaluate`, expanding window,
`min_train=750`, `step=21`, `embargo=horizon_days`, horizons
`(3, 5, 7, 10, 14)` trading days.

**Models compared:**
- `NullModel` (`signal_family="null"`) — unconditional historical
  distribution, no signal.
- `TsmomForecastModel` (`signal_family="tsmom"`) — the protected
  `tsmom_horizon_norm_v1` score mapped to a full predictive distribution
  (mean/sigma/quantiles), evaluated across the full grid.
- `LogisticDirectionModel` (`signal_family="logit"`) — evaluated on a
  reduced scope only (see below).

**Scope:** NullModel and TsmomForecastModel: full 4 underlyings × 5
horizons = **20/20 combinations**. LogisticDirectionModel: reduced to
`^GDAXI`, `h=5` only, `step=63` instead of 21 — its internal purged-CV
grid search over hyperparameters made the full grid infeasible in the
session time budget; this is one data point, not a survey of that model.

### Results

- **Brier score:** TSMOM's Brier score is worse than the null model's in
  **20 out of 20** tested `(underlying, horizon)` combinations. Average gap
  +0.0072 (TSMOM worse), ranging from +0.0012 (NDX, h=3) to +0.0212 (DAX,
  h=10 and h=14). The single logistic data point (DAX, h=5, reduced scope)
  is also worse than null, by +0.0014.
- **Calibration (ECE):** TSMOM's expected calibration error is worse than
  the null model's in nearly every combination, often by **5–15×** (e.g.
  DAX h=10: TSMOM ECE 0.0989 vs. null 0.0026 — roughly 38×; DAX h=5: TSMOM
  0.0478 vs. null 0.0041 — roughly 12×). This is the most consistent
  finding of the W4 measurement: TSMOM's probability outputs are reliably
  worse-calibrated than simply reporting the unconditional historical
  frequency, once actually deployed forward in time.
- **Hit rate:** mixed and small in magnitude either way; no consistent
  directional edge for TSMOM over null.
- **PSR (Probabilistic Sharpe Ratio):** ≈0.97–1.00 for **both** TSMOM and
  the null model in nearly every cell. **This is explicitly not evidence
  of skill for either model.** It follows mechanically from the strong,
  nearly uninterrupted positive drift of major equity indices over exactly
  this 2010–2026 sample window — the null model itself almost always
  predicts "up" and that alone produces a near-1.0 PSR in a 16-year bull
  market. PSR/DSR are only informative when comparing challenger to null
  directly; on that comparison neither model shows an edge once Brier/ECE
  are taken into account.

**Conclusion (W4):** the protected TSMOM baseline, mapped to a full
predictive distribution, does not demonstrate an out-of-sample advantage
over the trivial unconditional benchmark in this measurement — if
anything it is consistently worse-calibrated. This does not invalidate
TSMOM as the protected reference *score* (unchanged, still protected); it
means the specific distributional mapping built on top of it has not been
shown to beat "assume the historical distribution" out of sample.

---

## 2. Challenger signal families (Workstream W9)

**Date of measurement:** 2026-09-13. **Question:** on top of W4's finished
walk-forward harness, is there any signal family with a real, deflated
out-of-sample advantage over the null model? Six families were
**pre-registered before any measurement was run** (trial IDs
`W9-2026Q3-001` through `-006`), each probing a different hypothesis for
why TSMOM failed in W4 (sizing/confidence, regime-gating, reversal instead
of trend, a VIX-based risk filter, genuine cross-asset information, and a
maximally simple calendar-seasonality reference).

**Scope actually run:** DAX and ESTX50 at all 6 families × all 5 horizons;
NDX and SPX at 5 families (all but the EU-only `cross_asset_leadlag`) ×
horizons (5, 10) only, as a cross-underlying robustness check. This yields
**80 (family, underlying, horizon) cells measured**, out of 110
pre-registered — the reduction (NDX/SPX at horizons 3/7/14 for 5 families,
30 cells) is a documented, pre-decided-tractability reduction, not a
post-hoc exclusion of unfavorable cells.

### Results

- **78 of 80 measured cells (97.5%)** have a **worse** (higher) Brier score
  than the null model. The 2 nominally better cells (`seasonality_turn_of_month`
  and `voltarget_tsmom`, both at ESTX50/DAX h=14) have `ΔBrier` inside
  `[-0.0004, 0]` — indistinguishable from zero — and are not
  bootstrap-significant (p = 0.261 and 0.459 respectively).
- **Benjamini-Hochberg FDR (α=0.10)**, applied separately to the 80
  Brier-diff p-values and the 80 return-diff p-values: **0 of 80
  rejections on either metric.** Not one (family, underlying, horizon)
  cell clears BH-significance.
- **Best raw effect across the entire sweep:** the single lowest p-value
  anywhere in the 80-cell × 2-metric × 1-sided sweep (160 tests) is
  p=0.040, for `reversal_short_horizon` on SPX h=10 (signal-direction
  return). This does not survive BH correction (BH's rank-1 threshold at
  α=0.10/80 requires p ≤ 0.00125).
- **Best mean signal-direction-return edge over null, any family/cell:**
  **1.85 bp** (`vix_term_structure`, its most favorable single cell — see
  `w9_analysis.json`), against the **10 bp** minimum improvement
  `SIGNAL_REGISTRY.md` requires of any challenger versus the TSMOM
  baseline, and well short of even the low end of the realistic Turbo
  round-trip cost band (50–150 bp over 7 days at leverage 5–15, scaled
  linearly to horizon).
- **Cost threshold:** **0 of 80 cells** have a challenger mean
  signal-direction return exceeding even the low end of that cost band.
  There is no scenario in this measurement, deflated or not, where trading
  any of the six families after realistic Turbo costs is expected to add
  value over doing nothing.
- **Deflated Sharpe Ratio:** DSR ≈ 1.000 for all six families — explicitly
  **not informative** here, for the same reason as W4's PSR caveat above:
  every family's OOS return series inherits the sample period's secular
  drift, including the null model's own. A DSR of 1.0 shared identically
  by every family (and by both TSMOM and null in W4) is a symptom of the
  2010–2026 bull-market sample, not evidence of skill.

**Conclusion (W9):** no signal family tested beats the null model
out-of-sample once multiple-testing-deflated, and none would clear
realistic Turbo costs even if it did. All six families
(`voltarget_tsmom`, `lowvol_regime_trend`, `reversal_short_horizon`,
`vix_term_structure`, `cross_asset_leadlag`, `seasonality_turn_of_month`)
are recorded `dormant` in `state/registry/failed_hypotheses.json`. See
`SIGNAL_REGISTRY.md` §3 for the per-family table with trial IDs.

---

## 3. Path model / knock-out probability calibration (Workstream W5)

**Date of measurement:** 2026-09-12. **Underlying:** `^GDAXI` (DAX) daily
bars, 2015–2025 (558 historical start dates used for the realism checks).
**Question:** does the simulated P(KO) from `simulation/paths.py` match
what actually happened historically?

### Finding: P(KO) is systematically too high (over-predicted)

At the barrier distances most relevant to real turbo placements
(k = 1.5–2 EWMA-sigma), every resampling method tested **over-predicts**
knock-out risk relative to the realized historical touch rate — simulated
P(KO) runs **roughly 40–135% relatively higher** than what actually
happened (e.g. DAX, h=10, k=2, long: simulated 0.117 vs. realized 0.072,
+63%; short: simulated 0.084 vs. realized 0.036, +134%).

### Method comparison (mean absolute calibration error, lower = better)

Averaged over the (h, direction) cells at k ∈ {1.5, 2.0} — the range
closest to real turbo barrier placements:

| method | mean \|diff\|, k∈{1.5,2.0} | mean \|diff\|, all k tested |
|---|---:|---:|
| `block_bootstrap` (old default) | 0.0549 | 0.0334 |
| `monte_carlo` | 0.1133 | 0.0627 |
| `regime_bootstrap` (fixed) | 0.0216 | 0.0129 |
| `vol_scaled_bootstrap` (new default) | **0.0194** | **0.0122** |

`vol_scaled_bootstrap` (standardize each historical day by its own
trailing EWMA vol, block-bootstrap the standardized shocks, rescale by
current EWMA vol) has the lowest calibration error both in the
trading-relevant range (0.019, ~2.8× better than the old `block_bootstrap`
default of 0.055) and overall (0.012, ~2.7× better than 0.033). It is now
the default `method` for `simulate_paths`. `monte_carlo` is strictly
worse than `block_bootstrap` at every tested row and was ruled out as
default.

A confounding root-cause investigation (the "weekend-i.i.d." hypothesis —
that mixing blocked-weekday with i.i.d.-weekend sampling inflates P(KO))
was tested directly and **not supported**: making weekday sampling i.i.d.
too left P(KO) essentially unchanged (0.1056 vs. 0.1072, within Monte
Carlo noise for n=558). The actual driver, to the extent this validation
could establish it, is that the 750-day lookback window is drawn
*unconditionally* — it mixes calm and volatile regimes (including crash
episodes) into every draw regardless of which regime prevails at the
simulated start date.

### Residual bias after switching to `vol_scaled_bootstrap`

Even the best-calibrated method still over-predicts P(KO) at k=1.5–2
sigma: mean signed diff (realized − simulated) of **−0.019** at the
trading-relevant range. **Direction of the residual bias: conservative.**
An over-stated P(KO) mechanically depresses net EV in the downstream
EV/utility calculation and makes the ACTIONABLE gate (`LCB(EV) > 0`)
harder to clear, not easier — the system is biased toward proposing
*fewer* trades than a perfectly calibrated model would, not more. At
k=3–4 sigma several method/direction/horizon cells already flip to a
slight under-prediction, so the residual is not uniformly one-directional
far from the barrier — only close to it, which is also where it matters
most for gating.

**Downstream consumers of P(KO) should treat it as a conservative
(upper-bound-ish) estimate of true knock-out risk at small-to-moderate
barrier distances, not a calibrated probability.**

---

## 4. Product data coverage

- **BNP Paribas** (`derivate.bnpparibas.com`): live bid/ask, full DAX
  coverage confirmed (4340/4340 products in one pull, per prior source
  research).
- **Citi / CitiFirst** (`de.citifirst.com`): master data (financing level,
  barrier, ratio, ISIN) and closing-reference prices only — **not live
  quotes**. A market-hours pull observed `referencePriceMethod = "Closing
  Price"` and `ask = 0.0` on all returned rows; the adapter sets bid/ask to
  `None` for these, so Citi does not contribute to live pricing today.
- **gettex** (`gettex.wsd.com`): multiple issuers (BNP Paribas, UniCredit,
  Goldman Sachs, HSBC observed), with the product's leverage-implied
  bid/ask-to-underlying ratio **derived and independently verified**
  against gettex's own reported reference price. Live validation run
  2026-09-13 (a Sunday — German markets closed, so all quotes reflect the
  prior session's close):
  - **DAX:** 3000 raw rows fetched; 1695 (56.5%) had a ratio successfully
    derived and verified; 1304 (43.5%) rejected at the ratio-grid snap
    stage; 1 rejected at implied-spot verification. Consensus spot from
    accepted products: 25552.44 (DAX-derivation basis). Median
    `|S_implied − S_ref| / S_ref` across accepted products: **0.0497%**.
  - **NDX:** 3000 raw rows fetched; only 98 (3.3%) had a ratio derived and
    verified — the ratio grid used matches DAX-style products far more
    often than NDX-style ones. Median deviation for accepted products:
    0.0140%.
  - A direct ISIN-level cross-check against BNP's own live product feed
    found **zero overlap** with gettex's BNP-attributed DAX ISINs
    (0/1115), even after a broadened raw search across 9998 distinct
    ISINs. The most plausible explanation, consistent with prior research
    (`docs/data_sources.md` §9a), is that gettex continues to quote
    BNP-issued certificates no longer listed on BNP's own current-offering
    product finder (a normal secondary-market lifecycle state, not a
    data-quality defect in either feed) — a direct ISIN-level
    cross-validation could not be completed for that reason this session.
- **Börse Stuttgart** and **Börse Frankfurt / Deutsche Börse**: blocked
  (Cloudflare bot-management on Stuttgart; salted-hash signature headers
  from obfuscated JavaScript on Frankfurt). No adapter exists for either;
  neither was bypassed. See `docs/data_sources.md` for the full write-up.

---

## 5. Live pipeline results (scheduled CI runs, 2026-09-14 to 2026-09-18)

Sections 1–3 are offline research measurements. This section records what
the scheduled pipeline actually produced once it ran unattended in CI for
five days (22 scheduled runs, `.github/workflows/pipeline.yml`). It exists
because two of the findings below were only visible in production, and one
of them silently disabled the entire evaluation stage.

### Finding: a single freshness threshold rejected the whole universe

Until 2026-09-14 one threshold (`max_quote_age_s: 120`) answered two
different questions at once, and was measured against `evaluation_time` —
i.e. after the *complete* multi-source fetch had finished. Measured in CI
(run 34883437354): the fetch itself took 227.1 s (DAX) / 261.4 s (NDX), so
the **minimum** decision-time quote age in the entire run was already
171.5 s / 208.3 s — above the 120 s cutoff. No product could pass,
regardless of data quality. Result: `quote_stale` on 11,060 of 11,114 DAX
and 7,059 of 10,566 NDX candidates, `WATCH = 0`.

The gate was split into two thresholds, each derived only from the
measurement that answers its own question:

| | threshold | measures |
|---|---|---|
| `max_source_quote_age_s` | 900 s | `quote_timestamp` vs. that row's own `retrieved_at` |
| `max_quote_age_at_decision_s` | 450 s | `quote_timestamp` vs. `evaluation_time` |

Source-side age per issuer (`product_snapshots`, 69,424 rows, 2026-09-14,
`epoch(retrieved_at - quote_timestamp)`, p50 / p90 / max seconds): BNP
Paribas (n=66,418) −4.1 / 134.4 / 11,473.0; UniCredit (n=1,162) 170.6 /
301.7 / 9,982.7; HSBC (n=582) 171.4 / 363.8 / 595.5; Goldman Sachs
(n=1,187) 169.5 / 423.8 / 2,158,027.4; Citigroup (n=75) 797.3 / 999.8 /
1,085.3. Rows exceeding the 900 s cutoff: BNP 39, Goldman Sachs 12,
UniCredit 1, HSBC 0, Citigroup 34 of 75. The gate therefore rejects
genuinely dead quotes — including a 25-day-old Goldman Sachs quote — while
no longer rejecting the universe for the pipeline's own runtime.

**Caveat, measured and unresolved:** source-side age is systematically
negative for BNP (median −4.1 s, minimum −36.6 s) and negative at the
minimum for Goldman Sachs and HSBC. The issuer's quote timestamp precedes
our own retrieval clock, so the two clocks disagree by seconds. Harmless at
a 900 s cutoff (negative values pass trivially) and irrelevant to merging
(`universe/discover.py::_pick_winner` sorts on `quote_timestamp` itself),
but the quantity carries a systematic offset of a few seconds.

### Effect of the split, measured in CI

| | before (run 34883437354) | after (run 34888059675) |
|---|---|---|
| ACTIONABLE | 0 | **0** |
| WATCH | 0 | 12,497 |
| REJECT | 18,119 | 5,680 |
| DATA_QUALITY | 3,561 | 3,572 |
| products EV-evaluated | 120 | **874** |
| product × horizon evaluations | 600 | 4,370 |
| fetch duration DAX / NDX | 227.1 s / 261.4 s | 186.4 s / 128.3 s |

The decisive number is the second-to-last row: before the fix the pipeline
priced and path-simulated **120** products in CI, essentially only those
drawn by the shadow sample, because the regular pre-EV pool was empty. The
gap between 12,497 WATCH and 874 evaluated is by design, not loss — the
pre-EV pool is the 25 cheapest candidates per (direction, leverage bucket)
plus 3 per stratum drawn from the full pool for selection-bias protection
(`pipeline/scan.py::_select_ev_pool`).

**ACTIONABLE remained 0 in all 22 scheduled runs.** With 874 products fully
evaluated, no candidate reached a positive lower-bound EV. The fix restored
the measurement basis; it did not produce an edge, and was not expected to.

### Fetch duration and concurrency

The three product hosts are fetched concurrently (one thread per host).
This does not weaken the politeness rule: `derivate.bnpparibas.com`,
`de.citifirst.com` and `gettex.wsd.com` are distinct hosts, each adapter
holds its own per-host rate limiter (`adapters/base.py::_HostRateLimiter`,
lock-protected, ≥1.5 s between requests), and requests within one host stay
serialized. Measured: local 81.0 s / 74.9 s → 49.9 s / 57.4 s; CI 227.1 s /
261.4 s → 186.4 s / 128.3 s (CI is network-bound rather than rate-limit
bound, so it gains less). A scan still takes 1–3 minutes structurally.

Concurrency is covered by a test that a sequential loop cannot pass
(`tests/pipeline/test_universe.py`): three fake adapters block on a shared
`threading.Barrier(3)`. Verified in both directions — green as shipped, and
red with the pool forced to one worker (the first adapter blocks the full
5 s, then all three fail with `BrokenBarrierError`).

### After-hours runs correctly produce nothing

Four scheduled runs (21:13–21:45 UTC, i.e. after both the Xetra and the US
close) produced `WATCH = 0` with ≈16,500 REJECT. Both freshness gates fire
on the same products (DAX: `source_quote_stale` 9,900 and
`quote_age_at_decision` 9,900 of 10,022). This is correct behavior — the
quotes genuinely are hours old — not a regression.

### Learning loop: first run on real data

The evening chain (label → learn → position reevaluate) first processed
real data on 2026-09-17 (run 35287791715):

- `label`: labeled = 221, ko = 1, ambiguous = 0, missing_data = 0
- `learn`: posteriors_updated = 4, models_reweighted = 0, drift_events = 3
- `position reevaluate`: mails_sent = 0

`models_reweighted = 0` is consistent with sections 1–2: there is no model
with a measured advantage to re-weight toward. `drift_events = 3` are
Page-Hinkley detections, which reduce weights but never delete a model
(rule 32).

### Open risk: state growth is not yet bounded

The encrypted state artifact grew monotonically from 43.6 MB (2026-09-14)
to 168.8 MB (2026-09-18); the database reached 241.2 MB. `db compact` on
2026-09-17 removed **0 of 517,548 rows**. That is correct, not a defect:
with `keep_days: 5` and the first data written 2026-09-13, every row was
still inside the retention window, so nothing was eligible for thinning
(the 241.2 → 229.7 MB reduction that run came purely from the file
rewrite). Thinning first becomes eligible from 2026-09-18.

Projected effect, measured against the local database (69,428 rows):
reducing aged rows to one per (ISIN, UTC calendar day) leaves 26,499 rows,
a 61.8% reduction, i.e. roughly 2.9 snapshot rows per ISIN per day. Applied
to the CI volume (≈21,000 products per scan, several scans per day) the
steady state after 90 days is still on the order of millions of rows, so
thinning alone slows growth rather than bounding it. This is an open item,
not a solved one.

---

## 6. Conclusion

Across every avenue tested so far — the protected TSMOM baseline mapped to
a full predictive distribution (W4), six independently pre-registered
challenger signal families covering sizing, regime-gating, reversal,
volatility-term-structure, genuine cross-asset information, and calendar
seasonality (W9) — **no model has a measured, multiple-testing-deflated
out-of-sample advantage over the null (unconditional) benchmark, and no
tested effect survives realistic Turbo trading costs even before
deflation.** The knock-out path model (W5) is, after calibration work,
measurably closer to realized history than its previous default but still
runs conservative (over-predicts KO risk) at the barrier distances that
matter most for gating — a known, quantified, safe-direction bias, not a
calibrated probability. Given this evidence, the correct output of the
system is **"no trade"** for every scan until a model clears the
pre-registered ladder in `GOVERNANCE.md` (§2) on genuinely new,
out-of-sample data. Detection thresholds and gates are not lowered, and no
model is promoted, in order to manufacture trade suggestions where none
are supported by measurement. See `SIGNAL_REGISTRY.md` for the per-model
registry entries and `GOVERNANCE.md` for how these trials count against
the quarterly adaptation budget and multiple-testing deflation.
