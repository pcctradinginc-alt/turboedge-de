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
named there, which anyone with repository access can re-read. Section 6
(Phase D, 2026-09-19) is a fourth workstream, run directly against this
repository's own `backtest/walkforward.py`.

**Read this first if you read nothing else:** as of 2026-09-19, **no forecast
model and no challenger signal family in this codebase has a measured,
statistically significant out-of-sample advantage in net Expected Value over
doing nothing (the unconditional/null model).** The correct behavior of the
system today is to output "no trade" on every scan. Thresholds are not
lowered to manufacture suggestions. This is now confirmed live as well as
offline: across 22 scheduled CI runs (2026-09-14 to 2026-09-18), with up to
874 products fully priced and path-simulated per run, **not one candidate
reached a positive lower-bound EV** — see section 5. Section 6 (Phase D)
adds one nuance without changing this conclusion: three pre-registered
baselines show a consistent, 20/20-cell improvement in CRPS (a proper
distributional scoring rule) over the null model — real, but a
distributional finding, not a measured directional or net-EV edge; see §6.3
for why this does not change the "no trade" conclusion.

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

### 3.1 Out-of-sample calibrator promotion attempt (Workstream W10, 2026-09-19)

**Question:** the W5 finding above (P(KO) over-predicted by 40–135% at
k=1.5–2σ) was measured with code that is no longer in this repository. Can a
*calibrator* — fit walk-forward, out-of-sample, on a freshly rebuilt
empirical dataset — improve on raw P(KO) without eroding the conservative
safety margin that makes the ACTIONABLE gate harder to clear, not easier?
Implemented in `simulation/ko_calibration.py` (dataset) and
`backtest/ko_calibration.py` (walk-forward fit/evaluate/promote), run via
`turboedge research ko-calibration`. `ranking/gates.py` and
`pipeline/scan.py` were **not modified** — gates still consume raw P(KO)
regardless of this section's outcome, and the scan pipeline's population of
the new (additive, currently empty) `p_ko_raw`/`p_ko_calibrated`/
`ko_calibrator_version` columns on `candidate_sets` is deliberately left for
a later change.

**Dataset.** All 4 currently `enabled: true` underlyings in
`configs/universe.yaml` (DAX, NDX, EURUSD, XAU), `yfinance` daily bars
(~2010–2026, `configs/simulation.yaml`'s `n_paths=2000`,
`method=vol_scaled_bootstrap`, `block_size=5`, `lookback_days=750`,
`seed=20260101`), every 5th business day as an "as of" bar (a documented,
applied-before-any-result-is-seen stride — see
`simulation/ko_calibration.py`'s docstring), horizons `(3, 5, 7, 10, 14)`,
both directions, standardized barrier distances
`k ∈ {0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0}` (same
`pct = k·σ·√horizon` convention as `features/product.py::distance_to_barrier`).
Sigma and the volatility regime bucket are computed only from bars with
`available_at <= prediction_time`; realized KO is read from the real
subsequent bars' high/low (never close).

| underlying | "as of" bars | raw observations (× 5h × 2dir × 8σ) |
|---|---:|---:|
| DAX | 811 | 64,880 |
| NDX | 812 | 64,960 |
| EURUSD | 786 | 62,880 |
| XAU | 666 | 53,280 |
| **total** | **3,075** | **246,000** |

**Walk-forward evaluation.** `PurgedWalkForwardSplit` (embargo = horizon),
run independently per `(underlying, horizon)` — never pooling two
underlyings' bar-index axes into one purge/embargo computation. Within each
fold, isotonic/Platt calibrators are fit per `(direction, regime_bucket)`
stratum on that fold's training observations only and applied to its test
observations — every number below is genuinely out-of-sample both in time
and in calibrator fit. This yields **150,000 out-of-sample predictions per
method** (raw, identity, isotonic, Platt) pooled across all 4 underlyings,
all 5 horizons, both directions, all 8 σ-buckets.

**Results, out-of-sample, pooled overall:**

| method | n | Brier | Cox intercept | Cox slope | ECE | mean signed error | absolute calib. error |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 150,000 | 0.1053 | −0.299 | 0.840 | 0.0160 | −0.0131 | 0.2347 |
| identity | 150,000 | 0.1053 | −0.299 | 0.840 | 0.0160 | −0.0131 | 0.2347 |
| isotonic | 150,000 | 0.1051 | −0.472 | 0.584 | 0.0130 | +0.0080 | 0.2004 |
| platt | 150,000 | 0.1062 | +0.384 | 1.204 | 0.0349 | +0.0074 | 0.2247 |

(identity is numerically raw with a `[1e-6, 1-1e-6]` clip; walk-forward
evaluated only as the floor every real candidate must beat, never itself
eligible for promotion.)

**Breakdown at the trading-relevant σ-buckets (where the ACTIONABLE gate's
`min_distance_to_barrier_sigma` actually bites):**

| method | σ | n | Brier | Cox intercept | Cox slope | ECE | mean signed error | abs. calib. error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 1.5 | 18,750 | 0.1168 | −1.483 | 0.226 | 0.0231 | **−0.0205** | 0.4058 |
| isotonic | 1.5 | 18,750 | 0.1165 | −1.677 | 0.085 | 0.0219 | **+0.0121** | 0.3838 |
| platt | 1.5 | 18,750 | 0.1161 | −0.641 | 0.579 | 0.0268 | **+0.0250** | 0.3814 |
| raw | 2.0 | 18,750 | 0.0581 | −1.895 | 0.308 | 0.0042 | **−0.0042** | 0.4676 |
| isotonic | 2.0 | 18,750 | 0.0582 | −2.401 | 0.093 | 0.0193 | **+0.0114** | 0.4197 |
| platt | 2.0 | 18,750 | 0.0579 | −1.331 | 0.539 | 0.0093 | **−0.0093** | 0.4789 |

Two things line up with the original W5 finding: raw's mean signed error is
negative at both buckets (realized KO rate below simulated P(KO) — the same
conservative, over-predicting direction as the 2026-09-12 study), and raw's
Cox slope is far below 1 (0.23 at σ=1.5, 0.31 at σ=2.0) — a new, more
granular confirmation that raw P(KO) is not just biased but too *extreme*
relative to the true probability at these distances, consistent with W5's
aggregate `|diff|` finding.

**Promotion decision: NO PROMOTION.** Reasoning (the rule in
`backtest/ko_calibration.py::_decide_promotion`, applied exactly as
implemented, no threshold loosened to force a result):

- **Platt** fails the basic bar before the tail-risk check even applies:
  its pooled Brier (0.1062) is *worse* than raw's (0.1053) — a calibrator
  that degrades discrimination while narrowing ECE is not a genuine
  improvement.
- **Isotonic** does improve pooled Brier/ECE/absolute-calibration-error
  over raw, but at both trading-relevant σ-buckets its mean signed error
  flips sign — from raw's −0.0205/−0.0042 (conservative: realized KO rate
  below predicted) to +0.0121/+0.0114 (predicted P(KO) now *below* the
  realized rate, the unsafe direction for the ACTIONABLE gate) — crossing
  the safety tolerance (§ below). The exact same reversal shows up for
  Platt at σ=1.5 (+0.0250).
- No candidate clears the promotion rule (improve OOS calibration over raw
  **without** materially worsening the conservative tail-risk bias at
  σ∈{1.5, 2.0}). `ranking/gates.py` keeps consuming raw P(KO), exactly as
  it did before this session and as W5 recommended.

This is a **negative result, reported as such** (CLAUDE.md rule 26): a
calibrator that is measurably better in aggregate is still the wrong choice
here, because it buys that improvement by eroding the specific safety
margin (over- rather than under-predicting KO risk close to the barrier)
that the rest of the system relies on. No threshold, gate or significance
level was loosened to manufacture a different outcome.

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

### Learning loop on real data

The evening chain (label → learn → position reevaluate) first processed
real data on 2026-09-17, and scaled up sharply on the second night as the
forward ledger matured:

| | 2026-09-17 (run 35287791715) | 2026-09-18 (run 35406176313) |
|---|---|---|
| labeled entries | 221 | 2,854 |
| of which knocked out | 1 | 29 |
| ambiguous / missing_data | 0 / 0 | 0 / 0 |
| posteriors_updated | 4 | 20 |
| models_reweighted | **0** | **0** |
| drift_events | 3 | 19 |
| position reevaluate: mails_sent | 0 | 0 |

`models_reweighted = 0` on both nights is consistent with sections 1–2:
there is no model with a measured advantage to re-weight toward. The drift
events are Page-Hinkley detections, which reduce weights but never delete a
model (rule 32). Knock-outs run at roughly 1% of labeled entries (29 of
2,854), which is a realized-outcome observation, not a validated model.

### Open risk: state growth is not yet bounded

The encrypted state artifact grew from 43.6 MB (2026-09-14) to 176.5 MB
(2026-09-19), though not monotonically — recent runs fluctuate between
162.0 and 176.5 MB, since each run's artifact reflects that run's own pack.
The database itself does grow steadily: 241.2 MB on 2026-09-17, 277.6 MB on
2026-09-18.

`db compact` has so far removed **0 rows on every run** — 0 of 517,548 on
2026-09-17, 0 of 625,263 on 2026-09-18 — and both are correct rather than
defects, for two different reasons:

- On 2026-09-17 nothing had aged out at all: with `keep_days: 5` and the
  first data written 2026-09-13, every row was still inside the window.
- On 2026-09-18 the window had moved past 2026-09-13, but that day carried
  exactly one scheduled run (34775837861, 18:50 UTC). Thinning reduces aged
  rows to one per (ISIN, UTC calendar day), and with a single scan every
  ISIN already had exactly one row for that day, so there was nothing to
  reduce. An earlier version of this section predicted thinning would bite
  from 2026-09-18; that prediction was wrong for this reason.

The first day carrying multiple scans (2026-09-14, five runs) leaves the
window on the night of 2026-09-19, which is the first run where thinning
can actually remove anything. The file-size drops those runs did show
(241.2 → 229.7 MB, 277.6 → 275.0 MB) came purely from the file rewrite.

Projected effect, measured against the local database (69,428 rows):
reducing aged rows to one per (ISIN, UTC calendar day) leaves 26,499 rows,
a 61.8% reduction, i.e. roughly 2.9 snapshot rows per ISIN per day. Applied
to the CI volume (≈21,000 products per scan, several scans per day) the
steady state after 90 days is still on the order of millions of rows, so
thinning alone slows growth rather than bounding it. This is an open item,
not a solved one.

---

## 6. Phase D: distributional evaluation infrastructure and baseline measurement

**Date of measurement:** 2026-09-19. **Motivation:** W4/W9 (§1-2) evaluate
every model exclusively on `p_up`/`brier`/`log_loss`/`ece` — binary
metrics of the directional call. `models/forecast.py::HorizonForecast`
already carries a full predictive distribution (`mean`, `sigma`,
`quantiles`, `expected_shortfall_05`, `uncertainty`) on every prediction,
but `backtest/walkforward.py::walk_forward_evaluate` discarded everything
except `p_up` before scoring. A model whose *distribution* is informative
but whose binary hit rate is unremarkable therefore could not win under
W4/W9's own evaluation, by construction — not because it lacks value, but
because the harness never looked at anything but the coin-flip call. This
section reports the fix (a proper, quantile-based distributional scoring
harness) and the first measurement run through it.

### 6.1 What was built

- **Primary target:** `y_h(t) = ln(P_t+h / P_t) / (sigma_t * sqrt(h))`,
  `sigma_t` the causal EWMA daily volatility "as of" `t`
  (`features/volatility.py::causal_ewma_sigma`/`ewma_volatility`, only bars
  with `available_at <= t` ever enter it — covered by a dedicated
  look-ahead test). `features/volatility.py::normalized_horizon_target`
  vectorizes this over a whole bar series. Used only as an *internal*
  fitting target by the three new baseline models below (§6.2); every
  model still emits `HorizonForecast` in ordinary return units (mean/sigma
  in log-return space), converted back via the *current* `sigma_t` before
  returning — the payoff/path simulation, gates and every existing
  consumer are completely unaffected.
- **New metrics (`backtest/metrics.py`):** `pinball_loss`/`mean_pinball_loss`
  (quantile loss), `crps_from_quantiles`/`mean_crps_from_quantiles` (the
  standard quantile-averaging CRPS approximation, `CRPS ≈ 2 * mean(pinball
  loss over the quantile grid)` — Gneiting & Raftery 2007; Bracher et al.
  2021), `pinball_loss_by_level` (per-quantile breakdown), and
  `interval_coverage` (empirical coverage of a `[lower, upper]` interval
  against its nominal level — verified in tests to correctly flag a
  deliberately-too-narrow interval).
- **`backtest/walkforward.py::WalkForwardResult`** now additionally
  collects `oos_forecasts` (the full `HorizonForecast` per out-of-sample
  point, not just `p_up`) and reports `crps`, `pinball_by_quantile`,
  `coverage_90` (nominal 90%, `[q05, q95]`) and `coverage_50` (nominal 50%,
  `[q25, q75]`) per `(model, horizon)`. `p_up`, `brier`, `log_loss`, `ece`,
  `hit_rate` and `psr` are all still computed exactly as before — purely
  additive, verified by a dedicated regression test.
- **`storage/schemas.py::WalkforwardResultRecord`** and the
  `walkforward_results` table gain `crps`, `pinball_loss`, `coverage_90`,
  `coverage_50` via the existing additive `_migrate_table_columns`
  migration (nullable/empty-default on any pre-Phase-D row).
- **Four pre-registered baselines**, measured in this fixed order before any
  challenger, per `models/baselines.py` (b-d) and the existing
  `models/directional.py::NullModel` (a):
  - (a) unconditional empirical distribution — `NullModel`, unchanged.
  - (b) `RegimeConditionalEmpiricalModel` — the empirical distribution of
    `y_h`, conditioned on a trailing, pre-registered 3-bucket volatility
    regime (rolling tercile of causal EWMA vol); falls back to the
    unconditional distribution when the current regime has too few
    training samples.
  - (c) `RegularizedLinearLocationModel` — ridge regression of `y_h` on a
    single, independently re-derived causal trend z-score (same shape as
    the protected `tsmom_horizon_norm_v1` score, never imported or
    modified), residual quantiles for the shape.
  - (d) `RobustLocationScaleModel` — robust (median/MAD) location-scale fit
    of `y_h`, mapped onto a fixed-shape (`df=5`, pre-registered, never
    fit) Student-t distribution for quantiles/expected shortfall.
  - GAM and a from-scratch LightGBM challenger were **not** built in this
    pass — per the pre-registered priority order, baselines (a)-(d) are
    measured first, and doing so (plus building the harness itself) filled
    the available session budget. `models/quantile.py::RidgeReturnModel`
    already exists and is exercised indirectly through
    `LogisticDirectionModel`, but was not separately re-measured against
    the new CRPS/pinball/coverage metrics in this pass — an open item, not
    a negative result.

### 6.2 Measurement

Same methodology as W4 (§1): `backtest.walkforward.walk_forward_evaluate`,
expanding window, `min_train=750`, `step=21`, `embargo=horizon_days`,
horizons `(3, 5, 7, 10, 14)` trading days, `yfinance` daily bars for
`DAX`/`ESTX50`/`SPX`/`NDX` through 2026-09-18 (4881-6200 bars per
underlying). **Full scope: 4 underlyings × 5 horizons × 4 models = 80
cells, all measured (no reduction).** Total walk-forward compute: 1248s
(~21 min).

**Aggregate results (mean over all 20 `(underlying, horizon)` cells per model):**

| model | brier | crps | hit_rate | coverage_90 (nom. 0.90) | coverage_50 (nom. 0.50) | ece |
|---|---:|---:|---:|---:|---:|---:|
| `null` (a) | 0.2427 | 0.01511 | 0.586 | 0.924 | 0.548 | 0.0276 |
| `regime_conditional_empirical` (b) | 0.2437 | 0.01465 | 0.582 | 0.894 | 0.509 | 0.0324 |
| `regularized_linear_location` (c) | 0.2428 | 0.01461 | 0.584 | 0.887 | 0.503 | 0.0277 |
| `robust_location_scale_t` (d) | 0.2428 | 0.01476 | 0.586 | 0.931 | 0.538 | 0.0288 |

**CRPS wins vs. null, per baseline: 20/20 cells for (b), (c) and (d) alike**
(every one of DAX/ESTX50/SPX/NDX × 3/5/7/10/14d), a consistent -2.5% to
-3.5% mean relative CRPS improvement (min -0.99%, max -5.61% across
cells; `regularized_linear_location` largest mean improvement at -3.45%).
**Brier wins vs. null, per baseline: 3/20 (b), 8/20 (c), 9/20 (d)** — flat
to slightly worse on average (mean Δbrier +0.0010, +0.00009, +0.00004
respectively), consistent with W4/W9's own finding that this evaluation
axis shows essentially nothing here. **This is exactly the pattern the
Phase D hypothesis predicted:** a real, monotonically consistent
improvement in the *distribution* (CRPS, every single cell) that a
binary/Brier-only evaluation cannot see at all.

Per-quantile pinball loss (mean over all 20 cells) improves for every
baseline at every quantile level versus `null` except `q75`
(`robust_location_scale_t` alone is very slightly worse there, 0.00853 vs.
0.00877) — the tails (`q05`/`q95`) show the largest relative gains,
consistent with volatility-conditioning mattering most where the
unconditional model's fixed-width distribution is furthest from today's
actual regime.

### 6.3 Interpretation — not a directional edge, and not yet promotable

The most parsimonious explanation for a *consistent* CRPS improvement with
a *flat* Brier is **heteroskedasticity awareness, not forecasting skill**:
`NullModel` pools raw historical returns into one fixed-width distribution
per horizon regardless of current volatility; all three new baselines
rescale by the *current* causal `sigma_t` before returning quantiles. Since
volatility clustering is a well-established, uncontroversial property of
daily index returns (not an edge), a model whose predictive interval width
tracks current volatility will mechanically score better on a proper
scoring rule like CRPS than one that always reports the unconditional
average dispersion, with no directional information required at all. The
coverage numbers support this reading: coverage_90/50 are within roughly
1-4 percentage points of nominal for every baseline (regime/ridge trend
slightly narrow — 88-89% actual vs. 90% nominal, i.e. mildly overconfident;
`robust_location_scale_t`'s Student-t tails trend slightly wide, 93%) —
small, explainable miscalibrations, not a directional bias.

**Per `SIGNAL_REGISTRY.md` §1.10 and this project's own governance rule
("kein Modell befoerdern, nur weil seine Likelihood besser ist"), a CRPS
improvement alone — like a Brier improvement alone — is explicitly
insufficient for promotion.** Promotion requires a reproducible advantage
in downstream net Expected Value, after realistic Turbo costs, path/KO
risk and multiple-testing deflation (`GOVERNANCE.md` §2). **No such
net-EV measurement was performed in this pass** — that requires running
the full product/pricing/payoff simulation per candidate, a materially
larger undertaking than building and exercising the distributional
evaluation harness itself, and out of scope for this workstream.

**Recommendation:** none of (b)/(c)/(d) is promoted or added to the live
scan ensemble (`models.forecast.build_default_models`/`pipeline/scan.py`)
— they are wired only into the `turboedge backtest` measurement/persistence
path. The CRPS result is a genuine, reproducible, 20/20-consistent finding
worth carrying forward — specifically as a candidate for **volatility-aware
position sizing / interval width in the existing EV pipeline**, which is a
plausible mechanism for the "no measured edge" conclusion of §1-2 and §6
Conclusion below to eventually change without needing any new directional
signal — but until a net-EV measurement is run on this same infrastructure,
the honest conclusion is: **a measured distributional improvement,
explicitly not yet a trading edge.** No thresholds, gates or the protected
`tsmom_horizon_norm_v1` score were touched to produce this result.

---

## 6.5 Does the pre-EV cost prefilter discard promising products? (2026-09-20)

`ranking.yaml`'s `scan_filter.max_candidates_per_bucket: 25` simulates only
the cheapest 25 candidates per (direction, leverage_bucket) group, ranked by
`cost_rank_score` — a *cost* measure, not a return forecast. The suspicion
this tests: since ACTIONABLE requires `lcb_ev > 0` and `lcb_ev` is only ever
computed for simulated candidates, a product with a better expected value
but higher costs would never be measured at all, and "no trade" would then
be an artifact of the filter rather than a property of the market.

**Scope — this is a narrow measurement.** It reads `candidate_sets` in the
local state database: six scan runs from 2026-09-14 (two of them after the
close), DAX and NDX only, 36,086 WATCH candidates of which 2,349 carry an
`lcb_ev`. It is not a CI measurement and not a re-run; nothing was changed
or re-simulated. Under §11.1's W4 precedent (a *measurement* of existing
behavior is not a "Research-Änderung") it consumes no trial_id. Raising
`max_candidates_per_bucket` on the strength of it would be an adaptation and
would need one (2026Q4: 3 of 6 used).

### The filter behaves exactly as documented

| cost rank within its group | candidates | EV-evaluated |
|---|---|---|
| 1–25 | 2,124 | 2,124 (100%) |
| >25 | 33,962 | 225 (0.7%) |

The 225 are the shadow sample (`shadow_sample_per_stratum: 3`), drawn from
the *full* pool independently of cost rank — Spec §25's selection-bias
protection. That makes them an unbiased probe of precisely what the filter
discards, recorded on every run since the feature shipped.

### The discarded products are not better

| | n | best `lcb_ev` | p99 | median |
|---|---|---|---|---|
| filtered (cheapest 25) | 2,124 | **−0.00682** | −0.00969 | −0.04744 |
| shadow (random) | 225 | **−0.00749** | −0.00866 | −0.04975 |

The best randomly drawn candidate is *worse* than the best filtered one.
The shadow sample covers all 16 strata, and its ten best entries are all
DAX short, leverage bucket 2–3 — the same corner the filtered winners come
from — at cost ranks 68, 70, 72, 137, 156, 164, i.e. just behind the cutoff
rather than deep in the discarded field.

### Why the naive correlation misleads

Across all evaluated candidates, `corr(cost_rank_score, lcb_ev) = −0.15`
(r² = 0.022), which reads as "cost rank says almost nothing". That number is
Simpson's paradox: it mixes leverage buckets with structurally different
cost levels. The filter only ever ranks *within* one (run, direction,
bucket) group, and within groups the relationship is positive — Spearman
median +0.32, and quartile medians are monotone (−0.04664 / −0.04715 /
−0.04769 / −0.04946 from cheapest to dearest quartile). It is a weak
ordering, not a wrong one: 38 of 87 groups still rank negatively.

### Conclusion: the prefilter is not why ACTIONABLE is 0

Relaxing it cannot produce a trade proposal. The gap from the best measured
candidate to break-even is 0.00682; the entire difference between the
filtered and the random pool is 0.0007, an order of magnitude smaller. A
negative result, and a useful one: a structural suspicion about the
pipeline is ruled out, leaving the measured negative expected value in
sections 1–2 and 6 as the sole reason for "no trade".

**What this does not establish.** 225 of 33,962 is a 0.66% sample, and the
question is about an extreme value, where a small sample is weakest. This
exonerates the filter; it does not prove it optimal. Nor does it cover
EURUSD/XAU (no local candidates) or any date other than 2026-09-14.

## 6.6 Does the spot/quote timing offset inflate the measured EV? (2026-09-20)

Roughly a third of priced candidates show an intrinsic value *above* their
own mid quote (19,764 of 64,970) — economically impossible, since a turbo
below intrinsic would be arbitrage. The cause is a timing offset: the spot
used for `intrinsic_value` is not from the same instant as the issuer's
quote. The concern this tests is the worst case for section 7's central
claim: if a too-low spot inflates a short turbo's intrinsic value, and that
feeds the expected value, then the measured best candidate (−0.00682) would
be *optimistic*, and "no trade" would rest on numbers that are themselves
wrong in the flattering direction.

Same scope caveat as §6.5: local `candidate_sets`, six runs from
2026-09-14, DAX/NDX, no re-simulation, no trial_id consumed.

### The offset is concentrated in the *worst* candidates, not the best

Relative offset is `(intrinsic − mid) / ask`.

| | n | offset (median) | mean absolute offset |
|---|---|---|---|
| best 100 by `lcb_ev` | 100 | 0.00003 | **0.00071** |
| all others | 2,572 | 0.00027 | **0.01201** |

The top 100 carry a seventeen-fold *smaller* pricing inconsistency than the
rest. Individually, the twenty best candidates' offsets all lie between
−0.00099 and +0.00139, four of them negative — no systematic direction.

By EV quartile (worst to best), mean offset runs 0.01917 / 0.00574 /
0.00502 / 0.00307. So the `corr(offset, lcb_ev) = −0.39` (r² = 0.15)
measured across all candidates is carried entirely by the bad end:
products with an unclean price basis earn poor EV. That is the gate working
as intended, not a bias in favour of the winners.

### It also cannot reach the EV by construction

Entry is the ask, never the mid (`net_return = exit_bid / entry_ask − 1`,
`simulation/payoff.py`; Master Spec rule 11). The quantity measured here is
intrinsic-vs-*mid*, which enters the net return only indirectly through
`issuer_margin` — and `pipeline/scan.py` floors that at zero
(`max(issuer_margin_pct, 0.0)`), so a negative margin cannot cheapen a
candidate at all.

### Conclusion: the measured EV is not flattered by this defect

The −0.00682 is neither optimistic nor an artifact. The offset remains a
genuine data-quality defect worth fixing on its own merits (it is what
`implied_spot_deviation` already flags into DATA_QUALITY, where the mean
absolute offset is 0.00440 — the integrity check is catching the visible
cases), but it is not an explanation for ACTIONABLE = 0.

**What this does not establish.** It measures the offset's *association*
with EV, not a counterfactual: nothing here re-prices the candidates with a
correctly synchronised spot and re-runs the simulation. That would be the
only way to bound the effect exactly, and it needs price history the local
state database does not have (399 bars for DAX/NDX against
`lookback_days: 750`; no bars at all for EURUSD/XAU).

### Taken together with §6.5

Three structural explanations for "no ACTIONABLE" that would have been
defects rather than findings are now measured and ruled out: a blocking
gate (every evaluated candidate carries `lcb_ev_not_positive`, and no other
watch reason — §5), a prefilter discarding better products (§6.5), and a
pricing basis biased in the flattering direction (this section). What
remains is the measurement in sections 1–2 and 6: no forecast has an edge
that survives a turbo's costs.

## 6.7 W12-A: does Cboe volatility state add out-of-sample information? (2026-09-25)

**Trial:** `TR-2026Q3-abd750` (kind `feature`, status `experimental`).
**Question (H0-1, Research Wave 2 §12):** do Cboe volatility-state features
add forecast value over an own-history base, out of sample, after
multiple-testing correction and realistic Turbo costs?

**Answer: NO. DO NOT PROMOTE.**

### Prior

`VixTermStructure` (trial `W9-2026Q3-004`) already tested two of these
features — `vix9d_over_vix_minus_1` and `vix_over_vix3m_minus_1`, sourced
from yfinance — and is recorded `dormant`: Brier +0.0091 worse than null,
best edge 1.85 bp against SIGNAL_REGISTRY.md §1.10's 10 bp bar, 0/14 cells
surviving BH (§2 above). W12-A is broader (six official series, ten
features, VVIX/OVX/GVZ added) but that negative result was the honest prior.

### Data

Cboe's own daily-prices CSV endpoint
(`cdn-api.cboe.com/api/global/us_indices/daily_prices/<INDEX>_History.csv`),
chosen over yfinance because it is a published contract rather than an
inferred one, and because it is *slower*: on 2026-09-25 17:00 UTC it carried
data only through 09-24 while yfinance already served 09-25 — i.e. it
reflects what was genuinely published. `robots.txt` permits the path
(`/book/` and `*market_statistics/volume_reports/` are the only disallows,
no crawl delay); the endpoint answers 200 to the project's honest user
agent. Nothing was bypassed.

| Series | rows | from | shape |
|---|---:|---|---|
| VIX | 9,281 | 1990-01-02 | OHLC |
| VVIX | 5,112 | 2006-03-06 | close |
| VIX9D | 3,955 | **2011-01-04** | OHLC |
| VIX3M | 4,281 | 2009-09-18 | OHLC |
| OVX | 4,279 | 2009-09-18 | close |
| GVZ | 4,279 | 2009-09-18 | close |

83,723 observations persisted to `external_observations`. VIX9D binds the
common window to 2011.

**Availability model:** a trading day's close is published the following
day, so `available_at` = observation day + 1 at 00:00 UTC, and a prediction
at the close of day *t* uses Cboe data through *t−1*. Deliberately
conservative (the real publication is ~20:15 UTC the same day); it cannot
leak and covers the 07:40 UTC scan. Enforced by
`features/availability.py::assert_information_available_at_prediction`,
whose tests construct leaking inputs on purpose and assert they raise.

### Method

Target `y = ln(P[t+h]/P[t]) / (sigma_t·sqrt(h))`, `sigma_t` a causal EWMA
(λ=0.94). BASE features: own trend z-scores (21/63/126d), realised vol, vol
change. BASE+CBOE: identical plus the ten features. Ridge (α=1) for the
conditional mean, empirical training-residual quantiles for the
distribution. `PurgedWalkForwardSplit(horizon=h, embargo=h, min_train=750,
step=21)` with real label windows `[t, t+h]`, so overlapping labels are
purged rather than straddling the boundary. Four underlyings × five
horizons = 20 cells, ~3,000 OOS observations each (60,017 paired total).

### Results

| Metric | Outcome |
|---|---|
| CRPS | **20/20 cells worse** (+0.32% to +5.71%) |
| Brier | **20/20 cells worse** |
| BH (α=0.10) | **6/20 significant — all AGAINST Cboe, 0 in favour** |
| 90% interval coverage | 0.887 → 0.883 (nominal 0.900) |
| Economic (signal-direction return) | CBOE better in **7/20**, median Δ **−3.60 bp** |
| Best edge, any cell | BASE +59.18 bp; **CBOE +39.87 bp** |
| Realistic Turbo cost band | 50–150 bp / 7d — **0 CBOE cells clear even the low end** |

**Two measurement errors found and corrected before these numbers, both
mine, both recorded because they change how the result must be read:**

1. Rolling windows were first computed on the wide frame, whose union index
   reaches back to 1990 where VVIX (from 2006) is absent — a 252-day window
   returns NaN if one NaN sits inside it. This silently cut the CBOE sample
   by ~60% (3,347 → 1,295 OOS). Fixed by computing on each series' own
   gap-free history. The uncorrected run reported +1.17% to +16.46% CRPS
   degradation; the true figure is +0.32% to +5.71%. **The first run
   overstated the damage by roughly a factor of three.**
2. BASE and BASE+CBOE initially ran on different samples (the CBOE variant
   starts later). Both now start at the first row where all CBOE features
   exist. Residual difference 0.03–3.37% (mean 1.6%) is the US/EU holiday
   calendar offset, measured at 2.8% independently; the paired analysis uses
   common observations only.

**A caveat on the economic numbers, stated because it cuts against reading
them favourably:** the US indices show uniformly positive edges (NDX up to
+59 bp) and the European ones mixed-to-negative. That is the signature of
secular upward drift with a mostly-long signal, not forecast skill —
CLAUDE.md rule 11 forbids reading it as alpha. The only cell above the cost
band's low end belongs to BASE, not to Cboe.

### Conclusion

Adding Cboe volatility state makes the distributional forecast worse on
every cell, worse on Brier on every cell, and significantly worse on six
after FDR correction — with **not one cell significantly better**. The
economic measure adds nothing: median −3.60 bp, and no Cboe cell reaches
even the low end of the cost band. H0-1 is **not rejected**.

This is a **negative result, reported as such** and consistent with the
prior W9 finding rather than contradicting it. Nothing is promoted; the
adapter, schema, availability guard and feature builder remain in the tree
as measurement infrastructure for the remaining W12 families, which are
separate pre-registered hypotheses and are not started by this trial.

---

## 7. Conclusion

Across every avenue tested so far — the protected TSMOM baseline mapped to
a full predictive distribution (W4), six independently pre-registered
challenger signal families covering sizing, regime-gating, reversal,
volatility-term-structure, genuine cross-asset information, and calendar
seasonality (W9), and four pre-registered distributional baselines measured
under a new CRPS/pinball/coverage harness (Phase D, §6) — **no model has a
measured, multiple-testing-deflated out-of-sample advantage in net
Expected Value over the null (unconditional) benchmark, and no tested
effect survives realistic Turbo trading costs even before deflation.**
Phase D's baselines *do* show a consistent, 20/20-cell CRPS improvement
over the null model (§6.2) — a real, distributional (not directional)
finding — but per this project's own promotion rule, a proper-scoring-rule
improvement alone is not evidence of a net-EV edge, and none was measured.
The knock-out path model (W5) is, after calibration work,
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
