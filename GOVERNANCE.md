# Research Governance Framework

**Version 1.0.0**  
**Effective Date:** 2026-09-10

This document defines research governance policies for TurboEdge-DE's iterative model development, feature testing, and risk management.

---

## 1. Trial System and Adaptation Budget

### 1.1 Trial ID Assignment

Every research change—feature variant, threshold adjustment, model hyperparameter, sizing rule—receives a unique `trial_id`.

### 1.2 Quarterly Adaptation Budget

**Max changes per quarter: 6**

Each **test** increments the effective number of tested hypotheses,
triggering multiple-testing inflation penalties. Not each promotion.

This sentence previously read "each promotion from experimental to live",
and that was wrong in a way that had consequences (corrected 2026-09-26
after external review). Whether a hypothesis is later promoted is
irrelevant to multiple testing: the inflation is created the moment a
hypothesis is tested and its result is seen. Counting only the promoted
ones would permit running arbitrarily many experiments and charging the
budget for none of them, which is precisely the researcher-degrees-of-
freedom this budget exists to bound.

A trial is therefore charged to the quarter in which it is **opened and
measured**, regardless of its outcome and regardless of which quarter's
budget has room.

### 1.3 Tracking

Logged in `state/registry/trials.json`:

```json
{
  "trial_id": "2026Q3_001",
  "hypothesis": "Add EWMA volatility feature",
  "implementation_date": "2026-09-15",
  "status": "experimental|promoted|dormant|rejected",
  "promotion_date": "2026-10-01",
  "out_of_sample_net_ev_delta": 0.0012,
  "deflated_zscore": 1.8,
  "effective_sample": 412,
  "notes": "..."
}
```

---

## 2. Ladder Rule – Minimum Improvement Thresholds

Promoted changes must exceed:

### 2.1 Zscore Threshold (after deflation)

- **z ≥ 1.645** (10% significance after Benjamini-Hochberg FDR)
- **Deflated Sharpe Ratio (DSR) ≥ 0.6**
- **Probabilistic Sharpe Ratio (PSR): P(μ > 0) ≥ 95%**

### 2.2 Minimum Effect Size

- **Absolute net-EV improvement ≥ 0.0010** (10 bps) per forward trade
- **Win-rate improvement ≥ 2%** (e.g., 55% → 57%)

Small improvements below this ladder are tracked but remain experimental.

---

## 3. Multiple Testing Correction

### 3.1 Benjamini-Hochberg FDR

Applied to all trial promotions.

**False Discovery Rate (FDR) alpha:** 0.10

Only promotions with adjusted p-value < 0.10 advance to live.

### 3.2 Deflated Sharpe Ratio (DSR)

Accounts for:
- Overlapping labels (purged CV effective sample size)
- Multiple hypothesis testing
- Backtesting degrees of freedom

### 3.3 Probabilistic Sharpe Ratio (PSR)

Minimum posterior probability that true strategy mean > 0: **95%**

Calculated via Bayesian conjugate prior on returns.

---

## 4. Research Ruin Metric

### 4.1 Definition

**Portfolio loss threshold (X):** 20% of research capital over 12 months  
**Tolerated probability:** 5%

### 4.2 Calculation

Estimated via:
1. Historical return distribution (empirical)
2. Cluster-conditional correlations (rolling window)
3. Position sizing limits applied
4. Forward-looking simulation (1000 paths)

### 4.3 Monitoring

Recalculated weekly. If P(loss > 20% | 12m) exceeds 5.5%, capital allocation reduced until probability returns below 5%.

---

## 5. Sizing Constraints (§32)

### 5.1 Kelly Fraction

**Fractional Kelly: 0.25 × Kelly**

Never exceed 25% of full Kelly to account for estimation uncertainty.

### 5.2 Position Size Caps

| Cap                      | Value  | Notes                                |
|--------------------------|--------|--------------------------------------|
| max_position_fraction    | 2%     | Max single product per research capital |
| max_cluster_fraction     | 5%     | Max single correlation cluster       |
| max_total_notional       | 100%   | Sum of all open positions ≤ capital  |
| max_ko_loss_contribution | 8%     | Total KO scenario loss budget        |

### 5.3 Uncertainty Adjustment

Sizing reduced when:
- Model uncertainty (σ_forecast) > historical average
- Regime shift detected (drift score > threshold)
- Data quality degraded (source health < 0.8)

---

## 6. Promotion / Demotion Rules

### 6.1 Promotion (Experimental → Live)

**Criteria:**
- Trial passed ladder rule (DSR, PSR, FDR)
- Out-of-sample net-EV improvement confirmed
- Ablation study shows isolated positive contribution
- No adverse effect on existing champion
- Effective sample ≥ 100 distinct outcomes

**Action:** Feature added to next ensemble weights, logged in registry.

### 6.2 Demotion (Live → Dormant)

**Triggers:**
- Rolling 4-week Sharpe < 0.0 (net loss period)
- Drift detected: feature distribution shift z-score > 3
- Regret > 50 bps average (consistently choosing worse products)
- Calibration drift: ECE > 0.15

**Action:** Model weight set to 0.01 (minimum); kept in ensemble for recovery if regime shifts. Logged as dormant trial.

### 6.3 Archival (Dormant → Rejected)

After 2 quarters with 0 promotions and confirmed regime change unrelated to original hypothesis, move to failed_hypotheses.json.

---

## 7. Version Control & Registry

### 7.1 Signal Registry Versioning

Each signal or threshold change increments version:

```
tsmom_horizon_norm_v1  (current: threshold 0.5, PROTECTED)
tsmom_horizon_norm_v2  (if threshold changed: new trial_id required)
```

Protected baselines never revert; new versions created instead.

### 7.2 Configuration Versioning

`config_hash` stored in every prediction for reproducibility.

Config changes logged in `CHANGELOG.md` with timestamp and trial_id.

---

## 8. Version History Table

| Version | Release Date | Phase       | Key Changes                                         |
|---------|--------------|-------------|-----------------------------------------------------|
| 1.0.0   | 2026-09-10   | 0 + 1       | Initial: product adapters, cost engine, TSMOM base |

---

## 9. Research Ledger

Central append-only table: `ledger/forward_ledger.parquet`

**Mandatory fields per prediction:**
- signal_id, signal_version_hash
- trial_id, prediction_time, underlying, direction, horizon
- regime_bucket, cluster_id
- selected_wkn, issuer, entry_ask, financing_level_entry
- predicted_return, p_profit, p_ko, lcb_ev, uncertainty
- git_commit, config_hash, data_snapshot_hash
- (later) realized_pnl, mfe, mae, ko_hit, ambiguous_path

Not only ACTIONABLE candidates stored; stratified shadow sample of rejected candidates included to reduce selection bias.

---

## 10. Data Quality Gates

### 10.1 Source Health Requirements

Before scan proceeds:

- **Availability:** ≥ 95% (data present for last 20 days)
- **Freshness:** ≤ 1 hour staleness for critical sources
- **Missingness:** < 5% nulls in key fields
- **Schema consistency:** 0 schema violations in last 100 records
- **Cross-source agreement:** ± 2% on overlapping products

If any source fails: flag in report, apply quality discount to products, email alert if enabled.

### 10.2 Product Integrity Checks

- bid ≤ ask
- ratio > 0 & not NaN
- barrier valid (not knocked out; distance > 1% for active)
- financing_level consistent with history (no >10% daily jump without flag)
- quote_timestamp ≤ observation_time

Failed products marked DATA_QUALITY; not included in scoring.

---

## 11. Research Trials This Quarter (2026 Q3)

Trials consumed against the §1.2 quarterly adaptation budget (max 6/quarter)
in 2026 Q3. Full numbers: `docs/measured_results.md`; per-signal detail:
`SIGNAL_REGISTRY.md` §3; raw registry: `state/registry/failed_hypotheses.json`.

### 11.1 Trials consumed

- **W4 (2026-09-12):** measurement of the protected TSMOM baseline's
  distributional mapping (`TsmomForecastModel`) and one reduced-scope
  `LogisticDirectionModel` data point against the `NullModel` benchmark.
  This was a **measurement of existing signals**, not a new feature
  promotion — it does not itself consume a trial_id under §1.1 ("jede
  Research-Änderung", i.e. a change, gets a trial_id), but it is the
  evidentiary basis every W9 trial below was pre-registered against.
- **W9 (2026-09-13): 6 trial_ids consumed**, one per pre-registered
  signal family (§1.1's unit is the hypothesis/feature, not each
  underlying/horizon cell it is measured on):

  | trial_id | feature | quarter | status |
  |---|---|---|---|
  | W9-2026Q3-001 | `voltarget_tsmom` | 2026Q3 | dormant |
  | W9-2026Q3-002 | `lowvol_regime_trend` | 2026Q3 | dormant |
  | W9-2026Q3-003 | `reversal_short_horizon` | 2026Q3 | dormant |
  | W9-2026Q3-004 | `vix_term_structure` | 2026Q3 | dormant |
  | W9-2026Q3-005 | `cross_asset_leadlag` | 2026Q3 | dormant |
  | W9-2026Q3-006 | `seasonality_turn_of_month` | 2026Q3 | dormant |

  **6 of 6 trials in the 2026Q3 budget are consumed by W9 alone** (§1.2:
  max 6/quarter). No further new signal-family trials should be opened
  this quarter without either a documented regime-change justification
  (§6.3 / Master Spec §23) or explicit governance review to raise the
  budget.

  **Corrected 2026Q3 standing: 11 of 6 (2026-09-26, after external
  review).** Two separate accounting errors, in opposite directions:

  | Trial | Opened | Charged to | Correct |
  |---|---|---|---|
  | `W9-2026Q3-001..006` | 2026-09-13 | 2026Q3 | yes |
  | `PD-2026Q4-001..003` | 2026-09-19 | 2026**Q4** | no — Q3 work |
  | `TR-2026Q3-abd750` (Cboe) | 2026-09-25 | 2026Q3, override | charged, but on invalid grounds |
  | `TR-2026Q3-31e266` (CFTC) | 2026-09-25 | 2026Q3, override | same |

  The Cboe/CFTC overrides were logged "on the grounds that a measurement
  which promotes nothing does not inflate the multiple-testing count".
  §1.2 now states why that is wrong.

  The Phase D trials are the same error mirrored: opened 2026-09-19, which
  is Q3, but labelled and charged to Q4 — pre-spending a future quarter for
  work already done, and understating Q3. An external review read this as
  the correct handling; it is not, and the trial ids record the opening
  date that shows it.

  All five were genuinely opened and measured in Q3, so all five count in
  Q3. The honest record is that the quarter ran to **11 of 6**. The trial
  ids are left unchanged — renaming them now would rewrite the audit trail
  to make the books look tidy, which is the opposite of what this section
  is for.

  **Q3 is closed to new trials.** The next research question — including
  the approved `RO-MAE-PREDICTION` / `RO-MFE-PREDICTION` — opens against
  the 2026Q4 budget on 2026-10-01, with Q4 starting at 0 of 6, not 3.

### 11.2 Effective number of hypotheses tested (for deflation)

The unit is the (family, underlying, horizon) **cell actually measured** —
not the trial_id count, and not the pre-registered count (§3.1: "the actual
count measured, which may be smaller than what was pre-registered").

**Cumulative 2026Q3 research universe: 180 cells.**

| Wave | Cells | Deflated against |
|---|---|---|
| W9 challenger signals | 80 | 80 (own wave) |
| W12-A Cboe volatility state | 20 | 20 (own wave) |
| W12-D CFTC positioning | 20 | 20 (own wave) |
| Phase D distributional baselines | 60 | 60 (own wave) |
| **Total tested on overlapping data** | **180** | — |

**Each wave was deflated only within itself, and that understates the
penalty** (recorded 2026-09-26 after external review). These waves run over
the same underlyings and largely the same sample period, so the research
universe a later result must survive is the cumulative 180, not the 20 or 60
of its own wave. No conclusion in `docs/measured_results.md` changes as a
result — every W9, W12-A and W12-D cell failed against the *smaller*
denominator, and a larger one only makes them fail harder.

The one place it does matter is the Phase D distributional baselines, the
only measured improvement in the repository (CRPS better in 20/20 cells per
family). Their deflation used `n_trials=60`. Against the cumulative 180 the
penalty is larger, and **their DSR has not been recomputed on that basis**.
Until it is, "better in 20/20 cells" should be read as a within-wave result.

**Required from here:** deflation denominators are cumulative across all
cells measured on overlapping data in the same quarter, not per wave. A new
wave states the running total it deflated against.

### 11.3 Ladder Rule applied — no promotions

Applying §2 (Ladder Rule) to every trial above:

- **§2.1 (z-score / DSR / PSR):** 0 of 80 measured cells cleared
  Benjamini-Hochberg FDR at α=0.10 on either Brier-score or
  signal-direction-return difference vs. null. DSR was ≈1.000 for every
  family, but this is explicitly not informative in isolation (see
  `docs/measured_results.md` §2 — it is a sample-period artifact shared
  by the null model itself, not evidence of skill).
- **§2.2 (minimum effect size, ≥10 bps / ≥0.0010 net-EV):** the single
  best effect across the entire 2026Q3 sweep is **1.85 bp**
  (`vix_term_structure`, its most favorable cell) — roughly 5× below the
  10 bp minimum, and every family's own best single cell
  (`incremental_net_ev` in `state/registry/failed_hypotheses.json`) is
  1–2 orders of magnitude below the 0.0010 floor.
- **Conclusion: 0 of 6 W9 trials promoted.** None reached even the
  ordinary ladder, let alone the stricter mean-reversion-specific bar
  pre-registered for `reversal_short_horizon` (Master Spec §3.4: BH-sig.
  AND deflated z≥2.0 AND DSR≥0.6 — moot, since it did not clear the
  ordinary bar either). Per §6.2/§6.3, all six are logged `dormant` (not
  yet eligible for archival to "rejected" — that requires 2 quarters with
  0 promotions and a confirmed, unrelated regime change, per §6.3).

### 11.4 Registry reference

All six 2026Q3 trial outcomes are recorded in
`state/registry/failed_hypotheses.json` (`turboedge.learning.failed_hypotheses`,
never hand-edited), each with `trial_id`, `feature`, `status: "dormant"`,
`incremental_net_ev` (best single cell, not mean), `effective_sample`
(summed OOS prediction count across that family's measured cells,
34,542–48,304 per family), and `horizons`. Master Spec §23 ("Positive
Muster verstärken, negative Muster nicht löschen" / Rule 31): these
entries are retained, not deleted, so none of the six is blindly retried
next quarter without a documented regime-change justification.

### 11.5 Phase D distributional baselines — attributed to 2026Q4

Phase D (`docs/measured_results.md` §6) built and measured **three new
model families** on the normalized horizon target: the
`RegimeConditionalEmpiricalModel`, the `RegularizedLinearLocationModel`
and the `RobustLocationScaleModel`. They are registered as
`PD-2026Q4-001`, `-002` and `-003` in `research_trials`, all with status
`dormant`.

**Why these consume trial_ids at all.** §11.1 records that W4 did *not*
consume one, because it only measured signals that already existed. That
precedent does not extend here: Phase D did not re-measure an existing
model, it introduced three new ones. Under §1.1 ("every research
change — feature variant, threshold adjustment, model hyperparameter,
sizing rule — receives a unique `trial_id`") a new model family is
exactly the kind of change the budget exists to count. Treating them as
budget-free because none was promoted would understate `N_effective` in
every later deflation, which is the specific failure mode §3 is built to
prevent.

**Why 2026Q4 and not 2026Q3.** The 2026Q3 adaptation budget was already
fully consumed (6/6) by W9 before Phase D began (§11.1). The honest
options were to attribute the three trials to the next quarter, or to
raise the budget. **The budget was not raised** — §1.2's limit of 6 per
quarter is unchanged, and no exception was granted. Attribution to
2026Q4 is therefore the conservative reading: it counts the trials in
full, and it does so in the first quarter that actually has room for
them.

**What this does not mean.** None of the three is promoted, none is in
the live scan ensemble (`models.forecast.build_default_models` is
unchanged), and all three remain `dormant`. A measured CRPS improvement
is explicitly insufficient for promotion under `SIGNAL_REGISTRY.md`
§1.10 — promotion additionally requires a reproducible downstream net-EV
advantage after realistic Turbo costs, path/KO risk and multiple-testing
deflation, which has not been measured for these models.

---

## 12. Change Log for Research Framework

- **2026-09-10:** Framework v1.0.0 established for Phase 0+1. Trial budget: 6/quarter. Ruin threshold: 20% / 12mo / 5% probability.
- **2026-09-13:** §11 added — 2026Q3 research trials (W4 measurement, W9's
  6 pre-registered challenger families) recorded; 0 promotions; existing
  versioned values (budget, ladder thresholds, ruin metric, sizing caps)
  unchanged.
- **2026-09-19:** Bookkeeping fix, not a policy change: the six §11.1 W9
  trials had never actually been written to `research_trials` (only to
  `state/registry/failed_hypotheses.json`), so `count_research_trials_in_
  quarter("2026Q3")` was reading 0 instead of 6, undercounting the §1.2
  quarterly budget. Backfilled under their original trial_ids
  (`learning/trials.py::backfill_w9_trials`, `turboedge research
  backfill-trials`); `research_trials` now correctly reads 6/6 for 2026Q3.
  Budget (6/quarter), ladder thresholds, ruin metric and sizing caps are
  unchanged.
