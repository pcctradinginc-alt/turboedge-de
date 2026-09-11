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

Each promotion from experimental to live increments the effective number of tested hypotheses, triggering multiple-testing inflation penalties.

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

## 11. Change Log for Research Framework

- **2026-09-10:** Framework v1.0.0 established for Phase 0+1. Trial budget: 6/quarter. Ruin threshold: 20% / 12mo / 5% probability.
