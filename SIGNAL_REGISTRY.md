# Signal Registry

Registry of all signals implemented in TurboEdge-DE, with versioning, formulas, and change protocols.

---

## 1. Protected Baseline: tsmom_horizon_norm_v1

**Status:** PROTECTED  
**Version:** 1.0.0  
**Effective Date:** 2026-09-10  
**Last Modified:** 2026-09-10

### 1.1 Concept

Time-Series Momentum (TSMOM) normalized by realized volatility. Measures multi-horizon trend strength using log-returns. Protected baseline against which all other signals benchmarked.

### 1.2 Formula

For each lookback period $k \in \{21, 63, 126\}$ trading days:

$$z_k(t) = \frac{\ln(P_t / P_{t-k})}{\sigma_t(k) \cdot \sqrt{k}}$$

where:
- $P_t$ = underlying close price at time $t$
- $P_{t-k}$ = underlying close price $k$ trading days ago
- $\sigma_t(k)$ = realized volatility estimate over past $k$ days

**Realized volatility** calculated as:
$$\sigma_t(k) = \sqrt{\frac{1}{k}\sum_{i=1}^{k} \ln^2(P_{t-i+1}/P_{t-i})}$$

Apply **EWMA smoother** to volatility:
$$\sigma_{t,\text{EWMA}}(k) = \lambda \cdot \sigma_{t-1,\text{EWMA}}(k) + (1-\lambda) \cdot \sigma_t(k)^{\text{realized}}$$

**EWMA decay parameter:** $\lambda = 0.94$

### 1.3 Clipping

All $z_k$ values clipped to $[-3, +3]$ to suppress tail effects and prevent extreme signal dominance.

$$z_k^{\text{clipped}} = \max(-3, \min(+3, z_k))$$

### 1.4 Aggregation

Final signal score:
$$s(t) = \frac{1}{3}\left(z_{21}^{\text{clipped}} + z_{63}^{\text{clipped}} + z_{126}^{\text{clipped}}\right)$$

Range: $s(t) \in [-3, +3]$

### 1.5 Direction Interpretation

- $s(t) > 0$ → bullish (LONG candidate preferred)
- $s(t) < 0$ → bearish (SHORT candidate preferred)
- $|s(t)| < 0.5$ → weak signal (below threshold)

### 1.6 Threshold

**Threshold for WATCH category:** $|s(t)| \geq 0.5$

Products selected only if $|s(t)| \geq 0.5$. Threshold versionized in config; changes require new trial_id and formal governance review (not silent reoptimization).

### 1.7 Data Requirements

- **Minimum lookback:** 126 trading days historical data required
- **Staleness tolerance:** price data ≤ 1 day old
- **Gap handling:** overnight gaps included in log-returns; no gap-removal smoothing
- **Holiday handling:** weekend gaps treated separately in future path simulations

### 1.8 Output Schema

```python
class SignalSnapshot(BaseModel):
    signal_id: str = "tsmom_horizon_norm_v1"
    signal_version_hash: str  # SHA256 of this spec
    underlying_id: str
    prediction_time: datetime
    frozen_at: datetime  # when calculated, before quotes fetched
    score: float  # s(t) value
    components: dict[str, float]  # {"z_21": ..., "z_63": ..., "z_126": ...}
    direction_hint: Direction | None  # LONG if score > 0.5, SHORT if score < -0.5
    threshold: float = 0.5
    config_hash: str
    git_commit: str | None
    data_snapshot_hash: str  # hash of input price data
```

### 1.9 Immutability Rules

**This signal is PROTECTED and can never be:**
- Deleted
- Modified silently
- Re-optimized on new data

**If threshold or lookbacks must change:**
1. Create new trial_id (e.g., `2026Q4_001`)
2. Implement as `tsmom_horizon_norm_v2` with new version_hash
3. Run formal A/B test vs. v1 (min 4 weeks, min 100 outcomes)
4. Submit promotion to research governance (must pass DSR, PSR, FDR thresholds)
5. Update SIGNAL_REGISTRY.md with v2 entry

### 1.10 Benchmarking

All other signals evaluated relative to:
$$\Delta \text{net-EV} = \text{net-EV}_{\text{challenger}} - \text{net-EV}_{\text{tsmom\_v1}}$$

Challenger must achieve $\Delta \geq 10$ bps out-of-sample to justify complexity.

---

## 2. Signal Registry Changelog

| Signal ID                    | Version | Date       | Status    | Notes                                        |
|------------------------------|---------|------------|-----------|----------------------------------------------|
| tsmom_horizon_norm_v1        | 1.0.0   | 2026-09-10 | PROTECTED | Initial baseline; threshold 0.5; λ=0.94      |

`tsmom_horizon_norm_v1` itself (the score) is unchanged and remains
protected. Workstream W4 (2026-09-12) measured a full predictive
*distribution* built on top of this score (mean/sigma/quantiles via
`TsmomForecastModel`) against a trivial unconditional (null) benchmark —
see §3 below and `docs/measured_results.md` §1. That measurement did not
alter the protected score; it evaluated a distributional mapping built on
top of it.

---

## 3. Forecast Models and Challenger Signal Families (measured, none promoted)

Every entry below has been implemented and measured out-of-sample
(walk-forward, `min_train=750`, `step=21`, `embargo=horizon_days`,
horizons 3/5/7/10/14 trading days, `^GDAXI`/`^NDX`/`^GSPC`/`^STOXX50E`,
2010–2026 `yfinance` daily bars). **None has been promoted.** Full numbers:
`docs/measured_results.md` §1–2; raw source write-ups:
`w4_walkforward_results.md`, `w9_challenger_results.md` (outside this
repository, on the development machine).

### 3.1 Forecast models (`models/forecast.py`, `models/directional.py`, `models/quantile.py`) — Workstream W4, measured 2026-09-12

| model_id / signal_family | class | scope measured | result vs. null | status |
|---|---|---|---|---|
| `null` (`signal_family="null"`) | `NullModel` | 4 underlyings × 5 horizons (20/20) | reference benchmark | baseline, not a challenger |
| `tsmom` (`signal_family="tsmom"`, distributional mapping of the protected score) | `TsmomForecastModel` | 4 underlyings × 5 horizons (20/20) | worse Brier in 20/20 cells (avg +0.0072, up to +0.0212); ECE worse by 5–15× in nearly every cell | not promoted |
| `logit` (`signal_family="logit"`) | `LogisticDirectionModel` | DAX only, h=5 only, reduced step (1 data point, not a full sweep) | worse Brier than null (+0.0014) | not promoted, insufficient scope to evaluate further |
| — (`RidgeReturnModel`, `models/quantile.py`) | `RidgeReturnModel` | not separately walk-forward measured in W4 | not measured | not promoted |

### 3.2 Challenger signal families (`models/challengers.py`, `features/cross_asset.py`) — Workstream W9, pre-registered before measurement, measured 2026-09-13

| trial_id | signal_family | class | cells measured | mean ΔBrier vs. null | best single-cell edge vs. null (signal-dir. return) | BH-significant? | status |
|---|---|---|---|---:|---:|---|---|
| W9-2026Q3-001 | `voltarget_tsmom` | `VolTargetedTsmom` | 14 | +0.0179 (worse) | ~0 bp (best cell effectively 0) | 0/14 | dormant |
| W9-2026Q3-002 | `lowvol_regime_trend` | `LowVolRegimeTrend` | 14 | +0.0356 (worse; worst of the six) | +0.71 bp | 0/14 | dormant |
| W9-2026Q3-003 | `reversal_short_horizon` | `ShortHorizonReversal` | 14 | +0.0043 (worse) | +1.50 bp; lowest raw p-value in the whole sweep (p=0.040, SPX h=10 return) but does not survive BH correction against 80 comparisons; stricter pre-registered reversal-family bar (BH-significant AND deflated z≥2.0 AND DSR≥0.6) not cleared | 0/14 | dormant |
| W9-2026Q3-004 | `vix_term_structure` | `VixTermStructure` | 14 | +0.0091 (worse) | +1.85 bp (best edge of all six families) | 0/14 | dormant |
| W9-2026Q3-005 | `cross_asset_leadlag` | `CrossAssetLeadLag` | 10 (DAX/ESTX50 only, per pre-registration) | +0.0056 (worse) | +0.84 bp | 0/10 | dormant |
| W9-2026Q3-006 | `seasonality_turn_of_month` | `SeasonalityTurnOfMonth` | 14 | +0.0002 (worse, but closest to null) | +0.68 bp | 0/14 | dormant |

Across all 80 cells measured in W9: 78/80 (97.5%) had a worse Brier score
than the null model; 0/80 cleared Benjamini-Hochberg FDR (α=0.10) on
either Brier or signal-direction return; 0/80 exceeded even the low end
of the assumed realistic Turbo round-trip cost band. Best edge across the
entire sweep (1.85 bp) is well short of `SIGNAL_REGISTRY.md` §1.10's
required 10 bp minimum improvement over the TSMOM baseline. All six
families are recorded `dormant` in `state/registry/failed_hypotheses.json`
(Master Spec §23 hypothesis graveyard) with `effective_sample` in the
34,542–48,304 range per family and `incremental_net_ev` (each family's own
single best cell, not its average) 1–2 orders of magnitude below the
0.0010 (10 bp) ladder minimum in `GOVERNANCE.md` §2.2 — none is a close
call.

**Nothing in this section has been promoted to champion or added to the
live ensemble.** Per CLAUDE.md rule 26 ("keine Verbesserung nur anhand
In-Sample behaupten") and Master Spec §53, this is reported as the
unembellished, negative result it is.

### 3.3 Phase D distributional baselines (`models/baselines.py`) — measured 2026-09-19

W4/W9 (§3.1-3.2) evaluate every model only on `p_up`/Brier -- Phase D
(`docs/measured_results.md` §6) adds a proper distributional scoring harness
(CRPS, pinball loss, interval coverage) to `backtest/walkforward.py` and
measures four pre-registered baselines under it, in order: (a) the
unconditional empirical distribution (`NullModel`, unchanged), (b)
`RegimeConditionalEmpiricalModel`, (c) `RegularizedLinearLocationModel`, (d)
`RobustLocationScaleModel`.

**The 2026-09-19 figures below were superseded on 2026-09-26.** They were a
win count with no p-value, no bootstrap and no deflation, and the re-run did
not reproduce them (`docs/measured_results.md` §6.11-§6.13).

| model_id / signal_family | class | 2026-09-19 (as reported) | 2026-09-26 re-run | block-bootstrap p | status |
|---|---|---|---|---|---|
| `regime_conditional_empirical_v1` | `RegimeConditionalEmpiricalModel` | 20/20, mean -3.22% | **18/20, mean -1.40%** | **0.0135** | measured, not promoted |
| `regularized_linear_location_v1` | `RegularizedLinearLocationModel` | 20/20, mean -3.45% | **15/20, mean -1.54%** | **0.0142** | measured, not promoted |
| `robust_location_scale_t_v1` | `RobustLocationScaleModel` | 20/20, mean -2.47% | **14/20, mean -0.78%** | 0.1050 | measured, not promoted |

The first two survive Benjamini-Hochberg at α=0.10 against the quarter's
eleven model-level hypotheses (§6.13). That is **robust retrospective
evidence, not confirmatory**: the bootstrap was chosen after the original
result was seen, on the same data. The confirmatory test — method frozen,
new out-of-sample data — has not been run.

Brier-vs-null figures from the original run (better in 3/20, 8/20 and 9/20
respectively) are left as recorded; they were never the basis of any claim
here, and the re-run measured CRPS.

**Not promoted, and not added to the live scan ensemble** (they are wired
only into the `turboedge backtest` measurement path). Per this section's
own §1.10 rule (a challenger needs a measured net-EV advantage, not just a
better score on one axis) and this project's promotion rule, a CRPS
improvement alone is exactly as insufficient for promotion as a Brier
improvement alone was for §3.1/§3.2 -- no net-EV measurement was performed
against these baselines in this pass. See `docs/measured_results.md` §6 for
the full measurement, per-horizon/per-quantile breakdown, and the
heteroskedasticity-awareness interpretation of the CRPS result (most likely
explanation: these baselines rescale their predicted interval width by
*current* volatility, `NullModel` does not -- a distributional
improvement, not evidence of directional skill).

---

## 4. Change Protocol

To add or modify signals:

1. **Create trial_id** in governance system
2. **Implement** as new version (e.g., v2, v3)
3. **Run ablation** to show isolated contribution
4. **Walk-forward backtest** (min 4 weeks, min 100 labels)
5. **Calculate deflated stats** (DSR, PSR, FDR)
6. **Submit to promotion gate** (must pass thresholds)
7. **Update SIGNAL_REGISTRY.md** with new entry
8. **Archive** or demote old versions if superseded

Protected baselines never deleted; new versions created instead.

---

## 5. Implementation Notes

- **Signal calculation frozen before quote fetch.** Ensures no look-ahead bias; prices do not influence signal.
- **UTC timestamps required.** All times tz-aware UTC.
- **Reproducibility.** Version hash computed from formula, params, lookback list, EWMA λ, clipping bounds.
