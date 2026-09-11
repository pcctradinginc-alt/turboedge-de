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

---

## 3. Future Challenger Families (Placeholder)

Reserved for Phase 2+:
- `trend_plus_volatility_v1` (experimental)
- `mean_reversion_bounded_v1` (experimental)
- `cross_asset_leadlag_v1` (experimental)

No challengers implemented in Phase 0+1.

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
