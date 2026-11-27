# Exploratory ablation audit

Run `uv run turboedge research ablation --underlying DAX` against the restored
local archive. No new data requests. Cross-asset challenger only: refit full and
each leave-one-feature-out variant using identical purged walk-forward settings
and complete-case support. Match exact timestamps and outcomes. Positive Brier
and gross directional log-return gains favor keeping the feature. Report sample
coverage and exploratory moving-block intervals; these are not calibrated
simultaneous confidence intervals. Historical results are development evidence,
not untouched confirmation. No automatic feature removal, no promotion.

Remaining: other model families, actual product-level net economics, filter
counterfactual replay including rejected products, independent forward data,
and multiplicity-controlled confirmatory testing. Missing history must not be
interpreted as evidence of no utility.

The audit now schedules all six W9 challenger families (ten contrasts).
Vol targeting is compared against the same trend with unit multiplier;
short-horizon reversal against an intercept-only ridge on identical valid
feature rows; calendar buckets against a pooled distribution with the same
bucket sample eligibility. The multiplicity count derives from the actual
planned contrasts. Gross directional gains are not executable product P&L.
Incomplete checkpoints cannot be treated as full coverage.
