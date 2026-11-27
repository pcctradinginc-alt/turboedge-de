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
