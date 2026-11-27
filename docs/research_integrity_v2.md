# Research integrity v2 (exploratory, no promotion)

The weekly net-EV command reads frozen research cohorts rather than compacted
DuckDB snapshots. The old probe remains a retrospective diagnostic only.
A common receipt timestamp and a fixed 60-second maximum quote age are
conservative quote proxies, not guarantees of real fills. Both legs must
have a simultaneous observable exit within the fixed 24–48h window, or an
earlier documented KO. Pending cohorts are separate from mature missing exits.
The tape health census uses the same synchronization rule. Historical cohorts
that fail the stricter checks remain invalid/missing; do not delete or relabel
them to manufacture a positive result.

Ablation intervals use at least 20 expected bootstrap tail draws per
Bonferroni tail, batched to bound memory. Empirical coverage is not yet
validated. Filter diagnostics distinguish empty and unmatched datasets.

Remaining: dynamic ledger barriers, full
gate replay, fee policy, LCB calibration, further feature ablations and
live evidence. No risk thresholds or promotion rights changed.

Production autonomous discovery also uses frozen entry cohorts and paired exit
observations. The legacy injected-snapshot test path remains retrospective.

Ledger daily-bar KO scenarios are now marked ambiguous unless the issuer
confirms the KO. Observed barrier changes inform a conservative scenario,
not a reconstructed effective barrier timeline. A predeadline exit bid is
also ambiguous. Historical labels are not rewritten. Exact dynamic-barrier
reconstruction and maturity gating remain open.
