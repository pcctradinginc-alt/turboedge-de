# Decision required: the two unresolved position caps

**Status: DECISION PROPOSAL. Nothing changed. `GOVERNANCE.md` §5.2 still
records both values, and §11 reserves limit changes for a dated human
decision.**

This extends §5.2a, which established *what* the divergence is. This
document states what each option would mean for portfolio risk, and
recommends one.

---

## 1. What is actually in dispute

Of the four rows that looked like divergences, only two are a disagreement
about one quantity:

| | `GOVERNANCE.md` | Code + `configs/ranking.yaml` |
|---|---:|---:|
| `max_position_fraction` | 2 % | 5 % |
| `max_cluster_fraction` | 5 % | 15 % |

The other two were never the same quantity and are settled in §5.2a: the KO
figure is a portfolio-wide budget in the document and a per-position
expected-loss cap in the code (8 % could never bind as a per-position cap —
it would need `p_ko > 4`), and `max_total_notional` 100 % is not the
implemented `max_total_risk` 30 %.

A third cluster number exists and is the one that actually gates today:
`cluster.max_capital_fraction_per_cluster` = 25 %, used by
`cluster_risk_pass`, with no counterpart in §5.2.

## 2. How the caps interact in practice

`ranking/sizing.py` applies them in this order, each net of what is used:

```
fractional Kelly → ÷(1 + 2·uncertainty) → min(position) → min(cluster − used)
                 → min(total − used) → min(ko_budget / p_ko)
```

`ranking/cluster.py` separately caps the *number* of positions per cluster
at 3 (`max_active_positions_per_cluster`) and cluster capital at 25 %.

Four underlyings are enabled today — DAX, NDX, EURUSD, XAU — in four
declared clusters, though the runtime clusters are recomputed from
correlation and can merge DAX with NDX.

| | at 2 % / 5 % (document) | at 5 % / 15 % (code) |
|---|---:|---:|
| Worst single-position loss (full knock-out) | 2 % of capital | 5 % |
| Max capital in one cluster (3 positions) | 6 % | 15 % |
| Which cluster cap binds first | `max_cluster_fraction` 5 % → 2 positions | position count 3 → 15 %, under the 25 % gate cap |
| Correlated knock-out of a whole cluster | up to 6 % | up to 15 % |
| Portfolio ceiling with 3 clusters populated | ~18 %, under `max_total_risk` | 45 % wanted, **cut to 30 %** by `max_total_risk` |
| Does `max_ko_loss_contribution` bind? | never (0.02/p_ko ≥ 2 % for all p_ko ≤ 1) | when `p_ko > 0.4` |

Two consequences worth stating plainly:

- At the document's numbers, the **cluster cap becomes the binding
  constraint inside a cluster** (5 % allows two 2 % positions and part of a
  third), and the per-position KO budget never binds at all. The system
  becomes cluster-limited rather than count-limited.
- At the code's numbers, the **portfolio total is what stops accumulation**
  (30 %), not the cluster cap — and `max_total_risk` was itself inert until
  2026-10-24, so in practice nothing stopped it.

A turbo certificate can lose its entire value at the barrier. The
single-position figure is therefore a real loss bound, not a
value-at-risk — which is why the factor of 2.5 between the options is a
factor of 2.5 on the worst loss a single mistake can cause.

## 3. Options

**Option A — adopt 2 % / 5 % (the document's figures).**
Tightens. Worst single loss 2 %, worst correlated cluster loss 6 %. The
figures were chosen deliberately before implementation; the code's were
typed during it, in a commit that edited `GOVERNANCE.md` but not §5.2, with
no derivation anywhere. Cost: smaller suggested sizes, so a genuine edge
would be exploited more slowly. Since no confirmed alpha exists and no
candidate currently reaches ACTIONABLE, that cost is presently zero.

**Option B — adopt 5 % / 15 % (the implemented figures).**
Ratifies what runs today. Requires a derivation that does not exist: no
measurement in `docs/measured_results.md` supports either number, so this
would be ratifying a value because it is the status quo. Worst single loss
5 %, worst correlated cluster loss 15 %.

**Option C — adopt 2 % / 5 % now, revisit after forward evidence.**
Option A plus an explicit review trigger: once the forward ledger has
matured, labelled entries and a confirmed alpha exists, re-derive the caps
from measured knock-out frequency and realised excursions rather than from
judgement. Records that the number is provisional without leaving the
conflict open.

## 4. Recommendation

**Option C.** The reasoning is asymmetric rather than a preference between
two defensible numbers:

- The stricter figures were a decision; the looser ones were drift. §5.2a
  establishes that from the commit history.
- Neither is derived from evidence, so the tie-break cannot be empirical
  today — and when a tie-break has no evidence, the direction that bounds
  the loss is the defensible one.
- The cost of being too strict right now is precisely zero: nothing reaches
  ACTIONABLE, so no proposal is being shrunk. The cost of being too loose
  is a 2.5× larger loss on the first mistake after an alpha is confirmed.
- Leaving it unresolved is worse than either, because an autonomous process
  operating under an ambiguous limit is not operating under a limit.

I am not making this change. It is a hard risk limit.

## 5. Exactly what is needed

1. A dated entry in `GOVERNANCE.md` naming the approver and the chosen
   option, in the form §1.1a uses.
2. `configs/ranking.yaml` and the `SizingConfig` defaults updated together —
   they must not diverge, and `tests/test_risk_limit_consistency.py`
   enforces that a configured cap is never looser than the code default.
3. The §5.2 table updated to the single chosen value, and the two xfail
   markers in `tests/test_risk_limit_consistency.py` removed, so the suite
   stops reporting an open conflict.
4. A decision on `cluster.max_capital_fraction_per_cluster` (25 %) too: it
   gates `cluster_risk_pass` today and has no counterpart in §5.2. At the
   document's figures it would never bind; at the code's it binds after 5
   positions, above the count cap of 3, so it never binds either. It is
   currently inert under both options and should either be reconciled with
   `max_cluster_fraction` or removed.
5. A decision on whether the portfolio-wide KO budget §5.2 describes should
   be implemented. It does not exist in code, and `max_ko_loss_contribution`
   is a different quantity. Leaving a described-but-absent limit in a
   governance document is the condition that produced this whole
   reconciliation.

Until all of this is recorded, the open conflict stays visible in the test
suite. A green CI run does not mean it is resolved: the two divergences are
expected failures with both numbers in the message, so the suite reports
them without failing the build.

---

*Prepared 2026-10-24. No limit value was changed.*
