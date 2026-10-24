# Architecture audit: what earns its complexity

**Status: ASSESSMENT. One low-risk simplification implemented (noted in §4);
everything else is a recommendation. No protective mechanism, negative
result, historical research datum or recorded decision was removed.**

Classification is by **operational effect**, not by whether code is
reachable. Every package is imported from some entry point, so reachability
distinguishes nothing. Sizes are lines of Python.

---

## 1. Classification

### ESSENTIAL — safe data collection, correct valuation, reproducible research

| Component | Size | Why |
|---|---:|---|
| `storage/` | 6012 | The point-in-time archive. The irreplaceable asset: a bid/ask quote at a past minute cannot be re-fetched. |
| `pipeline/` | 3808 | `scan`/`scan_all` — the only thing that writes that archive. |
| `pricing/` | 2786 | Integrity, financing, fair value. Without it the archive is numbers of unknown meaning. |
| `external/` + adapters | 2309 + | The macro data factory. Point-in-time correct by construction. |
| `simulation/` | 1879 | Path generation behind every EV figure. |
| `ranking/` | 1975 | Gates and sizing — where NO TRADE is enforced. |
| `monitoring/` | 753 | `pipeline_health` makes silent data loss audible; `source_health` makes a dead source visible. |
| `state/` | 2070 | Pack/restore/retention. The 2026-10-23 loss was a failure *here*, which is an argument for its importance. |
| `features/`, `universe/` | 1193, 664 | Availability filtering and the product universe. |

### USEFUL — demonstrably helpful, not critical

| Component | Size | Why |
|---|---:|---|
| `backtest/` | 3929 | `purged_cv` is load-bearing for every model; `excursion_eval` is the only approved research harness. `ko_calibration` produced a real negative result (§3.1 of measured results). |
| `reporting/` | 2718 | The operator's only window into the system. |
| `notifications/` | 1000 | Gmail is the sole production output channel. |
| `learning/` | 2153 | Trials ledger, negative memory, posteriors. `failed_hypotheses` is a working retest gate. |
| `positions/` | 923 | Manual position ledger — and now the capital base for the portfolio caps. |

### DORMANT — no operational effect today, or no data to act on

| Component | Size | Evidence |
|---|---:|---|
| `alpha/` | 712 | `alpha_sources` has never held a row. **`AlphaRegistry` is re-exported from `alpha/__init__.py` and called by nothing, and there is no CLI command for alpha at all** — so the promotion ladder cannot be driven even by hand. Promotion is additionally blocked by seven unset gates. Keep: since 2026-10-24 the ACTIONABLE gate consults `PRODUCTION_STATES`, so this package is what makes NO TRADE structural. |
| `meta/` catalogue | 4330 (pkg) | Fourteen static questions, of which twelve have no harness. The queue is scored and read every week; the scoring is live, the answerability is not. |
| `research/` loop | 1896 | Runs weekly in production since 2026-10-24 12:54 and correctly reports `AWAIT_DATA`. **No screen has ever completed**, because the forward ledger has nothing matured. The mechanism works; it has nothing to work on. |
| `research/generations.py` | 180 | `SystemGeneration`/`GenerationStatus` are used **only by `research/__init__.py`**, which re-exports them. `GenerationEvidenceError` is used nowhere at all. |
| `models/` challengers | part of 3606 | Five challenger families, all measured without out-of-sample edge (§2 of measured results). Kept deliberately: they are the negative record. |

### REDUNDANT — duplicate or avoidable

| Item | Evidence | Recommendation |
|---|---|---|
| `ranking/ev.py::to_candidate_gate_input` | No production caller; `pipeline/scan.py` builds `GateInput` directly. A second, divergent path that had already fallen out of sync with the real one. | Remove, or keep with a docstring saying it is test-only. The 2026-10-24 audit threaded the new preconditions through it so it cannot resurrect the pre-audit behaviour. |
| Two cluster capital limits | `sizing.max_cluster_fraction` 15 % and `cluster.max_capital_fraction_per_cluster` 25 %. Under both options in `docs/decision_risk_limits.md` the 25 % one never binds, because the position-count cap of 3 binds first. | Reconcile or remove — operator decision, it is a risk limit. |
| `max_total_notional` in GOVERNANCE.md §5.2 | Describes a limit the code does not have; `max_total_risk` is a different quantity. | Already corrected in §5.2; no code change. |
| `max_ko_risk` promotion gate | A threshold on a quantity `ranking/ev.py` documents as explicitly uncalibrated. | See `docs/decision_promotion_gates.md` §5.3: consider removing rather than leaving unset. |
| `AlphaStatus.HEALTHY` absent from `PRODUCTION_STATES` | `NORMAL_PRODUCTION → HEALTHY` is a legal transition, so a healthy promoted alpha would stop counting as production. Error is in the safe direction. | Operator decision: widening that set changes alpha release criteria. |

## 2. What I did **not** classify as removable

- Negative results, the failed-hypotheses registry, `research_trials`, and
  the challenger families that failed. The failure record is the asset, not
  the cleanup target.
- Protective mechanisms, including the ones that currently never fire.
- `docs/proposal_confirmatory_autonomy.md` and its boundary tests.
- Anything in `state/` or `storage/`, despite `duckdb.py` being 6012 lines
  in one file. Splitting it is a plausible readability gain and a
  behaviour-neutral refactor of the module every other module depends on —
  the risk/benefit does not favour doing it during a safety audit.

## 3. The honest summary

The system is not too complex in the sense of having useless parts. It is
**complete in machinery and empty in evidence**: a full research factory,
an alpha promotion ladder with no alpha, fourteen catalogued questions with
two answerable, a loop that decides correctly and has nothing to decide
about, and five measured model families that all failed.

The useful simplification is therefore not deletion. It is to stop adding
layers above an empty evidence base, and to spend effort on the two things
that would change the picture: matured forward-ledger entries, and a
genuinely multi-issuer product universe.

## 4. Implemented here

One, behaviour-preserving and test-covered: the two economic preconditions
were threaded through `to_candidate_gate_input` so the redundant path
cannot produce a `GateInput` that silently omits them. Nothing was deleted.

Every other item above is a recommendation.

---

*Prepared 2026-10-24.*
