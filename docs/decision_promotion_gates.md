# Decision matrix: the seven unset promotion gates

**Status: ANALYSIS. No threshold was set. `PROMOTION_LOCKED` remains the
honest state, and `PromotionCriteria.is_complete` stays `False`.**

`alpha/promotion.py` leaves seven thresholds at `None` rather than giving
them plausible-looking defaults, and `is_complete` reports `False` so an
unset gate blocks rather than passing by default. This document asks, per
gate, whether that can change yet.

A note on what I could verify: the production state lives only inside
encrypted artifacts, which this system does not decrypt. Figures below are
either from `docs/measured_results.md`, from the code, or from the
2026-10-24 production research run — never from an unread database. Where a
quantity is unknown to me, it says so.

---

## 1. The matrix

| Gate | Needed? | Evidence exists? | Defensible threshold derivable? | Independent of the hypothesis under test? | Missing measurement |
|---|---|---|---|---|---|
| `max_expected_shortfall` | **Yes** — a turbo loses 100 % at the barrier, so tail loss is the governing risk | **No** — no position has ever been held | No | Would be, once derived from realised data | Realised per-position loss distribution from matured, labelled forward-ledger entries |
| `max_drawdown` | **Yes** — for demotion as much as promotion | **No** — same reason | No | Would be | A realised equity path, which requires executed positions |
| `max_ko_risk` | **Questionable as stated** | Partly — P(KO) is measured, but `ranking/ev.py` records it as explicitly **uncalibrated** | **No** — a threshold on `p_ko` is a threshold on a number that does not mean what it says | Yes | Out-of-sample P(KO) calibration. Attempted in W10 (`docs/measured_results.md` §3.1) and **no calibrator was promoted** |
| `max_alpha_correlation` | **Yes**, once more than one alpha exists | **No** — needs a correlation matrix across alphas, and there is not yet one alpha | No | Yes | An `AlphaCorrelationMatrix`; moot while the registry is empty |
| `max_issuer_concentration` | **Yes** — issuer margin and quote behaviour are issuer-specific | Partly — coverage is measured, and it is extreme: BNP dominates the product universe | **No** — any limit set now either blocks the whole universe or is vacuous | Yes | Either a genuinely multi-issuer universe, or a limit expressed per *comparable product set* rather than per universe |
| `min_data_quality` | **Yes** | Partly — `quality_score` exists per snapshot | **No** — the score has no promotion-relevant calibration; nothing establishes what value corresponds to what decision error | Yes | A link from `quality_score` to a measured outcome (e.g. fill plausibility, label ambiguity rate) |
| `min_forward_shadow_days` | **Yes** — it is the one gate that cannot be shortcut | **No** — and critically, the window is short and self-referential | **No** — a number chosen now would be chosen to fit the window that exists, which is the opposite of a pre-set bar | Yes | Elapsed time, nothing else. This gate's evidence accrues by waiting |

## 2. What the matrix says

Three distinct reasons, and they need different answers:

**Gates blocked by having no positions at all** —
`max_expected_shortfall`, `max_drawdown`. These cannot be derived from
anything but realised outcomes, and the system has never held a position by
design (manual execution, no broker). They are not "missing numbers"; they
are numbers that do not exist yet. Waiting is the only path, and it is a
long one.

**Gates blocked by an uncalibrated or uninterpreted input** —
`max_ko_risk`, `min_data_quality`. Here the quantity is measured but its
meaning is not established. W10 tried to promote a P(KO) calibrator and
none cleared its bar, which is recorded in §3.1 of the measured results. A
threshold on an uncalibrated probability would look like a risk control and
not be one.

**Gates blocked by the universe's shape** —
`max_issuer_concentration`, `max_alpha_correlation`. Both are well-defined
and both are currently unanswerable for structural reasons: one issuer
dominates the product universe, and there is not yet a single confirmed
alpha to correlate against anything.

**And one gate blocked only by time** — `min_forward_shadow_days`. This is
the one where a decision is possible *today without evidence*, because the
honest way to set it is to pick it **before** seeing how long the shadow
period turns out to be. That is what a pre-set bar means. Choosing it now,
while it cannot be fitted to a result, is strictly more defensible than
choosing it later.

## 3. Recommendation

**Set none of the six that need evidence. Consider setting
`min_forward_shadow_days` now, precisely because no evidence exists.**

That is not a contradiction. The other six need a measurement to be
defensible. This one needs the *absence* of one: its whole purpose is to fix
in advance how much forward observation is required, and every day that
passes makes a number chosen afterwards more suspect. A plausible anchor is
the independence structure already measured — `docs/measured_results.md`
records that at 14 days nine of every ten forward windows overlap, so a bar
expressed in *independent* windows rather than calendar days would follow
from something already known.

I am not setting it. It is an alpha release criterion, and the brief
reserves those.

Everything else stays `None`, and `PROMOTION_LOCKED` stays correct. Six
unset gates are not a gap to be closed by assigning numbers; they are an
accurate statement that this system has not yet earned a promotion
decision.

## 4. Consequence for the autonomy proposal

`docs/proposal_confirmatory_autonomy.md` §2.1 lists these seven gates as a
precondition. That precondition is **not** met and cannot be met by
analysis. The proposal stays inert, and
`tests/research/test_governance_boundary.py` enforces that nothing reads
it.

Note also that the gates being unset is now *visible* at the point it
matters: since the 2026-10-24 audit, ACTIONABLE requires a confirmed alpha
source in `PRODUCTION_STATES`, and an alpha cannot reach a production state
without clearing these gates. The chain from "no measured thresholds" to
"no trade proposals" is closed in code rather than relying on no model
happening to produce a positive EV.

## 5. Exactly what is needed

1. A ruling on whether `min_forward_shadow_days` is set now, and in what
   unit — calendar days or independent forward windows.
2. Acknowledgement that the other six remain unset, with
   `PROMOTION_LOCKED` as the recorded state.
3. A decision on whether `max_ko_risk` should be **removed** rather than
   left unset: it is a threshold on a quantity the repository documents as
   uncalibrated, and an unsettable gate is different from a gate that
   should not exist in that form. If P(KO) is never going to be calibrated,
   a tail-loss gate expressed in realised loss terms would do the job
   `max_ko_risk` is standing in for.
4. A decision on expressing `max_issuer_concentration` per comparable
   product set instead of per universe, which would make it answerable
   under today's issuer coverage.

---

*Prepared 2026-10-24. No threshold set; no gate removed.*
