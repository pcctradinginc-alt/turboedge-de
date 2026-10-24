# Proposal: once-approvable frozen rules for confirmatory research

**Status: PROPOSAL. NOT ACTIVE. NOT ACTIVATABLE BY THIS SYSTEM.**

This document proposes a rule set that, if a human approved it once, would
let the loop open and measure confirmatory trials without per-trial
approval. Nothing in this document is implemented, no code reads it, and
activating it is a source change plus a recorded human decision — see
"How activation would work" at the end. It was written because the loop can
now screen autonomously but stops at every confirmation, and the question
"what would have to be true for that gate to open safely?" deserves a
written answer rather than an ad-hoc one later.

The author of this proposal is the system. That is a reason to read it
sceptically: a proposal to loosen a constraint on oneself is exactly the
kind of document whose incentives are suspect. Every rule below is
therefore written to be *more* restrictive than the current manual process
in at least one respect, and the two existing blockers are treated as
preconditions rather than as problems to be reasoned around.

---

## 1. What is being proposed, precisely

Today: the loop may screen on DEVELOPMENT data. A confirmatory trial — the
kind whose p-value means something, which consumes the quarterly budget of
six, and which touches the CONFIRMATION partition — requires
`research approve` with a named human.

Proposed: a human approves **the rules** once. After that, the loop may
open and measure a confirmatory trial when every condition in §3 holds,
and may not otherwise. The human still approves every *promotion* to live
alpha. The gate moves from "approve each experiment" to "approve the
conditions under which experiments may run, then audit".

What is **not** proposed, and must not be read into this:

- No change to any trading risk limit.
- No automatic promotion of anything to live alpha.
- No broker execution, no order placement: unchanged and out of scope.
- No change to the quarterly budget of six, the ladder thresholds, the
  multiple-testing correction, or the purged-CV/embargo requirements.
- No reduction in what a confirmatory trial has to demonstrate. The rules
  below only decide *who opens it*, never *what it must show*.

---

## 2. Preconditions that are not yet met

These are stated first because they are the reason this cannot be
activated today, independent of whether the rules in §3 are agreed.

**2.1 Seven promotion gates are unset.** `alpha/promotion.py` leaves
`max_expected_shortfall`, `max_drawdown`, `max_ko_risk`,
`max_alpha_correlation`, `max_issuer_concentration`, `min_data_quality`
and `min_forward_shadow_days` as `None`, deliberately: an invented number
would look like a measured one. `PromotionCriteria.is_complete` is
therefore `False`, and the honest state is `PROMOTION_LOCKED`. A
confirmatory result that cannot be acted on is still worth having, so this
does not by itself block confirmatory autonomy — but a loop that produced
confirmable results it could never promote would accumulate conclusions
nobody can use, and that is a reason to resolve it first rather than a
reason to proceed.

**2.2 `GOVERNANCE.md` §5.2 and `configs/ranking.yaml` disagree about
three position limits, in both directions.** Recorded as xfail in
`tests/test_risk_limit_consistency.py` with both numbers:

| limit | `configs/ranking.yaml` | `GOVERNANCE.md` §5.2 |
|---|---|---|
| `max_position_fraction` | 0.05 | 0.02 |
| `max_cluster_fraction` | 0.15 | 0.05 |
| `max_ko_loss_contribution` | 0.02 | 0.08 |

Two of the three are looser in the config than in the governance document.
A rule set that referred to "the position limits" would be ambiguous about
which numbers it meant, and an autonomous process operating under an
ambiguous limit is not operating under a limit. **Reconciling these is a
human decision and a hard precondition.** This proposal does not pick a
side, and the system must not: choosing the looser number would be a
self-serving resolution of exactly the kind this document is suspect for.

---

## 3. The proposed frozen rules

"Frozen" means: each rule is a constant in source, each is checked before a
trial may open, any single failure blocks, and the loop may not modify,
override or recompute any of them. Changing a number here would be a
source change under review, not a runtime decision.

**R1 — Pre-registration before data.** The trial's primary hypothesis,
method, decision rule, denominator, horizon and stopping rule are written
to `docs/preregistration_<trial>.md` and committed *before* the
confirmatory partition is read. The loop may generate that document only
from the queue entry it is confirming — it may not author a new hypothesis.
This mirrors what `research synthetic-ev` already enforces: nobody gets to
choose when to run it, or to run it again.

**R2 — Exploratory support first, measured.** A confirmatory trial may only
follow a completed exploratory screen on the same question, by a registered
harness, whose recorded verdict is in `SUPPORTING_VERDICTS`. A screen that
failed, found no improvement, or had insufficient sample is not support.
This is already how `research_screens.produced_support` works; R2 makes it
a precondition rather than a signal.

**R3 — Independent data, declared in advance.** The confirmatory partition
must not overlap the DEVELOPMENT rows the screen used
(`research/partitions.py`), and the effective sample in that partition —
independent observation dates with average uniqueness applied, never row
counts — must meet the pre-registered minimum before the trial opens. If it
does not, the outcome is `AWAIT_DATA`. Shortening a window, lowering a
threshold or re-defining the denominator to reach the minimum is not
available to the loop, in this rule set or any other.

**R4 — Budget, counted at opening.** The quarterly budget of six primary
hypotheses (`GOVERNANCE.md` §1.2) is charged when the trial is **opened and
measured**, regardless of outcome. The loop may open at most **two** of the
six per quarter autonomously; the remaining four stay human-opened. Two is
proposed rather than six because a budget the loop can exhaust alone leaves
no room for a human to test something the loop did not rank highly, and
because a smaller first allowance is the honest way to start a mechanism
nobody has operated yet.

**R5 — One open confirmatory trial at a time.** Concurrency hides
peeking: two trials open at once make it tempting to read the first result
while the second is running. Sequential and pre-registered.

**R6 — No re-running, ever.** A confirmatory trial produces exactly one
result. A re-run voids the trial and is recorded as void — it does not
replace the result. `research synthetic-ev --force` already behaves this
way, and R6 denies the loop that flag entirely.

**R7 — Negative results are published to the same place as positive ones.**
The result is written to `research_trials`, the negative memory
(`learning/failed_hypotheses.py`) and the monthly report whatever it says.
A loop that could quietly drop a disconfirmation would be selecting its own
record.

**R8 — Promotion stays human.** A confirmatory result that clears every
ladder threshold produces `AWAIT_PROMOTION_APPROVAL` and stops. The loop
never writes `alpha_sources`, never moves a queue entry past `MEASURED`,
and never sets a promotion gate. §2.1 must still be resolved by a human
before any promotion, autonomous or not.

**R9 — A shelved family stays shelved until a human un-shelves it.**
`research_shelf` has no automatic expiry. The retest gate in
`failed_hypotheses.allow_retest` requires a materially different
condition, and "a quarter has passed" is not one.

**R10 — A kill switch that defaults to off.** A single configuration flag
disables autonomous confirmatory opening with no code change. Its absence
means disabled. Any run in which R1–R9 cannot all be evaluated — a missing
partition, an unreadable registry, an unparseable pre-registration — is
treated as a failure of this rule, not as a pass.

---

## 4. What would make this proposal wrong

Stated so that a reader can check it rather than take it on trust:

- **If a screen's verdict is a weak predictor of a confirmatory result**,
  R2 is not a filter but a rubber stamp, and the loop would spend the
  quarterly budget on questions a human would not have chosen. There is no
  evidence either way yet: as of 2026-10-24 the loop has executed no
  completed screen, because the forward ledger has nothing matured and
  labelled. **This proposal should not be activated before several screens
  and at least one human-opened confirmatory trial exist to compare
  against.**
- **If pre-registration documents generated from queue entries turn out to
  be vague**, R1 protects nothing. A human should read the first two the
  loop generates before any of them gates a trial.
- **If the two preconditions in §2 are resolved by relaxing limits rather
  than by choosing between them**, the whole frame is compromised,
  regardless of what §3 says.

---

## 5. How activation would work

Not implemented. The sequence would be:

1. A human resolves §2.2 (the three limit divergences) and records the
   decision in `GOVERNANCE.md` with a date and a reason.
2. A human decides whether §2.1 must be resolved first.
3. R1–R10 are implemented as checked constants, each with its own test,
   and the kill switch of R10 is added, defaulting to off.
4. A human reviews this document, amends it, and records approval in
   `GOVERNANCE.md` as a dated amendment naming the approver — the same
   mechanism as §1.1a.
5. Only then is the switch turned on, by a human.

The system may draft steps 3 and 4's text. It may not perform step 1, 2, 4
or 5. `controller.decide_next_action` carries `promotion_unlocked` as a
parameter that exists only to be `False`, threaded through so that "this is
locked" is recorded in the evidence rather than being invisible; whatever
replaces it for confirmatory autonomy should be built the same way.

---

*Written 2026-10-24 by the research loop, at the request of the operator,
as a proposal only. No code reads this file.*
