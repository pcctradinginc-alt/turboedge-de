# Pre-registration: 2026Q4-002 — does the *width* of the better forecast carry economic value?

**Written 2026-09-26. Frozen on commit. No measurement against this
specification has been run.**

Second trial under GOVERNANCE.md §11.2's one-primary-hypothesis rule, and the
direct successor to 2026Q4-001, which failed. That trial tested the only
channel the EV pipeline could feel at the time — the forecast mean — and
returned `mean ΔLCB_NetEV = -0.020547`, p = 1.0000
(`docs/measured_results.md` §6.15).

This trial tests the other channel, which three separate measurements now say
is the better-supported one:

* §6.12: `regime_conditional`'s CRPS advantage is real within its own wave,
  p ≈ 0.0135 under a moving-block bootstrap that respects both overlapping
  label windows and cross-cell correlation.
* §6.14: the EV pipeline never saw that advantage. `simulate_paths` had no
  volatility parameter, so path dispersion came entirely from history and was
  identical across both arms of 2026Q4-001.
* §6.16: the advantage is *sharpness, not overconfidence*.
  `regime_conditional` predicts total-horizon sigmas 26-38% narrower than the
  null and covers **better** out of sample on both interval levels
  (0.879/0.490 against nominal 0.90/0.50, where the null gives 0.870/0.473).

`simulate_paths` now accepts `target_sigma` (commit `b70c050`), so the channel
is reachable for the first time.

---

## 1. Primary hypothesis — exactly one

Same standardised turbo universe and the same machinery as 2026Q4-001, with
**one difference**: each arm's path dispersion is set to that arm's own
forecast sigma via `target_sigma`, instead of both arms inheriting the
historical dispersion.

    H0:  LCB_NetEV(regime_conditional, own sigma) - LCB_NetEV(null, own sigma)  <=  0
    H1:  ... > 0

**One primary p-value.** One-sided, α = 0.10, moving-block bootstrap over
prediction dates, block length 21.

## 2. The known bias, and why the design does not launder it

A narrower predicted distribution lowers knock-out probability mechanically.
`regime_conditional` predicts sigmas 0.62-0.76× the null's, so **it will look
better on EV for that reason alone, whether or not the narrowness is earned.**
This is the central threat to this trial's validity and it is stated here
rather than discovered later.

Three things bear on it, all fixed in advance:

1. **Coverage is the earned-ness check, and it was measured before this
   trial** (§6.16): narrower intervals that cover *better* are sharper, not
   overconfident. Had coverage been below the null's, this trial would not be
   worth running, and that check was not chosen after seeing any EV number.
2. **Both arms use their own sigma.** The null is not handicapped by being
   held at historical dispersion while the challenger gets its forecast.
3. **A pre-registered falsification arm** (§6) reruns the comparison with both
   arms forced to the *null's* sigma. If the effect survives there, it is not
   coming from width, and the primary result is confounded. This arm exists
   specifically so a PASS can be attacked, and its outcome is reported
   whatever it says.

## 3. Unreachable sigmas — decided before the result

`gap` cannot be scaled (rule 16), so a target below the gap-only floor is
unreachable and `simulate_paths` clamps with `target_sigma_met=False`.

Measured on the live grid before writing this: **9 of 10 checked cells are
reachable.** The exception is NDX h=10d, where `regime_conditional` predicts
0.02239 against a gap floor of 0.02389 — its predicted width is narrower than
historical overnight gap risk alone.

**Unreachable cells are included at the clamped sigma and counted.** They are
not dropped. Dropping a cell once its result is visible would be a
researcher's choice dressed as a data-quality decision; dropping it in advance
would silently discard the cells where the model is most aggressive, which is
exactly where its claim is most testable. The count of clamped cells, per arm,
is reported with the result, and §6 additionally reports the primary statistic
recomputed on reachable cells only — as stability analysis, never as the
primary.

## 4. Frozen method

Identical to 2026Q4-001 §4 as amended (Amendment B's `lcb_net_return` primary
statistic and the fair-value-straddling quote), with the single change in §1.

Models `NullModel` and `RegimeConditionalEmpiricalModel` exactly as at this
commit. Data: `YFinancePriceAdapter`, `lookback_days=4000`, DAX/NDX/EURUSD/XAU;
fetch date and per-underlying bar counts recorded with the result. Walk-forward
`min_train=750`, `step=21`, `embargo=horizon`, horizons 3/5/7/10/14. Grid:
barrier distances 2/5/10/15/20% each side, `ratio` 0.01, `fx` 1.0, `spread`
0.005 straddling fair value, `financing_spread` and `ref_rate` 0.02,
`exit_spread_pct` 0.005. 2,000 paths, seed 20261002, identical across arms.

## 5. Decision rule — stated before the result

| outcome | reading |
|---|---|
| p ≤ 0.10, mean Δ > 0, **and** the §6 falsification arm shows no effect | **PASS.** The forecast's width carries economic value on standardised terms. Necessary condition met; forward real-product arm becomes the next trial. Still no promotion. |
| p ≤ 0.10, mean Δ > 0, but the falsification arm **also** shows an effect | **CONFOUNDED.** Reported as such, promoted to nothing. The effect is not attributable to width. |
| p > 0.10 | **FAIL** → `failed_hypotheses.json`. The distributional improvement is real, measurable and economically inert through both channels. |
| mean Δ ≤ 0 at p ≤ 0.10 | **FAIL**, reported as evidence the sharper forecast is economically *harmful*. |

No threshold, cost assumption or grid may move after seeing the result. If the
cost assumptions dominate, that is the finding.

## 6. Stability analysis (secondary, never primary)

The falsification arm (both arms at the null's sigma). Per-underlying,
per-horizon and per-barrier-distance Δ. Share of cells with Δ > 0. Count of
clamped cells per arm. Primary statistic recomputed on reachable cells only.
Sign sensitivity to `spread` at 0.0025 and 0.01. Realised-vs-requested sigma
per arm, so a reader can confirm the mechanism did what §1 claims.

## 7. Budget and denominator

Charged to **2026Q4**, trial id assigned on the run date. Q4 opens 2026-10-01
at 0 of 6; this reserves the first unit. 2026Q4-001 was brought forward and
charged to Q3 as its 12th of 6 (its Amendment C), so it does not consume a Q4
unit.

**2026Q4 primary hypotheses pre-registered: 1.** BH denominator is the count of
Q4 primary hypotheses, not a cell count (§11.2).

Unlike 2026Q4-001, this trial is **not** brought forward. That override was
justified by having a frozen specification and an unanswered question; here the
argument does not hold, because the mechanism it depends on was written today
and has one day of scrutiny. Waiting costs five days and buys a chance to find
a defect in `_apply_volatility_scaling` before a result is attached to it.

## 8. Pre-committed statement of ignorance

The honest prior is still FAIL, but less confidently than for 2026Q4-001, and
the reason is §6.16: this is the first channel in this repository where the
measured evidence points the right way before the economics are tested.

Against that: 2026Q4-001 measured the mean channel at -0.0205, spread plus
financing is ~50bp per leg, and knock-out asymmetry does not care how well
calibrated a forecast is. A 26-38% narrower distribution is a large change in
KO probability, so the effect could be large in either direction — which is
also why the falsification arm is not optional.

If it passes, the first question is whether the coverage advantage in §6.16
holds in the specific regimes where the KO probability moved most, not whether
an edge has been found.

---

## 9. Amendment A (2026-09-26, before any measurement) — §2's stated bias is backwards

**Written before any measurement of the primary hypothesis.** Found while
building the harness, verified independently against real DAX bars rather than
taken on report.

### §2 claimed the wrong direction

§2 states: *"A narrower predicted distribution lowers knock-out probability
mechanically... it will look better on EV for that reason alone."* **That is
false**, and the error is not small.

Measured directly — identical terms, LONG, zero drift, DAX h=7d, 8,000 paths,
one seed, varying only `target_sigma` between the null's 0.0282 and
`regime_conditional`'s 0.0204:

| barrier distance | mean net return, σ=0.0282 | σ=0.0204 | Δ | p_ko 0.0282 | p_ko 0.0204 |
|---|---|---|---|---|---|
| 2% | +0.0545 | -0.0046 | **-0.0591** | 0.375 | 0.293 |
| 5% | -0.0064 | -0.0189 | **-0.0126** | 0.061 | 0.027 |
| 10% | -0.0096 | -0.0127 | -0.0030 | 0.012 | 0.000 |
| 15% | -0.0085 | -0.0098 | -0.0012 | 0.000 | 0.000 |
| 20% | -0.0074 | -0.0083 | -0.0009 | 0.000 | 0.000 |

The narrower distribution *does* lower knock-out probability, exactly as §2
said — 0.375 to 0.293 at the 2% barrier, 0.061 to 0.027 at 5%. And the payoff
still gets **worse at every barrier distance.**

### Why

A turbo is a convex claim on the underlying: leverage means the payoff rises
faster than the underlying on the upside. For a convex payoff, higher
dispersion raises the expectation (Jensen). The knock-out truncates the left
tail, which is what §2 was reasoning about, but the convexity in the surviving
right tail dominates — narrowing gives up more upside than it saves in
knock-out losses. The effect is strongest where leverage is highest, which is
why it is largest at the 2% barrier and fades monotonically outward.

### Consequences, all stated before the result

1. **The mechanical bias runs *against* `regime_conditional`, not for it.**
   §2's defence was built against a headwind that does not exist and a
   tailwind that does not exist either. The falsification arm in §6 stays —
   it now guards against a mean-channel effect leaking in, which is a real
   risk (§6.15 measured that channel) — but it is no longer protecting
   against a width tailwind, because there is none.
2. **A PASS would be stronger evidence than §2 implied, not weaker.** If a
   26-38% narrower forecast produces a positive ΔLCB_NetEV *despite* a
   convexity headwind of this size, that is a substantial result.
3. **§8's prior moves toward FAIL, and for a new reason.** The mechanical
   direction is now known to be adverse and large: -0.0591 at the 2% barrier
   against the -0.0205 that 2026Q4-001 measured for the whole mean channel.
   For the trial to pass, the forecast's accuracy would have to beat a
   structural headwind bigger than the entire effect the previous trial
   measured.

**No part of §1, §4, §5 or §6 changes.** The hypothesis, the method, the
decision rule and the stability analysis are as specified. What changes is that
a premise I stated about the mechanism was wrong, and the correction is
recorded here rather than discovered while reading a result — which is the
whole reason the premise was written down in advance.

### One thing this exposes beyond the trial

`p_ko` falling while expected return falls means **knock-out probability and
expected return are not monotonically related** in this product class. A gate
built on both (`lcb_ev > 0` *and* `p_ko` acceptable) is therefore not
double-counting one risk; it is reading two genuinely different ones. That is
worth knowing independently of this trial's outcome.


---

## 10. How it runs (added 2026-09-27)

`turboedge research synthetic-ev --trial 2026Q4-002`, from the monthly pipeline
job — the first scheduled job on or after the opening date. Automated for the
same reason 2026Q4-001 was: nobody chooses when it runs, and nobody can run it
again. That is what separates it from the retrospective evidence in
`docs/measured_results.md` §6.12.

Four guards, each with its own exit code, all verified by running them:

| guard | exit |
|---|---|
| before this trial's opening date (2026-10-01) | 2 |
| `docs/results_2026Q4_002.json` already exists | 3 |
| unknown trial name | 4 |
| `--force` overrides guard 2 and says so — and voids the trial | — |

`--not-before` can override the date, but an override requires a written
amendment in this document, as 2026Q4-001's Amendment C is. The guard is not
paperwork: it is the difference between a trial that ran once on a fixed date
and one whose timing was a choice.

Every monthly run after the trial completes will legitimately refuse, so the
step is `continue-on-error`. The result is committed to version control beside
this document with its provenance — fetch date, per-underlying bar counts after
the OHLC filter, git commit, the `sigma_source` used, and whether the run was
forced.

---

## 11. Amendment B (2026-09-27, before any measurement) — the scrutiny §7 asked for, and one scope limit it found

§7 declined to bring this trial forward on the grounds that *"the mechanism it
depends on was written today and has one day of scrutiny."* The purpose was
scrutiny, not the calendar. This records the scrutiny, performed before any
measurement of the primary hypothesis.

### Three adversarial checks on the mechanism

**1. Units — `forecast.sigma` is the quantity `target_sigma` expects.** The
most damaging undetected defect class would be feeding a differently-scaled
number. Checked against each model's own quantiles, where a normal
distribution implies `sigma ≈ (q95-q05)/3.29`:

| model | cell | `sigma` | implied by quantiles | ratio |
|---|---|---|---|---|
| null | DAX 3d | 0.01885 | 0.01792 | 0.95× |
| null | NDX 14d | 0.04853 | 0.05144 | 1.06× |
| regime_conditional | DAX 3d | 0.01392 | 0.01408 | 1.01× |
| regime_conditional | NDX 14d | 0.03026 | 0.03372 | 1.11× |

Consistent to 0.92-1.11×, with the spread explained by fat tails rather than a
scaling error. No units defect.

**2. Targeting is exact.** Reachable targets are hit to 0.00% on real DAX bars;
unreachable ones clamp at the gap-only floor with `target_sigma_met=False`
rather than silently returning a different dispersion. Drift is preserved to
six decimals when scaling and tilting are combined, `gap` is bit-identical, and
repeated runs are bit-identical.

**3. Shape largely transfers, not just width.** Scaling matches the second
moment; whether it also transmits the forecast's *shape* was untested. Measured
at `target_sigma = forecast.sigma`, 20,000 paths:

| h | τ=0.05 forecast / paths | τ=0.50 | τ=0.95 | path excess kurtosis |
|---|---|---|---|---|
| 3d | -0.02805 / **-0.02428** | 0.00032 / 0.00015 | 0.01826 / 0.01973 | +2.37 |
| 14d | -0.05737 / **-0.05588** | -0.00398 / -0.00399 | 0.03751 / 0.03851 | +0.77 |

Realised sigma matches the target exactly in both. Quantiles agree to
0.0001-0.0038 in log return, because the bootstrap already carries realistic
fat tails.

### The scope limit this found

**The paths' left tail is systematically thinner than the forecast's.** At
h=3d the forecast puts q05 at -0.0281 and the scaled paths at -0.0243 — a gap
of 0.0038 in log return, about 0.38% of the underlying level, shrinking to
0.0015 by h=14d.

For a long turbo a thinner left tail **understates knock-out risk**, which is
the anti-conservative direction. The effect is largest where leverage is
highest, so it partly offsets the convexity headwind Amendment A measured
(-0.0591 at the 2% barrier) rather than adding to it.

This is a limit of transmitting a distribution through its second moment, not a
defect: `simulate_paths` takes a sigma, not a shape. **Recorded as a bound on
what a PASS would mean** — a PASS would show the forecast's *width* carries
economic value, with its left-tail shape approximated by history rather than
taken from the forecast. §1, §4, §5 and §6 are unchanged.

### Verdict on the mechanism

No defect found. The scrutiny §7 asked for has been performed rather than
waited out.

---

## 12. Amendment C (2026-09-27, before any measurement) — run brought forward

**Written before the run.** The date guard is overridden and the trial runs on
2026-09-27 instead of 2026-10-01.

### Why the §7 reason is discharged

§7 gave one reason for waiting: *"waiting five days buys a chance to find a
defect in `_apply_volatility_scaling` before a result is attached to it."*
Amendment B performed that scrutiny — units, exactness, clamping, gap
invariance, drift interaction, determinism, and shape transfer — plus 41 tests
on the scaling and 55 on the harness. No defect was found, and one scope limit
was, and is recorded.

The reason was scrutiny, not the calendar. It has been discharged by doing the
work.

### What it costs, stated plainly

**Budget.** Charged to **2026Q3 as the 13th of 6**, not to Q4. Q3 already
stands at 12 of 6 after 2026Q4-001's own Amendment C. Making this a Q4 unit
would repeat the `PD-2026Q4-001..003` error that GOVERNANCE.md §11.5 exists to
record.

Deflation: the Q3 model-level count moves from twelve to thirteen. The rank-2
BH threshold moves from `2/12·0.10 = 0.0167` to `2/13·0.10 = 0.0154`; the Phase
D p-values of 0.0135 and 0.0142 both still clear it, so §6.13's conclusion is
unaffected.

**The precedent, which is the real cost.** This is the **second** date-guard
override in two days. A guard overridden twice is closer to a suggestion than
a rule, and saying otherwise would be dishonest. Two things limit the damage:
each override is written down before the run with its reason and its price, and
neither touched the hypothesis, the method or the decision rule — which is
where an override would actually corrupt a result.

**A third override should not happen without the calendar advancing.** If a
future trial's specification wants a waiting period, the waiting period should
be met or the specification should not claim one.

### Why now

The standing goal is to find an economic edge if one exists and report it.
2026Q4-001 closed the mean channel. This is the only remaining channel with
measured evidence pointing the right way (§6.16: 26-38% narrower intervals that
cover better), the mechanism to test it exists and has been scrutinised, and
four days of calendar buys nothing further that the scrutiny has not already
bought.

§8's prior stands: **FAIL**, and Amendment A strengthened it — the convexity
headwind at the 2% barrier (-0.0591) is larger than the entire mean-channel
effect 2026Q4-001 measured (-0.0205).
