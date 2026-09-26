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
