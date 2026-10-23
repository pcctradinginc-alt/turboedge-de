"""Which part of a forecast actually reaches the economic evaluation.

This is the measurement behind `docs/measured_results.md` §6.15 -- a model
that improved CRPS in 18 of 20 cells at p ~ 0.014 and was then measured
economically worse than doing nothing. The two facts are not in tension;
they describe different quantities, and these tests pin which.

A `HorizonForecast` carries, among others:

    mean         E[h-day log return]
    sigma        std of the h-day log return   -- the PREDICTIVE WIDTH
    uncertainty  standard error of `mean`      -- estimation error of a point
    quantiles    q05..q95                      -- what CRPS is computed on

`ranking/ev.py` consumes `mean` (as the path drift) and `uncertainty` (to
shift that drift in the optimistic/pessimistic scenarios, and in position
sizing). It does **not** consume `sigma` or `quantiles`: the simulated path
volatility always comes from the historical bootstrap over `bars`, and
`simulate_paths` is called without `target_sigma` (ev.py:252).

So CRPS scores the quantiles while the economics read the first moment plus
the standard error of that moment. A model predicting a narrower or wider
distribution produces the same paths as the null.

These tests change nothing. They make the boundary explicit, in both
directions:

* if someone starts feeding width into the simulation, the test naming that
  absence fails, and the change is forced to be deliberate and
  pre-registered rather than incidental;
* if someone removes the drift or sizing channels that DO work, that fails
  too.

Whether width *should* be fed through is an open research question, not a
cleanup task: `simulate_paths` cannot scale `gap` (CLAUDE.md rule 16), so a
target below the gap-only contribution is unreachable and `target_sigma_met`
would report False. It needs a pre-registered trial, not a patch.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import turboedge.ranking.ev as ev_module
from turboedge.models.forecast import HorizonForecast

EV_SOURCE = Path(inspect.getsourcefile(ev_module) or "")


def _attributes_read_in(function_name: str) -> set[str]:
    """Every `<something>.<attr>` read inside one function of ranking/ev.py."""
    tree = ast.parse(EV_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {
                n.attr
                for n in ast.walk(node)
                if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)
            }
    raise AssertionError(f"{function_name} not found in {EV_SOURCE.name}")


def test_the_forecast_object_still_separates_width_from_point_uncertainty() -> None:
    # The whole finding rests on these being different fields. If they are
    # ever merged, every test below becomes meaningless.
    fields = {f.name for f in dataclasses.fields(HorizonForecast)}

    assert {"mean", "sigma", "uncertainty", "quantiles"} <= fields


def test_the_scenario_drift_uses_the_mean_and_the_point_uncertainty() -> None:
    # What DOES reach the economics. Measured at ranking/ev.py:214-226.
    read = _attributes_read_in("_drift_for_scenario")

    assert "mean" in read
    assert "uncertainty" in read


def test_the_scenario_drift_does_not_use_the_predictive_width() -> None:
    # The finding itself. `sigma` is the quantity CRPS rewards and it does
    # not enter the drift.
    read = _attributes_read_in("_drift_for_scenario")

    assert "sigma" not in read, (
        "ranking/ev.py:_drift_for_scenario now reads forecast.sigma. That is a "
        "change in what the economic model consumes, and GOVERNANCE.md requires "
        "it to go through a pre-registered trial rather than arrive as a patch."
    )
    assert "quantiles" not in read


def test_the_path_simulation_is_called_without_a_volatility_target() -> None:
    # `simulate_paths` accepts `target_sigma`; the production call at
    # ranking/ev.py:252 does not pass it, so path volatility always comes
    # from the historical bootstrap over `bars`.
    tree = ast.parse(EV_SOURCE.read_text(encoding="utf-8"))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "simulate_paths"
    ]

    assert calls, "ranking/ev.py no longer calls simulate_paths"
    for call in calls:
        passed = {kw.arg for kw in call.keywords}
        assert "target_sigma" not in passed, (
            "ranking/ev.py now passes target_sigma into the production path "
            "simulation. This changes the economics of every candidate. It needs "
            "a pre-registered trial and a human decision -- and note that "
            "simulate_paths cannot scale `gap` (CLAUDE.md rule 16), so a target "
            "below the gap-only contribution is unreachable and target_sigma_met "
            "reports False."
        )


def test_the_predictive_width_is_only_logged_never_consumed() -> None:
    # `forecast.sigma` does appear once in ranking/ev.py, at line ~438 --
    # inside a `logger.error(...)` diagnostic for an implausible mean, which
    # is followed by `continue`. Logged, not consumed. Recorded here so the
    # earlier assertions are not mistaken for "sigma appears nowhere".
    source = EV_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)

    logged = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"error", "warning", "info", "debug"}
        ):
            logged |= {kw.arg for kw in node.keywords if kw.arg}

    assert "forecast_sigma" in logged, (
        "forecast.sigma is no longer logged in ranking/ev.py. If it moved into "
        "the economics instead, that is the change the other tests here guard."
    )
