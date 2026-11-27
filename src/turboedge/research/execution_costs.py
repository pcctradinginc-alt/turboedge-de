"""Explicit hypothetical costs on observed ask-to-bid returns, in EUR.

No spread or financing charge is added: both are already embedded in
the observed prices. This is a research sensitivity, not a broker tariff.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class ExecutionCosts:
    fixed_eur_per_order: float
    variable_bps_per_order: float
    slippage_bps_per_side: float

    def __post_init__(self) -> None:
        if not all(
            isfinite(x) and x >= 0
            for x in (
                self.fixed_eur_per_order,
                self.variable_bps_per_order,
                self.slippage_bps_per_side,
            )
        ):
            raise ValueError("execution costs must be finite and nonnegative")
        if max(self.slippage_bps_per_side, self.variable_bps_per_order) >= 10_000:
            raise ValueError("slippage and variable fees must be below 100 percent")

    def net_return(self, observed_return: float, notional_eur: float) -> float:
        """P&L divided by pre-cost entry notional, not a cash-budget return.

        Entry notional buys the quoted quantity. Slippage is additional
        adverse price movement on both sides. Variable fees apply to the
        slipped execution values; fixed fees reserve both orders even on KO.
        """
        if not isfinite(notional_eur) or notional_eur <= 0:
            raise ValueError("notional must be finite and positive")
        if not isfinite(observed_return) or observed_return < -1:
            raise ValueError("observed return must be finite and at least -1")
        slip = self.slippage_bps_per_side / 10_000
        fee = self.variable_bps_per_order / 10_000
        buy = 1 + slip
        sell = (1 + observed_return) * (1 - slip)
        return sell - buy - fee * (buy + sell) - 2 * self.fixed_eur_per_order / notional_eur


# Deliberately illustrative, not a claim about the user's broker.
COST_SCENARIOS = {
    "zero": ExecutionCosts(0, 0, 0),
    "low": ExecutionCosts(1, 0, 5),
    "medium": ExecutionCosts(5, 5, 10),
    "stress": ExecutionCosts(10, 10, 25),
}
NOTIONALS_EUR = (250, 500, 1000, 2500, 5000)

# Fixed ex-ante research convention, not a broker tariff or a live allocation.
PRIMARY_SCENARIO = "medium"
PRIMARY_NOTIONAL_EUR = 1000


def research_net_return(observed_return: float) -> float:
    return COST_SCENARIOS[PRIMARY_SCENARIO].net_return(observed_return, PRIMARY_NOTIONAL_EUR)
