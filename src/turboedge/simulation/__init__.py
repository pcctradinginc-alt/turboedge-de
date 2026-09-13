"""Path & payoff simulation engine (Master Spec §11-§13, §16, §26).

Sub-modules:

- :mod:`turboedge.simulation.overnight` -- per-trading-day OHLC decomposition
  into overnight-gap / intraday / high / low log-return components (§11.2).
- :mod:`turboedge.simulation.bootstrap` -- block, weekend-conditioned and
  regime-conditioned resampling of those daily components (§11.1 A/B).
- :mod:`turboedge.simulation.paths` -- :func:`~turboedge.simulation.paths.simulate_paths`,
  the public entry point combining bootstrap/Monte Carlo path generation
  (§11.1 A/B/C).
- :mod:`turboedge.simulation.barrier` -- discrete and Brownian-bridge-corrected
  barrier-touch detection (§11.3).
- :mod:`turboedge.simulation.payoff` -- full turbo/knockout payoff evaluation
  over simulated paths (§13, §16).
"""

from __future__ import annotations
