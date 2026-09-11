"""Pricing / cost engine (Master Spec Phase 1, §13).

Sub-modules:

- :mod:`turboedge.pricing.intrinsic` -- intrinsic value, implied underlying, leverage (§13.1).
- :mod:`turboedge.pricing.financing` -- implied financing spread inference (§13.3).
- :mod:`turboedge.pricing.gap_premium` -- overnight/weekend gap premium (§11.2, §13.4).
- :mod:`turboedge.pricing.issuer_margin` -- ask-price decomposition (§13.2).
- :mod:`turboedge.pricing.cross_issuer` -- cross-issuer consensus & dislocation scores (§13.5).
- :mod:`turboedge.pricing.integrity` -- product-level integrity checks (§6).
"""

from __future__ import annotations
