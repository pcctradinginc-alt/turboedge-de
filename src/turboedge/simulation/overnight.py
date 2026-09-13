"""Per-trading-day OHLC decomposition into overnight/intraday log-return components.

Formula reference: Master Spec §11.2 ("Overnight und Weekend getrennt") and
§26 ("Ambiguous Bars" -- discarded, never optimistically imputed, CLAUDE.md
rule 17/29).

Turbo-relevant path risk lives in two structurally different places within a
single daily OHLC bar:

- the **overnight/weekend gap** ``g = ln(Open_t / Close_{t-1})`` -- a jump
  the holder cannot react to (issuer's gap-premium risk, ``pricing/gap_premium.py``);
- the **intraday path** on day ``t`` itself -- ``ln(Close_t/Open_t)`` for the
  day's net move, and ``ln(High_t/Open_t)`` / ``ln(Low_t/Open_t)`` for the
  extremes a knockout barrier could have been touched at.

``bootstrap.py`` resamples these four numbers jointly, per trading day, so
that whatever historical co-movement exists between a day's overnight gap and
its own intraday range is preserved by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from turboedge.storage.schemas import UnderlyingBar

# A gap is a "weekend" gap (Master Spec §11.2: "Friday close -> Monday open")
# whenever more than one calendar day separates the two bars -- covers both
# ordinary weekends and holidays, deliberately not distinguished further
# (holiday calendars are out of scope for this milestone, documented in
# ``paths.py``).
_WEEKEND_GAP_CALENDAR_DAYS = 1

# Relative tolerance on the H >= max(O,C) / L <= min(O,C) consistency check,
# purely to absorb floating-point rounding in upstream vendor data -- not a
# license to accept genuinely inverted bars (CLAUDE.md rule 17: ambiguous /
# broken bars are discarded, never silently repaired).
_CONSISTENCY_TOLERANCE_REL = 1e-9


@dataclass(frozen=True, slots=True)
class DailyComponents:
    """Empirical per-trading-day log-return components, one entry per day.

    Entry ``i`` describes trading day ``dates[i]``: the overnight/weekend gap
    *into* that day (``gap[i] = ln(Open_i / Close_{i-1})``, using the
    previous *consistent* bar's close) and that day's own intraday session
    (``intraday[i] = ln(Close_i/Open_i)``, ``hi[i] = ln(High_i/Open_i) >= 0``,
    ``lo[i] = ln(Low_i/Open_i) <= 0``). Because ``gap`` needs a previous
    close, there is no entry for the very first bar in the input.

    ``n_discarded`` counts input bars dropped for internal OHLC inconsistency
    (``High < max(Open,Close)`` or ``Low > min(Open,Close)``) or a
    non-positive/duplicate calendar gap to the previous bar -- Master Spec
    §26: ambiguous/broken bars are never optimistically repaired, only
    counted and excluded.
    """

    dates: list[date]
    gap: npt.NDArray[np.float64]
    weekend_before: npt.NDArray[np.bool_]
    intraday: npt.NDArray[np.float64]
    hi: npt.NDArray[np.float64]
    lo: npt.NDArray[np.float64]
    n_discarded: int

    def __post_init__(self) -> None:
        n = self.gap.size
        for name, arr in (
            ("weekend_before", self.weekend_before),
            ("intraday", self.intraday),
            ("hi", self.hi),
            ("lo", self.lo),
        ):
            if arr.size != n:
                raise ValueError(f"{name} has size {arr.size}, expected {n} (== gap.size)")
        if len(self.dates) != n:
            raise ValueError(f"dates has length {len(self.dates)}, expected {n} (== gap.size)")


def _bar_is_consistent(open_: float, high: float, low: float, close: float) -> bool:
    if not (open_ > 0 and high > 0 and low > 0 and close > 0):
        return False
    scale = max(open_, high, low, close)
    tol = scale * _CONSISTENCY_TOLERANCE_REL
    return bool(high >= max(open_, close) - tol and low <= min(open_, close) + tol)


def daily_components_from_bars(bars: Sequence[UnderlyingBar]) -> DailyComponents:
    """Build a :class:`DailyComponents` series from consecutive daily OHLC bars.

    Bars are sorted by ``ts`` first. Any bar failing the ``High >=
    max(Open,Close)`` / ``Low <= min(Open,Close)`` consistency check (or with
    a non-positive price) is discarded outright (counted, never repaired).
    Among the remaining consistent bars, each consecutive pair contributes
    one :class:`DailyComponents` entry for the *later* bar; a pair with a
    non-positive or duplicate calendar-day gap (out-of-order/duplicate
    timestamps) is skipped and also counted in ``n_discarded``.
    """
    ordered = sorted(bars, key=lambda bar: bar.ts)
    consistent = [b for b in ordered if _bar_is_consistent(b.open, b.high, b.low, b.close)]
    n_discarded = len(ordered) - len(consistent)

    dates: list[date] = []
    gap: list[float] = []
    weekend_before: list[bool] = []
    intraday: list[float] = []
    hi: list[float] = []
    lo: list[float] = []

    for prev_bar, bar in pairwise(consistent):
        gap_days = (bar.ts.date() - prev_bar.ts.date()).days
        if gap_days <= 0:
            n_discarded += 1
            continue
        dates.append(bar.ts.date())
        gap.append(float(np.log(bar.open / prev_bar.close)))
        weekend_before.append(gap_days > _WEEKEND_GAP_CALENDAR_DAYS)
        intraday.append(float(np.log(bar.close / bar.open)))
        hi.append(float(np.log(bar.high / bar.open)))
        lo.append(float(np.log(bar.low / bar.open)))

    return DailyComponents(
        dates=dates,
        gap=np.asarray(gap, dtype=np.float64),
        weekend_before=np.asarray(weekend_before, dtype=np.bool_),
        intraday=np.asarray(intraday, dtype=np.float64),
        hi=np.asarray(hi, dtype=np.float64),
        lo=np.asarray(lo, dtype=np.float64),
        n_discarded=n_discarded,
    )
