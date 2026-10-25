"""How much capital the portfolio already has deployed, as a fraction.

WHY THIS EXISTS
---------------
`ranking/sizing.py` applies two portfolio-level caps net of what is already
used:

    fraction = min(fraction, max_cluster_fraction - cluster_fraction_used)
    fraction = min(fraction, max_total_risk     - total_risk_used)

Measured 2026-10-24: nothing ever supplied either argument. `pipeline/scan.py`
called `evaluate_product_horizons` without them, both defaulted to `0.0`, and
so both caps reduced to per-position caps of 15% and 30% -- each looser than
the 5% single-position cap, which means **neither ever bound**. Of the four
caps in `SizingConfig`, two were structurally inert in production, and
`ranking/utility.py:86-88` names one of them as a *primary* defence against
cluster and ruin risk.

That is an absent input being treated as a satisfied constraint, which is
the pattern this repository refuses elsewhere (`alpha/promotion.py` keeps
unmeasured gates as `None` and reports `is_complete = False` rather than
defaulting them). This module closes it the same way: it computes the real
usage when it can, and says so plainly when it cannot.

WHAT IT CAN AND CANNOT KNOW
---------------------------
The system never executes. The user executes manually, so the only record of
capital actually deployed is `positions_manual` -- `qty x entry_price`, in
EUR. Turning that into a fraction needs total research capital
(`risk.research_capital_eur`), which is a fact about the user's account and
is deliberately unset by default.

`forward_ledger` is NOT used, even though it carries
`suggested_position_fraction` and would need no capital figure. A ledger row
is a *proposal*; its `OPEN` status means "awaiting its label", not "the user
took it", and there is no link between `positions_manual` and
`forward_ledger` at all. Counting proposals would make the system behave as
if every suggestion had been acted on -- a number that looks right and is
wrong, which is worse than the honest gap.

Attributing a held position to a correlation cluster needs
`isin -> underlying_id` (from `product_snapshots`) and then
`underlying_id -> cluster_id`. A position whose ISIN was never scanned
cannot be attributed; it still counts toward the portfolio total, which is
cluster-agnostic, and is reported in `unattributed` rather than dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from turboedge.storage.schemas import Category, LedgerEntryStatus, PositionStatus

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from turboedge.storage.duckdb import Store

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class HeldFraction:
    """One open position's capital fraction and the cluster it belongs to.

    Kept per position rather than pre-summed per cluster because
    `ranking/cluster.py` counts positions as well as capital
    (`max_active_positions_per_cluster`). Handing it one aggregated entry
    per cluster would make three holdings count as one position.
    """

    position_id: str
    underlying_id: str | None
    cluster_id: str | None
    issuer: str | None
    fraction: float
    #: True for an ACTIONABLE proposal already emitted and not yet
    #: resolved, as opposed to a position actually held. Both consume the
    #: worst-case budgets -- GOVERNANCE.md 5.2 counts "bereits verbindlich
    #: reservierte neue Positionen" in the 8% total -- but only one of them
    #: is real, and a report that conflated them would misstate what is
    #: held.
    reserved: bool = False


@dataclass(frozen=True, slots=True)
class PortfolioUsage:
    """Capital already deployed, as fractions of total research capital.

    `capital_known` is the field to check first. When it is `False` every
    fraction here is `0.0` because the denominator is missing -- NOT because
    nothing is held. A caller that reads the zeros without the flag would be
    making exactly the mistake this module was written to fix.
    """

    total_fraction: float = 0.0
    held: tuple[HeldFraction, ...] = ()
    open_positions: int = 0
    unattributed: tuple[str, ...] = ()
    capital_known: bool = False

    @property
    def by_cluster(self) -> Mapping[str, float]:
        """Capital fraction per correlation cluster, attributed positions only."""
        sums: dict[str, float] = {}
        for item in self.held:
            if item.cluster_id is None:
                continue
            sums[item.cluster_id] = sums.get(item.cluster_id, 0.0) + item.fraction
        return sums

    @property
    def unattributed_fraction(self) -> float:
        """The part of the total that no cluster budget can be charged for."""
        return sum(item.fraction for item in self.held if item.cluster_id is None)

    @property
    def by_issuer(self) -> Mapping[str, float]:
        """Capital fraction per issuer, attributed positions only.

        The issuer limit (GOVERNANCE.md 5.2) bounds concentration and the
        shared liquidity and quote risk that comes with one issuer, which no
        amount of underlying diversification addresses.
        """
        sums: dict[str, float] = {}
        for item in self.held:
            if item.issuer is None:
                continue
            sums[item.issuer] = sums.get(item.issuer, 0.0) + item.fraction
        return sums

    @property
    def unattributed_issuers(self) -> tuple[str, ...]:
        return tuple(item.position_id for item in self.held if item.issuer is None)

    @property
    def reserved_fraction(self) -> float:
        """Capital committed by proposals already emitted, not yet resolved."""
        return sum(item.fraction for item in self.held if item.reserved)

    @property
    def enforceable(self) -> bool:
        """Whether the portfolio-level caps can be evaluated at all."""
        return self.capital_known

    def warning(self) -> str | None:
        """The warning a scan should carry, or `None` when all is well."""
        if not self.capital_known:
            return (
                "risk.research_capital_eur is unset, so the portfolio-level sizing "
                "caps (max_cluster_fraction, max_total_risk) cannot be evaluated "
                "against positions actually held and are NOT enforced against them. "
                "Set it in configs/risk.yaml to enable them."
            )
        if self.unattributed:
            return (
                f"{len(self.unattributed)} open position(s) could not be attributed "
                "to a correlation cluster (their ISIN is not in product_snapshots); "
                "they count toward the portfolio total but not toward any cluster: "
                f"{', '.join(self.unattributed)}"
            )
        return None


@dataclass(frozen=True, slots=True)
class PortfolioRiskVerdict:
    """Whether new ACTIONABLE proposals are defensible on portfolio grounds.

    Fail-closed on purpose. `passed` is `False` whenever the question could
    not be answered, not only when an answer was "too much risk" -- a
    portfolio limit that cannot be evaluated is not a limit that has been
    satisfied. The audit of 2026-10-24 found the opposite convention in
    place: the caps were netted against `0.0` because nobody supplied the
    used fractions, so "unknown" read as "nothing deployed".

    This withholds ACTIONABLE only. Scans, archiving, the shadow sample and
    every research path continue regardless -- the forward data collection
    is the most valuable thing the system has, and a risk question it
    cannot answer is no reason to stop recording quotes.
    """

    passed: bool
    reasons: tuple[str, ...] = ()

    def describe(self) -> str:
        return ", ".join(self.reasons) if self.reasons else "portfolio risk within caps"

    @property
    def breach_reasons(self) -> tuple[str, ...]:
        """Reasons that are an actual cap breach, not a missing input.

        The missing-input cases -- unknown capital, unattributable
        positions -- are already reported by `PortfolioUsage.warning()`,
        and a caller that warned about both would say the same thing twice
        for one condition. The verdict still fails closed either way; this
        only decides what is worth a second sentence.
        """
        return tuple(
            r
            for r in self.reasons
            if r.startswith(("total_risk_", "cluster_risk_", "issuer_risk_"))
        )


def portfolio_risk_verdict(
    usage: PortfolioUsage,
    *,
    max_total_risk: float,
    max_cluster_fraction: float,
    max_issuer_fraction: float,
) -> PortfolioRiskVerdict:
    """Evaluate the portfolio-level caps against what is actually held.

    Three worst-case capital limits, from the operator decision of
    2026-10-25 (GOVERNANCE.md 5.2): total exposure, per correlation
    cluster, per issuer. The fourth -- per position -- is enforced in
    `ranking/sizing.py` where a single position's size is decided.

    No value is chosen here: all three come from `SizingConfig`, which is
    the single place they live. This only answers whether the portfolio as
    a whole is inside them, and says why when it is not.
    """
    if not usage.capital_known:
        return PortfolioRiskVerdict(
            passed=False,
            reasons=(
                "research_capital_unknown: risk.research_capital_eur is unset, so "
                "total deployed capital cannot be computed and the portfolio-level "
                "caps cannot be evaluated against positions held",
            ),
        )

    reasons: list[str] = []
    if usage.total_fraction > max_total_risk:
        reasons.append(
            f"total_risk_exceeded: {usage.total_fraction:.4f} of capital deployed "
            f"against a cap of {max_total_risk:.4f}"
        )
    for cluster, fraction in sorted(usage.by_cluster.items()):
        if fraction > max_cluster_fraction:
            reasons.append(
                f"cluster_risk_exceeded: {cluster} holds {fraction:.4f} of capital "
                f"against a cap of {max_cluster_fraction:.4f}"
            )
    for issuer, fraction in sorted(usage.by_issuer.items()):
        if fraction > max_issuer_fraction:
            reasons.append(
                f"issuer_risk_exceeded: {issuer} holds {fraction:.4f} of capital "
                f"against a cap of {max_issuer_fraction:.4f}"
            )
    if usage.unattributed_issuers:
        # The issuer cap genuinely cannot be checked for a position whose
        # issuer is unknown, so the honest answer is that the check is
        # incomplete -- which fail-closed means blocking.
        reasons.append(
            f"unattributed_issuer: {len(usage.unattributed_issuers)} open position(s) "
            "could not be attributed to an issuer, so the per-issuer cap cannot be "
            "evaluated for them"
        )
    if usage.unattributed:
        # Capital that no cluster budget can be charged for. It counts in
        # the total, but the per-cluster cap genuinely cannot be checked for
        # it, so the honest answer is that the check is incomplete.
        reasons.append(
            f"unattributed_exposure: {len(usage.unattributed)} open position(s) "
            "could not be attributed to a correlation cluster, so the per-cluster "
            "cap cannot be evaluated for them"
        )

    return PortfolioRiskVerdict(passed=not reasons, reasons=tuple(reasons))


def portfolio_usage(
    store: Store,
    *,
    research_capital_eur: float | None,
    cluster_for_underlying: Callable[[str], str | None] | None = None,
) -> PortfolioUsage:
    """Capital deployed in open manual positions, as capital fractions.

    `cluster_for_underlying` maps an underlying id to its correlation
    cluster. Passed in rather than computed here so this module does not
    import the clustering, which needs bars and an as-of time that only the
    scan has.

    Also counts **reserved** exposure: ACTIONABLE proposals already written
    to the forward ledger and not yet resolved. GOVERNANCE.md 5.2 puts
    "bereits verbindlich reservierte neue Positionen" inside the 8% total,
    and without this two scans on the same day could each propose up to the
    full budget -- the system cannot know which proposal the operator acted
    on, so a proposal that has gone out has to count until it is resolved.
    WATCH candidates are not reserved: they are not a proposal to act on.
    """
    if research_capital_eur is None or research_capital_eur <= 0.0:
        # Deliberately not "assume zero deployed". The caller must see that
        # the question could not be answered.
        open_count = len(store.list_positions(status=PositionStatus.OPEN))
        if open_count:
            logger.warning(
                "portfolio_usage_unknown_capital",
                open_positions=open_count,
                detail="risk.research_capital_eur is unset; portfolio caps unenforced",
            )
        return PortfolioUsage(open_positions=open_count, capital_known=False)

    positions = store.list_positions(status=PositionStatus.OPEN)
    held: list[HeldFraction] = []
    unattributed: list[str] = []
    total = 0.0

    for position in positions:
        fraction = (position.qty * position.entry_price) / research_capital_eur
        total += fraction

        underlying = store.underlying_for_isin(position.isin) if position.isin is not None else None
        cluster = (
            cluster_for_underlying(underlying)
            if cluster_for_underlying is not None and underlying is not None
            else None
        )
        if cluster is None:
            unattributed.append(position.position_id)
        held.append(
            HeldFraction(
                position_id=position.position_id,
                underlying_id=underlying,
                cluster_id=cluster,
                issuer=store.issuer_for_isin(position.isin) if position.isin else None,
                fraction=fraction,
            )
        )

    # Proposals already emitted and not yet resolved. Their own
    # `suggested_position_fraction` is already a capital fraction, so this
    # needs no capital figure of its own -- but it is only reached when the
    # capital base is known, because the two must be comparable.
    for entry, _label in store.list_ledger_entries(
        status=LedgerEntryStatus.OPEN, category=Category.ACTIONABLE, is_shadow=False
    ):
        reserved_fraction = entry.suggested_position_fraction
        if reserved_fraction is None or reserved_fraction <= 0.0:
            continue
        total += reserved_fraction
        reserved_cluster = (
            cluster_for_underlying(entry.underlying) if cluster_for_underlying else None
        )
        if reserved_cluster is None:
            unattributed.append(entry.entry_id)
        held.append(
            HeldFraction(
                position_id=entry.entry_id,
                underlying_id=entry.underlying,
                cluster_id=reserved_cluster,
                issuer=entry.issuer,
                fraction=reserved_fraction,
                reserved=True,
            )
        )

    usage = PortfolioUsage(
        total_fraction=total,
        held=tuple(held),
        open_positions=len(positions),
        unattributed=tuple(unattributed),
        capital_known=True,
    )
    logger.info(
        "portfolio_usage",
        open_positions=usage.open_positions,
        total_fraction=round(usage.total_fraction, 6),
        clusters=len(usage.by_cluster),
        issuers=len(usage.by_issuer),
        reserved_fraction=round(usage.reserved_fraction, 6),
        unattributed=len(usage.unattributed),
    )
    return usage


__all__ = [
    "HeldFraction",
    "PortfolioRiskVerdict",
    "PortfolioUsage",
    "portfolio_risk_verdict",
    "portfolio_usage",
]
