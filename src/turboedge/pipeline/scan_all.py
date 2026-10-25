"""Multi-underlying scan orchestration (Contract v3 Abschnitt B/E, ``turboedge
scan-all``).

The forecast/path/EV machinery lives entirely in :func:`turboedge.pipeline.
scan.run_scan`, which this module calls once per active underlying (Master
Spec §4: "Nur Underlyings scannen, für die ein Forecast UND Produkte
vorliegen" -- a per-underlying failure is skipped, never aborts the batch).
What this module adds on top:

1. Correlation clusters (Master Spec §31) computed *once*, across every
   underlying's own bar history, before any individual scan runs -- so each
   ``run_scan`` call gets a real ``cluster_id`` instead of the
   single-underlying fallback it would otherwise use on its own.
2. Cluster-risk state (:class:`~turboedge.ranking.cluster.OpenClusterPosition`)
   threaded from one underlying's scan to the next *within the same batch*,
   so a second ACTIONABLE candidate in the same correlation cluster (e.g. two
   different DAX-correlated underlyings both proposing a long) is correctly
   gated against the cluster's capacity even though each is produced by a
   separate ``run_scan`` call.
3. Aggregation: total/by-underlying category counts, timing, and the
   ``reports/summary.json`` payload shape the CLI writes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import structlog

from turboedge.adapters.base import HealthCheckResult
from turboedge.adapters.registry import ProductSourceAdapter
from turboedge.config import TurboEdgeConfig
from turboedge.notifications.gmail import GmailNotifier
from turboedge.pipeline.scan import EstrSource, PriceSource, ScanOptions, ScanResult, run_scan
from turboedge.pipeline.universe import NoProductsError
from turboedge.positions.usage import portfolio_risk_verdict, portfolio_usage
from turboedge.provenance import new_run_id
from turboedge.ranking.cluster import OpenClusterPosition, cluster_id_for, compute_clusters
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import Category, LedgerEntryStatus, UnderlyingBar

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ScanAllResult:
    """Outcome of one :func:`run_scan_all` call."""

    run_ids: dict[str, str]
    results: dict[str, ScanResult]
    counts: dict[str, int]
    counts_by_underlying: dict[str, dict[str, int]]
    skipped: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    elapsed_by_underlying: dict[str, float] = field(default_factory=dict)

    def actionable_candidate_ids(self) -> list[str]:
        return [
            c.candidate_id
            for result in self.results.values()
            for c in result.candidates
            if c.category == Category.ACTIONABLE
        ]


def run_scan_all(
    cfg: TurboEdgeConfig,
    store: Store,
    state_dir: str | Path,
    *,
    underlying_ids: Sequence[str] | None,
    product_adapters: Sequence[ProductSourceAdapter],
    price_adapter: PriceSource,
    estr_adapter: EstrSource,
    reference_healthchecks: Sequence[Callable[[], HealthCheckResult]],
    notifier: GmailNotifier | None,
    email: bool,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    rng: np.random.Generator | None = None,
) -> ScanAllResult:
    """Scan every active underlying with the full forecast/EV pipeline
    enabled (``run_scan(..., rng=...)``).

    Args:
        underlying_ids: Explicit underlyings, or ``None`` to use every
            ``enabled: true`` entry in ``configs/universe.yaml``.
        rng: Seeded generator threaded into every ``run_scan`` call
            (CLAUDE.md rule 16); defaults to ``numpy.random.default_rng(
            cfg.simulation.seed)`` when not supplied.

    A per-underlying failure (``NoProductsError``, or any other exception
    from a single ``run_scan`` call) is recorded in ``skipped`` and does not
    abort the rest of the batch (mirrors ``pipeline.universe.run_universe``'s
    per-source resilience, applied here per-underlying).
    """
    ids = list(underlying_ids) if underlying_ids else cfg.universe.enabled_ids()
    effective_rng = rng if rng is not None else np.random.default_rng(cfg.simulation.seed)

    t_start = time.perf_counter()
    warnings: list[str] = []
    skipped: dict[str, str] = {}
    run_ids: dict[str, str] = {}
    results: dict[str, ScanResult] = {}
    elapsed_by_underlying: dict[str, float] = {}
    counts: dict[str, int] = {c.value: 0 for c in Category}
    counts_by_underlying: dict[str, dict[str, int]] = {}

    if not ids:
        return ScanAllResult(
            run_ids={},
            results={},
            counts=counts,
            counts_by_underlying={},
            skipped={},
            warnings=["no_underlyings_enabled"],
            elapsed_s=time.perf_counter() - t_start,
        )

    # -- correlation clusters across every underlying in this batch (Master
    # Spec §31), computed once from each one's own bar history --
    bars_by_underlying: dict[str, list[UnderlyingBar]] = {}
    for uid in ids:
        try:
            bars = price_adapter.fetch_daily_bars(uid, lookback_days=400)
        except Exception as exc:
            logger.warning("scan_all_bars_prefetch_failed", underlying_id=uid, error=str(exc))
            bars = store.latest_underlying_bars(uid, 400)
        bars_by_underlying[uid] = bars
    now = clock()
    cluster_assignments = compute_clusters(bars_by_underlying, as_of=now, cfg=cfg.ranking.cluster)

    # Capital already deployed in positions the user actually holds. Until
    # 2026-10-24 this accumulator started empty on every run, so the two
    # portfolio-level sizing caps were net of *this run's own proposals*
    # only and knew nothing about yesterday's open positions. Seeding it
    # here is what makes `max_cluster_fraction` and `max_total_risk` net of
    # the real portfolio rather than of nothing.
    #
    # When `risk.research_capital_eur` is unset there is no denominator to
    # turn a EUR position into a fraction, so the usage is reported as
    # unknown and carried into the scan's warnings instead of being silently
    # read as "nothing is held" -- the same refusal to treat an absent input
    # as a satisfied constraint that `alpha/promotion.py` makes.
    usage = portfolio_usage(
        store,
        research_capital_eur=cfg.risk.research_capital_eur,
        cluster_for_underlying=lambda u: cluster_id_for(cluster_assignments, u),
    )
    usage_warning = usage.warning()
    if usage_warning is not None and usage_warning not in warnings:
        warnings.append(usage_warning)

    # One entry per held position, not one aggregate per cluster:
    # `ranking/cluster.py` counts positions as well as capital
    # (`max_active_positions_per_cluster`), so collapsing three holdings into
    # one entry would under-count them by two.
    cluster_positions: dict[str, list[OpenClusterPosition]] = {}
    for item in usage.held:
        if item.cluster_id is None:
            continue
        cluster_positions.setdefault(item.cluster_id, []).append(
            OpenClusterPosition(
                underlying_id=item.underlying_id or "(unknown)",
                cluster_id=item.cluster_id,
                capital_fraction=item.fraction,
                counts_as_active_position=True,
            )
        )
    # Fail-closed: a portfolio limit that cannot be evaluated is not a
    # limit that has been satisfied. While `risk.research_capital_eur` is
    # unset this is False, so no candidate can reach ACTIONABLE -- and
    # scans, archiving, the shadow sample and every research path carry on
    # regardless, because the forward data collection is the most valuable
    # thing here and a risk question nobody can answer is no reason to stop
    # recording quotes.
    risk_verdict = portfolio_risk_verdict(
        usage,
        max_total_risk=cfg.ranking.sizing.max_total_risk,
        max_cluster_fraction=cfg.ranking.sizing.max_cluster_fraction,
        max_issuer_fraction=cfg.ranking.sizing.max_issuer_fraction,
    )
    for breach in risk_verdict.breach_reasons:
        # Only actual cap breaches. The missing-input cases are already in
        # `usage.warning()` above, and warning twice for one condition
        # makes a reader doubt both messages.
        note = f"portfolio risk check not passed: {breach}"
        if note not in warnings:
            warnings.append(note)

    portfolio_fraction_used = usage.total_fraction
    # Positions whose ISIN this system never scanned cannot be attributed to
    # a cluster, so no cluster list carries them. Held separately so that
    # recomputing the total from those lists below does not lose them.
    unattributed_fraction = usage.unattributed_fraction
    # Capital per issuer, held plus already reserved. Unlike the cluster and
    # the portfolio total this is not recomputed per underlying: an issuer's
    # usage changes only when a proposal is admitted, which the per-cluster
    # lists below already track, and projecting it onto candidates is the
    # scan's job (`pipeline/scan.py`).
    issuer_fraction_used: dict[str, float] = dict(usage.by_issuer)

    for uid in ids:
        u_start = time.perf_counter()
        cluster_id = cluster_id_for(cluster_assignments, uid)
        prior_positions = cluster_positions.get(cluster_id, [])
        run_id = new_run_id()
        run_ids[uid] = run_id
        try:
            result = run_scan(
                cfg,
                store,
                state_dir,
                options=ScanOptions(underlying_id=uid, email=email),
                product_adapters=product_adapters,
                price_adapter=price_adapter,
                estr_adapter=estr_adapter,
                reference_healthchecks=reference_healthchecks,
                notifier=notifier,
                run_id=run_id,
                clock=clock,
                rng=effective_rng,
                cluster_id=cluster_id,
                cluster_open_positions=prior_positions,
                portfolio_fraction_used=portfolio_fraction_used,
                portfolio_risk_pass=risk_verdict.passed,
                issuer_fraction_used=issuer_fraction_used,
            )
        except NoProductsError as exc:
            skipped[uid] = f"no_products:{exc}"
            logger.warning("scan_all_underlying_skipped", underlying_id=uid, reason=str(exc))
            elapsed_by_underlying[uid] = time.perf_counter() - u_start
            continue
        except Exception as exc:  # a single underlying's failure never aborts the batch
            skipped[uid] = f"error:{exc}"
            logger.error("scan_all_underlying_failed", underlying_id=uid, error=str(exc))
            elapsed_by_underlying[uid] = time.perf_counter() - u_start
            continue

        results[uid] = result
        elapsed_by_underlying[uid] = time.perf_counter() - u_start
        cluster_positions[cluster_id] = result.new_cluster_positions
        # Recomputed from the authoritative per-cluster lists rather than by
        # incremental addition, so the portfolio total cannot drift away from
        # them over a long batch.
        portfolio_fraction_used = unattributed_fraction + sum(
            position.capital_fraction
            for positions in cluster_positions.values()
            for position in positions
        )
        # And the issuer budgets this run has consumed, so the next
        # underlying's candidates are sized net of them. Read back from the
        # store rather than from `result`, which carries only a count -- and
        # the persisted rows are the more faithful source anyway.
        for entry, _label in store.list_ledger_entries(
            run_id=run_id,
            status=LedgerEntryStatus.OPEN,
            category=Category.ACTIONABLE,
            is_shadow=False,
        ):
            if entry.suggested_position_fraction:
                issuer_fraction_used[entry.issuer] = (
                    issuer_fraction_used.get(entry.issuer, 0.0) + entry.suggested_position_fraction
                )
        counts_by_underlying[uid] = {}
        for cat, count in result.counts.items():
            counts[cat.value] += count
            counts_by_underlying[uid][cat.value] = count
        for w in result.warnings:
            if w not in warnings:
                warnings.append(w)

    return ScanAllResult(
        run_ids=run_ids,
        results=results,
        counts=counts,
        counts_by_underlying=counts_by_underlying,
        skipped=skipped,
        warnings=warnings,
        elapsed_s=time.perf_counter() - t_start,
        elapsed_by_underlying=elapsed_by_underlying,
    )


__all__ = ["ScanAllResult", "run_scan_all"]
