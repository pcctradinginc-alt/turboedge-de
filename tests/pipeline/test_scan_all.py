"""Behaviour of the batch scan's portfolio-level risk plumbing
(``pipeline/scan_all.py``).

Until 2026-10-24 ``run_scan_all`` only aggregated: it computed correlation
clusters, threaded this run's own proposals from one underlying to the next,
and counted categories. Everything the portfolio-level sizing caps need in
order to be net of *positions actually held* was added that day, and
``tests/test_risk_limit_consistency.py`` pins it only structurally (an AST
check that ``portfolio_fraction_used`` is passed, a source check that
``portfolio_usage`` is still read) while ``tests/positions/test_usage.py``
unit-tests ``portfolio_usage`` on its own. A source-level pin survives a
rewrite that silently stops the value from reaching anything, so what is
asserted here is the behaviour instead:

1. with ``risk.research_capital_eur`` unset, the batch *says so* in its own
   warnings -- and nothing else changes, because the caps genuinely cannot
   be evaluated;
2. with it set, a position held in a correlation cluster lowers the
   suggested fraction of a candidate scanned in that same cluster, by
   exactly the cluster cap's remaining headroom;
3. a second underlying's scan is handed the first underlying's admitted
   candidates, so one cluster's capacity is shared across the batch rather
   than granted afresh per ``run_scan`` call;
4. ``portfolio_fraction_used`` accumulates across underlyings and keeps
   carrying the part of the portfolio no cluster can be charged for.

No network: every adapter is one of the in-memory fakes in
``tests/pipeline/conftest.py``, and each batch gets its own temporary
``Store``. ``ratio_reliability`` is set to ``SOURCE_REPORTED`` on every
product here (``make_dax_product`` leaves it ``UNVERIFIED``, which
``ranking/gates.py`` alone is enough to keep out of ACTIONABLE) because
tests 3 and 4 need candidates that are actually *admitted* -- only an
ACTIONABLE candidate consumes cluster capacity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from turboedge.alpha.schemas import AlphaSource, AlphaStatus
from turboedge.config import TurboEdgeConfig
from turboedge.meta.research_opportunity import Estimate, InformationFamily
from turboedge.pipeline import scan_all as scan_all_module
from turboedge.pipeline.scan import ScanResult, run_scan
from turboedge.pipeline.scan_all import ScanAllResult, run_scan_all
from turboedge.ranking.cluster import OpenClusterPosition
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Category,
    Direction,
    FieldReliability,
    ManualPosition,
    PositionStatus,
    ProductSnapshot,
    UnderlyingBar,
)

_EVAL_TIME = datetime(2026, 6, 15, 9, 0, tzinfo=UTC)

#: Total research capital for the tests that set it. Round, so that a
#: position's capital fraction below is readable rather than derived.
_CAPITAL_EUR = 100_000.0

#: A second underlying scanned alongside DAX. EUR-denominated (no FX leg)
#: and present in ``universe/underlying_map.py``, which ``run_scan``
#: requires -- an invented id is rejected as "unknown canonical
#: underlying_id" and lands in ``skipped`` instead of being scanned.
_PEER = "ESTX50"


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _ScanCall:
    """One ``run_scan`` call the batch made, as the batch parameterized it.

    ``cluster_open_positions`` and ``portfolio_fraction_used`` are what
    ``scan_all`` exists to supply and are not projected onto
    ``ScanAllResult``, so they are recorded at the call boundary.
    """

    underlying_id: str
    cluster_id: str
    cluster_open_positions: tuple[OpenClusterPosition, ...]
    portfolio_fraction_used: float
    portfolio_risk_pass: bool | None


@dataclass(frozen=True)
class _BatchRun:
    """One ``run_scan_all`` outcome plus what it asked each scan for."""

    result: ScanAllResult
    calls: tuple[_ScanCall, ...]
    #: ``(underlying, isin) -> suggested_position_fraction``, read back from
    #: the forward ledger -- the only persisted record of the size each
    #: candidate was actually sized at.
    suggested_fractions: dict[tuple[str, str], float] = field(default_factory=dict)

    def calls_for(self, underlying_id: str) -> _ScanCall:
        return next(c for c in self.calls if c.underlying_id == underlying_id)

    def fractions_for(self, underlying_id: str) -> list[float]:
        return [
            fraction
            for (underlying, _isin), fraction in self.suggested_fractions.items()
            if underlying == underlying_id
        ]

    def actionable_in(self, underlying_id: str) -> list[Any]:
        result = self.result.results[underlying_id]
        return [c for c in result.candidates if c.category == Category.ACTIONABLE]


def _production_alpha(*underlyings: str) -> AlphaSource:
    """A confirmed alpha in a production state, covering these underlyings.

    ACTIONABLE requires one (`ranking/gates.py`, audit 2026-10-24): a
    positive *modelled* `lcb_ev` is necessary but not sufficient, because
    until that audit nothing in the scan pipeline read `alpha_sources` at
    all. Tests about cluster propagation need a candidate to actually reach
    ACTIONABLE, so they have to satisfy that precondition rather than work
    around it.

    `frozen_at` is required for any state at or past CONFIRMATORY -- the
    schema enforces that confirmatory evidence is only meaningful for a
    hypothesis that was frozen before the data was seen.
    """
    unknown = Estimate.unknown("not measured in this test")
    return AlphaSource(
        alpha_id="A-TEST",
        name="test alpha",
        family=InformationFamily.MICROSTRUCTURE,
        version="1",
        description="d",
        economic_hypothesis="h",
        status=AlphaStatus.CANARY_PRODUCTION,
        created_at=_EVAL_TIME,
        frozen_at=_EVAL_TIME,
        underlyings=list(underlyings),
        horizons=["5"],
        directions=["long"],
        required_data_sources=["source_a"],
        nominal_sample=unknown,
        effective_sample=unknown,
        expected_net_ev=unknown,
        lcb_net_ev=unknown,
        posterior_probability_positive=unknown,
        uncertainty_score=unknown,
        drift_score=unknown,
        decay_score=unknown,
    )


def _with_capital(cfg: TurboEdgeConfig, research_capital_eur: float) -> TurboEdgeConfig:
    """``cfg`` with the one risk field these tests vary, and nothing else.

    Only the denominator that turns a held EUR position into a capital
    fraction is touched -- never a limit value (GOVERNANCE.md §5.2 records
    ``max_position_fraction``/``max_cluster_fraction`` as UNRESOLVED pending
    a human decision, and §11 reserves limit changes).
    """
    return cfg.model_copy(
        update={"risk": cfg.risk.model_copy(update={"research_capital_eur": research_capital_eur})}
    )


def _favorable_long(
    dax_product_factory: Callable[..., ProductSnapshot],
    *,
    isin: str,
    underlying_id: str = "DAX",
) -> ProductSnapshot:
    """A long open-end turbo priced essentially at intrinsic, barrier far
    below spot -- the same economics as ``test_scan_ev.py``'s
    ``_cheap_favorable_long``, retargetable onto another underlying.

    ``make_dax_product`` hardcodes ``underlying_id``/``underlying_raw`` to
    DAX; retargeting by ``model_copy`` keeps every other field (and so the
    product's economics against a ~24,000 spot) identical between the two
    underlyings, which is what makes a cross-underlying comparison mean
    anything.
    """
    product = dax_product_factory(
        isin=isin,
        issuer="BankA",
        direction=Direction.LONG,
        financing_level=18000.0,  # far below spot (24000) -> tiny P(KO)
        ratio=0.01,
        quote_timestamp=_EVAL_TIME,
    )
    return product.model_copy(
        update={
            # intrinsic = (24000-18000)*0.01 = 60.0; tight, cheap quote around it.
            "ask": 60.05,
            "bid": 60.03,
            "underlying_id": underlying_id,
            "underlying_raw": underlying_id,
            # The gate that would otherwise hold every candidate here at
            # WATCH; an admitted candidate is the point of tests 3 and 4.
            "ratio_reliability": FieldReliability.SOURCE_REPORTED,
        }
    )


def _held_product(
    dax_product_factory: Callable[..., ProductSnapshot],
    *,
    isin: str,
    underlying_id: str = "DAX",
    issuer: str = "BankA",
) -> ProductSnapshot:
    """A snapshot whose only job is to let ``Store.underlying_for_isin``
    resolve a held position's ISIN to ``underlying_id``.

    Never offered to the scan as a candidate -- a position the user holds
    and a product this run is considering are different things, and
    conflating them is what would make a holding look like a proposal.
    """
    return dax_product_factory(
        isin=isin,
        issuer=issuer,
        direction=Direction.LONG,
        financing_level=19000.0,
        ratio=0.01,
        quote_timestamp=_EVAL_TIME,
    ).model_copy(update={"underlying_id": underlying_id, "underlying_raw": underlying_id})


def _position(isin: str | None, *, qty: float, entry_price: float) -> ManualPosition:
    """One open manual position -- the only record of capital this system
    treats as actually deployed (``positions/usage.py``: a ``forward_ledger``
    row is a proposal, not a holding).
    """
    return ManualPosition(
        wkn=(isin or "XX0000")[-6:],
        isin=isin,
        qty=qty,
        entry_price=entry_price,
        entry_date=date(2026, 6, 1),
        status=PositionStatus.OPEN,
        created_at=_EVAL_TIME,
        updated_at=_EVAL_TIME,
    )


@pytest.fixture
def correlated_bars(bars_factory: Callable[..., list[UnderlyingBar]]) -> dict[str, Any]:
    """DAX and its peer on one shared return series, so they are certain to
    land in the *same* correlation cluster.

    Identical returns give ``rho = 1`` (distance 0), well inside
    ``cluster.distance_threshold`` -- a test about sharing one cluster's
    capacity must not be able to fail because two synthetic series happened
    to correlate below the cutoff. 400 low-noise, strongly positive-drift
    days is also enough history for every default forecast model to fit, so
    a candidate can reach ACTIONABLE at all.
    """
    dax = bars_factory("DAX", count=400, daily_drift=0.0025, noise_std=0.0015, seed=7)
    peer = [bar.model_copy(update={"underlying_id": _PEER}) for bar in dax]
    return {"DAX": dax, _PEER: peer}


@pytest.fixture
def run_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_product_adapter: Callable[..., Any],
    make_price_adapter: Callable[..., Any],
    make_estr_adapter: Callable[..., Any],
    correlated_bars: dict[str, Any],
) -> Callable[..., _BatchRun]:
    """Run one full ``run_scan_all`` batch against in-memory fakes.

    Each batch gets its own ``Store`` (named by ``label``) so that two
    batches can be compared within one test without one's forward-ledger
    rows or product snapshots leaking into the other's.

    ``run_scan`` is wrapped, not replaced: the real scan still runs, and the
    wrapper only records the arguments ``scan_all`` itself is responsible for
    computing. Nothing about the pipeline's own behaviour is faked here.
    """

    def _run(
        cfg: TurboEdgeConfig,
        *,
        label: str,
        underlying_ids: Sequence[str],
        products: Sequence[ProductSnapshot],
        snapshots: Sequence[ProductSnapshot] = (),
        positions: Sequence[ManualPosition] = (),
        alphas: Sequence[AlphaSource] = (),
    ) -> _BatchRun:
        calls: list[_ScanCall] = []

        def _recording_run_scan(*args: Any, **kwargs: Any) -> ScanResult:
            calls.append(
                _ScanCall(
                    underlying_id=kwargs["options"].underlying_id,
                    cluster_id=kwargs["cluster_id"],
                    cluster_open_positions=tuple(kwargs["cluster_open_positions"]),
                    portfolio_fraction_used=kwargs["portfolio_fraction_used"],
                    portfolio_risk_pass=kwargs["portfolio_risk_pass"],
                )
            )
            return run_scan(*args, **kwargs)

        monkeypatch.setattr(scan_all_module, "run_scan", _recording_run_scan)

        state_dir = tmp_path / label
        state_dir.mkdir(parents=True, exist_ok=True)
        with Store(state_dir / "turboedge.duckdb") as store:
            store.init_schema()
            if snapshots:
                store.append_product_snapshots(list(snapshots))
            for position in positions:
                store.insert_position(position)
            for alpha in alphas:
                store.upsert_alpha_source(alpha)

            result = run_scan_all(
                cfg,
                store,
                state_dir,
                underlying_ids=list(underlying_ids),
                product_adapters=[make_product_adapter("source_a", products=list(products))],
                price_adapter=make_price_adapter(bars_by_underlying=dict(correlated_bars)),
                estr_adapter=make_estr_adapter(),
                reference_healthchecks=[],
                notifier=None,
                email=False,
                clock=lambda: _EVAL_TIME,
                rng=np.random.default_rng(1234),
            )
            fractions = {
                (entry.underlying, entry.selected_isin): entry.suggested_position_fraction
                for entry, _label in store.list_ledger_entries()
                if entry.suggested_position_fraction is not None
            }

        return _BatchRun(result=result, calls=tuple(calls), suggested_fractions=fractions)

    return _run


# --------------------------------------------------------------------------
# 1. the missing denominator is reported, not silently read as "nothing held"
# --------------------------------------------------------------------------


def test_unset_research_capital_warns_in_the_batch_result(
    cfg: TurboEdgeConfig,
    run_batch: Callable[..., _BatchRun],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """``risk.research_capital_eur`` is unset by default (configs/risk.yaml
    leaves it commented out on purpose), so the two portfolio-level caps
    cannot be evaluated against real holdings at all. The batch must carry
    that in ``ScanAllResult.warnings``, where the CLI and
    ``reports/summary.json`` surface it -- an unenforceable cap that nobody
    is told about is indistinguishable from an enforced one.
    """
    assert cfg.risk.research_capital_eur is None, (
        "configs/risk.yaml is expected to leave research_capital_eur unset; "
        "this test is about what the batch does when it is missing"
    )

    run = run_batch(
        cfg,
        label="unset_capital",
        underlying_ids=["DAX"],
        products=[_favorable_long(dax_product_factory, isin="DE000FAVLNG1")],
        snapshots=[_held_product(dax_product_factory, isin="DE000HELD001")],
        positions=[_position("DE000HELD001", qty=1000.0, entry_price=14.0)],
    )

    assert run.result.skipped == {}, run.result.skipped
    unset_capital = [w for w in run.result.warnings if "research_capital_eur" in w]
    assert len(unset_capital) == 1, run.result.warnings
    assert "NOT enforced" in unset_capital[0]
    assert "max_cluster_fraction" in unset_capital[0]


def test_unset_research_capital_leaves_the_scan_otherwise_unchanged(
    cfg: TurboEdgeConfig,
    run_batch: Callable[..., _BatchRun],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """The warning is the *whole* consequence of the missing denominator.

    Without research capital a EUR position cannot be turned into a
    fraction, so there is no honest number to charge against either cap: the
    held position must not be seeded into the cluster accumulator at a
    guessed size, and the suggested fractions must come out exactly as they
    do with nothing held. This is the counterpart to the next test -- it is
    what makes that one's reduction attributable to the real holding rather
    than to anything else in the batch.
    """
    product = _favorable_long(dax_product_factory, isin="DE000FAVLNG1")

    nothing_held = run_batch(cfg, label="unset_none", underlying_ids=["DAX"], products=[product])
    position_held = run_batch(
        cfg,
        label="unset_held",
        underlying_ids=["DAX"],
        products=[product],
        snapshots=[_held_product(dax_product_factory, isin="DE000HELD001")],
        positions=[_position("DE000HELD001", qty=1000.0, entry_price=14.0)],
    )

    assert position_held.calls_for("DAX").cluster_open_positions == ()
    assert position_held.calls_for("DAX").portfolio_fraction_used == 0.0
    assert position_held.suggested_fractions == nothing_held.suggested_fractions


# --------------------------------------------------------------------------
# 2. with the denominator, a holding costs a later candidate real capacity
# --------------------------------------------------------------------------


def test_a_held_position_lowers_a_later_candidate_in_the_same_cluster(
    cfg: TurboEdgeConfig,
    run_batch: Callable[..., _BatchRun],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """The behaviour the whole 2026-10-24 wiring exists for.

    ``ranking/sizing.py`` applies ``min(fraction, max_cluster_fraction -
    cluster_fraction_used)``. Measured that day: nothing ever supplied
    ``cluster_fraction_used``, so it defaulted to 0.0 and the cap reduced to
    a per-position cap looser than the single-position cap, i.e.
    structurally inert. With research capital configured, an open position
    whose ISIN resolves to an underlying in this cluster must consume that
    headroom, and the candidate scanned in the same cluster must be sized
    into what is left.

    The holding is deliberately sized (4% of capital, against the 5%
    cluster cap decided on 2026-10-25) so that the cluster cap -- not
    ``max_position_fraction``, not ``max_total_risk`` and not
    ``max_issuer_fraction`` -- is the binding one, and the resulting size
    therefore identifies *which* cap bound. No limit value is modified to
    achieve that; only the position's size and its issuer are chosen.
    """
    sizing = cfg.ranking.sizing
    held_fraction = 0.04
    qty, entry_price = 1000.0, held_fraction * _CAPITAL_EUR / 1000.0
    assert held_fraction < sizing.max_cluster_fraction, (
        "the holding must leave the cluster cap with some headroom, so that "
        "a reduced-but-positive size is observable rather than a flat zero"
    )
    headroom = sizing.max_cluster_fraction - held_fraction
    assert headroom < sizing.max_position_fraction, (
        "with this much held, the cluster cap -- not the per-position cap -- "
        "must be the binding one for the reduction to mean what it says"
    )
    assert headroom < sizing.max_total_risk - held_fraction, (
        "the portfolio total must not be the binding cap either"
    )
    # The holding is from a different issuer than the candidates, so the
    # issuer cap (5% since 2026-10-25) leaves them their full budget and
    # cannot be what bound. Without this the issuer and cluster headroom
    # would be identical and the reduction would not identify either.
    held_issuer = "BankZ"
    assert headroom < sizing.max_issuer_fraction, (
        "the issuer cap must leave more headroom than the cluster cap here"
    )

    with_capital = _with_capital(cfg, _CAPITAL_EUR)
    products = [
        _favorable_long(dax_product_factory, isin="DE000FAVLNG1"),
        _favorable_long(dax_product_factory, isin="DE000FAVEU50", underlying_id=_PEER),
    ]

    nothing_held = run_batch(
        with_capital, label="cap_none", underlying_ids=["DAX", _PEER], products=products
    )
    position_held = run_batch(
        with_capital,
        label="cap_held",
        underlying_ids=["DAX", _PEER],
        products=products,
        snapshots=[_held_product(dax_product_factory, isin="DE000HELD001", issuer=held_issuer)],
        positions=[_position("DE000HELD001", qty=qty, entry_price=entry_price)],
    )

    assert not [w for w in position_held.result.warnings if "research_capital_eur" in w], (
        "with research capital configured the caps are enforceable and the "
        "unenforceable-caps warning must not be raised"
    )

    # The holding reached the scan as this cluster's already-used capacity.
    dax_call = position_held.calls_for("DAX")
    assert len(dax_call.cluster_open_positions) == 1
    seeded = dax_call.cluster_open_positions[0]
    assert seeded.underlying_id == "DAX", "the holding's ISIN resolved to its underlying"
    assert seeded.cluster_id == dax_call.cluster_id, "and to the cluster being scanned"
    assert seeded.cluster_id == position_held.calls_for(_PEER).cluster_id, (
        "DAX and its peer share one correlation cluster here, so the holding's "
        "capacity is the same capacity both underlyings are sized against"
    )
    assert seeded.capital_fraction == pytest.approx(held_fraction)
    assert seeded.counts_as_active_position

    # ...and cost the candidates scanned in it exactly the cap's headroom.
    baseline = nothing_held.fractions_for("DAX")
    reduced = position_held.fractions_for("DAX")
    assert baseline, "the baseline batch must have sized something to compare against"
    assert len(reduced) == len(baseline)
    assert min(baseline) > headroom, (
        "the baseline size must sit above the headroom, or the holding could "
        "not reduce anything and this test would pass vacuously"
    )
    assert all(fraction == pytest.approx(headroom) for fraction in reduced), reduced
    assert max(reduced) < min(baseline)


# --------------------------------------------------------------------------
# 3. one cluster's capacity is shared across the batch, not per run_scan call
# --------------------------------------------------------------------------


def test_the_second_underlying_sees_the_first_ones_admitted_candidates(
    cfg: TurboEdgeConfig,
    run_batch: Callable[..., _BatchRun],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """Two underlyings in one correlation cluster are scanned by two separate
    ``run_scan`` calls, each of which knows only what it is handed. The
    batch is what makes the second call aware of the first's admitted
    (ACTIONABLE) candidates -- otherwise each call would be granted the
    cluster's full capacity afresh and a cluster could be loaded up n times
    over within a single run.
    """
    with_capital = _with_capital(cfg, _CAPITAL_EUR)
    run = run_batch(
        with_capital,
        label="shared_cluster",
        underlying_ids=["DAX", _PEER],
        products=[
            _favorable_long(dax_product_factory, isin="DE000FAVLNG1"),
            _favorable_long(dax_product_factory, isin="DE000FAVEU50", underlying_id=_PEER),
        ],
        # ACTIONABLE now also needs a confirmed alpha source covering the
        # underlying; without one nothing is admitted and there is nothing
        # for the second underlying to see.
        alphas=[_production_alpha("DAX", _PEER)],
    )

    assert run.result.skipped == {}, run.result.skipped
    assert [c.underlying_id for c in run.calls] == ["DAX", _PEER]
    first, second = run.calls
    assert first.cluster_id == second.cluster_id, (
        "these two underlyings share a return series, so they must cluster "
        "together -- the rest of this test is about that shared cluster"
    )

    # Nothing is held, so the first call starts from an empty cluster...
    assert first.cluster_open_positions == ()
    admitted = run.actionable_in("DAX")
    assert admitted, "the first underlying must admit something to pass on"

    # ...and the second is handed exactly what the first admitted.
    assert second.cluster_open_positions == tuple(run.result.results["DAX"].new_cluster_positions)
    assert len(second.cluster_open_positions) == len(admitted)
    assert {p.underlying_id for p in second.cluster_open_positions} == {"DAX"}
    assert all(p.cluster_id == second.cluster_id for p in second.cluster_open_positions)
    assert all(p.counts_as_active_position for p in second.cluster_open_positions)

    # The consequence: the cluster's position-count budget is spent once
    # across the batch, not once per underlying.
    max_active = cfg.ranking.cluster.max_active_positions_per_cluster
    total_admitted = len(run.result.actionable_candidate_ids())
    assert total_admitted <= max_active, (
        f"{total_admitted} candidates admitted into one cluster, over the "
        f"max_active_positions_per_cluster limit of {max_active}"
    )


# --------------------------------------------------------------------------
# 4. the portfolio total accumulates, and never drops the unattributable part
# --------------------------------------------------------------------------


def test_the_portfolio_total_accumulates_and_keeps_the_unattributed_part(
    cfg: TurboEdgeConfig,
    run_batch: Callable[..., _BatchRun],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    """``portfolio_fraction_used`` feeds ``max_total_risk``, which is
    portfolio-wide and so cluster-agnostic. It must therefore grow as the
    batch admits candidates, *and* keep carrying the holdings that no
    cluster list can hold.

    A position whose ISIN this system never scanned cannot be mapped to an
    underlying, so no ``cluster_open_positions`` entry represents it. Since
    ``scan_all`` recomputes the total from those per-cluster lists after
    each underlying (rather than adding incrementally, so the total cannot
    drift away from them), an unattributed holding would vanish from the
    portfolio total at the first recomputation if it were not held
    separately.
    """
    with_capital = _with_capital(cfg, _CAPITAL_EUR)
    unattributed_fraction = 0.02
    # No product snapshot is appended for this ISIN, so
    # `Store.underlying_for_isin` returns None and the position is
    # attributable to no cluster.
    unscanned = _position(
        "DE000UNKNOWN",
        qty=1000.0,
        entry_price=unattributed_fraction * _CAPITAL_EUR / 1000.0,
    )
    run = run_batch(
        with_capital,
        label="unattributed",
        underlying_ids=["DAX", _PEER],
        products=[
            _favorable_long(dax_product_factory, isin="DE000FAVLNG1"),
            _favorable_long(dax_product_factory, isin="DE000FAVEU50", underlying_id=_PEER),
        ],
        positions=[unscanned],
    )

    assert run.result.skipped == {}, run.result.skipped
    unattributed_warning = [w for w in run.result.warnings if "could not be attributed" in w]
    assert len(unattributed_warning) == 1, run.result.warnings
    assert unscanned.position_id in unattributed_warning[0], (
        "the warning must name the position that could not be attributed, so "
        "the operator can tell which holding no cluster budget covers"
    )

    first, second = run.calls

    # Every scan in the batch sees the unattributed holding in the portfolio
    # total, even though no cluster list carries it.
    assert first.cluster_open_positions == ()
    assert first.portfolio_fraction_used == pytest.approx(unattributed_fraction)
    assert second.portfolio_fraction_used == pytest.approx(unattributed_fraction)
    assert not any(p.underlying_id == "(unknown)" for p in second.cluster_open_positions), (
        "an unattributed holding must not be smuggled into a cluster list"
    )

    # And it blocks ACTIONABLE, which is why nothing accumulated above: the
    # per-cluster cap cannot be evaluated for a holding whose cluster is
    # unknown, and a cap that cannot be evaluated has not been satisfied.
    assert first.portfolio_risk_pass is False
    assert second.portfolio_risk_pass is False


def test_an_unattributable_holding_blocks_actionable(
    cfg: TurboEdgeConfig,
    run_batch: Callable[..., _BatchRun],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    # Fail-closed, stated as its own property: a position whose correlation
    # cluster cannot be determined means the per-cluster cap cannot be
    # checked, so no new proposal may pass the portfolio check. The research
    # paths are untouched -- the batch still scans, still writes ledger
    # entries, and still records quotes.
    run = run_batch(
        _with_capital(cfg, _CAPITAL_EUR),
        label="blocked_by_unattributed",
        underlying_ids=["DAX"],
        products=[_favorable_long(dax_product_factory, isin="DE000FAVLNG1")],
        positions=[_position("DE000UNKNOWN", qty=100.0, entry_price=5.0)],
        alphas=[_production_alpha("DAX")],
    )

    assert run.calls[0].portfolio_risk_pass is False
    assert run.suggested_fractions, "the scan still ran and still wrote the ledger"


def test_the_portfolio_total_accumulates_across_underlyings(
    cfg: TurboEdgeConfig,
    run_batch: Callable[..., _BatchRun],
    dax_product_factory: Callable[..., ProductSnapshot],
) -> None:
    # The accumulation `max_total_risk` is applied net of, with a portfolio
    # the check can actually evaluate: nothing held, so no unattributed
    # exposure, and a confirmed alpha so candidates can reach ACTIONABLE.
    run = run_batch(
        _with_capital(cfg, _CAPITAL_EUR),
        label="accumulates",
        underlying_ids=["DAX", _PEER],
        products=[
            _favorable_long(dax_product_factory, isin="DE000FAVLNG1"),
            _favorable_long(dax_product_factory, isin="DE000FAVEU50", underlying_id=_PEER),
        ],
        alphas=[_production_alpha("DAX", _PEER)],
    )

    assert run.result.skipped == {}, run.result.skipped
    first, second = run.calls
    assert first.portfolio_risk_pass is True
    assert first.portfolio_fraction_used == pytest.approx(0.0)

    admitted = sum(p.capital_fraction for p in second.cluster_open_positions)
    assert admitted > 0.0, "the first underlying must admit something to accumulate"
    assert second.portfolio_fraction_used == pytest.approx(admitted)
    assert second.portfolio_fraction_used > first.portfolio_fraction_used
