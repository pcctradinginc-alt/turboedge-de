"""The portfolio-level sizing caps must be evaluated against real holdings.

`ranking/sizing.py` applies two caps net of what is already used:

    fraction = min(fraction, max_cluster_fraction - cluster_fraction_used)
    fraction = min(fraction, max_total_risk       - total_risk_used)

Measured 2026-10-24: nothing ever supplied either argument. Both defaulted
to `0.0`, so both caps reduced to per-position caps of 15% and 30% -- each
looser than the 5% single-position cap, which means **neither ever bound**.
Two of the four caps in `SizingConfig` were structurally inert in
production, and `ranking/utility.py:86-88` names one of them as a *primary*
defence against cluster and ruin risk.

These tests cover the piece that closes that: turning the positions the user
actually holds into capital fractions, and -- when the denominator for that
is missing -- saying so instead of returning zeros that read as "nothing is
held".
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from turboedge.positions.usage import portfolio_risk_verdict, portfolio_usage
from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import (
    Direction,
    ManualPosition,
    PositionStatus,
    ProductSnapshot,
    ProductType,
)

_AS_OF = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store(tmp_path / "t.duckdb") as s:
        s.init_schema()
        yield s


def _snapshot(isin: str, *, underlying: str | None) -> ProductSnapshot:
    return ProductSnapshot(
        isin=isin,
        wkn=isin[-6:],
        issuer="BankA",
        venue="stuttgart",
        underlying_raw=underlying or "unknown",
        underlying_id=underlying,
        direction=Direction.LONG,
        product_type=ProductType.TURBO_OPEN_END,
        financing_level=18000.0,
        knockout_barrier=18100.0,
        ratio=0.01,
        currency="EUR",
        underlying_currency="EUR",
        quanto=False,
        open_end=True,
        bid=4.99,
        ask=5.01,
        bid_size=100.0,
        ask_size=100.0,
        quote_timestamp=_AS_OF,
        quote_presence=True,
        bid_only=False,
        knocked_out=False,
        raw_hash=hashlib.sha256(isin.encode()).hexdigest(),
        observation_time=_AS_OF,
        available_at=_AS_OF,
        retrieved_at=_AS_OF,
        source="fake",
        parser_version="1",
        is_stale=False,
        quality_score=0.9,
    )


def _position(isin: str | None, *, qty: float, price: float) -> ManualPosition:
    return ManualPosition(
        wkn=(isin or "XX0000")[-6:],
        isin=isin,
        qty=qty,
        entry_price=price,
        entry_date=date(2026, 10, 1),
        status=PositionStatus.OPEN,
        created_at=_AS_OF,
        updated_at=_AS_OF,
    )


_CLUSTERS = {"DAX": "corr_1", "MDAX": "corr_1", "NDX": "corr_2"}


def _cluster_for(underlying: str) -> str | None:
    return _CLUSTERS.get(underlying)


# -- the missing denominator ---------------------------------------------


def test_unset_capital_is_reported_not_treated_as_nothing_held(store: Store) -> None:
    # The whole point. Returning 0.0 silently is what made the caps inert;
    # an absent input must not read as a satisfied constraint.
    store.insert_position(_position("DE000TEST001", qty=100, price=5.0))

    usage = portfolio_usage(store, research_capital_eur=None)

    assert not usage.capital_known
    assert not usage.enforceable
    assert usage.open_positions == 1, "it still knows something is held"
    assert usage.total_fraction == 0.0
    warning = usage.warning()
    assert warning is not None
    assert "research_capital_eur" in warning
    assert "NOT enforced" in warning


def test_a_zero_or_negative_capital_is_also_unknown(store: Store) -> None:
    assert not portfolio_usage(store, research_capital_eur=0.0).capital_known
    assert not portfolio_usage(store, research_capital_eur=-1.0).capital_known


def test_no_positions_and_no_capital_still_warns(store: Store) -> None:
    # The caps are unenforceable regardless of whether anything is held
    # right now, and the operator should know that before a position exists.
    usage = portfolio_usage(store, research_capital_eur=None)

    assert usage.open_positions == 0
    assert usage.warning() is not None


# -- with the denominator ------------------------------------------------


def test_a_held_position_becomes_a_capital_fraction(store: Store) -> None:
    store.append_product_snapshots([_snapshot("DE000TEST001", underlying="DAX")])
    store.insert_position(_position("DE000TEST001", qty=100, price=5.0))

    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )

    assert usage.capital_known
    assert usage.total_fraction == pytest.approx(500.0 / 25_000.0)
    assert usage.by_cluster == {"corr_1": pytest.approx(0.02)}
    assert usage.unattributed == ()
    assert usage.warning() is None


def test_positions_in_one_cluster_accumulate(store: Store) -> None:
    # Two underlyings, one correlation cluster. This is the sum
    # `max_cluster_fraction` is supposed to be net of.
    store.append_product_snapshots(
        [
            _snapshot("DE000TEST001", underlying="DAX"),
            _snapshot("DE000TEST002", underlying="MDAX"),
        ]
    )
    store.insert_position(_position("DE000TEST001", qty=100, price=5.0))
    store.insert_position(_position("DE000TEST002", qty=200, price=5.0))

    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )

    assert usage.by_cluster == {"corr_1": pytest.approx(0.06)}
    assert usage.total_fraction == pytest.approx(0.06)


def test_separate_clusters_stay_separate(store: Store) -> None:
    store.append_product_snapshots(
        [
            _snapshot("DE000TEST001", underlying="DAX"),
            _snapshot("DE000TEST003", underlying="NDX"),
        ]
    )
    store.insert_position(_position("DE000TEST001", qty=100, price=5.0))
    store.insert_position(_position("DE000TEST003", qty=100, price=5.0))

    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )

    assert usage.by_cluster == {
        "corr_1": pytest.approx(0.02),
        "corr_2": pytest.approx(0.02),
    }
    assert usage.total_fraction == pytest.approx(0.04)


def test_a_closed_position_consumes_nothing(store: Store) -> None:
    store.append_product_snapshots([_snapshot("DE000TEST001", underlying="DAX")])
    closed = _position("DE000TEST001", qty=100, price=5.0).model_copy(
        update={"status": PositionStatus.CLOSED, "exit_price": 6.0, "exit_date": date(2026, 10, 20)}
    )
    store.insert_position(closed)

    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )

    assert usage.open_positions == 0
    assert usage.total_fraction == 0.0


# -- what cannot be attributed -------------------------------------------


def test_an_unscanned_isin_counts_toward_the_total_but_no_cluster(store: Store) -> None:
    # A position whose ISIN this system never scanned has no underlying on
    # record, so no cluster can be charged for it. Dropping it would
    # understate deployed capital; guessing a cluster would be worse.
    store.insert_position(_position("DE000UNKNOWN", qty=100, price=5.0))

    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )

    assert usage.total_fraction == pytest.approx(0.02)
    assert usage.by_cluster == {}
    assert len(usage.unattributed) == 1
    warning = usage.warning()
    assert warning is not None
    assert "could not be attributed" in warning


def test_a_position_without_an_isin_is_unattributed(store: Store) -> None:
    store.insert_position(_position(None, qty=100, price=5.0))

    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )

    assert usage.total_fraction == pytest.approx(0.02)
    assert len(usage.unattributed) == 1


def test_an_underlying_outside_every_cluster_is_unattributed(store: Store) -> None:
    store.append_product_snapshots([_snapshot("DE000TEST009", underlying="SMI")])
    store.insert_position(_position("DE000TEST009", qty=100, price=5.0))

    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )

    assert usage.by_cluster == {}
    assert len(usage.unattributed) == 1


def test_without_a_cluster_resolver_nothing_is_attributed(store: Store) -> None:
    # Total capital is still answerable without a clustering; cluster
    # budgets are not.
    store.append_product_snapshots([_snapshot("DE000TEST001", underlying="DAX")])
    store.insert_position(_position("DE000TEST001", qty=100, price=5.0))

    usage = portfolio_usage(store, research_capital_eur=25_000.0)

    assert usage.total_fraction == pytest.approx(0.02)
    assert usage.by_cluster == {}


# -- the ISIN -> underlying lookup the attribution rests on ---------------


def test_underlying_for_isin_resolves_from_snapshots(store: Store) -> None:
    store.append_product_snapshots([_snapshot("DE000TEST001", underlying="DAX")])

    assert store.underlying_for_isin("DE000TEST001") == "DAX"


def test_underlying_for_isin_is_none_for_an_unknown_product(store: Store) -> None:
    assert store.underlying_for_isin("DE000NOPE000") is None


def test_underlying_for_isin_ignores_rows_without_an_underlying(store: Store) -> None:
    store.append_product_snapshots([_snapshot("DE000TEST001", underlying=None)])

    assert store.underlying_for_isin("DE000TEST001") is None


# -- per-position detail, because clusters count positions too ------------


def test_each_held_position_stays_a_separate_entry(store: Store) -> None:
    # `ranking/cluster.py` caps the *number* of active positions per cluster
    # (`max_active_positions_per_cluster`, 3) as well as their capital.
    # Pre-summing three holdings into one entry would under-count them by
    # two and let a fourth position in.
    store.append_product_snapshots(
        [
            _snapshot("DE000TEST001", underlying="DAX"),
            _snapshot("DE000TEST002", underlying="MDAX"),
        ]
    )
    store.insert_position(_position("DE000TEST001", qty=100, price=5.0))
    store.insert_position(_position("DE000TEST002", qty=100, price=5.0))

    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )

    in_cluster = [item for item in usage.held if item.cluster_id == "corr_1"]
    assert len(in_cluster) == 2, "two holdings must stay two entries"
    assert usage.by_cluster["corr_1"] == pytest.approx(0.04)
    assert {item.underlying_id for item in in_cluster} == {"DAX", "MDAX"}


def test_the_unattributed_fraction_is_separable_from_the_clusters(store: Store) -> None:
    # The portfolio total is recomputed from the per-cluster lists during a
    # batch scan, so the part no cluster carries has to be available on its
    # own or it would silently drop out.
    store.append_product_snapshots([_snapshot("DE000TEST001", underlying="DAX")])
    store.insert_position(_position("DE000TEST001", qty=100, price=5.0))
    store.insert_position(_position("DE000UNKNOWN", qty=100, price=5.0))

    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )

    assert usage.unattributed_fraction == pytest.approx(0.02)
    assert sum(usage.by_cluster.values()) == pytest.approx(0.02)
    assert usage.unattributed_fraction + sum(usage.by_cluster.values()) == pytest.approx(
        usage.total_fraction
    )


def test_an_unattributed_position_is_still_a_held_entry(store: Store) -> None:
    store.insert_position(_position("DE000UNKNOWN", qty=100, price=5.0))

    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )

    assert len(usage.held) == 1
    assert usage.held[0].cluster_id is None
    assert usage.held[0].fraction == pytest.approx(0.02)


# -- the fail-closed portfolio verdict -----------------------------------


def test_unknown_capital_fails_the_portfolio_check(store: Store) -> None:
    # The core of the audit's P0.3: a portfolio limit that cannot be
    # evaluated is not a limit that has been satisfied. Before this, the
    # caps were netted against 0.0, so "unknown" read as "nothing
    # deployed" -- and ACTIONABLE was reachable on that basis.
    usage = portfolio_usage(store, research_capital_eur=None)

    verdict = portfolio_risk_verdict(
        usage,
        max_total_risk=0.30,
        max_cluster_fraction=0.15,
        max_issuer_fraction=0.05,
    )

    assert not verdict.passed
    assert any("research_capital_unknown" in r for r in verdict.reasons)
    # Not a cap breach, so it must not be reported as a second warning on
    # top of the usage warning that already says this.
    assert verdict.breach_reasons == ()


def test_an_empty_portfolio_with_known_capital_passes(store: Store) -> None:
    # The check must be satisfiable, or it is a disabled feature pretending
    # to be a safeguard.
    usage = portfolio_usage(store, research_capital_eur=25_000.0)

    verdict = portfolio_risk_verdict(
        usage,
        max_total_risk=0.30,
        max_cluster_fraction=0.15,
        max_issuer_fraction=0.05,
    )

    assert verdict.passed
    assert verdict.reasons == ()
    assert "within caps" in verdict.describe()


def test_a_total_risk_breach_fails_and_is_a_breach_reason(store: Store) -> None:
    store.append_product_snapshots([_snapshot("DE000TEST001", underlying="DAX")])
    store.insert_position(_position("DE000TEST001", qty=1000, price=10.0))  # 0.4 of capital

    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )
    verdict = portfolio_risk_verdict(
        usage,
        max_total_risk=0.30,
        max_cluster_fraction=0.50,
        max_issuer_fraction=0.05,
    )

    assert not verdict.passed
    assert any("total_risk_exceeded" in r for r in verdict.reasons)
    assert verdict.breach_reasons, "a real breach is worth its own warning"


def test_a_cluster_breach_fails(store: Store) -> None:
    store.append_product_snapshots([_snapshot("DE000TEST001", underlying="DAX")])
    store.insert_position(_position("DE000TEST001", qty=1000, price=5.0))  # 0.2 of capital

    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )
    verdict = portfolio_risk_verdict(
        usage,
        max_total_risk=0.90,
        max_cluster_fraction=0.15,
        max_issuer_fraction=0.05,
    )

    assert not verdict.passed
    assert any("cluster_risk_exceeded" in r and "corr_1" in r for r in verdict.reasons)


def test_an_unattributable_holding_fails_the_check(store: Store) -> None:
    # The per-cluster cap genuinely cannot be evaluated for a position whose
    # cluster is unknown, so the honest answer is that the check is
    # incomplete -- which fail-closed means blocking.
    store.insert_position(_position("DE000UNKNOWN", qty=10, price=5.0))

    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )
    verdict = portfolio_risk_verdict(
        usage,
        max_total_risk=0.30,
        max_cluster_fraction=0.15,
        max_issuer_fraction=0.05,
    )

    assert not verdict.passed
    assert any("unattributed_exposure" in r for r in verdict.reasons)
    assert verdict.breach_reasons == (), "a missing input is not a cap breach"


def test_the_verdict_never_invents_a_cap(store: Store) -> None:
    # Both caps are passed in from SizingConfig, the single place they live.
    # GOVERNANCE.md §5.2 still records two of them as UNRESOLVED, and this
    # module must not quietly pick a value.
    import inspect

    source = inspect.getsource(portfolio_risk_verdict)

    assert "0.0" not in source.replace("0.0000", ""), source
    for literal in ("0.05", "0.15", "0.02", "0.30"):
        assert literal not in source, f"{literal} is hard-coded in the verdict"


# -- the four worst-case limits (operator decision 2026-10-25) -----------


def _at_fraction(fraction: float) -> tuple[float, float]:
    """(qty, entry_price) worth `fraction` of a 25,000 EUR capital base."""
    return 1000.0, fraction * 25_000.0 / 1000.0


def _held(store: Store, isin: str, *, fraction: float, underlying: str | None, issuer: str) -> None:
    if underlying is not None:
        store.append_product_snapshots(
            [_snapshot(isin, underlying=underlying).model_copy(update={"issuer": issuer})]
        )
    qty, price = _at_fraction(fraction)
    store.insert_position(_position(isin, qty=qty, price=price))


def _verdict(store: Store):  # type: ignore[no-untyped-def]
    usage = portfolio_usage(
        store, research_capital_eur=25_000.0, cluster_for_underlying=_cluster_for
    )
    return usage, portfolio_risk_verdict(
        usage,
        max_total_risk=0.08,
        max_cluster_fraction=0.05,
        max_issuer_fraction=0.05,
    )


def test_a_portfolio_inside_every_limit_passes(store: Store) -> None:
    _held(store, "DE000TEST001", fraction=0.02, underlying="DAX", issuer="BankA")
    _held(store, "DE000TEST003", fraction=0.02, underlying="NDX", issuer="BankB")

    usage, verdict = _verdict(store)

    assert verdict.passed, verdict.reasons
    assert usage.total_fraction == pytest.approx(0.04)


def test_a_cluster_over_five_percent_fails(store: Store) -> None:
    # DAX and MDAX share a correlation cluster, so two 3 % positions put 6 %
    # in one cluster even though each is inside the 2 % position cap.
    _held(store, "DE000TEST001", fraction=0.03, underlying="DAX", issuer="BankA")
    _held(store, "DE000TEST002", fraction=0.03, underlying="MDAX", issuer="BankB")

    _usage, verdict = _verdict(store)

    assert not verdict.passed
    assert any("cluster_risk_exceeded" in r and "corr_1" in r for r in verdict.reasons)


def test_one_issuer_over_five_percent_fails(store: Store) -> None:
    # Same issuer across two uncorrelated underlyings: no cluster is
    # breached, the total is inside 8 %, and the issuer cap is what catches
    # it. This is the limit that nothing enforced before 2026-10-25.
    _held(store, "DE000TEST001", fraction=0.03, underlying="DAX", issuer="BankA")
    _held(store, "DE000TEST003", fraction=0.03, underlying="NDX", issuer="BankA")

    _usage, verdict = _verdict(store)

    assert not verdict.passed
    assert any("issuer_risk_exceeded" in r and "BankA" in r for r in verdict.reasons)
    assert not any("cluster_risk_exceeded" in r for r in verdict.reasons)
    assert not any("total_risk_exceeded" in r for r in verdict.reasons)


def test_several_issuers_inside_their_own_limits_pass(store: Store) -> None:
    for i, issuer in enumerate(("BankA", "BankB", "BankC")):
        _held(
            store,
            f"DE000TEST00{i + 1}",
            fraction=0.02,
            underlying=("DAX", "MDAX", "NDX")[i],
            issuer=issuer,
        )

    usage, verdict = _verdict(store)

    assert sorted(usage.by_issuer) == ["BankA", "BankB", "BankC"]
    # corr_1 holds DAX+MDAX = 4 %, inside the 5 % cluster cap.
    assert verdict.passed, verdict.reasons


def test_a_total_over_eight_percent_fails(store: Store) -> None:
    # Spread across three clusters and three issuers so that neither of
    # those caps is what catches it.
    for i, (underlying, issuer) in enumerate(
        (("DAX", "BankA"), ("NDX", "BankB"), ("SMI", "BankC"))
    ):
        _held(store, f"DE000TEST00{i + 1}", fraction=0.03, underlying=underlying, issuer=issuer)

    _usage, verdict = _verdict(store)

    assert not verdict.passed
    assert any("total_risk_exceeded" in r for r in verdict.reasons)


def test_an_unattributable_issuer_fails_the_check(store: Store) -> None:
    # The issuer cap genuinely cannot be evaluated for a holding whose
    # issuer is unknown, so fail-closed means blocking.
    qty, price = _at_fraction(0.01)
    store.insert_position(_position("DE000UNKNOWN", qty=qty, price=price))

    _usage, verdict = _verdict(store)

    assert not verdict.passed
    assert any("unattributed_issuer" in r for r in verdict.reasons)


# -- reserved exposure closes the cross-scan gap -------------------------


def _actionable_entry(isin: str, *, issuer: str, fraction: float, underlying: str = "DAX"):  # type: ignore[no-untyped-def]
    from turboedge.storage.schemas import (
        Category,
        LedgerEntry,
        LedgerEntryStatus,
    )

    return LedgerEntry(
        run_id="run-1",
        candidate_id=f"cand-{isin}",
        signal_id="sig-1",
        signal_version_hash="h",
        trial_id="TR-test",
        prediction_time=_AS_OF,
        underlying=underlying,
        direction=Direction.LONG,
        horizon_days=5,
        regime_bucket=None,
        cluster_id=None,
        feature_hash="fh",
        model_hash="mh",
        config_hash="ch",
        git_commit=None,
        category=Category.ACTIONABLE,
        selected_wkn=None,
        selected_isin=isin,
        issuer=issuer,
        entry_bid=4.99,
        entry_ask=5.0,
        entry_spread=0.002,
        entry_quote_timestamp=_AS_OF,
        entry_underlying_timestamp=_AS_OF,
        financing_level_entry=19_000.0,
        barrier_entry=19_000.0,
        ratio=0.01,
        fx=1.0,
        predicted_return=0.0,
        p_profit=0.5,
        p_ko=0.05,
        expected_shortfall=-0.1,
        lcb_ev=0.01,
        uncertainty=0.02,
        shrinkage_intensity=0.5,
        is_shadow=False,
        shadow_stratum=None,
        suggested_position_fraction=fraction,
        exit_due=_AS_OF.date(),
        alternatives=[],
        feature_snapshot={},
        status=LedgerEntryStatus.OPEN,
    )


def test_a_proposal_already_emitted_consumes_the_budget(store: Store) -> None:
    # GOVERNANCE.md §5.2 counts "bereits verbindlich reservierte neue
    # Positionen" in the 8 % total. Without this, two scans on the same day
    # could each propose up to the full budget -- the system cannot know
    # which proposal the operator acted on.
    store.append_ledger_entries([_actionable_entry("DE000PROP001", issuer="BankA", fraction=0.02)])

    usage, verdict = _verdict(store)

    assert usage.reserved_fraction == pytest.approx(0.02)
    assert usage.total_fraction == pytest.approx(0.02)
    assert verdict.passed, verdict.reasons


def test_reserved_proposals_can_breach_the_issuer_limit(store: Store) -> None:
    # Three 2 % proposals from one issuer are 6 %, past the 5 % issuer cap,
    # even with nothing held at all.
    store.append_ledger_entries(
        [_actionable_entry(f"DE000PROP00{i}", issuer="BankA", fraction=0.02) for i in range(1, 4)]
    )

    _usage, verdict = _verdict(store)

    assert not verdict.passed
    assert any("issuer_risk_exceeded" in r for r in verdict.reasons)


def test_a_watch_candidate_reserves_nothing(store: Store) -> None:
    # WATCH is not a proposal to act on, so it consumes no budget. Counting
    # it would starve the limits on candidates nobody was asked to buy.
    from turboedge.storage.schemas import Category

    entry = _actionable_entry("DE000WATCH01", issuer="BankA", fraction=0.02).model_copy(
        update={"category": Category.WATCH}
    )
    store.append_ledger_entries([entry])

    usage, _verdict_result = _verdict(store)

    assert usage.reserved_fraction == 0.0
    assert usage.total_fraction == 0.0
