"""Tests for the missing-data and quote-executability audit.

The classification tests matter more than they look: every one of the eight
states has a different consequence for a forward result, and collapsing two
of them is how a dataset gets reported as healthier than it is. In
particular NOT_YET_DUE and OVERDUE_UNLABELED must never merge -- the first
is a young ledger, the second is data going missing, and
``docs/measured_results.md`` records a production run where telling them
apart was the whole question.

The bounds tests check the property that makes the approach honest: the
interval must contain the truth under its stated assumptions and must not
narrow when an assumption is relaxed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from turboedge.research.data_audit import (
    ObservationState,
    classify,
    exit_value_bounds,
    ledger_census,
    pnl_bounds,
    quote_age_seconds,
    quote_freshness,
    snapshot_staleness_advantage,
    staleness_advantage,
)
from turboedge.storage.schemas import (
    Category,
    Direction,
    ExitReason,
    FieldReliability,
    LedgerEntry,
    LedgerEntryStatus,
    LedgerLabel,
)

NOW = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)


@pytest.fixture
def make_entry() -> Callable[..., LedgerEntry]:
    def _make(**overrides: object) -> LedgerEntry:
        defaults: dict[str, object] = dict(
            run_id="run-1",
            candidate_id="cand-1",
            signal_id="tsmom_horizon_norm_v1",
            signal_version_hash="sig-hash-1",
            trial_id="TR-2026Q3-abc123",
            prediction_time=datetime(2026, 8, 1, 8, 0, tzinfo=UTC),
            underlying="DAX",
            direction=Direction.LONG,
            horizon_days=7,
            regime_bucket=None,
            cluster_id=None,
            feature_hash="feat-hash-1",
            model_hash="model-hash-1",
            config_hash="cfg-hash-1",
            git_commit=None,
            category=Category.WATCH,
            selected_wkn="ABC123",
            selected_isin="DE000ABC1234",
            issuer="TestBank",
            entry_bid=4.80,
            entry_ask=4.86,
            entry_spread=(4.86 - 4.80) / 4.86,
            entry_quote_timestamp=datetime(2026, 8, 1, 8, 0, tzinfo=UTC),
            entry_underlying_timestamp=datetime(2026, 8, 1, 8, 0, tzinfo=UTC),
            financing_level_entry=18000.0,
            barrier_entry=18000.0,
            ratio=0.01,
            fx=1.0,
            predicted_return=0.01,
            p_profit=0.55,
            p_ko=0.05,
            expected_shortfall=-0.1,
            lcb_ev=0.001,
            uncertainty=0.02,
            shrinkage_intensity=0.5,
            is_shadow=False,
            shadow_stratum=None,
            suggested_position_fraction=None,
            exit_due=date(2026, 8, 10),
            alternatives=[],
            feature_snapshot={},
            status=LedgerEntryStatus.OPEN,
        )
        defaults.update(overrides)
        return LedgerEntry(**defaults)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def make_label() -> Callable[..., LedgerLabel]:
    def _make(**overrides: object) -> LedgerLabel:
        defaults: dict[str, object] = dict(
            entry_id="placeholder",
            labeled_at=datetime(2026, 8, 10, 15, 30, tzinfo=UTC),
            exit_bid=5.10,
            exit_quote_timestamp=datetime(2026, 8, 10, 15, 30, tzinfo=UTC),
            financing_level_exit=18020.0,
            exit_reason=ExitReason.HORIZON,
            realized_selected_pnl=5.10 / 4.86 - 1,
            underlying_pnl=0.01,
            median_turbo_pnl=None,
            best_turbo_pnl=None,
            ideal_turbo_pnl=None,
            mfe=0.06,
            mae=-0.01,
            ko_hit=False,
            time_to_ko_days=None,
            ambiguous_path=False,
        )
        defaults.update(overrides)
        return LedgerLabel(**defaults)  # type: ignore[arg-type]

    return _make


class TestClassify:
    def test_an_open_entry_before_its_exit_is_young_not_missing(
        self, make_entry: Callable[..., LedgerEntry]
    ) -> None:
        entry = make_entry(exit_due=date(2026, 9, 1))
        assert classify((entry, None), now=NOW) is ObservationState.NOT_YET_DUE

    def test_an_open_entry_past_its_exit_is_overdue(
        self, make_entry: Callable[..., LedgerEntry]
    ) -> None:
        """The distinction the whole census exists for.

        A production run on 2026-10-23 reported ``labeled=0``, which was
        indistinguishable between "nothing was due" and "everything was due
        and failed" until the population was reported alongside it.
        """
        entry = make_entry(exit_due=date(2026, 8, 1))
        assert classify((entry, None), now=NOW) is ObservationState.OVERDUE_UNLABELED

    def test_the_boundary_day_counts_as_due(self, make_entry: Callable[..., LedgerEntry]) -> None:
        entry = make_entry(exit_due=NOW.date())
        assert classify((entry, None), now=NOW) is ObservationState.OVERDUE_UNLABELED

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({}, ObservationState.LABELED_HORIZON),
            (
                {"exit_reason": ExitReason.KO, "ko_hit": True, "realized_selected_pnl": -1.0},
                ObservationState.LABELED_KO,
            ),
            (
                {
                    "exit_reason": ExitReason.NO_EXIT_QUOTE_CONSERVATIVE,
                    "ambiguous_path": True,
                    "exit_quote_timestamp": None,
                },
                ObservationState.LABELED_NO_EXIT_QUOTE,
            ),
            ({"ambiguous_path": True}, ObservationState.LABELED_AMBIGUOUS),
            (
                {"exit_reason": ExitReason.EXPIRED_NO_DATA, "realized_selected_pnl": None},
                ObservationState.EXPIRED_NO_DATA,
            ),
            ({"realized_selected_pnl": None}, ObservationState.LABELED_WITHOUT_PNL),
        ],
    )
    def test_each_label_shape_maps_to_its_own_state(
        self,
        make_entry: Callable[..., LedgerEntry],
        make_label: Callable[..., LedgerLabel],
        overrides: dict[str, object],
        expected: ObservationState,
    ) -> None:
        row = (make_entry(), make_label(**overrides))
        assert classify(row, now=NOW) is expected

    def test_the_conservative_fallback_is_not_counted_as_a_real_exit(
        self, make_entry: Callable[..., LedgerEntry], make_label: Callable[..., LedgerLabel]
    ) -> None:
        """It carries a number, and that number is an assumption.

        Counting it as a real observation is what would make the forward
        sample look larger than the evidence in it.
        """
        label = make_label(
            exit_reason=ExitReason.NO_EXIT_QUOTE_CONSERVATIVE,
            ambiguous_path=True,
            exit_quote_timestamp=None,
        )
        state = classify((make_entry(), label), now=NOW)
        from turboedge.research.data_audit import TRUSTWORTHY_STATES

        assert state not in TRUSTWORTHY_STATES

    def test_a_naive_now_is_refused(self, make_entry: Callable[..., LedgerEntry]) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            classify((make_entry(), None), now=datetime(2026, 8, 15, 12, 0))


class TestLedgerCensus:
    def test_an_entirely_unlabeled_ledger_says_so_instead_of_reporting_health(
        self, make_entry: Callable[..., LedgerEntry]
    ) -> None:
        rows = [(make_entry(exit_due=date(2026, 9, 1)), None) for _ in range(5)]
        census = ledger_census(rows, now=NOW)
        assert census.total == 5
        assert census.trustworthy == 0
        assert "not measurable" in census.summary()

    def test_counts_split_by_underlying_and_issuer(
        self, make_entry: Callable[..., LedgerEntry], make_label: Callable[..., LedgerLabel]
    ) -> None:
        rows = [
            (make_entry(underlying="DAX", issuer="A"), make_label()),
            (make_entry(underlying="NDX", issuer="B", exit_due=date(2026, 8, 1)), None),
        ]
        census = ledger_census(rows, now=NOW)
        assert census.by_underlying["DAX"] == {ObservationState.LABELED_HORIZON.value: 1}
        assert census.by_issuer["B"] == {ObservationState.OVERDUE_UNLABELED.value: 1}

    def test_a_contrast_with_an_empty_side_is_marked_not_comparable(
        self, make_entry: Callable[..., LedgerEntry], make_label: Callable[..., LedgerLabel]
    ) -> None:
        """No evidence of bias must not come out of no data."""
        rows = [(make_entry(), make_label())]
        census = ledger_census(rows, now=NOW)
        spread = next(c for c in census.contrasts if c.name == "entry_spread")
        assert spread.n_trustworthy == 1
        assert spread.n_assumed == 0
        assert not spread.comparable
        assert math.isnan(spread.mean_assumed)

    def test_a_contrast_detects_a_planted_selection_effect(
        self, make_entry: Callable[..., LedgerEntry], make_label: Callable[..., LedgerLabel]
    ) -> None:
        """Entries that go missing have wider spreads here, by construction.

        This is the shape that would mean the labeled sample is not a
        random subset -- MNAR cannot be identified from it, but it can at
        least be *detected*.
        """
        rows: list[tuple[LedgerEntry, LedgerLabel | None]] = []
        for _ in range(10):
            rows.append((make_entry(entry_spread=0.01), make_label()))
        for _ in range(10):
            rows.append((make_entry(entry_spread=0.09, exit_due=date(2026, 8, 1)), None))
        census = ledger_census(rows, now=NOW)
        spread = next(c for c in census.contrasts if c.name == "entry_spread")
        assert spread.comparable
        assert spread.mean_assumed == pytest.approx(0.09)
        assert spread.mean_trustworthy == pytest.approx(0.01)
        assert spread.difference > 0.0


class TestExitValueBounds:
    def test_the_interval_brackets_the_intrinsic_value(self) -> None:
        bounds = exit_value_bounds(
            spot_at_exit=18500.0,
            barrier=18000.0,
            ratio=0.01,
            direction=Direction.LONG,
            ko_occurred=False,
        )
        intrinsic = (18500.0 - 18000.0) / 0.01
        assert bounds.lower < intrinsic < bounds.upper
        assert bounds.width > 0

    def test_a_knock_out_collapses_the_interval(self) -> None:
        bounds = exit_value_bounds(
            spot_at_exit=17900.0,
            barrier=18000.0,
            ratio=0.01,
            direction=Direction.LONG,
            ko_occurred=True,
        )
        assert bounds.lower == bounds.upper == 0.0
        assert bounds.ko
        assert bounds.width == 0.0

    def test_a_ko_residual_is_carried_through(self) -> None:
        """Mini futures keep a residual; the bounds must not zero it."""
        bounds = exit_value_bounds(
            spot_at_exit=17900.0,
            barrier=18000.0,
            ratio=0.01,
            direction=Direction.SHORT,
            ko_occurred=True,
            ko_residual=0.42,
        )
        assert bounds.lower == bounds.upper == 0.42

    def test_short_direction_inverts_the_intrinsic(self) -> None:
        long_bounds = exit_value_bounds(
            spot_at_exit=18500.0,
            barrier=18000.0,
            ratio=0.01,
            direction=Direction.LONG,
            ko_occurred=False,
        )
        short_bounds = exit_value_bounds(
            spot_at_exit=17500.0,
            barrier=18000.0,
            ratio=0.01,
            direction=Direction.SHORT,
            ko_occurred=False,
        )
        assert long_bounds.lower == pytest.approx(short_bounds.lower)

    def test_a_barrier_crossed_without_a_recorded_ko_clamps_to_zero(self) -> None:
        """Conservative, and it never raises the lower bound."""
        bounds = exit_value_bounds(
            spot_at_exit=17500.0,
            barrier=18000.0,
            ratio=0.01,
            direction=Direction.LONG,
            ko_occurred=False,
        )
        assert bounds.lower == 0.0
        assert bounds.upper == 0.0

    def test_wider_assumptions_never_narrow_the_interval(self) -> None:
        """The property that makes a bound a bound.

        If relaxing an assumption could tighten the interval, the interval
        would not be one -- it would be a fit.
        """
        common = dict(
            spot_at_exit=18500.0,
            barrier=18000.0,
            ratio=0.01,
            direction=Direction.LONG,
            ko_occurred=False,
        )
        tight = exit_value_bounds(**common, max_premium_fraction=0.05, max_spread_fraction=0.02)  # type: ignore[arg-type]
        wide = exit_value_bounds(**common, max_premium_fraction=0.20, max_spread_fraction=0.10)  # type: ignore[arg-type]
        assert wide.lower <= tight.lower
        assert wide.upper >= tight.upper
        assert wide.width >= tight.width

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"ratio": 0.0}, "ratio must be > 0"),
            ({"spot_at_exit": -1.0}, "must be >= 0"),
            ({"max_premium_fraction": 1.0}, r"max_premium_fraction must be in \[0, 1\)"),
            ({"max_spread_fraction": -0.1}, r"max_spread_fraction must be in \[0, 1\)"),
            ({"ko_residual": -1.0}, "ko_residual must be >= 0"),
        ],
    )
    def test_invalid_inputs_are_refused(self, overrides: dict[str, float], match: str) -> None:
        base: dict[str, object] = dict(
            spot_at_exit=18500.0,
            barrier=18000.0,
            ratio=0.01,
            direction=Direction.LONG,
            ko_occurred=False,
        )
        with pytest.raises(ValueError, match=match):
            exit_value_bounds(**{**base, **overrides})  # type: ignore[arg-type]


class TestPnlBounds:
    def test_a_total_loss_is_minus_one(self) -> None:
        bounds = exit_value_bounds(
            spot_at_exit=17900.0,
            barrier=18000.0,
            ratio=0.01,
            direction=Direction.LONG,
            ko_occurred=True,
        )
        assert pnl_bounds(entry_ask=4.86, bounds=bounds) == (-1.0, -1.0)

    def test_the_return_interval_contains_the_realised_return(self) -> None:
        """A worked case: entry at the ask, exit at a plausible real bid."""
        bounds = exit_value_bounds(
            spot_at_exit=18550.0,
            barrier=18000.0,
            ratio=0.01,
            direction=Direction.LONG,
            ko_occurred=False,
        )
        intrinsic = (18550.0 - 18000.0) / 0.01
        realised = (intrinsic * 1.01) / 50000.0 - 1.0
        low, high = pnl_bounds(entry_ask=50000.0, bounds=bounds)
        assert low <= realised <= high

    def test_a_non_positive_entry_ask_is_refused(self) -> None:
        bounds = exit_value_bounds(
            spot_at_exit=18500.0,
            barrier=18000.0,
            ratio=0.01,
            direction=Direction.LONG,
            ko_occurred=False,
        )
        with pytest.raises(ValueError, match="entry_ask must be > 0"):
            pnl_bounds(entry_ask=0.0, bounds=bounds)


class TestQuoteFreshness:
    def test_age_is_the_gap_between_quote_and_decision(
        self, make_entry: Callable[..., LedgerEntry]
    ) -> None:
        entry = make_entry(
            entry_quote_timestamp=datetime(2026, 8, 1, 7, 58, tzinfo=UTC),
            prediction_time=datetime(2026, 8, 1, 8, 0, tzinfo=UTC),
        )
        assert quote_age_seconds(entry) == 120.0

    def test_a_quote_newer_than_the_decision_is_reported_not_clamped(
        self, make_entry: Callable[..., LedgerEntry]
    ) -> None:
        """A negative age is a provenance defect; hiding it loses the signal."""
        entry = make_entry(
            entry_quote_timestamp=datetime(2026, 8, 1, 8, 5, tzinfo=UTC),
            prediction_time=datetime(2026, 8, 1, 8, 0, tzinfo=UTC),
        )
        assert quote_age_seconds(entry) == -300.0
        assert quote_freshness([entry]).n_negative == 1

    def test_an_empty_ledger_reports_no_quotes_rather_than_age_zero(self) -> None:
        stats = quote_freshness([])
        assert stats.n == 0
        assert math.isnan(stats.median_seconds)
        assert "no entry carried a quote timestamp" in stats.summary()

    def test_percentiles_track_the_tail(self, make_entry: Callable[..., LedgerEntry]) -> None:
        entries = [
            make_entry(
                entry_quote_timestamp=datetime(2026, 8, 1, 8, 0, tzinfo=UTC),
                prediction_time=datetime(2026, 8, 1, 8, 0, second=0, tzinfo=UTC)
                + __import__("datetime").timedelta(seconds=s),
            )
            for s in (1, 2, 3, 4, 600)
        ]
        stats = quote_freshness(entries)
        assert stats.median_seconds == 3.0
        assert stats.max_seconds == 600.0
        assert stats.p90_seconds > stats.median_seconds


class TestStalenessAdvantage:
    def test_a_perfect_monotone_association_is_one(self) -> None:
        assoc = staleness_advantage(
            quote_ages=list(range(40)), advantages=[float(x) for x in range(40)]
        )
        assert assoc.spearman == pytest.approx(1.0)
        assert assoc.interpretable

    def test_a_reversed_association_is_minus_one(self) -> None:
        assoc = staleness_advantage(
            quote_ages=list(range(40)), advantages=[float(-x) for x in range(40)]
        )
        assert assoc.spearman == pytest.approx(-1.0)

    def test_ranks_handle_ties_without_inventing_an_order(self) -> None:
        """All-equal ages carry no ordering, so the correlation is undefined."""
        assoc = staleness_advantage(quote_ages=[5.0] * 10, advantages=list(range(10)))
        assert math.isnan(assoc.spearman)

    def test_a_small_sample_is_flagged_as_not_interpretable(self) -> None:
        assoc = staleness_advantage(quote_ages=[1.0, 2.0, 3.0], advantages=[1.0, 2.0, 3.0])
        assert not assoc.interpretable

    def test_outliers_do_not_dominate_the_way_they_would_in_pearson(self) -> None:
        """Why this is a rank correlation: quote ages are heavy-tailed.

        The bulk of these points carries no association at all; one
        enormous pair carries a Pearson correlation of ~1 on its own. On
        ranks that pair is worth exactly one observation, so the reported
        association stays near zero -- which is the truth about the bulk.
        """
        rng = np.random.default_rng(4)
        ages = [*rng.random(60), 1e6]
        adv = [*rng.random(60), 1e6]
        pearson = float(np.corrcoef(ages, adv)[0, 1])
        spearman = staleness_advantage(quote_ages=ages, advantages=adv).spearman
        assert pearson > 0.99
        assert abs(spearman) < 0.3

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            staleness_advantage(quote_ages=[1.0, 2.0], advantages=[1.0])

    def test_a_constant_input_is_not_an_interpretable_null(self) -> None:
        """A nan on 100 points is a question about the data, not a result.

        Without this, an archive that records one quote timestamp per batch
        would report "no association found" when what it actually has is no
        variation to find one in.
        """
        assoc = staleness_advantage(quote_ages=[7.0] * 100, advantages=list(range(100)))
        assert not assoc.interpretable
        assert assoc.degenerate_side == "quote_age"
        assert assoc.distinct_ages == 1
        assert assoc.distinct_advantages == 100

    def test_the_degenerate_side_is_named(self) -> None:
        flat_cheapness = staleness_advantage(quote_ages=list(range(50)), advantages=[0.0] * 50)
        assert flat_cheapness.degenerate_side == "cheapness"
        both = staleness_advantage(quote_ages=[1.0] * 50, advantages=[0.0] * 50)
        assert both.degenerate_side == "both"
        neither = staleness_advantage(
            quote_ages=list(range(50)), advantages=[float(x % 7) for x in range(50)]
        )
        assert neither.degenerate_side is None
        assert neither.interpretable


class TestSnapshotStalenessAdvantage:
    """Phase 2's core question, on comparable products from different issuers."""

    @staticmethod
    def _snap(
        make_product_snapshot: Callable[..., object],
        *,
        isin: str,
        issuer: str,
        ask: float,
        age_seconds: float,
    ) -> object:
        """One comparable product: same bucket, differing only in ask and age."""
        observed = datetime(2026, 9, 10, 15, 30, tzinfo=UTC)
        return make_product_snapshot(
            isin=isin,
            wkn=isin[-6:],
            issuer=issuer,
            ask=ask,
            bid=ask * 0.99,
            # `observation_time` IS the quote's own timestamp in this
            # schema, so the age has to come from `retrieved_at`.
            observation_time=observed - timedelta(seconds=age_seconds),
            available_at=observed,
            retrieved_at=observed,
            quote_timestamp=observed - timedelta(seconds=age_seconds),
            # The schema default is UNVERIFIED, and an unverified ratio is
            # correctly refused -- every per-unit cost divides by it.
            ratio_reliability=FieldReliability.CROSS_SOURCE_VERIFIED,
        )

    def test_a_planted_association_is_recovered(
        self, make_product_snapshot: Callable[..., object]
    ) -> None:
        """Cheaper products are given older quotes here, by construction.

        This is the adverse-selection shape: whoever picks the cheapest
        product is partly picking the least recently repriced one.
        """
        snaps = [
            self._snap(
                make_product_snapshot,
                isin=f"DE000STL{i:04d}",
                issuer="A" if i % 2 == 0 else "B",
                ask=5.00 - 0.01 * i,
                age_seconds=1.0 + 10.0 * i,
            )
            for i in range(40)
        ]
        report = snapshot_staleness_advantage(snaps)  # type: ignore[arg-type]
        assert report.groups >= 1
        assert report.association.interpretable
        assert report.association.spearman > 0.8

    def test_no_association_when_age_is_unrelated_to_price(
        self, make_product_snapshot: Callable[..., object]
    ) -> None:
        rng = np.random.default_rng(9)
        snaps = [
            self._snap(
                make_product_snapshot,
                isin=f"DE000IND{i:04d}",
                issuer="A" if i % 2 == 0 else "B",
                ask=float(4.50 + rng.random()),
                age_seconds=float(rng.random() * 300.0),
            )
            for i in range(60)
        ]
        report = snapshot_staleness_advantage(snaps)  # type: ignore[arg-type]
        assert abs(report.association.spearman) < 0.3

    def test_a_single_issuer_group_is_dropped_and_counted(
        self, make_product_snapshot: Callable[..., object]
    ) -> None:
        """Choosing between issuers needs two issuers.

        Within one issuer the price differences are that issuer's own
        curve, which is a different question.
        """
        snaps = [
            self._snap(
                make_product_snapshot,
                isin=f"DE000ONE{i:04d}",
                issuer="OnlyBank",
                ask=4.50 + 0.1 * i,
                age_seconds=10.0 * i,
            )
            for i in range(6)
        ]
        report = snapshot_staleness_advantage(snaps)  # type: ignore[arg-type]
        assert report.groups == 0
        assert report.dropped["single_issuer_group"] >= 1

    def test_a_snapshot_without_a_quote_timestamp_is_dropped_not_treated_as_fresh(
        self, make_product_snapshot: Callable[..., object]
    ) -> None:
        snaps = [
            self._snap(
                make_product_snapshot, isin="DE000AAA0001", issuer="A", ask=4.5, age_seconds=5.0
            ),
            make_product_snapshot(
                isin="DE000BBB0001",
                wkn="BBB001",
                issuer="B",
                quote_timestamp=None,
                ratio_reliability=FieldReliability.CROSS_SOURCE_VERIFIED,
            ),
        ]
        report = snapshot_staleness_advantage(snaps)  # type: ignore[arg-type]
        assert report.dropped["no_quote_timestamp"] == 1

    def test_a_knocked_out_product_is_excluded(
        self, make_product_snapshot: Callable[..., object]
    ) -> None:
        snaps = [
            make_product_snapshot(
                knocked_out=True, ratio_reliability=FieldReliability.CROSS_SOURCE_VERIFIED
            )
        ]
        report = snapshot_staleness_advantage(snaps)  # type: ignore[arg-type]
        assert report.dropped["knocked_out"] == 1
        assert report.groups == 0

    def test_an_empty_archive_reports_nothing_rather_than_zero_association(self) -> None:
        report = snapshot_staleness_advantage([])
        assert report.groups == 0
        assert math.isnan(report.association.spearman)
        assert not report.association.interpretable


def test_the_audit_never_reaches_the_decision_path() -> None:
    """A diagnostic may be *reported*; it must never be *acted on*.

    Unlike `research/first_passage.py` this module is deliberately
    reachable from the CLI -- a read-only report nobody can run is not a
    report, and `cli_learn.py` already imports `research/product_pairs.py`
    for the same reason. What must not happen is an audit figure feeding a
    candidate, a gate, a weight or a simulation, so the invariant is named
    by package rather than by a blanket ban: a missingness correction that
    silently entered sizing would be exactly the kind of change
    GOVERNANCE.md requires to be deliberate.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "turboedge"
    decision_path = (
        "pipeline",
        "ranking",
        "simulation",
        "models",
        "alpha",
        "learning",
        "positions",
        "forecast",
        "meta",
    )
    pattern = re.compile(r"\bdata_audit\b")
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if path.relative_to(root).parts[0] in decision_path and pattern.search(path.read_text())
    ]
    assert offenders == [], f"data_audit reached the decision path via {offenders}"

    # And outside the decision path, the CLI is the only importer. A new
    # one appearing here is not necessarily wrong, but it is a decision.
    importers = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "research" not in path.relative_to(root).parts and pattern.search(path.read_text())
    )
    assert importers == ["cli_learn.py"], f"unexpected importer(s): {importers}"
