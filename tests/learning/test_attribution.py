"""Tests for ensemble attribution.

The two that matter most are refusals of a kind:

* ``test_an_entry_whose_run_wrote_no_forecasts_is_not_attributable`` --
  the 2,785 labels from before 2026-10-26 are exactly this case, and no
  amount of current model state can reconstruct weights that were never
  written.
* ``test_an_unresolved_entry_still_counts_in_the_denominator`` -- a report
  must not look complete by quietly excluding what it could not resolve.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, date, datetime

import pytest

from turboedge.learning.attribution import (
    ENSEMBLE_MODEL_ID,
    attribution_report,
    beats_null,
    resolve_attribution,
)
from turboedge.storage.schemas import (
    Category,
    Direction,
    ExitReason,
    ForecastRecord,
    LedgerEntry,
    LedgerEntryStatus,
    LedgerLabel,
)

NOW = datetime(2026, 10, 26, 8, 0, tzinfo=UTC)


@pytest.fixture
def make_entry() -> Callable[..., LedgerEntry]:
    def _make(**over: object) -> LedgerEntry:
        defaults: dict[str, object] = dict(
            run_id="run-1",
            candidate_id="cand-1",
            signal_id="tsmom_horizon_norm_v1",
            signal_version_hash="sig-1",
            trial_id="TR-2026Q4-aaa111",
            prediction_time=NOW,
            underlying="DAX",
            direction=Direction.LONG,
            horizon_days=7,
            regime_bucket=None,
            cluster_id=None,
            feature_hash="feat-1",
            model_hash="ensemble-hash",
            config_hash="cfg-1",
            git_commit=None,
            category=Category.WATCH,
            selected_wkn="ABC123",
            selected_isin="DE000ABC1234",
            issuer="TestBank",
            entry_bid=4.80,
            entry_ask=4.86,
            entry_spread=(4.86 - 4.80) / 4.86,
            entry_quote_timestamp=NOW,
            entry_underlying_timestamp=NOW,
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
            exit_due=date(2026, 11, 4),
            alternatives=[],
            feature_snapshot={},
            status=LedgerEntryStatus.LABELED,
        )
        defaults.update(over)
        return LedgerEntry(**defaults)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def make_label() -> Callable[..., LedgerLabel]:
    def _make(**over: object) -> LedgerLabel:
        defaults: dict[str, object] = dict(
            entry_id="placeholder",
            labeled_at=NOW,
            exit_bid=5.10,
            exit_quote_timestamp=NOW,
            financing_level_exit=18020.0,
            exit_reason=ExitReason.HORIZON,
            realized_selected_pnl=0.05,
            underlying_pnl=0.012,
            median_turbo_pnl=None,
            best_turbo_pnl=None,
            ideal_turbo_pnl=None,
            mfe=0.06,
            mae=-0.01,
            ko_hit=False,
            time_to_ko_days=None,
            ambiguous_path=False,
        )
        defaults.update(over)
        return LedgerLabel(**defaults)  # type: ignore[arg-type]

    return _make


def _forecast(
    model_id: str,
    *,
    mean: float,
    p_up: float | None = 0.55,
    weights: dict[str, float] | None = None,
    horizon: int = 7,
    underlying: str = "DAX",
) -> ForecastRecord:
    return ForecastRecord(
        run_id="run-1",
        underlying_id=underlying,
        horizon_days=horizon,
        prediction_time=NOW,
        frozen_at=NOW,
        p_up=p_up,
        mean=mean,
        sigma=0.02,
        quantiles={},
        expected_shortfall_05=-0.03,
        uncertainty=0.02,
        model_id=model_id,
        model_hash=f"{model_id}-hash",
        signal_family="tsmom",
        n_train=500,
        n_effective=420.0,
        component_weights=weights or {},
        config_hash="cfg-1",
        git_commit=None,
    )


def _full_run() -> list[ForecastRecord]:
    weights = {"tsmom_forecast_v1": 0.6, "logistic_direction_v1": 0.4}
    return [
        _forecast("tsmom_forecast_v1", mean=0.010, p_up=0.60),
        _forecast("logistic_direction_v1", mean=0.004, p_up=0.52),
        _forecast(ENSEMBLE_MODEL_ID, mean=0.0076, p_up=0.57, weights=weights),
    ]


class TestResolveAttribution:
    def test_the_prediction_time_weights_come_back(
        self, make_entry: Callable[..., LedgerEntry]
    ) -> None:
        """The whole point: the weights as they stood, not as they are now."""
        attribution = resolve_attribution(make_entry(), _full_run())

        assert attribution.weights == {
            "tsmom_forecast_v1": 0.6,
            "logistic_direction_v1": 0.4,
        }
        assert attribution.complete
        assert attribution.ensemble_mean == pytest.approx(0.0076)

    def test_each_component_carries_its_own_forecast_and_weight(
        self, make_entry: Callable[..., LedgerEntry]
    ) -> None:
        attribution = resolve_attribution(make_entry(), _full_run())
        by_id = {c.model_id: c for c in attribution.components}

        assert by_id["tsmom_forecast_v1"].mean == pytest.approx(0.010)
        assert by_id["tsmom_forecast_v1"].weight_at_prediction == pytest.approx(0.6)
        assert by_id["logistic_direction_v1"].weight_at_prediction == pytest.approx(0.4)
        assert ENSEMBLE_MODEL_ID not in by_id  # the combination is not a component

    def test_an_entry_whose_run_wrote_no_forecasts_is_not_attributable(
        self, make_entry: Callable[..., LedgerEntry]
    ) -> None:
        """The 2,785 labels from before 2026-10-26, exactly.

        Nothing in current model state can reconstruct weights that were
        never written, and the brief forbids trying.
        """
        attribution = resolve_attribution(make_entry(), [])

        assert not attribution.complete
        assert attribution.components == ()
        assert attribution.weights == {}
        assert attribution.ensemble_model_hash is None

    def test_a_weighted_model_with_no_forecast_row_is_named(
        self, make_entry: Callable[..., LedgerEntry]
    ) -> None:
        """Partial attribution is usable, but not as if it were whole."""
        rows = [
            _forecast("tsmom_forecast_v1", mean=0.010),
            _forecast(
                ENSEMBLE_MODEL_ID,
                mean=0.0076,
                weights={"tsmom_forecast_v1": 0.6, "gone_v1": 0.4},
            ),
        ]
        attribution = resolve_attribution(make_entry(), rows)

        assert not attribution.complete
        assert attribution.missing_models == ("gone_v1",)

    def test_rows_from_another_horizon_or_underlying_are_ignored(
        self, make_entry: Callable[..., LedgerEntry]
    ) -> None:
        rows = [
            *_full_run(),
            _forecast("other_v1", mean=0.9, horizon=14),
            _forecast("other_v2", mean=0.9, underlying="NDX"),
        ]
        attribution = resolve_attribution(make_entry(horizon_days=7), rows)

        assert {c.model_id for c in attribution.components} == {
            "tsmom_forecast_v1",
            "logistic_direction_v1",
        }

    def test_weights_serialised_as_json_are_read_back(
        self, make_entry: Callable[..., LedgerEntry]
    ) -> None:
        """The store round-trips this column as a JSON string."""
        rows = [
            _forecast("tsmom_forecast_v1", mean=0.01),
            _forecast(ENSEMBLE_MODEL_ID, mean=0.01),
        ]
        rows[-1] = rows[-1].model_copy(update={"component_weights": '{"tsmom_forecast_v1": 1.0}'})
        attribution = resolve_attribution(make_entry(), rows)

        assert attribution.weights == {"tsmom_forecast_v1": 1.0}
        assert attribution.complete


class TestAttributionReport:
    def _rows(
        self,
        make_entry: Callable[..., LedgerEntry],
        make_label: Callable[..., LedgerLabel],
        *,
        n: int = 25,
        forecasts: list[ForecastRecord] | None = None,
    ) -> list[tuple[LedgerEntry, LedgerLabel, object]]:
        out = []
        for i in range(n):
            entry = make_entry()
            label = make_label(entry_id=entry.entry_id, underlying_pnl=0.01 + 0.001 * (i % 5))
            rows = _full_run() if forecasts is None else forecasts
            out.append((entry, label, resolve_attribution(entry, rows)))
        return out

    def test_each_model_is_scored_against_the_underlying_return(
        self, make_entry: Callable[..., LedgerEntry], make_label: Callable[..., LedgerLabel]
    ) -> None:
        """A forecast is a claim about the market, not about a product.

        Scoring it against the leveraged product's P&L would grade the
        product selection too, and that is a separate question.
        """
        report = attribution_report(self._rows(make_entry, make_label))  # type: ignore[arg-type]

        assert report.entries_attributable == 25
        ids = {s.model_id for s in report.model_scores}
        assert ids == {"tsmom_forecast_v1", "logistic_direction_v1"}
        for score in report.model_scores:
            assert score.n == 25
            assert score.interpretable
            assert not math.isnan(score.mse)

    def test_the_closer_forecast_scores_the_lower_error(
        self, make_entry: Callable[..., LedgerEntry], make_label: Callable[..., LedgerLabel]
    ) -> None:
        """A sanity floor: the measurement has to be able to rank."""
        report = attribution_report(self._rows(make_entry, make_label))  # type: ignore[arg-type]
        by_id = {s.model_id: s for s in report.model_scores}

        # realised underlying_pnl averages ~0.012; tsmom predicts 0.010,
        # logistic predicts 0.004.
        assert by_id["tsmom_forecast_v1"].mse < by_id["logistic_direction_v1"].mse

    def test_the_mean_weight_is_reported_but_does_not_enter_the_score(
        self, make_entry: Callable[..., LedgerEntry], make_label: Callable[..., LedgerLabel]
    ) -> None:
        report = attribution_report(self._rows(make_entry, make_label))  # type: ignore[arg-type]
        by_id = {s.model_id: s for s in report.model_scores}

        assert by_id["tsmom_forecast_v1"].mean_weight == pytest.approx(0.6)
        assert by_id["logistic_direction_v1"].mean_weight == pytest.approx(0.4)

    def test_an_unresolved_entry_still_counts_in_the_denominator(
        self, make_entry: Callable[..., LedgerEntry], make_label: Callable[..., LedgerLabel]
    ) -> None:
        """A report must not look complete by excluding what it could not resolve."""
        good = self._rows(make_entry, make_label, n=10)
        bad = self._rows(make_entry, make_label, n=15, forecasts=[])
        report = attribution_report([*good, *bad])  # type: ignore[arg-type]

        assert report.entries_considered == 25
        assert report.entries_attributable == 10
        assert report.attributable_share == pytest.approx(0.4)
        assert "10 of 25" in report.summary()

    def test_the_ensemble_is_scored_too(
        self, make_entry: Callable[..., LedgerEntry], make_label: Callable[..., LedgerLabel]
    ) -> None:
        report = attribution_report(self._rows(make_entry, make_label))  # type: ignore[arg-type]
        assert report.ensemble_score is not None
        assert report.ensemble_score.model_id == ENSEMBLE_MODEL_ID
        assert report.ensemble_score.n == 25

    def test_an_empty_batch_reports_nothing_rather_than_zeroes(self) -> None:
        report = attribution_report([])
        assert report.entries_considered == 0
        assert math.isnan(report.attributable_share)
        assert "no labelled entry" in report.summary()

    def test_a_thin_score_is_flagged_not_interpretable(
        self, make_entry: Callable[..., LedgerEntry], make_label: Callable[..., LedgerLabel]
    ) -> None:
        report = attribution_report(self._rows(make_entry, make_label, n=5))  # type: ignore[arg-type]
        assert all(not s.interpretable for s in report.model_scores)


class TestBeatsNull:
    def test_an_uninterpretable_score_yields_unknown_not_false(
        self, make_entry: Callable[..., LedgerEntry], make_label: Callable[..., LedgerLabel]
    ) -> None:
        """`False` would claim a comparison was made. `None` says it was not."""
        from turboedge.learning.attribution import ModelScore

        thin = ModelScore(model_id="m", n=3, mse=0.001, directional_accuracy=0.6, mean_weight=0.5)
        assert beats_null(thin, null_mse=0.002) is None

    def test_a_real_comparison_answers(self) -> None:
        from turboedge.learning.attribution import ModelScore

        score = ModelScore(model_id="m", n=50, mse=0.001, directional_accuracy=0.6, mean_weight=0.5)
        assert beats_null(score, null_mse=0.002) is True
        assert beats_null(score, null_mse=0.0005) is False

    def test_a_missing_null_yields_unknown(self) -> None:
        from turboedge.learning.attribution import ModelScore

        score = ModelScore(model_id="m", n=50, mse=0.001, directional_accuracy=0.6, mean_weight=0.5)
        assert beats_null(score, null_mse=float("nan")) is None


def test_nothing_updates_a_weight_here() -> None:
    """Measuring and weighting stay separate.

    Turning these scores into weights is a governance step with its own
    statistical requirements. If this module ever writes a weight, that
    decision has been made in the wrong place.
    """
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[2]
        / "src"
        / "turboedge"
        / "learning"
        / "attribution.py"
    ).read_text()

    for forbidden in ("update_weights", "upsert_model_registry_entry", "ModelRegistry"):
        assert forbidden not in source, forbidden
