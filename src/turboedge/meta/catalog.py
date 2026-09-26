"""The fixed research catalog (Phase 2, §11 "Research ideas -- only from a catalog").

In version 1 the controller must never invent a research question. It may
only prioritise among a fixed, human-authored list. This module *is* that
list: fourteen static entries, each a falsifiable hypothesis with the ten
ranking inputs required by `research_opportunity.ResearchOpportunity`,
supplied here as `Estimate`s.

Every ranking input in this module is `DECLARED` or `UNKNOWN`, never
`MEASURED` -- nothing in a static catalog module measures anything. Three
inputs (`current_uncertainty`, `data_availability`,
`overlap_with_existing_research`) are left `UNKNOWN` on every entry on
purpose: they are genuinely measurable from repository state (the
uncertainty axes in `meta/uncertainty.py`, adapter/source health, and
`state/registry/failed_hypotheses.json` / `successful_research_patterns`
respectively), but nothing in *this* module reads the database, the
filesystem or the network, so claiming a value here would be exactly the
laundered guess the module docstring in `research_opportunity.py` warns
against. A later stage enriches these three from real repository state.

The other seven inputs are `DECLARED`, each with a `note` that is meant to
be argued with -- particularly `estimated_sample_size`, which is stated as
an honest count of *independent* observations, not raw rows. Two of the
fourteen entries (Cboe volatility state, CFTC positioning) were already
measured in Research Wave 2 (September 2026) and found not to clear the
promotion ladder; their `DECLARED` inputs here reflect that prior result
(low remaining information gain, low remaining economic value) rather than
treating the question as freshly open. Those measured outcomes themselves
are *not* stored on the catalog entries -- every entry here stays
`ResearchStatus.PROPOSED`, because the schema only allows a non-`PROPOSED`
status when `approved_by` is set, and the catalog is not the place a human
approval is recorded. Instead `W12_MEASURED_OUTCOMES` carries that fact
separately, keyed by `hypothesis_id`, for another stage to apply.

Underlyings and horizons are taken from what this repository actually
runs, not guessed: `configs/universe.yaml`'s `enabled: true` underlyings
are DAX, NDX, EURUSD and XAU (the canonical id used throughout
`universe/underlying_map.py`, `configs/universe.yaml` and
`tests/test_config.py` -- "XAUUSD" is only a display alias resolved by
`underlying_map.py`, never the id stored anywhere), and horizons are the
`"3d"/"5d"/"7d"/"10d"/"14d"` strings used in
`state/registry/failed_hypotheses.json`, matching
`models/forecast.py::HORIZONS = (3, 5, 7, 10, 14)`.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from turboedge.meta.research_opportunity import (
    Estimate,
    InformationFamily,
    ResearchOpportunity,
    ResearchStatus,
)

#: Enabled underlyings (`configs/universe.yaml`), in the id form used
#: everywhere else in this repository.
_ALL_UNDERLYINGS = ("DAX", "NDX", "EURUSD", "XAU")

#: Horizon ladder (`models/forecast.py::HORIZONS`), in the string form used
#: by `state/registry/failed_hypotheses.json`.
_ALL_HORIZONS = ("3d", "5d", "7d", "10d", "14d")

#: Boilerplate note for the three inputs this module can never measure.
#: Repeated verbatim (rather than varied for cosmetic reasons) because it
#: names a real, single computation path for every entry alike -- variety
#: here would suggest a different measurement exists per entry, which would
#: not be honest.
_CURRENT_UNCERTAINTY_NOTE = (
    "Measurable from the meta layer's own uncertainty axes (meta/uncertainty.py) once this "
    "question has a run attached to score; not computed by this static catalog module."
)
_DATA_AVAILABILITY_NOTE = (
    "Measurable via the relevant source adapter's health check (turboedge sources health) or, "
    "for a source with no adapter yet, its absence; not computed by this static catalog module."
)
_OVERLAP_NOTE = (
    "Measurable against state/registry/failed_hypotheses.json and the "
    "successful_research_patterns table by comparing information_family and feature construction; "
    "not computed by this static catalog module."
)


def _unmeasured_triplet() -> tuple[Estimate, Estimate, Estimate]:
    """The three inputs every catalog entry must leave `UNKNOWN`, in field order."""
    return (
        Estimate.unknown(_CURRENT_UNCERTAINTY_NOTE),
        Estimate.unknown(_DATA_AVAILABILITY_NOTE),
        Estimate.unknown(_OVERLAP_NOTE),
    )


def _opportunity(
    hypothesis_id: str,
    description: str,
    information_family: InformationFamily,
    affected_underlyings: tuple[str, ...],
    affected_horizons: tuple[str, ...],
    *,
    expected_information_gain: Estimate,
    expected_economic_value: Estimate,
    probability_of_resolving_uncertainty: Estimate,
    implementation_cost: Estimate,
    implementation_complexity: Estimate,
    estimated_sample_size: Estimate,
    leakage_risk: Estimate,
) -> ResearchOpportunity:
    current_uncertainty, data_availability, overlap_with_existing_research = _unmeasured_triplet()
    return ResearchOpportunity(
        hypothesis_id=hypothesis_id,
        description=description,
        information_family=information_family,
        affected_underlyings=list(affected_underlyings),
        affected_horizons=list(affected_horizons),
        expected_information_gain=expected_information_gain,
        expected_economic_value=expected_economic_value,
        probability_of_resolving_uncertainty=probability_of_resolving_uncertainty,
        implementation_cost=implementation_cost,
        implementation_complexity=implementation_complexity,
        estimated_sample_size=estimated_sample_size,
        current_uncertainty=current_uncertainty,
        data_availability=data_availability,
        leakage_risk=leakage_risk,
        overlap_with_existing_research=overlap_with_existing_research,
        status=ResearchStatus.PROPOSED,
    )


_CBOE_VOL_STATE = _opportunity(
    "RO-CBOE-VOL-STATE",
    "Cboe volatility-state features (VIX level/term-structure, VVIX, VIX9D, VIX3M, OVX, GVZ) "
    "carry incremental out-of-sample information about DAX/NDX/EURUSD/XAU returns, beyond what "
    "the underlying's own trend and realised-volatility history already capture.",
    InformationFamily.VOLATILITY_SURFACE,
    _ALL_UNDERLYINGS,
    _ALL_HORIZONS,
    expected_information_gain=Estimate.declared(
        0.05,
        "W12-A (TR-2026Q3-abd750, 2026-09-25) already measured the linear/ridge construction "
        "across all 20 cells and found it worse on CRPS and Brier everywhere; the remaining "
        "upside is confined to a materially different construction (nonlinear interactions, "
        "regime-conditional use), which is why this is scored low rather than zero.",
    ),
    expected_economic_value=Estimate.declared(
        0.03,
        "Best measured signal-direction edge was CBOE +39.87bp (NDX) against BASE +59.18bp, and "
        "0/20 cells cleared even the low end of the 50-150bp/7d Turbo cost band; a new "
        "construction would need to overturn a decisive negative, so this is scored near the "
        "floor, not zero (W12-A's own drift-versus-skill caveat leaves genuine tail upside).",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.15,
        "The linear-construction question is already resolved (NO). What remains open is a "
        "narrower question -- does a regime-conditional or nonlinear use of the same series do "
        "better -- which a small ablation could resolve cheaply, but is more likely to confirm "
        "the existing result than reverse it.",
    ),
    implementation_cost=Estimate.declared(
        3.0,
        "Adapter, schema, availability guard and feature builder already exist and are "
        "unmodified (kept as measurement infrastructure per W12-A); a follow-up only needs a new "
        "model class reusing the existing PurgedWalkForwardSplit harness across the same 20 "
        "cells -- roughly one engineer-week based on how the original W12-A run went.",
    ),
    implementation_complexity=Estimate.declared(
        0.2,
        "All data plumbing (adapter, schema, availability model, contract tests) is already "
        "built and tested; only a new model variant is required, which keeps complexity low.",
    ),
    estimated_sample_size=Estimate.declared(
        1700,
        "W12-A reports ~3,000 OOS rows per cell, 20 cells, 60,017 paired rows total -- but "
        "5-14-day-ahead labels overlap by construction, so independence is bounded by calendar "
        "span divided by horizon, not row count. At the 7d horizon that is ~3,000/7 ~= 430 "
        "independent windows per underlying; across 4 underlyings, ~1,700. W12-A's own first run "
        "lost a factor of ~3 to exactly this kind of overlap/window bug, which is why this "
        "estimate applies the same discount rather than quoting a raw row count.",
    ),
    leakage_risk=Estimate.declared(
        0.1,
        "available_at is already enforced as observation-day+1 00:00 UTC (deliberately "
        "conservative against the ~20:15 UTC real publication) and tested against adversarial "
        "leak fixtures in W12-A; a follow-up reusing the same adapter inherits that guard, so "
        "risk is low, not zero, since an independently built nonlinear feature could reintroduce "
        "it.",
    ),
)

_EUREX_POSITIONING = _opportunity(
    "RO-EUREX-POSITIONING",
    "Eurex derivatives positioning on the DAX complex (e.g. ODAX/FDAX open-interest skew, "
    "put/call ratio by strike bucket) forecasts subsequent DAX directional moves and volatility "
    "regime shifts beyond own-history features.",
    InformationFamily.POSITIONING,
    ("DAX",),
    _ALL_HORIZONS,
    expected_information_gain=Estimate.declared(
        0.35,
        "Eurex's own DAX/Bund positioning is the closest analogue to the CFTC financial-futures "
        "data that W12-D found degraded the forecast; unlike CFTC's US-listed proxies, an "
        "Eurex-native series would sit directly on the traded underlying, so a genuine edge "
        "remains plausible even though the closest comparable feature class failed.",
    ),
    expected_economic_value=Estimate.declared(
        0.3,
        "No positioning family has cleared the promotion ladder in this repository yet (CFTC: "
        "DO NOT PROMOTE, W12-D); scored moderate rather than low because Eurex's book covers the "
        "actual traded contract, which CFTC's CME-listed proxies do not.",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.2,
        "The one Eurex attempt to date (Research Wave 2, September 2026) was abandoned before "
        "producing a result -- effective sample was ~4% of nominal (Neff ~357) -- so this "
        "question is still open, but a repeat attempt faces the same availability constraint "
        "unless a different source or a coarser aggregation window is used.",
    ),
    implementation_cost=Estimate.declared(
        6.0,
        "The abandoned attempt already spent effort discovering the effective-sample problem; a "
        "viable repeat needs a new data source or aggregation plus building and testing a "
        "publication-lag model from scratch, roughly double the ~3 engineer-days the CBOE "
        "follow-up above costs by reusing existing infrastructure.",
    ),
    implementation_complexity=Estimate.declared(
        0.55,
        "Eurex does not publish a free daily positioning series comparable to CFTC's COT "
        "reports; the abandoned attempt approximated one from options open-interest snapshots, "
        "which is why effective sample collapsed. A viable version likely needs a paid Eurex "
        "data product or a coarser weekly aggregation -- meaningfully harder than the CBOE/CFTC "
        "adapters already built.",
    ),
    estimated_sample_size=Estimate.declared(
        357,
        "This is the measured effective sample (Neff) from the abandoned Research Wave 2 "
        "attempt, not a fresh guess -- it is the number that caused the test to be abandoned "
        "rather than run to a false conclusion, and it is the honest starting point for whether "
        "a retry is worth doing at all.",
    ),
    leakage_risk=Estimate.declared(
        0.35,
        "Eurex settlement/open-interest data for day t is typically reported after day t's "
        "close; without a verified publication-lag contract (the abandoned attempt did not get "
        "far enough to build one), the safe assumption is same-day risk until an availability "
        "model is derived and tested the way it was for Cboe (t+1) and CFTC (as-of+6d).",
    ),
)

_EUWAX_SENTIMENT = _opportunity(
    "RO-EUWAX-SENTIMENT",
    "Euwax retail order-flow / sentiment indicators (e.g. put/call turnover imbalance in "
    "retail-issued warrants and turbos) forecast short-horizon reversals or continuation in DAX, "
    "beyond own-history features.",
    InformationFamily.SENTIMENT,
    ("DAX",),
    ("3d", "5d", "7d"),
    expected_information_gain=Estimate.declared(
        0.25,
        "Euwax sentiment would be a genuinely novel information source -- retail flow, distinct "
        "from CFTC's institutional futures positioning and Cboe's US options market; scored "
        "moderate because retail-sentiment/contrarian effects are a well-documented category "
        "elsewhere, but nothing here has tested this specific source.",
    ),
    expected_economic_value=Estimate.declared(
        0.15,
        "No adapter has ever pulled this source, so there is no measured edge to anchor on, and "
        "retail-sentiment effects reported elsewhere are typically small and short-lived, which "
        "caps plausible economic value even if the sign is right.",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.1,
        "The source (boerse-stuttgart-group.de / Euwax) returns HTTP 403 including for "
        "robots.txt -- there is no legitimate crawl path today, so this question cannot be "
        "resolved via this source at all; scored low rather than zero only because a different, "
        "licensed data vendor could still answer it.",
    ),
    implementation_cost=Estimate.declared(
        8.0,
        "Highest cost in the catalog: needs sourcing a licensed or alternative vendor "
        "(commercial negotiation, or an entirely different scrape target) and then building "
        "adapter, schema, availability model and contract tests from nothing, rather than "
        "reusing existing infrastructure the way the CBOE/CFTC follow-ups can.",
    ),
    implementation_complexity=Estimate.declared(
        0.7,
        "Blocked at the access layer, not the parsing layer: the project's data-sources rule "
        "forbids bypassing bot protection, so this is a 'find or license an alternative source' "
        "problem, not a 'write an adapter' problem -- qualitatively harder than CBOE/CFTC, which "
        "only needed correct parsing of an already-open endpoint.",
    ),
    estimated_sample_size=Estimate.declared(
        0,
        "No accessible source exists today (HTTP 403 including robots.txt), so the honest "
        "sample size for this hypothesis, as currently specified, is zero independent "
        "observations -- this is a blocked question, not merely a small-sample one.",
    ),
    leakage_risk=Estimate.declared(
        0.3,
        "Unknown until a source exists, but retail sentiment/flow data is typically reported "
        "same-day or intraday by vendors, so a naive integration risks same-day leakage until a "
        "publication-lag contract is derived and tested, as was done for Cboe (t+1 00:00 UTC) "
        "and CFTC (as-of+6 days).",
    ),
)

_CFTC_POSITIONING = _opportunity(
    "RO-CFTC-POSITIONING",
    "CFTC Commitments-of-Traders leveraged-money positioning (normalised by open interest) in "
    "S&P 500 / Nasdaq-100 / EUR FX futures forecasts subsequent DAX/NDX/EURUSD/XAU returns beyond "
    "own-history features.",
    InformationFamily.POSITIONING,
    _ALL_UNDERLYINGS,
    _ALL_HORIZONS,
    expected_information_gain=Estimate.declared(
        0.03,
        "W12-D (TR-2026Q3-31e266, 2026-09-25) measured this exact construction (18 "
        "leveraged-money features, 20/20 cells) and found it worse on every metric, 20/20 "
        "BH-significant against; the remaining upside is confined to a materially different "
        "feature set (e.g. dealer/asset-manager splits alone, or restricting each series to the "
        "underlying it is actually denominated in), scored near-floor rather than zero.",
    ),
    expected_economic_value=Estimate.declared(
        0.02,
        "Median |t| against the null was 9.91 (versus CBOE's 1.57) and 90% coverage degraded "
        "from 0.888 to 0.838 -- the most decisive negative result measured in this repository to "
        "date; further economic value from this exact construction is implausible.",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.05,
        "Close to a closed question for the tested construction; a re-run would mostly confirm "
        "the existing result rather than resolve new uncertainty, which is why this is scored "
        "near the floor rather than at zero -- W12-D's own write-up notes this is a statement "
        "about this feature construction, not proof that positioning contains no information at "
        "all.",
    ),
    implementation_cost=Estimate.declared(
        1.5,
        "Feature selection and re-run only -- adapter, schema and availability model are already "
        "built and tested; the original run's own two iterations (54-feature, then 18-feature) "
        "show a further ablation is a small, fast follow-up, not a new data-engineering project.",
    ),
    implementation_complexity=Estimate.declared(
        0.15,
        "Adapter, contract tests (covering the two data traps found: unstable contract names "
        "across the archive, a mislabeled date column) and the as-of+6-day availability model "
        "already exist and are reusable; the only new work is choosing a narrower feature subset.",
    ),
    estimated_sample_size=Estimate.declared(
        380,
        "W12-D's own write-up states the 18-feature construction ran against '~380 independent "
        "weeks' -- 844 weeks of raw COT reports exist (2010-07-20 to 2026-09-15), but weekly "
        "positioning forward-filled onto a daily axis is still one independent observation per "
        "week, not per forward-filled day; this is that honest count, not the 10,128-row or "
        "60,017-paired figures quoted for the raw archive.",
    ),
    leakage_risk=Estimate.declared(
        0.1,
        "Publication-lag model (as-of Tuesday + 6 days, accounting for the Friday 15:30 ET "
        "release and holiday shifts) is already built and was the subject of dedicated "
        "availability testing; low residual risk, not zero, since a narrower feature subset "
        "re-derived independently could reintroduce an error.",
    ),
)

_RATES_CREDIT = _opportunity(
    "RO-RATES-CREDIT",
    "Rates and credit-spread indicators (e.g. 2s10s slope, Bund-Treasury spread, a broad credit "
    "index level) forecast regime shifts in DAX/NDX/EURUSD/XAU returns beyond own-history "
    "features -- a structurally different channel from both the vol-state (Cboe) and positioning "
    "(CFTC) families already measured and rejected.",
    InformationFamily.RATES_CREDIT,
    _ALL_UNDERLYINGS,
    _ALL_HORIZONS,
    expected_information_gain=Estimate.declared(
        0.4,
        "Rates/credit conditions are a structurally different information channel from vol-state "
        "(measured negative) and positioning (measured negative); macro regime is a commonly "
        "cited driver of cross-asset returns in the literature, and nothing in this repository "
        "has tested it yet, so this is scored higher than the two already-negative families.",
    ),
    expected_economic_value=Estimate.declared(
        0.3,
        "Plausible but unproven; two adjacent families (vol-state, positioning) both failed to "
        "clear the cost band, which is a relevant prior against any macro-conditioning signal "
        "surviving realistic Turbo costs, without this specifically having been tested.",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.55,
        "Public rates/credit series (FRED, ECB SDW/Bundesbank, a published credit-spread index) "
        "are well-documented and stable in construction, unlike the Eurex/Euwax cases -- a clean "
        "pre-registered test is very likely to produce a clear, reportable answer either way.",
    ),
    implementation_cost=Estimate.declared(
        5.0,
        "Three to four independent source adapters (US rates, EUR rates, a credit-spread index) "
        "each need their own contract tests and availability model -- roughly the combined cost "
        "of the CBOE and CFTC adapters together, based on how long each of those took.",
    ),
    implementation_complexity=Estimate.declared(
        0.35,
        "Multiple public series need aggregating, each behind its own adapter and availability "
        "model; more work than reusing an existing adapter, but every source is free, "
        "documented, and similar in shape to the CBOE adapter already built.",
    ),
    estimated_sample_size=Estimate.declared(
        2000,
        "Daily rates/credit series over the same ~15-16 year window used for CBOE/CFTC give "
        "roughly 4,000 raw trading days; with purged/embargoed CV at a 7d-median horizon, "
        "independent windows are bounded near total_days/horizon ~ 570 per underlying, "
        "aggregated across 4 underlyings ~ 2,000 -- the same overlap discount applied to the "
        "CBOE estimate above, not the raw row count.",
    ),
    leakage_risk=Estimate.declared(
        0.25,
        "Most rates/credit series settle same-day and publish with a short lag (FRED is often "
        "T+1); risk is moderate rather than low until an explicit availability model is derived "
        "and tested per series, the way it was for Cboe (t+1) and CFTC (as-of+6d).",
    ),
)

_MARKET_BREADTH = _opportunity(
    "RO-MARKET-BREADTH",
    "Market-breadth indicators (advance/decline line, share of index constituents above their "
    "own moving average) for DAX and NDX forecast subsequent index-level returns and volatility "
    "regime beyond the index's own trend/vol history.",
    InformationFamily.BREADTH_DISPERSION,
    ("DAX", "NDX"),
    ("5d", "7d", "10d", "14d"),
    expected_information_gain=Estimate.declared(
        0.3,
        "Breadth divergence (index rising while few constituents participate) is a "
        "long-documented equity phenomenon and genuinely untested here, but published effect "
        "sizes are typically small and inconsistent out of sample, so this is scored moderate "
        "rather than high.",
    ),
    expected_economic_value=Estimate.declared(
        0.2,
        "Breadth effects reported in the literature are usually below or near typical cash-equity "
        "transaction-cost bands; a Turbo's 50-150bp/7d cost band (the same band CBOE was measured "
        "against) is a high bar this family has not been shown to clear anywhere.",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.5,
        "Constituent-level price history for DAX/NDX is standard, stable data (unlike "
        "Eurex/Euwax); a pre-registered test is likely to produce a clean answer.",
    ),
    implementation_cost=Estimate.declared(
        4.0,
        "Point-in-time constituent lists plus roughly 140 individual price series to fetch and "
        "maintain is closer to the CFTC adapter's data-wrangling effort than to a single-series "
        "adapter like CBOE's.",
    ),
    implementation_complexity=Estimate.declared(
        0.5,
        "Requires per-constituent daily price history for two indices (roughly 40 DAX, 100 NDX "
        "names) rather than a single series, plus point-in-time index-membership handling to "
        "avoid survivorship bias (this project never retroactively historises current "
        "constituents) -- meaningfully more data plumbing than a single macro series.",
    ),
    estimated_sample_size=Estimate.declared(
        800,
        "~15 years of daily data at a 10d-median horizon gives roughly total_days/horizon ~ 380 "
        "independent windows per index; two indices (DAX, NDX) give ~ 760-800 -- not the raw "
        "~4,000 daily rows per index.",
    ),
    leakage_risk=Estimate.declared(
        0.15,
        "Constituent closes are available same-day after market close, the same timing profile "
        "the underlying index itself already uses elsewhere in this codebase; low risk given "
        "existing end-of-day conventions, not zero because point-in-time index membership needs "
        "its own care.",
    ),
)

_DISPERSION = _opportunity(
    "RO-DISPERSION",
    "Cross-sectional return dispersion among DAX/NDX constituents (the spread of single-name "
    "returns around the index return) forecasts subsequent index-level volatility regime and "
    "path/knock-out risk beyond realised index volatility alone.",
    InformationFamily.BREADTH_DISPERSION,
    ("DAX", "NDX"),
    _ALL_HORIZONS,
    expected_information_gain=Estimate.declared(
        0.3,
        "Dispersion is mechanically linked to index realised volatility (index vol is a "
        "variance-weighted average of single-name vol and average correlation), so part of any "
        "signal here may be redundant with volatility features already in BASE; scored moderate "
        "to reflect that overlap risk up front rather than after the fact.",
    ),
    expected_economic_value=Estimate.declared(
        0.2,
        "Most directly useful for path/knock-out-probability modelling (this project treats "
        "knock-out as strictly path-dependent) rather than directional return prediction; scored "
        "on the assumption it feeds path-risk sizing rather than a directional edge, which is a "
        "narrower and more modest win even if it works.",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.45,
        "Shares the constituent-data stability of market breadth; a clean test is achievable, "
        "but the redundancy with existing realised-volatility features noted above makes a "
        "positive, non-redundant result less likely than the raw hypothesis suggests.",
    ),
    implementation_cost=Estimate.declared(
        3.0,
        "Lower than market breadth's cost because it can reuse the same constituent price feed "
        "if built second; scored as the marginal cost assuming that data plumbing already "
        "exists, not built from zero.",
    ),
    implementation_complexity=Estimate.declared(
        0.5,
        "Shares the constituent-data infrastructure need with market breadth (point-in-time "
        "membership, roughly 140 individual series); if built alongside it the marginal cost is "
        "much lower, but standalone it carries the same complexity.",
    ),
    estimated_sample_size=Estimate.declared(
        800,
        "Same independent-window arithmetic as market breadth: ~15 years, 10d-median horizon, "
        "two indices -- roughly 800 independent observations, not the raw daily row count.",
    ),
    leakage_risk=Estimate.declared(
        0.15,
        "Same same-day-close timing profile as market breadth; low risk given existing "
        "end-of-day conventions already used for index-level bars.",
    ),
)

_EVENT_RISK = _opportunity(
    "RO-EVENT-RISK",
    "Scheduled macro/earnings-calendar event risk (e.g. ECB/Fed decisions, CPI releases, "
    "DAX-constituent earnings dates) changes the shape of the realised return distribution "
    "(fatter tails, gap risk) around event dates enough to warrant event-conditional path/KO "
    "modelling, beyond the unconditional gap-premium model already in production.",
    InformationFamily.EVENT_RISK,
    _ALL_UNDERLYINGS,
    ("3d", "5d", "7d"),
    expected_information_gain=Estimate.declared(
        0.35,
        "The gap-premium model already estimates unconditional overnight/weekend gap "
        "distributions; this hypothesis is specifically whether conditioning that distribution "
        "on a known scheduled-event calendar sharpens it further, which is untested and "
        "plausible given how much single-day event moves dominate tail risk in these "
        "underlyings.",
    ),
    expected_economic_value=Estimate.declared(
        0.25,
        "Would primarily improve path/knock-out-probability calibration around events (a "
        "risk-modelling win) rather than create a new directional edge; scored moderate, "
        "consistent with how the dispersion and MAE/MFE entries are also scored as path-risk "
        "rather than directional contributions.",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.5,
        "Public economic calendars (ECB/Fed/statistics-office release schedules) are stable, "
        "well-documented, point-in-time data with no ambiguity about publication timing -- a "
        "clean test is achievable.",
    ),
    implementation_cost=Estimate.declared(
        3.5,
        "Building and maintaining a point-in-time event calendar (avoiding the retroactive-"
        "historisation trap for e.g. which DAX constituents existed on a given earnings date) "
        "plus wiring it into the existing path simulator.",
    ),
    implementation_complexity=Estimate.declared(
        0.4,
        "Needs an events/calendar table (event date, type, underlying-relevance) joined against "
        "the existing path-simulation code; more schema work than a single time series, but no "
        "novel data-access problem like Eurex/Euwax.",
    ),
    estimated_sample_size=Estimate.declared(
        250,
        "Scheduled macro events (ECB + Fed decisions) run roughly 8-16 per year each; over a "
        "~15-year window that is on the order of a few hundred independent event windows total "
        "across underlyings -- a genuinely small-sample question regardless of how many daily "
        "price rows surround each event.",
    ),
    leakage_risk=Estimate.declared(
        0.05,
        "Scheduled events (rate decisions, CPI dates) are known well in advance by construction "
        "-- one of the few features here with essentially no publication-lag ambiguity, since "
        "the calendar itself, not the outcome, is the feature.",
    ),
)

_MAE_PREDICTION = _opportunity(
    "RO-MAE-PREDICTION",
    "A model conditioning on entry-time features (implied vol regime, trend state, "
    "time-to-horizon) can forecast a product's Maximum Adverse Excursion -- the worst intra-trade "
    "drawdown before exit or knock-out -- well enough to improve position sizing and stop/exit "
    "rules beyond the unconditional MAE distribution already computed by the labeler.",
    InformationFamily.PATH_STATISTICS,
    _ALL_UNDERLYINGS,
    _ALL_HORIZONS,
    expected_information_gain=Estimate.declared(
        0.35,
        "The labeler already computes realised MAE per matured ledger entry (an unconditional "
        "distribution); this hypothesis is whether entry-time conditioning features add "
        "predictive power over that unconditional distribution -- untested, and plausible since "
        "MAE is mechanically tied to the same vol/trend state the forecast models already use.",
    ),
    expected_economic_value=Estimate.declared(
        0.3,
        "A better MAE forecast would improve position sizing and stop placement directly, a real "
        "economic lever even without a new directional edge -- but it depends on having enough "
        "matured forward-ledger entries per cell to train and validate on, which this repository "
        "is only beginning to accumulate.",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.4,
        "The forward ledger began labeling real matured entries on 2026-09-17 and reached 2,854 "
        "labeled entries by 2026-09-18, growing daily since -- there is now real matured-path "
        "data to test against, unlike Eurex/Euwax, though still young relative to the ~15-year "
        "windows the macro families use.",
    ),
    implementation_cost=Estimate.declared(
        2.5,
        "Reuses existing labeled MAE data end to end; cost is mostly the modelling and "
        "validation harness, not data engineering.",
    ),
    implementation_complexity=Estimate.declared(
        0.3,
        "MAE is already computed and stored per matured entry; this is a modelling exercise "
        "(regress/classify against entry-time features) on data that already exists, not a new "
        "data-source integration.",
    ),
    estimated_sample_size=Estimate.declared(
        220,
        "Labeled-entry counts (2,854 by 2026-09-18) count trades, not independent paths: within "
        "one evening's labeling run, many entries share the same underlying/day/horizon and "
        "therefore the same realised price path, so they are not independent draws for a MAE "
        "model. The independence unit is closer to (trading day x underlying x horizon); at "
        "roughly 10 trading days of real ledger history x 4 underlyings x 5 horizons, that is on "
        "the order of 200 independent path-outcomes today, growing by roughly 20 per trading day "
        "as the ledger ages.",
    ),
    leakage_risk=Estimate.declared(
        0.2,
        "Entry-time features must be strictly available at or before the entry decision time; "
        "moderate risk because MAE-conditioning features could tempt using same-trade "
        "information (e.g. the realised path so far) if not disciplined, which would leak by "
        "construction.",
    ),
)

_MFE_PREDICTION = _opportunity(
    "RO-MFE-PREDICTION",
    "A model conditioning on entry-time features can forecast a product's Maximum Favorable "
    "Excursion -- the best intra-trade unrealised gain before exit or knock-out -- well enough to "
    "improve take-profit/exit-timing rules beyond the unconditional MFE distribution already "
    "computed by the labeler.",
    InformationFamily.PATH_STATISTICS,
    _ALL_UNDERLYINGS,
    _ALL_HORIZONS,
    expected_information_gain=Estimate.declared(
        0.3,
        "Symmetric hypothesis to MAE prediction, using the same labeled MFE distribution; scored "
        "slightly lower because take-profit timing has a smaller, more discretionary role in "
        "this system's rules than stop/exit sizing does.",
    ),
    expected_economic_value=Estimate.declared(
        0.25,
        "A better MFE forecast mainly sharpens exit timing rather than entry selection; a real "
        "but secondary lever compared to position sizing (RO-MAE-PREDICTION), hence the lower "
        "score.",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.4,
        "Identical data-maturity situation to MAE prediction: the same 2,854-labeled-entry "
        "ledger (as of 2026-09-18) carries MFE alongside MAE for every matured entry, so this "
        "question is equally testable and equally young.",
    ),
    implementation_cost=Estimate.declared(
        2.0,
        "Slightly cheaper than MAE prediction if built second: the labeled-data extraction and "
        "validation harness can be shared, leaving only a second target variable to model.",
    ),
    implementation_complexity=Estimate.declared(
        0.3,
        "Same data source and modelling approach as MAE prediction (already-labeled per-entry "
        "distribution, entry-time conditioning features); complexity is effectively identical.",
    ),
    estimated_sample_size=Estimate.declared(
        220,
        "Same independent-path accounting as MAE prediction: labeled-entry counts overstate "
        "independence because many entries share a realised path within a trading day; on the "
        "order of 200 independent path-outcomes exist today, from the same ledger.",
    ),
    leakage_risk=Estimate.declared(
        0.2,
        "Same entry-time-only conditioning requirement and same same-trade-information risk as "
        "MAE prediction.",
    ),
)

_CROSS_ASSET_RESIDUAL_MOMENTUM = _opportunity(
    "RO-CROSS-ASSET-RESIDUAL-MOMENTUM",
    "After regressing out each underlying's own trend/vol state, the residual return of one "
    "underlying (e.g. XAU, EURUSD) forecasts the near-term residual return of another (e.g. DAX, "
    "NDX) -- a genuine cross-asset lead-lag effect distinct from the already-measured, "
    "unconditional cross_asset_leadlag feature (W9-2026Q3-005, dormant).",
    InformationFamily.CROSS_ASSET,
    _ALL_UNDERLYINGS,
    ("3d", "5d", "7d"),
    expected_information_gain=Estimate.declared(
        0.2,
        "SIGNAL_REGISTRY.md records cross_asset_leadlag (W9-2026Q3-005) as already tested and "
        "dormant (0/10 cells significant, DAX/ESTX50 only, +0.84bp best edge); this hypothesis "
        "is the residual-momentum variant (own-trend regressed out first) rather than the raw "
        "lead-lag already tried, so it is not fully redundant, but the prior negative result for "
        "the simpler version is a real headwind, hence scored low rather than moderate.",
    ),
    expected_economic_value=Estimate.declared(
        0.15,
        "The already-tested unconditional version cleared 0/10 cells at +0.84bp best edge, far "
        "under any realistic cost band; the residual-conditioning refinement would need to "
        "unlock meaningfully more edge than that to matter economically.",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.35,
        "Own-history trend/vol regressions and cross-asset return data are both already built "
        "and stable in this codebase; a clean test is straightforward to run, though as with the "
        "CBOE follow-up it is more likely to confirm the existing negative than reverse it.",
    ),
    implementation_cost=Estimate.declared(
        2.0,
        "Smallest new-feature cost in the catalog: reuses the existing cross-asset and own-trend "
        "feature modules and adds one residualisation step, versus building a new adapter.",
    ),
    implementation_complexity=Estimate.declared(
        0.25,
        "Reuses the existing cross-asset feature module and the own-trend features already "
        "built for the protected baseline; the only new work is the residualisation step "
        "(regress return on own lagged trend/vol, use the residual as the cross-asset feature).",
    ),
    estimated_sample_size=Estimate.declared(
        900,
        "Same daily-overlap discount as the other cross-underlying entries: ~15 years of daily "
        "data, 5d-median horizon, four underlyings pairwise -- roughly total_days/horizon x "
        "underlying-pairs gives on the order of 900 independent windows, not the raw daily row "
        "count.",
    ),
    leakage_risk=Estimate.declared(
        0.15,
        "All four underlyings' bars already carry enforced available_at timestamps used "
        "elsewhere in the pipeline; residualisation adds no new same-day dependency beyond what "
        "the protected baseline already handles, though cross-underlying close-time misalignment "
        "(e.g. DAX closes before NDX opens) needs care to avoid treating a same-session close as "
        "prior information.",
    ),
)

_ISSUER_SPREAD_BEHAVIOUR = _opportunity(
    "RO-ISSUER-SPREAD-BEHAVIOUR",
    "An issuer's bid/ask spread on a given DAX/NDX/EURUSD/XAU turbo widens or narrows "
    "predictably around specific conditions (e.g. realised-volatility spikes, time-to-knockout, "
    "days since issuance) in a way forecastable ahead of the decision time, usable to prefer or "
    "discount candidates before entry, beyond the static spread already used in cost ranking.",
    InformationFamily.MICROSTRUCTURE,
    _ALL_UNDERLYINGS,
    _ALL_HORIZONS,
    expected_information_gain=Estimate.declared(
        0.3,
        "The scan pipeline's cost ranking already uses a static, current spread; nothing here "
        "tests whether spread *changes* are forecastable ahead of entry, which is a different "
        "and untested question from that static comparison.",
    ),
    expected_economic_value=Estimate.declared(
        0.35,
        "Entry is always the ask and spread is already one of the largest structural costs this "
        "system measures; even a modest ability to avoid entries just before a spread widens "
        "would be a direct EV improvement, not merely a distributional one.",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.4,
        "Historical bid/ask snapshots already exist for every scan (subject to the retention/"
        "compaction policy); this is an analysis of data already being collected for other "
        "purposes, so a test is very achievable, though issuer-level heterogeneity (BNP vs Citi "
        "vs gettex) may fragment the sample.",
    ),
    implementation_cost=Estimate.declared(
        4.5,
        "Panel construction across issuers and irregular product lifespans, plus handling the "
        "retention-driven survivorship risk noted under leakage, is more data-engineering-heavy "
        "than a single new time series.",
    ),
    implementation_complexity=Estimate.declared(
        0.35,
        "Requires joining product-snapshot history per ISIN with realised-volatility and "
        "time-to-knockout features; no new adapter, but the panel construction (per-ISIN, "
        "irregular product lifespans) is more involved than a single time series.",
    ),
    estimated_sample_size=Estimate.declared(
        60,
        "Independence here is bounded by distinct (issuer, underlying, volatility-regime) "
        "conditions observed across full product lifespans, not by the tens-of-thousands of "
        "per-scan snapshot rows; with roughly 3 issuers x 4 underlyings x a handful of realistic "
        "regime buckets, the honest count of genuinely distinct behavioural conditions to learn "
        "from is in the tens, not thousands, until many more months of history accumulate.",
    ),
    leakage_risk=Estimate.declared(
        0.2,
        "Must condition only on spread history strictly before the candidate decision time; "
        "moderate risk because product-snapshot retention/thinning (default keep_days=5) could "
        "bias which historical spreads remain available for products never referenced in the "
        "forward ledger -- an availability bias rather than a look-ahead one, but still worth "
        "flagging.",
    ),
)

_FINANCING_BEHAVIOUR = _opportunity(
    "RO-FINANCING-BEHAVIOUR",
    "An issuer's realised financing-level adjustments (the implied overnight financing rate "
    "embedded in a turbo's daily financing-level roll) deviate from a risk-free/reference-rate "
    "benchmark in a predictable, issuer-specific way that can be forecast ahead of the decision "
    "time and used to prefer cheaper-financed products, beyond the static financing-spread "
    "inference already in production.",
    InformationFamily.MICROSTRUCTURE,
    _ALL_UNDERLYINGS,
    _ALL_HORIZONS,
    expected_information_gain=Estimate.declared(
        0.25,
        "The pricing engine already infers a static, per-scan financing spread; this hypothesis "
        "is whether that spread's day-to-day *changes* are predictable (e.g. issuers adjusting "
        "faster or slower than the reference rate around central-bank meetings), a temporal "
        "question the static inference does not address.",
    ),
    expected_economic_value=Estimate.declared(
        0.2,
        "Financing cost compounds daily and is one of two cost components this project keeps "
        "strictly separate from gap premium; a forecastable component would let the system "
        "prefer issuers about to cheapen financing, but the effect is likely small per trade "
        "relative to spread (RO-ISSUER-SPREAD-BEHAVIOUR), hence the lower score.",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.35,
        "Multi-scan-per-day resolution collapses to one financing-level observation per "
        "(ISIN, calendar day) regardless of scan frequency, so the true observation frequency is "
        "daily at best -- a test is achievable but accumulates power more slowly than "
        "intraday-resolution hypotheses.",
    ),
    implementation_cost=Estimate.declared(
        2.5,
        "Reuses the existing financing-level history and inference module entirely; the new "
        "work is a day-over-day change model on an existing derived series.",
    ),
    implementation_complexity=Estimate.declared(
        0.3,
        "Reuses the existing financing-level history and inference module; the new work is a "
        "forecasting model on an existing derived series, not a new data source.",
    ),
    estimated_sample_size=Estimate.declared(
        90,
        "One financing-level observation per (ISIN, calendar day) regardless of scan frequency; "
        "independence is further bounded by issuer x underlying combinations (roughly a dozen) "
        "observed over the trading history accumulated so far (since 2026-09-13), giving on the "
        "order of a few dozen to ~90 genuinely distinct issuer-day observations today, far short "
        "of what a day-level regression would need for real power.",
    ),
    leakage_risk=Estimate.declared(
        0.15,
        "Financing levels are quoted, not settled, so today's level is observable at scan time "
        "with no publication lag by construction; low risk given the existing static inference "
        "already handles this timing correctly.",
    ),
)

_PRODUCT_SELECTION_EDGE = _opportunity(
    "RO-PRODUCT-SELECTION-EDGE",
    "Among products that pass the integrity and cost gates for the same underlying/direction/"
    "horizon, a learned ranking beyond the current cost-rank/median-product baseline selects "
    "products whose subsequent realised net EV is systematically better than the stored "
    "median-product counterfactual, net of winner's-curse shrinkage.",
    InformationFamily.PRODUCT_SELECTION,
    _ALL_UNDERLYINGS,
    _ALL_HORIZONS,
    expected_information_gain=Estimate.declared(
        0.3,
        "The system already stores a median-product counterfactual per decision specifically to "
        "make this comparison possible; whether a learned ranking beats it, after LCB shrinkage, "
        "has not been tested end to end yet.",
    ),
    expected_economic_value=Estimate.declared(
        0.3,
        "Product selection is a second, independent lever from forecast direction -- even with "
        "zero measured directional edge (the current state of this repository), a genuine "
        "product-selection edge would still show up as better realised EV among labeled entries, "
        "so this is not fully blocked by the 'no directional edge' finding.",
    ),
    probability_of_resolving_uncertainty=Estimate.declared(
        0.35,
        "Same maturity constraint as MAE/MFE prediction: needs a meaningful number of matured, "
        "labeled entries across enough distinct products per (underlying, horizon) cell to "
        "compare a ranking against the median-product counterfactual with any power; the ledger "
        "is real but young (2,854 labeled entries by 2026-09-18, growing).",
    ),
    implementation_cost=Estimate.declared(
        5.5,
        "Needs the ranking model, a rigorous purged-CV harness comparable to the W12-A/W12-D "
        "studies, and explicit winner's-curse validation against the stored median-product "
        "counterfactual -- the most methodologically involved entry in the catalog, though it "
        "reuses existing shrinkage/LCB code where possible.",
    ),
    implementation_complexity=Estimate.declared(
        0.45,
        "Needs a ranking model trained against realised outcomes plus explicit winner's-curse "
        "shrinkage validation (already implemented for the current LCB gate) to avoid simply "
        "re-discovering overfit product picks; more involved than a single-feature hypothesis.",
    ),
    estimated_sample_size=Estimate.declared(
        220,
        "Same independent-path accounting as MAE/MFE prediction: labeled-entry counts are "
        "trades, not independent paths, and many share the same (day, underlying, horizon) "
        "realised path; on the order of 200 independent path-outcomes exist today across strata, "
        "growing with the ledger.",
    ),
    leakage_risk=Estimate.declared(
        0.3,
        "Highest structural leakage risk in the catalog: a product-selection model trained on "
        "realised outcomes is exactly the setup where in-sample product picking can masquerade "
        "as skill unless purged/embargoed CV and the median-product counterfactual comparison "
        "are both enforced strictly.",
    ),
)

#: The fixed, human-authored research catalog (§11). Order matches the user's
#: original list; nothing here is generated or inferred.
CATALOG: tuple[ResearchOpportunity, ...] = (
    _CBOE_VOL_STATE,
    _EUREX_POSITIONING,
    _EUWAX_SENTIMENT,
    _CFTC_POSITIONING,
    _RATES_CREDIT,
    _MARKET_BREADTH,
    _DISPERSION,
    _EVENT_RISK,
    _MAE_PREDICTION,
    _MFE_PREDICTION,
    _CROSS_ASSET_RESIDUAL_MOMENTUM,
    _ISSUER_SPREAD_BEHAVIOUR,
    _FINANCING_BEHAVIOUR,
    _PRODUCT_SELECTION_EDGE,
)


class MeasuredOutcome(BaseModel):
    """What actually happened when a catalog question was run, kept apart from the catalog.

    The catalog entry itself must stay `PROPOSED` (`ResearchOpportunity`'s
    own validator requires `approved_by` for any other status, and this
    module records no approval). A `MeasuredOutcome` is the separate,
    approved fact of what a human decided after a trial ran -- meant to be
    applied to the stored opportunity by a later stage (the research queue),
    not folded into this static catalog.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str
    status: ResearchStatus
    approved_by: str
    note: str
    trial_id: str | None = None


#: Facts about four catalog questions already run in Research Wave 2
#: (September 2026), keyed by `hypothesis_id`. Distinguishes three genuinely
#: different epistemic states rather than collapsing them into one:
#:
#: * Cboe / CFTC were run to completion and did not clear the promotion
#:   ladder. `DORMANT` matches this repository's own precedent for exactly
#:   this situation -- `state/registry/failed_hypotheses.json` records six
#:   W9 signal families that cleared measurement but not the ladder, all as
#:   `dormant`, and W12-A's own write-up cites `VixTermStructure`
#:   (`W9-2026Q3-004`) as "recorded dormant" for the same reason. Neither
#:   result is proof the underlying idea is impossible (W12-D's own text:
#:   "not proof that positioning contains no information at all"), so
#:   `REJECTED` -- which reads as a closed, no-revisit verdict -- would
#:   overstate what was actually shown.
#: * Eurex was never measured at all: the attempt was abandoned once
#:   effective sample collapsed to ~4% of nominal, specifically to avoid
#:   manufacturing a false conclusion from an underpowered test. That is a
#:   materially different state from "measured and found wanting", so it
#:   gets `APPROVED` (a trial was authorised and attempted, but produced no
#:   trustworthy measurement to record) rather than `DORMANT`.
#: * Euwax could not even be attempted: the source blocks access (HTTP 403,
#:   including for its own robots.txt), and this project never bypasses
#:   access protection. That is a firm, principled stop on this specific
#:   avenue -- not a small-sample problem like Eurex, and not a measured
#:   negative like Cboe/CFTC -- so it gets `REJECTED`.
W12_MEASURED_OUTCOMES: Mapping[str, MeasuredOutcome] = {
    "RO-CBOE-VOL-STATE": MeasuredOutcome(
        hypothesis_id="RO-CBOE-VOL-STATE",
        status=ResearchStatus.DORMANT,
        approved_by="pcctradinginc@gmail.com (research wave 2, 2026-09)",
        note=(
            "Measured in full (TR-2026Q3-abd750, W12-A, 2026-09-25): 20/20 cells worse on CRPS, "
            "20/20 worse on Brier, 6/20 BH-significant against and 0 in favour, 0/20 cells clear "
            "even the low end of the realistic Turbo cost band. DO NOT PROMOTE. Recorded DORMANT, "
            "not REJECTED: this matches the repository's own precedent for the same situation "
            "(state/registry/failed_hypotheses.json's six W9 dormant entries; W12-A itself cites "
            "the prior VixTermStructure result as 'recorded dormant'), and the negative result is "
            "for this linear/ridge construction specifically, not a proof the family is dead."
        ),
        trial_id="TR-2026Q3-abd750",
    ),
    "RO-EUREX-POSITIONING": MeasuredOutcome(
        hypothesis_id="RO-EUREX-POSITIONING",
        status=ResearchStatus.DORMANT,
        approved_by="pcctradinginc@gmail.com (research wave 2, 2026-09)",
        note=(
            "Not completed: effective sample was ~4% of nominal (Neff ~357), so the test was "
            "abandoned as uninformative rather than run to a false conclusion. Recorded DORMANT "
            "rather than APPROVED: APPROVED keeps an entry in the actionable queue, which would "
            "present this as ready to run when the last attempt showed the available history "
            "cannot support the test. DORMANT is the honest state, and DORMANT -> APPROVED is a "
            "legal transition once a longer or higher-frequency positioning series exists. It is "
            "also distinct from MEASURED: nothing was measured, so there is no result to promote."
        ),
        trial_id=None,
    ),
    "RO-EUWAX-SENTIMENT": MeasuredOutcome(
        hypothesis_id="RO-EUWAX-SENTIMENT",
        status=ResearchStatus.REJECTED,
        approved_by="pcctradinginc@gmail.com (research wave 2, 2026-09)",
        note=(
            "Blocked: the source returns HTTP 403 including for its own robots.txt, so it is not "
            "accessible without bypassing access protection, which this project never does. "
            "Recorded REJECTED rather than DORMANT or APPROVED: this is a firm, principled stop on "
            "this specific data-access avenue (not a small-sample problem like Eurex, and not a "
            "measured negative result like Cboe/CFTC), though a different, licensed source could "
            "still answer the underlying question some other way."
        ),
        trial_id=None,
    ),
    "RO-CFTC-POSITIONING": MeasuredOutcome(
        hypothesis_id="RO-CFTC-POSITIONING",
        status=ResearchStatus.DORMANT,
        approved_by="pcctradinginc@gmail.com (research wave 2, 2026-09)",
        note=(
            "Measured in full (TR-2026Q3-31e266, W12-D, 2026-09-25): 20/20 cells worse on CRPS, "
            "pinball and Brier, 20/20 BH-significant against, median |t|=9.91, 90% coverage "
            "degraded 0.888 -> 0.838. DO NOT PROMOTE, more decisively than Cboe. Recorded DORMANT "
            "for the same reason as Cboe -- measured, ladder not cleared, but not proof "
            "positioning carries no information at all (W12-D's own conclusion), so REJECTED "
            "would overstate it."
        ),
        trial_id="TR-2026Q3-31e266",
    ),
}


def catalog_by_id() -> dict[str, ResearchOpportunity]:
    """The catalog indexed by `hypothesis_id`, for O(1) lookup by callers."""
    return {opportunity.hypothesis_id: opportunity for opportunity in CATALOG}


def catalog_for_family(family: InformationFamily) -> list[ResearchOpportunity]:
    """Catalog entries belonging to one `InformationFamily`, in catalog order."""
    return [opportunity for opportunity in CATALOG if opportunity.information_family is family]


__all__ = [
    "CATALOG",
    "W12_MEASURED_OUTCOMES",
    "MeasuredOutcome",
    "catalog_by_id",
    "catalog_for_family",
]
