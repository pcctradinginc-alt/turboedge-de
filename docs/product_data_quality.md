# Product Data Quality (Phase B — "Produktstammdaten härten")

This document records what Phase B actually measures about the reliability
of product master data (Bezugsverhältnis/ratio, KO-Barriere, Finanzierungslevel),
not what it assumes. It replaces impressions with numbers, dates and methods
— same convention as `docs/measured_results.md`.

**Read this first if you read nothing else:** every product this system has
priced from a live/issuer source so far reports its Bezugsverhältnis as
either issuer-`source_reported` or `derived_verified` (gettex's
derive-then-verify pipeline) — **never `unverified`** — because a gettex row
whose ratio derivation fails today is dropped before it ever becomes a
`ProductSnapshot` (see §3). The `unverified` tier therefore exists for a
source that has not earned a verified ratio and still gets one: today that
is exactly one place, `adapters/csv_import.py` (user-uploaded product lists,
"no validation of source" per `docs/data_sources.md`), which is intentionally
left untouched by this change and defaults to `unverified` for every field.
The new ACTIONABLE gate (§4) has measured **zero additional exclusions
today**, for the same reason `docs/measured_results.md` reports zero
ACTIONABLE candidates overall: there is currently nothing for the gate to
exclude. It is a safeguard for the day a candidate does clear the EV/KO/
cluster gates, not a correction to today's counts.

---

## 1. What was measured, and how

Every number below comes from running the **real, unmodified** adapter code
(`adapters/gettex.py`, `adapters/issuer_feeds.py`) against the **real
API-capture fixtures** already committed for contract testing under
`tests/fixtures/` (`docs/data_sources.md`: "No fabricated data in fixtures —
only real responses"). No numbers here are estimated, hand-written or copied
from adapter docstrings — every one is the direct output of a Python session
that imported the production adapter classes and called
`fetch_products(...)` on them, exactly as `pipeline/scan.py` does, mocking
only the HTTP transport (`respx`, the same library the contract test suite
uses) so the real parsing/derivation/verification code ran unmodified.

Sources combined into one run:

| Source | Fixture(s) used | Notes |
|---|---|---|
| BNP Paribas | `tests/fixtures/issuer_feeds/bnp_paribas/productlist_leverage_dax_market_hours_ask_present.json` | Real DAX capture during market hours (ask present); same fixture the contract test suite's main BNP smoke test uses. |
| Citi | `tests/fixtures/issuer_feeds/citi/productsearch_search_dax_probe_output.json` | Real DAX capture; every row in this particular capture is a `referencePriceMethod == "Closing Price"` row (no live quote), a genuine and already-documented property of this fixture, not an artifact of this measurement. |
| gettex | `tests/fixtures/gettex/{dax_turbosendlos,nasdaq100,sp500,eurostoxx50}_sample.json` | Real multi-issuer captures (BNP Paribas, Goldman Sachs, HSBC, UniCredit observed live in this feed). Each underlying fetched with its own fixture's real capture time as the adapter clock, so freshness gates are evaluated honestly against real timestamps rather than an arbitrarily-chosen shared "now". |

This is a **small, fixture-sized** sample (63 products total) — the
contract-test fixtures were captured to exercise adapter code paths, not to
be a statistically representative universe sample (the existing
`tests/adapters/test_gettex.py` module docstring makes the same point about
these same fixtures: "used for schema-shape realism and end-to-end smoke
coverage", with derivation *rate* claims instead resting on the large
live-session measurements already cited in `adapters/gettex.py`'s own
docstring — 2,000-row DAX/NDX sessions, BEFUND 2, 2026-09-13). Where a rate
in this document and the large live-session numbers in `adapters/gettex.py`
disagree in magnitude, the live-session numbers are the better-powered
estimate; this document's job is to show the **mechanism** (which field gets
which tier, and why) working correctly end to end, not to re-derive gettex's
own already-measured production derivation rate.

Measured 2026-09-19, against the fixtures as committed at commit `8578c5c`.

---

## 2. Product counts

| Issuer | Products | Source |
|---|---:|---|
| BNP Paribas | 31 | 30 direct (`issuer_feeds.py`) + 1 via gettex's multi-issuer feed |
| Citigroup | 25 | direct (`issuer_feeds.py`) |
| Goldman Sachs | 5 | via gettex's multi-issuer feed only (no direct adapter exists) |
| HSBC | 2 | via gettex's multi-issuer feed only (no direct adapter exists) |
| **Total** | **63** | |

UniCredit appears in the gettex fixtures' issuer census generally but had no
row survive parsing/derivation in this particular small sample — absence
here is a sample-size artifact, not a statement that UniCredit rows never
verify (see `adapters/gettex.py`'s own live-session measurements for that).

---

## 3. Field-level reliability distribution

### 3.1 `ratio_reliability` (Bezugsverhältnis) — the field the Kernsatz is about

| Issuer | source_reported | cross_source_verified | derived_verified | unverified |
|---|---:|---:|---:|---:|
| BNP Paribas | 30 | 0 | 1 | 0 |
| Citigroup | 25 | 0 | 0 | 0 |
| Goldman Sachs | 0 | 0 | 5 | 0 |
| HSBC | 0 | 0 | 2 | 0 |
| **Total** | **55 (87.3%)** | **0** | **8 (12.7%)** | **0 (0.0%)** |

The single BNP Paribas `derived_verified` row is a BNP-issued product
observed *through gettex's feed* (gettex derives-and-verifies its ratio
independently of BNP's own direct feed, which reports it — the two sources
can and do carry different reliability tiers for the same issuer, by
design).

**Why `unverified` measures 0% here, and what that does and does not mean:**
`adapters/gettex.py`'s derive-then-verify pipeline already distinguishes
`ratio_rejected`/`verification_failed`/`quanto_ambiguous` internally (via its
`_FetchStats` counters, unchanged by this project phase) — but a row that
fails any of those checks is dropped **before** a `ProductSnapshot` is ever
built for it (`ratio` is a required, pricing-critical field on that model;
there is nothing to attach an `unverified` ratio *to* for such a row). So
every `ProductSnapshot` gettex actually emits today already passed
verification and is correctly `derived_verified` — the mapping's
"otherwise `unverified`" branch (`_ratio_reliability_for_outcome`, see
`adapters/gettex.py`) is real and tested directly
(`tests/adapters/test_gettex.py::test_ratio_reliability_for_outcome_maps_every_non_ok_outcome_to_unverified`),
but it is not exercised by any row that survives to become a product in
production *today*. The place `unverified` is measured in this system is
`adapters/csv_import.py` (manual CSV upload — intentionally out of this
phase's scope, defaults to `unverified` for all three fields because it is
never touched).

Per-underlying gettex outcome breakdown for this run (from the adapter's own
`gettex_fetch_summary` log line, unmodified):

| Underlying | rows | ratio_derived | ratio_rejected | verification_failed | quanto_ambiguous | reason for 0 derivation |
|---|---:|---:|---:|---:|---:|---|
| DAX | 15 | 0 | 0 | 0 | 0 | every row's `leverage` (252–319) exceeds `max_leverage_for_reference_spot` (200) — no row could contribute to `S_ref`, so `gettex_no_reference_spot` fired before any row-level derivation ran |
| NDX | 10 | 0 | 0 | 0 | 0 | same cause; leverage 539–3,022 |
| SPX | 10 | 0 | 0 | 0 | 0 | same cause; leverage 287–823 |
| ESTX50 | 10 | 8 | 2 | 0 | 0 | leverage 78–169, within the reference-spot construction range — `S_ref` derived successfully, 8/10 rows' ratio snapped to the grid *and* passed the implied-spot verification |

The DAX/NDX/SPX fixtures happen to sample only very-high-leverage
(near-knockout) rows — a real, structural property of *this specific
capture*, not a bug in this measurement or in the adapter: gettex's own
docstring documents the identical moneyness-shrinks-with-leverage effect at
length (`_DEFAULT_MAX_LEVERAGE_FOR_REFERENCE_SPOT`'s docstring) and the large
live sessions cited there (2,000-row DAX/NDX pulls) show a nonzero,
substantial derivation rate once the sample includes lower-leverage rows —
this fixture-sized sample by itself under-samples that population.

### 3.2 `barrier_reliability` (KO-Barriere)

| Issuer | source_reported | cross_source_verified | derived_verified | unverified |
|---|---:|---:|---:|---:|
| BNP Paribas | 31 | 0 | 0 | 0 |
| Citigroup | 25 | 0 | 0 | 0 |
| Goldman Sachs | 5 | 0 | 0 | 0 |
| HSBC | 2 | 0 | 0 | 0 |
| **Total** | **63 (100.0%)** | **0** | **0** | **0** |

### 3.3 `financing_level_reliability` (Finanzierungslevel)

Identical distribution to §3.2 — `source_reported` for all 63 products.
Unlike `ratio`, gettex reports `koLevelRefCurAbsolute` and
`financingLevelRefCurAbsolute` directly as raw feed fields (no derivation
step exists for either), so gettex products get `source_reported` for these
two fields, the same tier BNP/Citi use for their own directly-reported
master data — only `ratio` is special-cased on gettex because it is the one
field gettex's endpoints never expose at all (see `adapters/gettex.py`'s
module docstring).

---

## 4. Live quote and independent-verification summary

| Metric | Value |
|---|---:|
| Total products measured | 63 |
| With a live, two-way quote (`quote_presence=True`) | 38 (60.3%) |
| `ratio_reliability != unverified` ("independently trustworthy" ratio) | 63 (100.0%) |
| … of which `source_reported` | 55 (87.3%) |
| … of which `derived_verified` | 8 (12.7%) |

Per-issuer live-quote share:

| Issuer | Live two-way quote |
|---|---:|
| BNP Paribas | 31/31 (100.0%) |
| Citigroup | 0/25 (0.0%) — this capture's rows are all `referencePriceMethod == "Closing Price"` (see `docs/data_sources.md`) |
| Goldman Sachs | 5/5 (100.0%) |
| HSBC | 2/2 (100.0%) |

Live-quote presence and ratio reliability are independent dimensions by
design: Citigroup's 0% live-quote share in this capture does not affect its
100% `source_reported` ratio reliability — master data can be trustworthy
even when the current quote is not tradable (`ranking/gates.py`'s
`no_live_quote`/`bid_only` REJECT reasons already handle the quote-liveness
dimension separately from the new `ratio_unverified` reason this phase adds).

---

## 5. Impact of the new ACTIONABLE gate

`ranking/gates.py::evaluate_gates` now additionally requires
`ratio_reliability != FieldReliability.UNVERIFIED` before assigning
`ACTIONABLE` (new reject-style reason `ratio_unverified`, surfaced only when
it is the thing actually stopping an otherwise-ACTIONABLE-eligible
candidate — see `tests/ranking/test_gates.py`). Measured impact:

- **Additional candidates excluded from ACTIONABLE today: 0.** As
  `docs/measured_results.md` documents, **zero** candidates have reached
  ACTIONABLE from any live pipeline run so far — no forecast model has a
  measured out-of-sample edge, so `lcb_ev > 0` alone is never satisfied. A
  gate that only tightens the *last* precondition of an already-unmet
  conjunction changes nothing observable today.
- **What the gate does change:** the moment any forecast model eventually
  clears the EV/KO/cluster gates (the stated goal of this whole research
  program), a candidate whose ratio is `unverified` — today, concretely,
  anything sourced only from `csv_import.py` — will be held at WATCH instead
  of silently becoming ACTIONABLE on unverified master data. It remains
  fully visible: still ranked, still written to the forward ledger
  (`forward_ledger`) and the shadow-portfolio sample
  (`learning/ledger.py::select_shadow_sample`) exactly as before, with the
  new `ratio_unverified` reason recorded — nothing is deleted or hidden, so
  this system can measure, later, how much (if any) EV such candidates would
  have realized, and how their outcomes compare to the reliably-sourced
  candidates that do reach ACTIONABLE.
- This is the intended shape of a hardening change per this phase's brief:
  strictly harder to reach ACTIONABLE, never easier, with a "no measurable
  change today, protection kicks in later" honest answer preferred over
  fabricating today's impact.

---

## 6. Test coverage added for this phase

| Area | What is tested |
|---|---|
| `storage/schemas.py` | `FieldReliability`'s four levels are individually assignable and round-trip through `model_dump`/`model_validate`; the default is `UNVERIFIED` when unset. |
| `ranking/gates.py` | Every non-`UNVERIFIED` level allows ACTIONABLE (parametrized over the three non-unverified tiers); `UNVERIFIED` blocks ACTIONABLE while still allowing WATCH and appends the `ratio_unverified` reason; a Hypothesis property test asserts no combination of `lcb_ev`/`p_ko`/`cluster_risk_pass` can make an `UNVERIFIED`-ratio candidate ACTIONABLE; backward compatibility when the field is left unset (`None`, every pre-existing caller). |
| `adapters/gettex.py` | `_ratio_reliability_for_outcome` maps `"ok"` → `DERIVED_VERIFIED` and every other outcome string (`ratio_rejected`/`verification_failed`/`quanto_ambiguous`) → `UNVERIFIED`; a real derive-and-verify fetch asserts every emitted snapshot is `DERIVED_VERIFIED` for ratio and `SOURCE_REPORTED` for barrier/financing level. |
| `adapters/issuer_feeds.py` | Both the real-fixture BNP smoke test and the real-fixture Citi closing-price test assert `SOURCE_REPORTED` on all three fields (the Citi case specifically proves reliability survives even when the quote itself is non-live). |
| `storage/duckdb.py` | Additive-migration test extended: an old on-disk table missing the three new columns gets them added (nullable `ALTER TABLE`), a pre-migration row reads back as `UNVERIFIED` (never guessed), and a fresh snapshot's explicit level round-trips exactly; a dedicated round-trip test covers all three fields together, default and explicit. |
| `pipeline/scan.py` (`tests/pipeline/test_scan_ev.py`) | End-to-end: the exact scenario that reaches ACTIONABLE in the sibling test (`test_run_scan_with_rng_writes_forecasts_and_ledger_and_can_reach_actionable`) is re-run with the same product forced to `ratio_reliability=UNVERIFIED` and the same RNG seed — asserts the candidate never becomes ACTIONABLE, and that it still appears in the forward ledger. |

**Test count:** 1,361 tests passing before this phase (the CI baseline this
phase started from, commit `8578c5c`) → **1,379 tests passing after** (+18,
verified via `pytest --collect-only` before/after per file):

| File | Before | After | Δ |
|---|---:|---:|---:|
| `tests/ranking/test_gates.py` | 22 | 29 | +7 |
| `tests/adapters/test_gettex.py` | 58 | 62 | +4 |
| `tests/adapters/test_issuer_feeds.py` | 48 | 48 | +0 (reliability assertions added to two existing real-fixture tests instead of new ones, to avoid duplicating fixture-loading boilerplate) |
| `tests/storage/test_schemas.py` | 13 | 18 | +5 |
| `tests/storage/test_duckdb.py` | 21 | 22 | +1 (plus one existing migration test's assertions updated, not counted as new) |
| `tests/pipeline/test_scan_ev.py` | 6 | 7 | +1 |
| **Total** | **1,361** | **1,379** | **+18** |

Full suite: `ruff check` clean, `ruff format --check` clean, `mypy src`
clean (106 source files), `pytest -q`: 1,379 passed, 6 deselected
(live/network tests, deselected by default, unchanged from before this
phase).

---

## 7. What this phase deliberately did not touch

Per the phase brief:

- No new issuer sources were researched — `docs/data_sources.md`'s seven
  already-attempted-and-failed paths (Börse Stuttgart, Börse Frankfurt, HSBC
  direct, Société Générale, UniCredit onemarkets, DZ Bank, Vontobel) are
  unchanged and remain a separate work item.
- No threshold, gate ordering, or risk parameter was changed except the one
  new, strictly-additive ACTIONABLE precondition described in §5 — every
  existing gate (spread, leverage, barrier distance, quote freshness, ...)
  behaves exactly as before.
- `state/retention.py` was not touched.
- `Instrument` (the slowly-changing master-data table) and `CandidateEvaluation`
  were deliberately left without their own reliability fields — the gate
  operates on `ProductSnapshot`/`GateInput` directly, and the reject/watch
  `reasons` list (already a free-form `list[str]`) is where the new
  `ratio_unverified` reason surfaces without needing a new column there.
