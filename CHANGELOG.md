# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-09-10

### Added

- **Phase 0: Product Data Feasibility**
  - Data source research (Round 1 + 2) with honest bot UA, no anti-bot bypass
  - Adapter architecture: `DataSourceAdapter` protocol, HTTP client with `tenacity` retry/backoff
  - **BNP Paribas** (`derivate.bnpparibas.com`) — WORKING (`POST /apiv2/api/v1/productlist/leverage`), live product quotes with ISIN/WKN/bid/ask/financing levels; pitfalls documented (no `ask` key outside trading hours, timestamps are Europe/Berlin local)
  - **Citi / CitiFirst** (`de.citifirst.com`) — WORKING (`POST /citi/v1/theq/api/ProductSearch/de-DE/Search`), secondary cross-validation source; gaps documented (pagination capped at 25 rows, direction filter unresolved)
  - **Börse Stuttgart KO-Finder** — BLOCKED, not implemented (Cloudflare bot-management blocks all requests, including `robots.txt`); no adapter exists — `boerse_stuttgart` is only an `enabled: false` documentation entry in `configs/sources.yaml`
  - **Börse Frankfurt / Deutsche Börse API** — BLOCKED, not implemented (requires salted-hash headers from obfuscated JS, no bypass attempted); no adapter exists — `deutsche_boerse` is only an `enabled: false` documentation entry in `configs/sources.yaml`
  - **yfinance** — WORKING (daily OHLC for DAX, NDX, GC=F, EURUSD=X); each ticker's timezone differs, must convert to UTC explicitly
  - **ECB €STR Data API** — WORKING (daily risk-free rate, SDMX-JSON format)
  - **CSV import adapter** — WORKING (manual product upload, legitimate fallback when automated sources unavailable)
  - Product universe discovery, deduplication (ISIN-based, preferring fresher/higher-quality quotes)
  - Quote health scoring (freshness, quality_score)
  - Financing level history tracking per product

- **Phase 1: Pricing & Cost Engine**
  - Intrinsic value calculation (Long/Short, ratio-adjusted, with underlying reference source)
  - Implied financing spread inference from financing level changes (Footnote 12: actual/360, calendar days, handles dividends/adjustments via `adjustment_suspected` flag). Requires ≥2 clean `product_snapshots` on different calendar days per ISIN; falls back to `configs/risk.yaml`'s `default_financing_spread` otherwise (reason `financing_spread_default`). Gap premium has no such warm-up requirement — it is estimated independently from the underlying's daily bar history.
  - Gap premium estimation (overnight and weekend log-gaps, empirical distribution, fair premium model)
  - Issuer margin decomposition (ask = intrinsic + trading spread + gap premium + financing drag + issuer margin)
  - Cross-issuer comparison (weighted median consensus spot, z-score residuals, wrapper edge, issuer markup score, quote dislocation score)
  - Integrity checks (bid ≤ ask, ratio > 0, barrier valid, quote freshness, ratio_factor_error detection)
  - Liquidity factor (geometric mean of quote presence, size coverage, freshness, spread quality)

- **Protected TSMOM Baseline (§7, Rule 10)**
  - Time-Series Momentum signal: 21/63/126-day lookbacks (trading days)
  - EWMA volatility normalization (λ = 0.94)
  - Clipping to [-3, +3] σ
  - Mean aggregation of three horizons
  - Threshold 0.5 (versionized `tsmom_horizon_norm_v1`, protected from silent reoptimization)
  - Status: PROTECTED (never revert; new versions required for changes)

- **Core Infrastructure**
  - Pydantic v2 schemas (ProductSnapshot, UnderlyingBar, SignalSnapshot, CostDecomposition, CandidateEvaluation, SourceHealthRecord, etc.)
  - DuckDB persistent storage (instruments, product_snapshots, underlying_prices, signals, candidate_sets, source_health, positions_manual, notifications_sent, runs tables)
  - Parquet immutable snapshots per scan run (date-partitioned, `state/snapshots/<table>/date=YYYY-MM-DD/<run_id>.parquet`)
  - Config system (YAML loading, pydantic validation, `config_hash()` for reproducibility)
  - Logging (structlog with JSON/console output modes, TTY detection)
  - Provenance tracking (git_commit, config_hash, data_snapshot_hash per prediction; all timestamps tz-aware UTC)

- **CLI Commands**
  - `turboedge universe [--underlying DAX]... [--source all|bnp_paribas|citi|...]` — fetch and deduplicate product universe; rich table output by underlying/issuer/direction/type
  - `turboedge scan --underlying DAX [--direction long|short] [--horizon 3d|5d|7d|10d|14d] [--top 20] [--report-out PATH] [--json-out PATH] [--email]` — full pipeline (health → bars → signal → quotes → gates → rank → report)
  - `turboedge sources health [--json-out PATH] [--email-on-fail] [--fail-on-error]` — monitor source availability, freshness, missingness, schema consistency, cross-source agreement
  - `turboedge notify test` — test email integration (dry-run if credentials missing)
  - `turboedge position add --wkn X [--isin Y] --qty 100 --price 4.86 --date 2026-09-10` — manual position ledger
  - `turboedge position list [--status open|closed]` — list positions (all statuses by default)
  - `turboedge position close --wkn X --price 5.42 --date 2026-09-15` — exit position
  - `turboedge db info` — introspect storage (tables, row counts)
  - Global options: `--config-dir PATH`, `--state-dir PATH`, `--log-format [console|json]`
  - Exit codes (per-command, not global): 0 (ok); 2 (config/argument error, all commands); 3 (`universe`/`scan` — every product source failed); 4 (`sources health --fail-on-error` with overall status FAIL); 1 (`sources health`/`notify test` — sending the alert/test email raised)

- **Email Notifications**
  - SMTP SSL to smtp.gmail.com:465 (via `email.mime`)
  - Environment variables: `GMAIL_USER`, `GMAIL_APP_PASSWORD` (Google App Password, not regular password), `TURBOEDGE_EMAIL_TO` (comma-separated recipients)
  - Jinja2 plaintext templates (no HTML, safe plain-text rendering)
  - Deduplication via hash(candidate_id, category, rounded core values) in `notifications_sent` table (prevents duplicate emails for same candidate across multiple runs)
  - Dry-run mode if credentials missing (logs output to console instead of sending)

- **Position Ledger**
  - Manual entry/exit recording (WKN, optional ISIN, quantity, price, date); no notes field
  - No automatic re-evaluation yet (Phase 5)
  - Persistent storage in DuckDB `positions_manual` table

- **Source Health Monitoring**
  - Health checks (availability ≥95%, freshness ≤1h for critical sources, missingness <5%, schema 0 violations, cross-source ±2%)
  - Per-source status: PASS/WARN/FAIL
  - Called before every scan (Spec §30); scan continues but products downweighted if source FAIL
  - Email alerts (optional `--email-on-fail`)

- **Governance & Documentation**
  - `CLAUDE.md` — 36 binding rules (verbatim Spec §49), current milestone, development commands
  - `GOVERNANCE.md` — research governance (trial_id assignment, adaptation budget 6/quarter, ladder rule, ruin metric 20%/5%, sizing caps, promotion/demotion rules)
  - `SIGNAL_REGISTRY.md` — protected TSMOM specification, version history, change protocol
  - `LICENSE` — proprietary (Copyright 2026 pccTradingINC, no order execution)
  - `.env.example` — credential and config path templates
  - `docs/csv_import_format.md` — CSV import schema, delimiter/number/ratio/timestamp/boolean formats, error handling
  - `docs/data_sources.md` — detailed data source status table, pitfalls (BNP `ask` key absence, timestamp timezones), principles (adapter encapsulation, contract tests, honest UA, rate limits, ToS compliance), GitHub Actions considerations
  - `README.md` — overview, architecture pipeline, setup instructions, configuration, CLI reference, secrets setup (Gmail App Password step-by-step), exit codes, data storage (DuckDB + Parquet), reproducibility, GitHub Actions workflows, development, troubleshooting, known limitations, roadmap
  - `.github/workflows/tests.yml` — Python 3.12, `uv sync --extra dev`, ruff/mypy/pytest (live tests excluded by default)
  - `.github/workflows/source-health.yml` — cron health checks (Mo–Fr 06:15 UTC), email alert on FAIL, JSON artifact upload
  - `.github/workflows/scan-report.yml` — cron scans (Mo–Fr 13:45 UTC), manual dispatch, 7-day state cache, report artifacts, email notification

### Changed

- **DuckDB additive schema migrations** — `Store.init_schema()` now migrates any table whose on-disk schema predates the current code, in addition to creating missing tables. It compares each table's actual columns (`PRAGMA table_info`) against the columns its DDL declares (derived from the DDL itself via a throwaway probe table, so the DDL text stays the single source of truth), adds any missing column as a nullable `ALTER TABLE ... ADD COLUMN`, and raises `StoreError` if an existing column's type does not match what the schema now expects — column removal or type changes are never applied automatically. Every applied migration is logged to a new `schema_migrations` table (`applied_at`, `table_name`, `column_name`, `action`) and to structlog. This closes a CI failure mode: `scan-report.yml` restores `state/turboedge.duckdb` from the GitHub Actions cache across runs, and `CREATE TABLE IF NOT EXISTS` alone does not add a newly introduced column (e.g. `product_snapshots.underlying_price_ref_timestamp`) to a table a previous run already created — the next `append_product_snapshots` would otherwise fail.
- `SCHEMA_VERSION` bumped `1.0.0` → `1.1.0` (new optional `ProductSnapshot.underlying_price_ref_timestamp` field, additive/non-breaking).
- **Citi closing-price-only rows no longer misclassified as DATA_QUALITY** — a product whose source explicitly reports no live two-way market at all (bid AND ask both missing, `quote_presence=False`, e.g. Citi's `referencePriceMethod == "Closing Price"` rows) is now categorized `REJECT` with reason `no_live_quote` instead of `DATA_QUALITY`: `pricing/integrity.check_product` no longer raises a spurious `missing_bid` failure for this specific case (master data — financing level, barrier, ISIN, underlying mapping — is still fully validated by every other check), and `ranking/gates.py`/`pipeline/scan.py` route it through a REJECT gate instead. A missing bid with `quote_presence` `True`/`None` (i.e. the source did not explicitly say "no live quote") remains a genuine `DATA_QUALITY` failure, unchanged.

### Not Included / Known Limitations

- **ACTIONABLE gate blocked** — Requires Path Model and LCB_EV calculation (Phase 3–4). Only WATCH, REJECT, DATA_QUALITY categories used in Phase 0–1.
- **No path-dependent KO modeling** — Single-value end-date forecast only. Barrier distance tracked but not path risk. Bootstrap/Monte Carlo models in Phase 3.
- **No multi-horizon forecasts** — Logistic regression and walk-forward CV for underlying direction in Phase 2.
- **No historical product data** — Phase 0–1 only stores forward snapshots from Phase 0–1 onward. No backfill of products killed before 2026-09-10.
- **Limited signal family** — Only protected TSMOM baseline. Challenger signals added in Phase 7 (ensemble weights).
- **No position re-evaluation** — Ledger records entries/exits but no daily HOLD/REDUCE/EXIT signals (Phase 5).
- **Alternative data not integrated** — No Eurex, Euwax, Cboe, ECB, FRED, CFTC (Phase 6).
- **No Stuttgart/Frankfurt adapters** — Both blocked (Cloudflare/salted-hash); no adapter code exists, only `enabled: false` documentation entries in `configs/sources.yaml`.
- **Citi gaps unresolved** — Pagination beyond 25 rows, direction filter field name, `ask == 0.0` ambiguity (after-hours vs. auction-only) documented but not fully closed; production use requires workaround.
- **yfinance unofficial** — No published API contract. Volume unreliable for indices/FX.
- **Financing spread inference warm-up** — Requires 5–10 days of historical snapshots for stable gap-premium and level-change estimates.
- **Spot consensus thin** — Only 2 sources (BNP, Citi); outlier detection limited by sample size.
- **No automatic recalibration** — Daily drift/calibration/regret monitoring not automated; manual review of WATCH candidates advised.
- **GitHub Actions cache eviction** — 7-day inactivity → state loss; time-critical production runs should use owned infrastructure.
- **No Markdown or graphical reports** — Plain-text email templates and JSON output only; HTML/Markdown report generation deferred.

### Testing & CI

- Contract tests for working sources (ECB, yfinance, CSV) with real fixtures and mocked HTTP calls (respx)
- Live tests excluded by default (`@pytest.mark.live`); run with `-m live` if needed
- GitHub Actions CI validates Python 3.12, ruff, mypy, pytest on every push/PR
- Cron health checks and scheduled scans (state caching, email notifications)

### Notes

- All timestamps UTC, tz-aware, validated at every stage
- No `pass` stubs; all functions have implementations
- Pydantic v2 strict mode for schema validation
- No README.md or doc commits in this milestone (documentation files added to repo, managed separately)

---

## [0.2.0] — 2026-09-13

### Added

- **Forecast engine** (`models/forecast.py`, `models/directional.py` —
  `NullModel`, `TsmomForecastModel`, `LogisticDirectionModel`;
  `models/quantile.py` — `RidgeReturnModel`; `models/calibration.py`,
  `models/ensemble.py`; `features/returns.py`, `features/volatility.py`,
  `features/trend.py`; `backtest/purged_cv.py`, `backtest/walkforward.py`,
  `backtest/metrics.py`, `backtest/significance.py`). `HorizonForecast`
  (p_up, mean, sigma, quantiles, expected shortfall, uncertainty) per
  underlying/horizon, walk-forward evaluated with purged/embargoed
  expanding-window CV. **Measured 2026-09-12 (Workstream W4): no model
  beats the null-model benchmark out-of-sample — see "Measured results"
  below and `docs/measured_results.md`.**
- **Six pre-registered challenger signal families**
  (`models/challengers.py`: `VolTargetedTsmom`, `LowVolRegimeTrend`,
  `ShortHorizonReversal`, `VixTermStructure`, `CrossAssetLeadLag`,
  `SeasonalityTurnOfMonth`; `features/cross_asset.py` for the VIX-term and
  cross-asset lead/lag feature sets). Pre-registered (Workstream W9,
  trial IDs `W9-2026Q3-001`–`006`) before measurement, per Master Spec §27
  / CLAUDE.md rule 26. **Measured 2026-09-13: 0 of 80 measured cells beat
  the null model after Benjamini-Hochberg deflation; all six recorded
  `dormant`.** See `SIGNAL_REGISTRY.md` §3.
- **Path simulation** (`simulation/{bootstrap,overnight,paths,barrier,payoff}.py`):
  `simulate_paths` (block_bootstrap / regime_bootstrap / monte_carlo /
  new `vol_scaled_bootstrap`), overnight/weekend gap modeling,
  `first_hit_index` barrier-touch detection, `brownian_bridge_hit_probability`
  analytic control, `simulate_product_payoff` (net returns, KO handling,
  MFE/MAE, ES95) against `ProductTerms` (classic/open-end/mini-future KO
  residual handling per direction).
- **Evaluation/ranking layer** (`ranking/{ev,shrinkage,lcb,utility,sizing,cluster,liquidity}.py`):
  product × horizon net-EV evaluation, winner's-curse shrinkage, lower
  confidence bound (LCB), utility-based ranking, Kelly-fraction position
  sizing, correlation-cluster risk limits.
- **Forward ledger & learning loop** (`learning/{ledger,labeler,counterfactual,posterior,registry,ensemble_weights,trials,drift}.py`):
  `ForwardLedger` (record/label/query, stratified shadow sampling),
  `label_due_entries` (real-bid labeling, never optimistic on ambiguous
  paths), `StrategyPosterior` (Normal-Inverse-Gamma per signal_family ×
  horizon), `ModelRegistry` (champion/challenger/dormant/protected status,
  weight updates), `new_trial_id` with quarterly budget check,
  `PageHinkley` drift detection, `learning/failed_hypotheses.py`
  (append-only hypothesis graveyard, `state/registry/failed_hypotheses.json`).
- **Monthly and weekly reports** (`reporting/{monthly,weekly,html,console,_common,redaction}.py`):
  monthly status report (works even with zero trades), weekly research
  tournament report.
- **Encrypted state persistence and pipeline scheduler**
  (`state/{__init__,crypto,archive}.py`, `cli_state.py`,
  `.github/workflows/pipeline.yml`): `turboedge state pack`/`unpack`
  (AES via `TURBOEDGE_STATE_KEY`, scrypt-derived, integrity-checked),
  `turboedge db compact` retention, and a single consolidated
  `pipeline.yml` workflow (modes `scan`/`eod`/`weekly`/`monthly`,
  scheduled Mon–Fri + weekly/monthly, encrypted `turboedge-state-enc`
  artifact instead of the previous 7-day Actions cache) replacing
  `scan-report.yml` (removed). `TURBOEDGE_PUBLIC_LOGS=1` redacts
  ISIN/WKN/prices from stdout and structured logs on this public repo.
- **gettex adapter validation**: multi-issuer (BNP Paribas, UniCredit,
  Goldman Sachs, HSBC observed) live quotes with a leverage-implied
  bid/ask-to-underlying ratio derived and independently verified against
  gettex's own reference price. Live-validated 2026-09-13: DAX 1695/3000
  rows (56.5%) ratio-derived+verified, median deviation from reference
  spot 0.0497%; NDX 98/3000 (3.3%), median deviation 0.0140%. See
  `docs/measured_results.md` §4 and `gettex_adapter_validation.md`.

### Changed

- **`simulate_paths` default `method` changed from `block_bootstrap` to
  `vol_scaled_bootstrap`** — data-driven, per Workstream W5's calibration
  comparison against 558 realized DAX start dates (2015–2025):
  `vol_scaled_bootstrap` has ~2.7–2.8× lower mean absolute KO-probability
  calibration error than the old default in both the trading-relevant
  (k=1.5–2σ) and full tested range. `regime_bootstrap`'s dormant-bucket
  bug (`min_bucket_days=250` never actually restricted the sample; 0%
  activation, silently degenerating to `block_bootstrap`) was also fixed
  (`min_bucket_days` lowered to 150), guarded by a new regression test.
  Both methods still **over-predict** P(KO) at small-to-moderate barrier
  distances — a known, conservative (not calibrated) residual bias; see
  `docs/measured_results.md` §3.
- **`simulate_product_payoff` vectorized** to remove the per-path/per-day
  Python loop for MFE/MAE on the simple fair-value fast path that W5
  flagged as a scaling bottleneck when scanning many products. Measured
  by the payoff-vectorization workstream, same machine, same process,
  before/after, no change in results:
  - Per-call at 300 paths: 6.13 ms (scalar reference implementation) →
    0.12 ms (vectorized) — **~51×**.
  - Full realistic scan workload (4000 products × 5 horizons × 2000
    paths = 20,000 calls, varying financing levels, no result caching):
    7.2–7.4 s total (0.36 ms/call), against ~700–820 s extrapolated for
    the old implementation — **roughly 95–110×**, comfortably inside the
    90 s budget for that workload.
  - Correctness, not just speed, is verified:
    `tests/simulation/test_payoff.py::test_vectorized_matches_reference_implementation`
    compares the vectorized path against the retained scalar reference
    implementation across all three product types × long/short,
    element-wise at `atol=rtol=1e-9` for `net_returns`/`ko`/`mfe`/`mae`
    plus every summary statistic — this is why the speedup did not
    change any result reported elsewhere in this changelog.
  - Remaining bottlenecks after the change (per cProfile): `np.quantile`
    overhead, `maximum`/`minimum.accumulate`, the fair-value array calls,
    and `first_hit_index`. No Python per-path loop remains.
- **Score-formula correction** in `ranking/utility.py`'s final ranking
  score (`score_jh`, Master Spec §18): the literal spec formula
  (`score = LCB(U) * liquidity_factor * calibration_factor *
  strategy_posterior_factor * positive_memory_factor`) is only
  order-preserving when `LCB(U) >= 0` — for `LCB(U) < 0`, multiplying a
  negative number by a smaller quality factor moves it toward zero
  (backwards: a worse-quality candidate would rank *above* an otherwise
  identical better one). Fixed by dividing (`LCB(U) / max(f, eps)`)
  instead of multiplying when `LCB(U) < 0`, restoring "worse factor →
  worse score" on both sides of zero; documented as a deliberate,
  reasoned deviation from the literal spec text, not a new research
  hypothesis (no separate trial_id opened). See the docstring of
  `ranking/utility.py::score` for the full derivation.
- **DuckDB schema migration** extended additively for every new table
  introduced by the forward ledger, forecasts, position evaluations, and
  schema-migration log (see `storage/schemas.py`/`storage/duckdb.py`);
  existing `Store.init_schema()` migration behavior (§0.1.0) unchanged.

### Fixed

- `regime_bootstrap`'s silent 0%-activation dormancy (see "Changed" above).

### Measured results

- **Workstream W4 (2026-09-12):** protected-TSMOM-derived forecast
  distribution and null-model walk-forward comparison, 4 underlyings × 5
  horizons (20/20 combinations) plus one reduced-scope logistic data
  point. **No model beat the null model out-of-sample** (worse Brier in
  20/20 cells; ECE worse by 5–15× in nearly every cell; PSR ≈0.97–1.00 for
  both models, an artifact of the 2010–2026 bull market, not evidence of
  skill).
- **Workstream W9 (2026-09-13):** six pre-registered challenger signal
  families, 80 measured (family, underlying, horizon) cells. **0 of 80
  cleared Benjamini-Hochberg FDR (α=0.10)** on Brier or return difference
  vs. null; best single-cell edge 1.85 bp against a required 10 bp; 0/80
  cleared realistic Turbo round-trip costs. All six recorded `dormant`.
- **Workstream W5 (2026-09-12):** KO-probability path-model calibration
  against 558 realized DAX start dates (2015–2025). Every method
  over-predicts P(KO) at trading-relevant barrier distances; best
  (`vol_scaled_bootstrap`) still over-predicts by a mean signed diff of
  −0.019 at k=1.5–2σ — a conservative, quantified residual bias, not a
  calibrated probability.
- Full numbers, methods and sample sizes: `docs/measured_results.md`
  (new). **Net effect on system behavior: the system currently outputs
  "no trade" — no model has a measured advantage over doing nothing, and
  thresholds are not lowered to manufacture suggestions.**

### Migration

- `SCHEMA_VERSION` bumped for the additive tables above (forward ledger,
  forecasts, position evaluations, schema-migration log); existing
  databases migrate automatically via `Store.init_schema()` on next run,
  consistent with the additive-migration policy established in 0.1.0.

---

## [Unreleased]

### Fixed

- **Split the single `max_quote_age_s` freshness gate into two** (Build
  Contract freshness/duration review, 2026-09-14): CI measured
  `fetch_duration_s` of 227.1s (DAX) / 261.4s (NDX) — well above the old
  120s "stale" cutoff, so essentially every candidate's decision-time quote
  age (`quote_timestamp` vs. `evaluation_time`, set only after the whole
  multi-source fetch completed) exceeded 120s regardless of how fresh the
  source's own data was, rejecting 11,060/11,114 DAX and 7,059/10,566 NDX
  candidates on staleness alone (run 34883437354, commit 1fe080e:
  ACTIONABLE=0, WATCH=**0**, REJECT=18,119, DATA_QUALITY=3,561, combined).
  `configs/risk.yaml` now has two independently-configurable thresholds:
  `max_source_quote_age_s` (900s — `quote_timestamp` vs. each snapshot's
  own `retrieved_at`, a source data-quality signal, reason
  `source_quote_stale`) and `max_quote_age_at_decision_s` (450s —
  `quote_timestamp` vs. `evaluation_time`, a tradability signal that must
  exceed the pipeline's own realistic fetch duration, reason
  `quote_age_at_decision`). Both values are derived from measured
  distributions, not guesswork — see the config file's comments, which
  also document a several-tens-of-seconds issuer/local clock offset found
  while reproducing the source-side numbers (BNP's median source age is
  slightly negative). Verified the split still rejects genuinely stale
  source data (a 25-day-old Goldman Sachs quote, a 9,695s-old BNP outlier)
  via `source_quote_stale` regardless of the decision-time fix. Re-ran the
  full pipeline in CI after the fix (run 34888059675, commit 90513f5):
  ACTIONABLE=0, WATCH=**12,497**, REJECT=5,680, DATA_QUALITY=3,572 —
  fetch_duration_s 186.4s (DAX) / 128.3s (NDX), `source_quote_stale`
  750/6, `quote_age_at_decision` 1,850/133 (DAX/NDX).
- **Parallelized per-adapter product fetch** (`pipeline/universe.py`):
  BNP/Citi/gettex are three independent hosts, each already rate-limited
  independently (`adapters/base.py`'s per-host `_HostRateLimiter`,
  `>=1.5s`/request, never relaxed or parallelized *within* one host's own
  request stream), but were previously fetched one after another for no
  reason tied to that politeness rule. Fetching them concurrently instead
  bounds total `fetch_duration_s` by the slowest single adapter rather than
  their sum: measured (2026-09-14) local fetch_duration_s dropped from
  81.0s/74.9s (DAX/NDX) to 49.9s/57.4s; the same-day CI figures above
  (227.1s/261.4s pre-fix to 186.4s/128.3s post-fix) confirm the effect
  survives CI's noisier, slower network path. A full scan still
  realistically takes on the order of a minute (local) to a few minutes
  (CI); see README's "Known Limitations" for what that does and does not
  mean for decision-time quote freshness. New concurrency-specific test
  coverage in `tests/pipeline/test_universe.py`: every adapter is still
  called when another is slow/fails, a failing adapter's error lands in
  `source_errors` without blocking a faster one, `counts_by_source`/log
  order stay in original adapter order regardless of completion order, and
  each adapter's own fetch duration is recorded (and now logged) on both
  the success and failure path.
- **Moved `premium_over_fair`/`premium_uncertainty_term` out of
  `CandidateEvaluation.reasons`** into their own optional float fields
  (`pipeline/scan.py`, `storage/schemas.py`, `storage/duckdb.py`, additive
  migration), matching the existing `financing_spread_source` pattern.
  These are per-candidate numeric measurements, not reject reasons; left in
  `reasons`, their formatted values (e.g. `"premium_over_fair=0.0014"`)
  almost never repeated across candidates, so `reject_reason_counts`
  (`pipeline/scan.py::_log_scan_diagnostics`) filled up with count-1
  pseudo-reasons that drowned out the real, repeated reject reasons the
  counter exists to surface: 81 distinct / 96 occurrences (DAX) and 72
  distinct / 78 occurrences (NDX) pseudo-reason entries measured pre-fix in
  CI (run 34883437354, `gh run view 34883437354 --log | grep
  scan_diagnostics`), 79 distinct / 96 occurrences (DAX) and 12 distinct /
  12 occurrences (NDX) measured pre-fix locally the same day. Confirmed
  absent from both `reject_reason_counts` post-fix (local and CI, run
  34888059675).

### Added

- **CLI wiring for the Wave-2 integration** (`cli.py`, `cli_learn.py`,
  Build Contract v3 §E), landed in parallel with this changelog entry:
  `scan-all`, `label`, `learn`, `position reevaluate`, `report monthly`,
  `research tournament`, `forecast`, `backtest` are now real CLI
  subcommands (`uv run turboedge --help`), backed by the previously
  library-only `pipeline/scan_all.py`, `positions/reevaluate.py`,
  `reporting/{weekly,monthly}.py`, `learning/*`, `backtest/walkforward.py`.
  `.github/workflows/pipeline.yml`'s four modes (`scan`/`eod`/`weekly`/
  `monthly`) call these exact command names and are no longer blocked on
  this wiring.

### Planned

- Further data sources (Eurex, Euwax, Cboe, FRED, CFTC) — no current plan.

