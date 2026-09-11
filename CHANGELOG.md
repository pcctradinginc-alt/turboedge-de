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

## [Unreleased]

### Planned (Phase 2–8)

- **Phase 2:** Multi-horizon underlying forecast, logistic regression, calibration, walk-forward validation
- **Phase 3:** Path models (bootstrap, regime-conditioned, Monte Carlo), overnight gap modeling, barrier detection, KO probability
- **Phase 4:** Product × horizon net-EV simulation, shrinkage, LCB, cluster risk analysis, ranking → ACTIONABLE gate unlocked
- **Phase 5:** Daily manual position re-evaluation, HOLD/REDUCE/EXIT emails, tracking PnL and maximum favorable/adverse excursion
- **Phase 6:** Alternative data (Eurex, Euwax, Cboe, ECB, FRED, CFTC)
- **Phase 7:** Bayesian strategy posterior, ensemble weights, positive memory, drift detection
- **Phase 8:** Experimental data (GDELT, SEC, EIA, attention alpha, lead/lag signals)

