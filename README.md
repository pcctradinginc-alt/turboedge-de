# TurboEdge-DE

**Research system for identifying and evaluating German turbocertificates (knock-out products).** Manual execution only. No broker integration, no order placement.

---

## Status

**Phase 0 + Phase 1 + Protected TSMOM Baseline + Gmail Notifier**

- Data source research (Round 2 complete; BNP Paribas + Citi confirmed working live)
- Product normalization, deduplication, storage (DuckDB + Parquet snapshots)
- Cost decomposition engine (intrinsic value, financing spread, gap premium, issuer margin, cross-issuer scores)
- Integrity checks and gates (WATCH, REJECT, DATA_QUALITY)
- Email notifications (Gmail SMTP with deduplication)
- Manual position ledger
- Source health monitoring

**Not yet implemented (Phase 2–8):** path-dependent KO probability modeling, LCB(EV) calculation, multi-horizon underlying forecasts, ensemble/adaptive strategy weights, alternative data feeds, position re-evaluation workflows.

**ACTIONABLE is technically locked** this milestone: `ranking/gates.py` only assigns it when `lcb_ev is not None and lcb_ev > 0 and p_ko is not None and cluster_risk_pass is True`. `lcb_ev`/`p_ko` are always `None` until the Phase 3–4 path model exists, so every candidate that clears REJECT/DATA_QUALITY falls through to WATCH. Only WATCH, REJECT, DATA_QUALITY are produced today.

---

## Architecture Overview

### Pipeline (`pipeline/scan.py`, `run_scan`)

1. **Source health** — check every enabled adapter + reference sources (ECB, yfinance)
2. **Underlying bars** — daily OHLC from yfinance, falls back to the last stored bars on fetch failure
3. **Signal** — EWMA volatility, gap distribution, protected TSMOM (21/63/126-day)
4. **`SignalSnapshot` persisted** — frozen before any product quote is fetched
5. **Products** — fetch from every enabled product source (BNP Paribas, Citi, CSV import), normalize, deduplicate by ISIN
6. **Pricing + gates** — integrity checks, cost decomposition (intrinsic, financing spread, gap premium, issuer margin), cross-issuer consensus, WATCH/REJECT/DATA_QUALITY classification (per product)
7. **Persist + report** — `candidate_sets` in DuckDB, Parquet snapshot, console table, optional email

### Spot resolution (`pipeline/scan.py`, `_resolve_spot`)

The spot used to price a product is the **cross-issuer consensus** (`pricing/cross_issuer.consensus_spot`: a robust median of `implied_underlying` across every quotable product on that underlying, MAD-outlier-filtered, and bias-corrected by averaging the LONG- and SHORT-side medians when both are present — a positive issuer margin baked into `mid` otherwise inflates LONG-implied spot and deflates SHORT-implied spot by the same amount). A source's own `underlying_price_ref` (e.g. BNP's `first.price`) is used **instead** only when it carries its own observation timestamp (`underlying_price_ref_timestamp`, distinct from the product's `quote_timestamp` — a source can batch/throttle its reference price independently of individual bid/ask ticks), that timestamp is fresh (`risk.max_quote_age_s`), and the value is within `risk.spot_ref_max_deviation_pct` of the consensus; otherwise the consensus is used and a `spot_ref_rejected` warning is recorded. Conflating a product's own quote freshness with its reference price's freshness previously caused near-identical products (same financing level/barrier/leverage) to show wildly different `issuer_margin_pct` — see `pricing/cross_issuer.py`/`pipeline/scan.py` docstrings for the measured example.

### Cost ranking (`cost_rank_score`, "Cost per exposure (h)")

Within WATCH, candidates are ranked by `cost_rank_score` = total round-trip cost over the scan horizon (spread + gap premium + financing + `max(issuer margin, 0)`, as a % of ask) **divided by leverage**. Dividing by leverage re-expresses cost as a % of underlying exposure instead of capital employed, so it does not mechanically favor low-leverage products (there is no leverage target in this system — a Hebel-2 and a Hebel-10 product with the same cost per unit of underlying exposure rank equally). `None` for a product with no ask (leverage cannot be computed).

### Modules

```
src/turboedge/
  cli.py                     typer app: universe, scan, sources, notify, position, db
  config.py                  YAML -> pydantic config; config_hash()

  adapters/
    base.py                  DataSourceAdapter protocol, HttpClient, HealthCheckResult
    issuer_feeds.py           BnpParibasTurboAdapter, CitiFirstTurboAdapter (live product quotes)
    ecb.py                   ECB EST (euro short-term rate) reference rate, SDMX-JSON
    fallback_prices.py       yfinance daily OHLC (DAX, NDX, GC=F, EURUSD=X)
    csv_import.py            CSV product import from state/imports/products/
    registry.py              Adapter factory; builds only cfg.sources[...].enabled adapters

  universe/
    underlying_map.py        Canonical underlying IDs (DAX, ESTX50, SPX, NDX, ...)
    classify.py               Product type & direction from raw data
    filters.py                Config-based product filtering
    discover.py                Merge sources, deduplicate by ISIN

  storage/
    schemas.py               pydantic v2: ProductSnapshot, SignalSnapshot, CostDecomposition, etc.
    duckdb.py                 append_*, query methods; DB schema creation
    snapshots.py               Parquet immutable archive (date-partitioned)

  pricing/
    intrinsic.py              Intrinsic value (Long/Short, ratio-adjusted)
    financing.py               Implied financing spread from level history
    gap_premium.py            Overnight/weekend gap distribution; fair premium
    issuer_margin.py           Decompose ask -> intrinsic + costs + margin
    cross_issuer.py            Consensus spot, z-scores, wrapper edge, dislocation scores
    integrity.py                Integrity checks (bid <= ask, ratio > 0, barrier valid)

  ranking/gates.py            WATCH/REJECT/DATA_QUALITY categories + reasons
  models/protected_baseline.py tsmom_horizon_norm_v1 (protected, never silently changed)
  monitoring/source_health.py  Health score (availability, freshness, missingness, schema, agreement)
  notifications/gmail.py       GmailNotifier (SMTP SSL smtp.gmail.com:465)
  positions/ledger.py          Manual position add/list/close (no re-evaluation yet)
  pipeline/scan.py             Orchestrates the pipeline above
```

---

## Data Sources

| Tier | Sources | Status |
|------|---------|--------|
| Live products | BNP Paribas (`derivate.bnpparibas.com`) — full DAX coverage with live bid/ask; Citi/CitiFirst (`de.citifirst.com`) — master data only, no live quotes observed | BNP working with live quotes (4340/4340 DAX); Citi returns closing-price reference data (25/33 DAX observed on 2026-09-11, 08:51 UTC) with referencePriceMethod="Closing Price" — sets bid/ask to None |
| Underlyings/rates | yfinance (daily OHLC), ECB EST rate (SDMX-JSON) | Working |
| Manual | CSV import from `state/imports/products/` | Working |
| Not implemented | Börse Stuttgart, Börse Frankfurt / Deutsche Börse | No adapter exists in code or `adapters/registry.py`; both remain `enabled: false` placeholder entries in `configs/sources.yaml` for documentation only |

Börse Stuttgart is blocked by a domain-wide Cloudflare bot-management block (rejects even `robots.txt`). Börse Frankfurt's API (`api.boerse-frankfurt.de`) requires salted-hash signature headers (`X-Client-TraceId`/`X-Security`) computed from obfuscated JavaScript; without them it returns HTTP 403. Neither was bypassed (no headless-browser challenge solving, no hash reproduction), per the research rules. See `docs/data_sources.md` for the full per-source write-up, including issuers checked and found unusable (HSBC, Société Générale, UniCredit onemarkets, DZ Bank, Vontobel, Morgan Stanley, ING, UBS KeyInvest, Deutsche Bank X-markets).

Known coverage gaps in the two working adapters:
- **BNP Paribas** — page size 1000, full coverage confirmed live (4340/4340 DAX products in one pull), with real bid/ask quotes. No `ask` key outside trading hours (key is absent, not `null`). `bidDate`/`askDate` have no UTC offset — must be localized as `Europe/Berlin`, not parsed as UTC. Only source with live two-way quotes; cross-issuer consensus is derived from BNP products only.
- **Citi** — first page only; response caps `items` at 25 regardless of `totalElementsCount` (25/33 DAX products observed live; `healthcheck()` reports WARN `citi_partial_universe`). **Critical: market-hours pull (2026-09-11, ~08:51 UTC) found `referencePriceMethod = "Closing Price"` and `ask = 0.0` on all 25 returned DAX rows — adapter sets bid/ask to `None`, marking them as stale/non-live.** Master data (financing level, barrier, ratio, ISIN) still usable for universe purposes. Underlying mapping (`_CITI_UNDERLYING_ISINS`) currently covers DAX only.

---

## Setup

### Requirements

- **Python 3.12+**
- **uv** package manager (≥0.4.0)

### Install

```bash
cd /Users/cc/Desktop/TURBO\ EDGE/turboedge-de

uv sync --extra dev

uv run pytest -q          # tests
uv run ruff check .       # linter
uv run mypy src            # type check
```

---

## Configuration

YAML config files live in `configs/` (loaded via pydantic):

| File | Purpose |
|------|---------|
| `default.yaml` | Log format, default top-N, default horizon, HTTP timeouts |
| `sources.yaml` | Per-source settings (base URL, timeout, rate limit, user agent, `enabled`) |
| `universe.yaml` | Underlying universe + product filters (leverage range, emittents, bid_only, knocked_out) |
| `gmail.yaml` | Email template paths, sender/recipient fields |
| `governance.yaml` | Adaptation budget, ladder rules, ruin metric |
| `risk.yaml` | Kelly fraction, position caps, gate thresholds, `default_financing_spread` |
| `models.yaml` | TSMOM threshold (protected `tsmom_horizon_norm_v1`) |

**Environment variables:**
- `TURBOEDGE_CONFIG_DIR` — config directory (default `./configs`)
- `TURBOEDGE_STATE_DIR` — state directory (default `./state`)
- `TURBOEDGE_LOG_FORMAT` — `console` or `json`
- `GMAIL_USER`, `GMAIL_APP_PASSWORD`, `TURBOEDGE_EMAIL_TO` — email secrets (see below)

---

## CLI Usage

All commands accept the global options `--config-dir`, `--state-dir`, `--log-format` before the subcommand. Every example below was checked against `uv run turboedge <cmd> --help`.

#### `turboedge universe [OPTIONS]`

Fetch and deduplicate the product universe from every enabled source.

```bash
turboedge universe                                       # enabled underlyings from universe.yaml
turboedge universe --underlying DAX --underlying NDX      # --underlying is repeatable
turboedge universe --underlying DAX --source bnp_paribas  # single adapter only ('all' is default)
```

#### `turboedge scan [OPTIONS]`

Run the full pipeline for **one** underlying (health -> bars -> signal -> quotes -> gates -> report). `--underlying` is a single value, not repeatable — run the command again per underlying.

```bash
turboedge scan --underlying DAX
turboedge scan --underlying DAX --top 20
turboedge scan --underlying DAX --direction long --horizon 7d --top 10
turboedge scan --underlying DAX --report-out ./reports/scan.txt --json-out ./reports/scan.json
turboedge scan --underlying DAX --email --top 10   # requires Gmail secrets; dry-run without them
```

**Horizons:** `3d`, `5d`, `7d` (default), `10d`, `14d`. **Directions:** `long`, `short`.

#### `turboedge sources health [OPTIONS]`

```bash
turboedge sources health
turboedge sources health --json-out ./reports/source_health.json
turboedge sources health --email-on-fail
turboedge sources health --fail-on-error   # exit 4 if overall status is FAIL
```

Checks (per source): availability (last 20 days), freshness, missingness, schema consistency, cross-source agreement; overall status PASS/WARN/FAIL.

#### `turboedge notify test`

Sends a test email. Dry-run (prints to console) if `GMAIL_APP_PASSWORD` is not set.

#### `turboedge position add / list / close`

```bash
turboedge position add --wkn ABC123 --qty 100 --price 4.86 --date 2026-09-10
turboedge position add --wkn ABC123 --isin DE000ABC1234 --qty 100 --price 4.86 --date 2026-09-10
turboedge position list
turboedge position list --status open
turboedge position close --wkn ABC123 --price 5.42 --date 2026-09-15
```

#### `turboedge db info`

Row counts for every DuckDB table (`instruments`, `product_snapshots`, `underlying_prices`, `signals`, `candidate_sets`, `source_health`, `positions_manual`, `notifications_sent`, `runs`).

---

## Exit Codes

Exit codes are per-command, not global — `scan` never exits `4`; only `sources health --fail-on-error` does.

| Command | Code | Meaning |
|---|---|---|
| any | `0` | Success |
| any | `2` | Invalid argument / config error (e.g. unknown `--underlying`, bad `--horizon`, malformed config YAML) |
| `universe`, `scan` | `3` | Every enabled product source failed (`NoProductsError`) — for `scan`, `--json-out`/`--report-out` are still written with the per-source errors and pre-flight health |
| `sources health` | `4` | A **critical** source FAIL **and** `--fail-on-error` was passed (critical: all enabled product sources except `csv_import`, plus `yfinance` and `ecb_estr`; `--email-on-fail` also gates to critical FAILs only, deduplicated per day) |
| `sources health`, `notify test` | `1` | Sending the alert/test email raised an exception |

---

## Secrets Setup

### Gmail App Password (required for email notifications)

1. **Enable 2-Step Verification** on your Google account: [myaccount.google.com/security](https://myaccount.google.com/security)
2. **Generate an App Password:** [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) → "Mail" + "Other (custom name)" → "TurboEdge-DE". Google gives you a 16-character password.
3. **Local `.env`** (copy `.env.example`, or write directly):
   ```bash
   GMAIL_USER=your-email@gmail.com
   GMAIL_APP_PASSWORD=abcdefghijklmnop
   TURBOEDGE_EMAIL_TO=alerts@example.com,trader@example.com
   TURBOEDGE_CONFIG_DIR=./configs
   TURBOEDGE_STATE_DIR=./state
   TURBOEDGE_LOG_FORMAT=console
   ```
4. **GitHub Actions secrets** (repo Settings → Secrets and variables → Actions): create `GMAIL_USER`, `GMAIL_APP_PASSWORD`, `TURBOEDGE_EMAIL_TO` — names must match exactly (`src/turboedge/notifications/gmail.py` reads them via `GmailCredentials.from_env()`).
5. **Test:** `uv run turboedge notify test` — dry-run (console output) if secrets are missing.

---

## Data Storage

**DuckDB:** `$TURBOEDGE_STATE_DIR/turboedge.duckdb` (default `./state/turboedge.duckdb`). Tables: `instruments`, `product_snapshots`, `underlying_prices`, `signals`, `candidate_sets`, `source_health`, `positions_manual`, `notifications_sent`, `runs`. All timestamps UTC, tz-aware. Inspect with `turboedge db info`.

**Parquet:** `$TURBOEDGE_STATE_DIR/snapshots/<table>/date=YYYY-MM-DD/<run_id>.parquet` — immutable, append-only, one file per table per run.

---

## Reproducibility

Every prediction is tagged with `git_commit`, `config_hash` (SHA256 of all config YAML), and `data_snapshot_hash` (SHA256 of the underlying-bar snapshot used). Checking out the same commit with the same config and snapshot data reproduces the same result.

---

## GitHub Actions Workflows

- **`.github/workflows/tests.yml`** — every push/PR: `uv sync --extra dev`, `ruff check`, `ruff format --check`, `mypy src`, `pytest -q --cov=turboedge` (live-marked tests excluded by default).
- **`.github/workflows/source-health.yml`** — cron Mon–Fri 06:15 UTC: `turboedge sources health --json-out ... --email-on-fail`, uploads the JSON report (14-day artifact retention).
- **`.github/workflows/scan-report.yml`** — cron Mon–Fri 13:45 UTC (or manual dispatch with `underlying`/`top`/`email` inputs): restores the `state/` cache, runs `scan --report-out ... --json-out ...` (fetches product universe internally), then "Summarize scan results" step writes category counts and warnings to GitHub Job Summary, saves cache and uploads reports (30-day retention).

State cache keys are per-run-id with a `turboedge-state-` prefix restore fallback; GitHub evicts any cache untouched for 7 days. Time-critical scans should not rely solely on scheduled Actions — cache eviction and scheduling jitter mean a run can start cold or be delayed. Use owned infrastructure for anything time-sensitive.

---

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

```bash
uv run pytest -q                    # live-marked tests excluded (pyproject.toml: addopts = "-m 'not live'")
uv run pytest -q -m live            # include live network tests
uv run pytest -q --cov=src          # with coverage
```

Contract tests for working adapters mock HTTP via `respx` against real captured fixtures (`tests/adapters/test_ecb.py`, `test_fallback_prices.py`, `test_csv_import.py`, and the BNP/Citi adapter tests).

### Troubleshooting

**`ModuleNotFoundError: turboedge` after `uv sync`** — usually a hidden `.venv` directory on macOS:
```bash
chflags -R nohidden /Users/cc/Desktop/TURBO\ EDGE/turboedge-de/.venv
```
then retry `uv run pytest -q`.

**Repo under iCloud Drive (`~/Desktop` is iCloud-synced on this machine)** — iCloud's file sync can corrupt or duplicate a `.venv` while `uv` writes to it (symptoms: `ModuleNotFoundError`, stray `" 2"`-suffixed files/directories under `.venv`). Two options:
- Clone/move the repo to a location outside iCloud Drive (e.g. `~/dev/`), or
- Keep it on Desktop but move the venv out of iCloud sync: `uv venv .venv.nosync && ln -s .venv.nosync .venv`. `.gitignore` already excludes both `.venv` and `.venv.nosync/`, and `uv sync`/`uv run` follow the `.venv` symlink transparently.

---

## Known Limitations

- **No path-dependent KO modeling.** Single-value end-date forecast only; barrier distance is tracked but not path risk. Requires Phase 3–4 bootstrap/Monte Carlo.
- **ACTIONABLE gate locked.** See Status above — `lcb_ev`/`p_ko` are always `None` this milestone.
- **Product data limited to BNP Paribas + Citi.** Börse Stuttgart and Börse Frankfurt have no adapter (see Data Sources). **Citi delivers closing-price reference data only, not live quotes** (observed 2026-09-11 during Xetra hours); BNP is the sole source with live bid/ask, so cross-issuer comparison currently works only within BNP products. Citi caps at 25 rows/underlying and maps DAX only. CSV import is an optional fallback for manual exports (empty directory → WARN, not FAIL).
- **yfinance is unofficial** — no published API contract; volume is unreliable for indices/FX; treated as best-effort.
- **Financing-spread inference needs history; gap premium does not.** The realized financing spread (`pricing/financing.py`) needs ≥2 `product_snapshots` for the same ISIN on different calendar days to invert a pair of financing-level observations; with fewer, `configs/risk.yaml`'s `default_financing_spread` (currently `0.025`) is used and the candidate is tagged `financing_spread_default`. Gap premium (`pricing/gap_premium.py`) is estimated from the underlying's daily bar history (yfinance), independent of product snapshots.
- **Spot consensus is thin.** No independent intraday market-data feed; consensus spot is derived only from BNP + Citi implied-underlying prices. A source's own reference price (`underlying_price_ref`) is only used as a validated override of the consensus — see "Spot resolution" above.
- **No recalibration loop.** Scans run offline on demand/schedule; drift/calibration/regret monitoring is not automated. Manual review of WATCH candidates advised.
- **GitHub Actions state-cache eviction.** See Workflows above.

---

## Roadmap

| Phase | Milestone |
|-------|-----------|
| 0–1 (done, 2026-09-10) | Data sources, cost engine, TSMOM baseline, Gmail notifier |
| 2 | Multi-horizon forecast, logistic regression, walk-forward CV |
| 3 | Path models (bootstrap, KO probability, LCB) |
| 4 | Product x horizon net-EV, sizing, ACTIONABLE gate, ranking |
| 5 | Daily position re-evaluation, HOLD/REDUCE/EXIT emails |
| 6 | Alternative data (Eurex, Euwax, Cboe, ECB, FRED, CFTC) |
| 7 | Ensemble weights, drift detection, positive memory |
| 8 | Experimental data (GDELT, SEC attention, lead/lag) |

No delivery dates are committed for phases 2–8.

---

## License

**Proprietary** — Copyright 2026 pccTradingINC. No order execution, no broker integration, research system only. See `LICENSE`.

---

## Disclaimer

This is a research and analysis tool, not investment advice.

- No historical backtest or live track record guarantees future performance.
- All signals are experimental and subject to change.
- Manual human review and risk management are essential before any trade.
- Turbo certificates are high-leverage, high-risk instruments (barrier breach = total loss).
- Past performance does not indicate future results.

Use at your own risk. Consult a licensed financial advisor before trading.
