# TurboEdge-DE

Identifies and evaluates German turbocertificates (knock-out products). **Research system — manual execution only. No broker integration, no order placement.**

---

## Measured results

**No forecast model and no tested challenger signal family currently has a
measured, statistically significant out-of-sample advantage over doing
nothing. The system currently outputs no trade suggestions, and thresholds
are not lowered to manufacture them.**

Full numbers, methods, sample sizes and dates: **[`docs/measured_results.md`](docs/measured_results.md)**.

- **Forecast models** (protected TSMOM baseline mapped to a full
  predictive distribution, plus a logistic-regression challenger):
  worse Brier score than a trivial unconditional (null) model in 20/20
  real-data walk-forward combinations tested; calibration error 5–15×
  worse in nearly every combination.
- **Six pre-registered challenger signal families** (vol-targeted sizing,
  low-vol regime gating, short-horizon reversal, VIX term structure,
  cross-asset lead/lag, turn-of-month seasonality): 78/80 measured cells
  worse than null on Brier score; 0/80 cleared Benjamini-Hochberg
  significance; 0/80 cleared realistic trading costs. All six recorded
  `dormant`.
- **Knock-out probability model:** after calibration work, the default
  path-simulation method is measurably better than before, but still
  over-predicts knock-out risk at realistic barrier distances — a known,
  conservative (not calibrated) bias, documented in
  `docs/measured_results.md` §3.

See `SIGNAL_REGISTRY.md` §3 for the per-model registry and
`GOVERNANCE.md` §11 for how these trials count against the quarterly
adaptation budget.

---

## Status

Data feasibility, cost/pricing engine, protected TSMOM baseline, forecast
engine, path/payoff simulation, product×horizon evaluation, forward
ledger, learning loop, and reporting are implemented, measured (see
above), and wired end-to-end into the CLI and the scheduled
`pipeline.yml` scan → email → label → learn loop (Build Contract v3
"integration wave" — see [CLI Usage](#cli-usage)).

**ACTIONABLE requires `lcb_ev > 0`, a computed `p_ko`, and
`cluster_risk_pass=True` (`ranking/gates.py`).** No model here currently
produces an `lcb_ev` expected to clear that gate on genuinely
out-of-sample data — the system correctly says "no trade" most or all of
the time until that changes.

---

## Architecture Overview

### Pipeline (`pipeline/scan.py`, extended by `pipeline/scan_all.py`)

1. **Source health** — every enabled adapter + ECB/yfinance checked before scanning (Master Spec §30).
2. **Underlying bars** — daily OHLC from yfinance, falls back to last stored bars on fetch failure.
3. **Forecast** (`models/forecast.py`, `directional.py`, `quantile.py`, `ensemble.py`) — per-horizon `HorizonForecast` (p_up, mean, sigma, quantiles, expected shortfall, uncertainty), ensembled from registry weights; frozen before any product quote is fetched (no look-ahead).
4. **Path simulation** (`simulation/{bootstrap,overnight,paths,barrier}.py`) — resampled price paths per direction/horizon (`vol_scaled_bootstrap` default), overnight/weekend gap modeling, barrier-touch detection.
5. **Products** — fetch from every enabled source (BNP, Citi, gettex, CSV), normalize, dedupe by ISIN.
6. **Product payoff & pricing** (`simulation/payoff.py`, `pricing/*`) — intrinsic value, financing spread, gap premium, issuer margin, cross-issuer consensus, integrity checks.
7. **Evaluation & ranking** (`ranking/{ev,shrinkage,lcb,utility,sizing,cluster}.py`) — net-EV per product × horizon, winner's-curse shrinkage, LCB, utility ranking, Kelly sizing, cluster-risk limits; `ranking/gates.py` assigns WATCH / REJECT / DATA_QUALITY / ACTIONABLE.
8. **Forward ledger** (`learning/ledger.py`) — every ACTIONABLE plus a stratified shadow sample of rejected candidates is recorded, independent of selection bias.
9. **Persist + report** — DuckDB, Parquet snapshot, console table, optional email.

### Learning loop and reports

`learning/labeler.py::label_due_entries` labels matured ledger entries
from real bid quotes (never optimistic on an ambiguous path);
`StrategyPosterior` (Normal-Inverse-Gamma per signal_family × horizon)
and `ModelRegistry` update model weights from realized outcomes;
`PageHinkley` detects drift; `learning/failed_hypotheses.py` records
negative results permanently (`state/registry/failed_hypotheses.json`) so
they are not silently retried; `learning/trials.py` enforces the
quarterly adaptation budget (`GOVERNANCE.md` §1.2).
`reporting/monthly.py` produces a report even with zero trades ("no
signal beat null this month" is a valid, expected output);
`reporting/weekly.py` compares champion/challenger/dormant models.

### Modules

| Package | Key files | Purpose |
|---|---|---|
| `cli.py`, `cli_state.py`, `config.py` | — | typer app (see CLI Usage); YAML→pydantic config, `config_hash()` |
| `adapters/` | `issuer_feeds.py`, `gettex.py`, `ecb.py`, `fallback_prices.py`, `csv_import.py`, `registry.py` | BNP (live bid/ask), Citi (master data only), gettex (multi-issuer, derived+verified ratio), ECB rate, yfinance, CSV import |
| `models/` | `forecast.py`, `directional.py`, `quantile.py`, `calibration.py`, `ensemble.py`, `challengers.py`, `protected_baseline.py` | `ForecastModel` protocol; `NullModel`/`TsmomForecastModel`/`LogisticDirectionModel`/`RidgeReturnModel`; 6 challenger families (dormant); `tsmom_horizon_norm_v1` (protected) |
| `features/`, `backtest/` | `returns.py`, `volatility.py`, `trend.py`, `cross_asset.py` / `purged_cv.py`, `walkforward.py`, `metrics.py`, `significance.py` | Feature engineering; purged/embargoed walk-forward CV, BH/DSR significance |
| `simulation/` | `bootstrap.py`, `overnight.py`, `paths.py`, `barrier.py`, `payoff.py` | `simulate_paths` (block/regime/monte_carlo/vol_scaled bootstrap), gap modeling, barrier-touch, `simulate_product_payoff` |
| `pricing/` | `intrinsic.py`, `financing.py`, `gap_premium.py`, `issuer_margin.py`, `cross_issuer.py`, `integrity.py` | Cost decomposition |
| `ranking/` | `ev.py`, `shrinkage.py`, `lcb.py`, `utility.py`, `sizing.py`, `cluster.py`, `liquidity.py`, `gates.py` | Net-EV, shrinkage, LCB, utility ranking, Kelly sizing, cluster risk, WATCH/REJECT/DATA_QUALITY/ACTIONABLE gates |
| `learning/` | `ledger.py`, `labeler.py`, `counterfactual.py`, `posterior.py`, `registry.py`, `ensemble_weights.py`, `trials.py`, `drift.py`, `failed_hypotheses.py` | Forward ledger, labeling, posteriors, model registry, drift, trial budget, hypothesis graveyard |
| `positions/` | `ledger.py`, `reevaluate.py` | Manual add/list/close; HOLD/REDUCE/EXIT/INVALIDATED (library code) |
| `reporting/` | `monthly.py`, `weekly.py`, `html.py`, `console.py`, `redaction.py` | Monthly/weekly reports |
| `state/`, `storage/` | `crypto.py`, `archive.py` / `schemas.py`, `duckdb.py`, `snapshots.py` | AES-256-GCM/scrypt pack/unpack; pydantic v2 schemas, DuckDB, Parquet |
| `universe/`, `monitoring/`, `notifications/` | `underlying_map.py`, `classify.py`, `filters.py`, `discover.py` / `source_health.py` / `gmail.py`, `templates.py` | Product universe; source health; Gmail notifier |

---

## Data Sources

| Tier | Sources | Status |
|------|---------|--------|
| Live products | BNP Paribas — full DAX coverage, live bid/ask | Working |
| | Citi/CitiFirst — master data and financing levels only | No live quotes (`referencePriceMethod = "Closing Price"`, `ask = 0.0` observed during market hours); bid/ask set to `None` |
| | gettex (BNP Paribas, UniCredit, Goldman Sachs, HSBC observed) | Working; ratio derived + independently verified (`docs/measured_results.md` §4) |
| Underlyings/rates | yfinance (daily OHLC), ECB €STR (SDMX-JSON) | Working |
| Manual | CSV import from `state/imports/products/` | Working |
| Not implemented | Börse Stuttgart, Börse Frankfurt | Blocked (Cloudflare bot management / salted-hash JS headers); neither bypassed; `enabled: false` docs-only entries in `configs/sources.yaml` |

Full per-source write-up, pitfalls and issuers checked and found unusable:
`docs/data_sources.md`.

---

## Setup

### Requirements

**Python 3.12+**, **uv** package manager (≥0.4.0).

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

YAML config files live in `configs/` (loaded via pydantic; every module
keeps its own defaults, these files are the integration-wave wiring
surface — Build Contract v3 §A):

| File | Purpose |
|------|---------|
| `default.yaml` | Log format, default top-N, default horizon, HTTP timeouts |
| `sources.yaml` | Per-source settings (base URL, timeout, rate limit, user agent, `enabled`) |
| `universe.yaml` | Underlying universe + product filters |
| `gmail.yaml` | Email template paths, sender/recipient fields |
| `governance.yaml` | Adaptation budget, ladder rules, ruin metric |
| `risk.yaml` | Kelly fraction, position caps, gate thresholds, `default_financing_spread` |
| `models.yaml` | TSMOM threshold (protected `tsmom_horizon_norm_v1`) |
| `forecast.yaml` | Model set, `min_train`, embargo, horizon ladder for the forecast engine |
| `simulation.yaml` | `n_paths`, path method (`vol_scaled_bootstrap` default), `block_size`, `lookback_days`, seed |
| `ranking.yaml` | `ev`/`utility`/`sizing`/`cluster` sub-sections (LCB z-scores, utility weights, Kelly caps, cluster limits) |
| `learning.yaml` | Posterior priors, ensemble-reweighting `eta`/`w_min`, trial budget |
| `reporting.yaml` | Report paths, thresholds for "insufficient data" |
| `state.yaml` | State retention (`db compact` keep-days) |

**Environment variables:**
- `TURBOEDGE_CONFIG_DIR` — config directory (default `./configs`)
- `TURBOEDGE_STATE_DIR` — state directory (default `./state`)
- `TURBOEDGE_LOG_FORMAT` — `console` or `json`
- `GMAIL_USER`, `GMAIL_APP_PASSWORD`, `TURBOEDGE_EMAIL_TO` — email secrets
- `TURBOEDGE_STATE_KEY` — passphrase for `turboedge state pack`/`unpack`
- `TURBOEDGE_PUBLIC_LOGS` — `1` redacts ISIN/WKN/prices from stdout/logs (set by `pipeline.yml`, since this is a public repository)

---

## CLI Usage

All commands accept the global options `--config-dir`, `--state-dir`,
`--log-format` before the subcommand.

All checked against `uv run turboedge <cmd> --help`:

```bash
turboedge universe [--underlying DAX]... [--source all|bnp_paribas|citi|gettex|...]
turboedge scan --underlying DAX [--direction long|short] [--horizon 3d|5d|7d|10d|14d] \
                [--top 20] [--report-out PATH] [--json-out PATH] [--email]
turboedge scan-all [--underlying DAX ...] [--email] [--json-out PATH]     # every active underlying, full forecast/EV pipeline
turboedge label                                                            # label matured forward-ledger entries
turboedge learn                                                            # update posteriors, ensemble weights, drift
turboedge forecast --underlying DAX                                        # diagnostic: fit models, print per-horizon forecast
turboedge backtest [--underlying DAX]                                      # walk-forward evaluate every model, persist results
turboedge sources health [--json-out PATH] [--email-on-fail] [--fail-on-error]
turboedge notify test
turboedge position add --wkn X [--isin Y] --qty 100 --price 4.86 --date 2026-09-10
turboedge position list [--status open|closed]
turboedge position close --wkn X --price 5.42 --date 2026-09-15
turboedge position reevaluate [--email]                                    # HOLD/REDUCE/EXIT/INVALIDATED per open position
turboedge report monthly [--month YYYY-MM] [--email]
turboedge research tournament [--email]                                    # weekly champion/challenger/dormant comparison
turboedge db info
turboedge db compact [--keep-days N]
turboedge state pack --out state.tar.enc
turboedge state unpack --in state.tar.enc
```

---

## Pipeline modes and schedule (`.github/workflows/pipeline.yml`)

One workflow, `concurrency: turboedge-state` (`cancel-in-progress: false`),
covers every state-changing job. Mode is resolved from the triggering
cron expression, or from the `mode` input on manual `workflow_dispatch`.

| Mode | Schedule (UTC, Mon–Fri unless noted) | Steps |
|---|---|---|
| `scan` | 07:40, 10:10, 13:45, 15:50, 18:10 | `turboedge scan-all --email` |
| `eod` | 21:40 | `turboedge label` → `turboedge learn` → `turboedge position reevaluate --email` → `turboedge db compact` |
| `weekly` | Saturday 06:00 | `turboedge research tournament --email`, then re-enable all workflows via the Actions API (60-day idle auto-disable on public repos), then a heartbeat commit if the repo has been idle 30+ days |
| `monthly` | 1st of month, 06:30 | `turboedge report monthly --email` |

State (DuckDB + snapshots/registry/ledger/trials) is restored from and
packed back into an **encrypted** GitHub Actions artifact
(`turboedge-state-enc`, `state.tar.enc`, 90-day retention) at the start
and end of every job — not the previous 7-day GitHub Actions cache. A
genuine first run starts fresh (logged); if prior successful runs exist
but none of the last 50 carry a usable artifact, the run fails loudly
instead of silently discarding history. `workflow_dispatch` with
`allow_fresh_state: true` forces a deliberate reset.

Two other workflows: **`tests.yml`** (every push/PR: ruff, mypy, pytest,
live tests excluded by default) and **`source-health.yml`** (cron Mon–Fri
06:15 UTC, source health check, email alert on FAIL). `pipeline.yml` sets
`TURBOEDGE_PUBLIC_LOGS=1` (redacts ISIN/WKN/prices from stdout/logs; no
unencrypted report is ever uploaded; job summaries carry only counts) —
this is a public repository. Scheduling jitter and, in the worst case,
the 60-day idle auto-disable mean scheduled Actions should not be relied
on alone for time-critical scans; use owned infrastructure for those.

---

## Secrets Setup

**Gmail** (email notifications): enable 2-Step Verification
([myaccount.google.com/security](https://myaccount.google.com/security)),
generate an App Password
([myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
→ "Mail" + "Other" → "TurboEdge-DE"), then set locally (copy
`.env.example`) and as GitHub Actions secrets (repo Settings → Secrets and
variables → Actions), names matching exactly:

```bash
GMAIL_USER=your-email@gmail.com
GMAIL_APP_PASSWORD=abcdefghijklmnop
TURBOEDGE_EMAIL_TO=alerts@example.com,trader@example.com
TURBOEDGE_STATE_KEY=<see below>
```

Test with `uv run turboedge notify test` — dry-run (console output) if
secrets are missing.

**`TURBOEDGE_STATE_KEY`** (encrypted state archive): a passphrase, minimum
**24 characters**, stretched via scrypt into a 256-bit AES-256-GCM key
(`state/crypto.py`). Generate one with:

```bash
python -c "import secrets;print(secrets.token_urlsafe(32))"
```

Set it locally and as a GitHub Actions secret alongside the Gmail ones
above. **Losing this key makes the encrypted state archive permanently
unrecoverable** — a wrong key on unpack fails loudly (exit 2) rather than
silently producing garbage.

---

## Data Storage

**DuckDB** (`$TURBOEDGE_STATE_DIR/turboedge.duckdb`): `instruments`,
`product_snapshots`, `underlying_prices`, `signals`, `candidate_sets`,
`source_health`, `positions_manual`, `notifications_sent`, `runs`, plus
additive tables for the forecast engine, forward ledger, and position
re-evaluation. Inspect with `turboedge db info`. **Parquet**
(`$TURBOEDGE_STATE_DIR/snapshots/<table>/date=YYYY-MM-DD/<run_id>.parquet`):
immutable, append-only. **Encrypted archive:** `turboedge state
pack`/`unpack` — what `pipeline.yml` uses between scheduled runs.

---

## Reproducibility

Every prediction is tagged with `git_commit`, `config_hash` (SHA256 of all
config YAML), and `data_snapshot_hash` (SHA256 of the underlying-bar
snapshot used) — the same commit, config, and snapshot data reproduce the
same result.

---

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

```bash
uv run pytest -q                    # live-marked tests excluded (pyproject.toml)
uv run pytest -q -m live            # include live network tests
uv run pytest -q --cov=src          # with coverage
```

**Troubleshooting:** `ModuleNotFoundError: turboedge` after `uv sync` is
usually a hidden `.venv` on macOS (this repo lives under iCloud-synced
Desktop, and `.venv` is a symlink to `.venv.nosync` to avoid iCloud
corrupting it mid-write):
```bash
chflags -R nohidden /Users/cc/Desktop/TURBO\ EDGE/turboedge-de/.venv
```

---

## Known Limitations

- **No model currently beats the null benchmark out-of-sample** — see
  "Measured results" above; the central limitation of the system, not a
  data or engineering gap.
- **P(KO) is conservative, not calibrated** — over-predicts knock-out risk
  at small-to-moderate barrier distances by roughly 40–135% relatively,
  even with the best-measured method (`vol_scaled_bootstrap`); treat it as
  an upper bound, not a calibrated probability.
- **Product data limited to BNP Paribas, Citi, and gettex.** No Stuttgart
  or Frankfurt adapter. Citi is closing-price reference data only, not
  live quotes. gettex's ratio derivation matches only 3.3% of fetched NDX
  rows (2026-09-13 validation); a direct ISIN cross-check against BNP's
  own feed found no overlap (`docs/measured_results.md` §4).
- **yfinance is unofficial** — no published API contract, best-effort.
- **Financing-spread inference needs history; gap premium does not** —
  fewer than 2 same-ISIN snapshots on different days falls back to
  `configs/risk.yaml`'s `default_financing_spread`.
- **Spot consensus is thin** — only BNP + gettex (Citi has no live
  prices); no independent intraday market-data feed.
- **State lives in a 90-day workflow artifact** — 90 days without a
  successful `pipeline.yml` run loses ledger/posterior/model history;
  deliberate reset requires `allow_fresh_state=true`.
- **2010–2026 sample is a near-uninterrupted bull market** — every
  Sharpe/PSR/DSR-style statistic in this codebase is inflated by secular
  drift and shared identically by the null model itself (see
  `docs/measured_results.md` §1–2); only paired comparisons against null
  are informative here.

---

## Roadmap

| Status | Milestone |
|---|---|
| Done | Data sources (BNP, Citi, gettex), cost engine, protected TSMOM baseline, Gmail notifier |
| Done, measured negative | Forecast engine (TSMOM-distribution, logistic, null) — no OOS edge over null |
| Done, measured negative | 6 pre-registered challenger signal families — 0/80 cells significant |
| Done | Path/KO simulation + calibration comparison, `vol_scaled_bootstrap` default |
| Done | Product×horizon EV/LCB/sizing/cluster evaluation layer |
| Done | Forward ledger, labeling, posteriors, model registry, drift detection |
| Done | Monthly/weekly reports, encrypted state, `pipeline.yml` scheduler |
| Done | CLI wiring for `scan-all`/`label`/`learn`/`position reevaluate`/`report monthly`/`research tournament`/`forecast`/`backtest` |
| Not started | A model clearing the `GOVERNANCE.md` §2 ladder on genuinely new OOS data — required before ACTIONABLE output is expected in practice |
| Not started | Additional data sources (Eurex, Euwax, Cboe, FRED, CFTC) |

No delivery dates are committed for unstarted items.

---

## License

**Proprietary** — Copyright 2026 pccTradingINC. No order execution, no broker integration, research system only. See `LICENSE`.

---

## Disclaimer

This is a research and analysis tool, not investment advice. No
historical backtest or live track record guarantees future performance;
all signals are experimental and, as measured, do not currently beat a
trivial null benchmark (see "Measured results" above). Manual human
review and risk management are essential before any trade — turbo
certificates are high-leverage, high-risk instruments (barrier breach =
total loss), and past performance does not indicate future results.

Use at your own risk. Consult a licensed financial advisor before trading.
