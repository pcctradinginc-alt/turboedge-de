# TurboEdge-DE – Development Guide for Claude

## Project Overview

TurboEdge-DE is a quantitative research system for identifying, evaluating, and ranking tradeable German turbocertificates (Knock-out products) across equities, FX, metals, and energy underlyings. Target state (Spec §54): multi-horizon trade proposals via Gmail, with robustness against survivor bias, winner's curse, look-ahead bias, and multiple-testing inflation. Current state: see "Current Milestone" below — cost/integrity analysis only, no trade proposals yet.

**Core principle:** Rigorous signal generation and product evaluation, but **no automatic execution**. The final trading decision remains human.

---

## Verbindliche Regeln (Master Spec §49, wörtlich)

1. Keine Live-Execution implementieren.
2. Keine Broker-API implementieren.
3. Produktive Ausgabe ausschließlich über Gmail/Reports.
4. Keine Look-ahead-Daten.
5. `available_at <= prediction_time` für jedes Feature erzwingen.
6. Keine zufälligen Zeitreihen-Splits.
7. Purged CV + Embargo verwenden.
8. Average Uniqueness bei überlappenden Samples berücksichtigen.
9. Jede komplexe Strategie gegen einfache Baselines testen.
10. Protected TSMOM Baseline niemals löschen.
11. Entry immer Ask, Exit immer Bid.
12. Spread immer berücksichtigen.
13. Finanzierung aus realem Financing-Level-Verlauf messen, wenn möglich.
14. Fair Gap Premium und Issuer Margin getrennt behandeln.
15. KO immer als pfadabhängiges Ereignis behandeln.
16. Overnight- und Weekend-Gaps explizit modellieren.
17. Ambiguous Bars niemals optimistisch auflösen.
18. Keine aktuellen Produkte rückwirkend historisieren.
19. Produktauswahl gegen Winner's Curse shrinken.
20. LCB statt maximalem Punktschätzer bevorzugen.
21. Medianprodukt als Counterfactual speichern.
22. Neue Features isoliert ablatieren.
23. Jede Research-Änderung bekommt eine Trial-ID.
24. Anpassungsbudget beachten.
25. Multiple Testing deflationieren.
26. Keine Verbesserung nur anhand In-Sample behaupten.
27. Jede Source hinter Adapter kapseln.
28. Parseränderungen mit Contract Tests absichern.
29. Fehlende Daten niemals still imputieren, wenn sie für Pricing kritisch sind.
30. Source Health vor jedem Scan prüfen.
31. Positive Muster verstärken, negative Muster nicht löschen.
32. Drift reduziert Gewichte, löscht aber nicht automatisch Modelle.
33. Jede Prediction vollständig reproduzierbar speichern.
34. Kein TODO-Code in einem als abgeschlossen markierten Milestone.
35. Nach jedem Milestone Tests ausführen.
36. Erst vertikalen Slice fertigstellen, dann verbreitern.

Konkretisierungen (z.B. Anpassungsbudget, Ruin-Grenzen) stehen versioniert in GOVERNANCE.md.

---

## Current Milestone: Full Pipeline Implemented — No Measured Edge Yet

Everything through the learning loop and reporting is implemented and
wired into the scheduled pipeline (`.github/workflows/pipeline.yml`):
product ingestion/pricing (Phase 0+1), the forecast/path-model/EV engine,
the forward ledger, the label/learn loop, position re-evaluation, and
monthly/weekly reporting. **ACTIONABLE is technically reachable** —
`ranking/gates.py::evaluate_gates` assigns it whenever `lcb_ev`/`p_ko`/
`cluster_risk_pass` (computed by the real EV pipeline in
`pipeline/scan.py`, which every `scan-all` run exercises) clear every
gate; it is not hard-disabled.

**In practice, no ACTIONABLE candidate is produced today, for one
reason: no forecast model has a measured, statistically significant
out-of-sample advantage over doing nothing.** See
`docs/measured_results.md` (updated 2026-09-13) — 0/20 forecast-model
cells and 0/80 pre-registered challenger-signal cells clear the ladder in
`GOVERNANCE.md` §2. The correct, deliberate behavior of the system right
now is "no trade" on every scan; thresholds are not lowered to
manufacture suggestions. This is a measured research result, not a
missing feature — see the roadmap in README.md.

**Data sources**
- Product sources: BNP Paribas issuer API (live bid/ask), Citi issuer API (master data + closing reference prices only, partial coverage), gettex, CSV import (optional, manual)
- Börse Stuttgart and Börse Frankfurt/Deutsche Börse: NOT implemented — blocked (Cloudflare bot management / anti-bot signature headers). Never bypass access protection; see docs/data_sources.md

**Pricing / Cost Engine**
- Intrinsic value (Long/Short), financing spread inference, gap premium estimation, issuer margin decomposition, cross-issuer comparison, integrity checks (bid ≤ ask, ratio > 0, barrier valid, etc.)

**Protected TSMOM Baseline (§7)**
- Lookbacks: [21, 63, 126] trading days
- EWMA volatility normalization
- Clip to [-3, +3]
- Aggregate: mean(z₂₁, z₆₃, z₁₂₆)
- Threshold: 0.5 (versionized, never silent reoptimization)
- Status: PROTECTED

**Forecast / Path / EV pipeline**
- Forecast models (TSMOM-distribution, logistic, null), path/KO simulation with calibration comparison, product×horizon EV/LCB/sizing/cluster evaluation — see `docs/measured_results.md` for measured performance of each

**Forward ledger, learning loop, reporting**
- Forward ledger (entry + counterfactual alternatives), labeling of matured entries, posterior/ensemble-weight updates, drift detection (Page-Hinkley), model registry (champion/challenger/dormant), monthly/weekly reports

**Meta layer (`src/turboedge/meta/`) — shadow only, never authoritative**
- Phase 1 asks, before "what is the best trade?", the prior question "does the
  system know enough here?": per-model trust (multiplicative, and a factor that
  cannot be computed is recorded in `missing_factors` and penalised, never
  defaulted to 1.0), six separate uncertainty axes, model disagreement, and a
  PROCEED/WATCH_ONLY/ABSTAIN decision. `shadow_mode=True` on every decision;
  nothing in `pipeline/scan.py` reads it. Against the current state it abstains
  on every horizon, because `walkforward_results` is empty and calibration has
  never been measured out-of-sample.
- Phase 2 ranks *research questions* from a fixed catalog of 14
  (`meta/catalog.py`) by a transparent value-of-information heuristic. Every
  ranking input is an `Estimate` carrying its provenance — MEASURED, DECLARED
  (a judgement, reasoning required) or UNKNOWN (penalised, never defaulted).
  **The system may reorder the queue freely and may only ever write status
  PROPOSED; every later state requires a named human via `research approve`
  (§8).** None of Phase 2's own weights are tuned against an outcome — there is
  no forward research data to tune them on, and doing so would be the parameter
  fishing GOVERNANCE.md forbids.

**Gmail Notifier**
- SMTP SSL to smtp.gmail.com:465
- Env: GMAIL_USER, GMAIL_APP_PASSWORD, TURBOEDGE_EMAIL_TO
- Dry-run mode if credentials missing
- Jinja2 templates (plaintext)
- Deduplication via hash in DuckDB

---

## Development Commands

```bash
# Install dependencies (dev extras for testing/linting)
uv sync --extra dev

# Run tests (live tests deselected by default)
uv run pytest -q

# Lint
uv run ruff check .

# Format check
uv run ruff format --check .

# Type check
uv run mypy src

# All together (CI-like)
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest -q
```

---

## Guiding Question (§53)

Before implementing any new feature, signal family, or model change, answer:

> **Does this change—after realistic cost modeling, path risk, winner's-curse correction, and multiple-testing deflation—demonstrably improve out-of-sample net Expected Value or system robustness?**

If empirically unanswerable, mark as `experimental`.

---

## Core CLI

Global options (`--config-dir`, `--state-dir`, `--log-format`) go BEFORE
the subcommand. Verified against `uv run turboedge --help` (and each
subcommand's own `--help`) on 2026-09-14 — re-check there if this list and
reality ever diverge again.

```bash
turboedge --config-dir ./configs --state-dir ./state --log-format console universe [--underlying DAX]...
turboedge scan --underlying DAX [--direction long|short] [--horizon 3d|5d|7d|10d|14d] [--top 20] [--report-out PATH] [--json-out PATH] [--email]
turboedge scan-all [--underlying DAX ...] [--email] [--json-out PATH]     # every active underlying, full forecast/EV pipeline
turboedge label                                                          # label matured forward-ledger entries
turboedge learn                                                          # update posteriors, ensemble weights, drift
turboedge forecast --underlying DAX                                      # diagnostic: fit models, print per-horizon forecast
turboedge backtest [--underlying DAX]                                    # walk-forward evaluate every model, persist results
turboedge sources health [--json-out PATH] [--email-on-fail]
turboedge notify test
turboedge position add --wkn XXXXX --qty 100 --price 4.86 --date 2026-09-10
turboedge position list [--status open|closed]
turboedge position close --wkn XXXXX --price 5.42 --date 2026-09-15
turboedge position reevaluate [--email]                                  # HOLD/REDUCE/EXIT/INVALIDATED per open position
turboedge report monthly [--month YYYY-MM] [--email]
turboedge research tournament [--email]                                  # weekly champion/challenger/dormant comparison
turboedge research queue [--limit 10] [--rescore/--no-rescore]           # ranked research priorities (shadow; no authorisation)
turboedge research approve <ID> --by WHO --note WHY [--trial-id ID]      # the human approval gate (Phase 2 §8)
turboedge db info
turboedge db compact [--keep-days N] [--hard-delete-after-days N]
turboedge state pack --out state.tar.enc [--include-snapshots]
turboedge state pack-snapshots --run-id ID [--run-id ID ...] --out PATH  # incremental per-scan Parquet snapshot artifact
turboedge state unpack --in state.tar.enc [--allow-missing]
turboedge state restore-snapshots --in state-full.tar.enc                # additive merge of state/snapshots/ only (manual/ad hoc use)
```

---

## Key Signals & Imports

- **numpy, scipy** – mathematical computation
- **polars** – fast dataframe operations
- **duckdb** – persistent state & ledger
- **pydantic v2** – schema validation
- **typer** – CLI framework
- **rich** – formatted console output
- **httpx + tenacity** – HTTP with retries & rate limiting
- **structlog** – JSON/console logging
- **jinja2** – email templates
- **yfinance** – underlying daily prices (fallback)

---

## Notes

- Local checkout lives in an iCloud-synced Desktop: `.venv` is a symlink to `.venv.nosync` (excluded from iCloud). If `ModuleNotFoundError: turboedge` appears, run `chflags -R nohidden .venv.nosync`.
- DuckDB schema changes must be additive and go through the migration in `Store.init_schema()` (CI restores old databases from the Actions cache).
- All timestamps must be tz-aware UTC.
- No `pass` stubs; all functions implemented.
- Contract tests mandatory for adapter changes.
- Ambiguous bar handling: fetch intraday data or mark ambiguous; never optimistically resolve.
- `turboedge db compact` (`state/retention.py`) permanently deletes `product_snapshots` rows older than `hard_delete_after_days` (default 90) after they've been thinned past `keep_days` (default 5) — **except** any ISIN referenced in `forward_ledger`, whether as the entry actually taken (`selected_isin`) or only as a discarded alternative/counterfactual (`alternatives`, Master Spec §21). Ledger-referenced ISINs are kept in full at any age, never thinned or deleted — required so W6's label/learn pipeline and counterfactual learning (`learning/counterfactual.py`) always have the complete path for realized P&L, KO timing, MFE/MAE and counterfactual comparison, matching rule 33's reproducibility requirement. Do not weaken or bypass this exemption when touching retention logic. Both defaults are measured (`state/retention.py`'s module docstring), not arbitrary: multi-scan-per-day resolution is only ever consumed operationally for a few days (financing-spread inference collapses to one observation/day regardless via `Store.financing_level_history`, and label/counterfactual learning only look at ledger-protected ISINs), and a 60-180-day sweep against a 21,700-ISIN/scan synthetic database showed compacted size scaling roughly linearly with `hard_delete_after_days`, with zero rows lost from any ledger ISIN and zero loss of day-level financing history adequacy at every value tested — 90 days keeps a wide margin over every actual need while cutting the old 400-day default by ~78%.
