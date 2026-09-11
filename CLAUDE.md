# TurboEdge-DE – Development Guide for Claude

## Project Overview

TurboEdge-DE is a quantitative research system for identifying, evaluating, and ranking tradeable German turbocertificates (Knock-out products) across equities, FX, metals, and energy underlyings. The system produces actionable, multi-horizon trade proposals via Gmail, with robustness against survivor bias, winner's curse, look-ahead bias, and multiple-testing inflation.

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

## Current Milestone: Phase 0 + Phase 1 + Minimal Gmail Notifier

**Phase 0 – Data Feasibility & Product Snapshot**
- Deutsche Börse adapter
- Stuttgart Börse adapter
- Product normalization & deduplication
- Quote health scoring
- Financing level history
- Persistence (DuckDB + Parquet)

**Phase 1 – Pricing / Cost Engine**
- Intrinsic value (Long/Short)
- Financing spread inference from level changes
- Gap premium estimation (overnight/weekend)
- Issuer margin decomposition
- Cross-issuer comparison
- Integrity checks (bid ≤ ask, ratio > 0, barrier valid, etc.)

**Protected TSMOM Baseline (§7)**
- Lookbacks: [21, 63, 126] trading days
- EWMA volatility normalization
- Clip to [-3, +3]
- Aggregate: mean(z₂₁, z₆₃, z₁₂₆)
- Threshold: 0.5 (versionized, never silent reoptimization)
- Status: PROTECTED

**Minimal Gmail Notifier**
- SMTP SSL to smtp.gmail.com:465
- Env: GMAIL_USER, GMAIL_APP_PASSWORD, TURBOEDGE_EMAIL_TO
- Dry-run mode if credentials missing
- Jinja2 templates (plaintext)
- Deduplication via hash in DuckDB

**ACTIONABLE BLOCKED**
- ACTIONABLE category never assigned in this milestone (LCB_EV requires Path Model from Phase 3+).
- Only WATCH, REJECT, DATA_QUALITY categories used.

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

## Core CLI (Phase 0+1 subset)

```bash
turboedge --config-dir ./configs --state-dir ./state --log-format console universe [--underlying DAX]...
turboedge scan --underlying DAX [--top 20] [--report-out ./reports/scan.txt] [--json-out ./reports/scan.json]
turboedge sources health [--json-out ./reports/health.json] [--email-on-fail]
turboedge notify test
turboedge db info
turboedge position add --wkn XXXXX --qty 100 --price 4.86 --date 2026-09-10
turboedge position list
turboedge position close --wkn XXXXX --price 5.42 --date 2026-09-15
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

- All timestamps must be tz-aware UTC.
- No `pass` stubs; all functions implemented.
- Contract tests mandatory for adapter changes.
- Ambiguous bar handling: fetch intraday data or mark ambiguous; never optimistically resolve.
