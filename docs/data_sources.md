# TurboEdge-DE Data Sources

This document describes the status, capabilities, and limitations of each data source used by TurboEdge-DE for product discovery, pricing, and underlying quotes.

## Data Source Status Table

| Source | Status | Purpose | Adapter | Key Limitations |
|--------|--------|---------|---------|-----------------|
| **BNP Paribas** (`derivate.bnpparibas.com`) | WORKING | Primary live product quotes (ISIN, WKN, bid/ask, financing levels, KO barriers) | `adapters/issuer_feeds.py` (`BnpParibasTurboAdapter`) | Page size 1000. **BEFUND 1 (2026-09-13 derivativeTypeId census):** the prior `derivativeTypeIds` filter (7/9/23/24 + zero-volume reserved ids) covered only ~4,304/4,340 of DAX's live turbo/mini book and ~2,124 of Nasdaq 100's — a rate-limited (<=1 req/s) exhaustive bisection scan of ids 1-699 found 26 populated ids total; two more (67/68, BNP's "Unlimited Turbo" brand, structurally identical open-end turbos, live-verified) were added, lifting coverage to ~11,529 DAX / ~10,167 Nasdaq 100 rows. The other 20 discovered ids (Optionsscheine, Faktor-Zertifikate, Discount/Bonus/Express families — no knock-out barrier) are deliberately excluded, each documented with its live structural evidence in `_BNP_DERIVATIVE_TYPE_CATALOG`. This pushed both underlyings' totals past an undocumented BNP backend limitation found the same session: `offset>=10000` hard-fails with HTTP 500 (confirmed live, independent of `derivativeTypeIds`/`limit`) — `_fetch_all_pages` now treats an HTTP error on any page after the first as this ceiling and returns the partial result (`bnp_partial_universe` WARN) instead of raising and discarding already-fetched pages; ~1,529 DAX / ~167 Nasdaq 100 rows beyond row 10,000 are structurally unreachable via this endpoint as it exists today. `ask` key absent outside trading hours (key missing, not `null`); `bidDate`/`askDate` are Europe/Berlin local (no UTC offset); dated `TURBO_CLASSIC` variant not yet observed in fixtures. **BEFUND 1, W1 (2026-09-11 scan review):** `first.price` (the underlying reference price) is batched/throttled independently of per-product `bid`/`askDate` — a live DAX pull found only 1-2 distinct `first.price` values across ~4300 rows despite continuously updating individual quotes, so a fresh product quote timestamp does **not** imply a fresh reference price. `first.price` carries its own timestamp `first.priceDate` (same naive Europe/Berlin convention) plus a same-day flag `first.isPriceToday`; the adapter now parses this into `ProductSnapshot.underlying_price_ref_timestamp` and `pipeline/scan.py._resolve_spot` validates freshness/deviation against the cross-issuer consensus before ever using `underlying_price_ref` as the pricing spot. |
| **Citi / CitiFirst** (`de.citifirst.com`) | WORKING — master data + closing reference prices only; no live quotes observed during trading hours on 2026-09-11 | Secondary/cross-validation source for master data (ISIN, financing level, KO barrier, ratio); NOT currently a live-quote source | `adapters/issuer_feeds.py` (`CitiFirstTurboAdapter`) | First page only — response caps `items` at 25 regardless of `totalElementsCount` (25/33 DAX products observed live; `healthcheck()` reports WARN `citi_partial_universe`); underlying mapping (`_CITI_UNDERLYING_ISINS`) covers DAX only; direction/subtype filter field name unresolved; a market-hours pull (2026-09-11, ~08:51 UTC, well inside Xetra hours) found `referencePriceMethod == "Closing Price"` and `ask == 0.0` on all 25/25 returned DAX rows — i.e. this endpoint is returning end-of-day/reference pricing, not a live two-way market, at least for DAX. The adapter now treats any row with a non-live `referencePriceMethod` (currently just `"Closing Price"`, the only value ever observed) as having no live quote at all: `bid`/`ask` forced to `None` regardless of their numeric value, `quote_presence=False`, `is_stale=True`, low `quality_score` — master data (financing level, KO barrier, ratio, ISIN, ...) is still populated and usable for cross-issuer universe purposes. robots.txt explicitly permits `/citi/v1/*` |
| **gettex** (`gettex.wsd.com`, Boerse Muenchen) | WORKING — live bid/ask, multi-issuer (BNP Paribas, Goldman Sachs, HSBC, UniCredit observed live); no `ratio`/`currency`/`maturity`/`quanto` field exists on any endpoint (Round 3 follow-up 9a, confirmed absent for every issuer via 4 disproven hypotheses) | Cross-issuer live bid/ask + master data (DAX, Nasdaq 100, S&P 500, Euro Stoxx 50) with `ratio` **derived and verified**, not sourced, from `leverage`/`financingLevelRefCurAbsolute`/bid-ask (see adapter docstring) | `adapters/gettex.py` (`GettexAdapter`) | Since `ratio` is not provided, it is reconstructed per product: (1) a robust, MAD-outlier-filtered, long/short-balanced median of `leverage`-implied spot (`S = F*leverage/(leverage∓1)`, ratio-independent), restricted to fresh rows with leverage <= 200 for numerical conditioning, gives a reference spot `S_ref` -- live-verified against gettex's own independently-reported `underlyings.price` to within 0.012% (2026-09-12/13 validation session, DAX) -- **`S_ref` is always derived this way, never dependent on any external cross-check existing**; (2) `ratio_raw = mid*fx/\|S_ref-F\|` is snapped to a canonical Bezugsverhaeltnis grid, (3) the snapped ratio must reproduce `S_ref` within an implied-spot tolerance via the exact algebraic inverse, else the product is dropped (`ratio_rejected`/`verification_failed`/`quanto_ambiguous`, all counted and logged, never silently imputed — CLAUDE.md rule 29). **BEFUND 2 (2026-09-13 measurement session):** a live 2000-row DAX/Nasdaq-100 diagnostic found derivation rates of 35.1%/5.0% respectively (matching the reported 56.5%/3.3% within session-to-session variance) and two systematic, fixable causes: (a) the FIXED 3% ratio-snap tolerance rejected high-leverage rows disproportionately (leverage 50-75: 0.2% pass rate) purely from numerical conditioning (moneyness shrinks with leverage) even though 97-99.7% of them verify correctly once the snap step is leverage-scaled — fixed via `_leverage_scaled_ratio_snap_tolerance` (widens with leverage, capped at 50%), paired with `_leverage_scaled_verification_tolerance` (tightens the *other* direction, two-hypothesis path only) after a contract test caught a false-accept edge case this combination alone would otherwise permit at very high leverage; (b) the generic `ProductSourceAdapter.fetch_products(underlying_ids)` contract never passes this adapter's `fx_hint` kwarg, so a non-EUR underlying was permanently limited to the quanto-only hypothesis — fixed via `GettexAdapter._learn_fx_by_issuer`, which learns a per-(issuer, underlying) non-quanto fx candidate from the *same* fetch's own data (grid-ratio inversion + MAD-filtered median) and only uses it once a live re-verification shows it explains a **majority** of that issuer's own previously-ambiguous rows (live-measured: BNP Paribas/HSBC/Goldman Sachs's NDX groups recovered at 92-99.6%; UniCredit's group converged on a wrong candidate and was correctly rejected by the majority gate). Combined: NDX derivation rose from 5.0% to 85.6% on the same 2000-row live sample. **BNP-outage resilience (2026-09-13 integration finding):** the scan pipeline's only `reference_spot` cross-check source was BNP's own live spot — a BNP outage silently zeroed gettex entirely even though `S_ref` never depended on BNP. `fetch_products` now accepts an additional `daily_close_reference` (e.g. yfinance daily close) used only when `reference_spot` is absent, checked at a wider (2%, vs. the live-quote check's 0.5%) tolerance appropriate for a prior-session close. `maturity` is never invented — `open_end`/`TURBO_OPEN_END` is only set once barrier==financing_level *and* the row survives verification, else `product_type=UNKNOWN`. Page size capped at `rowsPerPage=100`; `max_pages` (20 by default) covers a documented partial subset of DAX's ~13,855-row open-end-turbo book (`healthcheck()` reports WARN `gettex_partial_universe` with covered/total). No auth, no robots.txt restriction found. |
| **Börse Stuttgart KO-Finder** (`boerse-stuttgart.de`) | BLOCKED — not implemented | Turbo/KO master data (would provide ISIN, strike, barrier, ratio, maturity) | None. No adapter exists in code or in `adapters/registry.py`; `boerse_stuttgart` is only an `enabled: false` placeholder entry in `configs/sources.yaml` documenting why it is blocked | Cloudflare bot-management blocks all requests (even honest bot UA), including `robots.txt`. Historical SSR fixtures show schema but lack bid/ask (absent from payload entirely). No bypass attempted. |
| **Börse Frankfurt / Deutsche Börse** (`api.boerse-frankfurt.de`) | BLOCKED — not implemented | Turbo/KO master data and live quotes (would provide comprehensive data) | None. No adapter exists in code or in `adapters/registry.py`; `deutsche_boerse` is only an `enabled: false` placeholder entry in `configs/sources.yaml` documenting why it is blocked | API requires salted hash headers (`X-Client-TraceId`, `X-Security`) computed from obfuscated JavaScript. Requests without these headers receive HTTP 403 without the signature headers (this is the API gateway's own check, not a CORS preflight failure). No hash reproduction attempted. |
| **Deutsche Börse Cash Market / Xetra** (`www.deutsche-boerse-cash-market.com`) | NOT FOUND | Instrument master lists (cash-market equities, ETFs, bonds, futures) | N/A | Cash-market side does not cover exchange-traded structured products (turbos/KOs trade on Zertifikate-Börse venues, not Xetra cash market). Instrument CSV/sitemap confirmed reachable but certificate/warrant/turbo sitemap does not exist. |
| **HSBC** (`hsbc-zertifikate.de`) | NOT FOUND | KO product finder | None | Vaadin server-push application with stateful UIDL protocol. No REST/JSON endpoint. No product data embedded in HTML. |
| **Société Générale** (`sg-zertifikate.de`) | NOT FOUND | KO product finder | None | Angular SPA shell (15.8 KB) with zero embedded data. No API endpoint identified within research budget. |
| **UniCredit onemarkets** (`onemarkets.de`) | NOT FOUND | Leverage product finder | None | Broken/empty JavaScript assets (`improvedLeverageSearch.min.js`, `all-in-one.min.js` return 20-byte gzip stubs). No working endpoint found. |
| **DZ Bank** (`dzbank-wertpapiere.de`) | NOT FOUND | KO-Map widget | None | CMS component-render pattern. No JSON endpoint or server-rendered product data identified. |
| **Vontobel** (`zertifikate.vontobel.com`) | NOT FOUND | Product finder | None | JavaScript-rendered SPA shell. No embedded product data, no endpoint identified. |
| **Morgan Stanley** (`zertifikate.morganstanley.com`) | UNREACHABLE | Lever/KO products | None | Connection-level non-response (zero bytes, no HTTP status). Possible firewall block or IP filtering. No bypass attempted. |
| **ING** (`www.ing.de`) | UNREACHABLE | General banking domain (specific "ING Markets" leverage-products microsite not identified) | None | Connection-level timeout (zero bytes). Note: this is the general retail domain; dedicated leverage-products domain not confirmed. Worth re-checking if that separate domain exists. |
| **UBS KeyInvest** (`keyinvest-de.ubs.com`) | NOT ATTEMPTED | Leverage certificates | None | robots.txt explicitly disallows `/api/v2/page-api`, `/api/v2/page-api/*`, `/product/list/export*` (all known product-list paths). Policy: disallowed paths not probed. |
| **Deutsche Bank X-markets** (`xmarkets.db.com`) | NOT FOUND | Legacy ASP.NET WebForms leverage products | None | robots.txt permissive. Site is legacy stateful postback UI (impractical to script without full browser session). No product-finder link identified on homepage. Budget-limited further search. |
| **yfinance** (`query2.finance.yahoo.com`) | WORKING | Underlying daily OHLC bars (DAX, Nasdaq 100, Gold, EUR/USD) | `adapters/fallback_prices.py` | Unofficial/inferred (no published API contract). Each ticker's timezone differs (DAX = Europe/Berlin, NDX/GC=F = America/New_York, EURUSD=X = Europe/London) — must explicitly convert to UTC. Volume unreliable for indices/FX (frequently 0). Does not provide intraday bars. |
| **ECB €STR Data API** (`data-api.ecb.europa.eu`) | WORKING | Euro short-term rate (EST, reference rate for financing spread inference) | `adapters/ecb.py` | SDMX-JSON format (non-standard, complex date/value zipping). Daily business-day rate only (no intraday). Dates carry `+02:00` offset (CEST) — convert to UTC. |
| **CSV Import** (`state/imports/products/`) | WORKING | Manual product upload (legitimate fallback when automated sources unavailable) | `adapters/csv_import.py` | User-provided data (no validation of source). Requires manual curation. An absent/empty import directory is the everyday default (nobody uploaded anything) and reports `healthcheck()` WARN, never FAIL — this is an OPTIONAL source (see `monitoring/source_health.OPTIONAL_SOURCES`) that must never by itself trigger `sources health --email-on-fail`/`--fail-on-error`; FAIL is reserved for files that are actually present but unreadable or entirely unparseable. See `docs/csv_import_format.md` for schema. |

---

## Principles & Architecture

### Adapter Encapsulation
Every data source is accessed exclusively through an adapter module implementing the `DataSourceAdapter` protocol (`adapters/base.py`). No direct HTTP calls to sources exist elsewhere in the codebase. This enforces:
- Centralized retry/backoff policy (via `tenacity`)
- Uniform rate limiting (≤1 request/second per host)
- Consistent error handling and health-check reporting
- Easy mocking for tests (via `respx`)

### Contract Tests with Real Fixtures
Each adapter with working data includes:
- Fixture files (JSON/CSV captures from actual API responses or exported product lists)
- Contract tests asserting schema assumptions (e.g., presence of required fields, numeric value ranges, timestamp formats)
- No fabricated data in fixtures — only real responses, with PII/credentials redacted if necessary

### Honest User-Agent
The live product adapters (BNP Paribas, Citi, gettex) use the string configured in `configs/sources.yaml`:
```
TurboEdge-DE-Research/0.1 (+https://github.com/pcctradinginc-alt/turboedge-de)
```
Reference/fallback sources (ECB, yfinance, CSV import) and the two disabled placeholder
entries (`deutsche_boerse`, `boerse_stuttgart`) use a shorter research-only variant, also
from `configs/sources.yaml`:
```
TurboEdge-DE-Research/0.1 (+research-only; no-execution)
```
No browser impersonation, no TLS fingerprint tricks, no anti-bot token replication.

### Rate Limits & Pacing
Configured per-source in `configs/sources.yaml`:
- `min_interval_s`: minimum seconds between requests to the same host
- Default: ≤1 req/s, enforced by `HttpClient` with backoff via `tenacity`
- Different hosts can be queried in parallel (bursts across hosts are acceptable)

### Access-Control Respect
- **Cloudflare or salted-hash gated:** No bypass attempted. Stuttgart and Frankfurt have no adapter at all — they are `enabled: false` placeholder entries in `configs/sources.yaml` with no corresponding code in `adapters/registry.py`, not a built adapter that reports FAIL.
- **robots.txt disallowed:** Paths not probed (UBS KeyInvest)
- **Connection-level blocks:** Not re-attempted from different IPs or with browser UA (Morgan Stanley, ING)

### Source Health Before Every Scan
`turboedge sources health` is called at the start of every `turboedge scan` pipeline run. Provides:
- Availability: % of data present in last 20 days
- Freshness: age of the most recent quote (hours/minutes since `quote_timestamp`)
- Missingness: % of null values in key fields per source
- Schema consistency: 0 violations in recent records
- Cross-source agreement: ± divergence on overlapping products (e.g., BNP vs. Citi same ISIN)
- Overall status: PASS/WARN/FAIL

If a source reports FAIL, `scan` continues (affected products are excluded via the per-product `data_health_pass` gate rather than the whole run aborting) — `scan` itself never exits with a health-related code. `turboedge sources health --fail-on-error` exits code 4 when a *critical* source is FAIL — every enabled product source except the `OPTIONAL_SOURCES` (currently just `csv_import`, a user-curated manual fallback whose everyday state is "nothing uploaded"), plus `yfinance`/`ecb_estr`. The same critical-vs-optional split gates `--email-on-fail`, so an empty/absent CSV import directory on a CI runner never sends a false-alarm alert email or fails the job on its own.

### Terms-of-Service Compliance
No exhaustive legal review per source, but:
- **BNP Paribas:** robots.txt fully permissive (`Allow: /`). ToS reviewed; no anti-bot clause found.
- **Citi:** robots.txt explicitly permits `/citi/v1/*`. ToS reviewed; no anti-bot clause found.
- **yfinance:** Unofficial (no published API contract). Treated as a fallback/best-effort source.
- **ECB:** Public data API, no authentication required.

Regular compliance review recommended (see GOVERNANCE.md).

---

## Implications for Product Coverage

Spec § 5.1 prioritizes Deutsche Börse and Börse Stuttgart as authoritative sources for German turbo/KO certificates. Both are currently inaccessible via legitimate automated methods:

- **Stuttgart:** Cloudflare bot-management blocks even honest bot UA, including `robots.txt` fetch.
- **Frankfurt:** API requires salted-hash header computation from obfuscated JavaScript.

**Practical consequence:** Product discovery is limited to participating issuers with public APIs (BNP Paribas, Citi) plus manual CSV import. This introduces **selection bias**:

- Cross-issuer comparison possible only between BNP and Citi products (not a complete market picture)
- **Winner's curse & median product reference logic apply only to this subset**
- Unobserved products from other issuers (HSBC, SG, UniCredit, etc.) may exhibit different risk characteristics
- Cost decomposition (financing spread inference, issuer margin, gap premium) is calibrated on BNP/Citi data only

### Possible Paths Forward
1. **Licensed data feeds** — commercial subscriptions from Deutsche Börse, Eurex Data, or third-party consolidators
2. **Additional issuer APIs** — research and reverse-engineer remaining issuers' legitimate endpoints
3. **Manual CSV/Excel export** — users export product lists from their broker/exchange website and place in `state/imports/products/`
4. **Coordinate data sharing** — Reach out to issuers for direct data agreements (commercial or research terms)

---

## GitHub Actions Considerations

### BNP Paribas & Citi
- Plain JSON REST APIs, zero auth, zero anti-bot signature observed
- Should work fine from GitHub Actions hosted runners
- Use same `tenacity` retry/backoff and ≤1 req/s discipline
- Viable for unattended CI jobs

### gettex
- Plain JSON REST API, zero auth, zero anti-bot signature, no Cloudflare — same risk profile as BNP/Citi
- Should work fine from GitHub Actions hosted runners with the same `tenacity`/≤1 req/s discipline
- The ratio-derivation pipeline (robust median + MAD filtering across many rows) benefits from a reasonably large `rowsPerPage`/`max_pages` pull — too small a sample makes `S_ref` noisier and rejects more products via the verification gate, not a CI-environment-specific failure mode

### yfinance
- Unofficial (no published API contract)
- GitHub Actions IP ranges sometimes rate-limited in bursts by Yahoo
- Recommend modest request volume
- Fallback/best-effort source; missing data is non-fatal

### ECB
- Public data API
- No rate-limiting observed
- Reliable for CI jobs

### Stuttgart & Frankfurt
- **Do not expect success from GitHub Actions** — there is nothing to run: no adapter is registered for either source, so neither is part of `sources health` or the scan pipeline's health check today. If an adapter is ever built (a legitimate access path is found), it should be wired to report `FAIL` health cleanly rather than raise — `scan` would then continue unaffected (exit 0), and `sources health --fail-on-error` would exit 4.
- Email alerts on health-check failure recommended once such an adapter exists
