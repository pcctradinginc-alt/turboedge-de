# CSV Import Format

The TurboEdge-DE CSV import adapter accepts user-provided product lists in comma-separated or semicolon-separated value format. This is useful when exchange websites (Deutsche Börse, Börse Stuttgart, broker portals) are not accessible via legitimate automated API access.

## File Format

- **Encoding:** UTF-8, with or without BOM
- **Delimiters:** auto-detected (semicolon `;` or comma `,`)
- **Extension:** `.csv`
- **Location:** `state/imports/products/` (relative to working directory, or absolute path)

## Required Columns

All of the following columns must be present (header row required):

| Column | Type | Format | Example |
|--------|------|--------|---------|
| `isin` | String | 12-character ISIN | `DE000ABC1234` |
| `issuer` | String | Non-empty text | `BankX` |
| `underlying` | String | Any raw underlying label (auto-resolved) | `DAX`, `S&P 500`, `EUR/USD` |
| `direction` | String | "long"/"short" or variant | `long`, `short`, `call`, `put` |
| `financing_level` | Number | Decimal (German or English format) | `18000,50` or `18000.50` |
| `knockout_barrier` | Number | Decimal (German or English format) | `17500,00` or `17500.00` |
| `ratio` | Number or Ratio | Decimal or X:Y format | `0,01`, `100:1`, `10 : 1` |
| `bid` | Number | Decimal (German or English format) | `4,80` or `4.80` |
| `ask` | Number | Decimal (German or English format) | `4,86` or `4.86` |
| `quote_timestamp` | ISO-8601 Datetime | YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS | `2026-09-10` or `2026-09-10T15:30:00` |

## Optional Columns

| Column | Type | Format | Default | Notes |
|--------|------|--------|---------|-------|
| `wkn` | String | German securities identification | – | WKN if available |
| `product_type` | String | "turbo_open_end", "turbo_classic", "mini_future" | auto-classified | Over-rides automatic classification |
| `currency` | String | ISO-4217 code | `EUR` | Product currency |
| `underlying_currency` | String | ISO-4217 code | – | Underlying asset currency |
| `quanto` | Boolean | `true`/`false`, `ja`/`nein`, `1`/`0` | – | Quanto product flag |
| `open_end` | Boolean | `true`/`false`, `ja`/`nein`, `1`/`0` | auto | Open-ended product flag |
| `maturity` | Date | YYYY-MM-DD | – | Maturity date for classic turbos |
| `first_trading_day` | Date | YYYY-MM-DD | – | First trading date |
| `bid_size` | Number | Decimal (German or English format) | – | Bid volume |
| `ask_size` | Number | Decimal (German or English format) | – | Ask volume |
| `bid_only` | Boolean | `true`/`false`, `ja`/`nein`, `1`/`0` | `false` | Only bid side traded |
| `knocked_out` | Boolean | `true`/`false`, `ja`/`nein`, `1`/`0` | `false` | Product knocked out |
| `underlying_price_ref` | Number | Decimal (German or English format) | – | Reference underlying price from source |
| `venue` | String | Non-empty text | `unknown` | Exchange or venue identifier |

## Number Format

The adapter automatically detects German and English number formats:

- **German:** `1.234,56` (period as thousands separator, comma as decimal)
- **English:** `1234.56` (no thousands separator, period as decimal)
- **Ambiguous:** `1,234.56` or `1.234,5` — rightmost separator wins

## Ratio Format

Ratios (Bezugsverhältnis) can be specified as:

- **Decimal:** `0,01` or `0.01` (both forms supported)
- **Colon notation:** `100:1` or `10 : 1` (spaces allowed)

Example mappings:
- `100:1` → ratio = 0.01
- `10:1` → ratio = 0.1
- `0,01` → ratio = 0.01

## Timestamps

ISO-8601 format required:

- **Date only:** `2026-09-10` (interpreted as midnight Europe/Berlin local time, converted to UTC)
- **Date and time:** `2026-09-10T15:30:00` (interpreted as Europe/Berlin local time, converted to UTC)
- **With timezone:** `2026-09-10T15:30:00+02:00` (explicit timezone, converted to UTC)

Naive (timezone-unaware) values are **always localized to Europe/Berlin** using the IANA tzdata (via Python's `zoneinfo`) and then converted to UTC — never a hardcoded CET (UTC+1) or CEST (UTC+2) offset, since which one applies depends on the date (DST runs late March–late October). If the `Europe/Berlin` zone cannot be loaded in the runtime environment, the timestamp is rejected as a row error rather than guessed.

## Boolean Values

Boolean fields (`quanto`, `open_end`, `bid_only`, `knocked_out`) accept:

- **True:** `true`, `1`, `yes`, `ja`, `wahr`
- **False:** `false`, `0`, `no`, `nein`, `falsch`

Empty or omitted values default to `None` (nullable).

## Underlying Resolution

The `underlying` column is matched against a canonical list of known underlyings:

**Equities:** DAX, ESTX50, SPX, NDX, NKY, UKX, SMI
**FX:** EURUSD, USDJPY, GBPUSD, EURCHF
**Commodities:** XAU, XAG, BRENT, WTI, NATGAS

The adapter uses fuzzy matching and alias resolution, so common names and variations are supported:

- `DAX` → DAX
- `DAX 40` → DAX
- `DAX® (Performance)` → DAX
- `Euro Stoxx 50` → ESTX50
- `S&P 500` → SPX
- `Gold` → XAU
- `EUR/USD` → EURUSD

If the underlying cannot be resolved, `underlying_id` is set to `None` and the product is marked `DATA_QUALITY` in evaluation.

## Direction Classification

The `direction` column is matched against keywords:

- **Long:** "long", "call", "bull", "bullish"
- **Short:** "short", "put", "bear", "bearish"

Case-insensitive. If neither or both keywords are present, direction is classified as `DATA_QUALITY`.

## Example CSV (Semicolon-Delimited, German Format)

```csv
isin;wkn;issuer;underlying;direction;financing_level;knockout_barrier;ratio;bid;ask;quote_timestamp;venue;product_type;currency;open_end
DE000ABC1234;ABC123;TestBank;DAX;long;18000,50;18000,50;0,01;4,80;4,86;2026-09-10T15:30:00;Stuttgart;turbo_open_end;EUR;true
DE000XYZ5678;XYZ567;OtherBank;S&P 500;short;4200,00;4100,00;0,01;3,50;3,55;2026-09-10;Frankfurt;turbo_classic;EUR;false
DE000QQQ9999;QQQ999;ThirdBank;EUR/USD;long;1,1000;1,0800;0,001;1,20;1,21;2026-09-10T14:00:00;Stuttgart;;EUR;true
```

## Example CSV (Comma-Delimited, English Format)

```csv
isin,wkn,issuer,underlying,direction,financing_level,knockout_barrier,ratio,bid,ask,quote_timestamp,venue
DE000ABC1234,ABC123,TestBank,DAX,long,18000.50,18000.50,0.01,4.80,4.86,2026-09-10T15:30:00,Stuttgart
DE000XYZ5678,XYZ567,OtherBank,S&P 500,short,4200.00,4100.00,0.01,3.50,3.55,2026-09-10,Frankfurt
```

## Error Handling

- **Missing required fields:** Row rejected; error logged with filename and line number.
- **Invalid ISIN:** Row rejected.
- **Unparseable numbers:** Row rejected; specific field reported.
- **Unresolvable underlying:** Product marked with `underlying_id = None`; classified as `DATA_QUALITY`.
- **Unclassifiable direction:** Product marked with invalid direction enum; rejected at validation.

All errors are recorded in the adapter's `last_errors` list and logged via structlog for debugging.

## Freshness and Staleness

- **fresh:** Quote timestamp within 900 seconds (15 minutes) of import → `quality_score = 1.0`
- **stale:** Quote timestamp older than 900 seconds → `quality_score = 0.5`, `is_stale = true`

Threshold configurable via `CsvProductImportAdapter(stale_after_s=...)`.

## Health Check

`turboedge sources health` checks every enabled source, including `csv_import` (there is no
per-adapter health subcommand). For `csv_import` specifically, it checks:

1. Import directory exists and contains at least one `.csv` file
2. Percentage of malformed rows (>50% error → WARN)
3. Quote staleness (all quotes stale → WARN)
4. Otherwise → PASS

## Import Workflow

1. Export product list from the exchange website (search for "Knock-out products", "Turbo certificates", "Zertifikate") as CSV.
2. Place the CSV file(s) in `state/imports/products/`.
3. Run `turboedge universe` or `turboedge scan` — the adapter auto-discovers all `.csv` files.
4. Check `turboedge sources health` to verify parsing success.
