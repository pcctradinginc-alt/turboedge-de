# Research Data Factory: Point-in-Time Intraday Archive Integration

2026-11-08. Research only; no confirmed alpha, no production trading.

## The verified problem

The full scan Parquet archive existed but was never restored in normal research jobs; scan artifacts expire after 90 days. The deep EOD archive was persisted only to mutable DuckDB, whose five-day compactor removes intraday history for most products. Simply pointing discovery at the original Parquet directory would therefore have returned no durable observations.

## Solution: a narrow table within the existing archive

The EOD deep product fetch now uses the existing Parquet writer for a tiny, immutable, typed tape under state/snapshots/research_product_quotes. One pre-chosen, independently timestamped, comparable DAX LONG cross-issuer pair may enter every three calendar days. Real follow-up quotes of both tracked ISINs are preserved for up to 48 hours. Entry and follow-up records carry the cohort date and a frozen entry/follow-up role. No other products are duplicated into the rolling research tape.

The existing encrypted lean state pack/restore now includes ONLY this research tape subtree in addition to DuckDB, registry, ledger and trials. The large full-scan Parquet archive remains excluded and continues to use independent incremental encrypted artifacts. No second database, new trading path, new API, altered scanner depth, or live market action is introduced.

The production discovery CLI reads this tape through the existing ParquetResearchArchive.read_as_of implementation. It cannot silently fall back to the five-day-compacted DuckDB product quotes. All file write-times and row availability-times are checked; impossible source chronology is rejected. Follow-up quotes are never considered as new entries. Newest cohorts are omitted from validation until their full 48-hour exit period has elapsed; a mature missing exit still blocks positive evidence.

A weekly audit counts archive files, quote rows, source days, issuers, independent entry cohorts, complete actual Bid/KO exits, missing exits, pending exits, stale entries and temporal anomalies. The report contains no trade instruction. All existing exploratory research budgets, approval requirements, economic thresholds and promotion locks are unchanged.

## How long until 48 independent validation observations?

The engine still splits DEVELOPMENT and VALIDATION chronologically, with a two-day embargo and at least three calendar days between entry cohorts. Forty-eight validation observations therefore require roughly 288 calendar days of consistently eligible new cohorts even before accounting for weekends, missing pair matches and exits. No early statistical success may be claimed. Previously deleted high-frequency quotes cannot be recovered from DuckDB; available older encrypted scan artifacts are NOT silently relabeled as a pre-registered cohort series.

## Durability, limitations and operational controls

The narrow research tape is durable through the same rolling encrypted state chain as the rest of the system, including restores after GitHub runner restarts. This is not an independent permanent backup. GitHub's normal artifact retention is finite. For reliable multi-year recovery, configure and monitor the EXISTING optional durable state backend (docs/durable_state.md), whose EOD lean-state checkpoint now includes this tape. Full scan archives continue to be separately backed up with their original retention; there is no guarantee that expired artifacts can be reconstructed.

If an EOD deep fetch cannot produce a valid comparable issuer pair, no pair is invented. If the archive write fails, the job fails visibly. If no tape is present, discovery reports AWAIT_DATA and quality counters of zero rather than reading a different source. Purely synthetic positive software tests are not proof of economic alpha.

Acceptance checks: immutable Parquet capture, follow-up chronology, 3-day spacing, replay idempotence, strict PIT, no-DuckDB-fallback, lean encrypted state roundtrip with only the curated tape, incomplete-exit reporting, and full CI including Ruff, MyPy and pytest.
