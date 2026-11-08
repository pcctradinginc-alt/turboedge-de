# Market-hours Research Capture and E2E Verification (2026-11-08)

## Reason for PR #27

The Phase 2 PIT tape was collected only inside the EOD deep-fetch step, after
21:40 UTC; that time is **outside** the normal German certificate market
session. A technically successful fetch after closing is not proof of fresh,
simultaneously executable cross-issuer quotes. Running additional research
strategies against an empty or stale archive would be misleading.

## Schedule and state

The existing pipeline.yml receives a dedicated Monday-Friday cron at
**14:30 UTC**, during the German afternoon in both CET and CEST. The new
workflow mode is named research_capture; it may also be selected manually
via workflow_dispatch. Its steps restore the latest encrypted state,
execute the pre-existing deep research-only archive fetch with
--require-market-hours --monitor, publish a count-only job summary and pack
the same encrypted lean state artifact. Every state-changing mode is
serialized by the existing shared turboedge-state concurrency group,
preventing a concurrent scan or EOD from overwriting the tape.

The deep fetch is removed from the late EOD step, but EOD label/learn,
position reevaluation, compaction and its existing durable-state backend push
remain. Live scan frequency, gettex live cap, risk gates, trading permissions
and model weights do not change. Only the research deep fetch uses its
existing 150-page override, never the live scan.

The capture-mode CLI checks its session **before** fetching. Known exchange
closures (weekends, New Year's Day, Good Friday, Easter Monday, May Day,
Christmas Eve/Day/Boxing Day, New Year's Eve) are conservatively excluded.
A timestamp must also fall between 08:00 and 21:30 **Europe/Berlin**;
the timezone automatically handles daylight-saving changes. Unexpected
special exchange closures cannot be inferred from a calendar alone, so
fresh, independently time-stamped entry and exit bid/ask quotes remain
required. Closed sessions are skipped with an explicit summary, not counted
as profitable or as ordinary unsuccessful quote attempts.

A **Friday cohort entry is forbidden**: its 24-48 hour observation window
would fall on the weekend. A valid Friday quote *can* supply the observed
24-hour follow-up of a Thursday cohort. Holidays, missing ISINs, non-market
timestamps, absent Ask/Bid, crossed books and stale exit quotes never
manufacture a successful exit. A pending cohort is not a mature win.

## Operational E2E monitor

The previous quote archive is read after restoring the encrypted lean state.
A high-water marker stored at registry/research_capture_monitor.json inside
the SAME encrypted state is verified before each research fetch:
- Previously seen quote or entry counts may not silently disappear.
- A corrupted/rolled-back tape fails loudly before acquiring new quotes.
- After capture, a reported new quote must really be visible in the PIT
  Parquet archive.
- Legitimate three-day cohort spacing does not imply a broken data source:
  the monitor tracks actual fresh eligible issuer pairs separately from
  whether a new cohort was allowed to enter.
- One missing eligible pair on a market weekday is recorded; three separate
  eligible trading days without a fresh pair trigger a DEGRADED alert. Friday
  is follow-up-only and does not increment the missed-entry streak. Repeating
  the workflow on the same UTC day does not increment it again.
- Two or more observed mature missing exits trigger an alert when the count
  rises (they are NEVER classified as successful exits).
- Notifications use the existing Gmail credentials configured in Actions
  and contain only counts and the reason, no ISINs or price data. Without
  credentials or when SMTP fails, failure is logged; the job does not claim
  delivery. The evidence marker persists through the lean-state pack.

The monitor never approves hypotheses, enables confirmations, trades, or
changes risk limits. It is not proof that any alpha exists.

## Verification and limitations

Tests cover CET/CEST conversion, weekends and known closures, Thursday
entry -> Friday follow-up, rejected Friday entries, rejected weekend
follow-up, no duplicate alerts due to same-day retry, after-three-days
alerts, restoration rollback detection, and pipeline separation.

Software CI verifies this logic without relying on a live market.
The next scheduled market-hours run is needed to measure *actual* eligible
pairs and observed exit completeness. GitHub Actions cron is best-effort and
may be delayed, so runtime and quote-age gates remain strict. A holiday
not captured by the conservative closure list must still fail quote
freshness; it is not counted as proven trading availability.

All research data stay in the existing Parquet and encrypted lean-state
backup. Permanent multi-year recovery still depends on configuring and
monitoring the already-existing optional durable state backend.
