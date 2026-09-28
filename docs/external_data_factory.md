# External Data Factory

Point-in-time ingestion of economic and alternative data, with automatic
determination of when a dataset becomes usable for research.

## Why this exists

The repository already had working Cboe and CFTC adapters and 93,851 rows in
`external_observations`. It had no factory. No pipeline, CLI command or
workflow step ever called those adapters — the rows came from ad-hoc scripts
run during research sessions in September 2026. Nothing scheduled them,
nothing archived the payloads they were parsed from, and if the database were
rebuilt the data would be gone and unreproducible.

So the gap this closes is not "more sources". It is: collection that runs by
itself, an archive that survives a parser bug, availability semantics that can
be justified rather than asserted, and a rule — not a person's memory — that
decides when there is enough evidence to look.

## The five timestamps

Never conflated, ever:

| field | meaning |
| --- | --- |
| `observation_time` | the period the value describes |
| `source_release_time` | when the publisher released it, if known |
| `available_at` | the earliest moment TurboEdge may use it |
| `vintage_time` | which published revision this value is |
| `retrieved_at` | when TurboEdge downloaded it |

Only `available_at` may be compared against a prediction time. Substituting
`observation_time <= prediction_time` is the single most common way a
backtest quietly reads tomorrow's newspaper, and
`features/availability.py:assert_information_available_at_prediction` exists
to stop it.

Adapters never compute `available_at` themselves. They call
`external/adapter.py:resolve_available_at`, which is the one place the rule
lives. Six adapters each inventing their own availability rule is six chances
to get it wrong, and the one that gets it wrong will be the one nobody reviews.

## Availability precision

How well the moment of availability is *known* travels with every row:

- `EXACT_TIMESTAMP` — an auditable publication timestamp exists
- `EXACT_DATE` — an official date, no time
- `CONSERVATIVE_DATE` — a documented rule that deliberately errs late
- `INFERRED` — reconstructed from defensible evidence
- `UNKNOWN` — no defensible timing exists

Strict point-in-time research accepts the first three. The 93,851 pre-existing
Cboe/CFTC rows carry no precision at all, and are therefore treated as
`UNKNOWN` — not as fine. A series is only as honest as its weakest row, so one
`UNKNOWN` observation governs the whole series.

No adapter ever invents an intraday publication time. For a date-only release
the conservative rule is explicit: start of the observation day plus the
series' declared lag, recorded in `configs/external_data.yaml`. Erring late
costs a little statistical power. Erring early invents a forecast the system
could not have made.

## Backfill classes

What a publisher's history is actually worth:

- **`HISTORICAL_PIT_SAFE`** — real vintages are exposed (ALFRED). The history
  is genuine point-in-time evidence and may be used immediately.
- **`HISTORICAL_CONSERVATIVE`** — history exists, release timing reconstructed
  by a documented conservative rule. Fine for exploration, weaker for
  confirmation.
- **`FORWARD_ONLY`** — history is silently revised in place with no vintage
  record. Only snapshots TurboEdge takes itself count. Its readiness is
  computed from the forward archive alone, and `research_usable_from` is the
  collection start, never the publisher's first value.

That last one matters more than it looks. A `FORWARD_ONLY` series with ten
years of downloadable history and five days of self-archived data is five days
of evidence. Treating the ten years as observed would be a backtest of a
machine that never existed.

## Wave 1 sources

Verified live on 2026-09-28. Licence terms were read, not assumed.

| source | access | vintages | class | status |
| --- | --- | --- | --- | --- |
| ECB Data Portal | SDMX-JSON, no key | no | `HISTORICAL_CONSERVATIVE` | live |
| Deutsche Bundesbank | SDMX, **`Accept: text/csv` only** | no | `HISTORICAL_CONSERVATIVE` | live |
| EU Business & Consumer Surveys | JSON-stat 2.0 via Eurostat | no | `HISTORICAL_CONSERVATIVE` | live |
| Destatis truck-toll index | XLSX, weekly Thursdays | no | `FORWARD_ONLY` | live |
| FRED / ALFRED | JSON, API key | **yes** | `HISTORICAL_PIT_SAFE` | needs `FRED_API_KEY` |
| e-Stat (Japan) | JSON, application ID | no | `HISTORICAL_CONSERVATIVE` | needs `ESTAT_APP_ID` |

Two things worth knowing before debugging an adapter:

- The Bundesbank API answers **HTTP 406** for SDMX-JSON. Only `text/csv` is
  accepted, semicolon-separated, UTF-8 BOM, German decimal commas.
- `www.destatis.de/robots.txt` sets **`Crawl-delay: 30`**. That is honoured,
  not worked around.

FRED is the only source with genuine vintages, which is what makes it the only
one whose *history* is point-in-time safe. e-Stat is not in the written
specification; it was added on the user's direct instruction.

### Credentials

Both keys are optional and neither is ever handled by Claude. A source whose
variable is unset reports `AUTH_MISSING` and is skipped cleanly — not attempted
and failed.

```bash
gh secret set FRED_API_KEY
gh secret set ESTAT_APP_ID
```

Free registration: [FRED](https://fredaccount.stlouisfed.org/apikeys),
[e-Stat](https://www.e-stat.go.jp/api/). No credential is ever written to the
database, to a log line, or into an archived request fingerprint — `RawPayload`
rejects a fingerprint whose credential parameter still carries a value.

## Raw archive

Every response is written to `state/raw/<source>/date=YYYY-MM-DD/<hash>.<ext>`
**before** it is parsed, so a parser that crashes still leaves the evidence
behind. The file name is the SHA-256 of the bytes, which makes refetching an
unchanged file a no-op and a genuine revision a second file rather than an
overwrite. Reading one back verifies the hash: reparsing bytes that silently
changed would produce a "corrected" history that never existed.

Measured: ~950 KB for one Destatis XLSX, a few KB per ECB series response.

## Readiness

`external/readiness.py` answers "can this data safely be used yet?" from the
data's own properties, and from nothing else. Nothing in the module can see a
return, a P&L or a forecast.

States: `DISABLED`, `COLLECTING`, `SCHEMA_VALIDATED`, `PIT_VALIDATED`,
`EXPLORATORY_READY`, `VALIDATION_READY`, `CONFIRMATION_READY`,
`FORWARD_MATURE`, `DEGRADED`, `BLOCKED`.

The order of checks is deliberate: **quality and point-in-time integrity
before sample size.** A large sample whose availability is unknown is not weak
evidence, it is inadmissible evidence, and reporting it as "nearly ready"
would invite exactly the wrong fix — collect more of it.

Thresholds live in a frozen, versioned `ReadinessPolicy`. Every stored record
carries the `policy_version` that produced it. A threshold that can be edited
in place is a threshold that can be lowered the week an alpha needs it.

### Effective sample, not row count

Readiness counts distinct observation periods, never rows. Three vintages of
one month are one month of evidence; 500 daily rows containing eight ECB
decisions are closer to eight than to 500.

This is not a theoretical concern here. The weekly tournament reported three
significant signal families at n = 60,774 until the denominator was recomputed
properly; the same data gave 12,324 effective observations and nothing
significant (`docs/measured_results.md` §6.12).

### Maturity levels

`COLLECTING` (0) → `EXPLORATORY_READY` (1) → `VALIDATION_READY` (2) →
`CONFIRMATION_READY` (3) → `FORWARD_MATURE` (4).

There is no level 5 in this module. Production eligibility is not a data state:
it additionally requires a validated alpha, confirmation, positive economic
Turbo evidence, a forward shadow and the Evidence & Risk Gate. Data alone can
never create it.

## Research triggers

When a series first reaches a milestone, exactly one event is emitted, keyed on
a dedup key that excludes the timestamp. A daily re-emission would flood the
research queue with duplicates of one question and inflate the multiple-testing
denominator in `GOVERNANCE.md` §11.2 — which is how a real finding gets buried
under its own notifications. A demotion emits nothing, and a recovery does not
re-emit.

The trigger is handed to `ResearchQueue.seed_from_data_readiness`, which creates
a `ResearchOpportunity` in `PROPOSED` and nothing further. It starts no
research, consumes no research budget unit, and touches no `CONFIRMATION` or
`FORWARD` partition. A human still approves.

**A ready dataset is not a found alpha.** A dataset can be perfectly usable and
carry zero predictive value. That is a normal outcome and the expected one.

## Commands

```bash
turboedge external sources
turboedge external ingest
turboedge external readiness --json-out reports/eod/readiness.json
turboedge external triggers
```

`external ingest` runs daily in the `eod` job of `.github/workflows/pipeline.yml`,
with `continue-on-error`: an external publisher being down for an evening must
not fail the job that also labels the ledger and compacts retention. A failing
series is recorded per-series and the others still collect.

## Storage

Additive only. Nothing existing was rebuilt.

- `external_observations` gained four nullable columns: `source_release_time`,
  `vintage_time`, `availability_precision`, `revision_index`. The primary key
  is unchanged, so revisions still coexist by `available_at` and the 93,851
  existing rows remain readable.
- `external_sources` — the manifest, one row per source.
- `raw_payloads` — the archive index.
- `data_readiness` — append-only evaluation history, keyed on
  `(source, series_id, evaluated_at)`. A later, more permissive policy cannot
  erase the stricter answer it replaced.
- `research_triggers` — one row per emitted event, keyed on its dedup key.

`configs/external_data.yaml` is deliberately **not** part of `config_hash`.
Adding a macro series must not change the hash stamped on every run and every
`SignalSnapshot`: that hash is how two runs are compared for reproducibility,
and a new data series does not make an old forecast irreproducible.

## Deferred

Wave 2 (ENTSO-E, gas storage AGSI/ALSI, EIA, Kiel Trade Indicator, IMF
PortWatch) and Wave 3 (GDELT, Wikimedia analytics, ECMWF) are design-only. The
adapter protocol in `external/adapter.py` is what they will implement; nothing
about them is built, and no alpha test involving them has been run.

Cboe and CFTC are **not** reopened. Their measured results stand as research
history and are not reinterpreted, re-run or recombined with Wave 1 data.
