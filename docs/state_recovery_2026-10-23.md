# Recovery procedure: the eight unreachable state artifacts of 2026-10-23/24

**Status: PREPARED, NOT EXECUTED. Requires `TURBOEDGE_STATE_KEY`, which this
system never reads, exports or changes.**

Everything below has been verified against the Actions API; nothing has been
decrypted, unpacked, merged or written. The procedure stops at the first step
that needs the secret, which is step 4.

---

## 1. What happened

`.github/actions/turboedge-state` selected restore candidates with
`gh run list --workflow pipeline.yml --status success`. A run's conclusion is
the maximum over *all* its jobs, including jobs that touch no state. On
2026-10-23/24 a watchdog job inside `pipeline.yml` failed repeatedly, turning
eight runs red whose data jobs had all succeeded and all of which had
uploaded `turboedge-state-enc`.

Consequence, read from the log of the first green run afterwards
(40084459900, 2026-10-24 10:04):

```
Restored turboedge-state-enc artifact from pipeline.yml run 39956274493
```

That is 2026-10-23 12:49. The chain resumed from a day-old state, and the
eight runs' work was not carried forward. They are **siblings, not a chain**:
each restored from that same 12:49 state and added only its own work, so no
single artifact contains all of it.

The cause is fixed in two places: the watchdog now lives in its own workflow
(`.github/workflows/health-watch.yml`), and the restore no longer filters on
the run conclusion -- the artifact's existence is the stronger signal,
because "Pack and upload state" is the last step of every data job and is
gated on `if: success()`. Both are covered by tests in
`tests/test_workflow_modes.py`. **This document is only about the data
already lost.**

## 2. What exists, verified 2026-10-24

| Run | Time (UTC) | Job that succeeded | `turboedge-state-enc` | Snapshots artifact |
|---|---|---|---|---|
| 39968420761 | 10-23 14:17 | external | 35.2 MB | – |
| 39973892098 | 10-23 14:56 | scan | 34.6 MB | 1.55 MB |
| 39978874688 | 10-23 15:30 | external (**cancelled**) | – | – |
| 39999355264 | 10-23 17:57 | scan | 34.4 MB | 1.54 MB |
| 40012410917 | 10-23 19:34 | scan | 34.5 MB | 1.53 MB |
| 40025944915 | 10-23 21:19 | scan | 33.7 MB | 1.29 MB |
| 40042567854 | 10-23 23:58 | eod | 30.0 MB | – |
| 40047953810 | 10-24 01:02 | external | 33.0 MB | – |

None expired (`expired=false`). `turboedge-state-enc` has 14-day retention,
so **these expire around 2026-11-06/07**; the snapshot artifacts have 90-day
retention. Run 39978874688 has no artifact at all, which is consistent: its
`external` job was cancelled, so its `if: success()` pack step never ran.
Nothing is recoverable from it and nothing was packed to lose.

Both archive shapes are encrypted (`state.tar.enc`, `snapshot-batch.tar.enc`).
There is no unencrypted path to any of this data.

## 3. What is actually lost, and what is not

**Irreplaceable: four scans' product snapshots.** A bid/ask quote with its
issuer, barrier, financing level and timestamp, as it stood at 17:57 on
2026-10-23, cannot be fetched again at any later date. This is the part of
the project with no substitute, and it is the reason to run this recovery at
all rather than accept the gap.

**Recoverable by re-fetching, with a caveat: two external ingests.** The
adapters re-fetch history, and the next `external` run will. The caveat is
provenance, not values: `retrieved_at` and -- for sources without published
vintages -- `available_at` would carry the later retrieval, not the original.
ALFRED-style vintages are unaffected because the vintage is part of the data.
Re-fetching is point-in-time valid going forward; it is not a faithful
reconstruction of what was knowable at the original moment.

**Derived, recomputable only if its inputs survive: the eod labels and learn
output of 40042567854.** Labels come from stored bars and snapshots. If the
snapshots above are recovered, re-running the labeller reproduces them.

**Must not be reconstructed at all:** no historical product price may be
rebuilt from today's product information. That substitution is forbidden
outright, and a gap left visible is the correct outcome where the original
quote is gone.

## 4. The procedure

**Updated 2026-10-25.** The original version of this section assumed the
operator would set `TURBOEDGE_STATE_KEY` locally. That assumption was wrong
in a way worth recording: GitHub does not reveal a secret to anyone after it
is set, including the repository owner, so there may be no copy of the value
anywhere. The key was registered on 2026-09-13 and has never been changed.

That does not block the recovery, because the key does not need to leave
Actions. `.github/workflows/state-recovery.yml` runs the whole procedure
where the secret is already injected, exactly as every ordinary data job
decrypts the state five times a day. **Nobody handles the key.**

### 4a. The automated route (preferred)

1. Dispatch **State Recovery** with `dry_run: true` (the default). It
   restores the current state, downloads and unpacks each of the seven
   archives into its own directory, merges them oldest first, verifies, and
   uploads `turboedge-recovery-reports` — **without packing any state**.
2. Read the reports. Each `merge-<run_id>.json` gives per-table counts
   before and after, how many rows were inserted, and how many duplicate
   key groups exist afterwards. The job fails outright if any merge
   introduced a duplicate or moved a row count by more than it inserted.
3. If the numbers look right, dispatch again with `dry_run: false`. The
   merged state is uploaded as **`turboedge-state-recovered-enc`**, under
   its own name and from its own workflow, so `pipeline.yml`'s restore
   cannot select it.
4. Promote it deliberately: re-upload that archive as `turboedge-state-enc`
   from a `pipeline.yml` dispatch, or push it to the durable backend. This
   step is intentionally not automated.

Safety properties, each covered by a test in
`tests/test_workflow_modes.py`: `workflow_dispatch` only, `dry_run`
defaults to true, the result never carries the live artifact name, packing
happens only after a clean verification, no archive is ever unpacked over
`./state`, and the merge is insert-only.

### 4b. The manual route (only if Actions is unavailable)

Every step needs `TURBOEDGE_STATE_KEY` and must be run by the operator —
which is possible only if a copy of the value exists outside GitHub. Work
on a copy throughout; the live state is never the target.

1. **Back up the current state first.** Download the newest
   `turboedge-state-enc` from the current chain and keep it untouched. The
   weekly job also packs `turboedge-state-backup-enc` with 90-day retention;
   today's weekly (40084459900) produced one.

2. **Download the seven artifacts** into seven separate directories:

   ```bash
   for id in 39968420761 39973892098 39999355264 40012410917 \
             40025944915 40042567854 40047953810; do
     gh run download "$id" -n turboedge-state-enc -D "recover/$id"
   done
   ```

   Also the four snapshot artifacts (`-n turboedge-snapshots-<id>`).

3. **Unpack each into its own directory**, never over the live state:

   ```bash
   turboedge --state-dir ./recover/<id>/state state unpack \
     --in recover/<id>/state.tar.enc
   ```

   `state unpack` needs the key in the environment. **This is where this
   system stops.**

4. **Verify each unpacked state before using it.** `turboedge db info` and
   `turboedge --state-dir <dir> db compact` both exercise the schema. A state
   that will not open is not a candidate for merging.

5. **Merge additively, snapshots first — and de-duplicate explicitly.**

   The two tables behave differently, and this is the step where that
   matters most:

   - `external_observations` has `PRIMARY KEY (source, series_id,
     observation_time, available_at)`, so re-inserting a row already present
     is rejected and the merge is naturally idempotent.
   - `product_snapshots` has **no primary key**, and
     `Store.append_product_snapshots` is a plain `INSERT`. Re-inserting the
     same snapshot rows **duplicates** them. There is no uniqueness
     constraint to lean on, so the merge must do the de-duplication itself:
     insert only rows whose `(isin, observation_time, source)` is not
     already present, or equivalently whose `raw_hash` (sha256 of the raw
     source record) is new.

     ```sql
     INSERT INTO product_snapshots
     SELECT r.* FROM recovered.product_snapshots r
     WHERE NOT EXISTS (
       SELECT 1 FROM product_snapshots c
       WHERE c.isin = r.isin
         AND c.observation_time = r.observation_time
         AND c.source = r.source
     );
     ```

     Duplicated snapshots would not merely waste space: they would double a
     product's observation count, which feeds retention thinning and every
     effective-sample figure derived from it.

   Merge in chronological order of the runs above. Take `forward_ledger`,
   `ledger_labels`, `research_trials` and the registries from the newest
   artifact that contains each, not from an arbitrary one -- the siblings
   disagree about anything written after 12:49.

6. **Reconcile what the merge produced.** Row counts per table before and
   after, checking `product_snapshots` for duplicates introduced by the
   merge (`SELECT isin, observation_time, source, count(*) FROM
   product_snapshots GROUP BY 1,2,3 HAVING count(*) > 1` must return
   nothing more than it did before); the earliest and latest
   `observation_time` per source; that no
   `available_at < observation_time` appeared (the point-in-time invariant);
   and that `research_trials` did not grow -- the number of tested hypotheses
   is the denominator of the multiple-testing correction and recovery must
   not inflate it.

7. **Re-run the labeller** for the eod work of 40042567854 only after the
   snapshots are in place, so it labels from recovered data rather than from
   a gap.

8. **Publish the merged state** through one `workflow_dispatch` run, and
   record in `docs/measured_results.md` which runs were merged, which tables
   grew by how much, and that two external ingests were re-fetched rather
   than recovered.

## 4c. Merge semantics, in one place

`state/recovery.py` and `turboedge state merge-recovered`:

- **Insert-only.** Rows the current database lacks are added; nothing is
  updated and nothing is deleted. On a key present in both, the live row
  wins and the recovered one is discarded.
- **Identity per table.** DuckDB's PRIMARY KEY where one exists (29 of 36
  tables), a configured natural key for the seven that have none.
- **Fail-closed.** A table with neither is skipped and reported, never
  merged on a guess.
- **Idempotent.** Running it twice inserts nothing the second time, so an
  interrupted recovery can simply be repeated.
- **`schema_migrations` is never merged.** It records what *this*
  database's schema has been through; another database's log is not this
  one's history.

## 5. What this system will not do

- Read, display, export or change `TURBOEDGE_STATE_KEY`.
- Decrypt or unpack any artifact.
- Overwrite the live state.
- Reconstruct a historical quote from current product data.
- Treat a re-fetched external observation as if it carried its original
  retrieval provenance.

## 6. Time limit

`turboedge-state-enc` expires 14 days after upload: **around 2026-11-06/07**
for these runs. After that, only the four snapshot artifacts (90-day
retention) remain, which hold the irreplaceable Parquet quote archive but
not the DuckDB tables, ledger or registries.

---

*Prepared 2026-10-24. Verified against the Actions API; nothing executed.*
