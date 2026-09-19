# Durable State (Phase H)

## The problem this solves

Today, `turboedge`'s entire learning state — the DuckDB file (with
`forward_ledger`, `strategy_posteriors`, `model_registry`,
`research_trials`, and every other table) plus `state/registry/`,
`state/ledger/`, and `state/trials/` — lives **exclusively** in a GitHub
Actions workflow artifact, `turboedge-state-enc`
(`.github/actions/turboedge-state-pack/action.yml`, `retention_days: "14"`).
The weekly job additionally packs `turboedge-state-backup-enc` under a
90-day retention, but nothing restores it automatically — it is a manual,
`gh run download`-and-unpack fallback.

Measured, not hypothetical: on 2026-09-14 only one of three scheduled scans
fired, 37 minutes late — scheduled Actions runs are not reliable to the
minute. Public GitHub repositories additionally auto-disable scheduled
workflows after 60 days of repository inactivity. If the pipeline goes 14
days without producing a usable artifact, the forward ledger, posteriors,
model weights, and product history are gone — permanently, with no
recovery path, because nothing outside that one 14-day artifact chain ever
held a copy.

Phase H adds an **optional** durable layer underneath the existing
mechanism. It does not replace it: the GitHub Actions artifact is still
packed every run exactly as before, and is still the first thing every
restore tries. The durable backend is consulted only as a **fallback**,
and only in exactly the situation the artifact chain already treats as
dangerous (case B below) — never instead of the existing behavior, and
never in a way that makes the existing "abort rather than silently lose
history" guarantee weaker.

## The three-case restore logic, unchanged in spirit

`.github/actions/turboedge-state/action.yml`'s restore step has always
distinguished three cases:

- **A — genuine first run.** No previous successful `pipeline.yml` run
  exists at all. A fresh, empty state is correct and expected.
- **B — prior runs exist, but no usable artifact.** None of the last 50
  successful runs carry a usable `turboedge-state-enc` artifact (expired,
  never uploaded, or a failed download). This aborts the run loudly
  (`::error::` + exit 1) unless `allow_fresh_state: true` was explicitly
  passed via `workflow_dispatch`.
- **C — a usable artifact was found.** Restore it normally.

Phase H changes exactly one thing: **inside case B**, before aborting, the
restore step now also tries the durable backend (`turboedge state backend
get --key lean`). Only if *neither* the GitHub artifact search *nor* the
backend produces anything usable does the run abort (or force a fresh
state under `allow_fresh_state`). Case A and case C are untouched. This
makes case B **narrower** — a strictly larger set of situations now
recovers automatically — never wider.

If `TURBOEDGE_STATE_BACKEND` is unset (the default), `turboedge state
backend get` fails immediately without attempting any network call, so for
anyone who has not opted in, every job's behavior is byte-for-byte
identical to before this feature existed.

## What actually gets pushed, and when

The `eod` job (once per weekday, after `label` → `learn` → `position
reevaluate` → `db compact` — the most meaningfully-updated state of the
day) pushes its packed lean archive to the durable backend under the
logical key `lean`, in addition to uploading it as the normal
`turboedge-state-enc` GitHub Actions artifact. This is deliberately once a
day, not on every `scan` run (5x/day): it keeps a user's own bucket growing
at a bounded, predictable rate rather than uploading the ~168 MB archive
several times a day for years. If you want a different cadence (e.g. also
pushing from `scan`, or a separate key for a less-frequent long-term
snapshot), pass `push_to_backend: "true"` (and/or a different
`backend_key`) to another `./.github/actions/turboedge-state-pack` call in
`pipeline.yml` — the composite action already supports both inputs.

Every version is retained forever by default — this module never deletes
anything (see "Append-only, forever" below). If years of daily ~168 MB
archives becomes a storage-cost concern, configure a lifecycle/expiration
rule directly on your bucket (every S3-compatible provider supports this);
`turboedge` itself has no expiry logic for the durable backend and never
will, since expiring history automatically is exactly the failure mode
this feature exists to prevent.

## Setting it up

### Environment variables

None of these are ever written to the repository, logged, or printed —
`turboedge state backend *` treats them the same way `TURBOEDGE_STATE_KEY`
is already treated (see the main README's Secrets Setup section).

| Variable | Required | Purpose |
|---|---|---|
| `TURBOEDGE_STATE_BACKEND` | to opt in | `"s3"` for the S3-compatible remote backend, `"local"` for a local-path backend, unset/`"none"`/`"off"` to disable (default) |
| `TURBOEDGE_STATE_S3_BUCKET` | if backend=s3 | Bucket name |
| `TURBOEDGE_STATE_S3_ACCESS_KEY_ID` | if backend=s3 | Access key ID |
| `TURBOEDGE_STATE_S3_SECRET_ACCESS_KEY` | if backend=s3 | Secret access key |
| `TURBOEDGE_STATE_S3_ENDPOINT_URL` | optional | Custom endpoint for an S3-compatible provider; omit for AWS S3 itself |
| `TURBOEDGE_STATE_S3_REGION` | optional | Default `"auto"` (Cloudflare R2's convention); set a real AWS region for AWS S3 |
| `TURBOEDGE_STATE_S3_PREFIX` | optional | Default `"turboedge-state"`; object key prefix inside the bucket |
| `TURBOEDGE_STATE_LOCAL_ROOT` | if backend=local | Local directory root (e.g. a mounted network share) |

### Why S3-compatible, and which provider

The remote backend (`turboedge.state.backend_s3.S3CompatibleStateBackend`)
speaks the standard S3 API with a configurable `endpoint_url`, so it works
identically against **AWS S3** and against any S3-compatible provider —
**Cloudflare R2**, **Backblaze B2**, **Wasabi**, **MinIO** (self-hosted),
**DigitalOcean Spaces**, and others. This is deliberate: nothing in this
codebase locks you into one vendor for what is meant to be years of
learning history. Pick whichever provider you're comfortable trusting with
an encrypted blob and leaving running for years; a few concrete starting
points if you have no existing preference:

- **Cloudflare R2** — S3-compatible, no egress fees, generous free tier;
  `TURBOEDGE_STATE_S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com`,
  `TURBOEDGE_STATE_S3_REGION=auto`.
- **Backblaze B2** (S3-compatible API) — inexpensive, long-standing
  archival focus; `TURBOEDGE_STATE_S3_ENDPOINT_URL=https://s3.<region>.backblazeb2.com`.
- **AWS S3** — leave `TURBOEDGE_STATE_S3_ENDPOINT_URL` unset, set
  `TURBOEDGE_STATE_S3_REGION` to a real AWS region (e.g. `eu-central-1`).

Whichever you choose, create a dedicated bucket and an access key scoped to
only that bucket (S3 IAM policy or the provider's equivalent) — this
backend only ever needs `PutObject`/`GetObject`/`HeadObject`/
`ListObjectsV2` (never delete).

### Setup steps

1. Create a bucket with any S3-compatible provider (see above).
2. Create an access key scoped to that bucket only.
3. Install the optional dependency locally if you want to exercise the
   backend from your own machine: `uv sync --extra state-remote` (CI
   already installs it unconditionally via the same extra — see
   `.github/actions/turboedge-state/action.yml`'s "Install dependencies"
   step — so no workflow file needs editing just to turn this on).
4. Add the environment variables above as GitHub Actions repo secrets
   (Settings → Secrets and variables → Actions) — the same place
   `TURBOEDGE_STATE_KEY`/`GMAIL_*` already live. `pipeline.yml` reads them
   at the workflow level and passes them through to every job.
5. Verify locally before relying on it in CI:
   ```bash
   export TURBOEDGE_STATE_BACKEND=s3
   export TURBOEDGE_STATE_S3_BUCKET=...
   export TURBOEDGE_STATE_S3_ACCESS_KEY_ID=...
   export TURBOEDGE_STATE_S3_SECRET_ACCESS_KEY=...
   export TURBOEDGE_STATE_S3_ENDPOINT_URL=...   # omit for AWS S3
   uv run turboedge state backend health
   uv run turboedge state backend put --key lean --in state.tar.enc
   uv run turboedge state backend list-versions --key lean
   uv run turboedge state backend get --key lean --out restored.tar.enc
   ```
6. That's it — no other configuration. The next `eod` run starts pushing
   automatically, and every job's restore step gains the fallback
   automatically.

### The `local` backend

`TURBOEDGE_STATE_BACKEND=local` + `TURBOEDGE_STATE_LOCAL_ROOT=<path>`
selects `turboedge.state.backend.LocalStateBackend` instead of S3 — the
same versioned/atomic/checksum-verified contract, but writing to a plain
directory. This is genuinely durable when `<path>` is something other than
the pipeline runner's own ephemeral workspace (a self-hosted runner with a
mounted network share, an rclone/Dropbox sync target, etc.); it is also
what the test suite uses as a fast, hermetic stand-in for exercising the
same contract the S3 backend implements, since it needs no network and no
credentials.

## What survives, and for how long

With a durable backend configured, everything the lean archive already
covers — `forward_ledger` (entries + counterfactual alternatives, Master
Spec §21), `strategy_posteriors`, `model_registry` (champion/challenger/
dormant), `research_trials`, and the financing-level history embedded in
`product_snapshots` — is versioned forever, not for 14 days. Every `eod`
push is a complete, self-contained snapshot (the same lean archive shape
`state pack` already produces — DuckDB + `registry/`/`ledger/`/`trials/`,
still excluding the separately-archived `state/snapshots/` Parquet tree,
which keeps its own 90-day-retention incremental artifact chain per
`scan` run, unaffected by any of this). Because nothing is ever deleted or
overwritten, you can go back to *any* day's state, not just the most
recent one — see "Append-only, forever" below.

## Encryption

Nothing changes about encryption. `turboedge state pack` still produces
the AES-256-GCM-encrypted archive exactly as before
(`turboedge.state.crypto`, `TURBOEDGE_STATE_KEY`); `turboedge state backend
put` then uploads those already-encrypted bytes completely unchanged. The
S3-compatible backend never sees plaintext state, never performs any
encryption/decryption of its own, and would be useless to anyone who
obtained access to the bucket without also having `TURBOEDGE_STATE_KEY`.

## Append-only, forever

`put` always creates a brand-new, uniquely-named version — `<UTC timestamp
to the microsecond>-<8 random hex chars>` — and never overwrites or
deletes anything. There is no code path anywhere in
`turboedge.state.backend`/`turboedge.state.backend_s3` that deletes or
replaces an existing version. `list_versions` returns every version ever
written, oldest first.

## Integrity, atomicity, and failure behavior

- **Checksums always.** `put` computes a SHA-256 of the exact bytes being
  uploaded and stores it as metadata alongside the version; `get`
  re-verifies it after downloading and raises loudly
  (`StateBackendError`) on any mismatch instead of handing back corrupted
  data.
- **An aborted upload never becomes a valid version.** The S3-compatible
  backend uploads with an S3 `Content-MD5` header, which makes S3 itself
  reject a truncated/corrupted body server-side (HTTP 400) before it is
  ever stored; a failed `put_object` call simply means that version's
  object key was never created. The local backend copies to a hidden temp
  file, re-verifies its checksum, and only then atomically renames it into
  place (`os.replace`) — an interrupted copy leaves only the still-hidden
  temp file, invisible to `list_versions`/`get`.
- **An unreachable or misconfigured backend raises — it never produces an
  empty state.** Every method (`put`/`get`/`list_versions`) raises
  `StateBackendError` on a network failure, an auth failure, or a missing
  key/version; none of them ever return an empty result that a caller
  could mistake for "no history exists yet". `health()` is the one
  exception — it reports reachability as data (`HealthResult.reachable`)
  rather than raising, since "is it up right now" is exactly the question
  it exists to answer.
- **This is why the restore composite action's fallback is safe.** If the
  backend is configured but broken (wrong credentials, endpoint down,
  bucket deleted, ...), `state backend get` fails, the restore step falls
  through to the *existing* abort/`allow_fresh_state` logic — it never
  silently treats a broken backend as "no history, start fresh".

## CLI reference

```bash
turboedge state backend put --key KEY --in PATH            # new version from PATH
turboedge state backend get --key KEY --out PATH [--version V]   # latest, or a specific version
turboedge state backend list-versions --key KEY             # every version, oldest first
turboedge state backend health                              # reachability check
```

Exit codes follow the rest of the `state` command group: `0` success (for
`list-versions`, also zero versions found — that is not an error), `2`
backend not configured/misconfigured or a `put`/`get`/`list-versions`
failure, `1` for `health` reporting an unreachable-but-configured backend.

## What this does *not* change

- The existing `turboedge-state-enc`/`turboedge-state-backup-enc`
  GitHub Actions artifacts are untouched — same shape, same retention,
  same restore priority (tried first, always).
- `turboedge state pack`/`unpack`/`pack-snapshots`/`restore-snapshots`
  are untouched.
- No threshold, gate, or risk parameter exists anywhere in this feature —
  it is pure storage plumbing.
- `ranking/gates.py`, `pipeline/scan.py`, and `learning/labeler.py` are
  untouched.
