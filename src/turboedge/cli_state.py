"""``turboedge state pack/unpack`` and ``turboedge db compact``.

Kept as a separate module (rather than appended into ``cli.py`` directly) so
that ``cli.py`` -- a shared file outside this workstream's assignment --
only needs a two-line registration (see :func:`register_state_commands`) to
wire these commands in. This module never imports anything back from
``cli.py`` (that would be circular, since ``cli.py`` imports this module),
which is why ``register_state_commands`` takes ``app``/``db_app`` as
parameters instead of importing them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import structlog
import typer
from rich.console import Console
from rich.table import Table

from turboedge.state.archive import (
    FULL_INCLUDE_DIRS,
    LEAN_INCLUDE_DIRS,
    SNAPSHOTS_PREFIX,
    ArchiveConfig,
    StateArchiveError,
    pack_state,
    pack_state_files,
    unpack_state,
    unpack_state_subset,
)
from turboedge.state.backend import StateBackend, StateBackendError, backend_from_env
from turboedge.state.crypto import StateCryptoError
from turboedge.state.retention import (
    DEFAULT_HARD_DELETE_AFTER_DAYS,
    DEFAULT_KEEP_DAYS,
    compact_product_snapshots,
)
from turboedge.storage.duckdb import Store
from turboedge.storage.snapshots import snapshot_paths_for_run_ids

logger = structlog.get_logger(__name__)

# Separate Console instances (not imported from `cli.py`) for the same
# reason as the module docstring: no import back into the shared file.
# `stderr=True` (not `file=sys.stderr`) mirrors cli.py's `err_console` -- see
# its comment for why (typer.testing.CliRunner swaps sys.stderr per call).
console = Console()
err_console = Console(stderr=True)

state_app = typer.Typer(help="Encrypted state archive pack/unpack (public-repo hygiene)")

_STATE_KEY_ENV = "TURBOEDGE_STATE_KEY"
_STATE_KEY_HINT = 'python -c "import secrets;print(secrets.token_urlsafe(32))"'


def _require_state_key() -> str:
    key = os.environ.get(_STATE_KEY_ENV)
    if not key:
        err_console.print(
            f"[red]{_STATE_KEY_ENV} is not set.[/red] Generate one and store it as a secret "
            f"(e.g. GitHub Actions repo secret {_STATE_KEY_ENV}):\n"
            f"  {_STATE_KEY_HINT}"
        )
        raise typer.Exit(code=2)
    return key


def _state_dir_from_ctx(ctx: typer.Context) -> Path:
    app_ctx = ctx.obj
    state_dir: Path = app_ctx.state_dir
    return state_dir


@state_app.command("pack")
def state_pack(
    ctx: typer.Context,
    out: str = typer.Option(..., "--out", help="Output path for the encrypted state archive"),
    include_snapshots: bool = typer.Option(
        False,
        "--include-snapshots/--no-include-snapshots",
        help=(
            "Include state/snapshots/ (the immutable per-run Parquet archive) in the "
            "packed archive. Default: excluded -- the small 'lean' package "
            "(state/archive.py LEAN_INCLUDE_DIRS) every scan/eod/weekly/monthly pipeline "
            "run packs several times a day. --include-snapshots produces the 'full' "
            "shape (FULL_INCLUDE_DIRS) for a manual/ad hoc complete export; no "
            "pipeline.yml job packs it automatically -- the accumulated Parquet "
            "history is instead kept durable incrementally by every scan run via "
            "'state pack-snapshots' (see state/archive.py's module docstring)."
        ),
    ),
) -> None:
    """Checkpoint DuckDB, tar.gz the state directory, encrypt with
    TURBOEDGE_STATE_KEY.

    Exit codes: 0 ok, 2 TURBOEDGE_STATE_KEY missing/too short or any other
    pack failure.
    """
    passphrase = _require_state_key()
    state_dir = _state_dir_from_ctx(ctx)
    out_path = Path(out)
    config = ArchiveConfig(
        include_dirs=FULL_INCLUDE_DIRS if include_snapshots else LEAN_INCLUDE_DIRS
    )

    try:
        result = pack_state(state_dir, out_path, passphrase, config=config)
    except StateCryptoError as exc:
        err_console.print(f"[red]State pack failed: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    logger.info(
        "state_packed",
        out=str(result.out_path),
        file_count=result.file_count,
        total_bytes=result.total_bytes,
        archive_bytes=result.archive_bytes,
        include_snapshots=include_snapshots,
    )
    kind = "full (includes state/snapshots/)" if include_snapshots else "lean (no state/snapshots/)"
    console.print(
        f"[green]Packed {result.file_count} file(s), {result.total_bytes} bytes -> "
        f"{result.out_path} ({result.archive_bytes} bytes encrypted, {kind})[/green]"
    )


@state_app.command("pack-snapshots")
def state_pack_snapshots(
    ctx: typer.Context,
    run_id: Annotated[
        list[str] | None,
        typer.Option(
            "--run-id",
            help=(
                "Only pack the Parquet snapshot file(s) written under this run_id "
                "(repeatable) -- typically every value from one scan-all invocation's "
                "run_ids (reports/summary.json's 'run_ids' field, one per underlying)."
            ),
        ),
    ] = None,
    out: str = typer.Option(
        ..., "--out", help="Output path for the encrypted incremental snapshot archive"
    ),
) -> None:
    """Pack ONLY the immutable Parquet snapshot file(s) that THIS run wrote
    -- identified by --run-id -- into their own small encrypted archive.

    This is the incremental counterpart to ``state pack --include-snapshots``
    (which re-packs the *entire* accumulated state/snapshots/ history every
    time): ``turboedge scan-all`` writes one Parquet file per underlying
    (storage/snapshots.py's write_snapshot_parquet), and this command finds
    exactly those files via storage.snapshots.snapshot_paths_for_run_ids and
    packs only them, so a several-times-a-day scan job can durably archive
    its own new output without re-uploading anything it already shipped on a
    prior run. See state/archive.py's module docstring ("Parquet archiving")
    for why this replaced the old weekly-only full-archive approach.

    If none of the given --run-id values produced a snapshot file (e.g.
    every underlying in the batch was skipped before writing one), no
    archive is written -- --out will not exist -- and this exits 0 with a
    note rather than an error; the caller (pipeline.yml) should skip the
    upload step in that case.

    Exit codes: 0 ok (including the "nothing to pack" case), 2
    TURBOEDGE_STATE_KEY missing/too short, or a --run-id containing
    characters new_run_id() never produces.
    """
    passphrase = _require_state_key()
    state_dir = _state_dir_from_ctx(ctx)
    out_path = Path(out)
    run_ids = run_id or []

    try:
        files = snapshot_paths_for_run_ids(state_dir, run_ids)
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    if not files:
        logger.info("state_snapshots_pack_empty", run_id_count=len(run_ids))
        console.print(
            f"[yellow]No snapshot Parquet file(s) found for {len(run_ids)} run_id(s) -- "
            "nothing to pack.[/yellow]"
        )
        raise typer.Exit(code=0)

    try:
        result = pack_state_files(state_dir, out_path, passphrase, files)
    except StateCryptoError as exc:
        err_console.print(f"[red]Snapshot pack failed: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    logger.info(
        "state_snapshots_packed",
        out=str(result.out_path),
        file_count=result.file_count,
        total_bytes=result.total_bytes,
        archive_bytes=result.archive_bytes,
        run_id_count=len(run_ids),
    )
    console.print(
        f"[green]Packed {result.file_count} snapshot file(s), {result.total_bytes} bytes -> "
        f"{result.out_path} ({result.archive_bytes} bytes encrypted)[/green]"
    )


@state_app.command("unpack")
def state_unpack(
    ctx: typer.Context,
    in_path: str = typer.Option(..., "--in", help="Path to the encrypted state archive"),
    allow_missing: bool = typer.Option(
        False,
        "--allow-missing",
        help=(
            "If --in does not exist, warn and continue with a fresh state directory "
            "instead of failing (used by pipeline.yml when no prior artifact exists yet, "
            "or it has expired)"
        ),
    ),
) -> None:
    """Decrypt + verify a state archive, then atomically replace the state
    directory with its contents.

    Exit codes: 0 ok (including the --allow-missing "no archive found" case),
    2 TURBOEDGE_STATE_KEY missing/too short, archive missing without
    --allow-missing, wrong key, or a corrupted/unsafe archive.
    """
    passphrase = _require_state_key()
    state_dir = _state_dir_from_ctx(ctx)
    src = Path(in_path)

    if not src.is_file():
        if allow_missing:
            logger.warning("state_archive_missing_allow_missing", path=str(src))
            console.print(
                f"[yellow]No state archive found at {src} -- starting from a fresh state "
                "directory.[/yellow]"
            )
            state_dir.mkdir(parents=True, exist_ok=True)
            raise typer.Exit(code=0)
        err_console.print(f"[red]No state archive found at {src}[/red]")
        raise typer.Exit(code=2)

    try:
        result = unpack_state(src, state_dir, passphrase)
    except StateCryptoError as exc:
        err_console.print(
            f"[red]State unpack failed (wrong TURBOEDGE_STATE_KEY or corrupted archive): "
            f"{exc}[/red]"
        )
        raise typer.Exit(code=2) from exc
    except StateArchiveError as exc:
        err_console.print(f"[red]State unpack failed: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    logger.info(
        "state_unpacked",
        state_dir=str(result.state_dir),
        file_count=result.file_count,
        total_bytes=result.total_bytes,
    )
    console.print(
        f"[green]Unpacked {result.file_count} file(s), {result.total_bytes} bytes into "
        f"{result.state_dir}[/green]"
    )


@state_app.command("restore-snapshots")
def state_restore_snapshots(
    ctx: typer.Context,
    in_path: str = typer.Option(
        ..., "--in", help="Path to a full (--include-snapshots) encrypted state archive"
    ),
) -> None:
    """Additively restore ONLY state/snapshots/ (the immutable Parquet
    archive) from a full state archive into --state-dir, without touching
    anything else already there.

    Unlike ``state unpack``, this never replaces or deletes anything in
    --state-dir -- it is a partial, additive merge, meant to run AFTER a
    normal ``state unpack`` (which is lean by default and never carries
    state/snapshots/, see ``state pack --include-snapshots``). This is a
    manual, as-needed recovery tool (e.g. restoring onto a fresh state_dir
    from a full export someone took with ``state pack --include-snapshots``)
    -- it is NOT part of pipeline.yml's automated restore flow. The
    incremental per-scan Parquet snapshot artifact (``state pack-snapshots``,
    uploaded by every scan run) is what now keeps the accumulated Parquet
    history durable (CLAUDE.md rule 33); see state/archive.py's module
    docstring ("Parquet archiving") for the full history of why this
    replaced an earlier weekly-only full-archive approach that could not
    actually carry forward scan output written after it shipped.

    Exit codes: 0 ok, 2 TURBOEDGE_STATE_KEY missing/too short, archive
    missing, wrong key, or a corrupted/unsafe archive.
    """
    passphrase = _require_state_key()
    state_dir = _state_dir_from_ctx(ctx)
    src = Path(in_path)

    if not src.is_file():
        err_console.print(f"[red]No state archive found at {src}[/red]")
        raise typer.Exit(code=2)

    try:
        result = unpack_state_subset(src, state_dir, passphrase, prefixes=(SNAPSHOTS_PREFIX,))
    except StateCryptoError as exc:
        err_console.print(
            f"[red]Snapshot restore failed (wrong TURBOEDGE_STATE_KEY or corrupted archive): "
            f"{exc}[/red]"
        )
        raise typer.Exit(code=2) from exc
    except StateArchiveError as exc:
        err_console.print(f"[red]Snapshot restore failed: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    logger.info(
        "state_snapshots_restored",
        state_dir=str(result.state_dir),
        file_count=result.file_count,
        total_bytes=result.total_bytes,
    )
    console.print(
        f"[green]Merged {result.file_count} snapshot file(s), {result.total_bytes} bytes into "
        f"{result.state_dir}[/green]"
    )


backend_app = typer.Typer(
    help=(
        "Durable, versioned remote/local backend for encrypted state archives "
        "(Phase H -- see docs/durable_state.md). Opt-in via TURBOEDGE_STATE_BACKEND; "
        "a fallback layer UNDER the GitHub Actions artifact, not a replacement for it."
    )
)


def _require_backend() -> StateBackend:
    try:
        backend = backend_from_env()
    except StateBackendError as exc:
        err_console.print(f"[red]Durable state backend misconfigured: {exc}[/red]")
        raise typer.Exit(code=2) from exc
    if backend is None:
        err_console.print(
            "[yellow]TURBOEDGE_STATE_BACKEND is not set -- no durable remote/local backend "
            "configured. See docs/durable_state.md.[/yellow]"
        )
        raise typer.Exit(code=2)
    return backend


@backend_app.command("put")
def backend_put(
    key: str = typer.Option(..., "--key", help="Logical key to version this archive under"),
    in_path: str = typer.Option(
        ..., "--in", help="Local path of the (already-encrypted) archive to upload"
    ),
) -> None:
    """Upload --in as a brand-new, immutable version of --key.

    Never overwrites a previous version. Exit codes: 0 ok, 2 backend not
    configured/misconfigured, missing --in file, or any upload/verification
    failure (checksum mismatch, unreachable backend, ...).
    """
    backend = _require_backend()
    src = Path(in_path)
    if not src.is_file():
        err_console.print(f"[red]No such file: {src}[/red]")
        raise typer.Exit(code=2)

    try:
        version_id = backend.put(src, key)
    except StateBackendError as exc:
        err_console.print(f"[red]Backend put failed: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    logger.info("state_backend_put", key=key, version_id=version_id, in_path=str(src))
    console.print(f"[green]Uploaded {src} as key={key!r} version={version_id}[/green]")


@backend_app.command("get")
def backend_get(
    key: str = typer.Option(..., "--key", help="Logical key to fetch"),
    out: str = typer.Option(..., "--out", help="Local path to write the downloaded archive to"),
    version: str | None = typer.Option(
        None, "--version", help="Specific version_id to fetch (default: the latest)"
    ),
) -> None:
    """Download a version of --key (the latest, unless --version is given)
    to --out, verifying its checksum before writing.

    Exit codes: 0 ok, 2 backend not configured/misconfigured, key/version
    not found, unreachable backend, or a checksum mismatch. Never writes a
    partial/corrupted file on failure.
    """
    backend = _require_backend()
    out_path = Path(out)

    try:
        fetched_version = backend.get(key, out_path, version=version)
    except StateBackendError as exc:
        err_console.print(f"[red]Backend get failed: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    logger.info("state_backend_get", key=key, version_id=fetched_version, out_path=str(out_path))
    console.print(f"[green]Downloaded key={key!r} version={fetched_version} -> {out_path}[/green]")


@backend_app.command("list-versions")
def backend_list_versions(
    key: str = typer.Option(..., "--key", help="Logical key to list versions of"),
) -> None:
    """List every version of --key, oldest first.

    Exit codes: 0 ok (including zero versions found), 2 backend not
    configured/misconfigured or unreachable.
    """
    backend = _require_backend()
    try:
        versions = backend.list_versions(key)
    except StateBackendError as exc:
        err_console.print(f"[red]Backend list-versions failed: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    table = Table(title=f"backend versions for key={key!r}")
    table.add_column("version_id")
    table.add_column("created_at (UTC)")
    table.add_column("size (bytes)", justify="right")
    table.add_column("sha256")
    for v in versions:
        table.add_row(v.version_id, v.created_at.isoformat(), str(v.size), v.sha256)
    console.print(table)
    if not versions:
        console.print(f"[yellow]No versions found for key={key!r}.[/yellow]")


@backend_app.command("health")
def backend_health() -> None:
    """Check whether the configured durable backend is reachable right now.

    Exit codes: 0 reachable, 1 configured but unreachable, 2 not
    configured/misconfigured.
    """
    backend = _require_backend()
    result = backend.health()
    if result.reachable:
        console.print(f"[green]Backend reachable: {result.reason}[/green]")
        raise typer.Exit(code=0)
    err_console.print(f"[red]Backend unreachable: {result.reason}[/red]")
    raise typer.Exit(code=1)


def db_compact(
    ctx: typer.Context,
    keep_days: int = typer.Option(
        DEFAULT_KEEP_DAYS,
        "--keep-days",
        help="Compact product_snapshots rows older than N days to one row/isin/day",
    ),
    hard_delete_after_days: int = typer.Option(
        DEFAULT_HARD_DELETE_AFTER_DAYS,
        "--hard-delete-after-days",
        help=(
            "Permanently delete product_snapshots rows older than N days (must be > "
            "--keep-days), except ISINs present in forward_ledger (selected or as an "
            "alternative/counterfactual), which are never thinned or deleted"
        ),
    ),
) -> None:
    """Reduce ``product_snapshots`` rows older than --keep-days to one row
    per (isin, UTC calendar day), then permanently delete whatever is still
    older than --hard-delete-after-days; ISINs present in ``forward_ledger``
    (if that table exists yet -- as the selected pick or anywhere in
    ``alternatives``) keep their full history and are never thinned or
    deleted, however old. Runs CHECKPOINT, then rewrites the database file
    in place so the reduction actually shrinks it on disk (see
    ``state/retention.py`` module docstring).
    """
    app_ctx = ctx.obj
    db_path: Path = app_ctx.state_dir / "turboedge.duckdb"

    with Store(db_path) as store:
        store.init_schema()
        report = compact_product_snapshots(
            store, keep_days=keep_days, hard_delete_after_days=hard_delete_after_days
        )

    logger.info(
        "db_compact",
        keep_days=report.keep_days,
        hard_delete_after_days=report.hard_delete_after_days,
        rows_before=report.rows_before,
        rows_after=report.rows_after,
        rows_removed=report.rows_removed,
        rows_hard_deleted=report.rows_hard_deleted,
        protected_isin_count=report.protected_isin_count,
        forward_ledger_present=report.forward_ledger_present,
        db_size_bytes_before=report.db_size_bytes_before,
        db_size_bytes_after=report.db_size_bytes_after,
        file_rewritten=report.file_rewritten,
    )

    table = Table(title="db compact report")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("keep_days", str(report.keep_days))
    table.add_row("hard_delete_after_days", str(report.hard_delete_after_days))
    table.add_row("forward_ledger present", str(report.forward_ledger_present))
    table.add_row("protected ISINs", str(report.protected_isin_count))
    table.add_row("product_snapshots rows before", str(report.rows_before))
    table.add_row("product_snapshots rows after", str(report.rows_after))
    table.add_row("rows removed (total)", str(report.rows_removed))
    table.add_row("rows hard-deleted", str(report.rows_hard_deleted))
    table.add_row("db size before (bytes)", str(report.db_size_bytes_before))
    table.add_row("db size after (bytes)", str(report.db_size_bytes_after))
    table.add_row("db file physically rewritten", str(report.file_rewritten))
    console.print(table)


def register_state_commands(app: typer.Typer, db_app: typer.Typer) -> None:
    """Wire ``state pack``/``state unpack``/``state backend *`` and
    ``db compact`` into the main CLI. Called once from ``cli.py``::

        from turboedge.cli_state import register_state_commands
        register_state_commands(app, db_app)
    """
    state_app.add_typer(backend_app, name="backend")
    app.add_typer(state_app, name="state")
    db_app.command("compact")(db_compact)


__all__ = ["register_state_commands", "state_app"]
