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

import structlog
import typer
from rich.console import Console
from rich.table import Table

from turboedge.state.archive import StateArchiveError, pack_state, unpack_state
from turboedge.state.crypto import StateCryptoError
from turboedge.state.retention import DEFAULT_KEEP_DAYS, compact_product_snapshots
from turboedge.storage.duckdb import Store

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
) -> None:
    """Checkpoint DuckDB, tar.gz the state directory, encrypt with
    TURBOEDGE_STATE_KEY.

    Exit codes: 0 ok, 2 TURBOEDGE_STATE_KEY missing/too short or any other
    pack failure.
    """
    passphrase = _require_state_key()
    state_dir = _state_dir_from_ctx(ctx)
    out_path = Path(out)

    try:
        result = pack_state(state_dir, out_path, passphrase)
    except StateCryptoError as exc:
        err_console.print(f"[red]State pack failed: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    logger.info(
        "state_packed",
        out=str(result.out_path),
        file_count=result.file_count,
        total_bytes=result.total_bytes,
        archive_bytes=result.archive_bytes,
    )
    console.print(
        f"[green]Packed {result.file_count} file(s), {result.total_bytes} bytes -> "
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


def db_compact(
    ctx: typer.Context,
    keep_days: int = typer.Option(
        DEFAULT_KEEP_DAYS,
        "--keep-days",
        help="Compact product_snapshots rows older than N days to one row/isin/day",
    ),
) -> None:
    """Reduce ``product_snapshots`` rows older than --keep-days to one row
    per (isin, UTC calendar day); ISINs present in ``forward_ledger`` (if
    that table exists yet) keep their full history. Runs CHECKPOINT, then
    rewrites the database file in place so the reduction actually shrinks
    it on disk (see ``state/retention.py`` module docstring).
    """
    app_ctx = ctx.obj
    db_path: Path = app_ctx.state_dir / "turboedge.duckdb"

    with Store(db_path) as store:
        store.init_schema()
        report = compact_product_snapshots(store, keep_days=keep_days)

    logger.info(
        "db_compact",
        keep_days=report.keep_days,
        rows_before=report.rows_before,
        rows_after=report.rows_after,
        rows_removed=report.rows_removed,
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
    table.add_row("forward_ledger present", str(report.forward_ledger_present))
    table.add_row("protected ISINs", str(report.protected_isin_count))
    table.add_row("product_snapshots rows before", str(report.rows_before))
    table.add_row("product_snapshots rows after", str(report.rows_after))
    table.add_row("rows removed", str(report.rows_removed))
    table.add_row("db size before (bytes)", str(report.db_size_bytes_before))
    table.add_row("db size after (bytes)", str(report.db_size_bytes_after))
    table.add_row("db file physically rewritten", str(report.file_rewritten))
    console.print(table)


def register_state_commands(app: typer.Typer, db_app: typer.Typer) -> None:
    """Wire ``state pack``/``state unpack`` and ``db compact`` into the main
    CLI. Called once from ``cli.py``::

        from turboedge.cli_state import register_state_commands
        register_state_commands(app, db_app)
    """
    app.add_typer(state_app, name="state")
    db_app.command("compact")(db_compact)


__all__ = ["register_state_commands", "state_app"]
