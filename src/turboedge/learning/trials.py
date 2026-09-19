"""Research trial issuance & quarterly adaptation-budget enforcement.

Master Spec §27.1 ("Jeder Versuch zählt") / §27.2 ("Anpassungsbudget"): every
feature variant, threshold change, model change, hyperparameter family,
selection rule or sizing variant gets a `trial_id`, and the number of such
trials per quarter is capped (GOVERNANCE.md §1.2: 6/quarter, CLAUDE.md rule
24) so that later multiple-testing correction (Benjamini-Hochberg FDR, PSR/
DSR -- Master Spec §27.4) has an honest count of "hypotheses tested" to work
from.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

import structlog
from pydantic import BaseModel, ConfigDict, Field

from turboedge.storage.duckdb import Store
from turboedge.storage.schemas import ResearchTrial, TrialStatus

logger = structlog.get_logger(__name__)


class TrialsConfig(BaseModel):
    """This module's own config (wired into config.py/YAML by the
    integration wave). ``quarterly_budget`` defaults to the same value as
    ``configs/governance.yaml``'s ``research_adjustment_budget_per_quarter``
    (currently 6) -- a literal default here rather than a read of that YAML
    file at import time, per Build Contract v2: "Jedes neue Modul definiert
    seine eigene pydantic XxxConfig mit sinnvollen Defaults IM EIGENEN
    MODUL."
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    quarterly_budget: int = Field(default=6, ge=0)


class TrialBudgetExceeded(Exception):
    """Raised by :func:`new_trial_id` when a quarter's adaptation budget is
    already spent and ``force`` was not passed."""


def _quarter_label(dt: datetime) -> str:
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{quarter}"


def new_trial_id(
    store: Store,
    kind: str,
    description: str,
    *,
    config: TrialsConfig | None = None,
    as_of: datetime | None = None,
    force: bool = False,
) -> str:
    """Mint and persist a new trial id, format ``TR-YYYYQn-<6hex>``.

    Enforces the quarterly adaptation budget (Master Spec §27.2): once
    ``config.quarterly_budget`` trials already exist for the current
    quarter, raises :class:`TrialBudgetExceeded` unless ``force=True`` --
    which still creates the trial (a human override is not blocked) but
    logs a warning, since the budget stays informative for research
    governance either way (CLAUDE.md rule 24).

    Args:
        store: Open ``Store`` the trial is persisted through.
        kind: Short category, e.g. ``"feature"``, ``"threshold"``,
            ``"model"``, ``"sizing"``.
        description: Human-readable summary of what changed.
        config: Budget config; defaults to :class:`TrialsConfig`'s default.
        as_of: Clock override for testing; defaults to ``datetime.now(UTC)``.
        force: Bypass the budget check (still logs a warning).

    Returns:
        The newly minted, already-persisted ``trial_id``.

    Raises:
        TrialBudgetExceeded: Budget reached and ``force`` is ``False``.
    """
    cfg = config if config is not None else TrialsConfig()
    now = as_of if as_of is not None else datetime.now(UTC)
    quarter = _quarter_label(now)
    existing = store.count_research_trials_in_quarter(quarter)
    if existing >= cfg.quarterly_budget:
        if not force:
            raise TrialBudgetExceeded(
                f"quarterly adaptation budget ({cfg.quarterly_budget}) already "
                f"reached for {quarter} ({existing} trials already recorded); "
                "pass force=True to override (GOVERNANCE.md §1.2)"
            )
        logger.warning(
            "trial_budget_exceeded_forced",
            quarter=quarter,
            existing=existing,
            budget=cfg.quarterly_budget,
            kind=kind,
        )
    trial_id = f"TR-{quarter}-{secrets.token_hex(3)}"
    store.insert_research_trial(
        ResearchTrial(
            trial_id=trial_id,
            kind=kind,
            description=description,
            created_at=now,
            quarter=quarter,
            status=TrialStatus.EXPERIMENTAL,
        )
    )
    return trial_id


# -- W9 2026Q3 ledger backfill (GOVERNANCE.md §11.1) -------------------------
#
# Measured 2026-09-19: `research_trials` held 0 rows although GOVERNANCE.md
# §11.1 documents six consumed 2026Q3 trials (`W9-2026Q3-001` .. `-006`,
# `state/registry/failed_hypotheses.json`). `new_trial_id` is the only
# writer of `research_trials` and nothing ever called it for these six --
# they were recorded straight into the failed-hypotheses graveyard by a
# different code path (`turboedge.learning.failed_hypotheses`), in its own
# format, with `quarter=null`. Effect: `count_research_trials_in_quarter`
# undercounts the spent 2026Q3 budget (GOVERNANCE.md §1.2) and any N_effective
# derived from `research_trials` for multiple-testing deflation would be too
# small. This backfill adds the missing rows under their *original*
# trial_ids (never rewritten to the `TR-YYYYQn-<hex>` format `new_trial_id`
# mints for new trials -- these six already have a documented, citable id).
_W9_BACKFILL_QUARTER = "2026Q3"
# GOVERNANCE.md §11.1: "W9 (2026-09-13)".
_W9_BACKFILL_CREATED_AT = datetime(2026, 9, 13, tzinfo=UTC)
_W9_BACKFILL_TRIALS: tuple[tuple[str, str], ...] = (
    ("W9-2026Q3-001", "voltarget_tsmom"),
    ("W9-2026Q3-002", "lowvol_regime_trend"),
    ("W9-2026Q3-003", "reversal_short_horizon"),
    ("W9-2026Q3-004", "vix_term_structure"),
    ("W9-2026Q3-005", "cross_asset_leadlag"),
    ("W9-2026Q3-006", "seasonality_turn_of_month"),
)


def _w9_backfill_description(feature: str) -> str:
    return (
        f"GOVERNANCE.md §11.1 W9 (2026-09-13) pre-registered challenger signal "
        f"family {feature!r}, one of the six 2026Q3 trials consumed against the "
        "quarterly adaptation budget (§1.2). Measured dormant (ladder rule not "
        "cleared, §11.3); full per-cell numbers in "
        "state/registry/failed_hypotheses.json and SIGNAL_REGISTRY.md §3.2."
    )


def backfill_w9_trials(store: Store, *, dry_run: bool = False) -> list[str]:
    """Idempotently backfill the six GOVERNANCE.md §11.1 W9 2026Q3 trials
    into ``research_trials``.

    For each of the six documented ``trial_id``s, inserts a
    :class:`~turboedge.storage.schemas.ResearchTrial` row (``quarter =
    "2026Q3"``, ``status = DORMANT``, ``kind = "feature"``, ``created_at =
    2026-09-13`` -- the W9 measurement date) *unless a row with that
    trial_id already exists* (``trial_id`` is ``research_trials``'s primary
    key), in which case that trial_id is silently skipped rather than
    raising or duplicating -- safe to call repeatedly (running it twice adds
    nothing the second time), and safe against a CI run starting from a
    fresh, empty database.

    Original trial_ids (``W9-2026Q3-001`` .. ``-006``) are preserved exactly
    as documented in GOVERNANCE.md §11.1 -- never rewritten into the
    ``new_trial_id``-minted ``TR-YYYYQn-<hex>`` format, which is reserved for
    trials created going forward.

    Args:
        store: Open ``Store`` the backfill is persisted through.
        dry_run: If ``True``, computes and returns which trial_ids are still
            missing (would be inserted) without writing anything.

    Returns:
        The trial_ids that were inserted (or, under ``dry_run``, would be).
    """
    result: list[str] = []
    for trial_id, feature in _W9_BACKFILL_TRIALS:
        if store.get_research_trial(trial_id) is not None:
            continue
        if not dry_run:
            store.insert_research_trial(
                ResearchTrial(
                    trial_id=trial_id,
                    kind="feature",
                    description=_w9_backfill_description(feature),
                    created_at=_W9_BACKFILL_CREATED_AT,
                    quarter=_W9_BACKFILL_QUARTER,
                    status=TrialStatus.DORMANT,
                )
            )
        result.append(trial_id)
    return result


def effective_number_of_trials(store: Store, quarter: str | None = None) -> int:
    """Count of research trials issued so far, optionally scoped to one
    quarter (``"2026Q3"``).

    This is the raw "how many hypotheses were tested" input that multiple
    testing correction (Master Spec §27.4, implemented in
    ``backtest/significance.py``) consumes; this module only supplies the
    count, not the correction itself.
    """
    if quarter is None:
        return len(store.list_research_trials())
    return store.count_research_trials_in_quarter(quarter)


__all__ = [
    "TrialBudgetExceeded",
    "TrialsConfig",
    "backfill_w9_trials",
    "effective_number_of_trials",
    "new_trial_id",
]
