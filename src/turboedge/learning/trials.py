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
    "effective_number_of_trials",
    "new_trial_id",
]
