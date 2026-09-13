"""Self-learning subsystem: Forward Ledger, labeling, counterfactual
learning, Bayesian strategy memory, model registry, ensemble weighting,
research trials, and concept drift detection (Master Spec §20-27, §46).

See ``CONTRACT_v2.md`` (W6) and Master Spec §20-27 / §30 / §46 / §48 for the
governing rules. Source code itself stays versioned and human-reviewed --
"learning" here means model weights, Bayesian posteriors, the model
registry, and champion/challenger promotion, never autonomous code changes
(Master Spec §20).
"""

from turboedge.learning.counterfactual import CounterfactualResult, evaluate_counterfactual
from turboedge.learning.drift import PageHinkley, PageHinkleyConfig, record_drift_event
from turboedge.learning.ensemble_weights import update_weights
from turboedge.learning.failed_hypotheses import (
    FailedHypothesesConfig,
    FailedHypothesis,
    allow_retest,
    is_dormant,
    new_hypothesis,
)
from turboedge.learning.failed_hypotheses import (
    append as append_failed_hypothesis,
)
from turboedge.learning.failed_hypotheses import (
    load as load_failed_hypotheses,
)
from turboedge.learning.labeler import (
    ExitResolution,
    LabelerConfig,
    LabelRunResult,
    label_due_entries,
    resolve_exit,
    resolve_exit_with_fallback,
)
from turboedge.learning.ledger import ForwardLedger, select_shadow_sample
from turboedge.learning.posterior import (
    PosteriorConfig,
    StrategyPosterior,
    bootstrap_prior_from_backtest,
)
from turboedge.learning.registry import ModelRegistry, promote_if_ladder
from turboedge.learning.trials import (
    TrialBudgetExceeded,
    TrialsConfig,
    effective_number_of_trials,
    new_trial_id,
)

__all__ = [
    "CounterfactualResult",
    "ExitResolution",
    "FailedHypothesesConfig",
    "FailedHypothesis",
    "ForwardLedger",
    "LabelRunResult",
    "LabelerConfig",
    "ModelRegistry",
    "PageHinkley",
    "PageHinkleyConfig",
    "PosteriorConfig",
    "StrategyPosterior",
    "TrialBudgetExceeded",
    "TrialsConfig",
    "allow_retest",
    "append_failed_hypothesis",
    "bootstrap_prior_from_backtest",
    "effective_number_of_trials",
    "evaluate_counterfactual",
    "is_dormant",
    "label_due_entries",
    "load_failed_hypotheses",
    "new_hypothesis",
    "new_trial_id",
    "promote_if_ladder",
    "record_drift_event",
    "resolve_exit",
    "resolve_exit_with_fallback",
    "select_shadow_sample",
    "update_weights",
]
