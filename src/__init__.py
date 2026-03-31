"""
PHI: Predictive Hallucination Intervention via Onset Detection and Graduated Response.

Core library for hallucination onset detection, intervention policy,
claim-level labeling, factuality evaluation, and baseline methods.
"""

from src.onset_detector import (
    OnsetDetectorConfig,
    OnsetLinearProbe,
    MultiLayerOnsetDetector,
    find_onset_positions,
)
from src.intervention_actions import (
    Action,
    ACTION_NAMES,
    InterventionConfig,
    InterventionExecutor,
)
from src.claim_labeler import (
    ClaimLabelerConfig,
    ClaimExtractor,
    NLILabeler,
    ClaimLevelLabeler,
)
from src.factuality_eval import (
    FactualityConfig,
    FactualityEvaluator,
    paired_bootstrap_test,
)
from src.detector_calibration import (
    CalibrationMetrics,
    compute_auprc,
    compute_ece,
    compute_lead_time,
    evaluate_detector_calibration,
)
from src.completeness_eval import (
    CompletenessMetrics,
    compute_completeness,
    compute_helpfulness,
    evaluate_completeness_batch,
)
from src.budget_eval import (
    BudgetPoint,
    enforce_token_budget,
    compute_pareto_front,
    evaluate_under_budget,
)
from src.baselines import DoLaDecoder, ITIDecoder, SelfCheckGPT
from src.rule_policies import RULE_POLICIES

__all__ = [
    "OnsetDetectorConfig", "OnsetLinearProbe", "MultiLayerOnsetDetector", "find_onset_positions",
    "Action", "ACTION_NAMES", "InterventionConfig", "InterventionExecutor",
    "ClaimLabelerConfig", "ClaimExtractor", "NLILabeler", "ClaimLevelLabeler",
    "FactualityConfig", "FactualityEvaluator", "paired_bootstrap_test",
    "CalibrationMetrics", "compute_auprc", "compute_ece", "compute_lead_time",
    "evaluate_detector_calibration",
    "CompletenessMetrics", "compute_completeness", "compute_helpfulness",
    "evaluate_completeness_batch",
    "BudgetPoint", "enforce_token_budget", "compute_pareto_front", "evaluate_under_budget",
    "DoLaDecoder", "ITIDecoder", "SelfCheckGPT",
    "RULE_POLICIES",
]
