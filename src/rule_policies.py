"""
Rule-based intervention policies for ablation comparison.

These use the SAME detector as PHI but with fixed action rules,
isolating the contribution of the learned RL policy.
"""

from src.intervention_actions import Action


class AlwaysTruncatePolicy:
    """Truncate whenever detector fires. No learning needed."""

    def select_action(self, confidence: float, **kwargs) -> Action:
        if confidence > kwargs.get("threshold", 0.5):
            return Action.TRUNCATE
        return Action.CONTINUE


class AlwaysBacktrackPolicy:
    """Backtrack whenever detector fires."""

    def select_action(self, confidence: float, **kwargs) -> Action:
        if confidence > kwargs.get("threshold", 0.5):
            return Action.BACKTRACK
        return Action.CONTINUE


class AlwaysRestartPolicy:
    """Restart whenever detector fires."""

    def select_action(self, confidence: float, **kwargs) -> Action:
        if confidence > kwargs.get("threshold", 0.5):
            return Action.RESTART
        return Action.CONTINUE


class ThresholdCascadePolicy:
    """Escalating response based on confidence level.
    Low confidence → truncate, medium → backtrack, high → restart.
    """

    def __init__(self, low=0.5, mid=0.7, high=0.9):
        self.low = low
        self.mid = mid
        self.high = high

    def select_action(self, confidence: float, **kwargs) -> Action:
        if confidence >= self.high:
            return Action.RESTART
        elif confidence >= self.mid:
            return Action.BACKTRACK
        elif confidence >= self.low:
            return Action.TRUNCATE
        return Action.CONTINUE


class EntropyBasedPolicy:
    """Use token-level entropy instead of learned detector.
    Baselines the value of the onset detector itself.
    """

    def __init__(self, entropy_threshold=2.0):
        self.entropy_threshold = entropy_threshold

    def select_action(self, entropy: float = 0.0, **kwargs) -> Action:
        if entropy > self.entropy_threshold * 1.5:
            return Action.BACKTRACK
        elif entropy > self.entropy_threshold:
            return Action.TRUNCATE
        return Action.CONTINUE


RULE_POLICIES = {
    "always_truncate": AlwaysTruncatePolicy(),
    "always_backtrack": AlwaysBacktrackPolicy(),
    "always_restart": AlwaysRestartPolicy(),
    "threshold_cascade": ThresholdCascadePolicy(),
    "entropy_based": EntropyBasedPolicy(),
}
