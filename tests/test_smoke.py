"""Smoke tests: verify all modules import and key invariants hold.

Run: pytest tests/test_smoke.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


class TestImports:
    """Every public module must import without GPU or data."""

    def test_onset_detector(self):
        from src.onset_detector import OnsetDetectorConfig, OnsetLinearProbe, MultiLayerOnsetDetector

    def test_intervention_actions(self):
        from src.intervention_actions import Action, InterventionConfig, InterventionExecutor

    def test_claim_labeler(self):
        from src.claim_labeler import ClaimLabelerConfig, ClaimExtractor

    def test_factuality_eval(self):
        from src.factuality_eval import FactualityConfig, FactualityEvaluator

    def test_detector_calibration(self):
        from src.detector_calibration import compute_auprc, compute_ece

    def test_completeness_eval(self):
        from src.completeness_eval import compute_completeness, compute_helpfulness

    def test_budget_eval(self):
        from src.budget_eval import enforce_token_budget, compute_pareto_front

    def test_baselines(self):
        from src.baselines import DoLaDecoder, ITIDecoder, SelfCheckGPT

    def test_rule_policies(self):
        from src.rule_policies import RULE_POLICIES
        assert len(RULE_POLICIES) >= 4

    def test_top_level_init(self):
        import src
        assert hasattr(src, "OnsetDetectorConfig")
        assert hasattr(src, "Action")


class TestConfigUtils:
    def test_load_config(self):
        from config_utils import load_config
        cfg = load_config(str(ROOT / "configs" / "trace_config.yaml"))
        assert "generator" in cfg
        assert "detector" in cfg
        assert "intervention" in cfg
        assert "eval" in cfg

    def test_load_missing_returns_empty(self):
        from config_utils import load_config
        assert load_config("/nonexistent/path.yaml") == {}

    def test_apply_config_defaults(self):
        import argparse
        from config_utils import load_config, apply_config_defaults

        cfg = load_config(str(ROOT / "configs" / "trace_config.yaml"))
        parser = argparse.ArgumentParser()
        parser.add_argument("--learning_rate", type=float, default=None)
        parser.add_argument("--num_epochs", type=int, default=None)
        args = parser.parse_args([])

        apply_config_defaults(args, "train_onset_detector", cfg)
        assert args.learning_rate == 1e-3
        assert args.num_epochs == 20

    def test_cli_overrides_config(self):
        import argparse
        from config_utils import load_config, apply_config_defaults

        cfg = load_config(str(ROOT / "configs" / "trace_config.yaml"))
        parser = argparse.ArgumentParser()
        parser.add_argument("--learning_rate", type=float, default=None)
        args = parser.parse_args(["--learning_rate", "0.01"])

        apply_config_defaults(args, "train_onset_detector", cfg)
        assert args.learning_rate == 0.01


class TestActionEnum:
    def test_five_actions(self):
        from src.intervention_actions import Action
        assert len(Action) == 5

    def test_action_names(self):
        from src.intervention_actions import ACTION_NAMES
        expected = {"continue", "truncate", "backtrack", "retrieve", "restart"}
        assert set(ACTION_NAMES.values()) == expected
