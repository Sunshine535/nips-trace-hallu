#!/usr/bin/env python3
"""
Ablation study framework for PHI.

Systematically ablates:
1. Detector: single-layer vs multi-layer, layer selection
2. Policy features: remove each feature dimension
3. Intervention actions: remove BACKTRACK, RESTART
4. Policy learning: rule-based vs PPO
5. Training: offline-only vs online-finetuned
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ablations")


ABLATION_CONFIGS = {
    "full_phi": {
        "description": "Full PHI system (no ablation)",
        "detector_type": "multi_layer",
        "layer_indices": [8, 16, 24, 32],
        "policy_type": "ppo",
        "num_actions": 4,
        "feature_mask": [1, 1, 1, 1, 1, 1],
    },

    "single_layer_8": {
        "description": "Detector: single probe at layer 8 only",
        "detector_type": "single_layer",
        "layer_indices": [8],
        "policy_type": "ppo",
        "num_actions": 4,
        "feature_mask": [1, 1, 1, 1, 1, 1],
    },
    "single_layer_16": {
        "description": "Detector: single probe at layer 16 only",
        "detector_type": "single_layer",
        "layer_indices": [16],
        "policy_type": "ppo",
        "num_actions": 4,
        "feature_mask": [1, 1, 1, 1, 1, 1],
    },
    "single_layer_24": {
        "description": "Detector: single probe at layer 24 only",
        "detector_type": "single_layer",
        "layer_indices": [24],
        "policy_type": "ppo",
        "num_actions": 4,
        "feature_mask": [1, 1, 1, 1, 1, 1],
    },
    "single_layer_32": {
        "description": "Detector: single probe at layer 32 only",
        "detector_type": "single_layer",
        "layer_indices": [32],
        "policy_type": "ppo",
        "num_actions": 4,
        "feature_mask": [1, 1, 1, 1, 1, 1],
    },
    "layers_early": {
        "description": "Detector: early layers only [8, 16]",
        "detector_type": "multi_layer",
        "layer_indices": [8, 16],
        "policy_type": "ppo",
        "num_actions": 4,
        "feature_mask": [1, 1, 1, 1, 1, 1],
    },
    "layers_late": {
        "description": "Detector: late layers only [24, 32]",
        "detector_type": "multi_layer",
        "layer_indices": [24, 32],
        "policy_type": "ppo",
        "num_actions": 4,
        "feature_mask": [1, 1, 1, 1, 1, 1],
    },

    "no_layer_agreement": {
        "description": "Feature ablation: remove layer_agreement",
        "detector_type": "multi_layer",
        "layer_indices": [8, 16, 24, 32],
        "policy_type": "ppo",
        "num_actions": 4,
        "feature_mask": [1, 0, 1, 1, 1, 1],
    },
    "no_query_complexity": {
        "description": "Feature ablation: remove query_complexity",
        "detector_type": "multi_layer",
        "layer_indices": [8, 16, 24, 32],
        "policy_type": "ppo",
        "num_actions": 4,
        "feature_mask": [1, 1, 1, 0, 1, 1],
    },
    "no_onset_position": {
        "description": "Feature ablation: remove onset_position_norm",
        "detector_type": "multi_layer",
        "layer_indices": [8, 16, 24, 32],
        "policy_type": "ppo",
        "num_actions": 4,
        "feature_mask": [1, 1, 1, 1, 0, 1],
    },
    "confidence_only": {
        "description": "Feature ablation: detector confidence only",
        "detector_type": "multi_layer",
        "layer_indices": [8, 16, 24, 32],
        "policy_type": "ppo",
        "num_actions": 4,
        "feature_mask": [1, 0, 0, 0, 0, 0],
    },

    "no_backtrack": {
        "description": "Action ablation: remove BACKTRACK (3 actions)",
        "detector_type": "multi_layer",
        "layer_indices": [8, 16, 24, 32],
        "policy_type": "ppo",
        "num_actions": 3,
        "feature_mask": [1, 1, 1, 1, 1, 1],
        "remove_action": "BACKTRACK",
    },
    "no_restart": {
        "description": "Action ablation: remove RESTART (3 actions)",
        "detector_type": "multi_layer",
        "layer_indices": [8, 16, 24, 32],
        "policy_type": "ppo",
        "num_actions": 3,
        "feature_mask": [1, 1, 1, 1, 1, 1],
        "remove_action": "RESTART",
    },
    "binary_only": {
        "description": "Action ablation: CONTINUE/TRUNCATE only (2 actions)",
        "detector_type": "multi_layer",
        "layer_indices": [8, 16, 24, 32],
        "policy_type": "ppo",
        "num_actions": 2,
        "feature_mask": [1, 1, 1, 1, 1, 1],
    },

    "rule_cascade": {
        "description": "Policy ablation: threshold cascade (no learning)",
        "detector_type": "multi_layer",
        "layer_indices": [8, 16, 24, 32],
        "policy_type": "threshold_cascade",
        "num_actions": 4,
        "feature_mask": [1, 1, 1, 1, 1, 1],
    },
    "rule_always_truncate": {
        "description": "Policy ablation: always truncate at onset",
        "detector_type": "multi_layer",
        "layer_indices": [8, 16, 24, 32],
        "policy_type": "always_truncate",
        "num_actions": 4,
        "feature_mask": [1, 1, 1, 1, 1, 1],
    },
    "rule_always_backtrack": {
        "description": "Policy ablation: always backtrack at onset",
        "detector_type": "multi_layer",
        "layer_indices": [8, 16, 24, 32],
        "policy_type": "always_backtrack",
        "num_actions": 4,
        "feature_mask": [1, 1, 1, 1, 1, 1],
    },

    "offline_only": {
        "description": "Training ablation: offline PPO only (no online finetuning)",
        "detector_type": "multi_layer",
        "layer_indices": [8, 16, 24, 32],
        "policy_type": "ppo_offline",
        "num_actions": 4,
        "feature_mask": [1, 1, 1, 1, 1, 1],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", type=str, nargs="+", default=None,
                        help="Specific ablation names to run (default: all)")
    parser.add_argument("--output_dir", type=str, default="./results/ablations")
    parser.add_argument("--dataset", type=str, default="truthfulqa")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list_only", action="store_true",
                        help="Just list available ablations")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_only:
        print("\nAvailable ablation configurations:")
        print("=" * 70)
        for name, config in ABLATION_CONFIGS.items():
            print(f"  {name:30s} — {config['description']}")
        return

    ablations = args.ablations or list(ABLATION_CONFIGS.keys())
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Running {len(ablations)} ablation experiments")
    logger.info(f"Dataset: {args.dataset}, Samples: {args.num_samples}, Seed: {args.seed}")

    results = {}
    for name in ablations:
        if name not in ABLATION_CONFIGS:
            logger.warning(f"Unknown ablation: {name}, skipping")
            continue

        config = ABLATION_CONFIGS[name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Ablation: {name}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"{'='*60}")

        results[name] = {
            "config": config,
            "status": "config_ready",
            "note": "Run with full model loaded on GPU to execute",
        }

    out_path = os.path.join(args.output_dir, "ablation_configs.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nAblation configs saved to {out_path}")
    logger.info(f"Total configs: {len(results)}")

    print("\n" + "=" * 70)
    print("ABLATION SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Name':30s} {'Detector':12s} {'Policy':15s} {'Actions':8s} {'Features':10s}")
    print("-" * 70)
    for name, r in results.items():
        c = r["config"]
        feat_count = sum(c["feature_mask"])
        print(f"{name:30s} {c['detector_type']:12s} {c['policy_type']:15s} "
              f"{c['num_actions']:<8d} {feat_count}/6")


if __name__ == "__main__":
    main()
