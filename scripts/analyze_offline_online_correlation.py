#!/usr/bin/env python3
"""
Analyze correlation between offline simulated rewards and online factuality gains.
Validates the offline-to-online transfer assumption.

Reads offline training log and online evaluation results,
computes Spearman/Pearson correlations, and generates report.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("analyze_correlation")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline_log", type=str, required=True,
                        help="Path to offline training_log.json")
    parser.add_argument("--online_results", type=str, required=True,
                        help="Path to online chi_evaluation.json")
    parser.add_argument("--online_log", type=str, default=None,
                        help="Path to online_training_log.json (if available)")
    parser.add_argument("--output_dir", type=str, default="./results/correlation")
    return parser.parse_args()


def spearman_rank_correlation(x, y):
    """Compute Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    rank_x = np.argsort(np.argsort(x)).astype(float)
    rank_y = np.argsort(np.argsort(y)).astype(float)

    d = rank_x - rank_y
    d_sq = np.sum(d ** 2)
    rho = 1 - (6 * d_sq) / (n * (n ** 2 - 1))

    t = rho * np.sqrt((n - 2) / (1 - rho ** 2 + 1e-10))
    from scipy import stats
    p_value = 2 * stats.t.sf(abs(t), n - 2)

    return float(rho), float(p_value)


def pearson_correlation(x, y):
    """Compute Pearson correlation."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.sum((x - mean_x) * (y - mean_y))
    std_x = np.sqrt(np.sum((x - mean_x) ** 2))
    std_y = np.sqrt(np.sum((y - mean_y) ** 2))

    r = cov / (std_x * std_y + 1e-10)

    from scipy import stats
    t = r * np.sqrt((n - 2) / (1 - r ** 2 + 1e-10))
    p_value = 2 * stats.t.sf(abs(t), n - 2)

    return float(r), float(p_value)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.offline_log) as f:
        offline_log = json.load(f)

    with open(args.online_results) as f:
        online_results = json.load(f)

    report = {
        "offline_training_epochs": len(offline_log),
        "datasets_evaluated": list(online_results.keys()),
    }

    offline_rewards = [e["avg_reward"] for e in offline_log]
    report["offline_reward_trajectory"] = {
        "start": offline_rewards[0] if offline_rewards else None,
        "end": offline_rewards[-1] if offline_rewards else None,
        "max": max(offline_rewards) if offline_rewards else None,
        "improvement": (offline_rewards[-1] - offline_rewards[0]) if len(offline_rewards) > 1 else 0,
    }

    for ds_name, metrics in online_results.items():
        if not isinstance(metrics, dict):
            continue

        baseline_fact = metrics.get("no_intervention", {}).get("factuality", 0)
        chi_fact = metrics.get("chi_ours", {}).get("factuality", 0)
        improvement = chi_fact - baseline_fact

        report[f"online_{ds_name}"] = {
            "baseline_factuality": baseline_fact,
            "chi_factuality": chi_fact,
            "improvement": improvement,
        }

    if args.online_log and os.path.exists(args.online_log):
        with open(args.online_log) as f:
            online_log = json.load(f)

        online_rewards = [e.get("avg_reward", 0) for e in online_log]
        online_facts = [e.get("avg_factuality", 0) for e in online_log]

        min_len = min(len(offline_rewards), len(online_rewards))
        if min_len >= 3:
            rho, p_rho = spearman_rank_correlation(
                np.array(offline_rewards[:min_len]),
                np.array(online_rewards[:min_len]),
            )
            r, p_r = pearson_correlation(
                np.array(offline_rewards[:min_len]),
                np.array(online_rewards[:min_len]),
            )

            report["correlation_analysis"] = {
                "n_matched_epochs": min_len,
                "spearman_rho": rho,
                "spearman_p": p_rho,
                "pearson_r": r,
                "pearson_p": p_r,
                "correlation_significant": p_rho < 0.05,
                "interpretation": (
                    "Strong positive correlation validates offline-to-online transfer"
                    if rho > 0.5 and p_rho < 0.05
                    else "Weak or non-significant correlation — offline policy may not transfer well"
                ),
            }

            reward_to_fact_rho, _ = spearman_rank_correlation(
                np.array(online_rewards[:min_len]),
                np.array(online_facts[:min_len]),
            )
            report["reward_factuality_correlation"] = {
                "spearman_rho": reward_to_fact_rho,
                "interpretation": (
                    "Reward function well-aligned with factuality"
                    if reward_to_fact_rho > 0.5
                    else "Reward function may not directly optimize factuality"
                ),
            }

    out_path = os.path.join(args.output_dir, "offline_online_correlation.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Correlation analysis saved to {out_path}")
    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
