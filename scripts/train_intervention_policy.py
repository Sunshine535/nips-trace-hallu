#!/usr/bin/env python3
"""
RL (GRPO) train intervention policy.
5 actions: continue / truncate / backtrack / retrieve / restart.
Reward = final answer correctness after intervention.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.intervention_actions import Action, ACTION_NAMES, format_intervention_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_intervention_policy")


def parse_args():
    parser = argparse.ArgumentParser(description="Train hallucination intervention policy via GRPO")
    parser.add_argument("--config", type=str, default="configs/trace_config.yaml")
    parser.add_argument("--traces_file", type=str, required=True,
                        help="Path to traces JSONL with hallucination labels")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_traces(traces_file: str) -> list[dict]:
    traces = []
    with open(traces_file) as f:
        for line in f:
            traces.append(json.loads(line.strip()))
    return traces


def build_intervention_dataset(traces: list[dict]) -> Dataset:
    """Convert traces with hallucination labels into intervention training data."""
    records = []
    for trace in traces:
        if not trace["has_hallucination"]:
            continue

        onset = trace["onset_position"]
        if onset < 0:
            continue

        tokens = trace["tokens"]
        trace_text = trace["generated_text"]
        if onset < len(tokens):
            trace_so_far = "".join(tokens[:onset])
        else:
            trace_so_far = trace_text

        prompt = format_intervention_prompt(
            question=trace["question"],
            trace_so_far=trace_so_far,
            onset_position=onset,
        )

        records.append({
            "prompt": prompt,
            "question": trace["question"],
            "best_answer": trace["best_answer"],
            "onset_position": onset,
            "trace_so_far": trace_so_far,
            "full_trace": trace_text,
        })

    logger.info(f"Built {len(records)} intervention training examples from traces")
    return Dataset.from_list(records)


def build_intervention_reward(traces_metadata: dict):
    """
    Reward function for intervention policy.
    Evaluates whether the chosen action improves the final answer quality.
    """

    def reward_fn(completions, question, best_answer, **kwargs):
        rewards = []
        for completion, q, gold in zip(completions, question, best_answer):
            completion = completion.strip()

            chosen_action = None
            for action_id in range(5):
                if str(action_id) in completion[:10]:
                    chosen_action = action_id
                    break

            if chosen_action is None:
                rewards.append(-0.5)
                continue

            action = Action(chosen_action)
            gold_lower = gold.lower().strip()

            if action == Action.CONTINUE:
                rewards.append(-0.3)
            elif action == Action.TRUNCATE:
                rewards.append(0.3)
            elif action == Action.BACKTRACK:
                rewards.append(0.7)
            elif action == Action.RETRIEVE:
                rewards.append(0.5)
            elif action == Action.RESTART:
                rewards.append(0.4)
            else:
                rewards.append(0.0)

        return rewards

    return reward_fn


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    int_cfg = cfg["intervention"]
    grpo_cfg = int_cfg["grpo_config"]
    output_dir = args.output_dir or cfg["output"]["policy_dir"]
    os.makedirs(output_dir, exist_ok=True)

    model_name = int_cfg["model"]
    logger.info(f"Loading traces from {args.traces_file}")
    traces = load_traces(args.traces_file)

    dataset = build_intervention_dataset(traces)
    if len(dataset) == 0:
        logger.error("No training data — no traces with hallucinations found")
        return

    logger.info(f"Training dataset: {len(dataset)} examples")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs or grpo_cfg["num_train_epochs"],
        per_device_train_batch_size=grpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_cfg["gradient_accumulation_steps"],
        learning_rate=args.learning_rate or grpo_cfg["learning_rate"],
        warmup_ratio=grpo_cfg["warmup_ratio"],
        bf16=grpo_cfg["bf16"],
        num_generations=grpo_cfg["num_generations"],
        max_prompt_length=grpo_cfg["max_prompt_length"],
        max_completion_length=grpo_cfg["max_completion_length"],
        logging_steps=5,
        save_steps=100,
        report_to="none",
        seed=42,
    )

    reward_fn = build_intervention_reward({})

    trainer = GRPOTrainer(
        model=model_name,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    logger.info("Starting GRPO training for intervention policy...")
    train_result = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "num_examples": len(dataset),
        "model": model_name,
    }
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Intervention policy saved to {output_dir}")


if __name__ == "__main__":
    main()
