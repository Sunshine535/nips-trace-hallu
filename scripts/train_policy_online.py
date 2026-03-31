#!/usr/bin/env python3
"""
Online RL policy training for PHI.

Key difference from train_intervention_policy.py:
- Actions are ACTUALLY EXECUTED on the LLM during training
- Rewards come from REAL factuality evaluation of intervened outputs
- Solves the offline/online mismatch flagged in review

Training loop:
1. Generate text with onset detection (real hidden states)
2. Policy selects action based on actual detector output
3. Execute intervention (truncate/backtrack/restart)
4. Evaluate factuality of result via NLI
5. PPO update with real rewards
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.intervention_actions import Action, InterventionConfig, InterventionExecutor
from src.onset_detector import OnsetDetectorConfig, OnsetLinearProbe, MultiLayerOnsetDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_policy_online")


def parse_args():
    parser = argparse.ArgumentParser(description="Train intervention policy with online rollouts")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--detector_path", type=str, required=True)
    parser.add_argument("--detector_type", type=str, default="multi_layer")
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/online_policy")
    parser.add_argument("--dataset", type=str, default="truthfulqa",
                        choices=["truthfulqa", "halueval"])
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Samples per epoch for online training")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_cost", type=float, default=0.15,
                        help="Weight for action cost in reward")
    parser.add_argument("--pretrained_policy", type=str, default=None,
                        help="Path to pretrained offline policy for warm start")
    return parser.parse_args()


sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_intervention_policy import InterventionPolicyMLP


ACTION_COSTS = {
    Action.CONTINUE: 0.0,
    Action.TRUNCATE: 0.05,
    Action.BACKTRACK: 0.3,
    Action.RETRIEVE: 0.5,
    Action.RESTART: 0.8,
}


def load_dataset_samples(dataset_name, max_samples):
    from datasets import load_dataset
    if dataset_name == "truthfulqa":
        ds = load_dataset("truthful_qa", "generation", split="validation")
        samples = [{
            "question": ex["question"],
            "correct_answers": ex.get("correct_answers", [ex.get("best_answer", "")]),
            "incorrect_answers": ex.get("incorrect_answers", []),
        } for ex in ds]
    elif dataset_name == "halueval":
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
        samples = [{
            "question": ex.get("question", ""),
            "correct_answers": [ex.get("answer", "")],
            "incorrect_answers": [ex.get("hallucinated_answer", "")],
        } for ex in ds]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if max_samples and max_samples < len(samples):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[i] for i in indices]

    return samples


@torch.no_grad()
def online_rollout_step(
    model, tokenizer, detector, executor, policy,
    sample, layer_indices, args, device,
):
    """Single online rollout: generate → detect → act → evaluate."""
    prompt = f"Question: {sample['question']}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]

    generated_ids = prompt_ids.clone()
    onset_pos = -1
    max_conf = 0.0
    per_layer_confs = []

    for step in range(args.max_new_tokens):
        outputs = model(generated_ids, output_hidden_states=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        if onset_pos < 0 and step > 3:
            if args.detector_type == "multi_layer":
                all_hs = {idx: outputs.hidden_states[idx]
                          for idx in layer_indices if idx < len(outputs.hidden_states)}
                det_out = detector(all_hs)
            else:
                hs = outputs.hidden_states[layer_indices[0]]
                det_out = detector(hs)

            probs = torch.softmax(det_out["logits"], dim=-1)[:, -1, 1]
            conf = probs.item()
            max_conf = max(max_conf, conf)
            per_layer_confs.append(conf)

            if conf > args.threshold:
                onset_pos = prompt_len + step

    gen_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
    gen_len = generated_ids.shape[1] - prompt_len

    gen_norm = min(gen_len / 512.0, 1.0)
    q_complexity = min(len(sample["question"].split()) / 50.0, 1.0)
    onset_norm = onset_pos / max(gen_len, 1) if onset_pos >= 0 else 1.0
    hallu_density = max_conf * 0.5

    state = torch.tensor([
        max_conf,
        gen_norm,
        q_complexity,
        onset_norm,
        hallu_density,
    ], dtype=torch.float32, device=device).unsqueeze(0)

    action_idx, log_prob, entropy, value = policy.get_action(state)
    action = Action(action_idx.item())

    if onset_pos > 0 and action != Action.CONTINUE:
        result = executor.execute(action, generated_ids, onset_pos, original_prompt_ids=prompt_ids)
        final_text = tokenizer.decode(result["new_ids"][0, prompt_len:], skip_special_tokens=True)
    else:
        final_text = gen_text

    from src.claim_labeler import ClaimExtractor
    from src.factuality_eval import FactualityConfig, FactualityEvaluator

    try:
        evaluator = FactualityEvaluator(FactualityConfig())
        fact_result = evaluator.evaluate_single(
            final_text, sample["correct_answers"], sample.get("incorrect_answers", [])
        )
        factuality = fact_result["factuality_score"]
    except Exception:
        gen_lower = final_text.lower()
        factuality = 0.5
        for ans in sample["correct_answers"]:
            if ans and ans.lower() in gen_lower:
                factuality = 1.0
                break

    from src.completeness_eval import compute_completeness, compute_helpfulness

    try:
        completeness = compute_completeness(final_text, sample["correct_answers"], use_nli=False)
    except Exception:
        completeness = min(len(final_text.split()) / 50.0, 1.0)

    helpfulness = compute_helpfulness(factuality, completeness)

    cost = ACTION_COSTS.get(action, 0.0)
    reward = helpfulness - args.lambda_cost * cost

    return {
        "state": state.squeeze(0).cpu().numpy(),
        "action": action_idx.item(),
        "log_prob": log_prob.item(),
        "value": value.item(),
        "reward": reward,
        "factuality": factuality,
        "completeness": completeness,
        "helpfulness": helpfulness,
        "action_name": action.name,
        "onset_detected": onset_pos >= 0,
        "final_text_len": len(final_text.split()),
    }


def ppo_update(policy, optimizer, rollouts, args, device):
    """PPO update from online rollouts."""
    states = torch.tensor(np.array([r["state"] for r in rollouts]), dtype=torch.float32, device=device)
    actions = torch.tensor([r["action"] for r in rollouts], dtype=torch.long, device=device)
    old_log_probs = torch.tensor([r["log_prob"] for r in rollouts], dtype=torch.float32, device=device)
    rewards = torch.tensor([r["reward"] for r in rollouts], dtype=torch.float32, device=device)
    values = torch.tensor([r["value"] for r in rollouts], dtype=torch.float32, device=device)

    returns = rewards.clone()
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_loss = 0.0
    n = 0
    for _ in range(args.ppo_epochs):
        new_log_probs, entropy, new_values = policy.evaluate_action(states, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        value_loss = F.mse_loss(new_values, returns)
        entropy_loss = -entropy.mean()

        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    logger.info(f"Loading detector: {args.detector_path}")
    det_config = OnsetDetectorConfig(hidden_size=args.hidden_size)
    if args.detector_type == "multi_layer":
        detector = MultiLayerOnsetDetector(det_config, args.layer_indices)
    else:
        detector = OnsetLinearProbe(det_config)
    detector.load_state_dict(torch.load(args.detector_path, map_location="cpu"))
    detector = detector.to(device)
    detector.eval()

    int_config = InterventionConfig(max_new_tokens=args.max_new_tokens)
    executor = InterventionExecutor(model, tokenizer, int_config)

    policy = InterventionPolicyMLP(input_dim=5, hidden_dim=128, num_actions=5).to(device)

    if args.pretrained_policy:
        logger.info(f"Warm-starting from offline policy: {args.pretrained_policy}")
        try:
            offline_state = torch.load(args.pretrained_policy, map_location="cpu")
            compatible = {}
            for k, v in offline_state.items():
                if k in policy.state_dict() and v.shape == policy.state_dict()[k].shape:
                    compatible[k] = v
            policy.load_state_dict(compatible, strict=False)
            logger.info(f"  Loaded {len(compatible)}/{len(offline_state)} parameters")
        except Exception as e:
            logger.warning(f"  Could not load offline policy: {e}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    logger.info(f"Loading dataset: {args.dataset}")
    samples = load_dataset_samples(args.dataset, args.max_samples)
    logger.info(f"  {len(samples)} samples")

    best_reward = -float("inf")
    training_log = []
    start_epoch = 0

    latest_ckpt = sorted(
        [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint_epoch") and f.endswith(".pt")],
        key=lambda x: os.path.getmtime(os.path.join(args.output_dir, x)),
    ) if os.path.isdir(args.output_dir) else []
    if latest_ckpt:
        ckpt_path = os.path.join(args.output_dir, latest_ckpt[-1])
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        policy.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_reward = ckpt.get("best_reward", -float("inf"))
        training_log = ckpt.get("training_log", [])
        logger.info(f"  Resumed at epoch {start_epoch}, best_reward={best_reward:.4f}")

    for epoch in range(start_epoch, args.num_epochs):
        policy.train()
        rollouts = []
        action_counts = {a.name: 0 for a in Action}

        np.random.shuffle(samples)
        epoch_start = time.time()

        for sample in tqdm(samples, desc=f"Epoch {epoch}", leave=False):
            try:
                result = online_rollout_step(
                    model, tokenizer, detector, executor, policy,
                    sample, args.layer_indices, args, device,
                )
                rollouts.append(result)
                action_counts[result["action_name"]] += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

        if not rollouts:
            logger.warning(f"Epoch {epoch}: no successful rollouts")
            continue

        loss = ppo_update(policy, optimizer, rollouts, args, device)

        avg_reward = np.mean([r["reward"] for r in rollouts])
        avg_fact = np.mean([r["factuality"] for r in rollouts])
        avg_comp = np.mean([r.get("completeness", 0.5) for r in rollouts])
        avg_help = np.mean([r.get("helpfulness", 0.5) for r in rollouts])
        onset_rate = np.mean([r["onset_detected"] for r in rollouts])
        epoch_time = time.time() - epoch_start

        log_entry = {
            "epoch": epoch,
            "avg_reward": float(avg_reward),
            "avg_factuality": float(avg_fact),
            "avg_completeness": float(avg_comp),
            "avg_helpfulness": float(avg_help),
            "onset_detection_rate": float(onset_rate),
            "loss": float(loss),
            "action_distribution": {k: v / max(len(rollouts), 1) for k, v in action_counts.items()},
            "n_rollouts": len(rollouts),
            "epoch_time_s": epoch_time,
        }
        training_log.append(log_entry)

        logger.info(
            f"Epoch {epoch:3d}: reward={avg_reward:.4f} fact={avg_fact:.4f} "
            f"comp={avg_comp:.4f} help={avg_help:.4f} "
            f"onset_rate={onset_rate:.2f} loss={loss:.4f} "
            f"time={epoch_time:.0f}s rollouts={len(rollouts)}"
        )
        logger.info(f"  Actions: {log_entry['action_distribution']}")

        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(policy.state_dict(), os.path.join(args.output_dir, "best_policy.pt"))

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_reward": best_reward,
                "training_log": training_log,
            }, os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}.pt"))

    torch.save(policy.state_dict(), os.path.join(args.output_dir, "final_policy.pt"))
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    summary = {
        "best_avg_reward": float(best_reward),
        "final_avg_reward": float(avg_reward),
        "final_avg_factuality": float(avg_fact),
        "num_epochs": args.num_epochs,
        "training_mode": "online",
        "dataset": args.dataset,
        "model": args.model_name,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nOnline training complete. Best reward: {best_reward:.4f}")


if __name__ == "__main__":
    main()
