#!/usr/bin/env python3
"""
Train RL intervention policy using PPO with a small MLP.

5 actions: CONTINUE, TRUNCATE, BACKTRACK, RETRIEVE, RESTART
State: detector confidence + generation length + query complexity
Reward: factuality_score - cost(action)
Train on mix of TruthfulQA + HaluEval traces.
"""

import argparse
import glob
import json
import logging
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.intervention_actions import Action, ACTION_NAMES


def save_training_checkpoint(path, model, optimizer, epoch, step, **extra):
    torch.save({"epoch": epoch, "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                **extra}, path)


def load_training_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0)


def find_latest_checkpoint(output_dir, pattern="checkpoint_*.pt"):
    ckpts = sorted(glob.glob(os.path.join(output_dir, pattern)),
                   key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_intervention_policy")


def parse_args():
    parser = argparse.ArgumentParser(description="Train intervention policy via PPO")
    parser.add_argument("--config", type=str, default="configs/trace_config.yaml")
    parser.add_argument("--traces_dir", type=str, required=True,
                        help="Directory with collected traces")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["truthfulqa", "halueval"])
    parser.add_argument("--output_dir", type=str, default="./checkpoints/intervention_policy")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--value_coeff", type=float, default=0.5)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="'auto' to resume from latest checkpoint, or path")
    return parser.parse_args()


# ── MLP Policy ───────────────────────────────────────────────────────────────

class InterventionPolicyMLP(nn.Module):
    """
    MLP policy for intervention action selection.
    Input: [detector_confidence, generation_length_norm, query_complexity,
            onset_position_norm, hallucination_density]
    Output: action logits (5) + value (1)
    """

    def __init__(self, input_dim=5, hidden_dim=128, num_actions=5):
        super().__init__()
        self.policy_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_actions),
        )
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        logits = self.policy_head(state)
        value = self.value_head(state).squeeze(-1)
        return logits, value

    def get_action(self, state, temperature=1.0):
        logits, value = self.forward(state)
        probs = F.softmax(logits / temperature, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_action(self, state, action):
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        return dist.log_prob(action), dist.entropy(), value


# ── Environment ──────────────────────────────────────────────────────────────

ACTION_COSTS = {
    Action.CONTINUE: 0.0,
    Action.TRUNCATE: 0.05,
    Action.BACKTRACK: 0.3,
    Action.RETRIEVE: 0.5,
    Action.RESTART: 0.8,
}


def extract_state_features(trace: dict, detector_confidence: float = None) -> np.ndarray:
    """Extract state features from a trace for the policy."""
    tokens = trace.get("tokens", [])
    hallu_labels = trace.get("hallu_labels", [])
    onset_pos = trace.get("onset_position", -1)

    gen_length_norm = min(len(tokens) / 512.0, 1.0)
    query_len = len(trace.get("question", "").split())
    query_complexity = min(query_len / 50.0, 1.0)

    if onset_pos >= 0 and len(tokens) > 0:
        onset_norm = onset_pos / max(len(tokens), 1)
    else:
        onset_norm = 1.0

    if hallu_labels:
        hallu_density = sum(hallu_labels) / max(len(hallu_labels), 1)
    else:
        hallu_density = 0.0

    if detector_confidence is None:
        detector_confidence = hallu_density * 0.8 + np.random.uniform(0, 0.2)

    return np.array([
        detector_confidence,
        gen_length_norm,
        query_complexity,
        onset_norm,
        hallu_density,
    ], dtype=np.float32)


def compute_reward(trace: dict, action: Action) -> float:
    """
    Reward = factuality_score - cost(action).
    Factuality is estimated from hallucination labels and the action's expected effect.
    """
    has_hallu = trace.get("has_hallucination", False)
    hallu_labels = trace.get("hallu_labels", [])
    hallu_rate = sum(hallu_labels) / max(len(hallu_labels), 1) if hallu_labels else 0.0

    base_factuality = 1.0 - hallu_rate

    if not has_hallu:
        if action == Action.CONTINUE:
            factuality = base_factuality
        else:
            factuality = base_factuality - 0.1
    else:
        if action == Action.CONTINUE:
            factuality = base_factuality
        elif action == Action.TRUNCATE:
            factuality = min(base_factuality + 0.3, 1.0)
        elif action == Action.BACKTRACK:
            factuality = min(base_factuality + 0.5, 1.0)
        elif action == Action.RETRIEVE:
            factuality = min(base_factuality + 0.4, 1.0)
        elif action == Action.RESTART:
            factuality = 0.5 + np.random.uniform(-0.1, 0.1)
        else:
            factuality = base_factuality

    cost = ACTION_COSTS[action]
    reward = factuality - cost
    return reward


# ── PPO Training ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95):
        returns = []
        advantages = []
        gae = 0.0

        values = self.values + [0.0]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return returns, advantages

    def get_tensors(self, device):
        returns, advantages = self.compute_returns_and_advantages()
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        return states, actions, old_log_probs, returns_t, advantages_t

    def clear(self):
        self.__init__()


def ppo_update(policy, optimizer, buffer, args, device):
    """Run PPO update on the collected rollout buffer."""
    states, actions, old_log_probs, returns, advantages = buffer.get_tensors(device)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(args.ppo_epochs):
        indices = torch.randperm(len(states), device=device)

        for start in range(0, len(states), args.batch_size):
            end = min(start + args.batch_size, len(states))
            idx = indices[start:end]

            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_old_lp = old_log_probs[idx]
            batch_returns = returns[idx]
            batch_adv = advantages[idx]

            new_log_probs, entropy, values = policy.evaluate_action(batch_states, batch_actions)

            ratio = torch.exp(new_log_probs - batch_old_lp)
            clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
            policy_loss = -torch.min(ratio * batch_adv, clipped_ratio * batch_adv).mean()

            value_loss = F.mse_loss(values, batch_returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + args.value_coeff * value_loss + args.entropy_coeff * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss": total_value_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    traces = []
    for ds_name in args.datasets:
        jsonl_path = os.path.join(args.traces_dir, f"traces_{ds_name}.jsonl")
        if not os.path.exists(jsonl_path):
            logger.warning(f"Missing {jsonl_path}, skipping")
            continue
        with open(jsonl_path) as f:
            ds_traces = [json.loads(line.strip()) for line in f]
        traces.extend(ds_traces)
        logger.info(f"Loaded {len(ds_traces)} traces from {ds_name}")

    if not traces:
        logger.error("No traces found. Run collect_traces.py first.")
        return

    hallu_traces = [t for t in traces if t.get("has_hallucination", False)]
    normal_traces = [t for t in traces if not t.get("has_hallucination", False)]
    logger.info(f"Total traces: {len(traces)} (hallucinated: {len(hallu_traces)}, normal: {len(normal_traces)})")

    policy = InterventionPolicyMLP(input_dim=5, hidden_dim=args.hidden_dim, num_actions=5).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    logger.info(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    best_avg_reward = -float("inf")
    training_log = []
    reward_window = deque(maxlen=500)
    start_epoch = 0

    if args.resume_from_checkpoint:
        ckpt_path = find_latest_checkpoint(args.output_dir, "checkpoint_epoch*.pt")
        if ckpt_path:
            logger.info("Resuming from %s", ckpt_path)
            start_epoch, _ = load_training_checkpoint(ckpt_path, policy, optimizer)
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            best_avg_reward = ckpt_data.get("best_avg_reward", -float("inf"))
            training_log = ckpt_data.get("training_log", [])
            if ckpt_data.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])
            logger.info("  Resuming from epoch %d", start_epoch)

    for epoch in range(start_epoch, args.num_epochs):
        buffer = RolloutBuffer()
        epoch_rewards = []
        action_counts = {a.name: 0 for a in Action}

        np.random.shuffle(traces)

        for trace in traces:
            state_features = extract_state_features(trace)
            state_tensor = torch.tensor(state_features, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action_idx, log_prob, entropy, value = policy.get_action(state_tensor)

            action = Action(action_idx.item())
            reward = compute_reward(trace, action)

            buffer.add(
                state=state_features,
                action=action_idx.item(),
                log_prob=log_prob.item(),
                reward=reward,
                value=value.item(),
                done=True,
            )

            epoch_rewards.append(reward)
            reward_window.append(reward)
            action_counts[action.name] += 1

        update_info = ppo_update(policy, optimizer, buffer, args, device)
        scheduler.step()
        buffer.clear()

        avg_reward = np.mean(epoch_rewards)
        avg_running = np.mean(list(reward_window))

        action_dist = {k: v / max(len(traces), 1) for k, v in action_counts.items()}

        epoch_log = {
            "epoch": epoch,
            "avg_reward": avg_reward,
            "running_avg_reward": avg_running,
            "action_distribution": action_dist,
            **update_info,
        }
        training_log.append(epoch_log)

        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            logger.info(
                f"Epoch {epoch:4d}: reward={avg_reward:.4f} (running={avg_running:.4f}) "
                f"ploss={update_info['policy_loss']:.4f} vloss={update_info['value_loss']:.4f} "
                f"entropy={update_info['entropy']:.4f}"
            )
            logger.info(f"  Actions: {action_dist}")

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(policy.state_dict(), os.path.join(args.output_dir, "best_policy.pt"))

        if (epoch + 1) % 10 == 0:
            save_training_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}.pt"),
                policy, optimizer, epoch + 1, 0,
                best_avg_reward=best_avg_reward, training_log=training_log,
                scheduler_state_dict=scheduler.state_dict(),
            )

    torch.save(policy.state_dict(), os.path.join(args.output_dir, "final_policy.pt"))

    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    summary = {
        "best_avg_reward": best_avg_reward,
        "final_avg_reward": np.mean(epoch_rewards),
        "num_traces": len(traces),
        "num_epochs": args.num_epochs,
        "hidden_dim": args.hidden_dim,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nTraining complete. Best avg reward: {best_avg_reward:.4f}")
    logger.info(f"Policy saved to {args.output_dir}")


if __name__ == "__main__":
    main()
