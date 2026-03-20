#!/usr/bin/env python3
"""
Full CHI (Causal Hallucination Intervention) system evaluation.

End-to-end pipeline: detect onset → select intervention → continue/restart.
Datasets: TruthfulQA, HaluEval, FaithDial.
Metrics: factuality score, fluency (perplexity), avg intervention count, latency overhead.
Compare: no intervention, always truncate, oracle detector, CHI (ours).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.onset_detector import (
    OnsetDetectorConfig, OnsetLinearProbe, MultiLayerOnsetDetector, find_onset_positions,
)
from src.intervention_actions import (
    Action, ACTION_NAMES, InterventionConfig, InterventionExecutor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_chi")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate full CHI system")
    parser.add_argument("--config", type=str, default="configs/trace_config.yaml")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--detector_path", type=str, required=True,
                        help="Path to trained detector (.pt)")
    parser.add_argument("--policy_path", type=str, required=True,
                        help="Path to trained PPO policy (.pt)")
    parser.add_argument("--detector_type", type=str, default="multi_layer",
                        choices=["single_layer", "multi_layer"])
    parser.add_argument("--detector_layer", type=int, default=24)
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--hidden_size", type=int, default=3584)
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["truthfulqa", "halueval", "faithdial"])
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Dataset Loading ──────────────────────────────────────────────────────────

def load_eval_dataset(name, max_samples):
    if name == "truthfulqa":
        ds = load_dataset("truthful_qa", "generation", split="validation")
        samples = [{
            "question": ex["question"],
            "correct_answers": ex.get("correct_answers", [ex.get("best_answer", "")]),
            "incorrect_answers": ex.get("incorrect_answers", []),
        } for ex in ds]
    elif name == "halueval":
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
        samples = [{
            "question": ex.get("question", ""),
            "correct_answers": [ex.get("answer", "")],
            "incorrect_answers": [ex.get("hallucinated_answer", "")],
        } for ex in ds]
    elif name == "faithdial":
        ds = load_dataset("McGill-NLP/FaithDial", split="test")
        samples = [{
            "question": ex.get("history", [""])[-1] if ex.get("history") else "",
            "correct_answers": [ex.get("response", "")],
            "incorrect_answers": [],
        } for ex in ds]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return samples[:max_samples]


# ── Metrics ──────────────────────────────────────────────────────────────────

def check_factuality(generated: str, correct_answers: list[str], incorrect_answers: list[str]) -> float:
    """Score factuality: 1.0 if matches correct, 0.0 if matches incorrect, 0.5 otherwise."""
    gen_lower = generated.lower().strip()

    for ans in correct_answers:
        if ans and ans.lower().strip() in gen_lower:
            return 1.0

    for inc in incorrect_answers:
        if inc and inc.lower().strip() in gen_lower:
            return 0.0

    gen_words = set(gen_lower.split())
    correct_words = set()
    for a in correct_answers:
        correct_words.update(a.lower().split())
    incorrect_words = set()
    for a in incorrect_answers:
        incorrect_words.update(a.lower().split())

    correct_overlap = len(gen_words & correct_words) / max(len(correct_words), 1)
    incorrect_overlap = len(gen_words & incorrect_words) / max(len(incorrect_words), 1)

    if correct_overlap > incorrect_overlap:
        return 0.5 + 0.5 * correct_overlap
    elif incorrect_overlap > 0:
        return max(0.0, 0.5 - 0.5 * incorrect_overlap)
    return 0.5


@torch.no_grad()
def compute_perplexity(model, tokenizer, text, max_length=512):
    """Compute perplexity of generated text as a fluency proxy."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()


# ── Policy Loader ────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent))

def load_mlp_policy(policy_path, device):
    from train_intervention_policy import InterventionPolicyMLP
    policy = InterventionPolicyMLP(input_dim=5, hidden_dim=128, num_actions=5)
    policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
    policy = policy.to(device)
    policy.eval()
    return policy


def get_policy_action(policy, detector_confidence, gen_length, query, onset_pos, total_tokens):
    """Get action from MLP policy given state features."""
    gen_length_norm = min(gen_length / 512.0, 1.0)
    query_complexity = min(len(query.split()) / 50.0, 1.0)
    onset_norm = onset_pos / max(total_tokens, 1) if onset_pos >= 0 else 1.0
    hallu_density = detector_confidence * 0.5

    state = torch.tensor(
        [detector_confidence, gen_length_norm, query_complexity, onset_norm, hallu_density],
        dtype=torch.float32,
    ).unsqueeze(0).to(next(policy.parameters()).device)

    with torch.no_grad():
        logits, _ = policy(state)
        action_idx = logits.argmax(dim=-1).item()

    return Action(action_idx)


# ── Generation with Detection ────────────────────────────────────────────────

@torch.no_grad()
def generate_with_onset_detection(
    model, tokenizer, detector, prompt, layer_indices,
    max_new_tokens=512, threshold=0.5, detector_type="multi_layer",
):
    """Generate tokens with real-time onset detection."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    generated_ids = input_ids.clone()
    onset_detected_at = -1
    max_confidence = 0.0

    for step in range(max_new_tokens):
        outputs = model(generated_ids, output_hidden_states=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        if onset_detected_at < 0 and step > 3:
            if detector_type == "multi_layer":
                all_hs = {idx: outputs.hidden_states[idx] for idx in layer_indices
                          if idx < len(outputs.hidden_states)}
                det_out = detector(all_hs)
            else:
                hs = outputs.hidden_states[layer_indices[0]]
                det_out = detector(hs)

            probs = torch.softmax(det_out["logits"], dim=-1)[:, -1, 1]
            confidence = probs.item()
            max_confidence = max(max_confidence, confidence)

            if confidence > threshold:
                onset_detected_at = prompt_len + step

    text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
    gen_length = generated_ids.shape[1] - prompt_len
    return text, onset_detected_at, generated_ids, max_confidence, gen_length


# ── Evaluation Methods ───────────────────────────────────────────────────────

def evaluate_no_intervention(
    model, tokenizer, detector, samples, layer_indices, args,
):
    """Baseline: generate without any intervention."""
    results = []
    for sample in tqdm(samples, desc="No intervention"):
        prompt = f"Question: {sample['question']}\n\nAnswer:"
        start_time = time.time()

        text, onset, gen_ids, confidence, gen_len = generate_with_onset_detection(
            model, tokenizer, detector, prompt, layer_indices,
            max_new_tokens=args.max_new_tokens, threshold=args.threshold,
            detector_type=args.detector_type,
        )

        latency = time.time() - start_time
        factuality = check_factuality(text, sample["correct_answers"], sample.get("incorrect_answers", []))
        ppl = compute_perplexity(model, tokenizer, text) if text.strip() else float("inf")

        results.append({
            "factuality": factuality,
            "perplexity": min(ppl, 1000.0),
            "interventions": 0,
            "latency": latency,
        })
    return results


def evaluate_always_truncate(
    model, tokenizer, detector, executor, samples, layer_indices, args,
):
    """Always truncate at detected onset."""
    results = []
    for sample in tqdm(samples, desc="Always truncate"):
        prompt = f"Question: {sample['question']}\n\nAnswer:"
        start_time = time.time()

        text, onset, gen_ids, confidence, gen_len = generate_with_onset_detection(
            model, tokenizer, detector, prompt, layer_indices,
            max_new_tokens=args.max_new_tokens, threshold=args.threshold,
            detector_type=args.detector_type,
        )

        n_interventions = 0
        if onset > 0:
            result = executor.execute(Action.TRUNCATE, gen_ids, onset)
            text = tokenizer.decode(result["new_ids"][0], skip_special_tokens=True)
            n_interventions = 1

        latency = time.time() - start_time
        factuality = check_factuality(text, sample["correct_answers"], sample.get("incorrect_answers", []))
        ppl = compute_perplexity(model, tokenizer, text) if text.strip() else float("inf")

        results.append({
            "factuality": factuality,
            "perplexity": min(ppl, 1000.0),
            "interventions": n_interventions,
            "latency": latency,
        })
    return results


def evaluate_oracle(
    model, tokenizer, detector, executor, samples, layer_indices, args,
):
    """Oracle: always backtrack at true hallucination onset."""
    results = []
    for sample in tqdm(samples, desc="Oracle detector"):
        prompt = f"Question: {sample['question']}\n\nAnswer:"
        start_time = time.time()

        text, onset, gen_ids, confidence, gen_len = generate_with_onset_detection(
            model, tokenizer, detector, prompt, layer_indices,
            max_new_tokens=args.max_new_tokens, threshold=args.threshold,
            detector_type=args.detector_type,
        )

        n_interventions = 0
        if onset > 0:
            result = executor.execute(Action.BACKTRACK, gen_ids, onset)
            text = tokenizer.decode(result["new_ids"][0], skip_special_tokens=True)
            n_interventions = 1

        latency = time.time() - start_time
        factuality = check_factuality(text, sample["correct_answers"], sample.get("incorrect_answers", []))
        ppl = compute_perplexity(model, tokenizer, text) if text.strip() else float("inf")

        results.append({
            "factuality": factuality,
            "perplexity": min(ppl, 1000.0),
            "interventions": n_interventions,
            "latency": latency,
        })
    return results


def evaluate_chi(
    model, tokenizer, detector, executor, policy, samples, layer_indices, args,
):
    """CHI: detect onset → select intervention via learned policy → execute."""
    results = []
    action_counts = {a.name: 0 for a in Action}

    for sample in tqdm(samples, desc="CHI (ours)"):
        prompt = f"Question: {sample['question']}\n\nAnswer:"
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        start_time = time.time()

        text, onset, gen_ids, confidence, gen_len = generate_with_onset_detection(
            model, tokenizer, detector, prompt, layer_indices,
            max_new_tokens=args.max_new_tokens, threshold=args.threshold,
            detector_type=args.detector_type,
        )

        n_interventions = 0
        if onset > 0:
            action = get_policy_action(
                policy, confidence, gen_len, sample["question"], onset, gen_ids.shape[1],
            )
            action_counts[action.name] += 1
            n_interventions = 1

            result = executor.execute(
                action, gen_ids, onset, original_prompt_ids=prompt_ids,
            )
            text = tokenizer.decode(result["new_ids"][0], skip_special_tokens=True)

        latency = time.time() - start_time
        factuality = check_factuality(text, sample["correct_answers"], sample.get("incorrect_answers", []))
        ppl = compute_perplexity(model, tokenizer, text) if text.strip() else float("inf")

        results.append({
            "factuality": factuality,
            "perplexity": min(ppl, 1000.0),
            "interventions": n_interventions,
            "latency": latency,
        })

    return results, action_counts


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading generator: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    logger.info(f"Loading detector from {args.detector_path}")
    det_config = OnsetDetectorConfig(hidden_size=args.hidden_size)
    if args.detector_type == "multi_layer":
        detector = MultiLayerOnsetDetector(det_config, args.layer_indices)
    else:
        detector = OnsetLinearProbe(det_config)
    detector.load_state_dict(torch.load(args.detector_path, map_location="cpu"))
    detector = detector.to(device)
    detector.eval()

    logger.info(f"Loading policy from {args.policy_path}")
    policy = load_mlp_policy(args.policy_path, device)

    int_config = InterventionConfig(max_new_tokens=args.max_new_tokens)
    executor = InterventionExecutor(model, tokenizer, int_config)

    all_results = {}

    for ds_name in args.datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on: {ds_name}")
        logger.info(f"{'='*60}")

        try:
            samples = load_eval_dataset(ds_name, args.num_samples)
        except Exception as e:
            logger.warning(f"Failed to load {ds_name}: {e}")
            continue

        logger.info(f"Loaded {len(samples)} samples")

        methods = {}

        logger.info("\n[1/4] No intervention baseline")
        methods["no_intervention"] = evaluate_no_intervention(
            model, tokenizer, detector, samples, args.layer_indices, args,
        )

        logger.info("\n[2/4] Always truncate")
        methods["always_truncate"] = evaluate_always_truncate(
            model, tokenizer, detector, executor, samples, args.layer_indices, args,
        )

        logger.info("\n[3/4] Oracle detector")
        methods["oracle_detector"] = evaluate_oracle(
            model, tokenizer, detector, executor, samples, args.layer_indices, args,
        )

        logger.info("\n[4/4] CHI (ours)")
        chi_results, action_counts = evaluate_chi(
            model, tokenizer, detector, executor, policy, samples, args.layer_indices, args,
        )
        methods["chi_ours"] = chi_results

        ds_metrics = {}
        for method_name, result_list in methods.items():
            n = len(result_list)
            avg_factuality = np.mean([r["factuality"] for r in result_list])
            avg_ppl = np.mean([r["perplexity"] for r in result_list])
            avg_interventions = np.mean([r["interventions"] for r in result_list])
            avg_latency = np.mean([r["latency"] for r in result_list])

            ds_metrics[method_name] = {
                "factuality": avg_factuality,
                "perplexity": avg_ppl,
                "avg_interventions": avg_interventions,
                "avg_latency_s": avg_latency,
                "n_samples": n,
            }
            logger.info(
                f"  {method_name:20s}: factuality={avg_factuality:.4f} ppl={avg_ppl:.1f} "
                f"interventions={avg_interventions:.2f} latency={avg_latency:.2f}s"
            )

        ds_metrics["chi_action_distribution"] = {k: v for k, v in action_counts.items()}

        baseline_fact = ds_metrics["no_intervention"]["factuality"]
        chi_fact = ds_metrics["chi_ours"]["factuality"]
        ds_metrics["improvement_over_baseline"] = chi_fact - baseline_fact
        ds_metrics["improvement_over_truncate"] = chi_fact - ds_metrics["always_truncate"]["factuality"]

        all_results[ds_name] = ds_metrics

    output_path = os.path.join(args.output_dir, "chi_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("CHI EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    for ds_name, metrics in all_results.items():
        logger.info(f"\n{ds_name}:")
        for method, m in metrics.items():
            if isinstance(m, dict) and "factuality" in m:
                logger.info(f"  {method:20s}: fact={m['factuality']:.4f} ppl={m['perplexity']:.1f}")
        if "improvement_over_baseline" in metrics:
            logger.info(f"  CHI improvement: {metrics['improvement_over_baseline']:+.4f}")

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
