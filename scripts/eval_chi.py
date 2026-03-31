#!/usr/bin/env python3
"""
Full PHI (Predictive Hallucination Intervention) system evaluation.

End-to-end pipeline: detect onset → select intervention → continue/restart.
Datasets: TruthfulQA, HaluEval, FaithDial.
Metrics: factuality score, fluency (perplexity), avg intervention count, latency overhead.
Compare: no intervention, always truncate, detector-oracle (trained detector, NOT
ground-truth labels), DoLa, ITI, SelfCheckGPT, PHI (ours).

NOTE on "detector_oracle" baseline: this uses our *trained* onset detector with a
fixed backtrack action.  It is NOT a true ground-truth oracle — it upper-bounds what
a perfect *policy* could achieve given the current detector, but detection errors
still propagate.  The name is kept for backward compatibility with prior runs.
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
    parser = argparse.ArgumentParser(description="Evaluate full PHI system")
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
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["truthfulqa", "halueval", "faithdial"])
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds for replicated evaluation (e.g. 42 137 2024)")
    parser.add_argument("--baselines", type=str, nargs="+",
                        default=["no_intervention", "always_truncate", "detector_oracle",
                                 "dola", "iti", "selfcheckgpt"],
                        help="Baselines to evaluate (detector_oracle = trained detector "
                             "with fixed backtrack, NOT ground-truth labels)")
    parser.add_argument("--use_claim_eval", action="store_true", default=True,
                        help="Use claim-level NLI evaluation instead of string matching")
    parser.add_argument("--dola_premature_layer", type=int, default=16)
    parser.add_argument("--iti_alpha", type=float, default=15.0)
    parser.add_argument("--selfcheck_num_samples", type=int, default=5)
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

_factuality_evaluator = None


def get_factuality_evaluator():
    """Lazy-init the claim-level factuality evaluator."""
    global _factuality_evaluator
    if _factuality_evaluator is None:
        from src.factuality_eval import FactualityEvaluator, FactualityConfig
        config = FactualityConfig()
        _factuality_evaluator = FactualityEvaluator(config)
    return _factuality_evaluator


def check_factuality(
    generated: str,
    correct_answers: list[str],
    incorrect_answers: list[str],
    use_claim_level: bool = True,
) -> tuple[float, str]:
    """Score factuality.  Returns (score, method).

    *method* is ``"claim_nli"`` when the NLI evaluator succeeded, or
    ``"proxy_heuristic"`` when using the word-overlap fallback.  Downstream
    consumers should label metrics with a ``proxy_`` prefix when heuristic.
    """
    if use_claim_level:
        try:
            evaluator = get_factuality_evaluator()
            result = evaluator.evaluate_single(generated, correct_answers, incorrect_answers)
            return result["factuality_score"], "claim_nli"
        except Exception as e:
            logger.warning(f"Claim-level evaluation failed ({e}), falling back to heuristic")

    gen_lower = generated.lower().strip()

    for ans in correct_answers:
        if ans and ans.lower().strip() in gen_lower:
            return 1.0, "proxy_heuristic"

    for inc in incorrect_answers:
        if inc and inc.lower().strip() in gen_lower:
            return 0.0, "proxy_heuristic"

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
        return 0.5 + 0.5 * correct_overlap, "proxy_heuristic"
    elif incorrect_overlap > 0:
        return max(0.0, 0.5 - 0.5 * incorrect_overlap), "proxy_heuristic"
    return 0.5, "proxy_heuristic"


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
        fact_score, fact_method = check_factuality(
            text, sample["correct_answers"], sample.get("incorrect_answers", []),
            use_claim_level=args.use_claim_eval,
        )
        ppl = compute_perplexity(model, tokenizer, text) if text.strip() else float("inf")

        results.append({
            "factuality": fact_score,
            "factuality_method": fact_method,
            "perplexity": min(ppl, 1000.0),
            "interventions": 0,
            "latency": latency,
            "text": text,
            "tokens": gen_len,
        })
    return results


def evaluate_always_truncate(
    model, tokenizer, detector, executor, samples, layer_indices, args,
):
    """Always truncate at detected onset."""
    results = []
    for sample in tqdm(samples, desc="Always truncate"):
        prompt = f"Question: {sample['question']}\n\nAnswer:"
        prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        start_time = time.time()

        text, onset, gen_ids, confidence, gen_len = generate_with_onset_detection(
            model, tokenizer, detector, prompt, layer_indices,
            max_new_tokens=args.max_new_tokens, threshold=args.threshold,
            detector_type=args.detector_type,
        )

        n_interventions = 0
        token_count = gen_len
        if onset > 0:
            result = executor.execute(Action.TRUNCATE, gen_ids, onset)
            text = tokenizer.decode(result["new_ids"][0, prompt_len:], skip_special_tokens=True)
            token_count = max(0, result["new_ids"].shape[1] - prompt_len)
            n_interventions = 1

        latency = time.time() - start_time
        fact_score, fact_method = check_factuality(
            text, sample["correct_answers"], sample.get("incorrect_answers", []),
            use_claim_level=args.use_claim_eval,
        )
        ppl = compute_perplexity(model, tokenizer, text) if text.strip() else float("inf")

        results.append({
            "factuality": fact_score,
            "factuality_method": fact_method,
            "perplexity": min(ppl, 1000.0),
            "interventions": n_interventions,
            "latency": latency,
            "text": text,
            "tokens": token_count,
        })
    return results


def evaluate_detector_oracle(
    model, tokenizer, detector, executor, samples, layer_indices, args,
):
    """
    Detector-oracle baseline: always backtrack when the *trained* onset detector fires.

    IMPORTANT: This is NOT a ground-truth oracle.  Detection errors (false positives
    and false negatives) still propagate.  This baseline upper-bounds what a perfect
    *policy* could achieve given the current detector's accuracy, but does NOT
    represent a perfect-detection scenario.
    """
    results = []
    for sample in tqdm(samples, desc="Detector oracle"):
        prompt = f"Question: {sample['question']}\n\nAnswer:"
        prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        start_time = time.time()

        text, onset, gen_ids, confidence, gen_len = generate_with_onset_detection(
            model, tokenizer, detector, prompt, layer_indices,
            max_new_tokens=args.max_new_tokens, threshold=args.threshold,
            detector_type=args.detector_type,
        )

        n_interventions = 0
        token_count = gen_len
        if onset > 0:
            result = executor.execute(Action.BACKTRACK, gen_ids, onset)
            text = tokenizer.decode(result["new_ids"][0, prompt_len:], skip_special_tokens=True)
            token_count = max(0, result["new_ids"].shape[1] - prompt_len)
            n_interventions = 1

        latency = time.time() - start_time
        fact_score, fact_method = check_factuality(
            text, sample["correct_answers"], sample.get("incorrect_answers", []),
            use_claim_level=args.use_claim_eval,
        )
        ppl = compute_perplexity(model, tokenizer, text) if text.strip() else float("inf")

        results.append({
            "factuality": fact_score,
            "factuality_method": fact_method,
            "perplexity": min(ppl, 1000.0),
            "interventions": n_interventions,
            "latency": latency,
            "text": text,
            "tokens": token_count,
        })
    return results


def evaluate_chi(
    model, tokenizer, detector, executor, policy, samples, layer_indices, args,
):
    """PHI: detect onset → select intervention via learned policy → execute."""
    results = []
    action_counts = {a.name: 0 for a in Action}

    for sample in tqdm(samples, desc="PHI (ours)"):
        prompt = f"Question: {sample['question']}\n\nAnswer:"
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        prompt_len = prompt_ids.shape[1]
        start_time = time.time()

        text, onset, gen_ids, confidence, gen_len = generate_with_onset_detection(
            model, tokenizer, detector, prompt, layer_indices,
            max_new_tokens=args.max_new_tokens, threshold=args.threshold,
            detector_type=args.detector_type,
        )

        n_interventions = 0
        token_count = gen_len
        if onset > 0:
            action = get_policy_action(
                policy, confidence, gen_len, sample["question"], onset, gen_ids.shape[1],
            )
            action_counts[action.name] += 1
            n_interventions = 1

            result = executor.execute(
                action, gen_ids, onset, original_prompt_ids=prompt_ids,
            )
            text = tokenizer.decode(result["new_ids"][0, prompt_len:], skip_special_tokens=True)
            token_count = max(0, result["new_ids"].shape[1] - prompt_len)

        latency = time.time() - start_time
        fact_score, fact_method = check_factuality(
            text, sample["correct_answers"], sample.get("incorrect_answers", []),
            use_claim_level=args.use_claim_eval,
        )
        ppl = compute_perplexity(model, tokenizer, text) if text.strip() else float("inf")

        results.append({
            "factuality": fact_score,
            "factuality_method": fact_method,
            "perplexity": min(ppl, 1000.0),
            "interventions": n_interventions,
            "latency": latency,
            "text": text,
            "tokens": token_count,
        })

    return results, action_counts


# ── Main ─────────────────────────────────────────────────────────────────────

def evaluate_dola_baseline(model, tokenizer, samples, args):
    """DoLa: Decoding by Contrasting Layers baseline."""
    from src.baselines import DoLaDecoder
    dola = DoLaDecoder(model, tokenizer, premature_layer=args.dola_premature_layer)
    results = []
    for sample in tqdm(samples, desc="DoLa"):
        prompt = f"Question: {sample['question']}\n\nAnswer:"
        start_time = time.time()
        out = dola.generate(prompt, max_new_tokens=args.max_new_tokens)
        latency = time.time() - start_time
        fact_score, fact_method = check_factuality(
            out["text"], sample["correct_answers"],
            sample.get("incorrect_answers", []),
            use_claim_level=args.use_claim_eval,
        )
        ppl = compute_perplexity(model, tokenizer, out["text"]) if out["text"].strip() else float("inf")
        results.append({
            "factuality": fact_score,
            "factuality_method": fact_method,
            "perplexity": min(ppl, 1000.0),
            "interventions": 0,
            "latency": latency,
            "text": out["text"],
            "tokens": out.get("num_tokens", len(out["text"].split())),
        })
    return results


def evaluate_selfcheck_baseline(model, tokenizer, samples, args):
    """SelfCheckGPT baseline."""
    from src.baselines import SelfCheckGPT
    checker = SelfCheckGPT(model, tokenizer, num_samples=args.selfcheck_num_samples)
    results = []
    for sample in tqdm(samples, desc="SelfCheckGPT"):
        prompt = f"Question: {sample['question']}\n\nAnswer:"
        start_time = time.time()
        out = checker.generate_and_check(prompt, max_new_tokens=args.max_new_tokens)
        latency = time.time() - start_time
        fact_score, fact_method = check_factuality(
            out["text"], sample["correct_answers"],
            sample.get("incorrect_answers", []),
            use_claim_level=args.use_claim_eval,
        )
        ppl = compute_perplexity(model, tokenizer, out["text"]) if out["text"].strip() else float("inf")
        results.append({
            "factuality": fact_score,
            "factuality_method": fact_method,
            "perplexity": min(ppl, 1000.0),
            "interventions": 0,
            "latency": latency,
            "consistency": out.get("consistency_score", 0),
            "text": out["text"],
            "tokens": out.get("num_tokens", len(tokenizer.encode(out["text"]))),
        })
    return results


def evaluate_iti_baseline(model, tokenizer, samples, args):
    """ITI: Inference-Time Intervention (Li et al., 2023) baseline."""
    from src.baselines import ITIDecoder
    iti = ITIDecoder(model, tokenizer, alpha=args.iti_alpha)

    truthful_texts = []
    hallucinated_texts = []
    for s in samples[:50]:
        if s.get("correct_answers"):
            truthful_texts.append(s["correct_answers"][0])
        if s.get("incorrect_answers"):
            hallucinated_texts.append(s["incorrect_answers"][0])

    if truthful_texts and hallucinated_texts:
        iti.compute_directions(truthful_texts, hallucinated_texts)
    else:
        logger.warning("ITI: insufficient paired examples; running without learned directions")

    results = []
    for sample in tqdm(samples, desc="ITI"):
        prompt = f"Question: {sample['question']}\n\nAnswer:"
        start_time = time.time()
        out = iti.generate(prompt, max_new_tokens=args.max_new_tokens)
        latency = time.time() - start_time
        fact_score, fact_method = check_factuality(
            out["text"], sample["correct_answers"],
            sample.get("incorrect_answers", []),
            use_claim_level=args.use_claim_eval,
        )
        ppl = compute_perplexity(model, tokenizer, out["text"]) if out["text"].strip() else float("inf")
        results.append({
            "factuality": fact_score,
            "factuality_method": fact_method,
            "perplexity": min(ppl, 1000.0),
            "interventions": 0,
            "latency": latency,
            "text": out["text"],
            "tokens": out.get("num_tokens", len(tokenizer.encode(out["text"]))),
        })
    return results


def evaluate_rule_policy(
    model, tokenizer, detector, executor, rule_policy, samples, layer_indices, args, policy_name,
):
    """Evaluate a rule-based policy using the same detector as PHI."""
    results = []
    action_counts = {a.name: 0 for a in Action}

    for sample in tqdm(samples, desc=f"Rule: {policy_name}"):
        prompt = f"Question: {sample['question']}\n\nAnswer:"
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        prompt_len = prompt_ids.shape[1]
        start_time = time.time()

        text, onset, gen_ids, confidence, gen_len = generate_with_onset_detection(
            model, tokenizer, detector, prompt, layer_indices,
            max_new_tokens=args.max_new_tokens, threshold=args.threshold,
            detector_type=args.detector_type,
        )

        n_interventions = 0
        token_count = gen_len
        if onset > 0:
            action = rule_policy.select_action(confidence=confidence, threshold=args.threshold)
            action_counts[action.name] += 1
            n_interventions = 1

            if action != Action.CONTINUE:
                result = executor.execute(action, gen_ids, onset, original_prompt_ids=prompt_ids)
                text = tokenizer.decode(result["new_ids"][0, prompt_len:], skip_special_tokens=True)
                token_count = max(0, result["new_ids"].shape[1] - prompt_len)

        latency = time.time() - start_time
        fact_score, fact_method = check_factuality(
            text, sample["correct_answers"], sample.get("incorrect_answers", []),
            use_claim_level=args.use_claim_eval,
        )
        ppl = compute_perplexity(model, tokenizer, text) if text.strip() else float("inf")

        results.append({
            "factuality": fact_score,
            "factuality_method": fact_method,
            "perplexity": min(ppl, 1000.0),
            "interventions": n_interventions,
            "latency": latency,
            "text": text,
            "tokens": token_count,
        })

    return results, action_counts


def aggregate_seed_results(all_seed_results: dict) -> dict:
    """Aggregate results across seeds with bootstrap CIs."""
    from src.factuality_eval import paired_bootstrap_test

    aggregated = {}
    methods = list(all_seed_results[list(all_seed_results.keys())[0]].keys())

    for method in methods:
        if not isinstance(all_seed_results[list(all_seed_results.keys())[0]].get(method), list):
            continue

        all_facts = []
        for seed, results in all_seed_results.items():
            if method in results:
                facts = [r["factuality"] for r in results[method]]
                all_facts.extend(facts)

        if not all_facts:
            continue

        arr = np.array(all_facts)
        aggregated[method] = {
            "mean_factuality": float(arr.mean()),
            "std_factuality": float(arr.std()),
            "n_total": len(arr),
            "n_seeds": len(all_seed_results),
        }

        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(1000):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boot_means.append(sample.mean())
        boot_means = np.sort(boot_means)
        aggregated[method]["ci_95"] = [
            float(boot_means[int(0.025 * len(boot_means))]),
            float(boot_means[int(0.975 * len(boot_means))]),
        ]

    if "no_intervention" in aggregated and "chi_ours" in aggregated:
        baseline_scores = []
        chi_scores = []
        for seed, results in all_seed_results.items():
            if "no_intervention" in results and "chi_ours" in results:
                baseline_scores.extend([r["factuality"] for r in results["no_intervention"]])
                chi_scores.extend([r["factuality"] for r in results["chi_ours"]])

        if len(baseline_scores) == len(chi_scores) and len(baseline_scores) > 0:
            sig = paired_bootstrap_test(chi_scores, baseline_scores)
            aggregated["significance_vs_baseline"] = sig

    return aggregated


def run_single_seed(model, tokenizer, detector, executor, policy, args, samples, seed):
    """Run full evaluation for a single seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f"\n--- Seed {seed} ---")

    methods = {}

    if "no_intervention" in args.baselines:
        logger.info("[baseline] No intervention")
        methods["no_intervention"] = evaluate_no_intervention(
            model, tokenizer, detector, samples, args.layer_indices, args,
        )

    if "always_truncate" in args.baselines:
        logger.info("[baseline] Always truncate")
        methods["always_truncate"] = evaluate_always_truncate(
            model, tokenizer, detector, executor, samples, args.layer_indices, args,
        )

    if "detector_oracle" in args.baselines:
        logger.info("[baseline] Detector oracle")
        methods["detector_oracle"] = evaluate_detector_oracle(
            model, tokenizer, detector, executor, samples, args.layer_indices, args,
        )

    if "dola" in args.baselines:
        logger.info("[baseline] DoLa")
        methods["dola"] = evaluate_dola_baseline(model, tokenizer, samples, args)

    if "selfcheckgpt" in args.baselines:
        logger.info("[baseline] SelfCheckGPT")
        methods["selfcheckgpt"] = evaluate_selfcheck_baseline(model, tokenizer, samples, args)

    if "iti" in args.baselines:
        logger.info("[baseline] ITI")
        methods["iti"] = evaluate_iti_baseline(model, tokenizer, samples, args)

    if "rule_cascade" in args.baselines:
        logger.info("[baseline] Rule: threshold cascade")
        from src.rule_policies import RULE_POLICIES
        rule_results, rule_actions = evaluate_rule_policy(
            model, tokenizer, detector, executor,
            RULE_POLICIES["threshold_cascade"], samples, args.layer_indices, args,
            "threshold_cascade",
        )
        methods["rule_cascade"] = rule_results

    if "rule_always_backtrack" in args.baselines:
        logger.info("[baseline] Rule: always backtrack")
        from src.rule_policies import RULE_POLICIES
        rule_results, rule_actions = evaluate_rule_policy(
            model, tokenizer, detector, executor,
            RULE_POLICIES["always_backtrack"], samples, args.layer_indices, args,
            "always_backtrack",
        )
        methods["rule_always_backtrack"] = rule_results

    logger.info("[ours] PHI")
    chi_results, action_counts = evaluate_chi(
        model, tokenizer, detector, executor, policy, samples, args.layer_indices, args,
    )
    methods["chi_ours"] = chi_results
    methods["chi_action_distribution"] = action_counts

    return methods


def main():
    args = parse_args()

    from config_utils import load_config, apply_config_defaults
    cfg = load_config(args.config)
    apply_config_defaults(args, "eval_chi", cfg)

    os.makedirs(args.output_dir, exist_ok=True)

    seeds = args.seeds or [args.seed]
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
        logger.info(f"Seeds: {seeds}")
        logger.info(f"Baselines: {args.baselines}")
        logger.info(f"{'='*60}")

        try:
            samples = load_eval_dataset(ds_name, args.num_samples)
        except Exception as e:
            logger.warning(f"Failed to load {ds_name}: {e}")
            continue

        logger.info(f"Loaded {len(samples)} samples")

        seed_results = {}
        for seed in seeds:
            seed_results[seed] = run_single_seed(
                model, tokenizer, detector, executor, policy, args, samples, seed,
            )

        ds_metrics = {}
        last_seed = seeds[-1]
        for method_name, result_list in seed_results[last_seed].items():
            if not isinstance(result_list, list):
                continue
            n = len(result_list)
            avg_factuality = np.mean([r["factuality"] for r in result_list])
            avg_ppl = np.mean([r["perplexity"] for r in result_list])
            avg_interventions = np.mean([r["interventions"] for r in result_list])
            avg_latency = np.mean([r["latency"] for r in result_list])

            proxy_count = sum(1 for r in result_list
                              if r.get("factuality_method") == "proxy_heuristic")
            metric_dict = {
                "factuality": avg_factuality,
                "perplexity": avg_ppl,
                "avg_interventions": avg_interventions,
                "avg_latency_s": avg_latency,
                "n_samples": n,
            }
            if proxy_count > 0:
                metric_dict["proxy_factuality"] = avg_factuality
                metric_dict["proxy_ratio"] = proxy_count / n
            ds_metrics[method_name] = metric_dict

            tag = " [PROXY]" if proxy_count > 0 else ""
            logger.info(
                f"  {method_name:20s}: factuality={avg_factuality:.4f}{tag} ppl={avg_ppl:.1f} "
                f"interventions={avg_interventions:.2f} latency={avg_latency:.2f}s"
            )

        if len(seeds) > 1:
            ds_metrics["multi_seed_aggregate"] = aggregate_seed_results(seed_results)

        if "chi_action_distribution" in seed_results[last_seed]:
            ds_metrics["chi_action_distribution"] = seed_results[last_seed]["chi_action_distribution"]

        if "no_intervention" in ds_metrics and "chi_ours" in ds_metrics:
            baseline_fact = ds_metrics["no_intervention"]["factuality"]
            chi_fact = ds_metrics["chi_ours"]["factuality"]
            ds_metrics["improvement_over_baseline"] = chi_fact - baseline_fact

        try:
            from src.budget_eval import evaluate_under_budget
            budget_data = {}
            for method_name, result_list in seed_results[last_seed].items():
                if isinstance(result_list, list) and len(result_list) > 0:
                    if "factuality" in result_list[0]:
                        budget_data[method_name] = [
                            {"text": r.get("text", ""), "factuality": r["factuality"],
                             "tokens": r.get("tokens", 0), "latency": r.get("latency", 0)}
                            for r in result_list
                        ]
            if budget_data:
                ds_metrics["budget_matched"] = evaluate_under_budget(budget_data)
                logger.info("  Budget-matched evaluation completed")
        except Exception as e:
            logger.warning(f"  Budget-matched evaluation failed: {e}")

        all_results[ds_name] = ds_metrics

    output_path = os.path.join(args.output_dir, "chi_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total_eval_time = time.time()

    logger.info(f"\n{'='*60}")
    logger.info("PHI EVALUATION SUMMARY (keys: chi_ours = PHI method)")
    logger.info(f"{'='*60}")
    for ds_name, metrics in all_results.items():
        logger.info(f"\n{ds_name}:")
        for method, m in metrics.items():
            if isinstance(m, dict) and "factuality" in m:
                logger.info(f"  {method:20s}: fact={m['factuality']:.4f} ppl={m['perplexity']:.1f} "
                           f"lat={m.get('avg_latency_s', 0):.2f}s")
        if "improvement_over_baseline" in metrics:
            logger.info(f"  PHI improvement: {metrics['improvement_over_baseline']:+.4f}")
        if "multi_seed_aggregate" in metrics and "significance_vs_baseline" in metrics["multi_seed_aggregate"]:
            sig = metrics["multi_seed_aggregate"]["significance_vs_baseline"]
            logger.info(f"  Significance (paired bootstrap): p={sig['p_value']:.4f}")
        if "budget_matched" in metrics:
            bm = metrics["budget_matched"]
            if "pareto_front" in bm:
                logger.info("  Pareto front:")
                for pt in bm["pareto_front"]:
                    logger.info(f"    {pt['method']:20s} budget={pt['budget']} fact={pt['factuality']:.4f}")

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
