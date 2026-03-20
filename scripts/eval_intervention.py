#!/usr/bin/env python3
"""
Evaluate full pipeline: detect hallucination onset -> intervene -> measure accuracy improvement.
Compares: (1) no intervention baseline, (2) oracle intervention, (3) learned policy.
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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.onset_detector import OnsetDetectorConfig, OnsetLinearProbe, find_onset_positions
from src.intervention_actions import (
    Action, InterventionConfig, InterventionExecutor, format_intervention_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_intervention")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate intervention pipeline")
    parser.add_argument("--config", type=str, default="configs/trace_config.yaml")
    parser.add_argument("--detector_path", type=str, required=True,
                        help="Path to trained onset detector checkpoint")
    parser.add_argument("--policy_dir", type=str, required=True,
                        help="Path to trained intervention policy")
    parser.add_argument("--detector_layer", type=int, default=24,
                        help="Which transformer layer the detector was trained on")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--dataset", type=str, default="truthfulqa",
                        choices=["truthfulqa", "halueval"])
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_answer_correctness(generated: str, gold_answers: list[str]) -> bool:
    gen_lower = generated.lower().strip()
    for ans in gold_answers:
        if ans.lower().strip() in gen_lower:
            return True
    return False


@torch.no_grad()
def generate_with_detection(
    model, tokenizer, detector, prompt, layer_idx, max_new_tokens=512, threshold=0.5,
):
    """Generate tokens one-by-one, running onset detection at each step."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    generated_ids = input_ids.clone()
    onset_detected_at = -1

    for step in range(max_new_tokens):
        outputs = model(generated_ids, output_hidden_states=True)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        if onset_detected_at < 0 and step > 5:
            hidden = outputs.hidden_states[layer_idx]
            det_out = detector(hidden)
            probs = torch.softmax(det_out["logits"], dim=-1)[:, -1, 1]
            if probs.item() > threshold:
                onset_detected_at = prompt_len + step

    text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
    return text, onset_detected_at, generated_ids


def select_action_with_policy(policy_model, policy_tokenizer, question, trace_so_far, onset_pos):
    """Use the trained policy to select an intervention action."""
    prompt = format_intervention_prompt(question, trace_so_far, onset_pos)
    inputs = policy_tokenizer(prompt, return_tensors="pt").to(policy_model.device)

    with torch.no_grad():
        outputs = policy_model.generate(
            **inputs, max_new_tokens=10, temperature=0.0,
            pad_token_id=policy_tokenizer.pad_token_id,
        )

    response = policy_tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()

    for action_id in range(5):
        if str(action_id) in response[:10]:
            return Action(action_id)
    return Action.CONTINUE


def build_eval_prompt(question: str) -> str:
    return (
        f"Answer the following question with detailed reasoning.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = cfg["generator"]["model"]
    logger.info(f"Loading generator model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    # Load detector
    logger.info(f"Loading onset detector from {args.detector_path}")
    det_config = OnsetDetectorConfig(hidden_size=cfg["detector"]["hidden_size"])
    detector = OnsetLinearProbe(det_config)
    detector.load_state_dict(torch.load(args.detector_path, map_location="cpu"))
    detector = detector.to(model.device)
    detector.eval()

    # Load intervention policy
    logger.info(f"Loading policy from {args.policy_dir}")
    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_dir, trust_remote_code=True)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    policy_model.eval()

    # Load eval data
    if args.dataset == "truthfulqa":
        ds = load_dataset("truthful_qa", "generation", split="validation")
        samples = []
        for ex in ds:
            samples.append({
                "question": ex["question"],
                "correct_answers": ex.get("correct_answers", [ex.get("best_answer", "")]),
            })
    else:
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
        samples = []
        for ex in ds:
            samples.append({
                "question": ex.get("question", ""),
                "correct_answers": [ex.get("answer", ex.get("right_answer", ""))],
            })

    samples = samples[:args.num_samples]
    logger.info(f"Evaluating on {len(samples)} samples")

    int_config = InterventionConfig(
        backtrack_window=cfg["intervention"]["backtrack_window"],
        max_new_tokens=cfg["eval"]["max_new_tokens"],
    )
    executor = InterventionExecutor(model, tokenizer, int_config)

    # Evaluation loop
    results = {"baseline": [], "policy": [], "oracle_backtrack": []}
    action_counts = {a.name: 0 for a in Action}

    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        question = sample["question"]
        gold_answers = sample["correct_answers"]
        prompt = build_eval_prompt(question)

        # 1) Baseline: no intervention
        gen_text, onset_pos, gen_ids = generate_with_detection(
            model, tokenizer, detector, prompt, args.detector_layer,
            max_new_tokens=cfg["eval"]["max_new_tokens"],
        )
        baseline_correct = check_answer_correctness(gen_text, gold_answers)
        results["baseline"].append({"idx": idx, "correct": baseline_correct})

        # 2) Policy intervention
        if onset_pos > 0:
            trace_so_far = tokenizer.decode(
                gen_ids[0, :onset_pos], skip_special_tokens=True,
            )
            action = select_action_with_policy(
                policy_model, policy_tokenizer, question, trace_so_far, onset_pos,
            )
            action_counts[action.name] += 1

            prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
            result = executor.execute(
                action, gen_ids, onset_pos, original_prompt_ids=prompt_ids,
            )
            policy_text = tokenizer.decode(
                result["new_ids"][0], skip_special_tokens=True,
            )
            policy_correct = check_answer_correctness(policy_text, gold_answers)
        else:
            policy_correct = baseline_correct
        results["policy"].append({"idx": idx, "correct": policy_correct})

        # 3) Oracle: always backtrack at detected onset
        if onset_pos > 0:
            result = executor.execute(
                Action.BACKTRACK, gen_ids, onset_pos,
            )
            oracle_text = tokenizer.decode(result["new_ids"][0], skip_special_tokens=True)
            oracle_correct = check_answer_correctness(oracle_text, gold_answers)
        else:
            oracle_correct = baseline_correct
        results["oracle_backtrack"].append({"idx": idx, "correct": oracle_correct})

    # Compute metrics
    metrics = {}
    for method, res_list in results.items():
        acc = sum(r["correct"] for r in res_list) / max(len(res_list), 1)
        metrics[method] = {"accuracy": acc, "n_samples": len(res_list)}
        logger.info(f"{method}: accuracy = {acc:.4f}")

    metrics["action_distribution"] = action_counts
    metrics["improvement_over_baseline"] = (
        metrics["policy"]["accuracy"] - metrics["baseline"]["accuracy"]
    )

    output_path = os.path.join(args.output_dir, f"eval_intervention_{args.dataset}.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    print("\n" + "=" * 50)
    print(f"Baseline accuracy:      {metrics['baseline']['accuracy']:.4f}")
    print(f"Policy accuracy:        {metrics['policy']['accuracy']:.4f}")
    print(f"Oracle backtrack:       {metrics['oracle_backtrack']['accuracy']:.4f}")
    print(f"Improvement:            {metrics['improvement_over_baseline']:+.4f}")
    print(f"Action distribution:    {action_counts}")
    print("=" * 50)


if __name__ == "__main__":
    main()
