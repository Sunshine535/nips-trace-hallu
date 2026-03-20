#!/usr/bin/env python3
"""
Generate CoT traces with hallucination labels.
Uses Qwen3.5-27B to generate CoT traces on TruthfulQA/HaluEval.
Auto-labels hallucination onset points by comparing with ground truth.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_traces")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CoT traces with hallucination labels")
    parser.add_argument("--config", type=str, default="configs/trace_config.yaml")
    parser.add_argument("--dataset", type=str, default="truthfulqa",
                        choices=["truthfulqa", "halueval"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_truthfulqa():
    ds = load_dataset("truthful_qa", "generation", split="validation")
    samples = []
    for ex in ds:
        samples.append({
            "question": ex["question"],
            "best_answer": ex.get("best_answer", ""),
            "correct_answers": ex.get("correct_answers", []),
            "incorrect_answers": ex.get("incorrect_answers", []),
        })
    return samples


def load_halueval():
    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    samples = []
    for ex in ds:
        samples.append({
            "question": ex.get("question", ex.get("query", "")),
            "best_answer": ex.get("answer", ex.get("right_answer", "")),
            "correct_answers": [ex.get("answer", ex.get("right_answer", ""))],
            "incorrect_answers": [ex.get("hallucinated_answer", "")],
        })
    return samples


def build_cot_prompt(question: str) -> str:
    return (
        f"Answer the following question with a detailed chain-of-thought reasoning. "
        f"Think step by step before giving your final answer.\n\n"
        f"Question: {question}\n\n"
        f"Let me think through this step by step:\n"
    )


def label_hallucination_onset(
    trace_tokens: list[str],
    trace_text: str,
    correct_answers: list[str],
    incorrect_answers: list[str],
) -> list[int]:
    """
    Label each token position with 0 (normal) or 1 (hallucination onset/continuation).

    Heuristic: find the earliest sentence in the trace that contains information
    contradicting known correct answers or matching known incorrect answers.
    """
    labels = [0] * len(trace_tokens)

    sentences = re.split(r'(?<=[.!?])\s+', trace_text)
    if not sentences:
        return labels

    incorrect_set = set()
    for inc in incorrect_answers:
        for word in inc.lower().split():
            if len(word) > 3:
                incorrect_set.add(word)

    correct_set = set()
    for cor in correct_answers:
        for word in cor.lower().split():
            if len(word) > 3:
                correct_set.add(word)

    char_offset = 0
    onset_found = False
    for sentence in sentences:
        sentence_lower = sentence.lower()
        words = set(sentence_lower.split())

        incorrect_overlap = len(words & incorrect_set)
        correct_overlap = len(words & correct_set)

        is_hallucinated = (
            incorrect_overlap > 2
            and incorrect_overlap > correct_overlap
        )

        if is_hallucinated and not onset_found:
            onset_found = True

        sent_start = trace_text.find(sentence, char_offset)
        if sent_start == -1:
            sent_start = char_offset
        sent_end = sent_start + len(sentence)

        if onset_found:
            token_char_pos = 0
            for tidx, token in enumerate(trace_tokens):
                token_end = token_char_pos + len(token)
                if token_char_pos >= sent_start and token_char_pos < sent_end:
                    labels[tidx] = 1
                token_char_pos = token_end

        char_offset = sent_end

    return labels


@torch.no_grad()
def generate_traces_batch(model, tokenizer, prompts, gen_config):
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=gen_config["max_length"] // 2,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=gen_config["max_length"] // 2,
        temperature=gen_config["temperature"],
        top_p=gen_config["top_p"],
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    generated_texts = []
    for i, seq in enumerate(outputs.sequences):
        prompt_len = inputs["input_ids"][i].shape[0]
        gen_tokens = seq[prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        tokens = [tokenizer.decode([t], skip_special_tokens=True) for t in gen_tokens]
        generated_texts.append({"text": text, "tokens": tokens, "token_ids": gen_tokens.tolist()})

    return generated_texts


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    gen_cfg = cfg["generator"]
    output_dir = args.output_dir or cfg["output"]["traces_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "truthfulqa":
        samples = load_truthfulqa()
    else:
        samples = load_halueval()

    if args.max_samples:
        samples = samples[:args.max_samples]
    logger.info(f"Loaded {len(samples)} samples")

    model_name = gen_cfg["model"]
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    batch_size = args.batch_size or gen_cfg["batch_size"]
    num_traces = gen_cfg["num_traces_per_question"]

    all_traces = []
    for sample_idx in tqdm(range(0, len(samples), batch_size), desc="Generating traces"):
        batch = samples[sample_idx:sample_idx + batch_size]

        for trace_idx in range(num_traces):
            prompts = [build_cot_prompt(s["question"]) for s in batch]
            generated = generate_traces_batch(model, tokenizer, prompts, gen_cfg)

            for i, (sample, gen) in enumerate(zip(batch, generated)):
                hallu_labels = label_hallucination_onset(
                    gen["tokens"], gen["text"],
                    sample["correct_answers"],
                    sample["incorrect_answers"],
                )

                has_hallucination = any(l == 1 for l in hallu_labels)
                onset_pos = next((i for i, l in enumerate(hallu_labels) if l == 1), -1)

                trace_entry = {
                    "sample_idx": sample_idx + i,
                    "trace_idx": trace_idx,
                    "question": sample["question"],
                    "best_answer": sample["best_answer"],
                    "generated_text": gen["text"],
                    "tokens": gen["tokens"],
                    "token_ids": gen["token_ids"],
                    "hallu_labels": hallu_labels,
                    "has_hallucination": has_hallucination,
                    "onset_position": onset_pos,
                }
                all_traces.append(trace_entry)

    output_path = os.path.join(output_dir, f"traces_{args.dataset}.jsonl")
    with open(output_path, "w") as f:
        for trace in all_traces:
            f.write(json.dumps(trace) + "\n")

    n_hallu = sum(1 for t in all_traces if t["has_hallucination"])
    logger.info(f"Generated {len(all_traces)} traces, {n_hallu} with hallucinations")
    logger.info(f"Saved to {output_path}")

    stats = {
        "total_traces": len(all_traces),
        "traces_with_hallucination": n_hallu,
        "hallucination_rate": n_hallu / max(len(all_traces), 1),
        "dataset": args.dataset,
        "model": gen_cfg["model"],
    }
    stats_path = os.path.join(output_dir, f"trace_stats_{args.dataset}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
