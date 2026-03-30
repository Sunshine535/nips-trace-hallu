#!/usr/bin/env python3
"""
Collect annotated CoT traces with hidden states for hallucination onset detection.

Generates from Qwen/Qwen3.5-9B on TruthfulQA (817), HaluEval (10K), FaithDial (3.6K).
For each generation: hidden states at specified layers, output tokens, correctness labels.
Labels hallucination onset: first token position where generation diverges from factual.
Saves as HDF5 (hidden states) + JSONL (metadata).
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
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
logger = logging.getLogger("collect_traces")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect annotated traces with hidden states")
    parser.add_argument("--config", type=str, default="configs/trace_config.yaml")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["truthfulqa", "halueval", "faithdial"],
                        choices=["truthfulqa", "halueval", "faithdial"])
    parser.add_argument("--output_dir", type=str, default="./data/traces")
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per dataset (None=use all)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_traces_per_question", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Dataset Loaders ──────────────────────────────────────────────────────────

def load_truthfulqa(max_samples=None):
    ds = load_dataset("truthful_qa", "generation", split="validation")
    samples = []
    for ex in ds:
        samples.append({
            "question": ex["question"],
            "best_answer": ex.get("best_answer", ""),
            "correct_answers": ex.get("correct_answers", []),
            "incorrect_answers": ex.get("incorrect_answers", []),
            "source": "truthfulqa",
        })
    if max_samples:
        samples = samples[:max_samples]
    return samples


def load_halueval(max_samples=None):
    ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    samples = []
    for ex in ds:
        samples.append({
            "question": ex.get("question", ex.get("query", "")),
            "best_answer": ex.get("answer", ex.get("right_answer", "")),
            "correct_answers": [ex.get("answer", ex.get("right_answer", ""))],
            "incorrect_answers": [ex.get("hallucinated_answer", "")],
            "source": "halueval",
        })
    if max_samples:
        samples = samples[:max_samples]
    return samples


def load_faithdial(max_samples=None):
    ds = load_dataset("McGill-NLP/FaithDial", split="test")
    samples = []
    for ex in ds:
        knowledge = ex.get("knowledge", "")
        history = ex.get("history", [])
        question = history[-1] if history else ex.get("utterance", "")
        gold_response = ex.get("response", "")
        vhi = ex.get("BEGIN", [])

        samples.append({
            "question": question,
            "best_answer": gold_response,
            "correct_answers": [gold_response],
            "incorrect_answers": vhi if isinstance(vhi, list) else [],
            "knowledge": knowledge,
            "source": "faithdial",
        })
    if max_samples:
        samples = samples[:max_samples]
    return samples


DATASET_LOADERS = {
    "truthfulqa": load_truthfulqa,
    "halueval": load_halueval,
    "faithdial": load_faithdial,
}


# ── Trace Generation ─────────────────────────────────────────────────────────

def build_cot_prompt(question: str) -> str:
    return (
        f"Answer the following question with a detailed chain-of-thought reasoning. "
        f"Think step by step before giving your final answer.\n\n"
        f"Question: {question}\n\n"
        f"Let me think through this step by step:\n"
    )


_claim_labeler = None


def get_claim_labeler():
    """Lazy-init the claim-level labeler to avoid loading NLI model when not needed."""
    global _claim_labeler
    if _claim_labeler is None:
        from src.claim_labeler import ClaimLevelLabeler, ClaimLabelerConfig
        config = ClaimLabelerConfig()
        _claim_labeler = ClaimLevelLabeler(config)
    return _claim_labeler


def label_hallucination_onset(
    trace_tokens: list[str],
    trace_text: str,
    correct_answers: list[str],
    incorrect_answers: list[str],
    use_claim_level: bool = True,
) -> list[int]:
    """
    Label each token position: 0=normal, 1=hallucination onset/continuation.
    Uses claim-level NLI verification by default (use_claim_level=True).
    Falls back to heuristic word-overlap if NLI model unavailable.
    """
    if use_claim_level:
        try:
            labeler = get_claim_labeler()
            result = labeler.label_trace(trace_text, correct_answers, trace_tokens)
            return result["token_labels"]
        except Exception as e:
            logger.warning(f"Claim-level labeling failed ({e}), falling back to heuristic")

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
        words = set(sentence.lower().split())
        incorrect_overlap = len(words & incorrect_set)
        correct_overlap = len(words & correct_set)
        is_hallucinated = incorrect_overlap > 2 and incorrect_overlap > correct_overlap

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
                if sent_start <= token_char_pos < sent_end:
                    labels[tidx] = 1
                token_char_pos = token_end

        char_offset = sent_end

    return labels


@torch.no_grad()
def generate_and_collect_hidden_states(
    model,
    tokenizer,
    prompts: list[str],
    layer_indices: list[int],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[dict]:
    """Generate text and collect hidden states at specified layers."""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=1024,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    results = []
    for i in range(len(prompts)):
        prompt_len = inputs["input_ids"][i].shape[0]
        gen_tokens = outputs.sequences[i][prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        tokens = [tokenizer.decode([t], skip_special_tokens=True) for t in gen_tokens]
        results.append({
            "text": text,
            "tokens": tokens,
            "token_ids": gen_tokens.tolist(),
            "prompt_len": prompt_len,
        })

    return results, outputs


@torch.no_grad()
def extract_hidden_states_for_trace(
    model,
    tokenizer,
    full_text: str,
    layer_indices: list[int],
    max_length: int = 1024,
) -> dict[int, np.ndarray]:
    """Run a forward pass on the full generated text and extract hidden states."""
    inputs = tokenizer(
        full_text, return_tensors="pt", truncation=True, max_length=max_length,
    ).to(model.device)

    outputs = model(**inputs, output_hidden_states=True)

    hidden_dict = {}
    for layer_idx in layer_indices:
        if layer_idx < len(outputs.hidden_states):
            hs = outputs.hidden_states[layer_idx][0].cpu().float().numpy().astype(np.float16)
            hidden_dict[layer_idx] = hs

    return hidden_dict, inputs["input_ids"].shape[1]


def save_traces_hdf5_jsonl(
    traces: list[dict],
    hidden_states_list: list[dict],
    output_dir: str,
    dataset_name: str,
    layer_indices: list[int],
):
    """Save hidden states as HDF5 and metadata as JSONL."""
    os.makedirs(output_dir, exist_ok=True)

    h5_path = os.path.join(output_dir, f"hidden_states_{dataset_name}.h5")
    jsonl_path = os.path.join(output_dir, f"traces_{dataset_name}.jsonl")

    with h5py.File(h5_path, "w") as h5f:
        for layer_idx in layer_indices:
            grp = h5f.create_group(f"layer_{layer_idx}")
            for i, hs_dict in enumerate(hidden_states_list):
                if layer_idx in hs_dict:
                    grp.create_dataset(
                        f"trace_{i}",
                        data=hs_dict[layer_idx],
                        compression="gzip",
                        compression_opts=4,
                    )

        h5f.attrs["num_traces"] = len(traces)
        h5f.attrs["layer_indices"] = layer_indices
        h5f.attrs["dataset"] = dataset_name

    with open(jsonl_path, "w") as f:
        for trace in traces:
            serializable = {k: v for k, v in trace.items() if k != "hidden_states"}
            f.write(json.dumps(serializable) + "\n")

    logger.info(f"Saved {len(traces)} traces: HDF5={h5_path}, JSONL={jsonl_path}")
    return h5_path, jsonl_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    all_stats = {}

    for ds_name in args.datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {ds_name}")
        logger.info(f"{'='*60}")

        loader = DATASET_LOADERS[ds_name]
        samples = loader(max_samples=args.max_samples)
        logger.info(f"Loaded {len(samples)} samples from {ds_name}")

        all_traces = []
        all_hidden_states = []

        for batch_start in tqdm(range(0, len(samples), args.batch_size), desc=f"Generating [{ds_name}]"):
            batch = samples[batch_start:batch_start + args.batch_size]

            for trace_idx in range(args.num_traces_per_question):
                prompts = [build_cot_prompt(s["question"]) for s in batch]

                try:
                    gen_results, _ = generate_and_collect_hidden_states(
                        model, tokenizer, prompts, args.layer_indices,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM at batch {batch_start}, trace {trace_idx}. Skipping.")
                    torch.cuda.empty_cache()
                    continue

                for i, (sample, gen) in enumerate(zip(batch, gen_results)):
                    full_text = prompts[i] + gen["text"]

                    try:
                        hs_dict, seq_len = extract_hidden_states_for_trace(
                            model, tokenizer, full_text, args.layer_indices,
                        )
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"OOM extracting hidden states, skipping trace")
                        torch.cuda.empty_cache()
                        hs_dict = {}

                    hallu_labels = label_hallucination_onset(
                        gen["tokens"], gen["text"],
                        sample["correct_answers"],
                        sample["incorrect_answers"],
                    )

                    has_hallucination = any(l == 1 for l in hallu_labels)
                    onset_pos = next((j for j, l in enumerate(hallu_labels) if l == 1), -1)

                    trace_entry = {
                        "trace_id": len(all_traces),
                        "sample_idx": batch_start + i,
                        "trace_idx": trace_idx,
                        "dataset": ds_name,
                        "question": sample["question"],
                        "best_answer": sample["best_answer"],
                        "generated_text": gen["text"],
                        "tokens": gen["tokens"],
                        "token_ids": gen["token_ids"],
                        "hallu_labels": hallu_labels,
                        "has_hallucination": has_hallucination,
                        "onset_position": onset_pos,
                        "num_tokens": len(gen["tokens"]),
                        "prompt_len": gen["prompt_len"],
                    }
                    all_traces.append(trace_entry)
                    all_hidden_states.append(hs_dict)

        h5_path, jsonl_path = save_traces_hdf5_jsonl(
            all_traces, all_hidden_states, args.output_dir, ds_name, args.layer_indices,
        )

        n_hallu = sum(1 for t in all_traces if t["has_hallucination"])
        stats = {
            "dataset": ds_name,
            "total_traces": len(all_traces),
            "traces_with_hallucination": n_hallu,
            "hallucination_rate": n_hallu / max(len(all_traces), 1),
            "model": args.model_name,
            "h5_path": h5_path,
            "jsonl_path": jsonl_path,
        }
        all_stats[ds_name] = stats
        logger.info(f"{ds_name}: {len(all_traces)} traces, {n_hallu} with hallucination "
                     f"({stats['hallucination_rate']:.2%})")

    del model
    torch.cuda.empty_cache()

    stats_path = os.path.join(args.output_dir, "collection_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"\nCollection complete. Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
