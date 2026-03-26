#!/usr/bin/env python3
"""
Extract hidden states from pre-existing JSONL traces and save as HDF5.

Avoids re-generating text when traces already exist (e.g., from a previous
run that crashed before saving hidden states).

Usage:
    python scripts/extract_hidden_states.py \
        --jsonl_path traces_truthfulqa.jsonl \
        --output_dir data/traces \
        --dataset_name truthfulqa \
        --model_name Qwen/Qwen3.5-9B \
        --layer_indices 8 16 24 32
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_hidden_states")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract hidden states from existing JSONL traces into HDF5"
    )
    parser.add_argument("--jsonl_path", type=str, required=True,
                        help="Path to existing JSONL trace file")
    parser.add_argument("--output_dir", type=str, default="./data/traces",
                        help="Output directory for HDF5 + formatted JSONL")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name (truthfulqa, halueval, faithdial)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for forward passes (1 is safest for variable lengths)")
    parser.add_argument("--max_length", type=int, default=1024)
    return parser.parse_args()


def build_cot_prompt(question: str) -> str:
    return (
        f"Answer the following question with a detailed chain-of-thought reasoning. "
        f"Think step by step before giving your final answer.\n\n"
        f"Question: {question}\n\n"
        f"Let me think through this step by step:\n"
    )


@torch.no_grad()
def extract_hidden_states(model, tokenizer, full_text, layer_indices, max_length=1024):
    """Run forward pass and extract hidden states at specified layers."""
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


def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading traces from {args.jsonl_path}")
    traces = []
    with open(args.jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    logger.info(f"Loaded {len(traces)} traces")

    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    h5_path = os.path.join(args.output_dir, f"hidden_states_{args.dataset_name}.h5")
    jsonl_path = os.path.join(args.output_dir, f"traces_{args.dataset_name}.jsonl")

    all_hidden_states = []
    failed = 0

    for i, trace in enumerate(tqdm(traces, desc=f"Extracting [{args.dataset_name}]")):
        question = trace.get("question", "")
        gen_text = trace.get("generated_text", "")
        full_text = build_cot_prompt(question) + gen_text

        try:
            hs_dict, seq_len = extract_hidden_states(
                model, tokenizer, full_text, args.layer_indices, args.max_length,
            )
            all_hidden_states.append(hs_dict)
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM at trace {i}, using empty hidden states")
            torch.cuda.empty_cache()
            all_hidden_states.append({})
            failed += 1
        except Exception as e:
            logger.warning(f"Error at trace {i}: {e}")
            all_hidden_states.append({})
            failed += 1

    logger.info(f"Extraction complete: {len(traces) - failed}/{len(traces)} successful")

    logger.info(f"Saving HDF5 to {h5_path}")
    with h5py.File(h5_path, "w") as h5f:
        for layer_idx in args.layer_indices:
            grp = h5f.create_group(f"layer_{layer_idx}")
            saved = 0
            for i, hs_dict in enumerate(all_hidden_states):
                if layer_idx in hs_dict:
                    grp.create_dataset(
                        f"trace_{i}",
                        data=hs_dict[layer_idx],
                        compression="gzip",
                        compression_opts=4,
                    )
                    saved += 1
            logger.info(f"  Layer {layer_idx}: saved {saved}/{len(traces)} traces")

        h5f.attrs["num_traces"] = len(traces)
        h5f.attrs["layer_indices"] = args.layer_indices
        h5f.attrs["dataset"] = args.dataset_name

    logger.info(f"Saving JSONL to {jsonl_path}")
    with open(jsonl_path, "w") as f:
        for trace in traces:
            serializable = {k: v for k, v in trace.items() if k != "hidden_states"}
            f.write(json.dumps(serializable) + "\n")

    stats = {
        "dataset": args.dataset_name,
        "total_traces": len(traces),
        "successful_extractions": len(traces) - failed,
        "failed_extractions": failed,
        "model": args.model_name,
        "layer_indices": args.layer_indices,
        "h5_path": h5_path,
        "jsonl_path": jsonl_path,
    }
    logger.info(f"Stats: {json.dumps(stats, indent=2)}")

    del model
    torch.cuda.empty_cache()

    return stats


if __name__ == "__main__":
    main()
