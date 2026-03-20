#!/usr/bin/env python3
"""
Train linear probe on hidden states to detect hallucination onset.
Binary classification per token position.
Extracts hidden states from a frozen Qwen3.5-27B and trains a lightweight classifier.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.onset_detector import OnsetDetectorConfig, OnsetLinearProbe, MultiLayerOnsetDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_onset_detector")


def parse_args():
    parser = argparse.ArgumentParser(description="Train hallucination onset detector")
    parser.add_argument("--config", type=str, default="configs/trace_config.yaml")
    parser.add_argument("--traces_file", type=str, required=True,
                        help="Path to traces JSONL file")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--multi_layer", action="store_true",
                        help="Use multi-layer ensemble detector")
    return parser.parse_args()


class HiddenStateDataset(Dataset):
    """Dataset of pre-extracted hidden states with hallucination labels."""

    def __init__(self, hidden_states, labels, attention_masks):
        self.hidden_states = hidden_states
        self.labels = labels
        self.attention_masks = attention_masks

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return {
            "hidden_states": self.hidden_states[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_masks[idx],
        }


def load_traces(traces_file: str) -> list[dict]:
    traces = []
    with open(traces_file) as f:
        for line in f:
            traces.append(json.loads(line.strip()))
    return traces


@torch.no_grad()
def extract_hidden_states(
    model,
    tokenizer,
    traces: list[dict],
    layer_indices: list[int],
    max_seq_length: int,
    batch_size: int = 4,
) -> dict:
    """Extract hidden states from specified layers for all traces."""
    all_hidden = {idx: [] for idx in layer_indices}
    all_labels = []
    all_masks = []

    for i in tqdm(range(0, len(traces), batch_size), desc="Extracting hidden states"):
        batch_traces = traces[i:i + batch_size]

        texts = [t["generated_text"] for t in batch_traces]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=max_seq_length,
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)

        for layer_idx in layer_indices:
            hs = outputs.hidden_states[layer_idx].cpu()
            all_hidden[layer_idx].append(hs)

        batch_labels = []
        for j, trace in enumerate(batch_traces):
            seq_len = inputs["input_ids"][j].shape[0]
            labels = trace["hallu_labels"][:seq_len]
            labels += [0] * (seq_len - len(labels))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))

        padded_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True)
        all_labels.append(padded_labels)
        all_masks.append(inputs["attention_mask"].cpu())

    result = {}
    for idx in layer_indices:
        result[idx] = torch.cat(all_hidden[idx], dim=0)
    labels = torch.cat(all_labels, dim=0)
    masks = torch.cat(all_masks, dim=0)

    return result, labels, masks


def train_single_layer_probe(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    config: dict,
    device: torch.device,
) -> tuple:
    det_config = OnsetDetectorConfig(
        hidden_size=hidden_states.shape[-1],
        dropout=0.1,
    )
    probe = OnsetLinearProbe(det_config).to(device)

    dataset = HiddenStateDataset(hidden_states, labels, masks)
    train_size = int(len(dataset) * config["train_ratio"])
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"],
    )

    best_f1 = 0.0
    best_state = None
    history = []

    for epoch in range(config["num_epochs"]):
        probe.train()
        train_loss = 0.0
        for batch in train_loader:
            hs = batch["hidden_states"].to(device)
            lbl = batch["labels"].to(device)
            mask = batch["attention_mask"].to(device)

            out = probe(hs, labels=lbl, attention_mask=mask)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        probe.eval()
        val_metrics = {"loss": 0, "accuracy": 0, "f1": 0, "precision": 0, "recall": 0}
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                hs = batch["hidden_states"].to(device)
                lbl = batch["labels"].to(device)
                mask = batch["attention_mask"].to(device)
                out = probe(hs, labels=lbl, attention_mask=mask)
                for k in val_metrics:
                    if k in out:
                        val_metrics[k] += out[k].item()
                n_val += 1

        for k in val_metrics:
            val_metrics[k] /= max(n_val, 1)

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_info)

        logger.info(
            f"Epoch {epoch}: train_loss={epoch_info['train_loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.cpu() for k, v in probe.state_dict().items()}

    if best_state:
        probe.load_state_dict(best_state)
    return probe, history, best_f1


def main():
    args = parse_args()
    cfg = load_config(args.config)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    det_cfg = cfg["detector"]
    output_dir = args.output_dir or cfg["output"]["detector_dir"]
    os.makedirs(output_dir, exist_ok=True)

    num_epochs = args.num_epochs or det_cfg["num_epochs"]
    batch_size = args.batch_size or det_cfg["batch_size"]
    lr = args.learning_rate or det_cfg["learning_rate"]
    layer_indices = det_cfg["num_layers_to_probe"]

    logger.info(f"Loading traces from {args.traces_file}")
    traces = load_traces(args.traces_file)
    hallu_traces = [t for t in traces if t["has_hallucination"]]
    normal_traces = [t for t in traces if not t["has_hallucination"]]
    logger.info(f"Total: {len(traces)}, Hallucinated: {len(hallu_traces)}, Normal: {len(normal_traces)}")

    model_name = cfg["generator"]["model"]
    logger.info(f"Loading model for hidden state extraction: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    logger.info(f"Extracting hidden states from layers {layer_indices}")
    all_hidden, labels, masks = extract_hidden_states(
        model, tokenizer, traces, layer_indices, args.max_seq_length, batch_size=4,
    )

    del model
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": det_cfg["weight_decay"],
        "train_ratio": det_cfg["train_ratio"],
    }

    if args.multi_layer:
        logger.info("Training multi-layer ensemble detector")
        det_config = OnsetDetectorConfig(hidden_size=det_cfg["hidden_size"])
        detector = MultiLayerOnsetDetector(det_config, layer_indices).to(device)
        logger.info("Multi-layer training not yet fully wired — falling back to best single layer")

    results = {}
    best_overall_f1 = 0.0
    best_layer = None

    for layer_idx in layer_indices:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training probe for layer {layer_idx}")
        probe, history, best_f1 = train_single_layer_probe(
            all_hidden[layer_idx], labels, masks, train_config, device,
        )
        results[layer_idx] = {"best_f1": best_f1, "history": history}

        probe_path = os.path.join(output_dir, f"probe_layer{layer_idx}.pt")
        torch.save(probe.state_dict(), probe_path)
        logger.info(f"Layer {layer_idx}: best F1 = {best_f1:.4f}, saved to {probe_path}")

        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_layer = layer_idx

    logger.info(f"\nBest layer: {best_layer} with F1 = {best_overall_f1:.4f}")

    summary = {
        "best_layer": best_layer,
        "best_f1": best_overall_f1,
        "per_layer_results": {str(k): v["best_f1"] for k, v in results.items()},
        "config": train_config,
    }
    with open(os.path.join(output_dir, "detector_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
