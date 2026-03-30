#!/usr/bin/env python3
"""
Train MultiLayerOnsetDetector on collected traces from HDF5 + JSONL.

Loads hidden states from layers [8, 16, 24, 32] of Qwen3.5-9B.
Binary classification per token: hallucinating or not.
Evaluates: precision, recall, F1, false alarm rate.
Saves best detector checkpoint.
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.onset_detector import OnsetDetectorConfig, OnsetLinearProbe, MultiLayerOnsetDetector


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
logger = logging.getLogger("train_onset_detector")


def parse_args():
    parser = argparse.ArgumentParser(description="Train hallucination onset detector")
    parser.add_argument("--config", type=str, default="configs/trace_config.yaml")
    parser.add_argument("--traces_dir", type=str, required=True,
                        help="Directory containing HDF5 and JSONL trace files")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["truthfulqa", "halueval", "faithdial"])
    parser.add_argument("--output_dir", type=str, default="./checkpoints/detector")
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--hidden_size", type=int, default=4096,
                        help="Hidden size of the base model (Qwen3.5-9B=4096)")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="'auto' to resume from latest checkpoint, or path")
    return parser.parse_args()


class TraceHiddenStateDataset(Dataset):
    """Load hidden states from HDF5 and labels from JSONL."""

    def __init__(self, traces_dir, dataset_names, layer_indices, max_seq_length=512):
        self.layer_indices = layer_indices
        self.max_seq_length = max_seq_length
        self.items = []

        for ds_name in dataset_names:
            h5_path = os.path.join(traces_dir, f"hidden_states_{ds_name}.h5")
            jsonl_path = os.path.join(traces_dir, f"traces_{ds_name}.jsonl")

            if not os.path.exists(h5_path) or not os.path.exists(jsonl_path):
                logger.warning(f"Missing files for {ds_name}, skipping")
                continue

            traces = []
            with open(jsonl_path) as f:
                for line in f:
                    traces.append(json.loads(line.strip()))

            with h5py.File(h5_path, "r") as h5f:
                for i, trace in enumerate(traces):
                    layer_data = {}
                    valid = True
                    for layer_idx in layer_indices:
                        key = f"layer_{layer_idx}/trace_{i}"
                        if key in h5f:
                            hs = h5f[key][:]
                            seq_len = min(hs.shape[0], max_seq_length)
                            layer_data[layer_idx] = hs[:seq_len].astype(np.float32)
                        else:
                            valid = False
                            break

                    if not valid:
                        continue

                    labels = trace["hallu_labels"]
                    seq_len = layer_data[layer_indices[0]].shape[0]
                    labels = labels[:seq_len]
                    labels += [0] * (seq_len - len(labels))

                    self.items.append({
                        "layer_data": layer_data,
                        "labels": np.array(labels, dtype=np.int64),
                        "mask": np.ones(seq_len, dtype=np.float32),
                    })

            logger.info(f"Loaded {len(self.items)} traces from {ds_name}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "layer_data": {k: torch.from_numpy(v) for k, v in item["layer_data"].items()},
            "labels": torch.from_numpy(item["labels"]),
            "mask": torch.from_numpy(item["mask"]),
        }


def collate_traces(batch):
    """Custom collate to handle variable-length sequences and multi-layer data."""
    layer_indices = list(batch[0]["layer_data"].keys())
    max_len = max(b["labels"].shape[0] for b in batch)
    hidden_size = batch[0]["layer_data"][layer_indices[0]].shape[-1]
    bsz = len(batch)

    padded_layers = {}
    for layer_idx in layer_indices:
        padded = torch.zeros(bsz, max_len, hidden_size)
        for i, b in enumerate(batch):
            seq_len = b["layer_data"][layer_idx].shape[0]
            padded[i, :seq_len] = b["layer_data"][layer_idx]
        padded_layers[layer_idx] = padded

    padded_labels = torch.zeros(bsz, max_len, dtype=torch.long)
    padded_mask = torch.zeros(bsz, max_len)
    for i, b in enumerate(batch):
        seq_len = b["labels"].shape[0]
        padded_labels[i, :seq_len] = b["labels"]
        padded_mask[i, :seq_len] = b["mask"]

    return {
        "layer_data": padded_layers,
        "labels": padded_labels,
        "mask": padded_mask,
    }


def compute_metrics(logits, labels, mask):
    """Compute precision, recall, F1, accuracy, and false alarm rate."""
    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)

    active = mask.view(-1) == 1
    active_preds = preds.view(-1)[active]
    active_labels = labels.view(-1)[active]

    tp = ((active_preds == 1) & (active_labels == 1)).sum().float()
    fp = ((active_preds == 1) & (active_labels == 0)).sum().float()
    fn = ((active_preds == 0) & (active_labels == 1)).sum().float()
    tn = ((active_preds == 0) & (active_labels == 0)).sum().float()

    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = (2 * precision * recall / (precision + recall + 1e-8))
    accuracy = ((active_preds == active_labels).float().mean()).item()
    false_alarm_rate = (fp / (fp + tn + 1e-8)).item()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "false_alarm_rate": false_alarm_rate,
    }


def train_single_layer(
    dataset, layer_idx, hidden_size, config, device, output_dir=None,
):
    """Train a single-layer onset probe."""
    det_config = OnsetDetectorConfig(
        hidden_size=hidden_size,
        dropout=config["dropout"],
    )
    probe = OnsetLinearProbe(det_config).to(device)

    train_size = int(len(dataset) * config["train_ratio"])
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_traces,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], collate_fn=collate_traces,
    )

    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    best_f1 = 0.0
    best_state = None
    history = []
    start_epoch = 0

    if output_dir:
        ckpt_path = find_latest_checkpoint(output_dir, f"checkpoint_probe_layer{layer_idx}_epoch*.pt")
        if ckpt_path:
            logger.info("  Resuming probe layer %d from %s", layer_idx, ckpt_path)
            start_epoch, _ = load_training_checkpoint(ckpt_path, probe, optimizer)
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            best_f1 = ckpt_data.get("best_f1", 0.0)
            history = ckpt_data.get("history", [])
            if ckpt_data.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])

    for epoch in range(start_epoch, config["num_epochs"]):
        probe.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            hs = batch["layer_data"][layer_idx].to(device)
            lbl = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            out = probe(hs, labels=lbl, attention_mask=mask)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train += 1

        scheduler.step()

        probe.eval()
        val_logits_all, val_labels_all, val_mask_all = [], [], []
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                hs = batch["layer_data"][layer_idx].to(device)
                lbl = batch["labels"].to(device)
                mask = batch["mask"].to(device)
                out = probe(hs, labels=lbl, attention_mask=mask)
                val_loss += out["loss"].item()
                n_val += 1
                val_logits_all.append(out["logits"].cpu())
                val_labels_all.append(lbl.cpu())
                val_mask_all.append(mask.cpu())

        val_logits = torch.cat(val_logits_all, dim=0)
        val_labels = torch.cat(val_labels_all, dim=0)
        val_masks = torch.cat(val_mask_all, dim=0)
        metrics = compute_metrics(val_logits, val_labels, val_masks)

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_train, 1),
            "val_loss": val_loss / max(n_val, 1),
            **{f"val_{k}": v for k, v in metrics.items()},
        }
        history.append(epoch_info)

        logger.info(
            f"  Epoch {epoch:3d}: train_loss={epoch_info['train_loss']:.4f} "
            f"val_f1={metrics['f1']:.4f} val_prec={metrics['precision']:.4f} "
            f"val_rec={metrics['recall']:.4f} val_far={metrics['false_alarm_rate']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        if output_dir:
            save_training_checkpoint(
                os.path.join(output_dir, f"checkpoint_probe_layer{layer_idx}_epoch{epoch + 1}.pt"),
                probe, optimizer, epoch + 1, 0,
                best_f1=best_f1, history=history,
                scheduler_state_dict=scheduler.state_dict(),
            )

    if best_state:
        probe.load_state_dict(best_state)
    return probe, history, best_f1


def train_multi_layer_ensemble(
    dataset, layer_indices, hidden_size, config, device, output_dir=None,
):
    """Train the multi-layer ensemble detector end-to-end."""
    det_config = OnsetDetectorConfig(hidden_size=hidden_size, dropout=config["dropout"])
    detector = MultiLayerOnsetDetector(det_config, layer_indices).to(device)

    train_size = int(len(dataset) * config["train_ratio"])
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_traces,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], collate_fn=collate_traces,
    )

    optimizer = torch.optim.AdamW(
        detector.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    best_f1 = 0.0
    best_state = None
    history = []
    start_epoch = 0

    if output_dir:
        ckpt_path = find_latest_checkpoint(output_dir, "checkpoint_ensemble_epoch*.pt")
        if ckpt_path:
            logger.info("  Resuming ensemble from %s", ckpt_path)
            start_epoch, _ = load_training_checkpoint(ckpt_path, detector, optimizer)
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            best_f1 = ckpt_data.get("best_f1", 0.0)
            history = ckpt_data.get("history", [])
            if ckpt_data.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])

    for epoch in range(start_epoch, config["num_epochs"]):
        detector.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            all_hs = {idx: batch["layer_data"][idx].to(device) for idx in layer_indices}
            lbl = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            out = detector(all_hs, labels=lbl, attention_mask=mask)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train += 1

        scheduler.step()

        detector.eval()
        val_logits_all, val_labels_all, val_mask_all = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                all_hs = {idx: batch["layer_data"][idx].to(device) for idx in layer_indices}
                lbl = batch["labels"].to(device)
                mask = batch["mask"].to(device)
                out = detector(all_hs, labels=lbl, attention_mask=mask)
                val_logits_all.append(out["logits"].cpu())
                val_labels_all.append(lbl.cpu())
                val_mask_all.append(mask.cpu())

        val_logits = torch.cat(val_logits_all, dim=0)
        val_labels = torch.cat(val_labels_all, dim=0)
        val_masks = torch.cat(val_mask_all, dim=0)
        metrics = compute_metrics(val_logits, val_labels, val_masks)

        weights = detector.layer_weights.detach().cpu()
        weights_norm = torch.softmax(weights, dim=0).tolist()

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_train, 1),
            **{f"val_{k}": v for k, v in metrics.items()},
            "layer_weights": {str(idx): w for idx, w in zip(layer_indices, weights_norm)},
        }
        history.append(epoch_info)

        logger.info(
            f"  Epoch {epoch:3d}: val_f1={metrics['f1']:.4f} val_prec={metrics['precision']:.4f} "
            f"val_rec={metrics['recall']:.4f} val_far={metrics['false_alarm_rate']:.4f} "
            f"weights={[f'{w:.3f}' for w in weights_norm]}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in detector.state_dict().items()}

        if output_dir:
            save_training_checkpoint(
                os.path.join(output_dir, f"checkpoint_ensemble_epoch{epoch + 1}.pt"),
                detector, optimizer, epoch + 1, 0,
                best_f1=best_f1, history=history,
                scheduler_state_dict=scheduler.state_dict(),
            )

    if best_state:
        detector.load_state_dict(best_state)
    return detector, history, best_f1


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Loading traces from {args.traces_dir}")
    dataset = TraceHiddenStateDataset(
        args.traces_dir, args.datasets, args.layer_indices, args.max_seq_length,
    )
    logger.info(f"Total traces loaded: {len(dataset)}")

    if len(dataset) == 0:
        logger.error("No traces found. Run collect_traces.py first.")
        return

    config = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_ratio": args.train_ratio,
        "dropout": args.dropout,
    }

    # Phase 1: Train per-layer probes
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Training per-layer probes")
    logger.info("=" * 60)

    per_layer_results = {}
    best_overall_f1 = 0.0
    best_layer = None

    resume_dir = args.output_dir if args.resume_from_checkpoint else None

    for layer_idx in args.layer_indices:
        logger.info(f"\n--- Layer {layer_idx} ---")
        probe, history, best_f1 = train_single_layer(
            dataset, layer_idx, args.hidden_size, config, device,
            output_dir=resume_dir,
        )
        per_layer_results[layer_idx] = {"best_f1": best_f1, "history": history}

        probe_path = os.path.join(args.output_dir, f"probe_layer{layer_idx}.pt")
        torch.save(probe.state_dict(), probe_path)
        logger.info(f"Layer {layer_idx}: best F1 = {best_f1:.4f}")

        if best_f1 > best_overall_f1:
            best_overall_f1 = best_f1
            best_layer = layer_idx

    logger.info(f"\nBest single-layer probe: layer {best_layer} (F1={best_overall_f1:.4f})")

    # Phase 2: Train multi-layer ensemble
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Training multi-layer ensemble detector")
    logger.info("=" * 60)

    detector, ensemble_history, ensemble_f1 = train_multi_layer_ensemble(
        dataset, args.layer_indices, args.hidden_size, config, device,
        output_dir=resume_dir,
    )

    detector_path = os.path.join(args.output_dir, "multi_layer_detector.pt")
    torch.save(detector.state_dict(), detector_path)
    logger.info(f"Multi-layer ensemble: F1 = {ensemble_f1:.4f}")

    summary = {
        "best_single_layer": best_layer,
        "best_single_f1": best_overall_f1,
        "ensemble_f1": ensemble_f1,
        "layer_indices": args.layer_indices,
        "hidden_size": args.hidden_size,
        "per_layer_results": {str(k): v["best_f1"] for k, v in per_layer_results.items()},
        "config": config,
    }
    with open(os.path.join(args.output_dir, "detector_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nTraining complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
