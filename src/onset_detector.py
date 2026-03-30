"""
Linear probe on transformer hidden states for hallucination onset detection.
Binary classification per token position: hallucinating or not.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class OnsetDetectorConfig:
    hidden_size: int = 4096
    num_classes: int = 2
    dropout: float = 0.1
    use_layer_norm: bool = True


class OnsetLinearProbe(nn.Module):
    """
    Linear probe that takes hidden states from a specific transformer layer
    and classifies each token position as hallucination onset or not.
    """

    def __init__(self, config: OnsetDetectorConfig):
        super().__init__()
        self.config = config

        layers = []
        if config.use_layer_norm:
            layers.append(nn.LayerNorm(config.hidden_size))
        layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(config.hidden_size, config.num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            labels: (batch, seq_len) — 0=normal, 1=hallucination onset
            attention_mask: (batch, seq_len)
        """
        logits = self.classifier(hidden_states)  # (B, L, 2)

        output = {"logits": logits}

        if labels is not None:
            if attention_mask is not None:
                active = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_classes)[active]
                active_labels = labels.view(-1)[active]
            else:
                active_logits = logits.view(-1, self.config.num_classes)
                active_labels = labels.view(-1)

            loss = F.cross_entropy(active_logits, active_labels)
            output["loss"] = loss

            preds = active_logits.argmax(dim=-1)
            output["accuracy"] = (preds == active_labels).float().mean()

            tp = ((preds == 1) & (active_labels == 1)).sum().float()
            fp = ((preds == 1) & (active_labels == 0)).sum().float()
            fn = ((preds == 0) & (active_labels == 1)).sum().float()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            output["precision"] = precision
            output["recall"] = recall
            output["f1"] = f1

        return output


class MultiLayerOnsetDetector(nn.Module):
    """
    Ensemble of linear probes across multiple transformer layers.
    Final prediction is a weighted combination of per-layer logits.
    """

    def __init__(self, config: OnsetDetectorConfig, layer_indices: list[int]):
        super().__init__()
        self.layer_indices = layer_indices
        self.probes = nn.ModuleDict({
            str(idx): OnsetLinearProbe(config)
            for idx in layer_indices
        })
        self.layer_weights = nn.Parameter(torch.ones(len(layer_indices)) / len(layer_indices))

    def forward(
        self,
        all_hidden_states: dict[int, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            all_hidden_states: dict mapping layer_index -> (batch, seq_len, hidden_size)
        """
        weights = F.softmax(self.layer_weights, dim=0)

        combined_logits = None
        for i, idx in enumerate(self.layer_indices):
            hs = all_hidden_states[idx]
            out = self.probes[str(idx)](hs)
            logits = out["logits"] * weights[i]
            combined_logits = logits if combined_logits is None else combined_logits + logits

        output = {"logits": combined_logits, "layer_weights": weights.detach()}

        if labels is not None:
            if attention_mask is not None:
                active = attention_mask.view(-1) == 1
                active_logits = combined_logits.view(-1, 2)[active]
                active_labels = labels.view(-1)[active]
            else:
                active_logits = combined_logits.view(-1, 2)
                active_labels = labels.view(-1)

            output["loss"] = F.cross_entropy(active_logits, active_labels)
            preds = active_logits.argmax(dim=-1)
            output["accuracy"] = (preds == active_labels).float().mean()

        return output


def find_onset_positions(
    logits: torch.Tensor,
    threshold: float = 0.5,
    min_consecutive: int = 2,
) -> list[list[int]]:
    """
    Find hallucination onset positions from detector logits.

    Returns list of onset positions for each sequence in the batch.
    Onset = first position where P(hallu) > threshold for min_consecutive tokens.
    """
    probs = F.softmax(logits, dim=-1)[:, :, 1]  # (B, L) prob of class 1
    batch_onsets = []

    for b in range(probs.shape[0]):
        seq_probs = probs[b]
        above = (seq_probs > threshold).cpu().tolist()

        onsets = []
        count = 0
        for pos, is_above in enumerate(above):
            if is_above:
                count += 1
                if count >= min_consecutive:
                    onset_pos = pos - min_consecutive + 1
                    if not onsets or onset_pos > onsets[-1] + 5:
                        onsets.append(onset_pos)
            else:
                count = 0
        batch_onsets.append(onsets)

    return batch_onsets
