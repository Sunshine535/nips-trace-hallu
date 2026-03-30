"""
Detector calibration and deployment-centered evaluation metrics.

Provides:
- AUPRC (Area Under Precision-Recall Curve)
- ECE (Expected Calibration Error)
- Onset lead-time analysis
- False positive burden estimation
- Trigger curves across thresholds
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    auprc: float = 0.0
    ece: float = 0.0
    mean_lead_time: float = 0.0
    median_lead_time: float = 0.0
    false_positive_rate_at_recall_90: float = 0.0
    trigger_curve: dict = None

    def to_dict(self) -> dict:
        return {
            "auprc": self.auprc,
            "ece": self.ece,
            "mean_lead_time": self.mean_lead_time,
            "median_lead_time": self.median_lead_time,
            "fpr_at_recall_90": self.false_positive_rate_at_recall_90,
            "trigger_curve": self.trigger_curve,
        }


def compute_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area Under Precision-Recall Curve."""
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    tp = 0
    fp = 0
    total_pos = labels.sum()
    if total_pos == 0:
        return 0.0

    precisions = []
    recalls = []
    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / total_pos
        precisions.append(prec)
        recalls.append(rec)

    auprc = 0.0
    for i in range(1, len(recalls)):
        auprc += (recalls[i] - recalls[i - 1]) * precisions[i]
    return float(auprc)


def compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        bin_size = mask.sum()
        if bin_size == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_lead_time(
    predictions: list[int],
    true_onsets: list[int],
) -> list[float]:
    """
    Compute lead time: how many tokens before true onset the detector fires.
    Positive = early detection (good), negative = late detection.
    """
    lead_times = []
    for pred_onset, true_onset in zip(predictions, true_onsets):
        if true_onset < 0:
            continue
        if pred_onset < 0:
            continue
        lead_times.append(true_onset - pred_onset)
    return lead_times


def compute_trigger_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: list[float] = None,
) -> dict:
    """Compute precision, recall, F1, FPR at multiple thresholds."""
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05).tolist()

    curve = {}
    total_pos = labels.sum()
    total_neg = len(labels) - total_pos

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(total_pos, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        fpr = fp / max(total_neg, 1)

        curve[f"{t:.2f}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fpr": float(fpr),
            "n_triggers": int(preds.sum()),
        }

    return curve


def evaluate_detector_calibration(
    all_labels: np.ndarray,
    all_scores: np.ndarray,
    all_pred_onsets: list[int],
    all_true_onsets: list[int],
) -> CalibrationMetrics:
    """Full detector calibration evaluation."""
    auprc = compute_auprc(all_labels, all_scores)
    ece = compute_ece(all_labels, all_scores)
    trigger_curve = compute_trigger_curve(all_labels, all_scores)

    lead_times = compute_lead_time(all_pred_onsets, all_true_onsets)
    mean_lt = float(np.mean(lead_times)) if lead_times else 0.0
    median_lt = float(np.median(lead_times)) if lead_times else 0.0

    fpr_at_90 = 0.0
    for t_str, m in trigger_curve.items():
        if m["recall"] >= 0.9:
            fpr_at_90 = m["fpr"]
            break

    return CalibrationMetrics(
        auprc=auprc,
        ece=ece,
        mean_lead_time=mean_lt,
        median_lead_time=median_lt,
        false_positive_rate_at_recall_90=fpr_at_90,
        trigger_curve=trigger_curve,
    )
