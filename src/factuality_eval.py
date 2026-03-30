"""
Factuality evaluation via claim extraction + NLI verification.

Replaces naive string matching with:
1. Claim extraction from generated text
2. NLI-based claim verification against reference
3. Claim-level precision, recall, F1
4. Bootstrap confidence intervals
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from src.claim_labeler import ClaimExtractor, NLILabeler, ClaimLabelerConfig

logger = logging.getLogger(__name__)


@dataclass
class FactualityConfig:
    nli_model: str = "microsoft/deberta-v3-large-mnli"
    entailment_threshold: float = 0.7
    contradiction_threshold: float = 0.5
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    min_claim_length: int = 5


class FactualityEvaluator:
    """Evaluate factuality using claim-level NLI verification."""

    def __init__(self, config: Optional[FactualityConfig] = None):
        self.config = config or FactualityConfig()
        self.extractor = ClaimExtractor()
        labeler_config = ClaimLabelerConfig(
            nli_model=self.config.nli_model,
            entailment_threshold=self.config.entailment_threshold,
            contradiction_threshold=self.config.contradiction_threshold,
        )
        self.nli = NLILabeler(labeler_config)

    def evaluate_single(
        self,
        generated: str,
        correct_answers: list[str],
        incorrect_answers: Optional[list[str]] = None,
    ) -> dict:
        claims = self.extractor.extract_claims(generated, self.config.min_claim_length)
        if not claims:
            return {
                "factuality_score": 0.5,
                "claim_precision": 0.0,
                "claim_recall": 0.0,
                "claim_f1": 0.0,
                "num_claims": 0,
                "num_supported": 0,
                "num_contradicted": 0,
                "num_neutral": 0,
            }

        reference = " ".join(correct_answers)
        claim_texts = [c["text"] for c in claims]
        results = self.nli.classify_claims(claim_texts, reference)

        supported = sum(1 for r in results if r["is_supported"])
        contradicted = sum(1 for r in results if r["is_hallucination"])
        neutral = len(results) - supported - contradicted

        precision = supported / max(len(results), 1)
        ref_claims = self.extractor.extract_claims(reference, self.config.min_claim_length)
        recall = supported / max(len(ref_claims), 1) if ref_claims else precision
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        factuality_score = precision

        return {
            "factuality_score": factuality_score,
            "claim_precision": precision,
            "claim_recall": recall,
            "claim_f1": f1,
            "num_claims": len(results),
            "num_supported": supported,
            "num_contradicted": contradicted,
            "num_neutral": neutral,
            "per_claim": results,
        }

    def evaluate_batch(
        self,
        generations: list[str],
        references: list[list[str]],
    ) -> dict:
        per_sample = []
        for gen, ref in zip(generations, references):
            r = self.evaluate_single(gen, ref)
            per_sample.append(r)

        scores = [s["factuality_score"] for s in per_sample]
        precisions = [s["claim_precision"] for s in per_sample]
        f1s = [s["claim_f1"] for s in per_sample]

        agg = {
            "mean_factuality": float(np.mean(scores)),
            "mean_claim_precision": float(np.mean(precisions)),
            "mean_claim_f1": float(np.mean(f1s)),
            "n_samples": len(per_sample),
        }

        ci = self._bootstrap_ci(scores)
        agg["factuality_95ci"] = ci

        return agg

    def _bootstrap_ci(self, scores: list[float]) -> tuple[float, float]:
        arr = np.array(scores)
        n = len(arr)
        if n < 2:
            return (float(arr.mean()), float(arr.mean()))

        rng = np.random.default_rng(42)
        means = []
        for _ in range(self.config.bootstrap_samples):
            sample = rng.choice(arr, size=n, replace=True)
            means.append(sample.mean())

        means = np.sort(means)
        alpha = 1 - self.config.confidence_level
        lo = float(means[int(alpha / 2 * len(means))])
        hi = float(means[int((1 - alpha / 2) * len(means))])
        return (lo, hi)


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap significance test between two systems."""
    a = np.array(scores_a)
    b = np.array(scores_b)
    assert len(a) == len(b), "Same-length score arrays required"

    observed_diff = float(a.mean() - b.mean())
    n = len(a)
    rng = np.random.default_rng(seed)

    count_ge = 0
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diff = a[idx].mean() - b[idx].mean()
        diffs.append(diff)
        if diff <= 0:
            count_ge += 1

    p_value = count_ge / n_bootstrap
    diffs = np.sort(diffs)

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "ci_95": (float(diffs[int(0.025 * len(diffs))]), float(diffs[int(0.975 * len(diffs))])),
        "effect_size": observed_diff / max(np.std(np.array(diffs)), 1e-8),
    }
