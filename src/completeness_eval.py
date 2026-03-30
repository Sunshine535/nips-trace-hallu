"""
Completeness and helpfulness evaluation.

Ensures interventions don't improve factuality just by generating less.
Metrics:
- Response completeness: proportion of reference claims covered
- Helpfulness: combined factuality × completeness
- Abstention rate: fraction of responses that are effectively empty
- Token budget efficiency: factuality per token
"""

import logging
from dataclasses import dataclass

import numpy as np

from src.claim_labeler import ClaimExtractor

logger = logging.getLogger(__name__)


@dataclass
class CompletenessMetrics:
    completeness: float = 0.0
    helpfulness: float = 0.0
    abstention_rate: float = 0.0
    avg_response_length: float = 0.0
    factuality_per_token: float = 0.0

    def to_dict(self) -> dict:
        return {
            "completeness": self.completeness,
            "helpfulness": self.helpfulness,
            "abstention_rate": self.abstention_rate,
            "avg_response_length": self.avg_response_length,
            "factuality_per_token": self.factuality_per_token,
        }


def compute_completeness(
    generated: str,
    reference_answers: list[str],
    min_claim_length: int = 5,
) -> float:
    """
    Compute completeness: proportion of reference claims covered in the generation.
    """
    extractor = ClaimExtractor()
    reference = " ".join(reference_answers)
    ref_claims = extractor.extract_claims(reference, min_length=min_claim_length)
    if not ref_claims:
        return 1.0

    gen_lower = generated.lower()
    covered = 0
    for claim in ref_claims:
        claim_words = set(claim["text"].lower().split())
        gen_words = set(gen_lower.split())
        overlap = len(claim_words & gen_words) / max(len(claim_words), 1)
        if overlap > 0.5:
            covered += 1

    return covered / len(ref_claims)


def compute_helpfulness(factuality: float, completeness: float, alpha: float = 0.5) -> float:
    """
    Helpfulness = alpha * factuality + (1-alpha) * completeness.
    Penalizes methods that improve factuality by omitting information.
    """
    return alpha * factuality + (1 - alpha) * completeness


def is_abstention(text: str, min_tokens: int = 5) -> bool:
    """Check if a response is effectively empty/abstaining."""
    words = text.strip().split()
    if len(words) < min_tokens:
        return True

    abstention_phrases = [
        "i don't know", "i cannot", "i'm not sure",
        "i am not able", "no answer", "unable to",
    ]
    text_lower = text.lower()
    return any(p in text_lower for p in abstention_phrases)


def evaluate_completeness_batch(
    generations: list[str],
    references: list[list[str]],
    factualities: list[float],
) -> CompletenessMetrics:
    """Evaluate completeness metrics for a batch."""
    completeness_scores = []
    helpfulness_scores = []
    abstentions = 0
    total_tokens = 0

    for gen, ref, fact in zip(generations, references, factualities):
        comp = compute_completeness(gen, ref)
        completeness_scores.append(comp)

        help_score = compute_helpfulness(fact, comp)
        helpfulness_scores.append(help_score)

        if is_abstention(gen):
            abstentions += 1

        total_tokens += len(gen.split())

    n = len(generations)
    avg_len = total_tokens / max(n, 1)
    avg_fact = np.mean(factualities) if factualities else 0.0
    fpt = avg_fact / max(avg_len, 1) * 100

    return CompletenessMetrics(
        completeness=float(np.mean(completeness_scores)),
        helpfulness=float(np.mean(helpfulness_scores)),
        abstention_rate=abstentions / max(n, 1),
        avg_response_length=avg_len,
        factuality_per_token=fpt,
    )
