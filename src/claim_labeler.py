"""
Claim-level hallucination labeling using NLI + LLM verification.

Replaces heuristic word-overlap with:
1. Claim extraction: split generated text into atomic claims
2. NLI verification: check each claim against reference using entailment model
3. Span mapping: map verified claim labels back to token positions
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ClaimLabelerConfig:
    nli_model: str = "microsoft/deberta-v3-large-mnli"
    entailment_threshold: float = 0.7
    contradiction_threshold: float = 0.5
    min_claim_length: int = 5
    device: str = "auto"


class ClaimExtractor:
    """Extract atomic claims from generated text."""

    CLAIM_DELIMITERS = re.compile(r"(?<=[.!?;])\s+|(?<=\n)")
    STEP_PATTERN = re.compile(r"(?:Step\s+\d+[:.])|(Therefore|So|Thus|Hence|In conclusion)", re.I)

    @staticmethod
    def extract_claims(text: str, min_length: int = 5) -> list[dict]:
        sentences = ClaimExtractor.CLAIM_DELIMITERS.split(text.strip())
        claims = []
        char_offset = 0

        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) < min_length:
                char_offset = text.find(sent, char_offset) + len(sent)
                continue

            start = text.find(sent, max(0, char_offset - 5))
            if start == -1:
                start = char_offset
            end = start + len(sent)

            claims.append({
                "text": sent,
                "char_start": start,
                "char_end": end,
                "is_reasoning_step": bool(ClaimExtractor.STEP_PATTERN.search(sent)),
            })
            char_offset = end

        return claims


class NLILabeler:
    """Label claims using NLI entailment model."""

    LABEL_MAP = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def __init__(self, config: ClaimLabelerConfig):
        self.config = config

        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info(f"Loading NLI model: {config.nli_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.nli_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.nli_model
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def classify_claims(
        self,
        claims: list[str],
        reference: str,
        batch_size: int = 16,
    ) -> list[dict]:
        results = []
        for i in range(0, len(claims), batch_size):
            batch = claims[i:i + batch_size]
            pairs = [(reference, claim) for claim in batch]
            inputs = self.tokenizer(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

            for j, (p, claim) in enumerate(zip(probs, batch)):
                result = {
                    "claim": claim,
                    "entailment_prob": p[2].item(),
                    "contradiction_prob": p[0].item(),
                    "neutral_prob": p[1].item(),
                    "predicted_label": self.LABEL_MAP[p.argmax().item()],
                    "is_hallucination": p[0].item() > self.config.contradiction_threshold,
                    "is_supported": p[2].item() > self.config.entailment_threshold,
                }
                results.append(result)

        return results


class ClaimLevelLabeler:
    """
    Full claim-level hallucination labeler.
    Pipeline: text -> claims -> NLI verification -> token-level labels.
    """

    def __init__(self, config: Optional[ClaimLabelerConfig] = None):
        self.config = config or ClaimLabelerConfig()
        self.extractor = ClaimExtractor()
        self.nli = NLILabeler(self.config)

    def label_trace(
        self,
        generated_text: str,
        reference_answers: list[str],
        tokens: list[str],
    ) -> dict:
        claims = self.extractor.extract_claims(
            generated_text, min_length=self.config.min_claim_length
        )

        if not claims:
            return {
                "claims": [],
                "token_labels": [0] * len(tokens),
                "has_hallucination": False,
                "onset_position": -1,
                "hallucination_rate": 0.0,
                "claim_results": [],
            }

        reference_text = " ".join(reference_answers)
        claim_texts = [c["text"] for c in claims]
        nli_results = self.nli.classify_claims(claim_texts, reference_text)

        for claim, result in zip(claims, nli_results):
            claim.update(result)

        token_labels = [0] * len(tokens)
        onset_position = -1
        token_char_positions = self._map_tokens_to_chars(tokens, generated_text)

        for claim in claims:
            if not claim["is_hallucination"]:
                continue

            for tidx, (tstart, tend) in enumerate(token_char_positions):
                if tstart >= claim["char_start"] and tstart < claim["char_end"]:
                    token_labels[tidx] = 1
                    if onset_position < 0:
                        onset_position = tidx

        has_hallucination = any(l == 1 for l in token_labels)
        n_hallu = sum(token_labels)
        hallu_rate = n_hallu / max(len(tokens), 1)

        return {
            "claims": claims,
            "token_labels": token_labels,
            "has_hallucination": has_hallucination,
            "onset_position": onset_position,
            "hallucination_rate": hallu_rate,
            "claim_results": nli_results,
        }

    @staticmethod
    def _map_tokens_to_chars(tokens: list[str], text: str) -> list[tuple[int, int]]:
        positions = []
        offset = 0
        for token in tokens:
            clean = token.lstrip("Ġ▁ ")
            idx = text.find(clean, offset)
            if idx == -1:
                idx = offset
            positions.append((idx, idx + len(clean)))
            offset = idx + max(len(clean), 1)
        return positions
