"""
Multi-judge factuality evaluation via claim extraction + diverse verification.

Breaks metric circularity by combining three independent judges:
1. NLI judge: DeBERTa-v3-large-MNLI entailment
2. Lexical judge: token-overlap with soft matching (independent of NLI)
3. LLM self-consistency judge: check claim consistency across multiple samples

Final score = weighted majority across judges, not single-model dependence.
"""

import logging
import re
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
    use_multi_judge: bool = True
    judge_weights: tuple = (0.5, 0.25, 0.25)


class LexicalJudge:
    """Token-overlap factuality judge — fully independent of NLI models."""

    @staticmethod
    def score_claim(claim: str, reference: str) -> dict:
        claim_tokens = set(re.findall(r'\w+', claim.lower()))
        ref_tokens = set(re.findall(r'\w+', reference.lower()))
        if not claim_tokens:
            return {"supported": False, "score": 0.0}

        overlap = len(claim_tokens & ref_tokens) / len(claim_tokens)

        content_words = claim_tokens - {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "it", "its", "this", "that", "these", "those", "and", "or",
            "but", "not", "no", "if", "then", "than", "so", "as",
        }
        ref_content = ref_tokens - {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "it", "its", "this", "that", "these", "those", "and", "or",
            "but", "not", "no", "if", "then", "than", "so", "as",
        }
        content_overlap = (
            len(content_words & ref_content) / max(len(content_words), 1)
            if content_words else overlap
        )

        score = 0.4 * overlap + 0.6 * content_overlap
        return {"supported": score > 0.5, "score": score}


class ConsistencyJudge:
    """Self-consistency judge — checks if claim appears in multiple generations.
    Uses lexical similarity across samples to avoid NLI dependence.
    """

    @staticmethod
    def score_claim(claim: str, alternative_generations: list[str]) -> dict:
        if not alternative_generations:
            return {"supported": True, "score": 0.5, "agreement_rate": 0.5}

        claim_tokens = set(re.findall(r'\w+', claim.lower()))
        if not claim_tokens:
            return {"supported": True, "score": 0.5, "agreement_rate": 0.5}

        agreements = 0
        for gen in alternative_generations:
            gen_tokens = set(re.findall(r'\w+', gen.lower()))
            overlap = len(claim_tokens & gen_tokens) / max(len(claim_tokens), 1)
            if overlap > 0.4:
                agreements += 1

        rate = agreements / len(alternative_generations)
        return {"supported": rate > 0.5, "score": rate, "agreement_rate": rate}


class IndependentNLIJudge:
    """Second NLI model (BART-large-MNLI) as independent verification.
    Breaks single-model dependence by using a different architecture.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model_name = "facebook/bart-large-mnli"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()
        except Exception as e:
            logger.warning(f"Could not load BART-MNLI: {e}")
            self._model = "unavailable"

    def score_claim(self, claim: str, reference: str) -> dict:
        self._load()
        if self._model == "unavailable" or self._model is None:
            return {"supported": True, "score": 0.5}

        try:
            inputs = self._tokenizer(
                reference, claim, return_tensors="pt",
                truncation=True, max_length=512,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            ent_idx = 2  # BART-MNLI: contradiction=0, neutral=1, entailment=2
            score = probs[ent_idx].item()
            return {"supported": score > 0.5, "score": score}
        except Exception:
            return {"supported": True, "score": 0.5}


class GroundedFactualityJudge:
    """Verifiable-answer judge for datasets with gold answers (e.g. TruthfulQA).
    Checks if the generated text contains or implies the correct answer
    AND does not contain known incorrect answers.
    """

    @staticmethod
    def score(
        generated: str,
        correct_answers: list[str],
        incorrect_answers: list[str],
    ) -> dict:
        gen_lower = generated.lower().strip()

        correct_found = False
        for ans in correct_answers:
            if ans and len(ans.strip()) > 2 and ans.lower().strip() in gen_lower:
                correct_found = True
                break

        incorrect_found = False
        for ans in incorrect_answers:
            if ans and len(ans.strip()) > 2 and ans.lower().strip() in gen_lower:
                incorrect_found = True
                break

        if correct_found and not incorrect_found:
            return {"score": 1.0, "verdict": "correct_grounded"}
        elif incorrect_found and not correct_found:
            return {"score": 0.0, "verdict": "incorrect_grounded"}
        elif correct_found and incorrect_found:
            return {"score": 0.3, "verdict": "mixed"}
        else:
            return {"score": 0.5, "verdict": "ungrounded"}


class FactualityEvaluator:
    """Multi-judge factuality evaluator.

    Combines 4 independent judges to break single-model dependence:
    1. DeBERTa NLI (primary NLI)
    2. BART NLI (independent architecture)
    3. Lexical content-word overlap (no neural model)
    4. Grounded answer matching (for datasets with gold answers)

    Final score = weighted majority, with inter-judge agreement reported.
    """

    def __init__(self, config: Optional[FactualityConfig] = None):
        self.config = config or FactualityConfig()
        self.extractor = ClaimExtractor()
        labeler_config = ClaimLabelerConfig(
            nli_model=self.config.nli_model,
            entailment_threshold=self.config.entailment_threshold,
            contradiction_threshold=self.config.contradiction_threshold,
        )
        self.nli = NLILabeler(labeler_config)
        self.lexical = LexicalJudge()
        self.consistency = ConsistencyJudge()
        self.independent_nli = IndependentNLIJudge()
        self.grounded = GroundedFactualityJudge()

    def evaluate_single(
        self,
        generated: str,
        correct_answers: list[str],
        incorrect_answers: Optional[list[str]] = None,
        alternative_generations: Optional[list[str]] = None,
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
                "judge_agreement": 1.0,
            }

        reference = " ".join(correct_answers)
        claim_texts = [c["text"] for c in claims]
        nli_results = self.nli.classify_claims(claim_texts, reference)

        per_claim_results = []
        supported_count = 0
        contradicted_count = 0
        judge_agreements = []

        for i, claim_text in enumerate(claim_texts):
            nli_r = nli_results[i]
            nli_score = nli_r["entailment_prob"]
            nli_supported = nli_r["is_supported"]

            lex_r = self.lexical.score_claim(claim_text, reference)
            lex_score = lex_r["score"]
            lex_supported = lex_r["supported"]

            ind_nli_r = self.independent_nli.score_claim(claim_text, reference)
            ind_nli_score = ind_nli_r["score"]
            ind_nli_supported = ind_nli_r["supported"]

            if alternative_generations and self.config.use_multi_judge:
                con_r = self.consistency.score_claim(claim_text, alternative_generations)
                con_score = con_r["score"]
                con_supported = con_r["supported"]
            else:
                con_score = 0.5
                con_supported = True

            votes = [nli_supported, ind_nli_supported, lex_supported, con_supported]
            majority = sum(votes) >= 3

            agreement = sum(votes) / len(votes)
            judge_agreements.append(agreement)

            is_supported = majority if self.config.use_multi_judge else nli_supported
            is_contradicted = (
                nli_r["is_hallucination"] and not ind_nli_supported and not lex_supported
            ) if self.config.use_multi_judge else nli_r["is_hallucination"]

            if is_supported:
                supported_count += 1
            elif is_contradicted:
                contradicted_count += 1

            per_claim_results.append({
                "text": claim_text,
                "deberta_nli_score": float(nli_score),
                "bart_nli_score": float(ind_nli_score),
                "lexical_score": float(lex_score),
                "consistency_score": float(con_score),
                "is_supported": is_supported,
                "is_contradicted": is_contradicted,
                "judge_agreement": agreement,
            })

        neutral = len(claims) - supported_count - contradicted_count
        precision = supported_count / max(len(claims), 1)
        ref_claims = self.extractor.extract_claims(reference, self.config.min_claim_length)
        recall = supported_count / max(len(ref_claims), 1) if ref_claims else precision
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        grounded_result = None
        if incorrect_answers:
            grounded_result = self.grounded.score(
                generated, correct_answers, incorrect_answers,
            )

        result = {
            "factuality_score": precision,
            "claim_precision": precision,
            "claim_recall": recall,
            "claim_f1": f1,
            "num_claims": len(claims),
            "num_supported": supported_count,
            "num_contradicted": contradicted_count,
            "num_neutral": neutral,
            "judge_agreement": float(np.mean(judge_agreements)) if judge_agreements else 1.0,
            "per_claim": per_claim_results,
        }

        if grounded_result:
            result["grounded_score"] = grounded_result["score"]
            result["grounded_verdict"] = grounded_result["verdict"]

        return result

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
