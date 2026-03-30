"""
Baseline methods for hallucination mitigation comparison.

Implements:
- DoLa: Decoding by Contrasting Layers (Chuang et al., 2023)
- ITI: Inference-Time Intervention (Li et al., 2023)
- SelfCheckGPT: Self-consistency based hallucination detection (Manakul et al., 2023)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42


class DoLaDecoder:
    """
    DoLa: Decoding by Contrasting Layers.
    Uses the difference in logits between a mature layer and a premature layer
    to suppress hallucinated tokens during decoding.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        premature_layer: int = 16,
        mature_layer: int = -1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.premature_layer = premature_layer
        self.mature_layer = mature_layer

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
    ) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            outputs = self.model(generated_ids, output_hidden_states=True)
            hs = outputs.hidden_states

            mature_hs = hs[self.mature_layer][:, -1, :]
            premature_hs = hs[self.premature_layer][:, -1, :]

            lm_head = self.model.lm_head
            mature_logits = lm_head(mature_hs)
            premature_logits = lm_head(premature_hs)

            mature_probs = F.log_softmax(mature_logits, dim=-1)
            premature_probs = F.log_softmax(premature_logits, dim=-1)
            dola_logits = mature_probs - premature_probs

            if temperature > 0:
                dola_logits = dola_logits / temperature
                probs = F.softmax(dola_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = dola_logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        prompt_len = input_ids.shape[1]
        text = self.tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)

        return {
            "text": text,
            "num_tokens": generated_ids.shape[1] - prompt_len,
            "method": "dola",
        }


class ITIDecoder:
    """
    ITI: Inference-Time Intervention.
    Shifts activations at specific attention heads to steer away from hallucination.
    Uses pre-computed truthful/untruthful directions.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        intervention_layers: Optional[list[int]] = None,
        alpha: float = 15.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.alpha = alpha

        n_layers = model.config.num_hidden_layers
        if intervention_layers is None:
            mid = n_layers // 2
            self.intervention_layers = list(range(mid - 2, mid + 3))
        else:
            self.intervention_layers = intervention_layers

        self.directions = {}
        self._hooks = []

    def compute_directions(
        self,
        truthful_texts: list[str],
        hallucinated_texts: list[str],
    ):
        """Compute truthful directions from paired examples."""
        truthful_acts = self._collect_activations(truthful_texts)
        hallu_acts = self._collect_activations(hallucinated_texts)

        for layer in self.intervention_layers:
            if layer in truthful_acts and layer in hallu_acts:
                t_mean = truthful_acts[layer].mean(dim=0)
                h_mean = hallu_acts[layer].mean(dim=0)
                direction = t_mean - h_mean
                direction = direction / (direction.norm() + 1e-8)
                self.directions[layer] = direction

    @torch.no_grad()
    def _collect_activations(self, texts: list[str]) -> dict[int, torch.Tensor]:
        activations = {}
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=True)
            for layer in self.intervention_layers:
                if layer < len(outputs.hidden_states):
                    act = outputs.hidden_states[layer][:, -1, :].cpu()
                    if layer not in activations:
                        activations[layer] = []
                    activations[layer].append(act)

        return {k: torch.cat(v, dim=0) for k, v in activations.items()}

    def _intervention_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if layer_idx in self.directions:
                direction = self.directions[layer_idx].to(output[0].device)
                output_tensor = output[0]
                proj = (output_tensor * direction).sum(dim=-1, keepdim=True)
                output_tensor = output_tensor + self.alpha * direction.unsqueeze(0).unsqueeze(0)
                return (output_tensor,) + output[1:]
            return output
        return hook_fn

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> dict:
        self._register_hooks()
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            text = self.tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)
        finally:
            self._remove_hooks()

        return {
            "text": text,
            "num_tokens": outputs.shape[1] - prompt_len,
            "method": "iti",
        }

    def _register_hooks(self):
        self._remove_hooks()
        for layer_idx in self.intervention_layers:
            if layer_idx < len(self.model.model.layers):
                handle = self.model.model.layers[layer_idx].register_forward_hook(
                    self._intervention_hook(layer_idx)
                )
                self._hooks.append(handle)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class SelfCheckGPT:
    """
    SelfCheckGPT: Detect hallucinations via self-consistency.
    Generate multiple samples, check if claims are consistent across samples.
    Inconsistent claims are likely hallucinated.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        num_samples: int = 5,
        temperature: float = 0.7,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.temperature = temperature

    @torch.no_grad()
    def generate_and_check(
        self,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        samples = []
        for _ in range(self.num_samples):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            text = self.tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)
            samples.append(text)

        main_response = samples[0]
        other_samples = samples[1:]

        sentences = [s.strip() for s in main_response.split(".") if s.strip()]
        sentence_scores = []

        for sentence in sentences:
            if len(sentence.split()) < 3:
                sentence_scores.append(1.0)
                continue

            sentence_words = set(sentence.lower().split())
            consistencies = []
            for other in other_samples:
                other_words = set(other.lower().split())
                overlap = len(sentence_words & other_words)
                consistency = overlap / max(len(sentence_words), 1)
                consistencies.append(consistency)

            avg_consistency = sum(consistencies) / max(len(consistencies), 1)
            sentence_scores.append(avg_consistency)

        overall_consistency = sum(sentence_scores) / max(len(sentence_scores), 1)

        if overall_consistency < 0.3:
            main_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            main_response = self.tokenizer.decode(
                main_outputs[0, prompt_len:], skip_special_tokens=True
            )

        return {
            "text": main_response,
            "num_tokens": len(self.tokenizer.encode(main_response)),
            "consistency_score": overall_consistency,
            "sentence_scores": sentence_scores,
            "num_samples": self.num_samples,
            "method": "selfcheckgpt",
        }
