"""
Implementation of 5 intervention actions for hallucination mitigation.

Actions:
  0 - continue:   Do nothing, let generation proceed
  1 - truncate:   Stop generation at current position
  2 - backtrack:   Rewind to N tokens before onset and re-generate
  3 - retrieve:   Insert a retrieval-augmented context and continue
  4 - restart:    Discard trace entirely and regenerate from scratch
"""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class Action(IntEnum):
    CONTINUE = 0
    TRUNCATE = 1
    BACKTRACK = 2
    RETRIEVE = 3
    RESTART = 4


ACTION_NAMES = {
    Action.CONTINUE: "continue",
    Action.TRUNCATE: "truncate",
    Action.BACKTRACK: "backtrack",
    Action.RETRIEVE: "retrieve",
    Action.RESTART: "restart",
}


@dataclass
class InterventionConfig:
    backtrack_window: int = 3
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    max_interventions: int = 5
    retrieval_prefix: str = "[Retrieved context]: "


class InterventionExecutor:
    """Executes intervention actions on an ongoing generation."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: InterventionConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def execute(
        self,
        action: Action,
        input_ids: torch.Tensor,
        onset_position: int,
        original_prompt_ids: Optional[torch.Tensor] = None,
        retrieval_context: Optional[str] = None,
    ) -> dict:
        """
        Execute an intervention action.

        Args:
            action: Which action to take
            input_ids: Current token sequence (1, seq_len)
            onset_position: Token position where hallucination was detected
            original_prompt_ids: The original prompt (for restart)
            retrieval_context: External knowledge for retrieve action

        Returns:
            dict with 'new_ids', 'action_taken', 'tokens_removed', 'tokens_added'
        """
        if action == Action.CONTINUE:
            return self._do_continue(input_ids)
        elif action == Action.TRUNCATE:
            return self._do_truncate(input_ids, onset_position)
        elif action == Action.BACKTRACK:
            return self._do_backtrack(input_ids, onset_position)
        elif action == Action.RETRIEVE:
            return self._do_retrieve(input_ids, onset_position, retrieval_context)
        elif action == Action.RESTART:
            return self._do_restart(
                original_prompt_ids if original_prompt_ids is not None else input_ids
            )
        else:
            raise ValueError(f"Unknown action: {action}")

    def _do_continue(self, input_ids: torch.Tensor) -> dict:
        new_ids = self._generate(input_ids, max_new=self.config.max_new_tokens)
        return {
            "new_ids": new_ids,
            "action_taken": "continue",
            "tokens_removed": 0,
            "tokens_added": new_ids.shape[1] - input_ids.shape[1],
        }

    def _do_truncate(self, input_ids: torch.Tensor, onset_position: int) -> dict:
        truncated = input_ids[:, :onset_position]
        return {
            "new_ids": truncated,
            "action_taken": "truncate",
            "tokens_removed": input_ids.shape[1] - onset_position,
            "tokens_added": 0,
        }

    def _do_backtrack(self, input_ids: torch.Tensor, onset_position: int) -> dict:
        backtrack_pos = max(1, onset_position - self.config.backtrack_window)
        rewound = input_ids[:, :backtrack_pos]
        new_ids = self._generate(rewound, max_new=self.config.max_new_tokens)
        return {
            "new_ids": new_ids,
            "action_taken": "backtrack",
            "tokens_removed": input_ids.shape[1] - backtrack_pos,
            "tokens_added": new_ids.shape[1] - backtrack_pos,
        }

    def _do_retrieve(
        self,
        input_ids: torch.Tensor,
        onset_position: int,
        retrieval_context: Optional[str],
    ) -> dict:
        prefix = input_ids[:, :onset_position]

        if retrieval_context:
            context_text = f"\n{self.config.retrieval_prefix}{retrieval_context}\n"
        else:
            context_text = f"\n{self.config.retrieval_prefix}[No additional context available]\n"

        context_ids = self.tokenizer.encode(context_text, return_tensors="pt",
                                            add_special_tokens=False).to(prefix.device)
        augmented = torch.cat([prefix, context_ids], dim=1)
        new_ids = self._generate(augmented, max_new=self.config.max_new_tokens)
        return {
            "new_ids": new_ids,
            "action_taken": "retrieve",
            "tokens_removed": input_ids.shape[1] - onset_position,
            "tokens_added": new_ids.shape[1] - onset_position,
        }

    def _do_restart(self, original_prompt_ids: torch.Tensor) -> dict:
        new_ids = self._generate(original_prompt_ids, max_new=self.config.max_new_tokens)
        return {
            "new_ids": new_ids,
            "action_taken": "restart",
            "tokens_removed": -1,
            "tokens_added": new_ids.shape[1],
        }

    @torch.no_grad()
    def _generate(self, input_ids: torch.Tensor, max_new: int) -> torch.Tensor:
        return self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )


def format_intervention_prompt(
    question: str,
    trace_so_far: str,
    onset_position: int,
    action_descriptions: Optional[dict] = None,
) -> str:
    """Format the state as a prompt for the intervention policy model."""
    if action_descriptions is None:
        action_descriptions = {
            0: "Continue generating (no hallucination detected)",
            1: "Truncate the response at the current position",
            2: "Backtrack a few tokens and regenerate",
            3: "Insert retrieved factual context and continue",
            4: "Restart the entire generation from scratch",
        }

    actions_str = "\n".join(
        f"  Action {k}: {v}" for k, v in sorted(action_descriptions.items())
    )

    return (
        f"You are a hallucination intervention controller. "
        f"A language model is generating a chain-of-thought response. "
        f"A hallucination detector has flagged potential hallucination "
        f"at token position {onset_position}.\n\n"
        f"Question: {question}\n\n"
        f"Trace so far:\n{trace_so_far}\n\n"
        f"Available actions:\n{actions_str}\n\n"
        f"Choose the best action (respond with action number only):"
    )
