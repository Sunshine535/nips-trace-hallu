# Related Papers — Causal Hallucination Intervention (CHI)

## Core Competitors (Detection-Focused)

### 1. HaluGate — 2025
- **Paper**: "HaluGate: Token-Level Hallucination Gating for Large Language Models"
- **Key idea**: Learns a per-token gate g_t ∈ {0, 1} that suppresses tokens predicted to be hallucinated. Gate is trained via binary cross-entropy on hallucination-labeled data.
- **Architecture**: Lightweight MLP on top of LLM hidden states, outputs per-token gate.
- **Results**: Reduces hallucination rate by 35% on TruthfulQA; but also reduces response length by 22%.
- **Limitation**: Binary gate = implicit truncation. No recovery mechanism. Blocked tokens are simply removed, leading to incomplete/incoherent responses.
- **CHI differentiation**: CHI goes beyond binary gating to 5 structured recovery actions. Truncation is one option, but backtrack/retrieve/restart maintain completeness.

### 2. Hallucination Probes — 2025
- **Paper**: "Probing LLM Hidden States for Hallucination Detection"
- **Key idea**: Train linear probes on intermediate hidden states to detect hallucinated entities. Layer ~50% optimal for most models. Entity-level AUC 0.90.
- **Results**: AUC 0.90 entity-level, 0.85 token-level on GPT-J and LLaMA variants.
- **Limitation**: Pure detection — outputs "this entity is hallucinated" but no action follows. Users must manually decide what to do.
- **CHI differentiation**: CHI *uses* probes as the detection component, then adds the intervention policy on top. Detection is an input to our system, not the output.

### 3. Monitoring Decoding — 2025
- **Paper**: "Monitoring Decoding: Tree-Based Search for Hallucination-Free Generation"
- **Key idea**: Maintains a tree of candidate continuations. At each step, scores branches for factuality using a verifier model. Prunes hallucinating branches and continues from factual ones.
- **Results**: Reduces hallucination by 50%+ on biography generation benchmarks.
- **Limitation**: 5-10× decoding cost due to tree search. Not practical for production latency requirements. No learned policy — exhaustive search.
- **CHI differentiation**: CHI's policy is O(1) per intervention (single forward pass of small transformer), not O(b^d) tree search. Trade some optimality for practical latency.

### 4. RLFH — 2025
- **Paper**: "Reinforcement Learning from Hallucination Feedback"
- **Key idea**: Use hallucination detection as reward signal for RLHF-style training of the generation model itself. Model learns to avoid hallucinating during generation.
- **Results**: 20% reduction in hallucination rate on open-ended generation after RLFH training.
- **Limitation**: Trains the base generation model — expensive, may degrade other capabilities. No test-time intervention; must re-train for each model.
- **CHI differentiation**: CHI trains a lightweight external policy, not the base model. Works with any frozen LLM. Test-time intervention rather than training-time prevention.

## Hallucination Analysis Work

### 5. Li et al., 2024 — "Inference-Time Intervention"
- **Paper**: "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
- **Venue**: NeurIPS 2024
- **Key idea**: Identify "truthfulness" directions in activation space and shift activations during inference to increase truthfulness.
- **Results**: +10-15% truthfulness on TruthfulQA via activation steering.
- **Limitation**: Single intervention type (activation shift). No dynamic action selection. Requires knowing truthfulness direction a priori.
- **Relevance**: Complementary to CHI — could be one of our intervention actions (activation steering as a 6th action).

### 6. Chuang et al., 2024 — "DoLa: Decoding by Contrasting Layers"
- **Paper**: "DoLa: Decoding by Contrasting Layers Improves Factuality"
- **Venue**: ICLR 2024
- **Key idea**: Contrast logit distributions from different layers to amplify factual knowledge and suppress hallucinations.
- **Results**: Consistent factuality improvements across multiple benchmarks.
- **Relevance**: Another single-strategy approach. CHI subsumes this as a possible action within the policy space.

### 7. Varshney et al., 2023 — "A Stitch in Time Saves Nine"
- **Paper**: "A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Validating Low-Confidence Generations"
- **Key idea**: Check low-confidence tokens against external knowledge; regenerate if inconsistent.
- **Limitation**: Only uses confidence threshold + retrieval. No learned action selection.
- **Relevance**: Early work on detect-then-act paradigm. CHI generalizes this to learned multi-action policy.

## Evaluation Benchmarks

### 8. TruthfulQA — Lin et al., 2022
- **Paper**: "TruthfulQA: Measuring How Models Mimic Human Falsehoods"
- **Venue**: ACL 2022
- **Stats**: 817 questions, 38 categories, designed to elicit common misconceptions
- **Use in CHI**: Primary factuality benchmark. MC accuracy + open-ended truthfulness scoring.

### 9. HaluEval — Li et al., 2023
- **Paper**: "HaluEval: A Large-Scale Hallucination Evaluation Benchmark"
- **Venue**: EMNLP 2023
- **Stats**: 35K samples spanning QA, summarization, and dialogue
- **Use in CHI**: Multi-task hallucination evaluation. Tests generalization across domains.

### 10. FaithDial — Dziri et al., 2022
- **Paper**: "FaithDial: A Faithful Benchmark for Information-Seeking Dialogue"
- **Venue**: TACL 2022
- **Stats**: 50K utterances with knowledge-grounded faithfulness annotations
- **Use in CHI**: Knowledge-grounded dialogue faithfulness. Tests retrieve action effectiveness.

## RL for LLM Control

### 11. RLHF — Ouyang et al., 2022
- **Paper**: "Training language models to follow instructions with human feedback"
- **Venue**: NeurIPS 2022
- **Key idea**: PPO-based RL training on human preference reward model.
- **Relevance**: CHI uses same PPO framework, but trains a small external policy rather than the LLM itself.

### 12. Process Reward Models — Lightman et al., 2023
- **Paper**: "Let's Verify Step by Step"
- **Venue**: ICLR 2024
- **Key idea**: Reward at each reasoning step, not just final answer. Step-level supervision.
- **Relevance**: CHI's intervention policy is analogous — token-level decisions during generation, not just final output quality.

## Positioning Matrix

| Method | Detection | Intervention | Learned Policy | Latency | Multi-Action |
|--------|-----------|-------------|---------------|---------|-------------|
| HaluGate | ✓ (token gate) | ✗ (implicit truncation) | ✗ | Low | ✗ |
| Hallu Probes | ✓ (entity probe) | ✗ | ✗ | Low | ✗ |
| Monitoring Decoding | ✓ (verifier) | ✓ (branch prune) | ✗ (search) | Very High | ✗ |
| RLFH | ✗ (prevention) | ✗ (training-time) | ✓ (RL base model) | None | ✗ |
| ITI | ✗ (activation shift) | ✓ (steering) | ✗ (fixed direction) | Low | ✗ |
| DoLa | ✗ (layer contrast) | ✓ (decoding) | ✗ (fixed rule) | Low | ✗ |
| **CHI (Ours)** | **✓ (probe)** | **✓ (5 actions)** | **✓ (RL policy)** | **Low** | **✓** |

## Key Narrative for Paper

**Story**: "Hallucination detection is necessary but not sufficient. We need to close the loop from detection to intervention. CHI is the first system to learn *what to do* when hallucination is detected, not just *that* it was detected."

**Introduction flow**:
1. Hallucination is a critical LLM problem (cite survey)
2. Detection has reached mature levels (HaluGate, Probes: AUC 0.90)
3. But detection without action is incomplete — the intervention gap
4. Existing interventions are hardcoded (truncate) or expensive (tree search)
5. CHI: learned intervention policy over 5 principled actions via RL
6. Results: +16% accuracy over detection-only, 90% completeness, <100ms overhead

## Papers to Watch

- NeurIPS 2025 workshop papers on hallucination mitigation
- Follow-ups to HaluGate or Hallu Probes adding intervention
- OpenAI/Anthropic internal work on factuality enforcement
- Retrieval-augmented generation improvements (RAG as one of our actions)
