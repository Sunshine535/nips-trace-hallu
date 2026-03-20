# Research Proposal: Causal Hallucination Intervention (CHI)

## Title
Causal Hallucination Intervention: Beyond Detection to Learned Recovery Actions

## Problem Statement

The LLM hallucination literature has made significant progress on *detection* — we can now identify hallucinated tokens/entities with AUC > 0.90 using hidden-state probes (Hallucination Probes, 2025) and flag them with token-level gates (HaluGate, 2025). However, **detection without intervention is incomplete**: once a hallucination is detected at token t, existing systems either (a) simply flag the output for users, (b) apply a single hardcoded strategy (truncation), or (c) rely on expensive tree-search decoding (Monitoring Decoding, 5-10× cost).

**Core question**: Can we train a lightweight RL policy that selects the optimal intervention action from a discrete action space {truncate, backtrack, retrieve, restart, continue}, conditioned on hallucination type, severity, and context?

## Hypothesis

An RL-trained intervention policy over 5 actions will:
1. **Improve factual accuracy** by 10-15% over detection-only truncation (which sacrifices completeness)
2. **Maintain response completeness** at >90% (vs. ~71% for truncation-only)
3. **Add minimal latency** (<100ms per intervention) compared to tree-search methods (>500ms)
4. **Learn interpretable action preferences** that correlate with hallucination type and severity

## Theoretical Foundation

### Hallucination Taxonomy Driving Action Selection
CHI's action space is motivated by a taxonomy of hallucination causes:

| Hallucination Type | Cause | Optimal Action |
|-------------------|-------|---------------|
| Knowledge gap | Missing facts in parametric memory | **Retrieve** (inject external knowledge) |
| Conflation | Merging attributes of different entities | **Backtrack** (rewind to divergence point) |
| Fabrication | Confident generation of non-existent facts | **Truncate** (stop before damage spreads) |
| Premise error | Wrong assumption in initial reasoning | **Restart** (entire response based on wrong premise) |
| Detector false positive | Factually correct but unusual phrasing | **Continue** (ignore false alarm) |

### Why RL Over Supervised Learning?
The optimal action is not observable in static data — we can't label "the best action would have been to backtrack." Instead, we define reward as post-intervention factuality + fluency + completeness, and let the policy learn the optimal action-context mapping through interaction.

## Method

### Phase 1: Trace Generation with Hallucination Labels (Weeks 1-3)

**Goal**: Generate 100K labeled traces using Qwen3.5-27B with per-token hallucination annotations.

**Labeling procedure (self-consistency)**:
1. For each prompt p, generate N=5 responses {r₁, ..., r₅} at temperature T=0.7
2. For each claim/entity in response rᵢ, check consistency across {r₁, ..., r₅}
3. Claims consistent in ≥4/5 responses → factual; otherwise → hallucinated
4. Cross-validate against Wikipedia/knowledge base for entity-level claims
5. Output: per-token labels (factual/hallucinated/uncertain)

**Data sources**:
- TruthfulQA prompts (817 questions spanning 38 categories)
- HaluEval prompts (35K samples: QA, summarization, dialogue)
- FaithDial prompts (50K knowledge-grounded dialogues)
- Open-ended Wikipedia questions (20K, custom curated)

**Scale**: ~100K traces, ~10M tokens with hallucination labels. Estimated 72h on 4× A100.

### Phase 2: Onset Detector Training (Week 3)

**Architecture**: Linear probe on layer-24 hidden states of Qwen3.5-9B.
- Input: h_t ∈ R^4096 (Qwen3.5-9B hidden state at position t)
- Output: P(hallucination_onset | h_t) ∈ [0, 1]
- Training: binary cross-entropy on labeled traces
- Post-processing: Platt scaling for calibrated confidence scores

**Why layer 24?** Literature shows intermediate layers encode factuality information best (Hallu Probes found layer ~50% to be optimal). We validate by sweeping layers 16, 20, 24, 28, 32.

**Expected performance**: AUC ≥ 0.88, calibration ECE ≤ 0.05 (based on Hallu Probes results).

### Phase 3: Intervention Policy Training via RL (Weeks 4-6)

**Policy architecture**: 4-layer transformer decoder (128 dim, 4 heads)
- Input tokens: [DET_CONF] [HIDDEN_STATE_PROJ] [CONTEXT_EMB] [GEN_SO_FAR_EMB]
  - DET_CONF: scalar detector confidence, projected to 128 dim
  - HIDDEN_STATE_PROJ: linear projection of h_t from 4096 to 128
  - CONTEXT_EMB: mean-pooled prompt hidden states, projected to 128
  - GEN_SO_FAR_EMB: mean-pooled generated-text hidden states, projected to 128
- Output: softmax over 5 actions

**RL setup**:
- Algorithm: PPO (clip ε=0.2, GAE λ=0.95)
- Reward function:
  ```
  R = α·factuality(post_intervention) + β·fluency(post_intervention)
      + γ·completeness(post_intervention) - δ·latency_penalty
  
  α=0.5, β=0.2, γ=0.2, δ=0.1
  ```
- Factuality: NLI model scores claim-level factual accuracy against reference
- Fluency: inverse perplexity from reference LM
- Completeness: fraction of original prompt intent addressed in final response
- Latency: normalized wall-clock overhead of intervention action

**Episode structure**:
1. Sample prompt from training set
2. Begin generation with Qwen3.5-9B
3. At each token, run onset detector
4. When detector fires (confidence > threshold τ): policy selects action
5. Execute action → observe modified output
6. Compute reward on final output
7. Update policy via PPO

**Exploration**: ε-greedy initially (ε=0.3→0.05 over 5000 episodes), then pure policy.

### Phase 4: Evaluation (Weeks 6-7)

**Benchmarks**:
1. **TruthfulQA** (817 questions): MC accuracy + open-ended truthfulness
2. **HaluEval** (35K samples): hallucination detection F1 in QA/summarization/dialogue
3. **FaithDial** (50K dialogues): faithfulness to knowledge source

**Baselines**:
1. No intervention (vanilla generation)
2. Detection + always truncate
3. Detection + always backtrack
4. Detection + always retrieve
5. Detection + random action
6. Monitoring Decoding (tree search, 5 beams)
7. RLFH (RL from hallucination feedback, retrained generation model)

**Metrics**:
- Factual accuracy (task-specific)
- Response completeness (fraction of intent addressed)
- Fluency (perplexity, human eval 1-5)
- Intervention latency overhead
- Action distribution analysis

## Key Experiments

### Experiment 1: Action Distribution by Hallucination Type
Analyze which actions the policy selects for different hallucination types. Hypothesis: retrieve dominates for knowledge gaps; backtrack for conflation; truncate for severe fabrication.

### Experiment 2: Detection Threshold Sensitivity
Vary detector threshold τ ∈ {0.3, 0.5, 0.7, 0.9}. Lower τ → more interventions (higher accuracy, lower completeness). Analyze Pareto frontier of accuracy vs. completeness.

### Experiment 3: Intervention Timing
Compare early intervention (first hallucinated token) vs. delayed intervention (after 3 consecutive hallucination tokens). Hypothesis: delayed is better for reducing false positives.

### Experiment 4: Scaling with Detection Quality
Artificially degrade detector AUC from 0.90 to 0.70. How robust is the policy? The "continue" action should absorb false positives.

### Experiment 5: Transfer Across Domains
Train on TruthfulQA, evaluate on HaluEval and FaithDial. Does the intervention policy generalize across hallucination domains?

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Trace labeling is noisy | High | Self-consistency + knowledge base cross-validation; filter low-agreement traces |
| Intervention actions slow down generation | Medium | Budget: <100ms per intervention; precompute retrieval index |
| RL policy converges to single action | Medium | Entropy bonus in PPO; ensure reward differentiates actions |
| Backtrack implementation is complex | Medium | Fixed checkpoint every 10 tokens; rewind to nearest checkpoint |
| Detector AUC below 0.85 | Low | Layer sweep; use top-2 layers ensemble if needed |

## Compute Budget

| Phase | GPUs | Hours | GPU-Hours |
|-------|------|-------|-----------|
| Trace generation (Qwen3.5-27B) | 4× A100 | 72 | 288 |
| Detector training | 1× A100 | 4 | 4 |
| RL policy training | 8× A100 | 36 | 288 |
| Evaluation + ablations | 2× A100 | 24 | 48 |
| **Total** | | | **628** |

## Success Criteria

1. TruthfulQA accuracy ≥ 0.72 (vs. 0.58 no-intervention, 0.64 truncation-only)
2. Response completeness ≥ 0.90 (vs. 0.71 truncation-only)
3. FaithDial faithfulness ≥ 0.83 (vs. 0.72 baseline)
4. Average intervention latency < 100ms
5. Policy uses ≥ 3 of 5 actions with >5% frequency (non-degenerate)
