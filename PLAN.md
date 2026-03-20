# Execution Plan — Causal Hallucination Intervention (CHI)

## Project Timeline: 7 Weeks

---

## Week 1-2: Trace Generation Pipeline
**Goal**: Generate 100K labeled traces from Qwen3.5-27B with per-token hallucination annotations.

### Tasks
- [ ] Set up Qwen3.5-27B on 4× A100 with vLLM for efficient batched generation
- [ ] Implement self-consistency labeling pipeline:
  - Generate N=5 responses per prompt at T=0.7
  - Extract claims/entities from each response (spaCy NER + custom claim extractor)
  - Cross-check consistency: claim in ≥4/5 responses → factual
  - Cross-validate entities against Wikipedia dump (Dec 2025 snapshot)
- [ ] Data sources preparation:
  - TruthfulQA: 817 prompts → expand with paraphrases to ~2K
  - HaluEval: sample 20K from 35K for trace generation
  - FaithDial: sample 15K knowledge-grounded prompts
  - Open-ended Wikipedia: curate 20K questions about entities/events
- [ ] Per-token label propagation: entity-level labels → token-level via alignment
- [ ] Quality control: manually verify 500 random traces, target ≥85% label agreement
- [ ] Output format: JSON lines with {prompt, response, token_labels[], entity_labels[]}

### Trace Generation Details
```python
for prompt in prompts:
    responses = [generate(prompt, T=0.7) for _ in range(5)]
    claims = [extract_claims(r) for r in responses]
    for i, response in enumerate(responses):
        for claim in claims[i]:
            consistency = sum(claim in claims[j] for j in range(5) if j != i)
            label = "factual" if consistency >= 3 else "hallucinated"
        token_labels = propagate_to_tokens(response, claim_labels)
        yield {"prompt": prompt, "response": response, "labels": token_labels}
```

### Validation Criteria
- [ ] ≥100K traces generated with non-trivial hallucination rate (15-40% of responses contain ≥1 hallucination)
- [ ] Label quality: ≥85% agreement with manual annotation on 500-sample subset
- [ ] Balanced across data sources (roughly 25K each)
- [ ] Token-level labels align correctly with entity-level labels

---

## Week 3: Onset Detector Training
**Goal**: Train linear probe on Qwen3.5-9B hidden states for hallucination onset detection.

### Tasks
- [ ] Extract hidden states from Qwen3.5-9B for all 100K traces
  - Run inference on traces, cache hidden states at layers {16, 20, 24, 28, 32}
  - Storage: ~100K traces × 512 avg tokens × 4096 dim × 5 layers × fp16 ≈ 2TB → subsample to 50K traces
- [ ] Train linear probes at each candidate layer
  - Input: h_t ∈ R^4096, Output: P(onset | h_t)
  - Binary CE loss, AdamW lr=1e-3, batch_size=256
  - Train/val/test split: 70/15/15 of trace tokens
- [ ] Layer selection: pick layer with best AUC on validation set
- [ ] Confidence calibration via Platt scaling on validation set
- [ ] Analyze detector errors: false positive rate by token type, position, context length

### Detector Architecture
```python
class OnsetDetector(nn.Module):
    def __init__(self, hidden_dim=4096):
        self.probe = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden_state):
        logit = self.probe(hidden_state)  # [batch, 1]
        return torch.sigmoid(logit)
```

### Expected Results by Layer
| Layer | Expected AUC | Notes |
|-------|-------------|-------|
| 16 | 0.78 | Too early; factuality not yet encoded |
| 20 | 0.84 | Reasonable; some factuality signal |
| 24 | 0.88-0.90 | Sweet spot based on prior work |
| 28 | 0.87 | Slightly worse; post-decision representations |
| 32 | 0.83 | Late layers focus on generation, less factuality |

### Validation Criteria
- [ ] Best layer AUC ≥ 0.87 (achievable based on Hallu Probes precedent)
- [ ] Calibration ECE ≤ 0.05 after Platt scaling
- [ ] False positive rate ≤ 15% at 80% recall operating point
- [ ] Detector inference adds < 5ms per token (linear probe is cheap)

---

## Week 4-5: Intervention Policy RL Training
**Goal**: Train the 5-action intervention policy via PPO.

### Tasks
- [ ] Implement intervention policy transformer (4 layers, 128 dim, 4 heads)
- [ ] Implement all 5 action executors:
  - **Truncate**: End generation at current position, add <EOS>
  - **Backtrack**: Maintain checkpoint buffer every 10 tokens; rewind to nearest checkpoint, regenerate with temperature shift (T→0.3)
  - **Retrieve**: Query FAISS index over Wikipedia; inject top-3 passages into context; continue generation
  - **Restart**: Discard response; regenerate from prompt with lower temperature (T=0.3)
  - **Continue**: No action; proceed with generation as-is
- [ ] Build retrieval index: FAISS index over Wikipedia paragraphs using Contriever embeddings (~5M paragraphs, ~20GB index)
- [ ] Implement reward function:
  ```
  R = 0.5 * factuality_score(final_response, reference)
    + 0.2 * fluency_score(final_response)
    + 0.2 * completeness_score(final_response, prompt)
    - 0.1 * latency_penalty(action)
  ```
  - Factuality: DeBERTa-v3-large NLI model, claim-level entailment score
  - Fluency: 1 - (perplexity / max_perplexity), clamped to [0, 1]
  - Completeness: BERTScore between final response and reference response
  - Latency: action-specific (truncate=0.0, continue=0.0, backtrack=0.3, retrieve=0.5, restart=0.8)
- [ ] PPO training loop:
  - Episodes: 10,000 (expect convergence by ~6,000)
  - Batch size: 64 episodes, mini-batch: 16
  - Policy lr: 3e-5, value lr: 1e-4, clip ε=0.2
  - Entropy bonus coefficient: 0.05 (encourage exploration of all actions)
  - KL penalty: 0.01 (prevent policy collapse)
- [ ] Logging: action distribution per episode, reward components, KL divergence

### RL Episode Structure
```
episode():
    prompt = sample_prompt()
    context = encode_prompt(prompt)
    tokens = []
    actions_taken = []
    
    while not done:
        token = generate_next_token(context + tokens)
        tokens.append(token)
        hidden = get_hidden_state(layer=24)
        conf = detector(hidden)
        
        if conf > threshold:
            action = policy(conf, hidden, context, tokens)
            result = execute_action(action, tokens, prompt)
            tokens = result.new_tokens
            actions_taken.append(action)
            
            if action in [TRUNCATE, RESTART]:
                break  # episode ends differently per action
    
    reward = compute_reward(prompt, tokens, reference)
    return trajectory, reward
```

### Validation Criteria
- [ ] Policy reward increases monotonically (smoothed over 500 episodes)
- [ ] All 5 actions used with >5% frequency (non-degenerate policy)
- [ ] Policy outperforms best single-action baseline by ≥ 5% on reward
- [ ] Action distribution shows interpretable patterns (retrieve for knowledge-gap prompts, etc.)

---

## Week 5-6: Evaluation on Benchmarks
**Goal**: Complete results on TruthfulQA, HaluEval, FaithDial with all baselines.

### Tasks
- [ ] TruthfulQA evaluation:
  - MC accuracy (multiple choice): standard evaluation
  - Open-ended: generate response, score with GPT-4 judge for truthfulness + informativeness
  - Run: no intervention, truncate-only, backtrack-only, retrieve-only, random, Monitoring Decoding, CHI
- [ ] HaluEval evaluation:
  - QA subset (10K): extractive answer correctness
  - Summarization subset (10K): faithfulness to source
  - Dialogue subset (10K): consistency with context
  - Hallucination F1 across all subsets
- [ ] FaithDial evaluation:
  - Critic model score: faithfulness to knowledge snippet
  - BEGIN metric (standard FaithDial metric)
  - Response informativeness (F1 with reference response)
- [ ] Baseline implementations:
  - No intervention: vanilla Qwen3.5-9B generation
  - Single-action baselines: always truncate, always backtrack, always retrieve
  - Random action: uniform random selection from 5 actions when detector fires
  - Monitoring Decoding: 5-beam tree search with verifier (re-implement from paper)
  - RLFH: RL-finetune Qwen3.5-9B on hallucination reward (separate training run)
- [ ] Compute response completeness for all methods (critical metric for truncation comparison)
- [ ] Latency measurement: avg intervention overhead in ms per response

### Evaluation Matrix
```
Methods × Benchmarks × Metrics:
  7 methods × 3 benchmarks × 5 metrics = 105 cells
  + statistical significance (bootstrap 95% CI)
  + ablation variants (see Week 6)
```

---

## Week 6: Ablations + Analysis
**Goal**: Deep analysis of intervention policy behavior.

### Ablation Experiments
- [ ] **Detector quality**: Degrade detector AUC to {0.70, 0.75, 0.80, 0.85, 0.90} by adding noise to probe weights. Measure CHI accuracy degradation curve.
- [ ] **Threshold sweep**: τ ∈ {0.3, 0.5, 0.7, 0.9}. Plot Pareto frontier of accuracy vs. completeness.
- [ ] **Action space reduction**: Train policies with 3 actions {truncate, retrieve, continue} and 4 actions {truncate, backtrack, retrieve, continue}. Measure performance drop from removing actions.
- [ ] **Reward component ablation**: Remove each reward component one at a time. Which component drives which behavior?
- [ ] **Without RL**: Supervised policy trained on expert-annotated actions (50 samples per hallucination type). Compare with RL-trained policy.

### Analysis Experiments
- [ ] **Action distribution by hallucination type**: Cluster hallucinations by type (knowledge gap, conflation, fabrication, premise error) and analyze policy action frequencies per cluster.
- [ ] **Intervention timing analysis**: Compare intervening at first hallucinated token vs. waiting 3, 5, 10 tokens. Plot accuracy vs. timing delay.
- [ ] **Case study gallery**: 20 cherry-picked examples showing each action being used effectively (4 per action). Include in paper appendix.
- [ ] **Failure mode analysis**: Cases where CHI makes wrong intervention choice. Categorize failure modes.
- [ ] **Policy attention visualization**: What does the policy transformer attend to when selecting actions?

### Expected Analysis Findings
- Retrieve action dominates for knowledge-gap hallucinations (60%+ frequency)
- Backtrack dominates for mid-sentence conflation (40%+ frequency)
- Continue action usage inversely correlates with detector confidence
- Performance degrades gracefully with detector quality down to AUC 0.80
- 5-action policy outperforms 3-action by ~4% and 4-action by ~2%

---

## Week 7: Paper Writing + Final Experiments
**Goal**: Complete NeurIPS-quality draft paper.

### Paper Structure
1. **Introduction** (1.5 pages): The intervention gap — detection is mature, intervention is not
2. **Related Work** (1 page): Detection methods, decoding interventions, RL for LLM control
3. **Method** (2.5 pages): Trace generation, onset detector, intervention policy, RL training
4. **Experiments** (3 pages): Benchmarks, baselines, ablations, analysis
5. **Discussion** (0.5 pages): Limitations (latency of retrieve action, detector dependency), future work
6. **Conclusion** (0.5 pages)

### Figures to Generate
- [ ] Architecture diagram (system overview with trace gen → detector → policy → executor)
- [ ] Action distribution heatmap (action × hallucination type)
- [ ] Accuracy vs. completeness Pareto curve (all methods)
- [ ] Detector threshold sensitivity curves
- [ ] Case study examples (truncate vs. backtrack vs. retrieve scenarios)
- [ ] Policy reward training curve

### Tasks
- [ ] Write complete draft
- [ ] Generate all figures and tables
- [ ] Run any gap-filling experiments identified during writing
- [ ] Internal review and revision
- [ ] Supplementary: full results, hyperparameters, additional case studies

---

## Compute Budget Summary

| Phase | GPUs | Hours | GPU-Hours |
|-------|------|-------|-----------|
| Week 1-2: Trace generation | 4× A100 | 72 | 288 |
| Week 3: Hidden state extraction + detector | 2× A100 | 12 | 24 |
| Week 4-5: RL policy training | 8× A100 | 36 | 288 |
| Week 5-6: Evaluation (all baselines) | 4× A100 | 24 | 96 |
| Week 6: Ablations + analysis | 4× A100 | 16 | 64 |
| Week 7: Gap-filling experiments | 2× A100 | 8 | 16 |
| **Total** | | **168** | **776** |

---

## Critical Path

```
Week 1-2 (Traces) → Week 3 (Detector) → Week 4-5 (RL Policy)
                                              ↓
                                     Week 5-6 (Evaluation)
                                              ↓
                                     Week 6 (Ablations)
                                              ↓
                                     Week 7 (Paper)
```

**Bottleneck**: Trace generation (72h). Start immediately. Detector and policy can't begin until traces are ready.

**Fallback plan**: If trace labeling quality is poor (<80% agreement), switch to GPT-4 as labeler for a subset (more expensive but higher quality). If RL doesn't converge, use behavior cloning from expert-annotated actions as a strong baseline and frame RL as an improvement direction.

## Risk Mitigations

1. **Trace quality**: Generate 150K traces, filter to best 100K by label agreement score
2. **RL collapse to single action**: Entropy bonus 0.05; if still collapses, increase to 0.1 and add action-diversity reward term
3. **Retrieve action latency**: Pre-build FAISS index; use approximate search (nprobe=32) for <50ms retrieval
4. **Backtrack complexity**: Fixed 10-token checkpoints simplify implementation; no need for dynamic checkpointing
5. **RLFH baseline is expensive**: Budget 96 GPU-hours for RLFH baseline training; if too slow, use published RLFH numbers
