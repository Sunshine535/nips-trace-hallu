# PHI: Predictive Hallucination Intervention via Onset Detection and Graduated Response

## Problem Anchor (Immutable)

LLMs generate fluent but factually incorrect text (hallucinations). Current mitigations are:
- **Post-hoc**: detect after full generation, waste compute
- **Binary**: always reject/regenerate, no fine-grained control
- **No early warning**: don't exploit internal model signals before errors surface

**Gap**: No method combines real-time onset detection in hidden states with graduated, cost-aware interventions.

## Method: PHI

### Contribution 1 (Primary): Predictive Onset Detection
- Multi-layer ensemble of linear probes on hidden states at layers [8, 16, 24, 32]
- Trained on claim-level NLI-verified annotations (not word overlap)
- Predicts hallucination onset before errors appear in output text
- Calibrated with AUPRC, ECE, and lead-time metrics

### Contribution 2 (Supporting): Graduated Intervention Framework
- 4 actions: CONTINUE, TRUNCATE, BACKTRACK, RESTART
- PPO-trained policy learns cost-quality tradeoff
- State features: detector confidence, generation length, query complexity, onset position, hallucination density
- Evaluated online with real interventions during generation

### Evaluation Protocol
- **Factuality**: Claim-level NLI verification
- **Completeness**: Reference claim coverage (prevents winning by omission)
- **Helpfulness**: Combined factuality × completeness
- **Detector**: AUPRC, ECE, onset lead-time
- **Statistics**: 3 seeds, bootstrap 95% CIs, paired bootstrap significance
- **Baselines**: No intervention, Always truncate, Oracle, DoLa, ITI, SelfCheckGPT

## Validation Plan
1. Detector AUPRC >= 0.7 with positive lead-time on TruthfulQA + HaluEval
2. PHI claim-F1 >= +3 absolute over best baseline at matched token budget
3. Helpfulness gain (not just factuality by omission)
4. Cross-model transfer (Qwen → Llama)
5. Human audit of 300-500 cases for label validation

## ARIS Review History
- Round 1: 3/10 (RETHINK) — labels invalid, eval invalid, no baselines
- Round 2: 6/10 — claim-level NLI, baselines added, still missing calibration
- Round 3: 7/10 — calibration, completeness, PHI rename. Gap = execution.
