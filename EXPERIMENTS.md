# Experiments: PHI (Predictive Hallucination Intervention)

## Benchmarks
- TruthfulQA (generation split, 817 samples)
- HaluEval (qa_samples, 10K samples)
- FaithDial (test split, 3.6K samples)

## Baselines
- No mitigation (greedy decoding)
- Always truncate at onset
- Oracle detector + backtrack
- DoLa (Decoding by Contrasting Layers)
- ITI (Inference-Time Intervention)
- SelfCheckGPT (self-consistency detection)

## Metrics

### Factuality
- Claim-level NLI verification (DeBERTa-v3-large-MNLI)
- Claim precision, recall, F1

### Completeness
- Reference claim coverage (prevents winning by omission)
- Helpfulness = α×factuality + (1-α)×completeness
- Abstention rate

### Detector
- AUPRC (Area Under Precision-Recall Curve)
- ECE (Expected Calibration Error)
- Onset lead-time (tokens before hallucination)
- False positive burden at 90% recall
- Trigger curves across thresholds

### Efficiency
- Latency overhead (seconds)
- Token budget (generated tokens)
- Factuality-per-token efficiency

## Statistical Protocol
- 3 replications (seeds: 42, 137, 2024)
- Bootstrap 95% CIs (1000 samples)
- Paired bootstrap significance tests (10000 samples)
- Report effect sizes

## NeurIPS Minimum Publishable Standard
- Claim-F1 gain >= +3 absolute at matched token budget
- Significant gain on at least 2 benchmark families (p < 0.05)
- Helpfulness gain (not just factuality by omission)
- Detector AUPRC >= 0.7 with ECE < 0.1

## Current Status
- Code fully implemented with claim-level evaluation
- Pending: full online evaluation run
- Pending: cross-model transfer experiment (Qwen → Llama)
- Pending: human audit of 300-500 cases
