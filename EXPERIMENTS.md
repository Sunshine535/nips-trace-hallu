# Experiments: TRACE-Hallu (Revised v2)

## Benchmarks
- HaluEval.
- Long-form factual benchmark (LongFact/SAFE-style).
- HotpotQA and MuSiQue.

## Baselines
- No mitigation.
- Post-hoc detector + regenerate.
- SelfCheckGPT-style detector.
- Self-RAG style retrieve/critique.

## Metrics
- Claim precision/recall/F1.
- QA EM/F1.
- Latency, extra tokens, retrieval calls.
- Cost-normalized factuality score.

## Statistical Protocol
- Minimum 3 replications (seed or disjoint subset).
- Paired bootstrap on claim-F1 and EM/F1 deltas.
- Report 95% CI and effect sizes.

## NeurIPS Minimum Publishable Standard
- Claim-F1 gain `>= +3` absolute at matched cost.
- Significant gain on at least 2 benchmark families.
- Full per-step trace release for auditability.

## Current Status
- Pilot implementation and first result are now available.

## Implemented Pilot (2026-02-27)
- Script:
  - `methods/02_trace_hallu/scripts/run_trace_hallu_pilot.py`
- Command:
  ```bash
  python methods/02_trace_hallu/scripts/run_trace_hallu_pilot.py
  ```
- Input:
  - `methods/01_adathink/results/per_sample_Qwen3_8B_20260227_140410.csv`
- Output:
  - `methods/02_trace_hallu/results/trace_hallu_pilot_20260227_150036.json`
  - `methods/02_trace_hallu/results/trace_hallu_policy_20260227_150036.csv`

## Pilot Snapshot
- Detector test:
  - precision: `0.475`
  - recall: `1.000`
  - F1: `0.644`
- Policy test (`risk->256 else 64`):
  - accuracy: `0.525`
  - avg tokens: `43.25`
  - utility (`lambda_cost=0.15`): `0.4997`
- Baselines:
  - fixed64: acc `0.35`, tokens `21.25`, utility `0.3375`
  - fixed128: acc `0.40`, tokens `31.875`, utility `0.3813`
  - fixed256: acc `0.525`, tokens `43.25`, utility `0.4997`

## Limitation
- This is an offline proxy over GSM8K traces, not yet a true online claim-level hallucination intervention benchmark.
