# TRACE-Hallu: Causal Trajectory Auditing and Intervention for Long-CoT Hallucination

## Overview

This project tackles **online hallucination intervention** during LLM generation. Instead of detecting hallucinations after generation is complete (post-hoc), TRACE-Hallu identifies the **onset point** where hallucinations begin in a reasoning chain and applies targeted intervention (retrieve, verify, revise, or restart) at that point.

**Target venue:** NeurIPS 2026

**Status:** ~35% complete (pilot stage)

## Research Questions

1. Does onset-aware intervention reduce claim-level hallucination at matched latency vs post-hoc baselines?
2. Is onset-triggered intervention better than fixed/random triggers?
3. Does the gain persist under domain shift?

## Core Idea

```
Generation Trajectory:  Step1 → Step2 → [Onset!] → Step3(hallu) → Step4(hallu) → ...
                                           │
                                    ┌──────┴──────┐
                                    │  Detector   │
                                    │  (onset     │
                                    │   predictor)│
                                    └──────┬──────┘
                                           │
                              ┌────────────┼────────────┐
                              │            │            │
                           Retrieve     Verify      Revise/Restart
```

## Method

1. **Claim Graph Extraction:** Convert generation trajectory into step-level factual claims
2. **Onset Detection:** Predict hallucination onset from prefix-only features (entropy, confidence, consistency)
3. **Intervention Policy:** Select from {retrieve, verify, revise, restart, continue} based on onset signal
4. **Cost-Aware Optimization:** Maximize claim-level factuality under token/latency penalty

## Current Results (Pilot)

| Metric | Value |
|---|---|
| Detector F1 | 0.644 |
| Policy Accuracy | 0.525 |
| Average Tokens | 43.25 |
| Utility | 0.4997 |

The pilot runs on offline GSM8K-derived traces, not a full claim-level benchmark.

## Repository Structure

```
nips-trace-hallu/
├── README.md              # This file
├── PROPOSAL.md            # Falsifiable thesis and success criteria
├── PLAN.md                # Stage-gate execution plan
├── EXPERIMENTS.md          # Evaluation protocol and results
├── PAPERS.md              # Core references with URLs
├── README_RUN.md          # Runbook
├── environment.yml        # Conda environment spec
├── scripts/
│   └── run_trace_hallu_pilot.py   # Pilot experiment script
└── results/
    ├── trace_hallu_pilot_20260227_150036.json
    └── trace_hallu_policy_20260227_150036.csv
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate nips_trace_hallu
python scripts/run_trace_hallu_pilot.py
```

## Quantitative Success Criteria

- **Primary:** Claim-F1 >= +3.0 absolute over strongest post-hoc baseline at matched cost
- **Secondary:** No more than 10% latency increase at matched quality

## Key References

- LLM-Check (NeurIPS 2024)
- Long-form factuality in LLMs (NeurIPS 2024)
- SLED (NeurIPS 2024)
- Self-RAG (ICLR 2024)
- Auditing Meta-Cognitive Hallucinations (NeurIPS 2025)

See [PAPERS.md](PAPERS.md) for full list with direct URLs.

## Remaining Work

1. Implement real claim-level extraction pipeline
2. Build online intervention loop with LLM generation
3. Evaluate on LongFact/SAFE, HaluEval, multi-hop QA benchmarks
4. Compare against SelfCheckGPT, LLM-Check, SLED, Self-RAG
5. Multi-seed replication with statistical significance

## License

Research code for academic use.
