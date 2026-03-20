# CHI: Causal Hallucination Intervention — Beyond Detection to Action

[![NeurIPS 2026 Submission](https://img.shields.io/badge/NeurIPS-2026-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

## Overview

**Causal Hallucination Intervention (CHI)** moves beyond hallucination *detection* to systematically study and train *intervention* strategies. When a hallucination is detected mid-generation, what should the model do? Existing work can detect hallucinations (HaluGate, Probes, Monitoring Decoding) but provides no principled answer to the intervention question. CHI trains an RL-based intervention policy over 5 concrete actions: **truncate**, **backtrack**, **retrieve**, **restart**, **continue** — and learns when each action is optimal.

### Key Insight

Hallucination detection is a solved-ish problem (AUC > 0.90 on entity-level probes). But detection without intervention is like a smoke detector without a fire extinguisher. The real question is: *given that we've detected a hallucination at token t, what is the optimal recovery action?* This depends on hallucination type, severity, position, and downstream task. CHI frames this as an RL problem and trains an intervention policy that maximizes factual accuracy while minimizing disruption to fluency.

## Why CHI?

| Approach | Capability | What's Missing |
|----------|-----------|----------------|
| HaluGate (2025) | Token-level hallucination gating | Only binary gate; no recovery strategy |
| Hallucination Probes (2025) | Entity-level detection, AUC 0.90 | Detection only; no action after detection |
| Monitoring Decoding (2025) | Tree-based search to avoid hallucination | Expensive (5-10× decoding cost); no learned policy |
| RLFH (2025) | RL from hallucination feedback | Trains generation model, not intervention policy |
| **CHI (Ours)** | **Detection + 5-action intervention policy** | **First learned intervention system** |

## Architecture

```
                        ┌─────────────────────────────┐
                        │   Qwen3.5-27B (Teacher)     │
                        │   Generates traces with     │
                        │   hallucination labels       │
                        └──────────┬──────────────────┘
                                   │ labeled traces
                                   ▼
┌──────────────────────────────────────────────────────────┐
│                   Qwen3.5-9B (Student)                    │
│                                                           │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Onset        │    │ Intervention │    │ Generation   │ │
│  │ Detector     │───▶│ Policy       │───▶│ Executor     │ │
│  │ (probe on    │    │ (RL-trained  │    │ (applies     │ │
│  │  hidden      │    │  5 actions)  │    │  chosen      │ │
│  │  states)     │    │              │    │  action)     │ │
│  └─────────────┘    └──────────────┘    └──────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 5 Intervention Actions

| Action | Description | When Optimal |
|--------|-------------|-------------|
| **Truncate** | Stop generation at hallucination point | High confidence hallucination, end of sentence |
| **Backtrack** | Rewind to last factual checkpoint, re-generate | Mid-sentence hallucination with salvageable prefix |
| **Retrieve** | Pause, query retrieval system, inject context | Knowledge gap hallucination (missing facts) |
| **Restart** | Discard entire response, regenerate from scratch | Severe structural hallucination, wrong premise |
| **Continue** | Ignore detection signal (false positive handling) | Low confidence detection, minor factual uncertainty |

### Components

1. **Trace Generator** (Qwen3.5-27B): Generates long-form responses with per-token hallucination labels using self-consistency checking against a reference corpus. Produces training data for detector and policy.

2. **Onset Detector**: Linear probe on Qwen3.5-9B hidden states (layer 24 of 48) trained to predict hallucination onset. Binary classification: "hallucination starts at this token" vs. "factual continuation". Outputs confidence score.

3. **Intervention Policy**: Small transformer (4 layers, 128 dim) that takes:
   - Detector confidence score + hidden state at detection point
   - Generated text so far (encoded)
   - Original prompt/context
   - Outputs: distribution over 5 actions

4. **Execution Module**: Implements each action's mechanics (rewind buffer, retrieval API, restart logic).

## Quick Start

```bash
conda create -n chi python=3.11 && conda activate chi
pip install -r requirements.txt

# Phase 1: Generate labeled traces with Qwen3.5-27B
bash scripts/generate_traces.sh

# Phase 2: Train onset detector on hidden states
bash scripts/train_detector.sh

# Phase 3: RL-train intervention policy
bash scripts/train_policy.sh

# Phase 4: Evaluate on TruthfulQA + HaluEval + FaithDial
bash scripts/eval_all.sh
```

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | 8× A100-80GB |
| Trace generation | 4× A100 for Qwen3.5-27B, ~72h |
| Detector training | 1× A100, ~4h |
| Policy RL training | 8× A100, ~36h |
| Evaluation | 2× A100, ~12h |
| Storage | ~500GB (traces + model checkpoints) |

## Repository Structure

```
nips-trace-hallu/
├── src/
│   ├── trace/
│   │   ├── generator.py            # Qwen3.5-27B trace generation
│   │   ├── labeler.py              # Self-consistency hallucination labeling
│   │   └── reference_corpus.py     # Reference knowledge base interface
│   ├── detector/
│   │   ├── onset_probe.py          # Linear probe on hidden states
│   │   ├── hidden_extractor.py     # Extract hidden states at inference
│   │   └── calibration.py          # Detector confidence calibration
│   ├── policy/
│   │   ├── intervention_net.py     # 5-action policy transformer
│   │   ├── rl_trainer.py           # PPO for intervention policy
│   │   ├── reward.py               # Factuality + fluency reward
│   │   └── action_executor.py      # Truncate/backtrack/retrieve/restart/continue
│   ├── data/
│   │   ├── truthfulqa.py           # TruthfulQA loader
│   │   ├── halueval.py             # HaluEval loader
│   │   └── faithdial.py            # FaithDial loader
│   └── eval/
│       ├── factuality.py           # NLI-based factuality scoring
│       ├── fluency.py              # Perplexity + coherence metrics
│       └── intervention_analysis.py # Action distribution analysis
├── configs/
│   ├── trace_gen_27b.yaml
│   ├── detector_probe.yaml
│   ├── policy_ppo.yaml
│   └── eval_config.yaml
├── scripts/
├── PROPOSAL.md
├── PAPERS.md
├── PLAN.md
└── requirements.txt
```

## Expected Results

| Metric | No Intervention | Detection-only (Truncate) | CHI (Ours) | Target |
|--------|----------------|--------------------------|------------|--------|
| TruthfulQA Accuracy | 0.58 | 0.64 | 0.74+ | 0.75 |
| HaluEval F1 | 0.61 | 0.67 | 0.78+ | 0.80 |
| FaithDial Faithfulness | 0.72 | 0.76 | 0.85+ | 0.85 |
| Response Completeness | 1.00 | 0.71 | 0.91+ | 0.90 |
| Fluency (1-5 scale) | 4.2 | 3.1 | 4.0+ | 4.0 |
| Avg Latency Overhead | 0ms | 15ms | 85ms | <100ms |

## Citation

```bibtex
@inproceedings{chi2026,
  title={Causal Hallucination Intervention: Beyond Detection to Learned Recovery Actions},
  author={Anonymous},
  booktitle={NeurIPS},
  year={2026}
}
```

## License

MIT License
