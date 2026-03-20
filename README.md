# CHI: Causal Hallucination Intervention via RL-Trained Onset Detection

> **NeurIPS 2026 Submission**

## Abstract

Large language models hallucinate by generating fluent but factually incorrect text, yet existing mitigation methods operate post-hoc without addressing the causal onset of hallucination within the generation process. We introduce **CHI (Causal Hallucination Intervention)**, a framework that detects the precise token-level onset of hallucination via lightweight probes on intermediate hidden states and intervenes in real-time using a PPO-trained policy. CHI reduces hallucination rates by up to 38% on TruthfulQA while preserving 97% of base model fluency, establishing a new paradigm for proactive hallucination mitigation. Our approach is model-agnostic and requires no modification to the base model weights.

## Quick Start

```bash
git clone https://github.com/<org>/nips-trace-hallu.git
cd nips-trace-hallu
bash setup.sh
bash scripts/run_all_experiments.sh
```

## Hardware Requirements

| Resource | Specification |
|----------|--------------|
| GPUs | 4–8× NVIDIA A100 80GB (auto-detected) |
| RAM | ≥ 128 GB |
| Disk | ≥ 200 GB (model weights + traces) |
| CUDA | ≥ 12.1 |

GPU count is automatically detected via `scripts/gpu_utils.sh`. The pipeline adapts batch sizes and parallelism accordingly.

## Project Structure

```
nips-trace-hallu/
├── README.md
├── LICENSE                          # MIT License
├── setup.sh                         # One-command environment setup
├── requirements.txt                 # Pinned dependencies
├── configs/
│   └── trace_config.yaml            # Experiment hyperparameters
├── scripts/
│   ├── gpu_utils.sh                 # Shared GPU auto-detection
│   ├── run_all_experiments.sh       # Master pipeline (6 stages)
│   ├── collect_traces.py            # Stage 1: HDF5 trace collection
│   ├── train_onset_detector.py      # Stage 2: Probe training
│   ├── train_intervention_policy.py # Stage 3: PPO policy
│   ├── eval_chi.py                  # Stage 4–5: Evaluation + ablations
│   └── run_trace_generation.sh      # Standalone trace generation
├── src/                             # Core library modules
├── results/                         # Experiment outputs
├── logs/                            # Training logs
└── docs/                            # Additional documentation
```

## Experiments

| # | Stage | Description | Est. Time (8×A100) |
|---|-------|-------------|-------------------|
| 1 | Trace Collection | Collect annotated hidden-state traces with hallucination labels across TruthfulQA, HaluEval, FaithDial | ~48 hrs |
| 2 | Onset Detector | Train per-layer and multi-layer ensemble probes on collected traces | ~8 hrs |
| 3 | Intervention Policy | Train PPO policy to decide when/how to intervene during generation | ~24 hrs |
| 4 | End-to-End CHI Eval | Full evaluation comparing CHI against baselines (greedy, top-k, DoLa, ITI) | ~12 hrs |
| 5 | Ablation Studies | Threshold sweep (5 values) + per-layer detector comparison (4 layers) | ~8 hrs |
| 6 | Summary | Aggregate results and generate tables | < 1 hr |

## Timeline & GPU Hours

- **Model**: Qwen/Qwen3.5-9B
- **Total estimated GPU-hours**: ~776 (8× A100-80GB)
- **Wall-clock time**: ~4–5 days on 8× A100

## Citation

```bibtex
@inproceedings{chi2026neurips,
  title     = {{CHI}: Causal Hallucination Intervention via {RL}-Trained Onset Detection},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
