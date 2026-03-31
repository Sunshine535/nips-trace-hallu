# PHI: Predictive Hallucination Intervention via Onset Detection and Graduated Response

PHI monitors transformer hidden states during generation to **predict hallucination onset** before it becomes visible, then applies a **learned PPO policy** to select graduated interventions (continue, truncate, backtrack, retrieve, or restart). Evaluated end-to-end against DoLa, ITI, and SelfCheckGPT on TruthfulQA, HaluEval, and FaithDial with claim-level NLI factuality verification.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Sunshine535/nips-trace-hallu.git
cd nips-trace-hallu

# 2. Install (creates .venv with uv; falls back to conda)
bash setup.sh

# 3. Run full pipeline (6 stages, see below)
bash run.sh

# 4. (Optional) Run in background
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Progress

```bash
ls results/.phase_markers/          # See completed phases
cat results/.pipeline_done          # Shows PIPELINE_COMPLETE when all done
```

### Resume / Force Re-run

```bash
bash run.sh                         # Auto-skips completed phases
FORCE_RERUN=1 bash run.sh           # Force re-run all phases
```

---

## Requirements

| Component | Requirement |
|-----------|-------------|
| **GPU** | 4–8× NVIDIA A100-80GB (or equivalent; 24GB+ for single-GPU with reduced batch) |
| **VRAM** | ≥24GB per GPU (9B model in bf16 ≈ 18GB) |
| **Disk** | ≥100GB free (model cache + traces + checkpoints) |
| **Python** | 3.10+ |
| **CUDA** | 12.x |
| **OS** | Linux (tested on Ubuntu 22.04) |

### Key Dependencies

Installed automatically by `setup.sh`:

- PyTorch ≥ 2.4 (CUDA 12.8)
- transformers ≥ 4.46, datasets ≥ 2.21, accelerate ≥ 0.34
- h5py, scikit-learn, scipy, peft, wandb, evaluate
- flash-attn (optional, for faster inference)

See [`requirements.txt`](requirements.txt) for full list.

### HuggingFace Models (Auto-downloaded)

| Model | Size | Purpose |
|-------|------|---------|
| `Qwen/Qwen3.5-9B` | ~18GB | Primary generator |
| `microsoft/deberta-v3-large-mnli` | ~1.5GB | Claim-level NLI verification |
| `facebook/bart-large-mnli` | ~1.6GB | Independent NLI judge |

**China / Restricted regions**: Set `export HF_ENDPOINT=https://hf-mirror.com` before running.

---

## Project Structure

```
nips-trace-hallu/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── setup.sh                            # One-command environment setup
├── run.sh                              # One-command experiment launcher
├── requirements.txt                    # Python dependencies (PyTorch installed separately)
├── configs/
│   └── trace_config.yaml               # All hyperparameters and dataset configs
├── scripts/
│   ├── run_all_experiments.sh          # Master pipeline (Stage 1–6)
│   ├── gpu_utils.sh                    # GPU detection, batch size auto-scaling
│   ├── collect_traces.py               # Stage 1: Generate traces + extract hidden states + NLI labels
│   ├── extract_hidden_states.py        # Stage 1 (alt): Extract hidden states from existing JSONL
│   ├── train_onset_detector.py         # Stage 2: Per-layer probes + multi-layer ensemble
│   ├── train_intervention_policy.py    # Stage 3: Offline PPO policy (MLP, 5 actions)
│   ├── train_policy_online.py          # Stage 3b: Online PPO refinement with real rollouts
│   ├── eval_chi.py                     # Stage 4–5: Full evaluation + ablations
│   ├── run_ablations.py                # Generate ablation config matrix
│   ├── analyze_offline_online_correlation.py  # Offline↔online reward correlation
│   └── run_trace_generation.sh         # Standalone trace generation wrapper
├── src/                                # Core library
│   ├── __init__.py                     # Public API exports
│   ├── onset_detector.py               # Linear probe + multi-layer ensemble detector
│   ├── intervention_actions.py         # 5 intervention actions (continue/truncate/backtrack/retrieve/restart)
│   ├── claim_labeler.py                # Claim extraction + NLI hallucination labeling
│   ├── factuality_eval.py              # Multi-judge factuality evaluation + bootstrap CIs
│   ├── baselines.py                    # DoLa, ITI, SelfCheckGPT implementations
│   ├── detector_calibration.py         # AUPRC, ECE, lead-time, trigger curves
│   ├── completeness_eval.py            # Completeness, helpfulness, abstention metrics
│   ├── budget_eval.py                  # Budget-matched Pareto evaluation
│   └── rule_policies.py                # Rule-based policy baselines for ablation
├── tests/
│   └── test_trace_hallu.py             # Unit tests (no GPU needed)
├── results/                            # Experiment outputs (auto-created)
├── logs/                               # Training logs (auto-created)
├── data/                               # Trace data (auto-created)
│   └── traces/                         # HDF5 hidden states + JSONL metadata
└── checkpoints/                        # Model checkpoints (auto-created)
    ├── detector/                       # Onset detector weights
    ├── intervention_policy/            # Offline PPO policy
    └── online_policy/                  # Online-refined policy
```

---

## Method Overview

PHI operates in three phases:

### Phase 1: Predict — Hallucination Onset Detection

A **multi-layer ensemble detector** monitors hidden states at layers {8, 16, 24, 32} during generation. Each layer has a trained linear probe; their predictions are combined via learned softmax weights. Detection fires when the ensemble confidence exceeds a threshold for consecutive tokens.

### Phase 2: Intervene — Graduated Response via Learned Policy

Upon detection, a **PPO-trained MLP policy** selects from 5 graduated actions based on the current state (detector confidence, generation length, query complexity, onset position, hallucination density):

| Action | Cost | Description |
|--------|------|-------------|
| **Continue** | 0.00 | No intervention needed |
| **Truncate** | 0.05 | Stop at current position |
| **Backtrack** | 0.30 | Rewind N tokens and regenerate |
| **Retrieve** | 0.50 | Insert factual context and continue |
| **Restart** | 0.80 | Regenerate from scratch |

The policy is first trained offline on collected traces (Stage 3), then optionally refined online with real LLM rollouts (Stage 3b).

### Phase 3: Evaluate — Multi-Judge Factuality Verification

Factuality is measured via **4 independent judges** to avoid single-model dependence:
1. DeBERTa-v3-large NLI (primary)
2. BART-large NLI (independent architecture)
3. Lexical content-word overlap (no neural model)
4. Grounded answer matching (for datasets with gold answers)

Final score = weighted majority across judges.

---

## Pipeline Stages

The full pipeline (`run_all_experiments.sh`) runs 6 stages sequentially. Each stage is idempotent — completed stages are automatically skipped on re-run.

| Stage | Script | Description | Est. Time (8×A100) | Outputs |
|-------|--------|-------------|---------------------|---------|
| **1** | `collect_traces.py` | Generate CoT traces + extract hidden states (layers 8/16/24/32) + claim-level NLI labels | ~48 hrs | `data/traces/{hidden_states,traces}_*.{h5,jsonl}` |
| **2** | `train_onset_detector.py` | Train per-layer probes + multi-layer ensemble detector | ~8 hrs | `checkpoints/detector/{probe_layer*.pt, multi_layer_detector.pt}` |
| **3** | `train_intervention_policy.py` | Train offline PPO policy (MLP, 5D state → 5 actions) | ~24 hrs | `checkpoints/intervention_policy/{best_policy.pt, training_log.json}` |
| **3b** | `train_policy_online.py` | Online PPO fine-tuning with real rollouts | ~12 hrs | `checkpoints/online_policy/{best_online_policy.pt}` |
| **4** | `eval_chi.py` | Full evaluation: PHI vs 6 baselines × 3 datasets | ~12 hrs | `results/chi_evaluation.json` |
| **5** | `eval_chi.py` (ablations) | Threshold sweep (0.3–0.7) + per-layer detector comparison | ~8 hrs | `results/ablation_*.json` |
| **6** | Inline summary | Aggregate results, print summary table | < 1 min | Console output |

**Total: ~800 GPU-hours (8× A100-80GB), ~4–5 days wall clock.**

---

## Evaluation Protocol

### Metrics

- **Factuality**: Multi-judge claim-level verification (see Phase 3 above)
- **Completeness**: Proportion of reference claims covered (NLI-based)
- **Helpfulness**: α × factuality + (1−α) × completeness (prevents winning by omission)
- **Detector**: AUPRC, ECE, onset lead-time, false positive rate at 90% recall
- **Budget**: Factuality vs token budget Pareto analysis

### Statistical Rigor

- 3 seeds (42, 137, 2024)
- Bootstrap 95% confidence intervals (1000 samples)
- Paired bootstrap significance tests (PHI vs each baseline)

### Baselines

| Baseline | Description |
|----------|-------------|
| No intervention | Standard greedy decoding |
| Always truncate | Truncate at detector-flagged onset |
| Detector oracle | Trained detector + fixed backtrack (NOT ground-truth) |
| DoLa | Decoding by Contrasting Layers (Chuang et al., 2023) |
| ITI | Inference-Time Intervention (Li et al., 2023) |
| SelfCheckGPT | Self-consistency detection (Manakul et al., 2023) |
| Rule cascade | Confidence-escalated: low→truncate, mid→backtrack, high→restart |

---

## Configuration

All hyperparameters are in [`configs/trace_config.yaml`](configs/trace_config.yaml). Key sections:

```yaml
generator:
  model: "Qwen/Qwen3.5-9B"
  max_length: 2048
  temperature: 0.7

detector:
  hidden_size: 4096
  num_layers_to_probe: [8, 16, 24, 32]
  threshold: 0.5

intervention:
  num_actions: 5
  actions: ["continue", "truncate", "backtrack", "retrieve", "restart"]
  ppo_config:
    num_epochs: 100
    learning_rate: 3.0e-4

datasets:
  truthfulqa: { name: "truthful_qa", config: "generation", split: "validation" }
  halueval:   { name: "pminervini/HaluEval", config: "qa_samples", split: "data" }
  faithdial:  { name: "McGill-NLP/FaithDial", split: "test" }
```

---

## Running Individual Stages

If you want to run stages separately (e.g., on different machines):

```bash
source .venv/bin/activate   # or: conda activate nips-trace-hallu

# Stage 1: Collect traces
python scripts/collect_traces.py \
    --model_name Qwen/Qwen3.5-9B \
    --datasets truthfulqa halueval faithdial \
    --output_dir data/traces \
    --layer_indices 8 16 24 32 \
    --batch_size 4

# Stage 2: Train detector
python scripts/train_onset_detector.py \
    --traces_dir data/traces \
    --output_dir checkpoints/detector \
    --layer_indices 8 16 24 32 \
    --hidden_size 4096

# Stage 3: Train policy (offline)
python scripts/train_intervention_policy.py \
    --traces_dir data/traces \
    --output_dir checkpoints/intervention_policy

# Stage 4: Evaluate
python scripts/eval_chi.py \
    --model_name Qwen/Qwen3.5-9B \
    --detector_path checkpoints/detector/multi_layer_detector.pt \
    --policy_path checkpoints/intervention_policy/best_policy.pt \
    --datasets truthfulqa halueval faithdial \
    --output_dir results
```

---

## Tests

```bash
# Run unit tests (no GPU required)
pytest tests/ -v
```

Tests cover:
- Hidden-state HDF5 shape consistency across layers
- Label–token alignment assertions
- Group-aware train/val split (no question leakage)

---

## Collecting and Sharing Results

```bash
bash collect_results.sh
# → results_archive/nips-trace-hallu_results_YYYYMMDD_HHMMSS.tar.gz
# Includes: results/, logs/, checkpoints/ metadata, MANIFEST.json
```

---

## Citation

```bibtex
@inproceedings{phi2026neurips,
  title     = {{PHI}: Predictive Hallucination Intervention via Onset Detection and Graduated Response},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
