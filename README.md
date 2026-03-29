# CHI: Causal Hallucination Intervention via RL-Trained Onset Detection

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-trace-hallu.git
cd nips-trace-hallu

# 2. Install dependencies
bash setup.sh

# 3. Run all experiments
bash run.sh

# 4. (Optional) Run in background for long experiments
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done   # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/   # See which individual phases completed
```

### Save and Send Results

```bash
# Option A: Push to GitHub
git add results/ logs/
git commit -m "Experiment results"
git push origin main

# Option B: Package as tarball
bash collect_results.sh
# Output: results_archive/nips-trace-hallu_results_YYYYMMDD_HHMMSS.tar.gz
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
To force re-run all phases: `FORCE_RERUN=1 bash run.sh`

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
