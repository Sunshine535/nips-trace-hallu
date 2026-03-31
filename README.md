# PHI: Predictive Hallucination Intervention via Onset Detection and Graduated Response

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
│   ├── collect_traces.py            # Stage 1: Trace collection with claim-level NLI labeling
│   ├── train_onset_detector.py      # Stage 2: Multi-layer probe training
│   ├── train_intervention_policy.py # Stage 3: PPO intervention policy
│   ├── eval_chi.py                  # Stage 4–5: Online evaluation + ablations
│   └── run_trace_generation.sh      # Standalone trace generation
├── src/
│   ├── onset_detector.py            # Linear probe + multi-layer ensemble detector
│   ├── intervention_actions.py      # 5 intervention actions + executor
│   ├── claim_labeler.py             # Claim-level NLI hallucination labeling
│   ├── factuality_eval.py           # Claim-level factuality evaluation + bootstrap
│   ├── baselines.py                 # DoLa, ITI, SelfCheckGPT implementations
│   ├── detector_calibration.py      # AUPRC, ECE, lead-time, trigger curves
│   └── completeness_eval.py         # Completeness, helpfulness, abstention metrics
├── results/                         # Experiment outputs
├── logs/                            # Training logs
└── refine-logs/                     # ARIS research refinement history
```

## Method Overview

PHI operates in three phases:

1. **Predict**: A multi-layer ensemble detector monitors hidden states during generation to predict hallucination onset before it becomes visible in the output.

2. **Intervene**: Upon detection, a learned PPO policy selects from graduated interventions: continue, truncate, backtrack, or restart — balancing factuality improvement against computational cost.

3. **Evaluate**: Online evaluation with claim-level NLI verification, calibration metrics, and budget-matched comparisons against strong baselines.

## Experiments

| # | Stage | Description | Est. Time (8×A100) |
|---|-------|-------------|-------------------|
| 1 | Trace Collection | Collect hidden-state traces with claim-level NLI labels on TruthfulQA, HaluEval, FaithDial | ~48 hrs |
| 2 | Onset Detector | Train per-layer probes + multi-layer ensemble with calibration | ~8 hrs |
| 3 | Intervention Policy | Train PPO policy with 4 graduated actions | ~24 hrs |
| 4 | Online Evaluation | Full evaluation: PHI vs DoLa, ITI, SelfCheckGPT + detector-oracle ablation | ~12 hrs |
| 5 | Ablation Studies | Threshold sweep + per-layer comparison + budget analysis | ~8 hrs |
| 6 | Summary | Aggregate results, bootstrap CIs, significance tests | < 1 hr |

## Evaluation Protocol

- **Factuality**: Claim-level NLI verification (DeBERTa-v3-large-MNLI)
- **Completeness**: Reference claim coverage (prevents winning by omission)
- **Helpfulness**: Combined factuality × completeness score
- **Detector**: AUPRC, ECE, onset lead-time, false positive burden
- **Statistics**: 3 seeds, bootstrap 95% CIs, paired bootstrap significance tests
- **Baselines**: No intervention, Always truncate, Detector-oracle (trained detector, not ground-truth), DoLa, ITI, SelfCheckGPT

## Timeline & GPU Hours

- **Model**: Qwen/Qwen3.5-9B (primary), Llama-3-8B (transfer experiment)
- **Total estimated GPU-hours**: ~800 (8× A100-80GB)
- **Wall-clock time**: ~4–5 days on 8× A100

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
