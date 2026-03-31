# PHI: Experiment Runbook

## Prerequisites

```bash
bash setup.sh                          # creates .venv, installs PyTorch + deps
source .venv/bin/activate
```

## One-Command Full Pipeline

```bash
bash run.sh                            # wrapper for scripts/run_all_experiments.sh
# or run in background:
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

## Pipeline Stages

| Stage | Script | Output |
|-------|--------|--------|
| 1 — Trace Collection | `scripts/collect_traces.py` | `data/traces/traces_*.jsonl`, `data/traces/hidden_states_*.h5` |
| 2 — Onset Detector | `scripts/train_onset_detector.py` | `checkpoints/detector/multi_layer_detector.pt` |
| 3 — PPO Policy (offline) | `scripts/train_intervention_policy.py` | `checkpoints/intervention_policy/best_policy.pt` |
| 3b — Online Policy Refinement | `scripts/train_policy_online.py` | `checkpoints/online_policy/best_policy.pt` |
| 4 — Full Evaluation | `scripts/eval_chi.py` | `results/chi_evaluation.json` |
| 5 — Ablation Studies | `scripts/eval_chi.py` (threshold/layer sweeps) | `results/ablation_*.json` |
| 6 — Summary | inline aggregation | stdout summary |

## Checking Progress

```bash
ls results/.phase_markers/             # which stages completed
cat results/.pipeline_done             # PIPELINE_COMPLETE when all done
cat results/chi_evaluation.json | python -m json.tool | head -80
```

## Resume / Force Rerun

```bash
bash run.sh                            # auto-skips completed stages
FORCE_RERUN=1 bash run.sh              # rerun everything from scratch
```

## Config

All hyperparameters live in `configs/trace_config.yaml`.

## Baselines Evaluated (Stage 4)

- `no_intervention` — vanilla greedy decode
- `always_truncate` — truncate at detector onset
- `detector_oracle` — always backtrack at detector onset (NOT ground-truth; uses trained detector)
- `dola` — Decoding by Contrasting Layers
- `iti` — Inference-Time Intervention
- `selfcheckgpt` — SelfCheckGPT self-consistency
- `chi_ours` — PHI (learned PPO policy)
