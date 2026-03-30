# PHI Experiment Plan

## Claims to Validate

### Claim 1 (Primary): Predictive Onset Detection
PHI's multi-layer ensemble detector identifies hallucination onset in hidden states before it appears in output text, with AUPRC >= 0.7 and positive lead-time.

### Claim 2 (Supporting): Graduated Intervention Improves Factuality
PHI's learned intervention policy achieves higher claim-F1 than fixed strategies and strong baselines at matched token budgets.

### Claim 3 (Auxiliary): Cross-Model Generalization
Onset detection patterns transfer across model architectures (Qwen → Llama).

## Experiment Matrix

| Exp | Claim | Stage | GPU-hrs | Priority |
|-----|-------|-------|---------|----------|
| E1  | C1    | Trace Collection (3 datasets × 9B) | ~48 | P0 |
| E2  | C1    | Detector Training (4 layers + ensemble) | ~8 | P0 |
| E3  | C2    | PPO Policy Training | ~24 | P0 |
| E4  | C1+C2 | Full Online Evaluation (3 seeds) | ~36 | P0 |
| E5  | C2    | Ablation: threshold sweep | ~8 | P1 |
| E6  | C2    | Ablation: per-layer detector | ~8 | P1 |
| E7  | C2    | Ablation: action cost sensitivity | ~4 | P1 |
| E8  | C3    | Cross-model transfer (Llama-3-8B) | ~16 | P2 |
| E9  | C1    | Human audit (300-500 cases) | ~2 | P2 |

**Total: ~154 GPU-hours (P0+P1+P2)**

## Run Order

### Phase 1: Data Collection (E1)
```bash
bash run.sh  # Stage 1 only (set FORCE_RERUN=0)
```
Expected output: `data/traces/` with HDF5 + JSONL per dataset

### Phase 2: Training (E2 + E3)
```bash
bash run.sh  # Stages 2-3
```
Expected output: `checkpoints/detector/`, `checkpoints/intervention_policy/`

### Phase 3: Evaluation (E4 + E5 + E6 + E7)
```bash
python scripts/eval_chi.py \
    --detector_path checkpoints/detector/multi_layer_detector.pt \
    --policy_path checkpoints/intervention_policy/best_policy.pt \
    --seeds 42 137 2024 \
    --baselines no_intervention always_truncate oracle_detector dola selfcheckgpt \
    --output_dir results/full_eval
```

### Phase 4: Transfer (E8)
```bash
python scripts/eval_chi.py \
    --model_name meta-llama/Llama-3-8B \
    --detector_path checkpoints/detector/multi_layer_detector.pt \
    --policy_path checkpoints/intervention_policy/best_policy.pt \
    --datasets truthfulqa halueval \
    --output_dir results/transfer_llama
```

## Kill Conditions
- Detector AUPRC < 0.5 after E2 → rethink detector architecture
- No significant improvement over DoLa after E4 → investigate why
- Cross-model transfer completely fails → remove Claim 3

## Multi-GPU Setup
- Trace collection: `accelerate launch --num_processes N`
- Detector training: single GPU (fast)
- Policy training: single GPU (fast)
- Evaluation: `CUDA_VISIBLE_DEVICES=0,1,...` for parallel ablations

## Checkpointing
- All training scripts support `--resume_from_checkpoint auto`
- Phase markers in `results/.phase_markers/`
- Re-run `bash run.sh` to resume from last completed phase
