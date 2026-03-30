#!/bin/bash
# ============================================================================
# PHI Production Run Script
# Optimized for multi-GPU servers with checkpoint resume support.
# Usage: bash scripts/run_production.sh [--stage N] [--gpus 0,1,2,3]
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Parse args ---
STAGE_FILTER=""
CUSTOM_GPUS=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage) STAGE_FILTER="$2"; shift 2;;
        --gpus) CUSTOM_GPUS="$2"; shift 2;;
        --dry-run) DRY_RUN=1; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ -n "$CUSTOM_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$CUSTOM_GPUS"
fi

# --- Source GPU utilities ---
source "${SCRIPT_DIR}/gpu_utils.sh"
auto_setup

# --- Activate venv if present ---
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# --- Config ---
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-phi-hallu}"

MODEL_NAME="Qwen/Qwen3.5-9B"
LAYER_INDICES="8 16 24 32"
DATASETS="truthfulqa halueval faithdial"

TRACES_DIR="${PROJECT_DIR}/data/traces"
DETECTOR_DIR="${PROJECT_DIR}/checkpoints/detector"
POLICY_DIR="${PROJECT_DIR}/checkpoints/intervention_policy"
RESULTS_DIR="${PROJECT_DIR}/results"
LOG_DIR="${PROJECT_DIR}/logs"
PHASE_MARKER_DIR="${RESULTS_DIR}/.phase_markers"

mkdir -p "$TRACES_DIR" "$DETECTOR_DIR" "$POLICY_DIR" "$RESULTS_DIR" "$LOG_DIR" "$PHASE_MARKER_DIR"

FORCE_RERUN="${FORCE_RERUN:-0}"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $1" | tee -a "${LOG_DIR}/production.log"; }
phase_done() { touch "$PHASE_MARKER_DIR/phase_${1}.done"; log "[PHASE $1] Completed"; }
is_phase_done() {
    [[ "$FORCE_RERUN" == "1" ]] && return 1
    [[ -f "$PHASE_MARKER_DIR/phase_${1}.done" ]] && log "[PHASE $1] Already done, skipping" && return 0
    return 1
}
should_run_stage() {
    [ -z "$STAGE_FILTER" ] && return 0
    [ "$STAGE_FILTER" = "$1" ] && return 0
    return 1
}

log "========================================"
log " PHI Production Pipeline"
log " Model: ${MODEL_NAME}"
log " GPUs:  ${NUM_GPUS} × ${GPU_CLASS}"
log " CUDA:  ${CUDA_VISIBLE_DEVICES}"
log "========================================"

if [ $DRY_RUN -eq 1 ]; then
    log "[DRY RUN] Would run stages with above config"
    exit 0
fi

# === Stage 1: Trace Collection ===
if should_run_stage 1 && ! is_phase_done 1; then
    log "[Stage 1/6] Collecting traces with claim-level NLI labeling"
    python3 "${SCRIPT_DIR}/collect_traces.py" \
        --model_name "$MODEL_NAME" \
        --datasets $DATASETS \
        --output_dir "$TRACES_DIR" \
        --layer_indices $LAYER_INDICES \
        --batch_size "$(auto_batch_size 9 4)" \
        --num_traces_per_question 3 \
        --temperature 0.7 \
        --seed 42 \
        2>&1 | tee "${LOG_DIR}/stage1_collect_traces.log"
    phase_done 1
fi

# === Stage 2: Train Onset Detector ===
if should_run_stage 2 && ! is_phase_done 2; then
    log "[Stage 2/6] Training onset detector (per-layer + ensemble)"
    python3 "${SCRIPT_DIR}/train_onset_detector.py" \
        --traces_dir "$TRACES_DIR" \
        --datasets $DATASETS \
        --output_dir "$DETECTOR_DIR" \
        --layer_indices $LAYER_INDICES \
        --hidden_size 4096 \
        --num_epochs 20 \
        --batch_size 64 \
        --learning_rate 1e-3 \
        --resume_from_checkpoint auto \
        2>&1 | tee "${LOG_DIR}/stage2_train_detector.log"
    phase_done 2
fi

# === Stage 3: Train Intervention Policy (PPO) ===
if should_run_stage 3 && ! is_phase_done 3; then
    log "[Stage 3/6] Training PPO intervention policy"
    python3 "${SCRIPT_DIR}/train_intervention_policy.py" \
        --traces_dir "$TRACES_DIR" \
        --datasets truthfulqa halueval \
        --output_dir "$POLICY_DIR" \
        --num_epochs 100 \
        --batch_size 256 \
        --learning_rate 3e-4 \
        --hidden_dim 128 \
        --resume_from_checkpoint auto \
        2>&1 | tee "${LOG_DIR}/stage3_train_policy.log"
    phase_done 3
fi

# === Stage 4: Full Online Evaluation ===
if should_run_stage 4 && ! is_phase_done 4; then
    log "[Stage 4/6] Full online evaluation (3 seeds)"

    DETECTOR_PATH="${DETECTOR_DIR}/multi_layer_detector.pt"
    POLICY_PATH="${POLICY_DIR}/best_policy.pt"
    DETECTOR_TYPE="multi_layer"

    if [ ! -f "$DETECTOR_PATH" ]; then
        BEST_LAYER=$(python3 -c "import json; d=json.load(open('${DETECTOR_DIR}/detector_summary.json')); print(d['best_single_layer'])")
        DETECTOR_PATH="${DETECTOR_DIR}/probe_layer${BEST_LAYER}.pt"
        DETECTOR_TYPE="single_layer"
    fi

    python3 "${SCRIPT_DIR}/eval_chi.py" \
        --model_name "$MODEL_NAME" \
        --detector_path "$DETECTOR_PATH" \
        --policy_path "$POLICY_PATH" \
        --detector_type "$DETECTOR_TYPE" \
        --layer_indices $LAYER_INDICES \
        --hidden_size 4096 \
        --datasets $DATASETS \
        --output_dir "$RESULTS_DIR" \
        --num_samples 500 \
        --max_new_tokens 512 \
        --seeds 42 137 2024 \
        --baselines no_intervention always_truncate oracle_detector dola selfcheckgpt \
        --use_claim_eval \
        2>&1 | tee "${LOG_DIR}/stage4_eval.log"
    phase_done 4
fi

# === Stage 5: Ablation Studies ===
if should_run_stage 5 && ! is_phase_done 5; then
    log "[Stage 5/6] Ablation studies (parallel on ${NUM_GPUS} GPUs)"

    DETECTOR_PATH="${DETECTOR_DIR}/multi_layer_detector.pt"
    POLICY_PATH="${POLICY_DIR}/best_policy.pt"
    DETECTOR_TYPE="multi_layer"
    [ ! -f "$DETECTOR_PATH" ] && DETECTOR_TYPE="single_layer"

    GPU_IDX=0
    PIDS=()
    for THRESHOLD in 0.3 0.4 0.5 0.6 0.7; do
        OUT="${RESULTS_DIR}/ablation_threshold_${THRESHOLD}.json"
        [ -f "$OUT" ] && continue
        log "  GPU $((GPU_IDX % NUM_GPUS)) <- threshold=$THRESHOLD"
        (
            CUDA_VISIBLE_DEVICES=$((GPU_IDX % NUM_GPUS)) python3 "${SCRIPT_DIR}/eval_chi.py" \
                --model_name "$MODEL_NAME" \
                --detector_path "$DETECTOR_PATH" \
                --policy_path "$POLICY_PATH" \
                --detector_type "$DETECTOR_TYPE" \
                --datasets truthfulqa \
                --output_dir "${RESULTS_DIR}/tmp_thresh_${THRESHOLD}" \
                --num_samples 200 \
                --threshold "$THRESHOLD" \
                --baselines no_intervention \
                2>&1 | tee "${LOG_DIR}/ablation_thresh_${THRESHOLD}.log"
            mv "${RESULTS_DIR}/tmp_thresh_${THRESHOLD}/chi_evaluation.json" "$OUT" 2>/dev/null || true
        ) &
        PIDS+=($!)
        GPU_IDX=$((GPU_IDX + 1))
    done
    for pid in "${PIDS[@]}"; do wait "$pid" || true; done
    phase_done 5
fi

# === Stage 6: Summary ===
if should_run_stage 6 && ! is_phase_done 6; then
    log "[Stage 6/6] Generating summary"
    python3 -c "
import json, os, glob

results_dir = '${RESULTS_DIR}'
main_file = os.path.join(results_dir, 'chi_evaluation.json')
if os.path.exists(main_file):
    with open(main_file) as f:
        data = json.load(f)
    print('\n=== PHI Evaluation Summary ===')
    for ds, metrics in data.items():
        print(f'\nDataset: {ds}')
        for method, m in metrics.items():
            if isinstance(m, dict) and 'factuality' in m:
                print(f'  {method:20s}: fact={m[\"factuality\"]:.4f}')
        if 'improvement_over_baseline' in metrics:
            print(f'  PHI improvement: {metrics[\"improvement_over_baseline\"]:+.4f}')
        if 'multi_seed_aggregate' in metrics:
            agg = metrics['multi_seed_aggregate']
            if 'significance_vs_baseline' in agg:
                print(f'  Significance: p={agg[\"significance_vs_baseline\"][\"p_value\"]:.4f}')
"
    phase_done 6
fi

log "========================================"
log "PHI pipeline complete!"
log "Results: $RESULTS_DIR"
log "Logs:    $LOG_DIR"
log "========================================"

echo "PIPELINE_COMPLETE" > "${RESULTS_DIR}/.pipeline_done"
