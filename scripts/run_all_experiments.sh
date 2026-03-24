#!/bin/bash
# ============================================================================
# CHI (Causal Hallucination Intervention) — Full Experiment Pipeline
# collect_traces → train_detector → train_policy → eval → ablations → figures
# Hardware: 4–8× A100-80GB (auto-detected)
# Model: Qwen/Qwen3.5-9B
# ============================================================================
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=gpu_utils.sh
source "${SCRIPT_DIR}/gpu_utils.sh"
auto_setup

# --- Activate project venv (created by setup.sh) ---
PROJ_DIR_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

CONFIG="${PROJECT_DIR}/configs/trace_config.yaml"

TRACES_DIR="${PROJECT_DIR}/data/traces"
DETECTOR_DIR="${PROJECT_DIR}/checkpoints/detector"
POLICY_DIR="${PROJECT_DIR}/checkpoints/intervention_policy"
RESULTS_DIR="${PROJECT_DIR}/results"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "$TRACES_DIR" "$DETECTOR_DIR" "$POLICY_DIR" "$RESULTS_DIR" "$LOG_DIR"

MODEL_NAME="Qwen/Qwen3.5-9B"
LAYER_INDICES="8 16 24 32"
DATASETS="truthfulqa halueval faithdial"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $1"; }

log "========================================="
log " CHI Experiment Pipeline"
log " Model: ${MODEL_NAME}"
log " GPUs:  ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
log "========================================="

# ============================================================================
# Stage 1: Collect Annotated Traces (HDF5 + JSONL)
# ============================================================================
log "========================================="
log "[Stage 1/6] Collecting annotated traces"
log "========================================="

COLLECTION_DONE="${TRACES_DIR}/collection_stats.json"
if [ -f "$COLLECTION_DONE" ]; then
    log "Traces already collected, skipping."
else
    python "${SCRIPT_DIR}/collect_traces.py" \
        --model_name "$MODEL_NAME" \
        --datasets $DATASETS \
        --output_dir "$TRACES_DIR" \
        --layer_indices $LAYER_INDICES \
        --batch_size "$(auto_batch_size 9 4)" \
        --num_traces_per_question 3 \
        --temperature 0.7 \
        2>&1 | tee "${LOG_DIR}/stage1_collect_traces.log"
fi

# ============================================================================
# Stage 2: Train Hallucination Onset Detector
# ============================================================================
log "========================================="
log "[Stage 2/6] Training onset detector"
log "========================================="

DETECTOR_DONE="${DETECTOR_DIR}/detector_summary.json"
if [ -f "$DETECTOR_DONE" ]; then
    log "Detector already trained, skipping."
else
    python "${SCRIPT_DIR}/train_onset_detector.py" \
        --traces_dir "$TRACES_DIR" \
        --datasets $DATASETS \
        --output_dir "$DETECTOR_DIR" \
        --layer_indices $LAYER_INDICES \
        --hidden_size 3584 \
        --num_epochs 20 \
        --batch_size 64 \
        --learning_rate 1e-3 \
        2>&1 | tee "${LOG_DIR}/stage2_train_detector.log"
fi

# ============================================================================
# Stage 3: Train RL Intervention Policy (PPO)
# ============================================================================
log "========================================="
log "[Stage 3/6] Training intervention policy"
log "========================================="

POLICY_DONE="${POLICY_DIR}/training_summary.json"
if [ -f "$POLICY_DONE" ]; then
    log "Policy already trained, skipping."
else
    python "${SCRIPT_DIR}/train_intervention_policy.py" \
        --traces_dir "$TRACES_DIR" \
        --datasets truthfulqa halueval \
        --output_dir "$POLICY_DIR" \
        --num_epochs 100 \
        --batch_size 256 \
        --learning_rate 3e-4 \
        --hidden_dim 128 \
        2>&1 | tee "${LOG_DIR}/stage3_train_policy.log"
fi

# ============================================================================
# Stage 4: Full CHI Evaluation
# ============================================================================
log "========================================="
log "[Stage 4/6] Full CHI evaluation"
log "========================================="

DETECTOR_PATH="${DETECTOR_DIR}/multi_layer_detector.pt"
POLICY_PATH="${POLICY_DIR}/best_policy.pt"

if [ ! -f "$DETECTOR_PATH" ]; then
    BEST_LAYER=$(python -c "
import json
with open('${DETECTOR_DIR}/detector_summary.json') as f:
    d = json.load(f)
print(d['best_single_layer'])
")
    DETECTOR_PATH="${DETECTOR_DIR}/probe_layer${BEST_LAYER}.pt"
    DETECTOR_TYPE="single_layer"
    log "Using single-layer detector (layer $BEST_LAYER)"
else
    DETECTOR_TYPE="multi_layer"
    log "Using multi-layer ensemble detector"
fi

python "${SCRIPT_DIR}/eval_chi.py" \
    --model_name "$MODEL_NAME" \
    --detector_path "$DETECTOR_PATH" \
    --policy_path "$POLICY_PATH" \
    --detector_type "$DETECTOR_TYPE" \
    --layer_indices $LAYER_INDICES \
    --hidden_size 3584 \
    --datasets $DATASETS \
    --output_dir "$RESULTS_DIR" \
    --num_samples 500 \
    --max_new_tokens 512 \
    2>&1 | tee "${LOG_DIR}/stage4_eval_chi.log"

# ============================================================================
# Stage 5: Ablation Studies
# ============================================================================
log "========================================="
log "[Stage 5/6] Ablation studies"
log "========================================="

log "Threshold ablation (parallel across ${NUM_GPUS} GPU(s))"
GPU_IDX=0
PIDS=()
LABELS=()
for THRESHOLD in 0.3 0.4 0.5 0.6 0.7; do
    ABLATION_TAG="threshold_${THRESHOLD}"
    ABLATION_OUT="${RESULTS_DIR}/ablation_${ABLATION_TAG}.json"
    if [ -f "$ABLATION_OUT" ]; then
        log "Ablation $ABLATION_TAG already done, skipping."
        continue
    fi
    log "  GPU $((GPU_IDX % NUM_GPUS)) ← threshold=$THRESHOLD"
    (
        CUDA_VISIBLE_DEVICES=$((GPU_IDX % NUM_GPUS)) python "${SCRIPT_DIR}/eval_chi.py" \
            --model_name "$MODEL_NAME" \
            --detector_path "$DETECTOR_PATH" \
            --policy_path "$POLICY_PATH" \
            --detector_type "$DETECTOR_TYPE" \
            --layer_indices $LAYER_INDICES \
            --datasets truthfulqa \
            --output_dir "${RESULTS_DIR}/tmp_${ABLATION_TAG}" \
            --num_samples 200 \
            --threshold "$THRESHOLD" \
            2>&1 | tee "${LOG_DIR}/ablation_${ABLATION_TAG}.log"
        mv "${RESULTS_DIR}/tmp_${ABLATION_TAG}/chi_evaluation.json" "$ABLATION_OUT" 2>/dev/null || true
    ) &
    PIDS+=($!)
    LABELS+=("$ABLATION_TAG")
    GPU_IDX=$((GPU_IDX + 1))
done
FAIL=0
for j in "${!PIDS[@]}"; do
    wait "${PIDS[$j]}" || { log "ERROR: ${LABELS[$j]} failed"; FAIL=1; }
done
if [ $FAIL -ne 0 ]; then exit 1; fi

log "Single-layer detector ablation (parallel across ${NUM_GPUS} GPU(s))"
GPU_IDX=0
PIDS=()
LABELS=()
for LAYER in $LAYER_INDICES; do
    ABLATION_TAG="single_layer_${LAYER}"
    ABLATION_OUT="${RESULTS_DIR}/ablation_${ABLATION_TAG}.json"
    if [ -f "$ABLATION_OUT" ]; then
        log "Ablation $ABLATION_TAG already done, skipping."
        continue
    fi
    LAYER_DET="${DETECTOR_DIR}/probe_layer${LAYER}.pt"
    if [ ! -f "$LAYER_DET" ]; then
        log "No probe for layer $LAYER, skipping."
        continue
    fi
    log "  GPU $((GPU_IDX % NUM_GPUS)) ← single_layer=$LAYER"
    (
        CUDA_VISIBLE_DEVICES=$((GPU_IDX % NUM_GPUS)) python "${SCRIPT_DIR}/eval_chi.py" \
            --model_name "$MODEL_NAME" \
            --detector_path "$LAYER_DET" \
            --policy_path "$POLICY_PATH" \
            --detector_type single_layer \
            --detector_layer "$LAYER" \
            --datasets truthfulqa \
            --output_dir "${RESULTS_DIR}/tmp_${ABLATION_TAG}" \
            --num_samples 200 \
            2>&1 | tee "${LOG_DIR}/ablation_${ABLATION_TAG}.log"
        mv "${RESULTS_DIR}/tmp_${ABLATION_TAG}/chi_evaluation.json" "$ABLATION_OUT" 2>/dev/null || true
    ) &
    PIDS+=($!)
    LABELS+=("$ABLATION_TAG")
    GPU_IDX=$((GPU_IDX + 1))
done
FAIL=0
for j in "${!PIDS[@]}"; do
    wait "${PIDS[$j]}" || { log "ERROR: ${LABELS[$j]} failed"; FAIL=1; }
done
if [ $FAIL -ne 0 ]; then exit 1; fi

# ============================================================================
# Stage 6: Summary
# ============================================================================
log "========================================="
log "[Stage 6/6] Generating summary"
log "========================================="

python -c "
import json, os, glob

results_dir = '${RESULTS_DIR}'
main_results = os.path.join(results_dir, 'chi_evaluation.json')
if os.path.exists(main_results):
    with open(main_results) as f:
        data = json.load(f)
    print('\\n=== CHI Evaluation Summary ===')
    for ds, metrics in data.items():
        print(f'\\nDataset: {ds}')
        for method, m in metrics.items():
            if isinstance(m, dict) and 'factuality' in m:
                print(f'  {method:20s}: fact={m[\"factuality\"]:.4f} ppl={m[\"perplexity\"]:.1f}')

ablations = sorted(glob.glob(os.path.join(results_dir, 'ablation_*.json')))
if ablations:
    print('\\n=== Ablation Results ===')
    for path in ablations:
        name = os.path.basename(path).replace('ablation_', '').replace('.json', '')
        with open(path) as f:
            d = json.load(f)
        for ds, m in d.items():
            if 'chi_ours' in m:
                print(f'  {name}: factuality={m[\"chi_ours\"][\"factuality\"]:.4f}')
"

log "========================================="
log "CHI experiment pipeline complete!"
log "Traces:   $TRACES_DIR"
log "Detector: $DETECTOR_DIR"
log "Policy:   $POLICY_DIR"
log "Results:  $RESULTS_DIR"
log "Logs:     $LOG_DIR"
log "========================================="

# --- Pipeline completion marker ---
DONE_FILE="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/results/.pipeline_done"
mkdir -p "$(dirname "$DONE_FILE")"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "$(basename "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo ""
echo "[PIPELINE_COMPLETE] All experiments finished successfully."
echo "  Marker: $DONE_FILE"
echo "  Run 'bash collect_results.sh' to package results."
