#!/bin/bash
# ============================================================================
# PHI (Predictive Hallucination Intervention) — Full Experiment Pipeline
# collect_traces → train_detector → train_policy → eval → ablations → summary
# Hardware: 4–8× A100-80GB (auto-detected)
# Model: Qwen/Qwen3.5-9B
# ============================================================================
set -euo pipefail

# HF_ENDPOINT removed (use default huggingface.co)
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
elif command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q "^nips-trace-hallu "; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate nips-trace-hallu
fi
export PATH="$HOME/.local/bin:$PATH"

PHASE_MARKER_DIR="$PROJ_DIR_ROOT/results/.phase_markers"
mkdir -p "$PHASE_MARKER_DIR"
FORCE_RERUN="${FORCE_RERUN:-0}"

phase_done() {
    touch "$PHASE_MARKER_DIR/phase_${1}.done"
    echo "[PHASE $1] Completed at $(date)"
}
is_phase_done() {
    [[ "$FORCE_RERUN" == "1" ]] && return 1
    [[ -f "$PHASE_MARKER_DIR/phase_${1}.done" ]] && echo "[PHASE $1] Already completed. Skipping. (FORCE_RERUN=1 to override)" && return 0
    return 1
}

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
log " PHI Experiment Pipeline"
log " Model: ${MODEL_NAME}"
log " GPUs:  ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
log "========================================="

# ============================================================================
# Stage 1: Collect Annotated Traces (HDF5 + JSONL)
# ============================================================================
if ! is_phase_done 1; then
log "========================================="
log "[Stage 1/6] Collecting annotated traces"
log "========================================="

COLLECTION_DONE="${TRACES_DIR}/collection_stats.json"
if [ -f "$COLLECTION_DONE" ]; then
    log "Traces already collected, skipping."
else
    # --- Phase 1a: Extract hidden states from pre-existing JSONL traces ---
    NEED_FULL_COLLECT=""
    for DS in $DATASETS; do
        EXISTING_JSONL="${PROJECT_DIR}/traces_${DS}.jsonl"
        TARGET_H5="${TRACES_DIR}/hidden_states_${DS}.h5"
        TARGET_JSONL="${TRACES_DIR}/traces_${DS}.jsonl"

        if [ -f "$TARGET_H5" ] && [ -f "$TARGET_JSONL" ]; then
            log "  ${DS}: HDF5 + JSONL already in place, skipping."
        elif [ -f "$EXISTING_JSONL" ]; then
            log "  ${DS}: Found pre-existing traces at ${EXISTING_JSONL}"
            log "  ${DS}: Extracting hidden states (skipping text generation)..."
            python "${SCRIPT_DIR}/extract_hidden_states.py" \
                --jsonl_path "$EXISTING_JSONL" \
                --output_dir "$TRACES_DIR" \
                --dataset_name "$DS" \
                --model_name "$MODEL_NAME" \
                --layer_indices $LAYER_INDICES \
                2>&1 | tee "${LOG_DIR}/stage1_extract_${DS}.log"
        else
            NEED_FULL_COLLECT="${NEED_FULL_COLLECT} ${DS}"
        fi
    done

    # --- Phase 1b: Full collection for datasets without pre-existing traces ---
    if [ -n "$NEED_FULL_COLLECT" ]; then
        log "  Running full collect_traces for:${NEED_FULL_COLLECT}"
        python "${SCRIPT_DIR}/collect_traces.py" \
            --model_name "$MODEL_NAME" \
            --datasets $NEED_FULL_COLLECT \
            --output_dir "$TRACES_DIR" \
            --layer_indices $LAYER_INDICES \
            --batch_size "$(auto_batch_size 9 4)" \
            --num_traces_per_question 3 \
            --temperature 0.7 \
            2>&1 | tee "${LOG_DIR}/stage1_collect_traces.log"
    fi

    # --- Write collection stats ---
    python -c "
import json, os, glob
traces_dir = '${TRACES_DIR}'
stats = {}
for jsonl in glob.glob(os.path.join(traces_dir, 'traces_*.jsonl')):
    ds = os.path.basename(jsonl).replace('traces_', '').replace('.jsonl', '')
    with open(jsonl) as f:
        lines = f.readlines()
    n = len(lines)
    hallu = sum(1 for l in lines if json.loads(l).get('has_hallucination', False))
    stats[ds] = {'total': n, 'hallucinated': hallu, 'rate': hallu/max(n,1)}
with open(os.path.join(traces_dir, 'collection_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
"
fi

if [ ! -f "${TRACES_DIR}/collection_stats.json" ]; then
    log "ERROR: Stage 1 failed — collection_stats.json not found"
    exit 1
fi
phase_done 1
fi

# ============================================================================
# Stage 2: Train Hallucination Onset Detector
# ============================================================================
if ! is_phase_done 2; then
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
        --hidden_size 4096 \
        --num_epochs 20 \
        --batch_size 64 \
        --learning_rate 1e-3 \
        --resume_from_checkpoint auto \
        2>&1 | tee "${LOG_DIR}/stage2_train_detector.log"
fi

if [ ! -f "${DETECTOR_DIR}/detector_summary.json" ]; then
    log "ERROR: Stage 2 failed — detector_summary.json not found"
    exit 1
fi
phase_done 2
fi

# ============================================================================
# Stage 3: Train RL Intervention Policy (PPO)
# ============================================================================
if ! is_phase_done 3; then
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
        --resume_from_checkpoint auto \
        2>&1 | tee "${LOG_DIR}/stage3_train_policy.log"
fi

if [ ! -f "${POLICY_DIR}/training_summary.json" ]; then
    log "ERROR: Stage 3 failed — training_summary.json not found"
    exit 1
fi
phase_done 3
fi

# ============================================================================
# Stage 3b: Online Policy Fine-tuning
# Refines the offline policy with real LLM rollouts.
# ============================================================================
if ! is_phase_done 3b; then
log "========================================="
log "[Stage 3b] Online policy fine-tuning"
log "========================================="

ONLINE_DIR="${PROJECT_DIR}/checkpoints/online_policy"
mkdir -p "$ONLINE_DIR"

python "${SCRIPT_DIR}/train_policy_online.py" \
    --model_name "$MODEL_NAME" \
    --detector_path "${DETECTOR_DIR}/multi_layer_detector.pt" \
    --detector_type multi_layer \
    --layer_indices $LAYER_INDICES \
    --hidden_size 4096 \
    --output_dir "$ONLINE_DIR" \
    --dataset truthfulqa \
    --max_samples 200 \
    --num_epochs 20 \
    --pretrained_policy "${POLICY_DIR}/best_policy.pt" \
    2>&1 | tee "${LOG_DIR}/stage3b_online_policy.log"
phase_done 3b
fi

# ============================================================================
# Stage 4: Full PHI Evaluation
# ============================================================================
if ! is_phase_done 4; then
log "========================================="
log "[Stage 4/6] Full PHI evaluation"
log "========================================="

DETECTOR_PATH="${DETECTOR_DIR}/multi_layer_detector.pt"
POLICY_PATH="${POLICY_DIR}/best_policy.pt"

ONLINE_POLICY="${PROJECT_DIR}/checkpoints/online_policy/best_policy.pt"
ONLINE_FINAL="${PROJECT_DIR}/checkpoints/online_policy/final_policy.pt"
if [ -f "$ONLINE_POLICY" ]; then
    POLICY_PATH="$ONLINE_POLICY"
    log "Using online-refined policy: $ONLINE_POLICY"
fi

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
    --hidden_size 4096 \
    --datasets $DATASETS \
    --output_dir "$RESULTS_DIR" \
    --num_samples 500 \
    --max_new_tokens 512 \
    --baselines no_intervention always_truncate detector_oracle dola iti selfcheckgpt \
    2>&1 | tee "${LOG_DIR}/stage4_eval_chi.log"

if [ ! -f "${RESULTS_DIR}/chi_evaluation.json" ]; then
    log "ERROR: Stage 4 failed — chi_evaluation.json not found"
    exit 1
fi
phase_done 4
fi

# ============================================================================
# Stage 5: Ablation Studies
# ============================================================================
if ! is_phase_done 5; then
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
phase_done 5
fi

# ============================================================================
# Stage 6: Summary
# ============================================================================
if ! is_phase_done 6; then
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
    print('\\n=== PHI Evaluation Summary ===')
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
phase_done 6
fi

log "========================================="
log "PHI experiment pipeline complete!"
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
