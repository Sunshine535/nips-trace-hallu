#!/bin/bash
# Trace-Hallu: Full pipeline launcher
# Stage 1: Generate CoT traces with hallucination labels
# Stage 2: Train onset detector
# Stage 3: Train intervention policy (GRPO)
# Stage 4: Evaluate full pipeline
# 8x A100-80GB

set -euo pipefail

# Activate venv if available
_PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$_PROJ_ROOT/.venv/bin/activate" ]; then source "$_PROJ_ROOT/.venv/bin/activate"; fi
export PATH="$HOME/.local/bin:$PATH"

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${PROJECT_DIR}/configs/trace_config.yaml"

TRACES_DIR="${PROJECT_DIR}/data/traces"
DETECTOR_DIR="${PROJECT_DIR}/checkpoints/detector"
POLICY_DIR="${PROJECT_DIR}/checkpoints/intervention_policy"
RESULTS_DIR="${PROJECT_DIR}/results"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "$TRACES_DIR" "$DETECTOR_DIR" "$POLICY_DIR" "$RESULTS_DIR" "$LOG_DIR"

NUM_GPUS=8

# =========================================
# Stage 1: Generate traces
# =========================================
echo "========================================="
echo "[Stage 1] Generating CoT traces..."
echo "========================================="

for DATASET in truthfulqa halueval; do
    TRACE_FILE="${TRACES_DIR}/traces_${DATASET}.jsonl"
    if [ -f "$TRACE_FILE" ]; then
        echo "Traces for $DATASET already exist, skipping."
        continue
    fi

    echo "Generating traces for $DATASET..."
    accelerate launch \
        --num_processes $NUM_GPUS \
        --mixed_precision bf16 \
        "${SCRIPT_DIR}/generate_traces.py" \
        --config "$CONFIG" \
        --dataset "$DATASET" \
        --output_dir "$TRACES_DIR" \
        2>&1 | tee "${LOG_DIR}/generate_${DATASET}.log"
done

# =========================================
# Stage 2: Train onset detector
# =========================================
echo "========================================="
echo "[Stage 2] Training onset detector..."
echo "========================================="

TRACE_FILE="${TRACES_DIR}/traces_truthfulqa.jsonl"
if [ ! -f "$TRACE_FILE" ]; then
    echo "ERROR: Trace file not found: $TRACE_FILE"
    exit 1
fi

if [ -f "${DETECTOR_DIR}/detector_summary.json" ]; then
    echo "Detector already trained, skipping."
else
    python "${SCRIPT_DIR}/train_onset_detector.py" \
        --config "$CONFIG" \
        --traces_file "$TRACE_FILE" \
        --output_dir "$DETECTOR_DIR" \
        --multi_layer \
        2>&1 | tee "${LOG_DIR}/train_detector.log"
fi

# Find best detector layer
BEST_LAYER=$(python -c "
import json
with open('${DETECTOR_DIR}/detector_summary.json') as f:
    d = json.load(f)
print(d['best_layer'])
")
echo "Best detector layer: $BEST_LAYER"
DETECTOR_PATH="${DETECTOR_DIR}/probe_layer${BEST_LAYER}.pt"

# =========================================
# Stage 3: Train intervention policy (GRPO)
# =========================================
echo "========================================="
echo "[Stage 3] Training intervention policy..."
echo "========================================="

if [ -f "${POLICY_DIR}/training_metrics.json" ]; then
    echo "Policy already trained, skipping."
else
    accelerate launch \
        --num_processes $NUM_GPUS \
        --mixed_precision bf16 \
        "${SCRIPT_DIR}/train_intervention_policy.py" \
        --config "$CONFIG" \
        --traces_file "$TRACE_FILE" \
        --output_dir "$POLICY_DIR" \
        2>&1 | tee "${LOG_DIR}/train_policy.log"
fi

# =========================================
# Stage 4: Evaluate
# =========================================
echo "========================================="
echo "[Stage 4] Evaluating full pipeline..."
echo "========================================="

for DATASET in truthfulqa halueval; do
    echo "Evaluating on $DATASET..."
    python "${SCRIPT_DIR}/eval_intervention.py" \
        --config "$CONFIG" \
        --detector_path "$DETECTOR_PATH" \
        --policy_dir "$POLICY_DIR" \
        --detector_layer "$BEST_LAYER" \
        --output_dir "$RESULTS_DIR" \
        --dataset "$DATASET" \
        2>&1 | tee "${LOG_DIR}/eval_${DATASET}.log"
done

echo "========================================="
echo "Trace-Hallu pipeline complete!"
echo "Traces:   $TRACES_DIR"
echo "Detector: $DETECTOR_DIR"
echo "Policy:   $POLICY_DIR"
echo "Results:  $RESULTS_DIR"
echo "========================================="
