#!/bin/bash
# ============================================================================
# PHI (Predictive Hallucination Intervention) — ACP cluster launch script
# Target: /data/szs/250010072/nwh/nips-trace-hallu
# Model:  /data/szs/share/Qwen3.5-9B (local copy)
# Data:   /data/szs/share/trace-hallu (shared trace data)
# ============================================================================
set -euo pipefail

PROJECT_DIR="/data/szs/250010072/nwh/nips-trace-hallu"
DATA_DIR="/data/szs/share/trace-hallu"
MODEL_PATH="/data/szs/share/Qwen3.5-9B"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── HuggingFace cache ────────────────────────────────────────────────────────
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

# ── Sync code to PROJECT_DIR ─────────────────────────────────────────────────
if [ "$REPO_DIR" != "$PROJECT_DIR" ]; then
    echo "[setup] Syncing code to ${PROJECT_DIR} ..."
    mkdir -p "$PROJECT_DIR"
    rsync -a --exclude='.venv' --exclude='results' --exclude='logs' \
          --exclude='checkpoints' --exclude='data/traces' --exclude='__pycache__' \
          --exclude='.git' \
          "${REPO_DIR}/" "${PROJECT_DIR}/"
fi

cd "$PROJECT_DIR"

# ── Symlink results/ and logs/ to persistent storage ─────────────────────────
for dir in results logs; do
    target="${PROJECT_DIR}/${dir}"
    if [ ! -d "$target" ] && [ ! -L "$target" ]; then
        mkdir -p "$target"
    fi
    if [ -d "$target" ] && [ ! -L "$target" ]; then
        echo "[setup] ${dir}/ already exists as real directory, keeping it"
    fi
done

if [ -d "${DATA_DIR}" ]; then
    for dir in results logs; do
        persistent="${DATA_DIR}/${dir}"
        local_dir="${PROJECT_DIR}/${dir}"
        mkdir -p "$persistent"
        if [ -d "$local_dir" ] && [ ! -L "$local_dir" ]; then
            if [ -z "$(ls -A "$local_dir" 2>/dev/null)" ]; then
                rmdir "$local_dir"
                ln -sfn "$persistent" "$local_dir"
                echo "[setup] Symlinked ${dir}/ -> ${persistent}"
            else
                echo "[setup] ${dir}/ not empty, skipping symlink (copy manually if needed)"
            fi
        elif [ ! -e "$local_dir" ]; then
            ln -sfn "$persistent" "$local_dir"
            echo "[setup] Symlinked ${dir}/ -> ${persistent}"
        fi
    done
fi

# ── Docker / venv environment ────────────────────────────────────────────────
install_deps() {
    echo "[deps] Installing into current Python environment ..."

    pip install --upgrade pip

    pip install "torch==2.10.0" "torchvision" "torchaudio" \
        --index-url https://download.pytorch.org/whl/cu128

    pip install \
        "transformers>=4.46.0" \
        "datasets>=2.21.0" \
        "accelerate>=0.34.0" \
        "peft>=0.13.0" \
        "trl>=0.15.0" \
        "deepspeed>=0.16.0" \
        "h5py>=3.12.0" \
        "scikit-learn>=1.5.0" \
        "evaluate>=0.4.0" \
        "wandb>=0.18.0" \
        "pyyaml>=6.0.1" \
        "sentencepiece>=0.2.0" \
        "scipy>=1.14.0" \
        "matplotlib>=3.9.0" \
        "tqdm>=4.66.0" \
        "pandas>=2.2.0" \
        "huggingface_hub>=0.25.0" \
        "numpy>=1.26.0"

    pip install flash-attn --no-build-isolation 2>/dev/null \
        || echo "[deps] flash-attn skipped (optional)"
}

if ! python -c "import torch; assert torch.__version__.startswith('2.10')" 2>/dev/null; then
    install_deps
fi

# ── GPU diagnostics (use total_memory, not total_mem) ────────────────────────
echo ""
echo "============================================"
echo " Environment Check"
echo "============================================"
python -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.version.cuda}')
n = torch.cuda.device_count()
print(f'  GPUs     : {n}')
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / (1024**3)
    print(f'    GPU {i}: {props.name}  ({mem_gb:.1f} GB)')
"
echo "============================================"
echo ""

# ── Pre-flight: verify model exists ─────────────────────────────────────────
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model not found at ${MODEL_PATH}"
    echo "        Please download Qwen3.5-9B to that path first."
    exit 1
fi

# ── Link shared trace data if available ──────────────────────────────────────
TRACES_DIR="${PROJECT_DIR}/data/traces"
mkdir -p "$(dirname "$TRACES_DIR")"
if [ -d "${DATA_DIR}/traces" ] && [ ! -L "$TRACES_DIR" ] && [ ! -d "$TRACES_DIR" ]; then
    ln -sfn "${DATA_DIR}/traces" "$TRACES_DIR"
    echo "[setup] Symlinked data/traces/ -> ${DATA_DIR}/traces"
elif [ ! -d "$TRACES_DIR" ]; then
    mkdir -p "$TRACES_DIR"
fi

# ── Override model name to local path ────────────────────────────────────────
export PHI_MODEL_NAME="$MODEL_PATH"

# ── Patch configs to use local model ─────────────────────────────────────────
CONFIG="${PROJECT_DIR}/configs/trace_config.yaml"
if [ -f "$CONFIG" ]; then
    if grep -q 'Qwen/Qwen3.5-9B' "$CONFIG"; then
        sed -i "s|Qwen/Qwen3.5-9B|${MODEL_PATH}|g" "$CONFIG"
        echo "[setup] Patched trace_config.yaml model path -> ${MODEL_PATH}"
    fi
fi

# ── Launch pipeline ──────────────────────────────────────────────────────────
echo "============================================"
echo " Launching PHI experiment pipeline"
echo "  PROJECT_DIR : ${PROJECT_DIR}"
echo "  DATA_DIR    : ${DATA_DIR}"
echo "  MODEL_PATH  : ${MODEL_PATH}"
echo "  HF_HOME     : ${HF_HOME}"
echo "  Time        : $(date)"
echo "============================================"

export MODEL_NAME="$MODEL_PATH"

MAIN_SCRIPT="${PROJECT_DIR}/scripts/run_all_experiments.sh"

if [ -f "$MAIN_SCRIPT" ]; then
    sed -i "s|MODEL_NAME=\"Qwen/Qwen3.5-9B\"|MODEL_NAME=\"${MODEL_PATH}\"|g" "$MAIN_SCRIPT"
fi

bash "$MAIN_SCRIPT" 2>&1 | tee "${PROJECT_DIR}/run_acp.log"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "[DONE] Pipeline completed successfully."
    echo "  Results: ${PROJECT_DIR}/results/"
    echo "  Logs:    ${PROJECT_DIR}/logs/"
else
    echo "[FAIL] Pipeline failed with exit code ${EXIT_CODE}"
    echo "  Check:   ${PROJECT_DIR}/run_acp.log"
    exit $EXIT_CODE
fi
