#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="nips-trace-hallu"

echo "============================================"
echo " Environment Setup (uv preferred; conda fallback)"
echo "============================================"

VENV_DIR="$PROJ_DIR/.venv"

USE_CONDA=0
if ! command -v uv &>/dev/null; then
    echo "[1/5] uv not found — attempting install ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null && export PATH="$HOME/.local/bin:$PATH" || true
fi

if command -v uv &>/dev/null; then
    echo "[1/5] Using uv: $(uv --version)"

    if [ ! -d "$VENV_DIR" ]; then
        echo "[2/5] Creating Python 3.10 venv ..."
        uv venv "$VENV_DIR" --python 3.10 2>/dev/null || uv venv "$VENV_DIR"
    else
        echo "[2/5] Venv exists: $VENV_DIR"
    fi
    source "$VENV_DIR/bin/activate"

    echo "[3/5] Installing PyTorch + CUDA 12.8 ..."
    uv pip install "torch>=2.4" "torchvision" "torchaudio" \
        --index-url https://download.pytorch.org/whl/cu128

    echo "[4/5] Installing project dependencies ..."
    uv pip install -r "$PROJ_DIR/requirements.txt" \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match

    echo "[5/5] Installing flash-attn (optional) ..."
    uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped (optional)"

else
    USE_CONDA=1
    echo "[1/5] uv unavailable — falling back to conda"

    if ! command -v conda &>/dev/null; then
        echo "ERROR: Neither uv nor conda found. Install one and retry."
        exit 1
    fi

    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "[2/5] Conda env '${ENV_NAME}' exists, activating ..."
    else
        echo "[2/5] Creating conda env '${ENV_NAME}' (Python 3.10) ..."
        conda create -y -n "$ENV_NAME" python=3.10
    fi

    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "$ENV_NAME"

    echo "[3/5] Installing PyTorch + CUDA 12.8 via pip ..."
    pip install "torch>=2.4" torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

    echo "[4/5] Installing project dependencies ..."
    pip install -r "$PROJ_DIR/requirements.txt" \
        --extra-index-url https://download.pytorch.org/whl/cu128

    echo "[5/5] Installing flash-attn (optional) ..."
    pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped (optional)"

    VENV_DIR="conda:${ENV_NAME}"
fi

echo ""
echo "============================================"
python -c "
import torch
print(f'  PyTorch  : {torch.__version__}')
print(f'  CUDA     : {torch.version.cuda}')
print(f'  GPUs     : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo "============================================"
echo ""
echo "Setup complete!"
if [ "$USE_CONDA" -eq 1 ]; then
    echo "  Activate:  conda activate ${ENV_NAME}"
else
    echo "  Activate:  source $PROJ_DIR/.venv/bin/activate"
fi
echo "  Run:       bash scripts/run_all_experiments.sh"
