#!/usr/bin/env bash
# NeurIPS 2026 — nips-trace-hallu: conda env, PyTorch 2.4.0 + CUDA 12.1, deps, flash-attn, verify
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_NAME="${NIPS_CONDA_ENV:-nips-trace-hallu}"
PY_VER="${PYTHON_VERSION:-3.10}"

echo "============================================"
echo " nips-trace-hallu — environment setup"
echo "  env: $ENV_NAME  |  Python: $PY_VER"
echo "============================================"

if ! command -v conda &>/dev/null; then
  echo "[ERROR] conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[INFO] Conda env '$ENV_NAME' exists; activating."
else
  echo "[INFO] Creating conda env '$ENV_NAME'..."
  conda create -n "$ENV_NAME" "python=${PY_VER}" -y
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip setuptools wheel

echo "[INFO] Installing PyTorch 2.4.0 (cu121)..."
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[INFO] Installing project requirements..."
python -m pip install -r requirements.txt

echo "[INFO] Installing flash-attn (optional; may take several minutes)..."
export MAX_JOBS="${MAX_JOBS:-4}"
if python -m pip install flash-attn --no-build-isolation; then
  echo "[OK] flash-attn installed."
else
  echo "[WARN] flash-attn build/install failed — continuing without it."
fi

echo ""
echo "============================================"
echo " Verification"
echo "============================================"
python <<'PY'
import importlib, sys, torch

print(f"torch: {torch.__version__}")
if not torch.cuda.is_available():
    print("[ERROR] CUDA not available to PyTorch.")
    sys.exit(1)
n = torch.cuda.device_count()
print(f"CUDA devices: {n}")
for i in range(n):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")

for mod in ("transformers", "peft", "datasets", "accelerate", "h5py", "scipy", "sklearn"):
    m = "sklearn" if mod == "sklearn" else mod
    importlib.import_module(m)
    print(f"import ok: {mod}")

try:
    import flash_attn
    print("import ok: flash_attn")
except Exception as e:
    print(f"flash_attn: not available ({e})")

print("[OK] Core stack verified.")
PY

echo ""
echo "Done. Activate with: conda activate $ENV_NAME"
