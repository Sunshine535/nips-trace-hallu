#!/bin/bash
# =============================================================================
# RUN experiments only (environment assumed ready)
# Install deps first:  bash setup.sh
# Then run:            bash run.sh
# =============================================================================
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

echo "============================================================"
echo " PHI: Predictive Hallucination Intervention"
echo " Starting full experiment pipeline (6 stages)"
echo " Project: $(basename "$PROJ_DIR")"
echo " Time:    $(date)"
echo "============================================================"

# Activate venv or conda env
if [ -f "$PROJ_DIR/.venv/bin/activate" ]; then
    source "$PROJ_DIR/.venv/bin/activate"
    echo "[env] Activated venv: $PROJ_DIR/.venv"
elif [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" = "nips-trace-hallu" ]; then
    echo "[env] Conda env already active: $CONDA_DEFAULT_ENV"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate nips-trace-hallu 2>/dev/null && echo "[env] Activated conda: nips-trace-hallu" || echo "[env] No conda env 'nips-trace-hallu', using system Python"
else
    echo "[env] No .venv or conda found, using system Python"
fi

# Quick dependency check
python3 -c "import torch, transformers, datasets" 2>/dev/null || {
    echo "[ERROR] Missing dependencies. Run: bash setup.sh"
    exit 1
}

echo ""
echo "Running all experiments..."
echo "  Log file: $PROJ_DIR/run.log"
echo "  To run in background: nohup bash run.sh > run.log 2>&1 &"
echo ""

bash scripts/run_all_experiments.sh 2>&1 | tee "$PROJ_DIR/run.log"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================================"
    echo " Pipeline completed successfully!"
    echo " Results: $PROJ_DIR/results/"
    echo "============================================================"
else
    echo "============================================================"
    echo " Pipeline failed (exit code: $EXIT_CODE)"
    echo " Check log: $PROJ_DIR/run.log"
    echo "============================================================"
    exit $EXIT_CODE
fi
