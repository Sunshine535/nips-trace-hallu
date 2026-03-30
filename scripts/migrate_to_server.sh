#!/bin/bash
# ============================================================================
# PHI Server Migration Script
# Usage: bash scripts/migrate_to_server.sh <ssh_target> <remote_dir>
# Example: bash scripts/migrate_to_server.sh user@server /path/to/research
# ============================================================================
set -euo pipefail

SSH_TARGET="${1:?Usage: $0 <ssh_target> <remote_dir>}"
REMOTE_DIR="${2:?Usage: $0 <ssh_target> <remote_dir>}"
SSH_PORT="${3:-22}"
SSH_PASS="${4:-}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

echo "=== PHI Server Migration ==="
echo "  Source: $PROJECT_DIR"
echo "  Target: ${SSH_TARGET}:${REMOTE_DIR}/${PROJECT_NAME}"
echo "  Port:   $SSH_PORT"

SSH_CMD="ssh -o StrictHostKeyChecking=no -o PubkeyAuthentication=no -p $SSH_PORT"
SCP_CMD="scp -o StrictHostKeyChecking=no -o PubkeyAuthentication=no -P $SSH_PORT"
if [ -n "$SSH_PASS" ] && command -v sshpass &>/dev/null; then
    SSH_CMD="sshpass -p '$SSH_PASS' $SSH_CMD"
    SCP_CMD="sshpass -p '$SSH_PASS' $SCP_CMD"
fi

echo ""
echo "[1/4] Creating archive..."
ARCHIVE="/tmp/${PROJECT_NAME}_migrate_$(date +%Y%m%d_%H%M%S).tar.gz"
tar czf "$ARCHIVE" \
    -C "$(dirname "$PROJECT_DIR")" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='data/traces/*.h5' \
    --exclude='checkpoints/' \
    --exclude='skills/' \
    --exclude='templates/' \
    "$PROJECT_NAME"
echo "  Archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"

echo ""
echo "[2/4] Uploading to server..."
eval $SCP_CMD "$ARCHIVE" "${SSH_TARGET}:${REMOTE_DIR}/"

echo ""
echo "[3/4] Extracting on server..."
eval $SSH_CMD "$SSH_TARGET" "cd ${REMOTE_DIR} && tar xzf $(basename $ARCHIVE) && echo 'Extracted OK'"

echo ""
echo "[4/4] Setting up environment..."
eval $SSH_CMD "$SSH_TARGET" "cd ${REMOTE_DIR}/${PROJECT_NAME} && chmod +x run.sh scripts/*.sh && echo 'Setup complete'"

echo ""
echo "=== Migration Complete ==="
echo "To start experiments on the server:"
echo "  $SSH_CMD $SSH_TARGET"
echo "  cd ${REMOTE_DIR}/${PROJECT_NAME}"
echo "  bash scripts/run_production.sh"
echo ""
echo "To resume after interruption:"
echo "  bash scripts/run_production.sh  # auto-skips completed phases"
echo ""
echo "To run a specific stage:"
echo "  bash scripts/run_production.sh --stage 4"

rm -f "$ARCHIVE"
