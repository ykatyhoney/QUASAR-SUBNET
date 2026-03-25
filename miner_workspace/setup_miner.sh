#!/bin/bash
# =============================================================================
# QUASAR-SUBNET Miner Local Setup
# =============================================================================
# This script sets up a local miner workspace for kernel optimization.
# It clones the target flash-linear-attention repo, installs dependencies,
# and prepares the environment for local benchmarking.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FLA_REPO="https://github.com/troy12x/flash-linear-attention.git"
FLA_DIR="$SCRIPT_DIR/flash-linear-attention"

echo "============================================="
echo "  QUASAR-SUBNET Miner Local Setup"
echo "============================================="

# Step 1: Clone the target repository
if [ -d "$FLA_DIR" ]; then
    echo "[SETUP] flash-linear-attention already exists at $FLA_DIR"
    echo "[SETUP] To re-clone, delete it first: rm -rf $FLA_DIR"
else
    echo "[SETUP] Cloning flash-linear-attention..."
    git clone --depth 1 "$FLA_REPO" "$FLA_DIR"
    echo "[SETUP] Cloned successfully."
fi

# Step 2: Install flash-linear-attention in editable mode
echo "[SETUP] Installing flash-linear-attention in editable mode..."
cd "$FLA_DIR"
pip install -e . 2>&1 | tail -5
cd "$SCRIPT_DIR"

# Step 3: Install project dependencies
echo "[SETUP] Installing QUASAR-SUBNET dependencies..."
pip install torch triton 2>&1 | tail -3

# Step 4: Verify the installation
echo ""
echo "[SETUP] Verifying installation..."
cd "$SCRIPT_DIR"
python3 -c "
import sys, os
sys.path.insert(0, '$SCRIPT_DIR')
from quasar_import import import_quasar_attention
QA = import_quasar_attention()
print(f'QuasarAttention imported successfully: {QA}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================="
    echo "  Setup Complete!"
    echo "============================================="
    echo ""
    echo "Target kernel files to optimize:"
    echo "  $FLA_DIR/fla/ops/quasar/chunk.py"
    echo "  $FLA_DIR/fla/ops/quasar/chunk_intra_token_parallel.py"
    echo "  $FLA_DIR/fla/ops/quasar/forward_substitution.py"
    echo "  $FLA_DIR/fla/ops/quasar/fused_recurrent.py"
    echo "  $FLA_DIR/fla/ops/quasar/gate.py"
    echo "  $FLA_DIR/fla/ops/quasar/__init__.py"
    echo ""
    echo "Next steps:"
    echo "  1. Run benchmark:  python3 benchmark_local.py"
    echo "  2. Edit kernels:   Edit files in $FLA_DIR/fla/ops/quasar/"
    echo "  3. Re-benchmark:   python3 benchmark_local.py"
    echo "  4. Validate:       python3 validate_submission.py"
    echo "  5. Submit:         python3 submit_miner.py"
else
    echo "ERROR: Setup verification failed"
    exit 1
fi
