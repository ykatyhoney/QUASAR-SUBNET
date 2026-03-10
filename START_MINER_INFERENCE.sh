#!/bin/bash
# Script to start the Miner Inference Server
# Run from project root
#
# This server exposes the logit verification interface:
#   POST /inference — run inference with multi-step logit capture
#   GET  /health    — health check
#   GET  /model_info — basic metadata
#
# Uses port 8001 by default to avoid conflicts with the Validator API (8000).

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting Miner Inference Server"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Activated virtual environment"
fi

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "✅ Loaded .env file"
else
    echo "⚠️  No .env file found. Using defaults."
fi

# Use 8001 to avoid conflict with Validator API (8000)
export MINER_INFERENCE_PORT=${MINER_INFERENCE_PORT:-8001}
export PORT=$MINER_INFERENCE_PORT
export HOST=${HOST:-"0.0.0.0"}
export MODEL_NAME=${MODEL_NAME:-${REFERENCE_MODEL:-"Qwen/Qwen3-4B-Instruct-2507"}}
export DEVICE=${DEVICE:-"cuda"}

echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT (avoids Validator API on 8000)"
echo "  Model: $MODEL_NAME"
echo "  Device: $DEVICE"
echo ""

# Check CUDA
if [ "$DEVICE" = "cuda" ]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "✅ CUDA is available"
    else
        echo "⚠️  CUDA not available — falling back to CPU (slower)"
        export DEVICE=cpu
    fi
fi

echo ""
echo "Inference server will be at: http://$HOST:$PORT"
echo "  POST /inference  — Run inference with logit capture"
echo "  GET  /health     — Health check"
echo "  GET  /model_info — Model metadata"
echo ""
echo "Press CTRL+C to stop"
echo ""

python -u miner/inference_server.py
