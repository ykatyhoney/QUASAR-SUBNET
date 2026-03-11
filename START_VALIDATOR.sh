#!/bin/bash
# Script to start the Validator Neuron
# Run from project root
#
# Prerequisites:
#   - Docker with NVIDIA Container Toolkit
#   - Sandbox image built: docker build -t quasar-sandbox:latest -f validator/Dockerfile.sandbox .

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting QUASAR-SUBNET Validator Neuron"
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

# Set defaults if not in .env (mainnet defaults)
export VALIDATOR_API_URL=${VALIDATOR_API_URL:-"https://quasar-validator-api.onrender.com"}
export NETUID=${NETUID:-24}
export SUBTENSOR_NETWORK=${SUBTENSOR_NETWORK:-"finney"}
export WALLET_VALIDATOR_NAME=${WALLET_VALIDATOR_NAME:-"quasar_validator"}
export WALLET_HOTKEY=${WALLET_HOTKEY:-"default"}
export POLLING_INTERVAL=${POLLING_INTERVAL:-300}
export SUBTENSOR_CHAIN_ENDPOINT=${SUBTENSOR_CHAIN_ENDPOINT:-""}

# Inference verification settings
export ENABLE_LOGIT_VERIFICATION=${ENABLE_LOGIT_VERIFICATION:-"true"}
export REFERENCE_MODEL=${REFERENCE_MODEL:-"Qwen/Qwen3-4B-Instruct-2507"}
export COSINE_SIM_THRESHOLD=${COSINE_SIM_THRESHOLD:-0.99}
export MAX_ABS_DIFF_THRESHOLD=${MAX_ABS_DIFF_THRESHOLD:-0.1}

# Sandbox settings
export VALIDATOR_SANDBOX_IMAGE=${VALIDATOR_SANDBOX_IMAGE:-"quasar-sandbox:latest"}

# GPU normalization — adjusts measured TPS to a reference GPU baseline.
# Set GPU_NORMALIZATION_FACTOR to override auto-detection (e.g. "1.10" for RTX 6000 Pro).
# Or set GPU_NORMALIZATION_FACTORS as JSON to add custom GPU entries:
#   GPU_NORMALIZATION_FACTORS='{"My Custom GPU": 0.85}'
export GPU_NORMALIZATION_FACTOR=${GPU_NORMALIZATION_FACTOR:-""}
export GPU_NORMALIZATION_FACTORS=${GPU_NORMALIZATION_FACTORS:-""}

# Commit-reveal settings
export BLOCKS_UNTIL_REVEAL=${BLOCKS_UNTIL_REVEAL:-100}
export BLOCK_TIME_SECONDS=${BLOCK_TIME_SECONDS:-12}

echo "Configuration:"
echo "  API URL: $VALIDATOR_API_URL"
echo "  NetUID: $NETUID"
echo "  Network: $SUBTENSOR_NETWORK"
if [ -n "$SUBTENSOR_CHAIN_ENDPOINT" ]; then
    echo "  Chain Endpoint: $SUBTENSOR_CHAIN_ENDPOINT"
fi
echo "  Wallet: $WALLET_VALIDATOR_NAME/$WALLET_HOTKEY"
echo "  Polling Interval: $POLLING_INTERVAL seconds"
echo ""
echo "Inference Verification:"
echo "  Enabled: $ENABLE_LOGIT_VERIFICATION"
echo "  Reference Model: $REFERENCE_MODEL"
echo "  Cosine Similarity Threshold: $COSINE_SIM_THRESHOLD"
echo "  Max Absolute Difference: $MAX_ABS_DIFF_THRESHOLD"
echo ""
echo "Sandbox:"
echo "  Image: $VALIDATOR_SANDBOX_IMAGE"
echo ""
echo "GPU Normalization:"
if [ -n "$GPU_NORMALIZATION_FACTOR" ]; then
    echo "  Manual override: $GPU_NORMALIZATION_FACTOR"
else
    echo "  Auto-detect (set GPU_NORMALIZATION_FACTOR to override)"
fi
echo ""

# --- Pre-flight checks ---

# Check Docker daemon
echo "Checking Docker..."
if docker info > /dev/null 2>&1; then
    echo "✅ Docker daemon is running"
else
    echo "❌ Docker daemon is not running!"
    echo "   Validators require Docker for sandboxed testing and logit verification."
    echo "   If on RunPod: validators cannot run on RunPod (no Docker support)."
    echo "   See TROUBLESHOOTING.md for details."
    exit 1
fi

# Check sandbox image exists
if docker image inspect "$VALIDATOR_SANDBOX_IMAGE" > /dev/null 2>&1; then
    echo "✅ Sandbox image '$VALIDATOR_SANDBOX_IMAGE' found"
else
    echo "❌ Sandbox image '$VALIDATOR_SANDBOX_IMAGE' not found!"
    echo "   Build it with: docker build -t quasar-sandbox:latest -f validator/Dockerfile.sandbox ."
    exit 1
fi

# Check GPU in Docker
if docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU passthrough in Docker is working"
else
    echo "⚠️  GPU passthrough in Docker may not be configured."
    echo "   Install NVIDIA Container Toolkit — see TROUBLESHOOTING.md"
fi

# Check if API is running
echo ""
echo "Checking validator API..."
if curl -s "$VALIDATOR_API_URL/health" > /dev/null 2>&1; then
    echo "✅ Validator API is running"
else
    echo "❌ Validator API is not running at $VALIDATOR_API_URL!"
    echo "   Please check Validator API is running and try again."
    exit 1
fi

# Check CUDA for inference verification
if [ "$ENABLE_LOGIT_VERIFICATION" = "true" ]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "✅ CUDA is available for inference verification"
    else
        echo "⚠️  CUDA is not available (inference verification will be slower)"
    fi
fi

echo ""
echo "Starting validator neuron..."
echo "Press CTRL+C to stop"
echo ""

VALIDATOR_ARGS=(
    --netuid "$NETUID"
    --wallet.name "$WALLET_VALIDATOR_NAME"
    --wallet.hotkey "$WALLET_HOTKEY"
    --subtensor.network "$SUBTENSOR_NETWORK"
    --neuron.polling_interval "$POLLING_INTERVAL"
    --logging.debug
)

if [ -n "$SUBTENSOR_CHAIN_ENDPOINT" ]; then
    VALIDATOR_ARGS+=(--subtensor.chain_endpoint "$SUBTENSOR_CHAIN_ENDPOINT")
fi

python -m neurons.validator "${VALIDATOR_ARGS[@]}"
