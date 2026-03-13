#!/bin/bash
# Build and push miner Docker images.
#
# Supports two build backends:
#   - docker  (default) — requires Docker daemon; works everywhere
#   - crane   — no Docker daemon needed; works on RunPod
#
# Usage:
#   ./push_miner.sh              # Interactive
#   ./push_miner.sh gpu          # Push GPU image only
#   ./push_miner.sh cpu          # Push CPU image only
#   ./push_miner.sh both         # Push both
#   BUILDER=crane ./push_miner.sh gpu   # Use crane (RunPod)

set -e
cd "$(dirname "$0")"
cd ..  # repo root

# ── Load .env ──────────────────────────────────────────────────────
if [ -f .env ]; then
    set -a; source .env; set +a
fi

if [ -z "$DOCKER_USERNAME" ] || [ "$DOCKER_USERNAME" = "your_dockerhub_username" ]; then
    echo "❌ DOCKER_USERNAME is not set in .env"
    echo "   Add this to your .env file:"
    echo "   DOCKER_USERNAME=your_dockerhub_username"
    exit 1
fi

# ── Builder selection ──────────────────────────────────────────────
BUILDER="${BUILDER:-docker}"

if [ "$BUILDER" = "crane" ]; then
    if ! command -v crane &>/dev/null && ! command -v bazel &>/dev/null; then
        echo "❌ Neither crane nor bazel found."
        echo "   Install crane: go install github.com/google/go-containerregistry/cmd/crane@latest"
        echo "   Or install Bazel: see docker-build/README.md"
        exit 1
    fi
fi

# ── Docker Hub auth check ─────────────────────────────────────────
if [ ! -f "$HOME/.docker/config.json" ]; then
    echo "❌ Docker Hub credentials not found (~/.docker/config.json)"
    echo "   Run: echo 'YOUR_TOKEN' | docker login -u $DOCKER_USERNAME --password-stdin"
    exit 1
fi

GPU_IMAGE="${DOCKER_USERNAME}/quasar-miner-gpu:latest"
CPU_IMAGE="${DOCKER_USERNAME}/quasar-miner-cpu:latest"

echo ""
echo "Docker Hub username: $DOCKER_USERNAME"
echo "Builder:             $BUILDER"
echo "GPU image:           $GPU_IMAGE"
echo "CPU image:           $CPU_IMAGE"
echo ""

# ── Build & push functions ─────────────────────────────────────────

push_gpu_docker() {
    echo "🔨 Building GPU image with Docker..."
    docker build -f docker-build/Dockerfile.gpu \
        --build-arg CUDA_VERSION="${CUDA_VERSION:-12.2.0}" \
        -t "$GPU_IMAGE" .
    echo "📤 Pushing $GPU_IMAGE..."
    docker push "$GPU_IMAGE"
}

push_gpu_crane() {
    echo "🔨 Building GPU image with Bazel + crane..."
    cd docker-build
    ./load_config_from_env.sh
    bazel run //:push_miner_image_gpu
    cd ..
}

push_cpu_docker() {
    echo "🔨 Building CPU image with Docker..."
    docker build -f docker-build/Dockerfile.cpu \
        -t "$CPU_IMAGE" .
    echo "📤 Pushing $CPU_IMAGE..."
    docker push "$CPU_IMAGE"
}

push_cpu_crane() {
    echo "🔨 Building CPU image with Bazel + crane..."
    cd docker-build
    ./load_config_from_env.sh
    bazel run //:push_miner_image_cpu
    cd ..
}

push_gpu() {
    if [ "$BUILDER" = "crane" ]; then push_gpu_crane; else push_gpu_docker; fi
}

push_cpu() {
    if [ "$BUILDER" = "crane" ]; then push_cpu_crane; else push_cpu_docker; fi
}

# ── Choose target ──────────────────────────────────────────────────
choice="${1:-}"

if [ -z "$choice" ]; then
    echo "Which image to build and push?"
    echo "1) GPU image (for RunPod)"
    echo "2) CPU image (for Render)"
    echo "3) Both"
    read -p "Choice [1-3]: " choice
    case $choice in
        1) choice="gpu" ;;
        2) choice="cpu" ;;
        3) choice="both" ;;
        *) echo "Invalid choice"; exit 1 ;;
    esac
fi

case $choice in
    gpu)  push_gpu ;;
    cpu)  push_cpu ;;
    both) push_gpu; echo ""; push_cpu ;;
    *)    echo "Invalid choice: $choice"; echo "Usage: $0 [gpu|cpu|both]"; exit 1 ;;
esac

echo ""
echo "✅ Done! Images pushed to Docker Hub"
