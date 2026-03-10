#!/bin/bash
# Push miner images to Docker Hub using Bazel
# Works on RunPod (no Docker daemon needed — uses crane)
#
# Usage:
#   ./push_miner.sh          # Interactive: choose GPU, CPU, or both
#   ./push_miner.sh gpu      # Non-interactive: push GPU image only
#   ./push_miner.sh cpu      # Non-interactive: push CPU image only
#   ./push_miner.sh both     # Non-interactive: push both images

set -e
cd "$(dirname "$0")"

# Load Docker username from .env
cd ..
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi
cd docker-build

if [ -z "$DOCKER_USERNAME" ] || [ "$DOCKER_USERNAME" = "your_dockerhub_username" ]; then
    echo "❌ DOCKER_USERNAME is not set in .env"
    echo "   Add this to your .env file:"
    echo "   DOCKER_USERNAME=your_dockerhub_username"
    exit 1
fi

# Check Docker Hub credentials
if [ ! -f "$HOME/.docker/config.json" ]; then
    echo "❌ Docker Hub credentials not found (~/.docker/config.json)"
    echo ""
    echo "   Set up credentials (no Docker daemon required):"
    echo "   mkdir -p ~/.docker"
    echo "   echo 'YOUR_TOKEN' | docker login -u $DOCKER_USERNAME --password-stdin"
    echo ""
    echo "   Or manually:"
    echo "   printf '{\"auths\":{\"https://index.docker.io/v1/\":{\"auth\":\"%s\"}}}' \\"
    echo "     \"\$(echo -n '$DOCKER_USERNAME:YOUR_TOKEN' | base64)\" > ~/.docker/config.json"
    exit 1
fi

# Check Bazel is installed
if ! command -v bazel &>/dev/null; then
    echo "❌ Bazel is not installed."
    echo "   Install Bazelisk:"
    echo "   wget https://github.com/bazelbuild/bazelisk/releases/download/v1.28.1/bazelisk-linux-amd64"
    echo "   chmod +x bazelisk-linux-amd64 && sudo cp bazelisk-linux-amd64 /usr/local/bin/bazel"
    exit 1
fi

# Update docker_config.bzl
./load_config_from_env.sh

echo ""
echo "Docker Hub username: $DOCKER_USERNAME"
echo "GPU image: $DOCKER_USERNAME/quasar-miner-gpu:latest"
echo "CPU image: $DOCKER_USERNAME/quasar-miner-cpu:latest"
echo ""

# Determine mode: interactive or from CLI argument
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
    gpu)
        echo "Building and pushing GPU image..."
        bazel run //:push_miner_image_gpu
        ;;
    cpu)
        echo "Building and pushing CPU image..."
        bazel run //:push_miner_image_cpu
        ;;
    both)
        echo "Building and pushing GPU image..."
        bazel run //:push_miner_image_gpu
        echo ""
        echo "Building and pushing CPU image..."
        bazel run //:push_miner_image_cpu
        ;;
    *)
        echo "Invalid choice: $choice"
        echo "Usage: $0 [gpu|cpu|both]"
        exit 1
        ;;
esac

echo ""
echo "✅ Done! Images pushed to Docker Hub"
