#!/bin/bash
# Build the inference Docker image with auto-detected CUDA version.
#
# Usage:
#   ./build_inference.sh                  # auto-detect CUDA
#   ./build_inference.sh 12.4.0           # explicit version
#   CUDA_VERSION=11.8.0 ./build_inference.sh

set -e
cd "$(dirname "$0")"

if [ -n "$1" ]; then
    CUDA_VERSION="$1"
elif [ -z "$CUDA_VERSION" ]; then
    # Auto-detect from nvidia-smi
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' | head -1)
        if [ -n "$CUDA_VERSION" ]; then
            # nvidia-smi reports major.minor (e.g. 12.2), append .0
            case "$CUDA_VERSION" in
                *.*.* ) ;; # already has patch
                *     ) CUDA_VERSION="${CUDA_VERSION}.0" ;;
            esac
            echo "✅ Detected CUDA $CUDA_VERSION from nvidia-smi"
        else
            echo "⚠️  nvidia-smi found but could not parse CUDA version, using default"
        fi
    else
        echo "⚠️  nvidia-smi not found, using default CUDA version"
    fi
fi

CUDA_VERSION="${CUDA_VERSION:-12.2.0}"
IMAGE_NAME="${IMAGE_NAME:-quasar-miner-gpu:latest}"

echo "Building $IMAGE_NAME with CUDA $CUDA_VERSION"
docker build \
    -f Dockerfile.inference \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    -t "$IMAGE_NAME" \
    .
echo "Done: $IMAGE_NAME (CUDA $CUDA_VERSION)"
