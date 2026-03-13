# Docker Build

Build and push QUASAR miner inference Docker images.

**Important:** All dependencies (python3, pip packages, curl) are installed at **build time**. Validator containers run on an isolated network with no internet access, so runtime package installation will fail.

## Prerequisites

- **Docker Hub account** with an access token ([create one here](https://hub.docker.com/settings/security))
- **Docker** installed (or Bazel + crane for daemon-less environments)
- **Docker Hub credentials** in `~/.docker/config.json`:

  ```bash
  echo 'YOUR_ACCESS_TOKEN' | docker login -u YOUR_USERNAME --password-stdin
  ```

- **`DOCKER_USERNAME`** set in the project `.env` file:

  ```bash
  # In /root/QUASAR-SUBNET/.env
  DOCKER_USERNAME=your_dockerhub_username
  ```

## Quick Start

```bash
cd docker-build
bash push_miner.sh gpu
```

## Build Methods

### Method 1: Docker (recommended)

Uses `Dockerfile.gpu` / `Dockerfile.cpu` to build proper images with all dependencies pre-installed.

```bash
# From repo root
docker build -f docker-build/Dockerfile.gpu -t $DOCKER_USERNAME/quasar-miner-gpu:latest .
docker push $DOCKER_USERNAME/quasar-miner-gpu:latest
```

Or use the script:

```bash
cd docker-build
bash push_miner.sh gpu   # or cpu, both
```

### Method 2: Bazel + crane (RunPod / no Docker daemon)

For environments without a Docker daemon. Requires a pre-built base image that already includes python3 and pip dependencies.

```bash
BUILDER=crane ./push_miner.sh gpu
```

## What Gets Built

| Target | Image Name | Base Image | Use Case |
|--------|-----------|------------|----------|
| GPU | `<USER>/quasar-miner-gpu:latest` | `nvidia/cuda:12.2.0-runtime-ubuntu22.04` | RunPod, GPU servers |
| CPU | `<USER>/quasar-miner-cpu:latest` | `python:3.10-slim` | Render, CPU servers |

Both images include:
- `miner/inference_server.py` — the FastAPI inference server
- All Python dependencies pre-installed (torch, transformers, fastapi, etc.)
- Environment: `MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507`, `PYTHONUNBUFFERED=1`

## Why Build-Time Installation Matters

Validators run miner containers in a **sandboxed environment**:
- Read-only root filesystem
- Internal-only Docker network (no internet)
- Dropped Linux capabilities

If your image tries to `apt-get install` or `pip install` at startup, it will fail with DNS resolution errors and your submission will score 0. Everything must be pre-installed in the image.

## File Structure

```
docker-build/
├── Dockerfile.gpu          # GPU image (recommended build method)
├── Dockerfile.cpu          # CPU image (recommended build method)
├── BUILD.bazel             # Bazel image definitions (crane-based push)
├── WORKSPACE               # External dependencies (rules_oci, base images)
├── push_miner.sh           # Build + push script (supports docker & crane)
├── load_config_from_env.sh # Generates docker_config.bzl from .env
└── docker_config.bzl       # Auto-generated: DOCKER_USERNAME
```

## Local Testing

```bash
# Build locally
docker build -f docker-build/Dockerfile.gpu -t quasar-miner-gpu:latest .

# Test
docker run --gpus all -p 8001:8000 quasar-miner-gpu:latest
curl http://localhost:8001/health
```

## Relationship to Miner

When the miner submits to `/submit_kernel`, it includes a `docker_image` field:
- Default: `<DOCKER_USERNAME>/quasar-miner-gpu:latest`
- Override: Set `MINER_DOCKER_IMAGE` in `.env`

The validator pulls this image and runs it in a sandboxed container to verify the miner's logits match a reference model.
