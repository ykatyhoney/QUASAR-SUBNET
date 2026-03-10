# Docker Build (Bazel)

Build and push QUASAR miner inference Docker images using [Bazel](https://bazel.build) + [rules_oci](https://github.com/bazel-contrib/rules_oci).

This approach uses `crane` to push images directly to Docker Hub **without requiring a Docker daemon**. This is essential for environments like RunPod where `docker run` / `docker build` are not available.

## Prerequisites

- **Docker Hub account** with an access token ([create one here](https://hub.docker.com/settings/security))
- **Bazel** (via Bazelisk):

  ```bash
  wget https://github.com/bazelbuild/bazelisk/releases/download/v1.28.1/bazelisk-linux-amd64
  chmod +x bazelisk-linux-amd64
  sudo cp bazelisk-linux-amd64 /usr/local/bin/bazel
  ```

- **Docker Hub credentials** in `~/.docker/config.json`:

  ```bash
  mkdir -p ~/.docker
  echo 'YOUR_ACCESS_TOKEN' | docker login -u YOUR_USERNAME --password-stdin 2>/dev/null || \
    printf '{"auths":{"https://index.docker.io/v1/":{"auth":"%s"}}}' \
      "$(echo -n 'YOUR_USERNAME:YOUR_ACCESS_TOKEN' | base64)" > ~/.docker/config.json
  ```

- **`DOCKER_USERNAME`** set in the project `.env` file:

  ```bash
  # In /root/QUASAR-SUBNET/.env
  DOCKER_USERNAME=your_dockerhub_username
  ```

## Quick Start

```bash
cd docker-build
bash push_miner.sh
```

This interactive script will:
1. Load `DOCKER_USERNAME` from `.env`
2. Update `docker_config.bzl`
3. Let you choose GPU, CPU, or both images
4. Build and push to Docker Hub

## Manual Build

```bash
cd docker-build

# Update Bazel config from .env
bash load_config_from_env.sh

# Push GPU image (for RunPod / CUDA environments)
bazel run //:push_miner_image_gpu

# Push CPU image (for Render / CPU environments)
bazel run //:push_miner_image_cpu
```

## What Gets Built

| Target | Image Name | Base Image | Use Case |
|--------|-----------|------------|----------|
| `push_miner_image_gpu` | `<DOCKER_USERNAME>/quasar-miner-gpu:latest` | `nvidia/cuda:12.1.0-runtime-ubuntu22.04` | RunPod, GPU servers |
| `push_miner_image_cpu` | `<DOCKER_USERNAME>/quasar-miner-cpu:latest` | `python:3.10-slim` | Render, CPU servers |

Both images include:
- `miner/inference_server.py` — the FastAPI inference server
- `miner/requirements.inference.txt` — Python dependencies
- Environment: `MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507`, `PYTHONUNBUFFERED=1`

## File Structure

```
docker-build/
├── BUILD.bazel              # Image definitions and push targets
├── WORKSPACE                # External dependencies (rules_oci, base images)
├── MODULE.bazel             # Bazel module declaration
├── .bazelversion            # Pins Bazel to 7.4.1
├── docker_config.bzl        # Auto-generated: DOCKER_USERNAME (from .env)
├── load_config_from_env.sh  # Generates docker_config.bzl from .env
├── push_miner.sh            # Interactive build+push script
└── miner/
    ├── BUILD.bazel              # Exports inference server files
    ├── inference_server.py      # Miner inference server source
    └── requirements.inference.txt  # Python dependencies
```

## Local Testing (without pushing)

Create a local tarball for testing:

```bash
bazel build //:quasar_miner_tarball_gpu
# Output: bazel-bin/quasar_miner_tarball_gpu/tarball.tar

# Load into local Docker (requires Docker daemon)
docker load < bazel-bin/quasar_miner_tarball_gpu/tarball.tar
docker run --gpus all -p 8001:8000 quasar-miner-gpu:latest
```

## How It Works

1. `load_config_from_env.sh` reads `DOCKER_USERNAME` from `.env` and writes `docker_config.bzl`
2. `BUILD.bazel` uses `oci_image` to layer the miner files onto a base image
3. `oci_push` uses `crane` (not Docker) to push the built image to Docker Hub
4. The miner neuron (`neurons/miner.py`) includes the image name in submissions so validators can pull it

## Relationship to Miner

When the miner submits to `/submit_kernel`, it includes a `docker_image` field:
- Default: `<DOCKER_USERNAME>/quasar-miner-gpu:latest`
- Override: Set `MINER_DOCKER_IMAGE` in `.env`

The validator pulls this image and runs it in a sandboxed container to verify the miner's logits match a reference model.
