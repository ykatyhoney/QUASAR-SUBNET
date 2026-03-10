<div align="center">

## QUASAR-SUBNET

**Long-context kernel optimization & inference verification subnet on Bittensor**

[![QUASAR](./banner.png)](https://github.com/SILX-LABS/QUASAR-SUBNET)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

- [Overview](#overview)
- [Architecture](#architecture)
  - [Validator API](#validator-api)
  - [Miner Neuron](#miner-neuron)
    - [BYOC Mode (Bring Your Own Code)](#byoc-mode-bring-your-own-code)
  - [Validator Neuron](#validator-neuron)
  - [Miner Inference Server](#miner-inference-server)
- [Quick Start](#quick-start)
  - [Environment Setup](#environment-setup)
  - [Miner Setup (RunPod or any GPU)](#miner-setup-runpod-or-any-gpu)
  - [Validator Setup (Vast.ai, Lambda, AWS — Docker required)](#validator-setup-vastai-lambda-aws--docker-required)
  - [Ports & Conflicts](#ports--conflicts)
- [Docker & Image Publishing](#docker--image-publishing)
  - [Miner: Build & Push with Bazel (RunPod)](#miner-build--push-with-bazel-runpod)
  - [Miner: Manual Docker Build (Local)](#miner-manual-docker-build-local)
  - [Validator: Sandbox Image](#validator-sandbox-image)
- [Key Environment Variables](#key-environment-variables)
- [GPU Hosting Compatibility](#gpu-hosting-compatibility)
- [Required Imports (Critical!)](#required-imports-critical)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

QUASAR-SUBNET is a Bittensor subnet that:

- Optimizes **flash-linear attention kernels** via miner submissions
- Uses a centralized **Validator API** to orchestrate rounds, commit–reveal, and scoring
- Verifies miner behavior via a **logit-level inference verification** protocol
- Supports both **local miners** and **Docker-based miner inference servers**

Miners:
- Fork and optimize a target repository (e.g. `troy12x/flash-linear-attention`)
- Run kernel benchmarks and report results to the Validator API
- Expose an HTTP inference server (locally or via Docker) for logit verification
- Build and push a Docker image to Docker Hub so validators can verify logits

Validators:
- Coordinate benchmarking rounds through the Validator API
- Run miner kernel code inside a **sandboxed Docker container** for performance testing
- Verify miner outputs using **logit verification** against a reference model
- Submit final weights to the Bittensor network

---

## Architecture

### Validator API

Location: `validator_api/`
Entry point: `validator_api/app.py`

- Central FastAPI service used by miners and validators
- Stores benchmark results, commit–reveal state, and miner metadata
- Backed by PostgreSQL or SQLite via `DATABASE_URL`
- Deployed in production using Docker/Render (see `render.yaml` and `Dockerfile`)

Local startup:

```bash
./START_SERVER.sh
```

Health check:

```bash
curl http://localhost:8000/health
```

### Miner Neuron

Location: `neurons/miner.py`
Role: Bittensor neuron that:

- Connects to the Bittensor network (`--subtensor.network`, `--netuid`)
- Forks the target GitHub repo using `GITHUB_TOKEN` and `GITHUB_USERNAME`
- Runs optimization loops on `chunk.py` and related kernels
- Submits results (including `docker_image`) to the Validator API
- Optionally runs in **test mode** for local development

Typical local run (using `.env`):

```bash
./START_MINER.sh
```

Under the hood this is equivalent to:

```bash
python -m neurons.miner \
  --wallet.name "$WALLET_MINER_NAME" \
  --wallet.hotkey "$WALLET_HOTKEY" \
  --subtensor.network "$SUBTENSOR_NETWORK" \
  --netuid "$NETUID"
```

> **Note**: The miner requires `GITHUB_TOKEN` (with `repo` scope) and `GITHUB_USERNAME` in `.env`.

#### Docker Image for Logit Verification

Miners **must** build and push a Docker image containing their inference server so validators can verify logits. Set `DOCKER_USERNAME` in `.env` and the miner will automatically include `<DOCKER_USERNAME>/quasar-miner-gpu:latest` as the `docker_image` in submissions. Without this, logit verification will fail.

Build & push:

```bash
cd docker-build && bash push_miner.sh
```

#### BYOC Mode (Bring Your Own Code)

Two BYOC modes are available:

**BYOC Reference** — LLM uses your code as a reference to guide optimization:

```bash
BYOC_FILE_PATH=./my_chunk.py ./START_MINER.sh
```

**BYOC Direct** — Skip the LLM entirely, submit your code directly (no GPU needed for code gen):

```bash
# Single file
BYOC_DIRECT=true BYOC_FILE_PATH=./my_chunk.py ./START_MINER.sh

# Directory with multiple target files
BYOC_DIRECT=true BYOC_DIR=./my_kernels/ ./START_MINER.sh
```

BYOC Direct will: copy your files into the fork, validate imports, run benchmarks, commit+push to GitHub, and submit to the Validator API.

**Tips**:

- Make sure your code includes all [required imports](#required-imports-critical) or the validator will reject with score 0.0
- Use absolute paths for `BYOC_FILE_PATH` / `BYOC_DIR`
- Target files: `chunk.py`, `fused_recurrent.py`, `gate.py`, `forward_substitution.py`, `chunk_intra_token_parallel.py`, `__init__.py`

### Validator Neuron

Location: `neurons/validator.py`
Role: Bittensor neuron that:

- Polls the Validator API for new submissions
- Validates miner code, imports, and kernel behavior
- Runs benchmark tasks inside a **sandboxed Docker container** (`quasar-sandbox:latest`)
- Verifies miner outputs using **logit verification** against a reference model
- Queries the actual Bittensor chain for block numbers (with time-based fallback)

**Prerequisites** (on the validator machine — requires full Docker support):

```bash
# 1. Build the sandbox image for miner code benchmarking
docker build -t quasar-sandbox:latest -f validator/Dockerfile.sandbox .

# 2. Install NVIDIA Container Toolkit (if not already installed)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# 3. Verify GPU passthrough works
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi
```

Without the sandbox image, all performance tests will fail with `pull access denied`.

Local run (using `.env`):

```bash
./START_VALIDATOR.sh
```

### Miner Inference Server

Location (Python): `miner/inference_server.py`
Location (Docker build): `docker-build/` and `miner/Dockerfile.inference`

This server exposes the **logit verification** interface:

- `POST /inference` – runs generation & returns captured logits at multiple decoding steps
- `GET /health` – health check
- `GET /model_info` – basic metadata

Local dev startup (bound to port 8001 to avoid conflicts):

```bash
./START_MINER_INFERENCE.sh
```

Containerized startup (built image):

```bash
docker run -d --gpus all -p 8001:8000 \
  -e MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
  your_dockerhub_username/quasar-miner-gpu:latest
```

---

## Quick Start

### Environment Setup

```bash
git clone https://github.com/SILX-LABS/QUASAR-SUBNET
cd QUASAR-SUBNET

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**Set up environment variables**:

```bash
cp .env.example .env
# Edit .env and fill in your values (see Key Environment Variables below)
```

### Miner Setup (RunPod or any GPU)

Miners do **not** need Docker running. They run Python directly and use Bazel to push images.

```bash
# Terminal 1: Start the miner inference server
./START_MINER_INFERENCE.sh

# Terminal 2: Start the miner neuron
./START_MINER.sh

# One-time: Build & push Docker image for logit verification
cd docker-build && bash push_miner.sh
```

### Validator Setup (Vast.ai, Lambda, AWS — Docker required)

Validators **must** run on a machine with full Docker support (not RunPod).

```bash
# One-time setup: Build sandbox image
docker build -t quasar-sandbox:latest -f validator/Dockerfile.sandbox .

# Terminal 1: Start the Validator API
./START_SERVER.sh

# Terminal 2: Start the validator neuron
./START_VALIDATOR.sh
```

### Ports & Conflicts

| Service                 | Default Port | Notes                              |
|-------------------------|:------------:|------------------------------------|
| Validator API           | `8000`       | Set via `PORT` env var             |
| Miner Inference Server  | `8001`       | Set via `MINER_INFERENCE_PORT`     |
| Challenge container     | `8080`       | From `docker-compose.yml`          |
| Miner neuron axon       | auto/`8091`  | Set via `--axon.port`              |
| Validator neuron axon   | auto/`8092`  | Set via `--axon.port`              |

If you see `address already in use`, check which process is using the port (`lsof -i :8000`).

---

## Docker & Image Publishing

### Miner: Build & Push with Bazel (RunPod)

Bazel uses `crane` to push images directly to Docker Hub — **no Docker daemon required**. This works on RunPod and any environment.

1. Set `DOCKER_USERNAME` in `.env`
2. Set up Docker Hub credentials:

   ```bash
   # Create ~/.docker/config.json for crane authentication
   mkdir -p ~/.docker
   echo 'YOUR_TOKEN' | docker login -u YOUR_USERNAME --password-stdin 2>/dev/null || \
     printf '{"auths":{"https://index.docker.io/v1/":{"auth":"%s"}}}' \
       "$(echo -n 'YOUR_USERNAME:YOUR_TOKEN' | base64)" > ~/.docker/config.json
   ```

3. Install Bazel (if not installed):

   ```bash
   wget https://github.com/bazelbuild/bazelisk/releases/download/v1.28.1/bazelisk-linux-amd64
   chmod +x bazelisk-linux-amd64
   sudo cp bazelisk-linux-amd64 /usr/local/bin/bazel
   ```

4. Build and push:

   ```bash
   cd docker-build
   bash push_miner.sh
   ```

This pushes:
- `index.docker.io/$DOCKER_USERNAME/quasar-miner-gpu:latest` (CUDA)
- `index.docker.io/$DOCKER_USERNAME/quasar-miner-cpu:latest` (CPU)

### Miner: Manual Docker Build (Local)

If you have Docker available (not on RunPod):

```bash
# GPU image (auto-detects CUDA version)
cd miner && bash build_inference.sh

# Or manually
docker build -f miner/Dockerfile.inference -t quasar-miner-gpu:latest miner/
docker push your_username/quasar-miner-gpu:latest
```

### Validator: Sandbox Image

The validator runs miner kernel code inside a sandboxed Docker container. Build this once on the validator machine:

```bash
docker build -t quasar-sandbox:latest -f validator/Dockerfile.sandbox .
```

The sandbox image contains Python 3.11, PyTorch (CUDA), Triton, and `flash-linear-attention`. Miner code is mounted read-only at `/workspace` at runtime.

---

## Key Environment Variables

Defined in `.env` (see `.env.example` for full documentation):

| Variable                    | Default / Example                        | Used By      |
|-----------------------------|------------------------------------------|--------------|
| `DATABASE_URL`              | `postgresql://...` or `sqlite:///...`    | Validator API |
| `VALIDATOR_API_URL`         | `https://quasar-validator-api.onrender.com` | All         |
| `GITHUB_TOKEN`              | `ghp_...`                                | Miner        |
| `GITHUB_USERNAME`           | `your_username`                          | Miner        |
| `DOCKER_USERNAME`           | `your_dockerhub_username`                | Miner        |
| `MINER_DOCKER_IMAGE`        | (auto: `<DOCKER_USERNAME>/quasar-miner-gpu:latest`) | Miner |
| `NETUID`                    | `24`                                     | All neurons  |
| `SUBTENSOR_NETWORK`         | `finney` / `test`                        | All neurons  |
| `WALLET_MINER_NAME`         | `quasar_miner`                           | Miner        |
| `WALLET_VALIDATOR_NAME`     | `quasar_validator`                       | Validator    |
| `WALLET_HOTKEY`             | `default`                                | All neurons  |
| `VALIDATOR_HOTKEYS`         | `hotkey1,hotkey2,...`                     | Validator API |
| `ENABLE_LOGIT_VERIFICATION` | `true`                                   | Validator    |
| `REFERENCE_MODEL`           | `Qwen/Qwen3-4B-Instruct-2507`           | Validator    |
| `COSINE_SIM_THRESHOLD`      | `0.99`                                   | Validator    |
| `MAX_ABS_DIFF_THRESHOLD`    | `0.1`                                    | Validator    |
| `VALIDATOR_SANDBOX_IMAGE`   | `quasar-sandbox:latest`                  | Validator    |
| `MODEL_NAME`                | `Qwen/Qwen3-4B-Instruct-2507`           | Inference    |
| `TARGET_SEQUENCE_LENGTH`    | `100000`                                 | Miner        |

---

## GPU Hosting Compatibility

| Provider | Miner | Validator | Notes |
|----------|:-----:|:---------:|-------|
| **RunPod** | ✅ | ❌ | No Docker daemon — miners use Bazel to push, run Python directly |
| **Vast.ai** | ✅ | ✅ | Full Docker support; sync clock with NTP (see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)) |
| **Lambda Labs** | ✅ | ✅ | Full Docker and GPU support |
| **AWS EC2 (GPU)** | ✅ | ✅ | Full VM with Docker |
| **GCP GPU VMs** | ✅ | ✅ | Full VM with Docker |
| **Paperspace Core** | ✅ | ✅ | VM-level access with Docker |

**Key constraint**: Validators require `docker run` for sandboxed performance testing and logit verification. RunPod pods cannot run Docker containers (iptables blocked).

---

## Required Imports (Critical!)

The validator checks for these **MANDATORY** imports in `chunk.py`. Missing any of these will cause validation to fail with score 0.0:

```python
from fla.utils import autocast_custom_bwd
from fla.utils import autocast_custom_fwd
from fla.utils import autotune_cache_kwargs
from fla.utils import check_shared_mem
from fla.utils import input_guard
```

If you're using BYOC mode, ensure your code includes all these imports.

---

## Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for detailed solutions to common issues including:

- Docker daemon errors on RunPod
- Clock drift on Vast.ai causing 401 authentication errors
- NVIDIA Container Toolkit setup for GPU passthrough
- Sandbox container errors and debugging

---

## Contributing

We welcome contributions of all kinds:

- Improving kernel optimization strategies
- Enhancing validator scoring and metrics
- Extending the inference verification pipeline
- Docker and deployment improvements
- Documentation and testing

To contribute:

1. Fork the repo and create a feature branch
2. Make your changes with tests where applicable
3. Open a PR with a clear description and rationale

---

## License

QUASAR-SUBNET is released under the [MIT License](./LICENSE).
