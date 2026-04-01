<div align="center">

## QUASAR-SUBNET

**Long-context kernel optimization & inference verification subnet on Bittensor**

[![QUASAR](./banner.png)](https://github.com/SILX-LABS/QUASAR-SUBNET)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

| | |
|---|---|
| **Subnet UID** | `24` (mainnet) / `383` (testnet) |
| **Network** | `finney` |
| **Validator API** | `https://quasar-validator-api.onrender.com` |
| **Reference Model** | `Qwen/Qwen3-4B-Instruct-2507` |

</div>

---

- [Overview](#overview)
- [Validator Operation](#validator-operation)
  - [Requirements](#validator-requirements)
  - [Setup](#validator-setup)
  - [Running](#running-the-validator)
- [Miner Operation](#miner-operation)
  - [Requirements](#miner-requirements)
  - [Setup](#miner-setup)
  - [Running](#running-the-miner)
  - [BYOC Mode (Bring Your Own Code)](#byoc-mode-bring-your-own-code)
- [Reward Distribution](#reward-distribution)
- [Architecture](#architecture)
- [Key Environment Variables](#key-environment-variables)
- [GPU Hardware & Hosting](#gpu-hardware--hosting)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

QUASAR-SUBNET is a Bittensor subnet where miners compete to optimize **flash-linear attention kernels** and validators verify performance via sandboxed benchmarks and logit-level inference checks.

**Miners** fork a target repository, optimize kernel code, and submit results. The validator measures actual throughput (tokens/sec) in a sandboxed Docker container and verifies correctness via logit comparison against a reference model.

**Validators** pull competition weights from the centralized Validator API and set them on the Bittensor chain. They also run the sandboxed benchmark and logit verification pipeline for pending submissions.

---

## Validator Operation

### Validator Requirements

- **GPU**: RTX 5090 or RTX 6000 Pro recommended (48GB VRAM ideal)
- **Docker**: Full Docker daemon with NVIDIA Container Toolkit (not RunPod)
- **Bittensor wallet**: Registered on subnet 24 with sufficient stake
- **OS**: Ubuntu 22.04+ recommended

### Validator Setup

```bash
# 1. Clone and install
git clone https://github.com/SILX-LABS/QUASAR-SUBNET
cd QUASAR-SUBNET
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 2. Configure environment
cp .env.example .env
```

Edit `.env` with your values:

```bash
NETUID=24
SUBTENSOR_NETWORK=finney
WALLET_VALIDATOR_NAME=your_validator_wallet
WALLET_HOTKEY=default
VALIDATOR_API_URL=https://quasar-validator-api.onrender.com
ENABLE_LOGIT_VERIFICATION=true
REFERENCE_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

```bash
# 3. Install NVIDIA Container Toolkit (if not already installed)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# 4. Verify GPU passthrough
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi

# 5. Build the sandbox image (one-time)
docker build -t quasar-sandbox:latest -f validator/Dockerfile.sandbox .
```

### Running the Validator

```bash
./START_VALIDATOR.sh
```

This runs `neurons/validator.py` which, on each cycle:

1. Syncs metagraph UIDs with the Validator API
2. Fetches pending submissions and validates them (clone repo, benchmark in sandbox, verify logits)
3. Pulls competition weights from `/get_weights` (completed round or active round interim data)
4. Caches received weights locally
5. Commits weights on-chain every **6 hours** (configurable via `WEIGHT_COMMIT_INTERVAL_HOURS`)
6. Sleeps for the polling interval (default 300s) and repeats

**Interim weight commits**: Bittensor's `activity_cutoff` (~5000 blocks / ~16.7h) requires validators to set weights regularly. Since competition rounds last 48 hours, the validator automatically re-commits the last valid weights every 6 hours to stay active. Weights are committed immediately when a new round completes, and then repeated on the 6-hour cadence until the next round finishes.

The validator needs its hotkey to be authorized on the Validator API. Contact the subnet owner to have your hotkey added to the production `VALIDATOR_HOTKEYS`.

---

## Miner Operation

### Miner Requirements

- **GPU**: RTX 5090 recommended (CUDA required for kernel optimization)
- **GitHub account**: Personal Access Token with `repo` scope
- **Docker Hub account**: For pushing the inference server image
- **Bittensor wallet**: Registered on subnet 24

### Miner Setup

```bash
# 1. Clone and install
git clone https://github.com/SILX-LABS/QUASAR-SUBNET
cd QUASAR-SUBNET
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 2. Configure environment
cp .env.example .env
```

Edit `.env` with your values:

```bash
NETUID=24
SUBTENSOR_NETWORK=finney
WALLET_MINER_NAME=your_miner_wallet
WALLET_HOTKEY=default
GITHUB_TOKEN=ghp_your_token_here
GITHUB_USERNAME=your_github_username
DOCKER_USERNAME=your_dockerhub_username
TARGET_SEQUENCE_LENGTH=100000
```

```bash
# 3. Build and push Docker image for logit verification (one-time)
cd docker-build && bash push_miner.sh && cd ..
```

Without the Docker image, logit verification will fail and your submissions will not rank.

### Running the Miner

```bash
# Terminal 1: Start the inference server (for logit verification)
./START_MINER_INFERENCE.sh

# Terminal 2: Start the miner neuron
./START_MINER.sh
```

The miner neuron will:

1. Fork the target repository (`flash-linear-attention`) to your GitHub
2. Run optimization loops on kernel code using an LLM agent
3. Benchmark each iteration locally
4. Submit results (commit hash, TPS, Docker image) to the Validator API
5. Wait for the optimization interval (default 300s) and repeat

### BYOC Mode (Bring Your Own Code)

If you have your own optimized kernel code, skip the LLM agent and submit directly:

**BYOC Reference** — LLM uses your code as a starting point:

```bash
BYOC_FILE_PATH=./my_chunk.py ./START_MINER.sh
```

**BYOC Direct** — Submit your code directly (no LLM, no GPU needed for code gen):

```bash
# Single file
BYOC_DIRECT=true BYOC_FILE_PATH=./my_chunk.py ./START_MINER.sh

# Directory with multiple target files
BYOC_DIRECT=true BYOC_DIR=./my_kernels/ ./START_MINER.sh
```

Target files: `chunk.py`, `fused_recurrent.py`, `gate.py`, `forward_substitution.py`, `chunk_intra_token_parallel.py`, `__init__.py`

Your code **must** include these imports or the validator will reject it with score 0.0:

```python
from fla.utils import autocast_custom_bwd
from fla.utils import autocast_custom_fwd
from fla.utils import autotune_cache_kwargs
from fla.utils import check_shared_mem
from fla.utils import input_guard
```

---

## Reward Distribution

Competition runs in **48-hour evaluation rounds**. During each round, validators independently benchmark and verify miner submissions. Ranking is based on **validator-measured TPS** (not miner-claimed). The top 4 submissions receive rewards:

| Rank | Reward Share |
|:----:|:-----------:|
| 1st | 60% |
| 2nd | 25% |
| 3rd | 10% |
| 4th | 5% |

**Ranking criteria** (in order):

1. Logit verification must pass (mandatory)
2. Weighted score = `validated_tokens_per_sec × league_multiplier` (descending)
3. Submission timestamp (first submission wins ties)

**League multipliers** by target sequence length:

| Sequence Length | League | Multiplier |
|:-:|:-:|:-:|
| ≤ 100K | 100k | 0.5× |
| ≤ 500K | 500k | 1.5× |
| ≤ 1M | 1M | 3.0× |

### Weight Commit Cadence

While evaluation rounds last 48 hours, validators **commit weights on-chain every 6 hours** to stay within Bittensor's `activity_cutoff` (5000 blocks / ~16.7h):

```
Round N completes  ─► commit Round N weights (immediate)
      +6h          ─► re-commit Round N weights
      +12h         ─► re-commit Round N weights
      ...
      +48h         ─► Round N+1 completes ─► commit Round N+1 weights
```

This ensures validators remain `ACTIVE` in the metagraph at all times. If the API is temporarily unreachable, the validator repeats its last cached weights. The interval is configurable via `WEIGHT_COMMIT_INTERVAL_HOURS` (default `6`).

### GPU Normalization in Scoring

Since validators and miners run on different GPUs, raw TPS values differ across hardware. Validator-measured TPS is **normalized to a reference GPU baseline** (RTX 5090) so that rankings are hardware-independent. The miner's self-reported TPS is informational only — ranking is entirely based on what the validator measures and normalizes.

---

## Architecture

```
┌─────────────┐     submit      ┌──────────────────┐     get_weights     ┌───────────────┐
│    Miner     │ ──────────────► │   Validator API   │ ◄────────────────── │   Validator    │
│ neurons/     │                 │ validator_api/     │                     │ neurons/       │
│ miner.py     │                 │ app.py            │ ──── set_weights ──►│ validator.py   │
└─────────────┘                 └──────────────────┘      (on-chain)      └───────────────┘
       │                                │                                         │
       │ push docker image              │ PostgreSQL                              │ sandbox benchmark
       ▼                                │ (Supabase)                              │ + logit verification
  Docker Hub                            │                                         ▼
                                        └───────────────────────────────── Docker sandbox
```

- **Validator API** (`validator_api/app.py`): Central FastAPI service managing rounds, submissions, rankings, and weights. Deployed on Render with PostgreSQL.
- **Miner Neuron** (`neurons/miner.py`): Optimizes kernels, benchmarks locally, submits to API.
- **Validator Neuron** (`neurons/validator.py`): Validates submissions in Docker sandbox, verifies logits, pulls weights from API, and commits weights on-chain every 6 hours (interim repeat-weights to stay within activity_cutoff).
- **Miner Inference Server** (`miner/inference_server.py`): HTTP server for logit verification (`POST /inference`, `GET /health`).

---

## Key Environment Variables

| Variable | Default | Used By | Description |
|---|---|---|---|
| `NETUID` | `24` | All | Subnet UID |
| `SUBTENSOR_NETWORK` | `finney` | All | `finney` (mainnet) or `test` |
| `VALIDATOR_API_URL` | `https://quasar-validator-api.onrender.com` | All | Central API endpoint |
| `WALLET_MINER_NAME` | `quasar_miner` | Miner | Bittensor wallet name |
| `WALLET_VALIDATOR_NAME` | `quasar_validator` | Validator | Bittensor wallet name |
| `WALLET_HOTKEY` | `default` | All | Hotkey name |
| `GITHUB_TOKEN` | — | Miner | GitHub PAT with `repo` scope |
| `GITHUB_USERNAME` | — | Miner | GitHub username |
| `DOCKER_USERNAME` | — | Miner | Docker Hub username |
| `TARGET_SEQUENCE_LENGTH` | `100000` | Miner | Optimization target (affects league) |
| `ENABLE_LOGIT_VERIFICATION` | `true` | Validator | Must be `true` for production |
| `REFERENCE_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Validator | Model for logit comparison |
| `POLLING_INTERVAL` | `300` | Validator | Seconds between validation cycles |
| `WEIGHT_COMMIT_INTERVAL_HOURS` | `6` | Validator | Hours between on-chain weight commits (keep under ~16h) |
| `GPU_NORMALIZATION_FACTOR` | auto | Validator | Override auto-detected GPU factor |

See `.env.example` for the full list with documentation.

---

## GPU Hardware & Hosting

### Recommended GPUs

| Role | GPU | Notes |
|---|---|---|
| **Miner** | RTX 5090 | Optimize kernels for Blackwell architecture |
| **Validator** | RTX 5090 / RTX 6000 Pro | 48GB VRAM ideal for reference model + sandbox |

### GPU Normalization

Validator-measured TPS is normalized to a reference baseline so different hardware produces comparable scores:

| GPU | Factor |
|---|:---:|
| RTX 5090 | 1.00 (reference) |
| RTX 6000 Pro (Blackwell) | 1.10 |
| H100 80GB | 1.30 |
| RTX 4090 | 0.65 |
| A100 80GB SXM | 0.75 |

Override with `GPU_NORMALIZATION_FACTOR` in `.env`.

### Hosting Compatibility

| Provider | Miner | Validator | Notes |
|---|:---:|:---:|---|
| **Vast.ai** | ✅ | ✅ | Full Docker; sync clock with NTP |
| **Lambda Labs** | ✅ | ✅ | Full Docker and GPU support |
| **AWS EC2** | ✅ | ✅ | Full VM with Docker |
| **GCP GPU VMs** | ✅ | ✅ | Full VM with Docker |
| **RunPod** | ✅ | ❌ | No Docker daemon — miners use Bazel for image push |
| **Targon (SN4)** | ✅ | ❌ | Kubernetes pods lack full Docker support (containerd shim + docker-proxy issues) |

Validators **require** `docker run` with GPU passthrough and port mapping for sandbox benchmarking and logit verification. RunPod and Targon do not fully support this — use Vast.ai, Lambda Labs, or a full VM provider instead.

### Ports

| Service | Port | Env Var |
|---|:---:|---|
| Validator API | `8000` | `PORT` |
| Miner Inference Server | `8001` | `MINER_INFERENCE_PORT` |

---

## Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for detailed solutions including:

- Docker daemon errors on RunPod
- Clock drift on Vast.ai causing 401 authentication errors
- NVIDIA Container Toolkit setup
- Sandbox container errors

---

## Contributing

We welcome contributions:

1. Fork the repo and create a feature branch
2. Make your changes with tests where applicable
3. Open a PR with a clear description

---

## License

QUASAR-SUBNET is released under the [MIT License](./LICENSE).
