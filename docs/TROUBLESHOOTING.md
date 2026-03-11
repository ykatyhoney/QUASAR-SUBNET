# Troubleshooting Guide

Common issues and solutions for running QUASAR-SUBNET miners and validators.

---

## Table of Contents

- [RunPod: Docker Not Available](#runpod-docker-not-available)
- [Vast.ai: Clock Drift Causing 401 Errors](#vastai-clock-drift-causing-401-errors)
- [Vast.ai: GPU Not Found in Docker Containers](#vastai-gpu-not-found-in-docker-containers)
- [Validator: Sandbox Image Pull Errors](#validator-sandbox-image-pull-errors)
- [Validator: Cannot Parse Test Results](#validator-cannot-parse-test-results)
- [Validator: 401 Unauthorized After Moving Machines](#validator-401-unauthorized-after-moving-machines)
- [Miner: docker_image Not Set (Logit Verification Fails)](#miner-docker_image-not-set-logit-verification-fails)
- [Miner: Bazel Push Authentication Failed](#miner-bazel-push-authentication-failed)
- [Validator: GPU Normalization Factor Not Detected](#validator-gpu-normalization-factor-not-detected)
- [General: Port Already in Use](#general-port-already-in-use)

---

## RunPod: Docker Not Available

**Symptoms:**

```
failed to start daemon: Error initializing network controller: ...
iptables v1.8.7 (nf_tables): Could not fetch rule set generation id: Permission denied
```

Or:

```
failed to mount ... operation not permitted
```

**Cause:** RunPod pods run inside Docker containers themselves. Docker-in-Docker operations (`docker run`, `docker build`, `docker daemon`) are blocked at the kernel level due to iptables and mount restrictions.

**Impact:**
- `docker run` — does not work
- `docker build` — does not work
- `bazel run //:push_miner_image_gpu` — **works** (uses `crane`, no Docker daemon needed)

**Solution for miners:**
Miners do not need Docker. Run everything directly with Python:

```bash
# Run inference server directly
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507" python3 -u miner/inference_server.py

# Run miner neuron directly
python -m neurons.miner --netuid 24 ...

# Push Docker image via Bazel (no daemon required)
cd docker-build && bash push_miner.sh
```

**Solution for validators:**
Validators cannot run on RunPod. Move to a service with full Docker support:
- **Vast.ai** (cheapest, common in Bittensor ecosystem)
- **Lambda Labs** (bare metal)
- **AWS EC2** / **GCP GPU VMs** (full VM)

See the [GPU Hosting Compatibility](./README.md#gpu-hosting-compatibility) table in README.md.

---

## Vast.ai: Clock Drift Causing 401 Errors

**Symptoms:**

```
HTTPError: 401 Client Error: Unauthorized
```

Or in API logs:

```
Signature expired (3589s old, max 120s)
```

**Cause:** Vast.ai rents GPUs from independent hosts that often lack NTP synchronization. System clocks can drift by minutes or hours. The validator API rejects signatures older than 120 seconds.

**Diagnosis:**

Run the clock checker on your Vast.ai machine:

```bash
python3 check_clock.py
```

Output showing drift:

```
https://www.google.com     drift: +3589s  (59.8 min)  [DRIFTED — will cause 401 auth errors!]
```

**Fix:**

```bash
# Option 1: Sync via HTTP (works even if NTP/UDP is blocked)
sudo date -s "$(python3 -c "
import email.utils, time
from urllib.request import urlopen, Request
r = urlopen(Request('https://www.google.com', method='HEAD'), timeout=10)
print(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(
    time.mktime(email.utils.parsedate(r.headers['Date'])) - time.timezone
)))
")"

# Option 2: NTP (if UDP port 123 is open)
apt-get update && apt-get install -y ntpdate
ntpdate -u pool.ntp.org
```

**Prevent future drift** with a cron job (re-sync every 30 minutes):

```bash
(crontab -l 2>/dev/null; echo '*/30 * * * * ntpdate -u pool.ntp.org 2>/dev/null') | crontab -
```

After syncing, restart the validator.

---

## Vast.ai: GPU Not Found in Docker Containers

**Symptoms:**

```
[VALIDATOR] Sandbox container error: 500 Server Error ... 
"failed to discover GPU vendor from CDI: no known GPU vendor found"
```

**Cause:** The NVIDIA Container Toolkit is not installed on the Vast.ai machine. Docker cannot pass GPUs into containers without it.

**Fix:**

```bash
# 1. Verify NVIDIA driver works on the host
nvidia-smi

# 2. Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit

# 3. Configure Docker to use NVIDIA runtime
nvidia-ctk runtime configure --runtime=docker

# 4. Restart Docker daemon
systemctl restart docker

# 5. Verify GPU works inside Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi
```

---

## Validator: Sandbox Image Pull Errors

**Symptoms:**

```
[VALIDATOR] Sandbox container error: 404 Client Error ...
"pull access denied for quasar-sandbox, repository does not exist"
```

**Cause:** The sandbox image (`quasar-sandbox:latest`) has not been built locally on the validator machine. It is a local-only image, not pulled from Docker Hub.

**Fix:**

```bash
cd /path/to/QUASAR-SUBNET
docker build -t quasar-sandbox:latest -f validator/Dockerfile.sandbox .
```

Verify:

```bash
docker images | grep quasar-sandbox
```

The sandbox image contains Python 3.11, PyTorch (CUDA 12.4), Triton, and `flash-linear-attention`. It's ~10GB.

---

## Validator: Cannot Parse Test Results

**Symptoms:**

```
[VALIDATOR] Could not parse test results from output (2656 bytes)
```

**Cause:** The sandbox container ran but the test script crashed inside it. The output contains an error traceback instead of `RESULT:` / `VRAM_MB:` lines.

**Diagnosis:**

With debug logging enabled, the validator will print the container output:

```
[VALIDATOR] Could not parse test results from output (2656 bytes)
[VALIDATOR]   | Traceback (most recent call last):
[VALIDATOR]   | ...
[VALIDATOR]   | ModuleNotFoundError: No module named 'fla.layers.quasar'
```

**Common causes:**

1. **`ModuleNotFoundError: No module named 'fla.layers.quasar'`** — This module comes from the **miner's fork** of flash-linear-attention, not the base `fla` package. The miner's repo is mounted at `/workspace` and the test script adds it to `sys.path`. If the miner's repo doesn't contain `fla/layers/quasar.py`, this is a **bad submission** (correctly scored 0.0).

2. **CUDA not available inside container** — Check that NVIDIA Container Toolkit is installed (see above).

3. **Out of memory** — The sandbox has a default memory limit of 16GB (`SANDBOX_MEMORY_LIMIT`).

**If the error is from the miner's code**, this is expected behavior — the validator correctly gives score 0.0 for broken submissions.

---

## Validator: 401 Unauthorized After Moving Machines

**Symptoms:** Validator worked on Machine A, get 401 errors on Machine B with the same config.

**Checklist:**

1. **Wallet exists on the new machine?**

   ```bash
   ls ~/.bittensor/wallets/your_wallet_name/hotkeys/
   ```

   If missing, regenerate the wallet:

   ```bash
   # First: regenerate coldkey
   btcli wallet regen_coldkey --wallet.name your_wallet_name
   
   # Then: regenerate hotkey
   btcli wallet regen_hotkey --wallet.name your_wallet_name --wallet.hotkey default
   ```

   > **Important**: `regen_hotkey` requires the coldkey to exist first.

2. **Hotkey matches `VALIDATOR_HOTKEYS`?**

   ```bash
   python3 -c "
   import bittensor as bt
   w = bt.wallet(name='your_wallet_name', hotkey='default')
   print(f'Hotkey: {w.hotkey.ss58_address}')
   "
   ```

   Verify this exact string is in the `VALIDATOR_HOTKEYS` env var on the API server.

3. **API server restarted** after adding the hotkey to `VALIDATOR_HOTKEYS`? Env vars are loaded once at startup.

4. **Clock synced?** See [Clock Drift](#vastai-clock-drift-causing-401-errors) section above.

---

## Miner: docker_image Not Set (Logit Verification Fails)

**Symptoms (validator logs):**

```
[VALIDATOR] No docker_image for submission XXX - FAIL (mandatory)
```

**Cause:** The miner did not include a `docker_image` in its submission. This happens when neither `DOCKER_USERNAME` nor `MINER_DOCKER_IMAGE` is set in the miner's `.env`.

**Fix (on the miner machine):**

```bash
# In .env, set at least one:
DOCKER_USERNAME=your_dockerhub_username
# OR for a custom image name:
MINER_DOCKER_IMAGE=your_username/your-custom-image:latest
```

Then build and push the image:

```bash
cd docker-build && bash push_miner.sh
```

The miner will now include `<DOCKER_USERNAME>/quasar-miner-gpu:latest` in submissions.

---

## Miner: Bazel Push Authentication Failed

**Symptoms:**

```
Error: PUT https://index.docker.io/...: UNAUTHORIZED: authentication required
```

**Cause:** Docker Hub credentials not configured for `crane` (Bazel's push tool).

**Fix:**

```bash
mkdir -p ~/.docker

# Option 1: Use docker login (even without daemon, credential file is created)
echo 'YOUR_ACCESS_TOKEN' | docker login -u YOUR_USERNAME --password-stdin

# Option 2: Create config manually
printf '{"auths":{"https://index.docker.io/v1/":{"auth":"%s"}}}' \
  "$(echo -n 'YOUR_USERNAME:YOUR_ACCESS_TOKEN' | base64)" > ~/.docker/config.json
```

Create an access token at [Docker Hub Security Settings](https://hub.docker.com/settings/security) with **Read/Write** permissions.

---

## Validator: GPU Normalization Factor Not Detected

**Symptoms (validator logs):**

```
[GPU-NORM] WARNING: Could not detect GPU. Using factor 1.0 (reference baseline).
```

Or:

```
[GPU-NORM] WARNING: GPU 'Some Custom GPU Name' not in normalization table. Using factor 1.0.
```

**Cause:** The validator's GPU is either not visible to PyTorch/nvidia-smi, or it uses a non-standard name not in the default normalization table.

**Fix:**

1. **GPU not detected:** Ensure CUDA drivers and the NVIDIA Container Toolkit are installed:

   ```bash
   nvidia-smi                 # Should show your GPU
   python3 -c "import torch; print(torch.cuda.get_device_name(0))"  # Should print GPU name
   ```

2. **GPU not in table:** Override manually in `.env`:

   ```bash
   # Set an explicit normalization factor
   GPU_NORMALIZATION_FACTOR=0.75

   # Or add your GPU to the lookup table (JSON)
   GPU_NORMALIZATION_FACTORS='{"My Custom GPU Name": 0.75}'
   ```

   See `quasar/gpu_normalization.py` for the full default table and factor semantics.

3. **Verify the factor is applied:** Check validator startup logs for:

   ```
   [VALIDATOR] GPU normalization: NVIDIA GeForce RTX 5090 (factor=1.00, matched='NVIDIA GeForce RTX 5090')
   ```

**Factor semantics:** `factor = your_gpu_throughput / rtx5090_throughput`. An RTX 5090 has factor 1.0. An H100 (~30% faster) has factor 1.30. An A100 (~30% slower) has factor 0.70.

---

## General: Port Already in Use

**Symptoms:**

```
ERROR: [Errno 98] Address already in use
```

**Diagnosis:**

```bash
lsof -i :8000   # Check Validator API port
lsof -i :8001   # Check inference server port
```

**Fix:**

```bash
# Kill the process using the port
kill $(lsof -t -i :8000)

# Or use a different port
PORT=8002 ./START_SERVER.sh
MINER_INFERENCE_PORT=8003 ./START_MINER_INFERENCE.sh
```
