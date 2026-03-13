# The MIT License (MIT)
# Copyright 2026 SILX INC
#
# Inference Verification Module for QUASAR-SUBNET
# Based on const's qllm/quasar.py architecture
#
# This module implements:
# - Reference model for logit comparison
# - Logit verification (cosine similarity + max absolute diff)
# - Container execution via Docker SDK for miner submissions
# - Throughput-based scoring with verification gate

"""
Inference verification for QUASAR-SUBNET.

Miners submit Docker images exposing an inference server. Validators run
containers, measuring generation throughput while verifying correctness
by comparing logits at a random decode step.

Container interface: POST /inference
  Request:  {prompt: List[int], gen_len: int, logits_at_step: int, logits_at_steps: List[int]}
  Response: {tokens: List[int], captured_logits_multi: Dict[int, List[float]], elapsed_sec: float}

Verification: cosine similarity + max absolute diff on captured logits
Scoring: throughput (tok/sec) if verified, infinity if failed
Leader: epsilon-dominance, winner-take-all weights
"""

import os
import asyncio
import random
import numpy as np
import torch
import time
import hashlib
import socket
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ CONFIGURATION                                                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class InferenceVerificationConfig:
    """Configuration for inference verification."""

    # Network configuration
    netuid: int = int(os.environ.get("NETUID", 383))

    # Reference model (DeepSeek-V3.2 for better code understanding and verification)
    reference_model: str = os.environ.get("REFERENCE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

    # Inference configuration
    prompt_length: int = 128  # Random prompt token length
    generate_length: int = 512  # Number of tokens to generate
    logit_capture_range: Tuple[int, int] = (1, 50)  # Legacy single-step range (unused)
    num_logit_checks: int = 3  # Number of random steps to capture logits at
    inference_timeout: int = 300  # Timeout in seconds

    # Verification thresholds (from const's implementation)
    cosine_sim_threshold: float = 0.99  # Minimum cosine similarity
    max_abs_diff_threshold: float = 0.1  # Maximum absolute difference

    # Commit-reveal timing
    blocks_until_reveal: int = 100  # ~20 minutes (100 blocks * 12s/block)
    block_time: int = 12  # Bittensor block time in seconds

    # Scoring parameters
    epsilon: float = 0.01  # Epsilon for dominance comparison
    tempo: int = 360  # Weight update interval in blocks

    # Defaults
    default_wallet: str = os.environ.get("WALLET_NAME", "default")
    default_hotkey: str = os.environ.get("HOTKEY_NAME", "default")
    default_network: str = os.environ.get("NETWORK", "finney")


# Global config instance
CONFIG = InferenceVerificationConfig()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ LOGGING                                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def log(msg: str, level: str = "info"):
    """Colored logging output."""
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {
        "info": "\033[36m\u25b8\033[0m",      # Cyan
        "success": "\033[32m\u2713\033[0m",   # Green
        "error": "\033[31m\u2717\033[0m",     # Red
        "warn": "\033[33m\u26a0\033[0m",      # Yellow
        "start": "\033[33m\u2192\033[0m",     # Yellow arrow
    }
    print(f"\033[90m{ts}\033[0m {colors.get(level, ' ')} {msg}")


def log_header(title: str):
    """Log a section header."""
    sep = "\u2500" * 60
    print(f"\n\033[1m{sep}\033[0m")
    print(f"\033[1m{title}\033[0m")
    print(f"\033[1m{sep}\033[0m\n")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ VERIFICATION                                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class VerificationResult:
    """Result of logit verification."""
    verified: bool
    cosine_sim: Optional[float] = None
    max_abs_diff: Optional[float] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified": self.verified,
            "cosine_sim": self.cosine_sim,
            "max_abs_diff": self.max_abs_diff,
            "reason": self.reason
        }


def verify_logits(
    miner_logits: List[float],
    reference_logits: List[float],
    cosine_threshold: float = CONFIG.cosine_sim_threshold,
    max_diff_threshold: float = CONFIG.max_abs_diff_threshold
) -> VerificationResult:
    """
    Compare miner's logits against reference within tolerance.

    Uses cosine similarity + max absolute difference as verification metrics.
    This handles numerical instability that would cause divergence with
    direct logit comparison.
    """
    miner = np.array(miner_logits, dtype=np.float32)
    reference = np.array(reference_logits, dtype=np.float32)

    if miner.shape != reference.shape:
        return VerificationResult(
            verified=False,
            reason=f"shape_mismatch: miner={miner.shape}, reference={reference.shape}"
        )

    norm_m = np.linalg.norm(miner)
    norm_r = np.linalg.norm(reference)

    if norm_m < 1e-9 or norm_r < 1e-9:
        return VerificationResult(
            verified=False,
            reason="zero_norm: logit vectors have near-zero norm"
        )

    cosine_sim = float(np.dot(miner, reference) / (norm_m * norm_r))
    max_abs_diff = float(np.max(np.abs(miner - reference)))
    verified = (cosine_sim >= cosine_threshold) and (max_abs_diff <= max_diff_threshold)

    return VerificationResult(
        verified=verified,
        cosine_sim=cosine_sim,
        max_abs_diff=max_abs_diff,
        reason=None if verified else f"threshold_failed: cosine={cosine_sim:.4f}, max_diff={max_abs_diff:.4f}"
    )


def compute_kl_divergence(
    miner_logits: List[float],
    reference_logits: List[float],
    temperature: float = 1.0
) -> float:
    """
    Compute KL divergence between miner and reference logit distributions.

    Alternative verification method:
    "validator gives zero score if KL > epsilon"
    """
    import torch.nn.functional as F

    miner_t = torch.tensor(miner_logits, dtype=torch.float32)
    reference_t = torch.tensor(reference_logits, dtype=torch.float32)

    miner_probs = F.softmax(miner_t / temperature, dim=-1)
    reference_probs = F.softmax(reference_t / temperature, dim=-1)

    kl_div = F.kl_div(
        miner_probs.log(),
        reference_probs,
        reduction='sum'
    )

    return float(kl_div)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ REFERENCE MODEL                                                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class ReferenceModel:
    """
    Reference model for inference verification.

    Runs the honest base model to produce ground-truth logits for comparison
    with miner outputs.
    """

    def __init__(self, model_name: str = CONFIG.reference_model):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False

    async def load(self):
        """Load the reference model and tokenizer."""
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        log(f"Loading reference model: {self.model_name}", "info")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True

        log(f"Reference model loaded on {self.device}", "success")

    async def inference(
        self,
        prompt: List[int],
        gen_len: int,
        logits_at_step: int,
        logits_at_steps: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference and capture logits at one or more steps.

        Args:
            logits_at_step:  Legacy single-step capture (used if logits_at_steps is None).
            logits_at_steps: List of 1-indexed steps to capture logits at.
        """
        if not self._loaded:
            await self.load()

        capture_set = set(logits_at_steps) if logits_at_steps else {logits_at_step}

        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prompt], device=device)

        generated_tokens = []
        captured_logits = None
        captured_logits_multi: Dict[int, List[float]] = {}
        past_key_values = None

        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            for step in range(gen_len):
                step_1indexed = step + 1
                if step_1indexed in capture_set:
                    logits_list = next_token_logits[0].cpu().float().tolist()
                    captured_logits_multi[step_1indexed] = logits_list
                    if step_1indexed == (logits_at_steps[0] if logits_at_steps else logits_at_step):
                        captured_logits = logits_list

                next_token = next_token_logits.argmax(dim=-1)
                generated_tokens.append(next_token.item())

                outputs = self.model(
                    next_token.unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]

        elapsed_sec = time.perf_counter() - start_time

        return {
            "tokens": generated_tokens,
            "captured_logits": captured_logits,
            "captured_logits_multi": captured_logits_multi,
            "elapsed_sec": elapsed_sec,
        }

    def get_vocab_size(self) -> int:
        """Get vocabulary size for logit dimension verification."""
        if self.tokenizer is None:
            return 0
        return len(self.tokenizer)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ PROMPT GENERATION                                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def generate_random_prompt(length: int, vocab_size: int = 32000) -> List[int]:
    """
    Generate a random prompt for inference verification.

    Using random tokens ensures miners can't pre-compute responses
    and must actually run the model.
    """
    return [random.randint(10, vocab_size - 1) for _ in range(length)]


def generate_verification_challenge(
    reference_model: ReferenceModel,
    config: InferenceVerificationConfig = CONFIG
) -> Dict[str, Any]:
    """Generate a verification challenge for a miner.

    Captures logits at multiple random steps spread across the full generation
    range so miners can't stop generating early or skip intermediate tokens.
    """
    vocab_size = reference_model.get_vocab_size() or 32000
    gen_len = config.generate_length

    # Pick N distinct steps uniformly across [1, gen_len-1].
    # One early, rest spread through the full range -- miners must run the
    # entire generation to pass all checks.
    n_checks = max(1, config.num_logit_checks)
    all_steps = list(range(1, gen_len))
    capture_steps = sorted(random.sample(all_steps, min(n_checks, len(all_steps))))

    return {
        "prompt": generate_random_prompt(config.prompt_length, vocab_size),
        "gen_len": gen_len,
        "logits_at_step": capture_steps[0],
        "logits_at_steps": capture_steps,
    }


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ CONTAINER EXECUTION (Docker SDK)                                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

CONTAINER_INTERNAL_PORT = 8000
CONTAINER_STARTUP_TIMEOUT = int(os.environ.get("CONTAINER_STARTUP_TIMEOUT", "120"))
CONTAINER_PULL_TIMEOUT = int(os.environ.get("CONTAINER_PULL_TIMEOUT", "600"))
CONTAINER_GPU_ENABLED = os.environ.get("CONTAINER_GPU_ENABLED", "true").lower() == "true"
CONTAINER_MEMORY_LIMIT = os.environ.get("CONTAINER_MEMORY_LIMIT", "16g")
CONTAINER_SHM_SIZE = os.environ.get("CONTAINER_SHM_SIZE", "2g")


@dataclass
class ContainerInferenceResult:
    """Result from running inference in a miner's container."""
    success: bool
    tokens: List[int] = field(default_factory=list)
    captured_logits: Optional[List[float]] = None
    captured_logits_multi: Dict[int, List[float]] = field(default_factory=dict)
    elapsed_sec: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "tokens": self.tokens,
            "captured_logits": self.captured_logits,
            "elapsed_sec": self.elapsed_sec,
            "error": self.error
        }


def _find_free_port() -> int:
    """Find a free TCP port on the host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_container_health(
    host_port: int,
    timeout: int = CONTAINER_STARTUP_TIMEOUT,
    poll_interval: float = 2.0,
) -> bool:
    """Poll the container's /health endpoint until it responds or timeout."""
    import requests as _requests

    url = f"http://127.0.0.1:{host_port}/health"
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            r = _requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "healthy":
                    return True
        except Exception as e:
            last_err = e
        time.sleep(poll_interval)
    if last_err:
        log(f"Last health-check error: {last_err}", "warn")
    return False


def run_container_inference(
    hotkey: str,
    docker_image: str,
    prompt: List[int],
    gen_len: int,
    logits_at_step: int,
    logits_at_steps: Optional[List[int]] = None,
    timeout: int = CONFIG.inference_timeout,
    gpu_enabled: bool = CONTAINER_GPU_ENABLED,
) -> ContainerInferenceResult:
    """
    Run inference on a miner's Docker container via the Docker SDK.

    Lifecycle:
        1. Pull the image (if not cached locally).
        2. Create / reuse an internal Docker network (no internet access).
        3. Start a hardened container (read-only, cap-drop ALL, pids limit).
        4. Wait for /health to report "healthy".
        5. POST /inference with the challenge payload.
        6. Parse the response into a ContainerInferenceResult.
        7. Always stop and remove the container in the finally block.

    The miner container must expose a FastAPI server on port 8000 with:
        POST /inference  -> {tokens, captured_logits_multi: {step: logits}, elapsed_sec}
        GET  /health     -> {status: "healthy", ...}

    The container MUST return captured_logits_multi with logits for every
    step listed in the request's logits_at_steps array. Returning only the
    legacy single captured_logits field will fail verification.
    """
    import requests as _requests

    try:
        import docker as _docker
    except ImportError:
        log("docker SDK not installed - run: pip install docker", "error")
        return ContainerInferenceResult(
            success=False, error="docker python SDK not installed"
        )

    container = None
    client = None
    host_port = _find_free_port()

    try:
        client = _docker.from_env()

        # 1. Pull image
        log(f"Pulling image {docker_image} for {hotkey[:12]}...", "start")
        try:
            client.images.pull(docker_image)
            log(f"Image ready: {docker_image}", "success")
        except _docker.errors.ImageNotFound:
            return ContainerInferenceResult(
                success=False, error=f"image not found: {docker_image}"
            )
        except _docker.errors.APIError as e:
            return ContainerInferenceResult(
                success=False, error=f"docker pull failed: {e}"
            )

        # 2. Create / reuse an internal Docker network.
        #    "internal=True" blocks all outbound internet access while
        #    still allowing the validator to reach the container via the
        #    mapped port on the host.
        _INTERNAL_NET_NAME = "quasar-verify-internal"
        try:
            verify_network = client.networks.get(_INTERNAL_NET_NAME)
        except _docker.errors.NotFound:
            verify_network = client.networks.create(
                _INTERNAL_NET_NAME,
                driver="bridge",
                internal=True,
            )
            log(f"Created internal Docker network: {_INTERNAL_NET_NAME}", "info")

        # 3. Start container with security hardening
        run_kwargs: Dict[str, Any] = {
            "image": docker_image,
            "detach": True,
            "auto_remove": False,
            "ports": {f"{CONTAINER_INTERNAL_PORT}/tcp": ("127.0.0.1", host_port)},
            "environment": {"DEVICE": "cuda" if gpu_enabled else "cpu"},
            "mem_limit": CONTAINER_MEMORY_LIMIT,
            "shm_size": CONTAINER_SHM_SIZE,
            "labels": {
                "quasar.hotkey": hotkey,
                "quasar.role": "miner-verification",
            },
            "name": f"quasar-verify-{hotkey[:12]}-{int(time.time())}",
            # --- Security hardening ---
            # Internal network: container can serve on the mapped port
            # but CANNOT reach the internet or LAN.
            "network": verify_network.name,
            "cap_drop": ["ALL"],
            # Re-add only the minimum capabilities needed for GPU,
            # serving on the mapped port, and common container runtimes
            # (apt, user switching, file ownership in tmpfs mounts).
            "cap_add": [
                "SYS_PTRACE",  # CUDA runtimes (stacktrace, debugger attach)
                "SETUID",      # seteuid — apt, gosu, su (switch to non-root)
                "SETGID",      # setegid/setgroups — apt privilege separation
                "CHOWN",       # chown in writable tmpfs dirs (apt, pip)
                "FOWNER",      # operations on files regardless of owner (pip cache)
                "DAC_OVERRIDE",  # bypass read/write/execute permission checks (apt partial dirs)
            ],
            "read_only": True,
            "security_opt": ["no-new-privileges"],
            "pids_limit": 1024,
            # Writable dirs the inference server / runtime may need.
            # The root filesystem stays read-only for security; only
            # these tmpfs mounts are writable.
            "tmpfs": {
                "/tmp": "size=2G",
                "/root/.cache": "size=4G",
                "/var/lib/apt/lists": "size=64M",
                "/var/cache/apt": "size=256M",
                "/var/log": "size=64M",
                "/var/tmp": "size=256M",
                "/run": "size=64M",
                "/home": "size=1G",
            },
        }
        if gpu_enabled:
            run_kwargs["device_requests"] = [
                _docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])
            ]

        log(f"Starting container on host port {host_port}...", "start")
        container = client.containers.run(**run_kwargs)

        # 4. Wait for health
        log("Waiting for container health...", "info")
        healthy = _wait_for_container_health(host_port, timeout=CONTAINER_STARTUP_TIMEOUT)
        if not healthy:
            logs_tail = ""
            try:
                logs_tail = container.logs(tail=40).decode(errors="replace")
            except Exception:
                pass
            return ContainerInferenceResult(
                success=False,
                error=(
                    f"container did not become healthy within "
                    f"{CONTAINER_STARTUP_TIMEOUT}s. Logs:\n{logs_tail}"
                ),
            )
        log("Container healthy", "success")

        # 5. Call /inference -- validator measures wall-clock time independently
        inference_url = f"http://127.0.0.1:{host_port}/inference"
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "gen_len": gen_len,
            "logits_at_step": logits_at_step,
        }
        if logits_at_steps:
            payload["logits_at_steps"] = logits_at_steps
        log(f"Calling /inference (timeout={timeout}s)...", "info")
        wall_start = time.perf_counter()
        resp = _requests.post(inference_url, json=payload, timeout=timeout)
        wall_elapsed = time.perf_counter() - wall_start
        if resp.status_code != 200:
            return ContainerInferenceResult(
                success=False,
                error=f"/inference returned HTTP {resp.status_code}: {resp.text[:500]}",
            )

        data = resp.json()
        miner_claimed_sec = float(data.get("elapsed_sec", 0))
        # Use validator-measured wall time, not the miner's self-reported value.
        # Wall time includes HTTP overhead (~ms), but that's a conservative
        # penalty the miner can't game. Log the discrepancy for auditing.
        if miner_claimed_sec > 0 and wall_elapsed > 0:
            ratio = miner_claimed_sec / wall_elapsed
            if ratio < 0.5:
                log(f"Miner claimed {miner_claimed_sec:.2f}s but wall time was {wall_elapsed:.2f}s "
                    f"(ratio {ratio:.2f}) — likely inflated speed", "warn")

        # Parse multi-step logits if the container supports it
        captured_logits_multi: Dict[int, List[float]] = {}
        raw_multi = data.get("captured_logits_multi")
        if isinstance(raw_multi, dict):
            captured_logits_multi = {int(k): v for k, v in raw_multi.items()}

        return ContainerInferenceResult(
            success=True,
            tokens=data.get("tokens", []),
            captured_logits=data.get("captured_logits"),
            captured_logits_multi=captured_logits_multi,
            elapsed_sec=wall_elapsed,
        )

    except Exception as e:
        log(f"Container execution failed for {hotkey[:8]}...: {e}", "error")
        return ContainerInferenceResult(success=False, error=str(e))
    finally:
        if container is not None:
            try:
                container.stop(timeout=10)
            except Exception:
                pass
            try:
                container.remove(force=True)
            except Exception:
                pass
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ SCORING & LEADER SELECTION                                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class MinerEvaluation:
    """Evaluation result for a single miner."""
    hotkey: str
    block: int
    docker_image: str
    score: float  # 1/throughput (lower is better) or inf if failed
    verified: bool
    throughput: float = 0.0  # tokens/sec
    verification: Optional[VerificationResult] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hotkey": self.hotkey,
            "block": self.block,
            "docker_image": self.docker_image,
            "score": self.score,
            "verified": self.verified,
            "throughput": self.throughput,
            "verification": self.verification.to_dict() if self.verification else None,
            "error": self.error
        }


async def evaluate_miner(
    hotkey: str,
    docker_image: str,
    reference: ReferenceModel,
    config: InferenceVerificationConfig = CONFIG
) -> MinerEvaluation:
    """
    Evaluate a miner: verify correctness + measure throughput.

    1. Generate a random prompt
    2. Run inference on miner's container (synchronous Docker call)
    3. Run inference on reference model
    4. Compare logits at random step
    5. Calculate throughput if verified
    """
    challenge = generate_verification_challenge(reference, config)
    prompt = challenge["prompt"]
    gen_len = challenge["gen_len"]
    logits_at_step = challenge["logits_at_step"]

    log(f"  prompt_len={len(prompt)}, gen_len={gen_len}, capture_step={logits_at_step}", "info")

    # run_container_inference is synchronous (Docker SDK); run in thread to
    # avoid blocking the event loop if called from an async context.
    loop = asyncio.get_event_loop()
    miner_result = await loop.run_in_executor(
        None,
        lambda: run_container_inference(hotkey, docker_image, prompt, gen_len, logits_at_step),
    )

    if not miner_result.success:
        return MinerEvaluation(
            hotkey=hotkey,
            block=0,
            docker_image=docker_image,
            score=float("inf"),
            verified=False,
            error=miner_result.error
        )

    reference_result = await reference.inference(prompt, gen_len, logits_at_step)

    if miner_result.captured_logits is None:
        log("  Miner did not return captured logits", "error")
        return MinerEvaluation(
            hotkey=hotkey,
            block=0,
            docker_image=docker_image,
            score=float("inf"),
            verified=False,
            error="no_logits"
        )

    verification = verify_logits(
        miner_result.captured_logits,
        reference_result["captured_logits"]
    )

    log(f"  cosine={verification.cosine_sim:.4f}, max_diff={verification.max_abs_diff:.4f}", "info")

    if not verification.verified:
        log("  FAILED verification", "error")
        return MinerEvaluation(
            hotkey=hotkey,
            block=0,
            docker_image=docker_image,
            score=float("inf"),
            verified=False,
            verification=verification
        )

    elapsed = miner_result.elapsed_sec
    if elapsed <= 0:
        return MinerEvaluation(
            hotkey=hotkey,
            block=0,
            docker_image=docker_image,
            score=float("inf"),
            verified=True,
            error="invalid_elapsed"
        )

    throughput = gen_len / elapsed
    score = 1.0 / throughput if throughput > 0 else float("inf")

    log(f"  Throughput: {throughput:.1f} tok/sec", "success")

    return MinerEvaluation(
        hotkey=hotkey,
        block=0,
        docker_image=docker_image,
        score=score,
        verified=True,
        throughput=throughput,
        verification=verification
    )


def beats(
    evaluations: Dict[str, MinerEvaluation],
    i: str,
    j: str,
    epsilon: float = CONFIG.epsilon
) -> bool:
    """Check if miner i beats miner j with epsilon-dominance."""
    if i not in evaluations or j not in evaluations:
        return False
    return evaluations[i].score < evaluations[j].score - epsilon


def select_leader(
    evaluations: Dict[str, MinerEvaluation],
    epsilon: float = CONFIG.epsilon
) -> Optional[str]:
    """
    Select leader using epsilon-dominance.

    The leader is the verified miner that is not dominated by any other
    verified miner. Tie-breaker: earliest submission block.
    """
    if not evaluations:
        return None

    verified = [hk for hk in evaluations if evaluations[hk].verified]
    if not verified:
        return None

    candidates = []
    for hk in verified:
        dominated = any(beats(evaluations, other, hk, epsilon) for other in verified if other != hk)
        if not dominated:
            candidates.append(hk)

    if not candidates:
        return None

    return min(candidates, key=lambda hk: evaluations[hk].block)


def calculate_weights(
    evaluations: Dict[str, MinerEvaluation],
    hotkeys: List[str]
) -> Dict[str, float]:
    """
    Calculate weights for all miners (winner-take-all).

    The leader gets 100% of the weight, everyone else gets 0%.
    """
    leader = select_leader(evaluations)

    weights = {}
    for hk in hotkeys:
        if hk in evaluations:
            weights[hk] = 1.0 if hk == leader else 0.0
        else:
            weights[hk] = 0.0

    return weights


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║ COMMIT-REVEAL HELPERS                                                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def compute_commitment_hash(data: str, salt: bytes = None) -> str:
    """Compute commitment hash for commit-reveal scheme."""
    if salt is None:
        salt = os.urandom(32)

    content = salt + data.encode()
    return hashlib.sha256(content).hexdigest()


def get_reveal_block(current_block: int, blocks_until_reveal: int = CONFIG.blocks_until_reveal) -> int:
    """Calculate the block at which commitment will be revealed."""
    return current_block + blocks_until_reveal


def estimate_reveal_time_minutes(blocks_until_reveal: int = CONFIG.blocks_until_reveal) -> int:
    """Estimate time until reveal in minutes."""
    return (blocks_until_reveal * CONFIG.block_time) // 60
