# The MIT License (MIT)
# Copyright 2026 SILX INC
#
# GPU Normalization for QUASAR-SUBNET
#
# Normalizes validator-measured TPS to a reference GPU baseline so that
# different validator hardware produces comparable scores.
#
#   normalized_tps = measured_tps / gpu_factor
#
# A factor > 1.0 means the validator GPU is *faster* than the reference,
# so the raw TPS is scaled down.  A factor < 1.0 means it is slower, so
# the raw TPS is scaled up.

import os
import json
import subprocess
from typing import Dict, Optional, Tuple

# Reference GPU against which all normalization factors are defined.
REFERENCE_GPU = "NVIDIA GeForce RTX 5090"

# Default normalization factors for common GPUs used in Bittensor subnets.
# These are approximate relative throughput multipliers for memory-bandwidth-
# bound linear-attention kernels.  Override via GPU_NORMALIZATION_FACTORS
# env var (JSON string) to fine-tune for your workload.
#
# Factor semantics:  factor = gpu_throughput / reference_throughput
#   RTX 5090 @ 1.00 means "this IS the reference"
#   H100     @ 1.30 means "H100 is ~30% faster than RTX 5090"
#   A100     @ 0.70 means "A100 is ~30% slower than RTX 5090"
DEFAULT_GPU_FACTORS: Dict[str, float] = {
    # ── Blackwell (sm_120) ──
    "NVIDIA GeForce RTX 5090": 1.00,
    "NVIDIA RTX 5090": 1.00,
    "NVIDIA RTX PRO 6000 Blackwell": 1.10,
    "NVIDIA RTX 6000 Pro": 1.10,
    # ── Hopper (sm_90) ──
    "NVIDIA H100 80GB HBM3": 1.30,
    "NVIDIA H100": 1.30,
    "NVIDIA H200": 1.45,
    # ── Ada Lovelace (sm_89) ──
    "NVIDIA GeForce RTX 4090": 0.65,
    "NVIDIA RTX 4090": 0.65,
    "NVIDIA GeForce RTX 4080 SUPER": 0.48,
    "NVIDIA GeForce RTX 4080": 0.45,
    "NVIDIA RTX 6000 Ada Generation": 0.60,
    "NVIDIA L40S": 0.55,
    "NVIDIA L40": 0.50,
    # ── Ampere (sm_80 / sm_86) ──
    "NVIDIA A100 80GB PCIe": 0.70,
    "NVIDIA A100-SXM4-80GB": 0.75,
    "NVIDIA A100-SXM4-40GB": 0.70,
    "NVIDIA A100-PCIE-40GB": 0.65,
    "NVIDIA A100": 0.70,
    "NVIDIA A6000": 0.45,
    "NVIDIA A40": 0.40,
    "NVIDIA GeForce RTX 3090": 0.40,
    "NVIDIA RTX 3090": 0.40,
    "NVIDIA GeForce RTX 3090 Ti": 0.42,
    "NVIDIA GeForce RTX 3080 Ti": 0.35,
    "NVIDIA GeForce RTX 3080": 0.32,
}


def _load_custom_factors() -> Dict[str, float]:
    """Load user-supplied GPU normalization factors from env var (JSON)."""
    raw = os.environ.get("GPU_NORMALIZATION_FACTORS", "").strip()
    if not raw:
        return {}
    try:
        custom = json.loads(raw)
        if not isinstance(custom, dict):
            return {}
        return {str(k): float(v) for k, v in custom.items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def get_gpu_factors() -> Dict[str, float]:
    """Return the merged normalization table (defaults + env overrides)."""
    factors = dict(DEFAULT_GPU_FACTORS)
    factors.update(_load_custom_factors())
    return factors


def detect_gpu_name() -> Optional[str]:
    """Detect the primary GPU name via PyTorch CUDA or nvidia-smi fallback."""
    # Method 1: PyTorch (preferred — matches what the sandbox actually uses)
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass

    # Method 2: nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        if out:
            return out.split("\n")[0].strip()
    except Exception:
        pass

    return None


def _fuzzy_match(gpu_name: str, factors: Dict[str, float]) -> Optional[float]:
    """Try substring matching if exact key lookup fails."""
    name_lower = gpu_name.lower()
    for key, factor in factors.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return factor
    return None


def get_normalization_factor(
    gpu_name: Optional[str] = None,
) -> Tuple[float, str, str]:
    """
    Return (factor, detected_gpu_name, matched_key) for the validator's GPU.

    If the GPU is not in the table, falls back to 1.0 (assumes reference-class
    performance) and prints a warning.
    """
    override = os.environ.get("GPU_NORMALIZATION_FACTOR", "").strip()
    if override:
        try:
            factor = float(override)
            detected = gpu_name or detect_gpu_name() or "unknown"
            return (factor, detected, "ENV_OVERRIDE")
        except ValueError:
            pass

    if gpu_name is None:
        gpu_name = detect_gpu_name()

    if gpu_name is None:
        print(
            "[GPU-NORM] WARNING: Could not detect GPU. "
            "Using factor 1.0 (reference baseline). "
            "Set GPU_NORMALIZATION_FACTOR env var to override."
        )
        return (1.0, "unknown", "UNDETECTED")

    factors = get_gpu_factors()

    # Exact match
    if gpu_name in factors:
        return (factors[gpu_name], gpu_name, gpu_name)

    # Fuzzy match
    fuzzy = _fuzzy_match(gpu_name, factors)
    if fuzzy is not None:
        matched = next(
            k for k in factors if _fuzzy_match(gpu_name, {k: factors[k]}) is not None
        )
        print(
            f"[GPU-NORM] Fuzzy matched '{gpu_name}' -> '{matched}' (factor={fuzzy:.2f})"
        )
        return (fuzzy, gpu_name, matched)

    print(
        f"[GPU-NORM] WARNING: GPU '{gpu_name}' not in normalization table. "
        f"Using factor 1.0. Add it to GPU_NORMALIZATION_FACTORS or set "
        f"GPU_NORMALIZATION_FACTOR to override."
    )
    return (1.0, gpu_name, "UNKNOWN")


def normalize_tps(measured_tps: float, factor: float) -> float:
    """Normalize measured TPS to the reference GPU baseline."""
    if factor <= 0:
        return measured_tps
    return measured_tps / factor
