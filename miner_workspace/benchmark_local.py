#!/usr/bin/env python3
"""
QUASAR-SUBNET Local Benchmark
==============================
Mirrors exactly what the validator runs to measure your TPS.

The validator test script (neurons/validator.py _build_test_script):
  - batch_size=1, hidden_size=512, head_dim=64, num_heads=8
  - 3 warmup runs, 10 timed runs
  - TPS = (batch_size * seq_len * num_runs) / elapsed
  - Reports RESULT: <tps> and VRAM_MB: <vram>

This script replicates that exact measurement plus additional diagnostics.

Usage:
    python3 benchmark_local.py                    # Default: 100k seq_len
    python3 benchmark_local.py --seq-len 500000   # Custom seq_len
    python3 benchmark_local.py --all-leagues      # Test all leagues
    python3 benchmark_local.py --stacked          # Multi-layer benchmark
"""

import sys
import os
import time
import types
import argparse
import json

# Suppress triton noise for cleaner output
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
# Reduce CUDA memory fragmentation — critical for 900k+ sequence lengths
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# Add script directory to path for quasar_import
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from quasar_import import import_quasar_attention as _import_quasar_attention


# ── League multipliers (same as validator_api/app.py) ────────────────────────

LEAGUE_MULTIPLIERS = {
    "100k": 0.5,
    "200k": 0.75,
    "300k": 1.0,
    "400k": 1.25,
    "500k": 1.5,
    "600k": 1.75,
    "700k": 2.0,
    "800k": 2.25,
    "900k": 2.5,
    "1M": 3.0,
}


def get_league(seq_len: int) -> str:
    if seq_len >= 1_000_000:
        return "1M"
    elif seq_len >= 900_000:
        return "900k"
    elif seq_len >= 800_000:
        return "800k"
    elif seq_len >= 700_000:
        return "700k"
    elif seq_len >= 600_000:
        return "600k"
    elif seq_len >= 500_000:
        return "500k"
    elif seq_len >= 400_000:
        return "400k"
    elif seq_len >= 300_000:
        return "300k"
    elif seq_len >= 200_000:
        return "200k"
    return "100k"


# ── Validator-equivalent benchmark ───────────────────────────────────────────

def run_validator_benchmark(seq_len: int, verbose: bool = True) -> dict:
    """
    Run the EXACT benchmark the validator uses.

    Validator test script parameters (from neurons/validator.py:_build_test_script):
      - batch_size = 1
      - hidden_size = 512
      - head_dim = 64
      - num_heads = 8
      - warmup = 3 iterations
      - timed = 10 iterations
      - TPS = (batch_size * seq_len * num_runs) / elapsed
    """
    QuasarAttention = _import_quasar_attention()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: Running on CPU. Results will not match validator (GPU required).")

    batch_size = 1
    hidden_size = 512
    head_dim = 64
    num_heads = 8

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Validator-Equivalent Benchmark")
        print(f"  seq_len={seq_len:,}  batch={batch_size}  hidden={hidden_size}")
        print(f"  heads={num_heads}  head_dim={head_dim}")
        print(f"{'='*60}")

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        if verbose:
            print(f"  GPU memory: {free_mem:.2f} GB free")

    # Initialize model
    quasar = QuasarAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        mode="chunk",
        use_short_conv=True,
    ).to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Warmup (3 iterations, same as validator)
    if verbose:
        print("  Warmup: 3 iterations...", end=" ", flush=True)
    for _ in range(3):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
            _ = quasar(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if verbose:
        print("done")

    # Timed runs (10 iterations, same as validator)
    num_runs = 10
    if verbose:
        print(f"  Benchmark: {num_runs} iterations...", end=" ", flush=True)

    start = time.time()
    for _ in range(num_runs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
            _ = quasar(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens_per_sec = (batch_size * seq_len * num_runs) / elapsed

    # VRAM
    vram_mb = 0.0
    if device.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    if verbose:
        print("done")

    # League info
    league = get_league(seq_len)
    multiplier = LEAGUE_MULTIPLIERS.get(league, 1.0)
    weighted_score = tokens_per_sec * multiplier

    result = {
        "seq_len": seq_len,
        "tokens_per_sec": tokens_per_sec,
        "vram_mb": vram_mb,
        "elapsed_sec": elapsed,
        "league": league,
        "multiplier": multiplier,
        "weighted_score": weighted_score,
    }

    if verbose:
        print(f"\n  --- Results ---")
        print(f"  RESULT: {tokens_per_sec:.2f}")
        print(f"  VRAM_MB: {vram_mb:.2f}")
        print(f"  Elapsed: {elapsed:.3f}s ({elapsed/num_runs:.4f}s per run)")
        print(f"  League: {league} (multiplier: {multiplier}x)")
        print(f"  Weighted Score: {weighted_score:.2f}")
        print(f"  (This is the score validators use for ranking)")

    # Cleanup
    del x, quasar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    return result


def run_stacked_benchmark(seq_len: int, n_layers: int = 16, hidden_size: int = 1536):
    """Run multi-layer stacked benchmark (matches test_quasar_stacked_benchmark)."""
    QuasarAttention = _import_quasar_attention()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_heads = max(4, hidden_size // 128)
    head_dim = hidden_size // num_heads
    batch_size = 1

    print(f"\n{'='*60}")
    print(f"  Stacked Benchmark ({n_layers} layers)")
    print(f"  seq_len={seq_len:,}  hidden={hidden_size}  heads={num_heads}")
    print(f"{'='*60}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"  GPU memory: {free_mem:.2f} GB free")

    layers = torch.nn.ModuleList([
        QuasarAttention(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            mode="chunk",
            use_short_conv=True,
        )
        for _ in range(n_layers)
    ]).to(device)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Forward-only benchmark
    for mode_name, mode_bwd in [("fwd-only", False), ("fwd+bwd", True)]:
        if mode_bwd:
            x = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)

        # Warmup
        try:
            for _ in range(3):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
                    y = x
                    for layer in layers:
                        y, _, _ = layer(y)
                    loss = y.float().mean()
                if mode_bwd:
                    loss.backward()
                    for p in layers.parameters():
                        if p.grad is not None:
                            p.grad = None
                    if x.grad is not None:
                        x.grad = None
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM during {mode_name} warmup, skipping")
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()

        runs = 5
        t0 = time.time()
        try:
            for _ in range(runs):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
                    y = x
                    for layer in layers:
                        y, _, _ = layer(y)
                    loss = y.float().mean()
                if mode_bwd:
                    loss.backward()
                    for p in layers.parameters():
                        if p.grad is not None:
                            p.grad = None
                    if x.grad is not None:
                        x.grad = None
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM during {mode_name} benchmark, skipping")
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        step_s = (t1 - t0) / runs
        tok_s = (batch_size * seq_len) / step_s if step_s > 0 else 0
        print(f"  {mode_name}: {step_s:.4f}s/step  {tok_s:.0f} tok/s")

    del layers, x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all_leagues():
    """Benchmark across all league sequence lengths to find optimal league."""
    seq_lens = [100_000, 200_000, 300_000, 400_000, 500_000,
                600_000, 700_000, 800_000, 900_000, 1_000_000]

    print(f"\n{'='*60}")
    print(f"  All-League Benchmark")
    print(f"  Testing {len(seq_lens)} sequence lengths")
    print(f"{'='*60}")

    results = []
    for seq_len in seq_lens:
        try:
            result = run_validator_benchmark(seq_len, verbose=False)
            results.append(result)
            league = result["league"]
            tps = result["tokens_per_sec"]
            ws = result["weighted_score"]
            print(f"  {league:>4s} ({seq_len:>10,}): {tps:>12,.0f} tok/s  x{result['multiplier']:.2f} = {ws:>14,.0f} weighted")
        except torch.cuda.OutOfMemoryError:
            print(f"  {get_league(seq_len):>4s} ({seq_len:>10,}): OOM - skipped")
        except Exception as e:
            print(f"  {get_league(seq_len):>4s} ({seq_len:>10,}): Error - {e}")

    if results:
        best = max(results, key=lambda r: r["weighted_score"])
        print(f"\n  BEST LEAGUE: {best['league']} (seq_len={best['seq_len']:,})")
        print(f"  Best weighted score: {best['weighted_score']:,.0f}")
        print(f"  TPS at best league: {best['tokens_per_sec']:,.0f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="QUASAR-SUBNET Local Benchmark")
    parser.add_argument("--seq-len", type=int, default=100_000,
                        help="Sequence length to benchmark (default: 100000)")
    parser.add_argument("--all-leagues", action="store_true",
                        help="Test all league sequence lengths")
    parser.add_argument("--stacked", action="store_true",
                        help="Run stacked multi-layer benchmark")
    parser.add_argument("--n-layers", type=int, default=16,
                        help="Number of layers for stacked benchmark (default: 16)")
    parser.add_argument("--hidden-size", type=int, default=1536,
                        help="Hidden size for stacked benchmark (default: 1536)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    print("QUASAR-SUBNET Local Benchmark")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {total_mem:.1f} GB")

    if args.all_leagues:
        results = run_all_leagues()
        if args.json:
            print(json.dumps(results, indent=2, default=str))
    elif args.stacked:
        run_stacked_benchmark(args.seq_len, args.n_layers, args.hidden_size)
    else:
        result = run_validator_benchmark(args.seq_len)
        if args.json:
            print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
