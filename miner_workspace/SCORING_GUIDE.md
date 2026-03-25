# QUASAR-SUBNET Scoring & Strategy Guide

## How You Get Scored (TL;DR)

Your **weighted_score** determines your rank:

```
weighted_score = validated_tokens_per_sec × league_multiplier
```

Top 4 miners get rewards: **60% / 25% / 10% / 5%**

## League Multipliers

| League | Seq Length | Multiplier | Strategy |
|--------|-----------|-----------|----------|
| 100k   | ≤100,000  | 0.5×      | Easiest, lowest reward |
| 200k   | ≤200,000  | 0.75×     | |
| 300k   | ≤300,000  | 1.0×      | Baseline |
| 400k   | ≤400,000  | 1.25×     | |
| 500k   | ≤500,000  | 1.5×      | |
| 600k   | ≤600,000  | 1.75×     | |
| 700k   | ≤700,000  | 2.0×      | |
| 800k   | ≤800,000  | 2.25×     | |
| 900k   | ≤900,000  | 2.5×      | |
| 1M     | ≥1,000,000| 3.0×      | Hardest, 6x reward vs 100k |

**Key insight**: If you get 10,000 TPS at 100k (score=5,000), you need only 1,667 TPS at 1M to match (score=5,001). Higher leagues are almost always better if your kernel handles them.

## What Validators Actually Measure

The validator runs THIS exact script in a Docker sandbox:

```python
batch_size = 1
seq_len = <your_target_sequence_length>
hidden_size = 512
head_dim = 64
num_heads = 8

quasar = QuasarAttention(
    hidden_size=512, head_dim=64, num_heads=8,
    mode="chunk", use_short_conv=True
).to("cuda")

x = torch.randn(1, seq_len, 512, device="cuda")

# 3 warmup runs
for _ in range(3):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = quasar(x)

# 10 timed runs
start = time.time()
for _ in range(10):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = quasar(x)
torch.cuda.synchronize()
elapsed = time.time() - start

tokens_per_sec = (1 * seq_len * 10) / elapsed
```

## Requirements to Pass Validation

### 1. Required Imports (chunk.py ONLY)
```python
from fla.utils import autocast_custom_bwd
from fla.utils import autocast_custom_fwd
from fla.utils import autotune_cache_kwargs
from fla.utils import check_shared_mem
from fla.utils import input_guard
```
Missing ANY = score 0.0

### 2. Forbidden Imports (ALL files)
```python
from fla.ops.gla  # FORBIDDEN
from fla.ops.kda  # FORBIDDEN
```

### 3. Forbidden Calls (ALL .py files under fla/)
exec(), eval(), compile(), os.system(), subprocess.run(), etc.

### 4. Forbidden Modules (ALL .py files under fla/)
importlib, ctypes, socket, http, urllib, requests, paramiko, fabric

### 5. Logit Verification (MANDATORY)
Your Docker image must produce logits matching the reference model (Qwen/Qwen3-4B-Instruct-2507) with:
- Cosine similarity ≥ 0.99
- Max absolute difference ≤ 0.1

### 6. Over-Claim Detection
If your claimed TPS is >50% higher than validator-measured AND the absolute difference >1,000 tok/s, you get **flagged** and excluded from ALL future rounds.

## Files You Can Optimize

```
fla/ops/quasar/
├── chunk.py                        # Main chunked attention (HIGHEST IMPACT)
├── chunk_intra_token_parallel.py   # Intra-token parallel processing
├── forward_substitution.py         # Forward substitution kernel
├── fused_recurrent.py              # Fused recurrent computation
├── gate.py                         # Gating mechanism
└── __init__.py                     # Module initialization
```

## Optimization Strategies

### 1. Triton Kernel Optimization (chunk.py)
- Optimize block sizes for your GPU architecture
- Reduce memory transfers between global/shared memory
- Improve tiling patterns for better GPU occupancy
- Use autotune to find optimal configurations

### 2. Memory Efficiency
- Reduce peak VRAM usage (allows larger batch/seq)
- Optimize intermediate tensor allocation
- Use in-place operations where safe

### 3. Computation Fusion
- Fuse sequential operations into single kernels
- Reduce kernel launch overhead
- Minimize synchronization points

### 4. League Strategy
Use `python3 benchmark_local.py --all-leagues` to find your best league.
The optimal league depends on your GPU — higher leagues give better multipliers
but may have lower raw TPS.

## Local Testing Workflow

```bash
# 1. Setup
bash setup_miner.sh

# 2. Get baseline benchmark
python3 benchmark_local.py

# 3. Edit kernel files
vim flash-linear-attention/fla/ops/quasar/chunk.py

# 4. Re-install and benchmark
cd flash-linear-attention && pip install -e . && cd ..
python3 benchmark_local.py

# 5. Validate (checks imports, security, etc.)
python3 validate_submission.py --benchmark

# 6. Find best league
python3 benchmark_local.py --all-leagues

# 7. Submit (when ready)
python3 submit_miner.py --seq-len 100000 --network test --dry-run
python3 submit_miner.py --seq-len 100000 --network test
```

## GPU Normalization

Validator TPS is normalized to RTX 5090 baseline:
- RTX 5090: 1.00× (reference)
- H100: 1.30× (your TPS ÷ 1.30)
- RTX 4090: 0.65× (your TPS ÷ 0.65 = higher normalized)
- A100: 0.70×

This means slower GPUs get a boost in normalized scoring.

## Tiebreaker Rules

When weighted scores are equal:
1. **Earlier submission wins** (submit fast!)
2. **Lower submission ID wins** (tiebreaker)

## Competition Rounds

- Each round lasts **48 hours**
- Previous winner becomes baseline for next round
- New submissions must **exceed the baseline weighted score**
- Only **validated, logit-verified, revealed** submissions rank
