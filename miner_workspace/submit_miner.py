#!/usr/bin/env python3
"""
QUASAR-SUBNET Miner Submission
===============================
Submits your optimized kernel to the validator API after local validation.

This script:
1. Validates your code (import checks, security scan)
2. Runs the benchmark to get accurate TPS
3. Commits changes to your GitHub fork
4. Submits to the validator API

Prerequisites:
  - GITHUB_TOKEN and GITHUB_USERNAME environment variables
  - Bittensor wallet configured
  - Docker image built (for logit verification)

Usage:
    python3 submit_miner.py --wallet-name my_wallet --wallet-hotkey default
    python3 submit_miner.py --dry-run  # Validate + benchmark without submitting
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import argparse
import requests


VALIDATOR_API_URL = os.getenv(
    "VALIDATOR_API_URL", "https://quasar-validator-api.onrender.com"
)


def get_commit_hash(repo_path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path, capture_output=True, text=True
    )
    return result.stdout.strip()


def get_fork_url(repo_path: str) -> str:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo_path, capture_output=True, text=True
    )
    return result.stdout.strip()


def commit_changes(repo_path: str, message: str) -> bool:
    try:
        subprocess.run(["git", "add", "-A"], cwd=repo_path, check=True)
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path, capture_output=True, text=True
        )
        if not result.stdout.strip():
            print("  No changes to commit")
            return True
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_path, check=True
        )
        subprocess.run(["git", "push"], cwd=repo_path, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Git error: {e}")
        return False


def sign_message(message: str, wallet_path: str, hotkey_name: str) -> str:
    """Sign message with bittensor wallet."""
    try:
        import bittensor as bt
        wallet = bt.Wallet(name=os.path.basename(wallet_path), hotkey=hotkey_name)
        from cryptography.hazmat.primitives.asymmetric import ed25519
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
            wallet.hotkey.private_key[:32]
        )
        signature = private_key.sign(message.encode("utf-8"))
        return signature.hex()
    except Exception as e:
        print(f"  Warning: Could not sign message: {e}")
        return ""


def submit_to_api(
    fork_url: str,
    commit_hash: str,
    tokens_per_sec: float,
    vram_mb: float,
    target_seq_len: int,
    miner_hotkey: str,
    signature: str,
    docker_image: str = "",
    network: str = "finney",
    benchmarks: dict = None,
):
    payload = {
        "miner_hotkey": miner_hotkey,
        "fork_url": fork_url,
        "commit_hash": commit_hash,
        "target_sequence_length": target_seq_len,
        "tokens_per_sec": tokens_per_sec,
        "vram_mb": vram_mb,
        "docker_image": docker_image,
        "signature": signature,
        "network": network,
    }
    if benchmarks:
        payload["benchmarks"] = benchmarks

    timestamp = str(int(time.time()))
    headers = {
        "Content-Type": "application/json",
        "X-Miner-Hotkey": miner_hotkey,
        "X-Timestamp": timestamp,
    }

    print(f"\n  Submitting to {VALIDATOR_API_URL}/submit_kernel ...")
    print(f"  TPS: {tokens_per_sec:.2f}")
    print(f"  VRAM: {vram_mb:.2f} MB")
    print(f"  Target seq_len: {target_seq_len}")
    print(f"  Network: {network}")
    print(f"  Docker image: {docker_image or 'NOT SET (logit verification will fail!)'}")

    for attempt in range(3):
        try:
            response = requests.post(
                f"{VALIDATOR_API_URL}/submit_kernel",
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code == 429:
                detail = response.json().get("detail", "")
                print(f"  Rate limited: {detail}")
                import re
                match = re.search(r"wait (\d+) seconds", detail)
                wait = int(match.group(1)) + 5 if match else 30
                print(f"  Waiting {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code == 422:
                print(f"  Validation error: {response.json().get('detail', response.text[:500])}")
                return False

            response.raise_for_status()
            result = response.json()
            print(f"  Submission successful! ID: {result.get('submission_id')}")
            return True

        except Exception as e:
            print(f"  Attempt {attempt+1}/3 failed: {e}")
            time.sleep(10)

    print("  All submission attempts failed")
    return False


def main():
    parser = argparse.ArgumentParser(description="Submit miner optimization")
    parser.add_argument("--repo-path", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=100_000)
    parser.add_argument("--wallet-name", type=str, default="quasar_miner")
    parser.add_argument("--wallet-hotkey", type=str, default="default")
    parser.add_argument("--network", type=str, default="finney",
                        choices=["finney", "test"])
    parser.add_argument("--docker-image", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and benchmark only, don't submit")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Skip benchmark (use provided --tps value)")
    parser.add_argument("--tps", type=float, default=None,
                        help="Manual TPS value (only with --skip-benchmark)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = args.repo_path or os.path.join(script_dir, "flash-linear-attention")

    if not os.path.isdir(repo_path):
        print(f"ERROR: Repository not found at {repo_path}")
        sys.exit(1)

    print("="*60)
    print("  QUASAR-SUBNET Miner Submission")
    print("="*60)

    # Step 1: Validate
    print("\n[Step 1] Validating code...")
    sys.path.insert(0, script_dir)
    from validate_submission import validate_imports
    errors, warnings = validate_imports(repo_path)

    for w in warnings:
        print(w)

    if errors:
        print(f"\n  VALIDATION FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("  Validation PASSED")

    # Step 2: Benchmark
    if args.skip_benchmark and args.tps:
        tokens_per_sec = args.tps
        vram_mb = 0.0
        print(f"\n[Step 2] Using provided TPS: {tokens_per_sec:.2f}")
    else:
        print(f"\n[Step 2] Running benchmark (seq_len={args.seq_len})...")
        from benchmark_local import run_validator_benchmark
        result = run_validator_benchmark(args.seq_len)
        tokens_per_sec = result["tokens_per_sec"]
        vram_mb = result["vram_mb"]

    if args.dry_run:
        print(f"\n  DRY RUN complete. TPS: {tokens_per_sec:.2f}")
        print("  Remove --dry-run to submit.")
        return

    # Step 3: Commit and push
    print(f"\n[Step 3] Committing changes...")
    commit_changes(repo_path, f"optimize: TPS {tokens_per_sec:.0f} at seq_len {args.seq_len}")

    # Step 4: Get submission details
    commit_hash = get_commit_hash(repo_path)
    fork_url = get_fork_url(repo_path)
    print(f"  Commit: {commit_hash}")
    print(f"  Fork URL: {fork_url}")

    # Step 5: Get wallet info
    try:
        import bittensor as bt
        wallet = bt.Wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
        miner_hotkey = wallet.hotkey.ss58_address
        print(f"  Miner hotkey: {miner_hotkey}")
    except Exception as e:
        print(f"  ERROR: Could not load wallet: {e}")
        sys.exit(1)

    # Step 6: Sign and submit
    docker_image = args.docker_image or os.getenv("MINER_DOCKER_IMAGE", "")
    if not docker_image:
        docker_username = os.getenv("DOCKER_USERNAME", "")
        if docker_username:
            docker_image = f"{docker_username}/quasar-miner-gpu:latest"

    signature_data = f"{fork_url}{commit_hash}{tokens_per_sec}{{}}"
    signature = sign_message(signature_data, args.wallet_name, args.wallet_hotkey)

    print(f"\n[Step 4] Submitting...")
    submit_to_api(
        fork_url=fork_url,
        commit_hash=commit_hash,
        tokens_per_sec=tokens_per_sec,
        vram_mb=vram_mb,
        target_seq_len=args.seq_len,
        miner_hotkey=miner_hotkey,
        signature=signature,
        docker_image=docker_image,
        network=args.network,
    )


if __name__ == "__main__":
    main()
