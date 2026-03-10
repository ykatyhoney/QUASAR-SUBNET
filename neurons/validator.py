# The MIT License (MIT)
# Copyright 2026 SILX INC

import os
import sys
import time
import asyncio
import subprocess
import tempfile
import torch
import numpy as np
import bittensor as bt
import traceback
import requests
import shutil
import json
from typing import List, Dict, Optional

# Add the parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import quasar
from quasar.base.validator import BaseValidatorNeuron
from quasar.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
)
from quasar.utils.context_builder import (
    build_full_context,
    validate_repo_structure,
    estimate_context_tokens,
)

# --- Constants ---
VALIDATOR_API_URL = os.getenv(
    "VALIDATOR_API_URL", "https://quasar-validator-api.onrender.com"
)


class PerformanceValidator:
    """Validates miner performance claims by cloning repos and running tests."""

    # Required imports that must be present in chunk.py (critical for validation)
    # These are the imports that validators check - missing any will cause score 0.0
    REQUIRED_IMPORTS_CHUNK_PY = [
        "from fla.utils import autocast_custom_bwd",
        "from fla.utils import autocast_custom_fwd",
        "from fla.utils import autotune_cache_kwargs",
        "from fla.utils import check_shared_mem",
        "from fla.utils import input_guard",
    ]

    # Optional imports that are typically present but not strictly required
    OPTIONAL_IMPORTS = [
        "import torch",
        "import torch.nn.functional as F",
        "import triton",
        "import triton.language as tl",
        "from fla.ops.utils.index import prepare_chunk_indices",
        "from fla.ops.quasar.forward_substitution import forward_substitution_kernel",
        "from fla.utils import IS_AMD",
    ]

    # Forbidden imports that must NOT be present
    FORBIDDEN_IMPORTS = [
        "from fla.ops.gla",
        "from fla.ops.kda",
        "import fla.ops.gla",
        "import fla.ops.kda",
    ]

    # Dangerous built-ins / modules that must never appear in miner code.
    # Checked via AST so dynamic tricks (string concat, getattr) in source
    # are caught at the call-site level.
    DANGEROUS_CALLS = {
        "__import__",
        "exec",
        "eval",
        "compile",
        "os.system",
        "os.popen",
        "os.exec",
        "os.execl",
        "os.execle",
        "os.execlp",
        "os.execlpe",
        "os.execv",
        "os.execve",
        "os.execvp",
        "os.execvpe",
        "os.spawn",
        "os.spawnl",
        "os.spawnle",
        "subprocess.run",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.check_output",
        "subprocess.check_call",
    }
    DANGEROUS_IMPORT_MODULES = {
        "importlib",
        "ctypes",
        "socket",
        "http",
        "urllib",
        "requests",
        "paramiko",
        "fabric",
    }

    def __init__(self, validator_instance=None):
        """
        Initialize PerformanceValidator.

        Args:
            validator_instance: Optional reference to the main Validator instance
                               for accessing logit verification methods.
        """
        self.validator_api_url = VALIDATOR_API_URL
        self.temp_dir = tempfile.mkdtemp(prefix="quasar_validator_")
        self.validator_instance = validator_instance  # Reference to main Validator for logit verification
        print(f"[VALIDATOR] Initialized with temp dir: {self.temp_dir}")

    # --- AST helpers -----------------------------------------------------------

    @staticmethod
    def _ast_imported_names(tree) -> set:
        """Return all imported module/name strings from an AST tree."""
        import ast

        names: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names.add(module)
                for alias in node.names:
                    names.add(
                        f"{module}.{alias.name}" if module else alias.name
                    )
        return names

    @staticmethod
    def _ast_called_names(tree) -> set:
        """Return dotted call-names (e.g. 'os.system') found in the AST."""
        import ast

        calls: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                parts = []
                while isinstance(func, ast.Attribute):
                    parts.append(func.attr)
                    func = func.value
                if isinstance(func, ast.Name):
                    parts.append(func.id)
                if parts:
                    parts.reverse()
                    calls.add(".".join(parts))
        return calls

    # --- Main validation entry point -----------------------------------------

    def validate_imports(self, repo_path: str) -> tuple[bool, List[str]]:
        """Validate miner repo using AST parsing.

        1. Scan .py files under fla/ for dangerous calls / imports.
        2. Check the 5 target files exist with required/forbidden imports.
        """
        import ast
        import glob as _glob

        errors: List[str] = []

        # --- Phase 1: deep-scan .py files under fla/ for dangerous patterns ---
        # Scoped to fla/ to avoid false positives from the base repo's own
        # setup.py, tests/, benchmarks/, etc. which legitimately use os/subprocess.
        fla_root = os.path.join(repo_path, "fla")
        all_py = (
            _glob.glob(os.path.join(fla_root, "**", "*.py"), recursive=True)
            if os.path.isdir(fla_root)
            else []
        )
        for py_path in all_py:
            rel = os.path.relpath(py_path, repo_path)
            try:
                with open(py_path, "r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source, filename=rel)
            except SyntaxError:
                errors.append(f"{rel}: SyntaxError - file cannot be parsed")
                continue

            # Dangerous calls
            called = self._ast_called_names(tree)
            for dc in self.DANGEROUS_CALLS:
                if dc in called:
                    errors.append(f"{rel}: Dangerous call detected: {dc}()")

            # Dangerous module imports
            imported = self._ast_imported_names(tree)
            for mod in self.DANGEROUS_IMPORT_MODULES:
                for imp in imported:
                    if imp == mod or imp.startswith(f"{mod}."):
                        errors.append(
                            f"{rel}: Dangerous module imported: {imp}"
                        )

        # --- Phase 2: target-file checks (required + forbidden imports) ------
        quasar_dir = os.path.join(repo_path, "fla/ops/quasar")
        target_files = [
            "chunk.py",
            "chunk_intra_token_parallel.py",
            "forward_substitution.py",
            "fused_recurrent.py",
            "gate.py",
        ]

        for filename in target_files:
            file_path = os.path.join(quasar_dir, filename)
            if not os.path.exists(file_path):
                errors.append(f"Missing file: {filename}")
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source, filename=filename)
            except SyntaxError:
                errors.append(f"{filename}: SyntaxError - cannot parse")
                continue

            imported = self._ast_imported_names(tree)

            # Forbidden imports (AST-level)
            for forbidden in self.FORBIDDEN_IMPORTS:
                mod = (
                    forbidden.replace("from ", "")
                    .replace("import ", "")
                    .strip()
                )
                for imp in imported:
                    if imp == mod or imp.startswith(f"{mod}."):
                        errors.append(
                            f"{filename}: Forbidden import found: {forbidden}"
                        )

            # Required imports (only for chunk.py)
            if filename == "chunk.py":
                for required in self.REQUIRED_IMPORTS_CHUNK_PY:
                    import_name = required.split("import")[-1].strip()
                    expected = f"fla.utils.{import_name}"
                    if expected not in imported:
                        errors.append(
                            f"{filename}: Missing required import: {required}"
                        )

        return len(errors) == 0, errors

    def fetch_pending_submissions(
        self, limit: int = 10, network: Optional[str] = None
    ) -> List[Dict]:
        """Fetch pending (unvalidated) submissions from validator API for the given network.
        Uses authenticated endpoint that returns full details including fork_url.
        """
        try:
            headers = {}
            params = {"limit": limit}
            if self.validator_instance:
                headers = self.validator_instance._api_auth_headers()
                if network is None:
                    net = (
                        getattr(
                            self.validator_instance.subtensor, "network", None
                        )
                        or "finney"
                    )
                    network = (
                        "test" if str(net).lower() == "test" else "finney"
                    )
            if network is None:
                network = "finney"
            params["network"] = network

            response = requests.get(
                f"{self.validator_api_url}/get_pending_validations",
                params=params,
                headers=headers,
                timeout=30,
            )
            if response.status_code != 200:
                detail = response.text[:500]
                print(
                    f"[VALIDATOR] Error fetching pending submissions: "
                    f"status {response.status_code}, detail: {detail}"
                )
                return []
            data = response.json()
            return data.get("submissions", [])
        except Exception as e:
            print(f"[VALIDATOR] Error fetching pending submissions: {e}")
            return []

    def clone_miner_repo(self, fork_url: str) -> str:
        """Clone miner's fork repository to temporary directory."""
        repo_name = fork_url.split("/")[-1].replace(".git", "")
        repo_path = os.path.join(self.temp_dir, repo_name)

        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        print(f"[VALIDATOR] Cloning repo: {fork_url}")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", fork_url, repo_path],
                check=True,
                capture_output=True,
                timeout=120,
            )
            print(f"[VALIDATOR] Repo cloned to: {repo_path}")
            return repo_path
        except subprocess.TimeoutExpired:
            print(f"[VALIDATOR] Clone timeout for {fork_url}")
            raise
        except subprocess.CalledProcessError as e:
            print(f"[VALIDATOR] Clone failed: {e.stderr}")
            raise

    def checkout_commit(self, repo_path: str, commit_hash: str) -> None:
        try:
            subprocess.run(
                ["git", "checkout", commit_hash],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", commit_hash],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            subprocess.run(
                ["git", "checkout", commit_hash],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

    # Docker image used to sandbox miner code execution.
    # Must have Python, PyTorch, Triton, and fla base dependencies installed.
    # Build it once on your validator machine:
    #   docker build -t quasar-sandbox:latest -f validator/Dockerfile.sandbox .
    SANDBOX_IMAGE = os.getenv(
        "VALIDATOR_SANDBOX_IMAGE",
        "quasar-sandbox:latest",
    )
    # Hard resource limits for the sandbox container.
    SANDBOX_MEMORY_LIMIT = os.getenv("SANDBOX_MEMORY_LIMIT", "16g")
    try:
        SANDBOX_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "300"))
    except (TypeError, ValueError):
        SANDBOX_TIMEOUT = 300

    def _build_test_script(self, sequence_length: int) -> str:
        """Return the Python source for the benchmark test script."""
        return f"""
import sys, os, time, json, types
sys.path.insert(0, "/workspace")

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
os.environ["TRITON_PRINT_DEBUG"] = "1"

import torch

# Stub broken upstream fla modules before importing QuasarAttention.
# Some fla versions reference ops that don't exist in the cloned repo,
# causing an ImportError cascade.  We catch, purge partial state, stub
# the missing symbol, and retry.
def _import_quasar_attention():
    try:
        from fla.layers.quasar import QuasarAttention
        return QuasarAttention
    except ImportError as exc:
        err = str(exc)
        missing_name = None
        stub_target = None
        if "cannot import name" in err:
            parts = err.split("'")
            if len(parts) >= 2:
                missing_name = parts[1]
        if "from '" in err:
            stub_target = err.split("from '")[1].split("'")[0]
        for key in list(sys.modules):
            if key == "fla" or key.startswith("fla."):
                del sys.modules[key]
        if stub_target:
            stub = types.ModuleType(stub_target)
            if missing_name:
                setattr(stub, missing_name, None)
            sys.modules[stub_target] = stub
        from fla.layers.quasar import QuasarAttention
        return QuasarAttention

QuasarAttention = _import_quasar_attention()

def test_quasar():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    seq_len = {sequence_length}
    hidden_size = 512
    head_dim = 64
    num_heads = 8

    quasar = QuasarAttention(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        mode="chunk",
        use_short_conv=True,
    ).to(device)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(3):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
            _ = quasar(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    num_runs = 10
    start = time.time()
    for _ in range(num_runs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad():
            _ = quasar(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    tokens_per_sec = (batch_size * seq_len * num_runs) / elapsed

    vram_bytes = 0
    if device.type == "cuda":
        vram_bytes = torch.cuda.max_memory_allocated()
    vram_mb = vram_bytes / (1024 * 1024)

    print(f"RESULT: {{tokens_per_sec:.2f}}")
    print(f"VRAM_MB: {{vram_mb:.2f}}")

if __name__ == "__main__":
    test_quasar()
"""

    def run_performance_test(
        self, repo_path: str, sequence_length: int
    ) -> Dict[str, float]:
        """Run performance test inside a sandboxed Docker container.

        The miner's cloned repo is mounted **read-only** into the container.
        The container runs with:
          - --network none   (no network access)
          - --cap-drop ALL   (drop all Linux capabilities)
          - --read-only      (read-only root filesystem)
          - --security-opt no-new-privileges
          - bounded memory and timeout

        Returns:
            Dict with keys: tokens_per_sec, vram_mb.
            If the failure is due to validator infrastructure (Docker
            not running, GPU unavailable), the dict also contains
            ``infra_failure: True`` so callers can leave the
            submission pending for retry.
        """
        print(
            f"[VALIDATOR] Running sandboxed performance test "
            f"(seq_len={sequence_length})..."
        )

        # Write the test script into the repo dir (it will be mounted r/o
        # inside the container, so we write it on the host side first).
        test_script_name = f"test_temp_{sequence_length}.py"
        temp_test_script = os.path.join(repo_path, test_script_name)

        try:
            with open(temp_test_script, "w") as f:
                f.write(self._build_test_script(sequence_length))

            # --- Docker SDK sandbox execution --------------------------------
            try:
                import docker as _docker
            except ImportError:
                return self._run_performance_fallback(
                    repo_path, temp_test_script
                )

            try:
                client = _docker.from_env()
                client.ping()
            except Exception as dock_err:
                print(
                    f"[VALIDATOR] CRITICAL: Docker daemon not reachable: "
                    f"{dock_err}. Marking as infra failure so submission "
                    f"stays pending."
                )
                return {
                    "tokens_per_sec": 0.0,
                    "vram_mb": 0.0,
                    "infra_failure": True,
                }

            run_kwargs = {
                "image": self.SANDBOX_IMAGE,
                "command": ["python3", f"/workspace/{test_script_name}"],
                "detach": True,
                "auto_remove": False,
                # Mount miner repo read-only
                "volumes": {
                    os.path.abspath(repo_path): {
                        "bind": "/workspace",
                        "mode": "ro",
                    }
                },
                # --- Security hardening ---
                "network_mode": "none",
                "cap_drop": ["ALL"],
                "read_only": True,
                "security_opt": ["no-new-privileges"],
                "mem_limit": self.SANDBOX_MEMORY_LIMIT,
                "pids_limit": 512,
                # Writable dirs -- Triton compiles .so kernels at runtime
                # so /tmp and the cache dir must allow execution.
                "tmpfs": {
                    "/tmp": "exec,size=2G",
                    "/root/.triton": "exec,size=1G",
                },
                "environment": {
                    "TRITON_CACHE_DIR": "/root/.triton",
                },
                "labels": {"quasar.role": "perf-sandbox"},
            }

            # GPU is required for meaningful benchmarks.
            try:
                run_kwargs["device_requests"] = [
                    _docker.types.DeviceRequest(
                        count=1, capabilities=[["gpu"]]
                    )
                ]
            except Exception as gpu_err:
                print(
                    f"[VALIDATOR] CRITICAL: Cannot configure GPU for "
                    f"sandbox container: {gpu_err}. "
                    f"CPU-only benchmarks are not comparable -- aborting."
                )
                return {
                    "tokens_per_sec": 0.0,
                    "vram_mb": 0.0,
                    "infra_failure": True,
                }

            container = None
            try:
                container = client.containers.run(**run_kwargs)
                result = container.wait(timeout=self.SANDBOX_TIMEOUT)
                output = container.logs().decode(errors="replace")
            except Exception as e:
                err_str = str(e).lower()
                print(f"[VALIDATOR] Sandbox container error: {e}")
                # Infra errors — mark for retry (not the miner's fault)
                if (
                    "pull access denied" in err_str
                    or "not found" in err_str
                    or "404" in err_str
                    or "gpu vendor" in err_str
                    or "nvidia" in err_str
                ):
                    print(
                        f"[VALIDATOR] Sandbox image '{self.SANDBOX_IMAGE}' not available. "
                        f"Build it with: docker build -t {self.SANDBOX_IMAGE} "
                        f"-f validator/Dockerfile.sandbox ."
                    )
                    return {
                        "tokens_per_sec": 0.0,
                        "vram_mb": 0.0,
                        "infra_failure": True,
                    }
                # Other container errors (OOM, timeout, etc.) are miner faults
                return {"tokens_per_sec": 0.0, "vram_mb": 0.0}
            finally:
                if container is not None:
                    try:
                        container.stop(timeout=5)
                    except Exception:
                        pass
                    try:
                        container.remove(force=True)
                    except Exception:
                        pass

            return self._parse_test_output(output)

        except Exception as e:
            print(f"[VALIDATOR] Test setup failed: {e}")
            return {
                "tokens_per_sec": 0.0,
                "vram_mb": 0.0,
                "infra_failure": True,
            }
        finally:
            if os.path.exists(temp_test_script):
                os.remove(temp_test_script)

    @staticmethod
    def _run_performance_fallback(
        repo_path: str, test_script: str
    ) -> Dict[str, float]:
        """Refuse to run when Docker is unavailable.

        Running miner-controlled code without a container sandbox exposes
        the validator to RCE.  Return zero so the submission is scored as
        invalid rather than risk host compromise.
        """
        print(
            "[VALIDATOR] CRITICAL: Docker SDK not available. "
            "Refusing to execute miner code without sandbox. "
            "Install the 'docker' Python package and ensure Docker "
            "daemon is running."
        )
        return {"tokens_per_sec": 0.0, "vram_mb": 0.0, "infra_failure": True}

    @staticmethod
    def _parse_test_output(output: str) -> Dict[str, float]:
        """Parse RESULT: / VRAM_MB: lines from test script stdout."""
        tokens_per_sec = 0.0
        vram_mb = 0.0
        for line in output.split("\n"):
            if "RESULT:" in line:
                try:
                    tokens_per_sec = float(line.split("RESULT:")[1].strip())
                except ValueError:
                    pass
            if "VRAM_MB:" in line:
                try:
                    vram_mb = float(line.split("VRAM_MB:")[1].strip())
                except ValueError:
                    pass

        if tokens_per_sec > 0:
            print(
                f"[VALIDATOR] Test result: {tokens_per_sec:.2f} "
                f"tokens/sec | VRAM: {vram_mb:.2f} MB"
            )
        else:
            print(
                f"[VALIDATOR] Could not parse test results from output "
                f"({len(output)} bytes)"
            )
            tail = output[-1500:] if len(output) > 1500 else output
            for line in tail.strip().split("\n"):
                print(f"[VALIDATOR]   | {line}")
        return {"tokens_per_sec": tokens_per_sec, "vram_mb": vram_mb}

    def verify_performance(
        self, claimed: float, actual: float, tolerance: float = 0.1
    ) -> bool:
        """Verify if actual performance is close to claimed performance."""
        if actual <= 0:
            return False

        # Calculate percentage difference
        diff = abs(claimed - actual) / claimed
        is_valid = diff <= tolerance

        print(f"[VALIDATOR] Performance verification:")
        print(f"  Claimed: {claimed:.2f} tokens/sec")
        print(f"  Actual: {actual:.2f} tokens/sec")
        print(f"  Difference: {diff:.2%}")
        print(f"  Valid: {is_valid}")

        return is_valid

    def validate_submission(self, submission: Dict) -> Dict:
        """Validate a single submission."""
        fork_url = submission.get("fork_url")
        commit_hash = submission.get("commit_hash")
        repo_hash = submission.get(
            "repo_hash"
        )  # Repository context hash from miner
        raw_performance = submission.get("tokens_per_sec")
        target_sequence_length = submission.get(
            "target_sequence_length", 100000
        )
        claimed_benchmarks_json = submission.get("benchmarks")

        # Guard: claimed_performance must be a positive number
        try:
            claimed_performance = (
                float(raw_performance) if raw_performance is not None else None
            )
        except (TypeError, ValueError):
            claimed_performance = None
        if not claimed_performance or claimed_performance <= 0:
            print(
                f"[VALIDATOR] ❌ Submission {submission.get('id')}: "
                f"invalid claimed_performance={claimed_performance}"
            )
            return {
                "submission_id": submission.get("id"),
                "miner_hotkey": submission.get("miner_hotkey"),
                "claimed_performance": claimed_performance,
                "actual_performance": 0.0,
                "score": 0.0,
                "is_valid": False,
                "errors": ["claimed_performance is None or <= 0"],
                "reason": "Invalid claimed performance",
            }

        # Parse claimed benchmarks if available
        claimed_benchmarks = {}
        if claimed_benchmarks_json:
            try:
                claimed_benchmarks = json.loads(claimed_benchmarks_json)
            except Exception as e:
                print(f"[VALIDATOR] Failed to parse benchmarks: {e}")

        print(f"\n[VALIDATOR] Validating submission: {submission.get('id')}")
        print(f"  Fork URL: {fork_url}")
        print(f"  Commit: {commit_hash}")
        print(
            f"  Claimed performance: {claimed_performance:.2f} tokens/sec @ seq_len={target_sequence_length}"
        )
        if claimed_benchmarks:
            print(f"  Claimed benchmarks:")
            for seq_len, metrics in claimed_benchmarks.items():
                print(
                    f"    {seq_len}: {metrics.get('tokens_per_sec', 0):.2f} tokens/sec | VRAM: {metrics.get('vram_mb', 0):.2f} MB"
                )

        try:
            # Clone the repository
            repo_path = self.clone_miner_repo(fork_url)

            if commit_hash:
                self.checkout_commit(repo_path, commit_hash)

            # Validate imports - check for forbidden imports and required imports
            print(f"[VALIDATOR] Checking imports...")
            imports_valid, import_errors = self.validate_imports(repo_path)
            if not imports_valid:
                print(f"[VALIDATOR] ❌ Import validation failed:")
                for error in import_errors:
                    print(f"  - {error}")
                return {
                    "submission_id": submission.get("id"),
                    "miner_hotkey": submission.get("miner_hotkey"),
                    "claimed_performance": claimed_performance,
                    "actual_performance": 0.0,
                    "score": 0.0,
                    "is_valid": False,
                    "errors": import_errors,
                    "reason": "Import validation failed",
                }
            print(f"[VALIDATOR] ✅ Import validation passed")

            # Run benchmarks for all reported sequence lengths.
            # JSON keys are always strings, so normalize to int to avoid
            # TypeError when sorting mixed str/int.
            seq_lengths_to_test = sorted(
                set([512, 1024, 2048, int(target_sequence_length)])
            )
            if claimed_benchmarks:
                seq_lengths_to_test = sorted(
                    set(
                        [int(k) for k in claimed_benchmarks.keys()]
                        + [int(target_sequence_length)]
                    )
                )

            results_by_seq_len: Dict[int, Dict[str, float]] = {}
            infra_failure = False
            for seq_len in seq_lengths_to_test:
                res = self.run_performance_test(repo_path, seq_len)
                results_by_seq_len[seq_len] = res
                if res.get("infra_failure"):
                    infra_failure = True
                    break

            target_results = results_by_seq_len.get(
                int(target_sequence_length),
                {"tokens_per_sec": 0.0, "vram_mb": 0.0},
            )
            actual_performance = float(
                target_results.get("tokens_per_sec", 0.0)
            )

            # Calculate score: higher actual = higher rewards, lower actual = zero
            # If actual >= claimed * 0.9, give full reward (10% tolerance)
            # If actual < claimed * 0.9, give zero reward
            tolerance = 0.9  # 90% of claimed
            score = 0.0
            if actual_performance >= claimed_performance * tolerance:
                # Bonus for exceeding claimed performance
                score = (
                    1.0
                    + (actual_performance - claimed_performance)
                    / claimed_performance
                )
            else:
                # Below tolerance, zero reward
                score = 0.0

            print(f"[VALIDATOR] Performance verification:")
            print(
                f"  Claimed: {claimed_performance:.2f} tokens/sec @ seq_len={target_sequence_length}"
            )
            print(
                f"  Actual: {actual_performance:.2f} tokens/sec @ seq_len={target_sequence_length}"
            )
            print(
                f"  Difference: {(actual_performance - claimed_performance) / claimed_performance * 100:.2f}%"
            )
            print(f"  Score: {score:.4f} (higher actual = higher rewards)")

            # Compare all reported sequence lengths
            print(f"[VALIDATOR] Benchmark comparison:")
            for seq_len in (
                sorted(int(k) for k in claimed_benchmarks.keys())
                if claimed_benchmarks
                else []
            ):
                claimed = claimed_benchmarks.get(
                    str(seq_len), claimed_benchmarks.get(seq_len, {})
                )
                claimed = (
                    claimed.get("tokens_per_sec", 0)
                    if isinstance(claimed, dict)
                    else 0
                )
                actual = results_by_seq_len.get(seq_len, {}).get(
                    "tokens_per_sec", 0
                )
                diff = (actual - claimed) / claimed * 100 if claimed > 0 else 0
                print(
                    f"  {seq_len}: claimed={claimed:.2f}, actual={actual:.2f}, diff={diff:.2f}%"
                )

            # Note: Logit verification is now handled in evaluate_performance_submissions()
            # after performance validation passes, to avoid duplicate calls and ensure
            # we have the submission result before running verification
            verification_result = None

            # Clean up
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)

            result = {
                "submission_id": submission.get("id"),
                "miner_hotkey": submission.get("miner_hotkey"),
                "claimed_performance": claimed_performance,
                "actual_performance": actual_performance,
                "results_by_seq_len": results_by_seq_len,
                "score": score,
                "fork_url": fork_url,
                "commit_hash": commit_hash,
                "repo_hash": repo_hash,  # Include repo_hash in result
            }

            if infra_failure:
                result["infra_failure"] = True

            # Add verification result if available
            if verification_result:
                result["verification"] = verification_result

            return result

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # Miner code fault — score zero, mark as validated
            print(f"[VALIDATOR] Miner code error: {e}")
            traceback.print_exc()
            return {
                "submission_id": submission.get("id"),
                "miner_hotkey": submission.get("miner_hotkey"),
                "claimed_performance": claimed_performance,
                "actual_performance": 0.0,
                "score": 0.0,
                "error": str(e),
            }
        except Exception as e:
            # Infrastructure fault — leave submission pending for retry
            print(f"[VALIDATOR] Infra failure during validation: {e}")
            traceback.print_exc()
            return {
                "submission_id": submission.get("id"),
                "miner_hotkey": submission.get("miner_hotkey"),
                "claimed_performance": claimed_performance,
                "actual_performance": 0.0,
                "score": 0.0,
                "error": str(e),
                "infra_failure": True,
            }

    def cleanup(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"[VALIDATOR] Cleaned up temp dir: {self.temp_dir}")


class Validator(BaseValidatorNeuron):
    """
    Simplified Validator for QUASAR-SUBNET.
    Evaluates miners by calling the challenge container.

    Now includes logit verification from const's qllm architecture to prevent
    miners from returning bogus values quickly.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("🚀 Initializing QUASAR Validator...")

        # Set polling interval from config (default 5 minutes = 300 seconds)
        polling_interval = getattr(config.neuron, "polling_interval", 300)
        if hasattr(self, "neuron"):
            self.neuron.polling_interval_seconds = polling_interval
        elif hasattr(self, "_polling_interval_seconds"):
            self._polling_interval_seconds = polling_interval
        bt.logging.info(
            f"⏱️ Polling interval: {polling_interval}s ({polling_interval/60:.1f} minutes)"
        )

        # Initialize PerformanceValidator for speed optimization validation
        # Pass self reference so PerformanceValidator can access logit verification methods
        self.performance_validator = PerformanceValidator(
            validator_instance=self
        )
        bt.logging.info("⚡ Performance validator initialized")

        # ═══════════════════════════════════════════════════════════════════════════
        # LOGIT VERIFICATION (from const's qllm architecture)
        # Reference model for verifying miners are running the actual model
        # ═══════════════════════════════════════════════════════════════════════════
        self.reference_model = None
        self.reference_model_name = os.getenv(
            "REFERENCE_MODEL", "Qwen/Qwen3-4B-Instruct-2507"
        )
        self.logit_verification_enabled = (
            os.getenv("ENABLE_LOGIT_VERIFICATION", "true").lower() == "true"
        )

        # Verification thresholds (from const's implementation)
        self.cosine_sim_threshold = float(
            os.getenv("COSINE_SIM_THRESHOLD", "0.99")
        )
        self.max_abs_diff_threshold = float(
            os.getenv("MAX_ABS_DIFF_THRESHOLD", "0.1")
        )

        bt.logging.info(
            f"🔍 Logit verification: {'ENABLED' if self.logit_verification_enabled else 'DISABLED'}"
        )
        if self.logit_verification_enabled:
            bt.logging.info(f"   Reference model: {self.reference_model_name}")
            bt.logging.info(
                f"   Cosine sim threshold: {self.cosine_sim_threshold}"
            )
            bt.logging.info(
                f"   Max abs diff threshold: {self.max_abs_diff_threshold}"
            )

        # Initialize scores as numpy (must match base class which uses numpy throughout)
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.load_state()

        bt.logging.info(f"📡 Validator API URL: {VALIDATOR_API_URL}")

    def _api_auth_headers(self) -> Dict[str, str]:
        """Generate authentication headers with timestamp nonce for replay protection."""
        hotkey = self.wallet.hotkey.ss58_address
        timestamp = str(int(time.time()))
        message = f"{hotkey}:{timestamp}".encode()
        signature = self.wallet.hotkey.sign(message).hex()
        return {
            "Hotkey": hotkey,
            "Signature": signature,
            "Timestamp": timestamp,
        }

    def load_reference_model(self):
        """Load the reference model for logit verification (lazy loading)."""
        if self.reference_model is not None:
            return

        if not self.logit_verification_enabled:
            return

        try:
            from quasar.inference_verification import ReferenceModel

            print(
                f"[VALIDATOR] 🔍 Loading reference model: {self.reference_model_name}...",
                flush=True,
            )
            bt.logging.info(
                f"Loading reference model: {self.reference_model_name}"
            )

            self.reference_model = ReferenceModel(self.reference_model_name)

            # Load the model - handle both sync and async contexts
            import asyncio

            try:
                # Try to get the running event loop
                asyncio.get_running_loop()
                # If we're in an async context, schedule the coroutine
                # Use a thread-safe method to run it
                import concurrent.futures
                import threading

                # Create a future to wait for the result
                future = concurrent.futures.Future()

                def run_in_thread():
                    """Run the async load in a separate thread with its own event loop."""
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        new_loop.run_until_complete(
                            self.reference_model.load()
                        )
                        new_loop.close()
                        future.set_result(True)
                    except Exception as e:
                        future.set_exception(e)

                # Run in a separate thread to avoid event loop conflict
                thread = threading.Thread(target=run_in_thread, daemon=True)
                thread.start()
                thread.join(timeout=300)  # 5 minute timeout

                if thread.is_alive():
                    raise TimeoutError(
                        "Model loading timed out after 5 minutes"
                    )

                # Check if there was an exception
                if future.exception():
                    raise future.exception()

            except RuntimeError:
                # No running event loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.reference_model.load())
                loop.close()

            print(
                f"[VALIDATOR] ✅ Reference model loaded successfully",
                flush=True,
            )
            bt.logging.success("Reference model loaded successfully")

        except Exception as e:
            print(
                f"[VALIDATOR] ⚠️ Failed to load reference model: {e}",
                flush=True,
            )
            bt.logging.warning(f"Failed to load reference model: {e}")
            self.reference_model = None

    def run_logit_verification(
        self,
        submission_id: int,
        docker_image: str = None,
        repo_path: str = None,
        fork_url: str = None,
        commit_hash: str = None,
    ) -> Dict:
        """
        Run logit verification for a submission.

        This is the core verification from const's qllm architecture:
        1. Build repository context (same as miner used during generation)
        2. Generate random prompt with context
        3. Run inference on miner's container (or local test)
        4. Run inference on reference model with same context
        5. Compare logits at random step

        Args:
            submission_id: Submission ID for tracking
            docker_image: Docker image to verify (optional, for container-based miners)
            repo_path: Path to cloned repository (for context building)
            fork_url: Fork URL (for cloning if repo_path not provided)
            commit_hash: Commit hash to checkout (for deterministic context)

        Returns:
            Dict with verification results
        """
        if not self.logit_verification_enabled:
            return {
                "verified": False,
                "reason": "Logit verification disabled but mandatory - enable ENABLE_LOGIT_VERIFICATION=true",
            }

        # Ensure reference model is loaded
        self.load_reference_model()

        if self.reference_model is None:
            return {
                "verified": False,
                "reason": "Reference model failed to load - cannot verify",
            }

        try:
            from quasar.inference_verification import (
                generate_verification_challenge,
                verify_logits,
                CONFIG,
            )
            import asyncio
            import hashlib

            print(
                f"[VALIDATOR] 🔍 Running logit verification for submission {submission_id}...",
                flush=True,
            )

            # Build repository context (same as miner used)
            repo_context = None
            repo_hash = None

            if repo_path and os.path.exists(repo_path):
                try:
                    print(
                        f"[VALIDATOR]   Building repository context from {repo_path}...",
                        flush=True,
                    )

                    # Validate repository structure
                    is_valid, warnings = validate_repo_structure(repo_path)
                    if warnings:
                        print(
                            f"[VALIDATOR]   ⚠️  Repository validation warnings:",
                            flush=True,
                        )
                        for warning in warnings:
                            print(f"      - {warning}", flush=True)

                    # Build full context (same parameters as miner)
                    repo_context = build_full_context(
                        repo_path=repo_path,
                        target_file="chunk.py",
                        include_tree=True,
                        max_files=50,  # Match miner default
                        max_size=200000,  # Match miner default
                        byoc_mode=False,  # Validator doesn't use BYOC
                    )

                    # Calculate repo hash for consistency tracking
                    repo_hash = hashlib.sha256(
                        repo_context.encode()
                    ).hexdigest()[:16]
                    context_tokens = estimate_context_tokens(repo_context)
                    print(
                        f"[VALIDATOR]   ✅ Context built: ~{context_tokens} tokens, hash: {repo_hash}",
                        flush=True,
                    )
                    bt.logging.info(
                        f"Repository context built: ~{context_tokens} tokens, hash: {repo_hash}"
                    )

                except Exception as e:
                    print(
                        f"[VALIDATOR]   ⚠️  Failed to build context: {e}. Using context-free verification.",
                        flush=True,
                    )
                    bt.logging.warning(
                        f"Failed to build repository context: {e}"
                    )
                    repo_context = None
            elif fork_url and commit_hash:
                # Clone repo if not provided
                try:
                    print(
                        f"[VALIDATOR]   Cloning repository for context building...",
                        flush=True,
                    )
                    # Use performance_validator's clone method
                    cloned_repo_path = (
                        self.performance_validator.clone_miner_repo(fork_url)
                    )
                    self.performance_validator.checkout_commit(
                        cloned_repo_path, commit_hash
                    )

                    repo_context = build_full_context(
                        repo_path=cloned_repo_path,
                        target_file="chunk.py",
                        include_tree=True,
                        max_files=50,
                        max_size=200000,
                        byoc_mode=False,
                    )
                    repo_hash = hashlib.sha256(
                        repo_context.encode()
                    ).hexdigest()[:16]
                    context_tokens = estimate_context_tokens(repo_context)
                    print(
                        f"[VALIDATOR]   ✅ Context built from cloned repo: ~{context_tokens} tokens",
                        flush=True,
                    )

                    # Cleanup cloned repo
                    if os.path.exists(cloned_repo_path):
                        shutil.rmtree(cloned_repo_path)

                except Exception as e:
                    print(
                        f"[VALIDATOR]   ⚠️  Failed to clone/build context: {e}",
                        flush=True,
                    )
                    repo_context = None

            # Logit verification is mandatory: miners MUST provide a docker_image.
            # Without one, the submission cannot be verified and will not rank.
            if not docker_image:
                print(
                    f"[VALIDATOR]   No docker_image for submission {submission_id} - FAIL (mandatory). "
                    f"Miner must set DOCKER_USERNAME env var and push their inference container.",
                    flush=True,
                )
                return {
                    "verified": False,
                    "reason": "No docker_image provided. Miner must set DOCKER_USERNAME and push "
                    "their inference container (see Dockerfile.inference).",
                    "repo_hash": repo_hash,
                }

            # Generate challenge with multiple capture steps
            challenge = generate_verification_challenge(self.reference_model)
            prompt = challenge["prompt"]
            gen_len = challenge["gen_len"]
            logits_at_step = challenge["logits_at_step"]
            logits_at_steps = challenge.get("logits_at_steps") or [
                logits_at_step
            ]

            if repo_context:
                print(
                    f"[VALIDATOR]   Using context-aware verification (repo_hash: {repo_hash})",
                    flush=True,
                )

            print(
                f"[VALIDATOR]   Challenge: prompt_len={len(prompt)}, gen_len={gen_len}, "
                f"capture_steps={logits_at_steps}",
                flush=True,
            )

            # Run reference model inference - handle event loop conflicts
            try:
                asyncio.get_running_loop()
                import concurrent.futures
                import threading

                future = concurrent.futures.Future()

                def run_inference():
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(
                            self.reference_model.inference(
                                prompt,
                                gen_len,
                                logits_at_step,
                                logits_at_steps=logits_at_steps,
                            )
                        )
                        new_loop.close()
                        future.set_result(result)
                    except Exception as e:
                        future.set_exception(e)

                thread = threading.Thread(target=run_inference, daemon=True)
                thread.start()
                thread.join(timeout=300)

                if thread.is_alive():
                    raise TimeoutError("Inference timed out after 5 minutes")

                if future.exception():
                    raise future.exception()

                reference_result = future.result()

            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                reference_result = loop.run_until_complete(
                    self.reference_model.inference(
                        prompt,
                        gen_len,
                        logits_at_step,
                        logits_at_steps=logits_at_steps,
                    )
                )
                loop.close()

            ref_multi = reference_result.get("captured_logits_multi", {})
            if (
                not ref_multi
                and reference_result.get("captured_logits") is not None
            ):
                ref_multi = {
                    logits_at_step: reference_result["captured_logits"]
                }

            if not ref_multi:
                return {
                    "verified": False,
                    "reason": "Reference model failed to capture logits",
                }

            from quasar.inference_verification import (
                run_container_inference as _run_container,
            )

            print(
                f"[VALIDATOR]   Running container inference for {docker_image}...",
                flush=True,
            )
            miner_result = _run_container(
                hotkey=str(submission_id),
                docker_image=docker_image,
                prompt=prompt,
                gen_len=gen_len,
                logits_at_step=logits_at_step,
                logits_at_steps=logits_at_steps,
            )

            if not miner_result.success:
                print(
                    f"[VALIDATOR]   Container execution failed: {miner_result.error}",
                    flush=True,
                )
                ref_elapsed = reference_result.get("elapsed_sec", 1)
                return {
                    "verified": False,
                    "reason": f"Container execution failed: {miner_result.error}",
                    "reference_throughput": (
                        gen_len / ref_elapsed if ref_elapsed > 0 else 0
                    ),
                    "reference_elapsed_sec": ref_elapsed,
                    "repo_hash": repo_hash,
                }

            # Require multi-step logits -- no fallback to legacy single-step.
            # A miner returning only `captured_logits` (single step) while
            # ignoring `logits_at_steps` would bypass the multi-step check.
            miner_multi = miner_result.captured_logits_multi or {}

            if not miner_multi:
                print(
                    f"[VALIDATOR]   Miner container did not return captured_logits_multi "
                    f"(legacy single-step not accepted)",
                    flush=True,
                )
                return {
                    "verified": False,
                    "reason": "Container must return captured_logits_multi for all requested steps",
                    "repo_hash": repo_hash,
                }

            # Verify ALL captured steps -- miner must pass every one
            all_passed = True
            worst_cosine = 1.0
            worst_max_diff = 0.0
            fail_reason = None
            steps_checked = 0

            for step in logits_at_steps:
                ref_logits = ref_multi.get(step)
                miner_logits = miner_multi.get(step)

                if ref_logits is None:
                    continue
                if miner_logits is None:
                    all_passed = False
                    fail_reason = f"Missing logits at step {step}"
                    print(
                        f"[VALIDATOR]   Step {step}: FAIL (miner returned no logits)",
                        flush=True,
                    )
                    break

                step_result = verify_logits(
                    miner_logits,
                    ref_logits,
                    cosine_threshold=self.cosine_sim_threshold,
                    max_diff_threshold=self.max_abs_diff_threshold,
                )
                steps_checked += 1
                worst_cosine = min(worst_cosine, step_result.cosine_sim or 1.0)
                worst_max_diff = max(
                    worst_max_diff, step_result.max_abs_diff or 0.0
                )

                if not step_result.verified:
                    all_passed = False
                    fail_reason = f"Step {step}: {step_result.reason}"
                    print(
                        f"[VALIDATOR]   Step {step}: FAIL (cosine={step_result.cosine_sim:.4f}, "
                        f"max_diff={step_result.max_abs_diff:.4f})",
                        flush=True,
                    )
                    break
                else:
                    print(
                        f"[VALIDATOR]   Step {step}: PASS (cosine={step_result.cosine_sim:.4f}, "
                        f"max_diff={step_result.max_abs_diff:.4f})",
                        flush=True,
                    )

            miner_throughput = (
                gen_len / miner_result.elapsed_sec
                if miner_result.elapsed_sec > 0
                else 0
            )
            ref_throughput = (
                gen_len / reference_result["elapsed_sec"]
                if reference_result["elapsed_sec"] > 0
                else 0
            )

            print(
                f"[VALIDATOR]   Result: {'PASS' if all_passed else 'FAIL'} "
                f"({steps_checked} steps checked, worst_cosine={worst_cosine:.4f}, "
                f"worst_max_diff={worst_max_diff:.4f})",
                flush=True,
            )
            print(
                f"[VALIDATOR]   Throughput: miner={miner_throughput:.1f} tok/s (validator-timed), "
                f"reference={ref_throughput:.1f} tok/s",
                flush=True,
            )

            return {
                "verified": all_passed,
                "reason": None if all_passed else fail_reason,
                "cosine_similarity": worst_cosine,
                "max_abs_diff": worst_max_diff,
                "steps_checked": steps_checked,
                "miner_throughput": miner_throughput,
                "reference_throughput": ref_throughput,
                "reference_elapsed_sec": reference_result["elapsed_sec"],
                "miner_elapsed_sec": miner_result.elapsed_sec,
                "repo_hash": repo_hash,
            }

        except Exception as e:
            print(f"[VALIDATOR] ❌ Logit verification failed: {e}", flush=True)
            traceback.print_exc()
            return {
                "verified": False,
                "reason": f"Verification error: {str(e)}",
                "repo_hash": repo_hash if "repo_hash" in locals() else None,
            }

    def record_verification_result(self, submission_id: int, result: Dict):
        """Record logit verification result to the API.

        Logit verification is mandatory. All submissions must pass to rank.
        verified=None (e.g. no docker_image, disabled, model unavailable) is
        recorded as False so the submission is excluded from rankings.
        """
        try:
            verified = result.get("verified")
            if verified is None:
                reason = result.get("reason", "Verification not applicable")
                print(
                    f"[VALIDATOR] ⚠️ Logit verification not applicable for submission {submission_id} "
                    f"({reason}) - recording as FAILED (verification is mandatory)",
                    flush=True,
                )
                verified = False
                result = {
                    **result,
                    "verified": False,
                    "reason": f"mandatory_fail: {reason}",
                }

            params = {
                "submission_id": submission_id,
                "verified": bool(verified),
            }

            if result.get("cosine_similarity") is not None:
                params["cosine_similarity"] = result.get("cosine_similarity")
            if result.get("max_abs_diff") is not None:
                params["max_abs_diff"] = result.get("max_abs_diff")
            if (
                result.get("throughput_verified") is not None
                or result.get("reference_throughput") is not None
            ):
                params["throughput"] = result.get(
                    "throughput_verified"
                ) or result.get("reference_throughput")
            if result.get("reason"):
                params["reason"] = result.get("reason")

            response = requests.post(
                f"{VALIDATOR_API_URL}/record_verification",
                params=params,
                headers=self._api_auth_headers(),
                timeout=30,
            )
            if response.status_code == 200:
                print(
                    f"[VALIDATOR] ✅ Verification result recorded for submission {submission_id}: verified={verified}",
                    flush=True,
                )
            else:
                bt.logging.warning(
                    f"Failed to record verification for {submission_id}: {response.status_code} - {response.text}"
                )
                print(
                    f"[VALIDATOR] ⚠️ Failed to record verification: {response.status_code} - {response.text}",
                    flush=True,
                )
        except Exception as e:
            bt.logging.warning(
                f"Failed to record verification result for {submission_id}: {e}"
            )
            print(
                f"[VALIDATOR] ⚠️ Failed to record verification result: {e}",
                flush=True,
            )

    def evaluate_performance_submissions(self) -> Dict[str, float]:
        """Evaluate performance submissions by cloning repos and running tests.

        Returns:
            Dictionary mapping miner_hotkey to score (0.0 to 1.0).
        """
        print(
            f"[VALIDATOR] ⚡ Evaluating performance submissions...", flush=True
        )
        bt.logging.info("⚡ Evaluating performance submissions...")

        evaluated_scores = {}  # hotkey -> score

        try:
            # Fetch pending submissions from API (batch size configurable for backlog drain)
            batch_size = int(os.getenv("VALIDATOR_PENDING_BATCH_SIZE", "10"))
            submissions = self.performance_validator.fetch_pending_submissions(
                limit=batch_size
            )

            if not submissions:
                print(
                    "[VALIDATOR] No performance submissions to evaluate",
                    flush=True,
                )
                return evaluated_scores

            print(
                f"[VALIDATOR] Found {len(submissions)} performance submissions",
                flush=True,
            )

            for submission in submissions:
                # Skip already validated submissions
                if submission.get("validated", False):
                    continue

                # Validate the submission
                result = self.performance_validator.validate_submission(
                    submission
                )

                # Infrastructure failure — skip this submission so it
                # stays pending and can be retried on the next cycle.
                if result.get("infra_failure"):
                    print(
                        f"[VALIDATOR] ⚠️ Infra failure for submission "
                        f"{submission.get('id')} — will retry later",
                        flush=True,
                    )
                    continue

                miner_hotkey = result.get("miner_hotkey") or "unknown"
                score = result.get("score", 0.0)

                # Use the score from validate_submission (already calculated)
                # Normalize to 0-1 range (assuming max reasonable score is around 2.0)
                normalized_score = min(score / 2.0, 1.0)

                if score > 0:
                    print(
                        f"[VALIDATOR] ✅ Valid submission from {miner_hotkey[:12]}... - Score: {score:.4f} (normalized: {normalized_score:.4f})",
                        flush=True,
                    )

                    submission_id = submission.get("id")

                    # ═══════════════════════════════════════════════════════════════════════
                    # LOGIT VERIFICATION (from const's qllm architecture)
                    # Run after performance test passes to verify miner is running actual model
                    # ═══════════════════════════════════════════════════════════════════════
                    if self.logit_verification_enabled and submission_id:
                        print(
                            f"[VALIDATOR] 🔍 Running logit verification...",
                            flush=True,
                        )
                        docker_image = submission.get("docker_image")
                        # Get fork_url and commit_hash from submission or result
                        fork_url = submission.get("fork_url") or result.get(
                            "fork_url"
                        )
                        commit_hash = submission.get(
                            "commit_hash"
                        ) or result.get("commit_hash")
                        # Note: repo_path is not available here (already cleaned up)
                        # Context will be built from fork_url + commit_hash
                        verification_result = self.run_logit_verification(
                            submission_id=submission_id,
                            docker_image=docker_image,
                            repo_path=None,
                            fork_url=fork_url,
                            commit_hash=commit_hash,
                        )

                        # Logit verification is mandatory: only True passes
                        if verification_result.get("verified") == True:
                            evaluated_scores[miner_hotkey] = normalized_score
                        else:
                            status = (
                                "FAILED"
                                if verification_result.get("verified") == False
                                else "NOT RUN"
                            )
                            print(
                                f"[VALIDATOR] ❌ Logit verification {status} - score set to 0",
                                flush=True,
                            )
                            normalized_score = 0.0
                            evaluated_scores[miner_hotkey] = 0.0
                    else:
                        verification_result = None
                        evaluated_scores[miner_hotkey] = normalized_score

                    # Mark submission as validated in API and record score
                    # Send the validator-measured actual_tokens_per_sec so rankings
                    # use the real value, not the miner-claimed one.
                    actual_tps = result.get("actual_performance", 0.0)
                    if submission_id:
                        try:
                            payload = {
                                "submission_id": submission_id,
                                "score": normalized_score,
                                "actual_tokens_per_sec": actual_tps,
                            }
                            # Include verification fields if available
                            if verification_result is not None:
                                verified = verification_result.get("verified")
                                if verified is None:
                                    verified = False
                                payload["verified"] = bool(verified)
                                if (
                                    verification_result.get(
                                        "cosine_similarity"
                                    )
                                    is not None
                                ):
                                    payload["cosine_similarity"] = (
                                        verification_result[
                                            "cosine_similarity"
                                        ]
                                    )
                                if (
                                    verification_result.get("max_abs_diff")
                                    is not None
                                ):
                                    payload["max_abs_diff"] = (
                                        verification_result["max_abs_diff"]
                                    )
                                tp = verification_result.get(
                                    "miner_throughput"
                                ) or verification_result.get(
                                    "reference_throughput"
                                )
                                if tp is not None:
                                    payload["throughput"] = tp
                                if verification_result.get("reason"):
                                    payload["reason"] = verification_result[
                                        "reason"
                                    ]

                            response = requests.post(
                                f"{VALIDATOR_API_URL}/mark_validated_with_verification",
                                json=payload,
                                headers=self._api_auth_headers(),
                                timeout=30,
                            )
                            # Fallback: API hasn't been updated yet
                            if response.status_code == 404:
                                response = requests.post(
                                    f"{VALIDATOR_API_URL}/mark_validated",
                                    json=payload,
                                    headers=self._api_auth_headers(),
                                    timeout=30,
                                )
                                if verification_result is not None:
                                    self.record_verification_result(
                                        submission_id, verification_result
                                    )
                            if response.status_code == 200:
                                print(
                                    f"[VALIDATOR] ✅ Submission {submission_id} marked validated: score={normalized_score:.4f}, actual_tps={actual_tps:.2f}",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[VALIDATOR] ⚠️ Failed to mark submission as validated: {response.status_code} - {response.text}",
                                    flush=True,
                                )
                        except Exception as e:
                            print(
                                f"[VALIDATOR] Failed to mark submission as validated: {e}",
                                flush=True,
                            )
                else:
                    print(
                        f"[VALIDATOR] ❌ Invalid submission from {miner_hotkey[:12]}...",
                        flush=True,
                    )
                    evaluated_scores[miner_hotkey] = 0.0

                    # Record failure for IP banning (Phase 4)
                    ip_address = submission.get("ip_address")
                    if ip_address:
                        try:
                            requests.post(
                                f"{VALIDATOR_API_URL}/record_failure",
                                json={"ip_address": ip_address},
                                headers=self._api_auth_headers(),
                                timeout=10,
                            )
                            print(
                                f"[VALIDATOR] 📝 Recorded failure for IP: {ip_address}",
                                flush=True,
                            )
                        except Exception as e:
                            print(
                                f"[VALIDATOR] Failed to record failure: {e}",
                                flush=True,
                            )

                    # Still mark as validated to avoid re-processing (with score 0.0)
                    submission_id = submission.get("id")
                    if submission_id:
                        try:
                            fail_payload = {
                                "submission_id": submission_id,
                                "score": 0.0,
                            }
                            response = requests.post(
                                f"{VALIDATOR_API_URL}/mark_validated",
                                json={
                                    "submission_id": submission_id,
                                    "score": 0.0,
                                },
                                headers=self._api_auth_headers(),
                                timeout=30,
                            )
                            if response.status_code == 404:
                                response = requests.post(
                                    f"{VALIDATOR_API_URL}/mark_validated",
                                    json=fail_payload,
                                    headers=self._api_auth_headers(),
                                    timeout=30,
                                )
                            if response.status_code == 200:
                                print(
                                    f"[VALIDATOR] ✅ Submission {submission_id} marked as validated with score 0.0",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[VALIDATOR] ⚠️ Failed to mark submission as validated: {response.status_code} - {response.text}",
                                    flush=True,
                                )
                        except Exception as e:
                            print(
                                f"[VALIDATOR] Failed to mark submission as validated: {e}",
                                flush=True,
                            )

            if evaluated_scores:
                print(
                    f"[VALIDATOR] ✅ Evaluated {len(evaluated_scores)} performance submissions",
                    flush=True,
                )

        except Exception as e:
            print(
                f"[VALIDATOR] ⚠️ Failed to evaluate performance submissions: {e}",
                flush=True,
            )
            bt.logging.warning(
                f"Failed to evaluate performance submissions: {e}"
            )
            traceback.print_exc()

        return evaluated_scores

    def load_state(self):
        """Load validator state from disk (numpy format, compatible with base class)."""
        try:
            # Try numpy format first (base class format)
            npz_path = self.config.neuron.full_path + "/state.npz"
            pt_path = self.config.neuron.full_path + "/state.pt"

            if os.path.exists(npz_path):
                state = np.load(npz_path, allow_pickle=True)
                self.step = int(state.get("step", 0))
                scores = state.get("scores", self.scores)
                if isinstance(scores, torch.Tensor):
                    scores = scores.detach().cpu().numpy()
                self.scores = np.array(scores, dtype=np.float32)
                if hasattr(state, "files") and "hotkeys" in state.files:
                    self.hotkeys = list(state["hotkeys"])
                bt.logging.success("💾 State loaded from npz successfully.")
            elif os.path.exists(pt_path):
                state = torch.load(pt_path, weights_only=False)
                self.step = state.get("step", 0)
                scores = state.get("scores", self.scores)
                if isinstance(scores, torch.Tensor):
                    scores = scores.detach().cpu().numpy()
                self.scores = np.array(scores, dtype=np.float32)
                bt.logging.success(
                    "💾 State loaded from pt (legacy) successfully."
                )
        except Exception as e:
            bt.logging.warning(
                f"⚠️ Failed to load state (starting fresh): {e}"
            )

    def should_set_weights(self) -> bool:
        """Override base class: we handle set_weights() in forward(), not in sync()."""
        return False

    def set_weights(self):
        """
        Override set_weights to return success status for proper error handling.
        Returns True if weights were successfully set on-chain, False otherwise.
        """
        # Convert to numpy if it's a torch tensor
        scores = self.scores
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        # Check if scores contains any NaN values and log a warning if it does.
        if np.isnan(scores).any():
            bt.logging.warning(
                f"Scores contain NaN values. This may be due to a lack of responses from miners, or a bug in your reward functions."
            )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        norm = np.linalg.norm(scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = scores / norm

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

        # Diagnostic checks before setting weights
        try:
            # Check if validator is registered
            validator_uid = None
            for uid in range(self.metagraph.n):
                if (
                    self.metagraph.hotkeys[uid]
                    == self.wallet.hotkey.ss58_address
                ):
                    validator_uid = uid
                    break

            if validator_uid is None:
                error_msg = "Validator hotkey not found in metagraph - validator may not be registered on subnet"
                print(f"[VALIDATOR] ❌ {error_msg}", flush=True)
                bt.logging.error(error_msg)
                return False

            # Check if validator has permit (can set weights)
            if not self.metagraph.validator_permit[validator_uid]:
                error_msg = f"Validator UID {validator_uid} does not have validator_permit - cannot set weights"
                print(f"[VALIDATOR] ❌ {error_msg}", flush=True)
                bt.logging.error(error_msg)
                return False

            # Check if we have UIDs to set weights for
            if len(uint_uids) == 0 or len(uint_weights) == 0:
                error_msg = "No UIDs or weights to set (all weights are zero)"
                print(f"[VALIDATOR] ⚠️ {error_msg}", flush=True)
                bt.logging.warning(error_msg)
                return False

            print(
                f"[VALIDATOR] 🔍 Attempting to set weights for {len(uint_uids)} miners...",
                flush=True,
            )
            bt.logging.info(
                f"Setting weights for {len(uint_uids)} miners: UIDs={uint_uids[:5]}... (showing first 5)"
            )

        except Exception as diag_error:
            print(
                f"[VALIDATOR] ⚠️ Diagnostic check failed: {diag_error}",
                flush=True,
            )
            bt.logging.warning(f"Diagnostic check failed: {diag_error}")

        # Set the weights on chain via our subtensor connection.
        result, msg = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.config.netuid,
            uids=uint_uids,
            weights=uint_weights,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=self.spec_version,
        )
        if result is True:
            print(
                f"[VALIDATOR] ✅ set_weights on chain successfully!",
                flush=True,
            )
            bt.logging.info("set_weights on chain successfully!")
            return True
        else:
            # Print detailed error message
            error_msg = f"set_weights failed: {msg}"
            print(f"[VALIDATOR] ❌ {error_msg}", flush=True)
            bt.logging.error(error_msg)

            # Provide helpful diagnostics based on common error messages
            if msg and isinstance(msg, str):
                msg_lower = msg.lower()
                if "cooldown" in msg_lower or "too soon" in msg_lower:
                    print(
                        f"[VALIDATOR] 💡 This is a cooldown issue - will retry in next cycle",
                        flush=True,
                    )
                elif "stake" in msg_lower or "balance" in msg_lower:
                    print(
                        f"[VALIDATOR] 💡 Check validator stake/balance - may need more TAO",
                        flush=True,
                    )
                elif "not registered" in msg_lower or "not found" in msg_lower:
                    print(
                        f"[VALIDATOR] 💡 Validator may not be registered on subnet {self.config.netuid}",
                        flush=True,
                    )
                elif "permit" in msg_lower:
                    print(
                        f"[VALIDATOR] 💡 Validator may not have validator_permit",
                        flush=True,
                    )

            return False

    def save_state(self):
        """Save validator state to disk (numpy format, compatible with base class)."""
        try:
            scores = self.scores
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            np.savez(
                self.config.neuron.full_path + "/state.npz",
                step=self.step,
                scores=scores,
                hotkeys=self.hotkeys,
            )
            bt.logging.info("💾 State saved.")
        except Exception as e:
            bt.logging.error(f"❌ Failed to save state: {e}")

    async def forward(self):
        """Main validation loop with dynamic polling based on submission rate."""
        print("[VALIDATOR] ➡️ Starting validation cycle...", flush=True)
        bt.logging.info("➡️ Starting validation cycle...")

        try:
            # Check submission rate and adjust polling interval dynamically
            polling_interval = 300  # Default: 5 minutes
            try:
                response = requests.get(
                    f"{VALIDATOR_API_URL}/get_submission_rate",
                    params={"window_minutes": 10},
                    headers=self._api_auth_headers(),
                    timeout=10,
                )
                if response.status_code == 200:
                    data = response.json()
                    submissions_per_min = data.get("submissions_per_minute", 0)

                    # Adjust polling interval based on submission rate
                    if submissions_per_min > 5:
                        # High activity: poll every 1 minute
                        polling_interval = 60
                        activity_level = "HIGH"
                    elif submissions_per_min > 1:
                        # Medium activity: poll every 2 minutes
                        polling_interval = 120
                        activity_level = "MEDIUM"
                    else:
                        # Low activity: poll every 5 minutes (default)
                        polling_interval = 300
                        activity_level = "LOW"

                    print(
                        f"[VALIDATOR] 📊 Submission rate: {submissions_per_min:.2f}/min ({activity_level} activity), "
                        f"polling every {polling_interval}s",
                        flush=True,
                    )
                    bt.logging.info(
                        f"📊 Submission rate: {submissions_per_min:.2f}/min, polling every {polling_interval}s"
                    )
                else:
                    print(
                        f"[VALIDATOR] ⚠️ Failed to get submission rate (status {response.status_code}), using default 5min",
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"[VALIDATOR] ⚠️ Failed to get submission rate: {e}, using default 5min",
                    flush=True,
                )
                bt.logging.warning(f"Failed to get submission rate: {e}")

            # Sync metagraph UIDs to the API so weights are set to correct UIDs
            # NOTE: UID sync is non-critical - if it fails, validator continues normally
            # Weights can still be set using hotkeys from the API response
            try:
                self.metagraph.sync(subtensor=self.subtensor)
                uid_map = {}
                for uid_idx in range(self.metagraph.n):
                    hk = self.metagraph.hotkeys[uid_idx]
                    uid_map[hk] = uid_idx
                if uid_map:
                    net = getattr(self.subtensor, "network", None) or "finney"
                    network = (
                        "test" if str(net).lower() == "test" else "finney"
                    )
                    # Increased timeout
                    sync_resp = requests.post(
                        f"{VALIDATOR_API_URL}/sync_uids",
                        json={"network": network, "uid_map": uid_map},
                        headers=self._api_auth_headers(),
                        timeout=(5, 30),
                    )
                    if sync_resp.status_code == 200:
                        sync_data = sync_resp.json()
                        if sync_data.get("updated", 0) > 0:
                            print(
                                f"[VALIDATOR] 🔄 Synced {sync_data['updated']} UIDs to API",
                                flush=True,
                            )
                    else:
                        print(
                            f"[VALIDATOR] ⚠️ UID sync failed (status {sync_resp.status_code}) - non-critical",
                            flush=True,
                        )
            except requests.exceptions.Timeout as e:
                print(
                    f"[VALIDATOR] ⚠️ UID sync timed out (non-critical, continuing): {e}",
                    flush=True,
                )
                bt.logging.warning(f"UID sync timed out (non-critical): {e}")
            except Exception as e:
                print(
                    f"[VALIDATOR] ⚠️ UID sync failed (non-critical, continuing): {e}",
                    flush=True,
                )
                bt.logging.warning(f"UID sync failed (non-critical): {e}")

            # Evaluate performance submissions
            print(
                "[VALIDATOR] ⚡ Evaluating performance submissions...",
                flush=True,
            )
            evaluated_scores = self.evaluate_performance_submissions()

            if not evaluated_scores:
                print(
                    f"[VALIDATOR] ⚠️ No pending submissions to evaluate",
                    flush=True,
                )
            else:
                print(
                    f"[VALIDATOR] ✅ Evaluation complete: {len(evaluated_scores)} submissions evaluated",
                    flush=True,
                )
                bt.logging.success(
                    f"✅ Evaluation complete: {len(evaluated_scores)} submissions"
                )
                for hotkey, score in evaluated_scores.items():
                    print(
                        f"[VALIDATOR]   {hotkey[:12]}...: score={score:.4f}",
                        flush=True,
                    )

            # Ensure round finalization runs: calling get_current_round triggers
            # ensure_current_round on the API, which finalizes expired rounds and
            # creates a new active round if needed.
            try:
                net = getattr(self.subtensor, "network", None) or "finney"
                network = "test" if str(net).lower() == "test" else "finney"
                response = requests.get(
                    f"{VALIDATOR_API_URL}/get_current_round",
                    params={"network": network},
                    headers=self._api_auth_headers(),
                    timeout=15,
                )
                if response.status_code == 200:
                    round_data = response.json()
                    time_remaining = round_data.get(
                        "time_remaining_seconds", 3600
                    )
                    round_number = round_data.get("round_number")
                    total_submissions = round_data.get("total_submissions", 0)

                    if time_remaining <= 0:
                        print(
                            f"[VALIDATOR] ⏰ Round {round_number} expired - triggered finalization via API",
                            flush=True,
                        )
                    elif time_remaining < 3600:
                        hours_remaining = time_remaining / 3600
                        print(
                            f"[VALIDATOR] ⏰ Round {round_number}: {hours_remaining:.1f}h remaining, {total_submissions} submissions",
                            flush=True,
                        )
                    else:
                        hours_remaining = time_remaining / 3600
                        print(
                            f"[VALIDATOR] Round {round_number}: {hours_remaining:.1f}h remaining, {total_submissions} submissions",
                            flush=True,
                        )
                else:
                    print(
                        f"[VALIDATOR] ⚠️ Round check returned status {response.status_code}",
                        flush=True,
                    )
            except Exception as e:
                print(f"[VALIDATOR] ⚠️ Round check failed: {e}", flush=True)

            # Fetch weights from API and update self.scores for on-chain submission
            try:
                net = getattr(self.subtensor, "network", None) or "finney"
                network = "test" if str(net).lower() == "test" else "finney"
                response = requests.get(
                    f"{VALIDATOR_API_URL}/get_weights",
                    params={"network": network},
                    headers=self._api_auth_headers(),
                    timeout=10,
                )
                if response.status_code == 200:
                    weights_data = response.json()
                    weight_entries = weights_data.get("weights", [])

                    if weight_entries:
                        # Build hotkey -> UID mapping from metagraph (the authoritative source)
                        hotkey_to_uid = {}
                        for uid_idx in range(self.metagraph.n):
                            hk = self.metagraph.hotkeys[uid_idx]
                            hotkey_to_uid[hk] = uid_idx

                        # Reset scores to zero, then populate from API weights
                        self.scores = np.zeros(
                            self.metagraph.n, dtype=np.float32
                        )
                        resolved_count = 0

                        for entry in weight_entries:
                            hotkey = entry.get("hotkey", "")
                            weight = entry.get("weight", 0.0)

                            if hotkey in hotkey_to_uid:
                                uid = hotkey_to_uid[hotkey]
                                self.scores[uid] = float(weight)
                                resolved_count += 1
                                print(
                                    f"[VALIDATOR]   UID {uid} ({hotkey[:12]}...): weight={weight:.4f}",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[VALIDATOR] ⚠️ Hotkey {hotkey[:12]}... not found in metagraph, skipping",
                                    flush=True,
                                )

                        if resolved_count > 0:
                            print(
                                f"[VALIDATOR] 📊 Updated self.scores for {resolved_count} miners from API weights",
                                flush=True,
                            )
                            bt.logging.info(
                                f"Updated scores for {resolved_count} miners, calling set_weights()"
                            )
                            self.save_state()
                            success = self.set_weights()
                            if success:
                                print(
                                    f"[VALIDATOR] ✅ Weights successfully set on Bittensor chain",
                                    flush=True,
                                )
                                bt.logging.success(
                                    f"✅ Weights submitted to Bittensor chain"
                                )
                            else:
                                # Error message already printed in set_weights() method
                                print(
                                    f"[VALIDATOR] ⚠️ Weight setting failed - will retry in next cycle",
                                    flush=True,
                                )
                                bt.logging.warning(
                                    f"⚠️ Weight setting failed - will retry in next cycle"
                                )
                        else:
                            print(
                                f"[VALIDATOR] ⚠️ No miners from API weights found in metagraph",
                                flush=True,
                            )
                    else:
                        print(
                            f"[VALIDATOR] ⚠️ No weights available yet from API",
                            flush=True,
                        )
            except Exception as e:
                print(
                    f"[VALIDATOR] ⚠️ Weight fetching/submission failed: {e}",
                    flush=True,
                )
                bt.logging.warning(f"Weight submission failed: {e}")

            # Wait before next cycle using dynamic interval
            print(
                f"[VALIDATOR] ⏱️ Waiting {polling_interval}s before next cycle...",
                flush=True,
            )
            time.sleep(polling_interval)

        except Exception as e:
            print(f"[VALIDATOR] ❌ Error in forward: {e}", flush=True)
            bt.logging.error(f"❌ Error in forward: {e}")
            traceback.print_exc()
            # Wait 5 minutes on error before retrying
            time.sleep(300)

    def resolve_uid_from_metagraph(self, hotkey: str) -> int:
        """Resolve a miner's UID from the metagraph using their hotkey."""
        for uid_idx in range(self.metagraph.n):
            if self.metagraph.hotkeys[uid_idx] == hotkey:
                return uid_idx
        return -1


if __name__ == "__main__":
    import argparse

    # Create parser and add all Bittensor base arguments
    parser = argparse.ArgumentParser(description="QuasarSubnet Validator")
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.Axon.add_args(parser)
    Validator.add_args(parser)  # Adds validator-specific Bittensor args

    # Add custom validator-specific arguments
    parser.add_argument(
        "--neuron.polling_interval",
        type=int,
        default=300,
        help="Polling interval in seconds (default: 300 = 5 minutes)",
    )

    # Create config - bt.Config will parse sys.argv automatically
    config = bt.Config(parser)

    # Parse args again to get custom arguments
    args = parser.parse_args()

    # Update config with custom args
    if hasattr(args, "polling_interval"):
        config.neuron.polling_interval = args.polling_interval

    # Run validator
    validator = Validator(config=config)

    print("[VALIDATOR] Starting validator loop...", flush=True)
    bt.logging.info("🚀 Starting validator loop...")
    validator.run()
