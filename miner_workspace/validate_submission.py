#!/usr/bin/env python3
"""
QUASAR-SUBNET Submission Validator
===================================
Pre-validates your kernel code EXACTLY as the validator does before submission.

Checks:
1. Required imports in chunk.py (5 mandatory FLA utility imports)
2. No forbidden imports (fla.ops.gla, fla.ops.kda)
3. No dangerous calls (exec, eval, os.system, subprocess, etc.)
4. No dangerous module imports (importlib, ctypes, socket, etc.)
5. All target files exist and parse correctly
6. Runs the actual benchmark

Usage:
    python3 validate_submission.py                         # Validate default repo
    python3 validate_submission.py --repo-path /path/to/fla  # Custom path
"""

import ast
import os
import sys
import glob as _glob
import argparse


# ── Validator constants (from neurons/validator.py PerformanceValidator) ──────

REQUIRED_IMPORTS_CHUNK_PY = [
    "from fla.utils import autocast_custom_bwd",
    "from fla.utils import autocast_custom_fwd",
    "from fla.utils import autotune_cache_kwargs",
    "from fla.utils import check_shared_mem",
    "from fla.utils import input_guard",
]

FORBIDDEN_IMPORTS = [
    "from fla.ops.gla",
    "from fla.ops.kda",
    "import fla.ops.gla",
    "import fla.ops.kda",
]

DANGEROUS_CALLS = {
    "__import__", "exec", "eval", "compile",
    "os.system", "os.popen",
    "os.exec", "os.execl", "os.execle", "os.execlp", "os.execlpe",
    "os.execv", "os.execve", "os.execvp", "os.execvpe",
    "os.spawn", "os.spawnl", "os.spawnle",
    "subprocess.run", "subprocess.call", "subprocess.Popen",
    "subprocess.check_output", "subprocess.check_call",
}

DANGEROUS_IMPORT_MODULES = {
    "importlib", "ctypes", "socket", "http",
    "urllib", "requests", "paramiko", "fabric",
}

TARGET_FILES = [
    "chunk.py",
    "chunk_intra_token_parallel.py",
    "forward_substitution.py",
    "fused_recurrent.py",
    "gate.py",
]


def _ast_imported_names(tree) -> set:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.add(module)
            for alias in node.names:
                names.add(f"{module}.{alias.name}" if module else alias.name)
    return names


def _ast_called_names(tree) -> set:
    calls = set()
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


def validate_imports(repo_path: str) -> tuple:
    """Validate miner repo using AST parsing — exact replica of validator logic."""
    errors = []
    warnings = []

    # Phase 1: deep-scan .py files under fla/ for dangerous patterns
    fla_root = os.path.join(repo_path, "fla")
    all_py = (
        _glob.glob(os.path.join(fla_root, "**", "*.py"), recursive=True)
        if os.path.isdir(fla_root)
        else []
    )

    print(f"\n  Phase 1: Scanning {len(all_py)} Python files under fla/...")

    for py_path in all_py:
        rel = os.path.relpath(py_path, repo_path)
        try:
            with open(py_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=rel)
        except SyntaxError:
            errors.append(f"  {rel}: SyntaxError - file cannot be parsed")
            continue

        # Dangerous calls
        called = _ast_called_names(tree)
        for dc in DANGEROUS_CALLS:
            if dc in called:
                errors.append(f"  {rel}: Dangerous call detected: {dc}()")

        # Dangerous module imports
        imported = _ast_imported_names(tree)
        for mod in DANGEROUS_IMPORT_MODULES:
            for imp in imported:
                if imp == mod or imp.startswith(f"{mod}."):
                    errors.append(f"  {rel}: Dangerous module imported: {imp}")

    # Phase 2: target-file checks
    quasar_dir = os.path.join(repo_path, "fla/ops/quasar")
    print(f"  Phase 2: Checking target files in {quasar_dir}...")

    for filename in TARGET_FILES:
        file_path = os.path.join(quasar_dir, filename)
        if not os.path.exists(file_path):
            errors.append(f"  Missing file: {filename}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            errors.append(f"  {filename}: SyntaxError - cannot parse")
            continue

        imported = _ast_imported_names(tree)

        # Forbidden imports
        for forbidden in FORBIDDEN_IMPORTS:
            mod = forbidden.replace("from ", "").replace("import ", "").strip()
            for imp in imported:
                if imp == mod or imp.startswith(f"{mod}."):
                    errors.append(f"  {filename}: Forbidden import found: {forbidden}")

        # Required imports (only for chunk.py)
        if filename == "chunk.py":
            for required in REQUIRED_IMPORTS_CHUNK_PY:
                import_name = required.split("import")[-1].strip()
                expected = f"fla.utils.{import_name}"
                if expected not in imported:
                    errors.append(f"  {filename}: Missing required import: {required}")

        # Info: file size
        lines = source.count("\n") + 1
        warnings.append(f"  {filename}: {lines} lines, {len(source)} bytes")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate miner submission")
    parser.add_argument("--repo-path", type=str, default=None,
                        help="Path to flash-linear-attention repo")
    parser.add_argument("--benchmark", action="store_true",
                        help="Also run the benchmark after validation")
    parser.add_argument("--seq-len", type=int, default=100_000,
                        help="Sequence length for benchmark (default: 100000)")
    args = parser.parse_args()

    # Find repo path
    repo_path = args.repo_path
    if not repo_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_path = os.path.join(script_dir, "flash-linear-attention")
    if not os.path.isdir(repo_path):
        print(f"ERROR: Repository not found at {repo_path}")
        print("Run setup_miner.sh first or specify --repo-path")
        sys.exit(1)

    print("="*60)
    print("  QUASAR-SUBNET Submission Validator")
    print("="*60)
    print(f"  Repository: {repo_path}")

    errors, warnings = validate_imports(repo_path)

    # Report
    print(f"\n  --- File Info ---")
    for w in warnings:
        print(w)

    if errors:
        print(f"\n  --- ERRORS ({len(errors)}) ---")
        for e in errors:
            print(f"  FAIL: {e}")
        print(f"\n  VALIDATION FAILED - {len(errors)} error(s)")
        print("  These errors will cause score 0.0 or rejection by the validator.")
        sys.exit(1)
    else:
        print(f"\n  VALIDATION PASSED")
        print("  All imports, security checks, and file requirements are satisfied.")

    # Run benchmark if requested
    if args.benchmark:
        print("\n  Running benchmark...")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from benchmark_local import run_validator_benchmark
        run_validator_benchmark(args.seq_len)


if __name__ == "__main__":
    main()
