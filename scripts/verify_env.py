#!/usr/bin/env python3
"""
scripts/verify_env.py — Environment verification for FL + PPO project
Paper: Adaptive FL with PPO-based Client and Quantization Selection

Checks:
  - All required packages importable at correct versions
  - CPU/RAM system info (psutil)
  - PyTorch CPU-only mode confirmed (no CUDA required)
  - Gymnasium + SB3 compatibility
  - No CIFAR-10 download (Stage 1 only)

Exit codes:
  0 — all checks passed
  1 — one or more checks failed
"""

import sys
import os
import platform
import importlib
from typing import Tuple, List

# ─── Helpers ──────────────────────────────────────────────────────────────────

GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
RED    = "\033[0;31m"
CYAN   = "\033[0;36m"
RESET  = "\033[0m"

def ok(msg: str):   print(f"{GREEN}  [OK]{RESET}    {msg}")
def warn(msg: str): print(f"{YELLOW}  [WARN]{RESET}  {msg}")
def fail(msg: str): print(f"{RED}  [FAIL]{RESET}  {msg}")
def step(msg: str): print(f"\n{CYAN}{'─'*60}{RESET}\n{CYAN}  {msg}{RESET}")

errors: List[str] = []

def check(condition: bool, ok_msg: str, fail_msg: str, is_warning: bool = False):
    if condition:
        ok(ok_msg)
    else:
        if is_warning:
            warn(fail_msg)
        else:
            fail(fail_msg)
            errors.append(fail_msg)

# ─── Step 1: System Info ──────────────────────────────────────────────────────
step("1/6  System Information")

print(f"  OS:             {platform.system()} {platform.release()}")
print(f"  Architecture:   {platform.machine()}")
print(f"  Python:         {sys.version}")
print(f"  Python path:    {sys.executable}")

# RAM
try:
    import psutil
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    total_gb  = vm.total  / 1024**3
    avail_gb  = vm.available / 1024**3
    swap_gb   = swap.total / 1024**3
    cpu_count = psutil.cpu_count(logical=True)
    cpu_phys  = psutil.cpu_count(logical=False)

    ok(f"psutil imported")
    print(f"  CPU logical:    {cpu_count}  physical: {cpu_phys}")
    print(f"  RAM total:      {total_gb:.2f} GB")
    print(f"  RAM available:  {avail_gb:.2f} GB  ({vm.percent:.1f}% used)")
    print(f"  Swap total:     {swap_gb:.2f} GB")

    check(avail_gb >= 1.5,
          f"Sufficient RAM available ({avail_gb:.2f} GB)",
          f"Low RAM ({avail_gb:.2f} GB free). Some experiments may use swap.",
          is_warning=True)
except ImportError:
    fail("psutil not installed — run setup.sh first")
    errors.append("psutil missing")

# ─── Step 2: Core Package Versions ───────────────────────────────────────────
step("2/6  Core Package Versions")

REQUIRED: List[Tuple[str, str, str]] = [
    # (import_name, display_name, min_version)
    ("torch",               "torch",               "2.1.0"),
    ("torchvision",         "torchvision",          "0.16.0"),
    ("flwr",                "flwr",                "1.7.0"),
    ("stable_baselines3",   "stable-baselines3",   "2.2.0"),
    ("gymnasium",           "gymnasium",           "0.29.0"),
    ("numpy",               "numpy",               "1.26.0"),
    ("pandas",              "pandas",              "2.0.0"),
    ("matplotlib",          "matplotlib",          "3.8.0"),
    ("psutil",              "psutil",              "5.9.0"),
    ("yaml",                "pyyaml",              "6.0.0"),
    ("tqdm",                "tqdm",                "4.66.0"),
    ("click",               "click",               "8.1.0"),
    ("prometheus_client",   "prometheus-client",   "0.19.0"),
    ("shimmy",              "shimmy",              "1.3.0"),
]

from packaging.version import Version  # noqa: E402

def check_package(import_name: str, display_name: str, min_ver: str):
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "unknown")
        try:
            meets = Version(ver) >= Version(min_ver)
        except Exception:
            meets = True  # can't parse, assume ok
        if meets:
            ok(f"{display_name:<24} {ver}")
        else:
            fail(f"{display_name:<24} {ver}  (requires >= {min_ver})")
            errors.append(f"{display_name} version too old: {ver}")
    except ImportError as e:
        fail(f"{display_name:<24} NOT FOUND — {e}")
        errors.append(f"{display_name} not importable")

# packaging itself first
try:
    from packaging.version import Version
    ok(f"{'packaging':<24} {importlib.import_module('packaging').__version__}")
except ImportError:
    warn("packaging not installed — version comparisons skipped")
    # redefine to skip version checks
    def Version(v): return v  # type: ignore[misc]

for args in REQUIRED:
    check_package(*args)

# ─── Step 3: PyTorch Capabilities ─────────────────────────────────────────────
step("3/6  PyTorch Capabilities")

try:
    import torch

    # CPU only for this VM
    check(not torch.cuda.is_available(),
          "CUDA not available (expected: CPU-only build)",
          "CUDA available — are you using the GPU build? Check memory!", is_warning=True)

    # Static INT8 quantization support
    try:
        import torch.quantization
        ok("torch.quantization module available (Static INT8 supported)")
    except ImportError:
        fail("torch.quantization not available — INT8 quant will not work")
        errors.append("torch.quantization missing")

    # Verify a tiny forward pass works
    try:
        x = torch.randn(1, 3, 32, 32)
        ok(f"torch.randn() forward pass OK  (tensor shape {list(x.shape)})")
    except Exception as e:
        fail(f"torch basic operation failed: {e}")
        errors.append("torch basic op failed")

    # Half precision
    try:
        x_half = x.half()
        ok(f"FP16 (.half()) conversion OK")
    except Exception as e:
        warn(f"FP16 conversion failed: {e}")

except Exception as e:
    fail(f"pytorch import/test failed: {e}")
    errors.append("torch broken")

# ─── Step 4: Flower Sanity Check ──────────────────────────────────────────────
step("4/6  Flower (flwr) Sanity Check")

try:
    import flwr as fl
    # Check key submodules
    from flwr.server import ServerConfig   # noqa: F401
    from flwr.common import ndarrays_to_parameters  # noqa: F401
    from flwr.server.strategy import FedAvg  # noqa: F401
    ok(f"flwr {fl.__version__} — server, common, strategy imports OK")
except ImportError as e:
    fail(f"flwr import failed: {e}")
    errors.append("flwr broken")

# ─── Step 5: SB3 + Gymnasium Check ───────────────────────────────────────────
step("5/6  Stable-Baselines3 + Gymnasium")

try:
    import gymnasium as gym
    import numpy as np

    # Create a simple env to check compatibility
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
    ok(f"gymnasium {gym.__version__} — CartPole-v1 step OK")

    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env  # noqa: F401
    ok(f"stable-baselines3 {sb3.__version__} — PPO importable")

except Exception as e:
    fail(f"gymnasium/SB3 check failed: {e}")
    errors.append("gymnasium/SB3 broken")

# ─── Step 6: Project Structure Check ─────────────────────────────────────────
step("6/6  Project Structure")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED_DIRS = [
    "docs", "configs", "scripts",
    "src/fl", "src/models", "src/data",
    "src/compression", "src/heterogeneity",
    "src/rl", "src/experiments",
    "docker",
    "outputs/logs", "outputs/metrics",
    "outputs/plots", "outputs/checkpoints",
]
REQUIRED_FILES = [
    "docs/SPEC.md",
    "docs/RUNSHEET.md",
    "requirements.txt",
    "scripts/setup.sh",
    "scripts/verify_env.py",
]

for d in REQUIRED_DIRS:
    path = os.path.join(PROJECT_ROOT, d)
    check(os.path.isdir(path),
          f"Dir  exists: {d}",
          f"Dir  missing: {d}")

for f in REQUIRED_FILES:
    path = os.path.join(PROJECT_ROOT, f)
    check(os.path.isfile(path),
          f"File exists: {f}",
          f"File missing: {f}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{CYAN}{'═'*60}{RESET}")

if errors:
    print(f"{RED}  VERIFICATION FAILED — {len(errors)} error(s):{RESET}")
    for e in errors:
        print(f"{RED}    • {e}{RESET}")
    print(f"{CYAN}{'═'*60}{RESET}\n")
    sys.exit(1)
else:
    print(f"{GREEN}  ALL CHECKS PASSED — environment is ready!{RESET}")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"{CYAN}{'═'*60}{RESET}\n")
    sys.exit(0)
