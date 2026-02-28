#!/usr/bin/env bash
# =============================================================================
# scripts/setup.sh — Environment setup for FL + PPO project
# Paper: Adaptive FL with PPO-based Client and Quantization Selection
# Target: Ubuntu 22.04, Python 3.10, CPU-only VM (3.8 GB RAM)
# =============================================================================
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh
#   source .venv/bin/activate
#   python scripts/verify_env.py
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
VENV_DIR=".venv"
PYTHON="python3"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="${PROJECT_ROOT}/requirements.txt"

# ANSI colors
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
RESET="\033[0m"

echo_step() { echo -e "${CYAN}[SETUP]${RESET} $1"; }
echo_ok()   { echo -e "${GREEN}[OK]${RESET}    $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${RESET}  $1"; }
echo_err()  { echo -e "${RED}[ERROR]${RESET} $1"; }

cd "${PROJECT_ROOT}"

# ── Step 0: System checks ─────────────────────────────────────────────────────
echo_step "Step 0: Checking system prerequisites..."

# Check Python 3.10+
PY_VERSION=$("${PYTHON}" --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "${PY_VERSION}" | cut -d. -f1)
PY_MINOR=$(echo "${PY_VERSION}" | cut -d. -f2)

if [ "${PY_MAJOR}" -lt 3 ] || [ "${PY_MINOR}" -lt 10 ]; then
    echo_err "Python 3.10+ required. Found: ${PY_VERSION}"
    echo_err "On Ubuntu 22.04: sudo apt install python3.10 python3.10-venv"
    exit 1
fi
echo_ok "Python ${PY_VERSION}"

# Check available RAM (warn if < 2.5 GB free)
FREE_MB=$(free -m | awk '/^Mem:/{print $7}')
echo_step "Available RAM: ${FREE_MB} MB"
if [ "${FREE_MB}" -lt 1500 ]; then
    echo_warn "Low free RAM (${FREE_MB} MB). Consider closing other apps."
    echo_warn "Installation may be slow due to swap usage."
fi

# Check pip
"${PYTHON}" -m pip --version > /dev/null 2>&1 || {
    echo_err "pip not found. Run: sudo apt install python3-pip"
    exit 1
}
echo_ok "pip available"

# ── Step 1: Create virtual environment ───────────────────────────────────────
echo_step "Step 1: Creating virtual environment in ${VENV_DIR}..."

if [ -d "${VENV_DIR}" ]; then
    echo_warn "Existing .venv found. Reusing it."
    echo_warn "To start fresh: rm -rf .venv && ./scripts/setup.sh"
else
    "${PYTHON}" -m venv "${VENV_DIR}" --prompt "fl-ppo"
    echo_ok "Virtual environment created at ${VENV_DIR}"
fi

# Activate venv for the rest of this script
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
echo_ok "Activated: $(which python)"

# ── Step 2: Upgrade pip and wheel ─────────────────────────────────────────────
echo_step "Step 2: Upgrading pip, setuptools, wheel..."
pip install --quiet --upgrade pip setuptools wheel
echo_ok "pip $(pip --version | awk '{print $2}')"

# ── Step 3: Install requirements ──────────────────────────────────────────────
echo_step "Step 3: Installing requirements from ${REQ_FILE}..."
echo_step "  NOTE: torch+cpu wheels are ~200 MB. This may take 5-10 min on first run."
echo_step "  RAM note: pip itself may use 500 MB. Monitor with: watch -n1 free -m"

# Use --no-cache-dir to avoid disk cache eating into limited storage
# Use one-shot install to let pip resolve deps correctly with the --extra-index-url
pip install \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r "${REQ_FILE}" 2>&1 | tee /tmp/fl_ppo_install.log | grep -E "(Successfully|ERROR|WARNING|already)" || true

# Check for install errors
if grep -q "^ERROR" /tmp/fl_ppo_install.log; then
    echo_err "Installation errors detected. Check /tmp/fl_ppo_install.log"
    echo_err "Common fix: ensure internet access and sufficient disk space."
    exit 1
fi
echo_ok "Requirements installed."

# ── Step 4: Create required output directories ────────────────────────────────
echo_step "Step 4: Ensuring output directories exist..."
mkdir -p \
    outputs/logs \
    outputs/metrics \
    outputs/plots \
    outputs/checkpoints \
    configs \
    docker
echo_ok "Output directories ready."

# ── Step 5: Create __init__.py stubs for src packages ────────────────────────
echo_step "Step 5: Creating src package __init__.py stubs..."
for pkg in src src/fl src/models src/data src/compression src/heterogeneity src/rl src/experiments; do
    init_file="${pkg}/__init__.py"
    if [ ! -f "${init_file}" ]; then
        echo "# ${pkg} package" > "${init_file}"
        echo_ok "  Created ${init_file}"
    fi
done

# ── Step 6: Print installed versions ─────────────────────────────────────────
echo_step "Step 6: Installed package versions:"
echo ""
python -c "
import sys
import importlib

packages = [
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('flwr', 'flwr'),
    ('stable_baselines3', 'stable_baselines3'),
    ('gymnasium', 'gymnasium'),
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('matplotlib', 'matplotlib'),
    ('psutil', 'psutil'),
    ('yaml', 'pyyaml'),
    ('tqdm', 'tqdm'),
    ('click', 'click'),
]

print(f'  Python:         {sys.version.split()[0]}')
for import_name, pkg_name in packages:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, '__version__', 'unknown')
        print(f'  {pkg_name:<22} {ver}')
    except ImportError:
        print(f'  {pkg_name:<22} NOT INSTALLED')
"
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo -e "${GREEN}════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}  Setup complete!${RESET}"
echo -e "${GREEN}════════════════════════════════════════════════════${RESET}"
echo ""
echo "  Next steps:"
echo "    source .venv/bin/activate"
echo "    python scripts/verify_env.py"
echo ""
