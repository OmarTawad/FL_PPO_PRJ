#!/usr/bin/env bash
# scripts/run_smoke_fl.sh — Run Phase 6 FL smoke test (2 rounds, 3 clients, FP32)
#
# Usage: bash scripts/run_smoke_fl.sh
# From project root: ~/FL_joint_client

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo " Adaptive FL — Phase 6 Smoke Test"
echo " Config: configs/exp1_smoke.yaml"
echo " Rounds: 2  |  Clients: 3  |  Quant: FP32"
echo "=========================================="

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[INFO] venv activated: $(which python)"
fi

# Ensure output dirs exist (server.py also creates them, but belt-and-suspenders)
mkdir -p outputs/metrics outputs/plots outputs/checkpoints

# Run the simulation
python src/fl/server.py --config configs/exp1_smoke.yaml

echo ""
echo "=========================================="
echo " Smoke test complete."
echo " JSON logs:"
ls -la outputs/metrics/run_*/round_*.json 2>/dev/null | tail -10 || echo "  (no logs found)"
echo "=========================================="
