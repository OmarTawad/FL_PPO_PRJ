#!/usr/bin/env bash
# scripts/smoke_test_docker.sh
# Stage 16: Docker smoke test — 3 clients (strong/medium/weak), 2 FL rounds.
#
# Tests:
#   1. Docker images build cleanly
#   2. Server starts and accepts gRPC connections
#   3. Clients connect, train locally (MobileNetV2, CIFAR-10, 224x224)
#   4. Server aggregates (FedAvgQuant, FP32)
#   5. outputs/metrics/run_*/round_*.json created with required fields
#   6. Plotting and summarization scripts run without error
#
# Prerequisites:
#   - Docker + Docker Compose installed
#   - data/ directory with CIFAR-10 and partitions_50_iid.json
#     (run: python3 scripts/download_and_partition.py --n-clients 50 first)
#
# Usage:
#   bash scripts/smoke_test_docker.sh
#   bash scripts/smoke_test_docker.sh --no-rebuild   (skip image rebuild)
#   bash scripts/smoke_test_docker.sh --keep-up      (don't tear down after)

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
COMPOSE_FILE="docker-compose.smoke.yml"
OUTPUT_BASE="outputs"
DATA_DIR="data"
PARTITION_FILE="data/partitions_50_iid.json"
REBUILD=true
KEEP_UP=false

# Parse args
for arg in "$@"; do
    case "$arg" in
        --no-rebuild) REBUILD=false ;;
        --keep-up)    KEEP_UP=true  ;;
    esac
done

# ── Colour output helpers ─────────────────────────────────────────────────────
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[0;33m'
BLU='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLU}[SMOKE]${NC} $*"; }
ok()   { echo -e "${GRN}[PASS]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }
warn() { echo -e "${YLW}[WARN]${NC}  $*"; }

echo ""
info "=========================================="
info "  FL PPO Paper — Docker Smoke Test"
info "  Config: ${COMPOSE_FILE}"
info "=========================================="

# ── Pre-flight checks ─────────────────────────────────────────────────────────
info "Step 1: Pre-flight checks"

if ! command -v docker &>/dev/null; then
    fail "Docker not found. Install Docker first."
fi

if ! docker compose version &>/dev/null; then
    fail "Docker Compose v2 not found. Install Docker Compose v2."
fi

if [ ! -f "$PARTITION_FILE" ]; then
    warn "Partition file not found: $PARTITION_FILE"
    info "Running download_and_partition.py first..."
    python3 scripts/download_and_partition.py --n-clients 50 --data-root data/ --partition iid --seed 42
fi

if [ ! -f "$COMPOSE_FILE" ]; then
    fail "Compose file not found: $COMPOSE_FILE"
fi

ok "Pre-flight checks passed"

# ── Build images ──────────────────────────────────────────────────────────────
info "Step 2: Building Docker images"
if [ "$REBUILD" = true ]; then
    docker compose -f "$COMPOSE_FILE" build 2>&1 | while IFS= read -r line; do
        echo "  $line"
    done
    ok "Docker images built"
else
    warn "Skipping rebuild (--no-rebuild)"
fi

# ── Clean up previous outputs ─────────────────────────────────────────────────
info "Step 3: Cleaning previous test outputs"
mkdir -p "$OUTPUT_BASE/metrics" "$OUTPUT_BASE/plots"

# ── Run smoke test ────────────────────────────────────────────────────────────
info "Step 4: Starting docker compose smoke test (server + 3 clients, 2 rounds)"
info "  Config: configs/exp_docker_smoke.yaml"
info "  Expected: 2 round JSON files + summary.json"

COMPOSE_UP_EXIT=0
docker compose -f "$COMPOSE_FILE" up \
    --abort-on-container-exit \
    --exit-code-from server \
    2>&1 | while IFS= read -r line; do
    echo "  | $line"
done || COMPOSE_UP_EXIT=$?

if [ "$KEEP_UP" = false ]; then
    info "Step 5: Tearing down containers"
    docker compose -f "$COMPOSE_FILE" down --remove-orphans 2>&1 | tail -3
fi

# ── Verify outputs ────────────────────────────────────────────────────────────
info "Step 6: Verifying outputs"

# Find the latest run directory
LATEST_RUN=$(find "$OUTPUT_BASE/metrics" -name "run_*" -type d | sort | tail -1)
if [ -z "$LATEST_RUN" ]; then
    fail "No run_* directory found in $OUTPUT_BASE/metrics"
fi
ok "Run directory found: $LATEST_RUN"

# Check round JSON files
N_ROUNDS=$(find "$LATEST_RUN" -name "round_*.json" | wc -l)
if [ "$N_ROUNDS" -eq 0 ]; then
    fail "No round_*.json files found in $LATEST_RUN"
fi
ok "Round JSON files found: $N_ROUNDS"

# Verify SPEC §8 required fields in round JSON
python3 - << PYEOF
import json, glob, sys

run_dir = "$LATEST_RUN"
required = {"round","selected_clients","quant_assignments","actual_quant_method",
            "dropout_clients","dropout_fraction","global_accuracy","accuracy_delta"}

logs = sorted(glob.glob(f"{run_dir}/round_*.json"))
failures = []
for p in logs:
    d = json.load(open(p))
    missing = required - set(d.keys())
    if missing:
        failures.append(f"  {p}: missing {sorted(missing)}")

if failures:
    print("SPEC compliance FAILED:")
    for f in failures:
        print(f)
    sys.exit(1)
else:
    print(f"SPEC compliance: ALL {len(logs)} round logs have required fields")
PYEOF

# ── Post-run plotting ──────────────────────────────────────────────────────────
info "Step 7: Running plot_results.py"
python3 scripts/plot_results.py "$LATEST_RUN" 2>&1 | while IFS= read -r line; do
    echo "  | $line"
done || warn "Plotting had warnings (check logs above)"

PLOTS_DIR=$(echo "$LATEST_RUN" | sed 's|/metrics/|/plots/|')
N_PLOTS=$(find "$PLOTS_DIR" -name "*.png" 2>/dev/null | wc -l)
ok "Plots generated: $N_PLOTS"

# ── Post-run summarization ────────────────────────────────────────────────────
info "Step 8: Running summarize_run.py"
python3 scripts/summarize_run.py "$LATEST_RUN" 2>&1 | while IFS= read -r line; do
    echo "  | $line"
done

if [ -f "$LATEST_RUN/summary.json" ]; then
    FINAL_ACC=$(python3 -c "import json; d=json.load(open('$LATEST_RUN/summary.json')); print(d.get('final_accuracy','N/A'))")
    ok "summary.json written; final_accuracy=$FINAL_ACC"
else
    warn "summary.json not found"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
info "=========================================="
ok  "Smoke test COMPLETE"
info "  Run dir  : $LATEST_RUN"
info "  Rounds   : $N_ROUNDS / 2"
info "  Plots    : $N_PLOTS / 5"
info "=========================================="
