# FL PPO — Adaptive Federated Learning with PPO-based Client and Quantization Selection

[![Paper](https://img.shields.io/badge/Paper-Adaptive_FL_PPO-blue)](docs/SPEC.md)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2_CPU-orange)](requirements.txt)
[![Flower](https://img.shields.io/badge/Flower-1.7.0-purple)](requirements.txt)

> **"A Reinforcement Learning Framework for Joint Client Selection and Model Compression in Heterogeneous Federated Learning"**

A PPO (Proximal Policy Optimization) RL agent jointly decides **which clients** participate in each FL round and **what quantization precision** (FP32/FP16/INT8) they use — adapting to heterogeneous client resources in real time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  PPO Agent (Server-side)                │
│  State s_t = [client features × N] + [accuracy, Δ]     │
│  Action a_t = MultiDiscrete([4]*N) → skip/FP32/FP16/8  │
│  Reward r_t = α·ΔAcc − β·Dropout − γ·Var(Q)            │
└──────────────────────┬──────────────────────────────────┘
                       │ per-client quant assignment
                       ▼
┌─────────────────────────────────────────────────────────┐
│         Flower Server (FedAvgQuant strategy)            │
│  Aggregates FP32 · Tracks dropout · Writes round JSON  │
└────┬────────────────────────────────────────────────────┘
     │ gRPC (Docker) or in-process (local simulation)
     ├── Client 0  (strong:       2.0 CPU, 2048MB)
     ├── Client 1  (medium:       1.0 CPU, 1024MB)
     ├── ...
     └── Client 49 (extreme_weak: 0.25 CPU,  256MB) → skip/INT8
```

**Key constraints (paper-aligned, never modified):**
- Quantization is server→client only. Clients **always** return FP32 updates.
- INT8 uses static quantization with fallback chain: INT8 → FP16 → FP32.
- All-skip guard: if PPO skips all clients, server forces highest-reliability client with FP32.
- State dimension: `N × 7 + 3`. For 50 clients: `353`.

---

## Repository Structure

```
FL_PPO_PRJ/
├── src/
│   ├── fl/
│   │   ├── server.py          # In-process simulation loop
│   │   ├── client.py          # FlowerClient (NumPyClient, always FP32 return)
│   │   └── strategy.py        # FedAvgQuant (quant config + dropout tracking + JSON logging)
│   ├── rl/
│   │   ├── env.py             # FLEnv Gymnasium wrapper (1 step = 1 FL round)
│   │   ├── agent.py           # PPO training (SB3)
│   │   ├── state.py           # StateBuilder: N×7+3 vector
│   │   └── reward.py          # r = α·ΔAcc − β·Dropout − γ·Var(Q)
│   ├── compression/
│   │   ├── quantizer.py       # Dispatcher → fp32/fp16/int8
│   │   ├── int8.py            # Static INT8 + fallback chain (NEVER silent)
│   │   ├── fp16.py            # FP16 (model.half())
│   │   └── fp32.py            # FP32 (no-op baseline)
│   ├── data/
│   │   ├── cifar.py           # CIFAR-10 loader (224×224, hardguard)
│   │   └── partitioner.py     # IID + Dirichlet(α=0.1) partition logic
│   ├── models/
│   │   ├── mobilenetv2.py     # MobileNetV2 (10 classes)
│   │   └── trainer.py         # train_local(), evaluate()
│   ├── heterogeneity/
│   │   ├── profiles.py        # 4 tiers: strong/medium/weak/extreme_weak
│   │   └── dropout.py         # DropoutTracker
│   └── common/config.py       # YAML → typed dataclasses
├── scripts/
│   ├── download_and_partition.py    # One-time: CIFAR-10 + 50-client partition JSON
│   ├── plot_results.py              # 5 paper-aligned plots from round logs
│   ├── summarize_run.py             # summary.json + SPEC §8 compliance check
│   └── smoke_test_docker.sh         # Docker smoke test + auto-verification
├── configs/
│   ├── exp1_smoke.yaml              # 3 clients, 2 rounds (basic smoke)
│   ├── exp1_ppo_smoke.yaml          # 3 clients, 2 rounds, adaptive PPO
│   ├── exp_50rounds_3clients.yaml   # 3 clients, 50 rounds (50-round proof)
│   ├── exp_docker_smoke.yaml        # Docker: 3 clients, 2 rounds
│   ├── exp_docker_grpc_50rounds.yaml# Docker: 3 clients, 50 rounds (gRPC proof)
│   ├── exp_docker_50clients.yaml    # Docker: 50 clients, 50 rounds (paper scale)
│   └── exp[1-10]*.yaml             # Individual paper experiments
├── docker/
│   ├── Dockerfile.base             # Python 3.11 + PyTorch CPU + deps
│   ├── Dockerfile.server           # server_main.py entrypoint
│   └── Dockerfile.client           # client_main.py entrypoint
├── server_main.py                  # gRPC server entrypoint (Docker)
├── client_main.py                  # gRPC client entrypoint (Docker)
├── docker-compose.yml              # 1 server + 50 heterogeneous clients
├── docker-compose.smoke.yml        # 1 server + 3 clients (safe smoke)
├── docker-compose.grpc_50rounds.yml# 1 server + 3 clients, 50 rounds (gRPC proof)
├── RESOURCE_DISTRIBUTION.md        # Client tier specs + memory safety
└── data/                           # Created by download_and_partition.py
    ├── cifar10/
    ├── partitions_50_iid.json
    └── partitions_50_dirichlet0.1.json
```

---

## 1. Environment Setup

```bash
cd /root/FL_PPO_PRJ

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate    # Linux/macOS

# Install PyTorch CPU (~200 MB download)
pip install torch==2.2.2+cpu torchvision==0.17.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Setup (required once before any run)

```bash
# Download CIFAR-10 + generate IID and non-IID partition files
python3 scripts/download_and_partition.py --n-clients 50 --partition iid --seed 42
python3 scripts/download_and_partition.py --n-clients 50 --partition dirichlet \
    --dirichlet-alpha 0.1 --seed 42
```

Creates:
- `data/cifar10/` — raw CIFAR-10 files
- `data/partitions_50_iid.json` — 50 clients × 1,000 train indices
- `data/partitions_50_dirichlet0.1.json` — non-IID partition

---

## 3. Local Runs (No Docker)

### Basic smoke test
```bash
python3 src/fl/server.py --config configs/exp1_smoke.yaml
```

### 50-round proof (3 clients, 50 rounds)
```bash
python3 src/fl/server.py --config configs/exp_50rounds_3clients.yaml
```

### PPO adaptive mode (paper method)
```bash
python3 src/rl/agent.py --config configs/exp1_ppo_smoke.yaml
```

### All paper experiments
```bash
# Exp1: Baseline (homogeneous IID, FP32 only)
python3 src/fl/server.py --config configs/exp1_homogeneous_iid.yaml

# Exp2: Weak client dropout with FP32
python3 src/fl/server.py --config configs/exp2_heterogeneous_iid.yaml

# Exp3: Non-IID Dirichlet α=0.1
python3 src/fl/server.py --config configs/exp3_heterogeneous_noniid.yaml

# Exp6: FP16 for weakest client (rescued survival)
python3 src/fl/server.py --config configs/exp6_fp16_weakest.yaml

# Exp7: INT8 for weakest client
python3 src/fl/server.py --config configs/exp7_int8_weakest.yaml

# Exp9: Uniform INT8 all clients
python3 src/fl/server.py --config configs/exp9_uniform_int8.yaml

# Exp10: Mixed FP32/FP16/INT8 oracle
python3 src/fl/server.py --config configs/exp10_mixed_quant.yaml

# PPO Adaptive (full 50 rounds, paper method)
python3 src/rl/agent.py --config configs/exp_ppo_adaptive.yaml
```

---

## 4. Docker Deployment

### Build images
```bash
docker compose -f docker-compose.smoke.yml build
```

### Docker smoke test (3 clients, 2 rounds — safe on 22 GB)
```bash
bash scripts/smoke_test_docker.sh

# Or manually:
docker compose -f docker-compose.smoke.yml up --abort-on-container-exit
docker compose -f docker-compose.smoke.yml down
```

### Docker 50-round gRPC validation (3 clients, 50 rounds)
```bash
docker compose -f docker-compose.grpc_50rounds.yml up --build --abort-on-container-exit
docker compose -f docker-compose.grpc_50rounds.yml down
```

### Full 50-client experiment (paper scale)
```bash
# ⚠️ Read RESOURCE_DISTRIBUTION.md before running
# ⚠️ Requires stochastic dropout_mode (set in config)
docker compose up --abort-on-container-exit 2>&1 | tee outputs/run_50clients.log
docker compose down
```

---

## 5. Post-Run Analysis

```bash
# After any run, find your output directory:
ls outputs/metrics/

# Generate 5 plots from real round logs
python3 scripts/plot_results.py outputs/metrics/run_<id>

# Generate summary.json with SPEC §8 compliance verification
python3 scripts/summarize_run.py outputs/metrics/run_<id>
```

---

## 6. Expected Outputs

After a 50-round run:

```
outputs/
├── metrics/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── round_001.json   ← 8 required fields (SPEC §8)
│       ├── round_002.json
│       ├── ...
│       ├── round_050.json
│       └── summary.json     ← "num_rounds": 50, SPEC compliance check
└── plots/
    └── run_YYYYMMDD_HHMMSS/
        ├── accuracy.png
        ├── loss.png
        ├── dropout.png
        ├── quant_distribution.png
        └── reward.png
```

Each `round_NNN.json` (SPEC §8 required fields):
```json
{
  "round": 1,
  "selected_clients": ["0", "1", "2"],
  "quant_assignments": {"0": 32, "1": 32, "2": 32},
  "actual_quant_method": {"0": "fp32", "1": "fp32", "2": "fp32"},
  "dropout_clients": [],
  "dropout_fraction": 0.0,
  "global_accuracy": 0.087,
  "accuracy_delta": 0.0
}
```

`summary.json`:
```json
{
  "num_rounds": 50,
  "status": "complete",
  "final_accuracy": 0.312,
  "paper_compliance": {
    "rounds_with_all_required_fields": 50,
    "rounds_missing_fields": 0
  }
}
```

---

## Client Resource Tiers (Docker)

| Tier | Count | IDs | CPUs | RAM | Behavior |
|------|-------|-----|------|-----|----------|
| Strong | 10 (20%) | 0–9 | 2.0 | 2048 MB | Always completes FP32 |
| Medium | 20 (40%) | 10–29 | 1.0 | 1024 MB | FP32/FP16 safe; INT8 marginal |
| Weak | 15 (30%) | 30–44 | 0.5 | 512 MB | FP32 OOM risk; needs FP16/INT8 |
| Extreme-weak | 5 (10%) | 45–49 | 0.25 | 256 MB | Almost always fails at FP32 |

See `RESOURCE_DISTRIBUTION.md` for full memory safety analysis.

---

## Paper Alignment Summary

| Requirement | Status | Location |
|-------------|--------|---------|
| state_dim = N×7+3 | ✅ | `src/rl/state.py`, `src/common/config.py` |
| action = MultiDiscrete([4]×N) | ✅ | `src/rl/env.py` |
| All-skip guard | ✅ | `src/rl/env.py` |
| r = α·ΔAcc − β·Dropout − γ·Var(Q) | ✅ | `src/rl/reward.py` |
| INT8 fallback chain | ✅ | `src/compression/quantizer.py`, `int8.py` |
| CIFAR-10, MobileNetV2, 224×224 | ✅ | `src/data/cifar.py`, `src/fl/client.py` |
| SPEC §8 per-round JSON fields | ✅ | `src/fl/strategy.py` |
| Exp08 guard | ✅ | `raise RuntimeError("Exp08 disabled")` |

---

## Versions

| Package | Version |
|---------|---------|
| Python | 3.11+ (3.12.3 on KVM VM) |
| flwr | 1.7.0 |
| torch | 2.2.2+cpu |
| torchvision | 0.17.2+cpu |
| stable-baselines3 | 2.2.1 |
| gymnasium | 0.29.1 |
| numpy | 1.26.4 |
