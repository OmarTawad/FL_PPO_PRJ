# SPEC.md — Adaptive FL with PPO-based Client and Quantization Selection

**Version:** 0.1.0  
**Date:** 2026-02-28  
**Status:** Stage 0 — Specification Locked

---

## 1. Overview

This project implements the paper: **"Adaptive Federated Learning with PPO-based Client and Quantization Selection"**.

The core idea: a PPO RL agent acts as the FL orchestrator. Each round it jointly selects:
- **Which clients** participate (`S_t ⊆ C`)
- **What quantization level** each selected client uses (`Q_i^t ∈ {32, 16, 8}`)

The agent observes device heterogeneity, historical dropout, and model accuracy to learn an adaptive policy that maximizes accuracy while tolerating resource-constrained clients.

---

## 2. System Architecture Summary

```
┌──────────────────────────────────────────────────────┐
│                   PPO Agent (RL)                     │
│  State s_t → Policy π → Action a_t = (S_t, Q^t)     │
└────────────────────┬─────────────────────────────────┘
                     │ action
                     ▼
┌──────────────────────────────────────────────────────┐
│             Flower FL Server (FedAvg-based)          │
│  Selects clients per S_t, sends quant config Q_i^t   │
└──────┬────────────────────────────────────────┬──────┘
       │                                        │
       ▼                                        ▼
┌──────────────┐                      ┌────────────────┐
│  Client i    │   …   N clients      │  Client N      │
│  Quant Q_i   │                      │  Quant Q_N     │
│  MobileNetV2 │                      │  MobileNetV2   │
│  CIFAR-10    │                      │  CIFAR-10      │
└──────────────┘                      └────────────────┘
```

**Framework**: Flower (`flwr`)  
**Model**: MobileNetV2 (PyTorch / torchvision)  
**Dataset**: CIFAR-10  
**RL**: PPO via `stable-baselines3`  
**Gym interface**: `gymnasium`  

---

## 3. Algorithm Definition

### 3.1 State Space `s_t`

At round `t`, the RL agent observes a concatenated feature vector comprising per-client and global features.

**Per-client features** (7 per client, repeated for each client `i`):

| Feature | Description |
|---------|-------------|
| `cpu_cores_i` | Number of vCPUs available to client i |
| `mem_limit_mb_i` | Memory limit in MB for client i (normalized) |
| `mem_used_frac_i` | Current memory utilization fraction [0, 1] |
| `reliability_i` | Historical success rate = rounds_completed / rounds_attempted |
| `last_dropout_i` | 1 if client dropped out last round, else 0 |
| `consecutive_drops_i` | Count of consecutive dropout rounds (normalized) |
| `last_quant_i` | Quantization bits used last round: 0=not selected, 8/16/32 normalized |

**Global features** (3, shared across all clients):

| Feature | Description |
|---------|-------------|
| `global_accuracy_{t-1}` | FL model top-1 accuracy from previous round |
| `ΔAcc_{t-1}` | One-round accuracy delta: `Acc_{t-1} − Acc_{t-2}` (0.0 on round 1) |
| `round_fraction` | `t / T` — normalized round index in [0, 1] |

> **ΔAcc_{t-1} definition**: `ΔAcc_{t-1} = Acc_{t-1} − Acc_{t-2}` (single-step delta). No multi-round trend unless explicitly added in a later stage.

State vector layout: `[f_0₁, …, f_0₇, f_1₁, …, f_1₇, …, f_{N-1,7}, acc, ΔAcc_{t-1}, t/T]`

Dimension = `N_clients × 7 + 3`. This value is computed by `Config.state_dim` and passed to the PPO network at construction time — it is **not hardcoded** anywhere in `src/`. Example: 3 clients → 3×7+3 = 24.

### 3.2 Action Space `a_t`

`a_t = (S_t, Q^t)`

- **Client selection** `S_t`: binary mask of length `N` (which clients participate)
- **Quantization** `Q_i^t`: per-selected-client bit assignment ∈ {32, 16, 8}

**Encoding**: Each client chooses from `{skip=0, fp32=1, fp16=2, int8=3}` (4 options).  
Total discrete actions = `MultiDiscrete([4] * N_clients)`.

For 3 clients: `4^3 = 64` joint actions — feasible with SB3 `MultiDiscrete`.

**Constraint**: At least 1 client must be selected per round (enforced via action masking or post-processing fallback: if all clients mapped to `skip`, force-select the one with highest reliability).

### 3.3 Reward Function `r_t`

```
r_t = α · ΔAcc_t  −  β · Dropout_t  −  γ · Var(Q_t)
```

| Term | Symbol | Default Weight | Description |
|------|--------|---------------|-------------|
| Accuracy gain | α · ΔAcc_t | α = 1.0 | `Acc_t − Acc_{t-1}`: one-round global accuracy improvement |
| Dropout penalty | β · Dropout_t | β = 0.5 | Fraction of **selected** clients that failed this round |
| Quantization variance | γ · Var(Q_t) | γ = 0.1 | Variance of bit-widths over **selected** clients only |

**Explicit definitions**:
- `ΔAcc_t = Acc_t − Acc_{t-1}` (single-step; same type of signal as the state feature (state uses ΔAcc_{t-1}))
- `Dropout_t = |{i ∈ S_t : client i failed}| / |S_t|` (fraction of selected clients that dropped)
- `Var(Q_t) = variance({Q_i^t : i ∈ S_t})` — computed **only over selected clients** `S_t`; skipped clients (action=0) are excluded entirely.

**Rationale for γ term**: Penalizes extreme heterogeneity in quantization that can destabilize FedAvg aggregation. Encourages consistent quality levels unless resource constraints force deviation.

**Accuracy metric**: Top-1 accuracy on a server-held global test set (10% of CIFAR-10 test split).

---

## 4. Quantization Approach

### 4.1 Supported Levels

| Level | Torch Dtype | Method |
|-------|-------------|--------|
| 32-bit | `torch.float32` | No-op (baseline) |
| 16-bit | `torch.float16` | `model.half()` + `model.to(device)` |
| 8-bit | `torch.qint8` | PyTorch Static INT8 (preferred) |

### 4.2 Static INT8 — Implementation Constraint and Fallback Policy

**Paper alignment**: The paper specifies static INT8. PyTorch's `torch.quantization.quantize_static()` requires:
1. A **calibration dataset** representative of the training distribution.
2. **Fuseable layer patterns** (Conv → BN → ReLU) — MobileNetV2 supports `torch.ao.quantization.fuse_modules`.
3. The model must be in **eval mode** during calibration (running statistics collection).

**Known limitation**: Static INT8 calibration introduces **per-round overhead** (calibration requires a forward pass over ~128 samples). FP32/FP16 are fully implemented; static INT8 is implemented in Phase 4 using a 128-sample calibration set.

**Backend auto-selection** (SPEC.md §4.2 / `src/compression/int8.py`):
At the start of each INT8 quantization attempt, `try_static_int8()` logs `torch.backends.quantized.supported_engines` and probes each candidate in order: **fbgemm → x86 → onednn → qnnpack**. The first backend that successfully completes fuse → prepare → calibrate → convert → inference is selected. If all fail, fallback is triggered. An explicit `backend=` parameter overrides this.

**Fallback policy** (NEVER silent):

| Condition | Action | Log Key |
|-----------|--------|--------|
| Auto-selected backend succeeds end-to-end | Use static INT8 | `quant_method: static_int8` |
| All backends fail probe / op unsupported | Log and fall back to **FP16** | `QUANT_UNSUPPORTED`, `quant_method: fp16_fallback` |
| FP16 also fails | Fall back to **FP32** | `QUANT_UNSUPPORTED`, `quant_method: fp32_fallback` |
| INT8 action requested but pipeline failed | **INT8 disabled this client this round** | `int8_disabled: true` |

**This fallback is NEVER silent** — every round JSON logs `quant_method` and `int8_disabled` per client. Dynamic INT8 (`quantize_dynamic`) is NOT used as a fallback; it is excluded to keep the method space clean.

**Communication model** (critical): Quantization governs **server → client model distribution only**:
- **Server → Client**: The server distributes the current global model to each selected client at the quantization precision assigned by the PPO action (`Q_i^t ∈ {FP32, FP16, Static INT8}`).
- **Client → Server**: Clients always return updated parameters to the server **in FP32** (full precision). Quantization is NOT applied to gradient or weight updates going back to the server.
- **Aggregation**: The server performs weighted FedAvg in FP32. No quantization step occurs during aggregation.

---

## 5. Federated Learning Details

| Parameter | Value |
|-----------|-------|
| Framework | Flower (`flwr`) v1.7.0 |
| Strategy | Custom `FedAvgQuant` — per-client quantized model distribution + dropout tracking |
| Rounds | 50 |
| Local epochs per round | 2 (reduce to 1 for early smoke tests on VM) |
| Batch size (default) | 32; **VM-safe default: 8–16** (see VM Defaults below) |
| Input resolution | 224×224 (paper-aligned); **do not resize to 32×32** |
| Optimizer | SGD (lr=0.01, momentum=0.9) |
| Aggregation | Weighted FedAvg, always in **FP32**. Quantization is for model weight *distribution* only — update aggregation is never quantized. |
| Global test set | 10% CIFAR-10 test split, server-side only |
| Min clients per round | 1 |
| Max clients per round | N (all) |

### VM-Safe Training Defaults (3.8 GB RAM)

Keep 224×224 resolution (paper-aligned) but control memory via:

| Parameter | VM-safe Value | Notes |
|-----------|-------------|-------|
| `batch_size` | 8 (weak/extreme) / 16 (medium) / 32 (strong) | Override per profile |
| `num_workers` | 0 | Disable DataLoader workers; no shared memory issues |
| `prefetch_factor` | N/A (disabled when `num_workers=0`) | — |
| `local_epochs` | 1 for smoke tests, 2 for full experiments | Configurable |
| `pin_memory` | `False` | CPU-only, pin_memory wastes RAM |

---

## 6. Dataset & Partitioning

| Mode | Description | Experiments |
|------|-------------|-------------|
| IID homogeneous | Equal splits, uniform label distribution | Exp1 |
| IID heterogeneous (size) | Unequal data sizes (e.g., 60%/30%/10%) | Exp2, Exp4 |
| Non-IID Dirichlet α=0.1 | Strongly skewed label distribution | Exp3 |

Partition metadata (sizes, label histograms) saved to `outputs/metrics/partitions.json` per run.

---

## 7. Resource Profiles (Heterogeneity)

Clients are assigned resource profiles controlling memory and CPU limits:

| Profile | Memory Limit | CPU Cores | Expected Behavior |
|---------|-------------|-----------|-------------------|
| `strong` | 2048 MB | 1.5 | Always completes training |
| `medium` | 1024 MB | 1.0 | Completes FP32/FP16; may OOM on INT8 edge cases |
| `weak` | 512 MB | 0.5 | OOM on FP32 large batches; FP16/INT8 required |
| `extreme_weak` | 128 MB | 0.25 | Almost always fails at model load |

Dropout is simulated via two modes (selected by config):
1. **Docker mode**: Docker `--memory` hard limit enforces real OOM kills; `--cpus` enforces CPU limits. Both result in actual training failures/dropout.
2. **Stochastic mode** (default for smoke tests): Probability-based dropout derived from profile + assigned quant level. No Docker required.

> **CPU enforcement note**: In docker mode, `cpu_cores` is enforced via Docker `--cpus` flag (hard cgroup limit). In stochastic mode, `cpu_cores` is a **descriptor only** — it informs the dropout probability model but does not physically constrain the process.

---

## 8. Metrics to Log

Every round `t`, log to `outputs/metrics/run_<id>/round_<t>.json`:

```json
{
  "round": 1,
  "selected_clients": [0, 1, 2],
  "quant_assignments": {"0": 32, "1": 16, "2": 8},
  "actual_quant_method": {"0": "static_fp32", "1": "static_fp16", "2": "static_int8"},
  "quant_fallback": {"2": false},
  "dropout_clients": [],
  "dropout_fraction": 0.0,
  "global_accuracy": 0.312,
  "accuracy_delta": 0.031,
  "reward": 0.278,
  "ppo_action": [1, 2, 3],
  "client_stats": {
    "0": {"mem_used_mb": 380, "train_time_s": 12.3, "loss": 1.84},
    "1": {"mem_used_mb": 512, "train_time_s": 9.8, "loss": 1.91},
    "2": {"mem_used_mb": 210, "train_time_s": 8.1, "loss": 1.96}
  }
}
```

Aggregate per run: `outputs/metrics/run_<id>/summary.json`

Plots saved to: `outputs/plots/run_<id>/`
- `accuracy_curve.png`
- `reward_curve.png`
- `dropout_rate.png`
- `quant_distribution.png`

---

## 9. Configuration Schema (YAML)

Location: `configs/<exp_name>.yaml`

```yaml
experiment:
  name: "exp1_homogeneous_iid"
  rounds: 50
  seed: 42
  dropout_mode: stochastic    # stochastic | docker

clients:
  count: 3
  profiles:
    - id: 0
      profile: strong         # strong | medium | weak | extreme_weak
      mem_limit_mb: 2048
      cpu_cores: 1.5
      data_fraction: 0.33
    - id: 1
      profile: medium
      mem_limit_mb: 1024
      cpu_cores: 1.0
      data_fraction: 0.33
    - id: 2
      profile: weak
      mem_limit_mb: 512
      cpu_cores: 0.5
      data_fraction: 0.34

data:
  dataset: cifar10
  partition: iid              # iid | dirichlet
  dirichlet_alpha: 0.1        # used only if partition=dirichlet
  reduced_fraction: 1.0       # 0.6 for Exp4

quantization:
  mode: adaptive              # adaptive | fixed_fp32 | fixed_fp16 | fixed_int8 | mixed
  fixed_bits: 32              # used only if mode=fixed_*
  calibration_samples: 128
  # No dynamic INT8 fallback. If static INT8 fails: FP16 → FP32, logged as QUANT_UNSUPPORTED.

fl:
  local_epochs: 2
  batch_size: 32              # VM-safe: use per_profile_batch overrides below
  per_profile_batch:          # per-profile batch size override
    strong: 32
    medium: 16
    weak: 8
    extreme_weak: 8
  num_workers: 0              # 0 = no DataLoader workers (VM-safe, no shared mem)
  pin_memory: false           # false for CPU-only
  optimizer: sgd
  lr: 0.01
  momentum: 0.9
  min_clients_per_round: 1

rl:
  algorithm: ppo
  total_timesteps: 50         # = number of FL rounds
  gamma_discount: 0.99
  reward_alpha: 1.0
  reward_beta: 0.5
  reward_gamma_var: 0.1
  policy: MlpPolicy
  checkpoint_dir: outputs/checkpoints

logging:
  level: info
  metrics_dir: outputs/metrics
  plots_dir: outputs/plots
```

---

## 10. Assumptions & Open Questions

| # | Assumption | Rationale |
|---|-----------|-----------|
| A1 | PPO policy runs centrally on the server | Paper does not specify distributed RL; central is simplest |
| A2 | Reward uses global top-1 accuracy on server-held test set | Standard FL evaluation metric |
| A3 | Dropout penalty = fraction of *selected* clients that dropped | Normalizes by selection size not total clients |
| A4 | INT8 calibration uses 128 CIFAR-10 samples from server's held-out set | Small enough for VM RAM (3.8 GB) |
| A5 | FedAvg always upcasts to FP32 before averaging | Numerical stability |
| A6 | Non-IID uses Dirichlet α=0.1 (strongly heterogeneous) | Paper explicitly specifies α=0.1 |
| A7 | "Survival" = client completes round without dropout | Binary outcome per client per round |
| A8 | Initial experiments use 3 clients; scalable via config | VM RAM constraint (3.8 GB) |
| A9 | Docker resource limits used for Exp5+ real OOM enforcement | Paper describes actual OOM events |
| A10 | Stochastic dropout used for Exp1-4 smoke tests (no Docker needed) | Faster iteration and reproducibility |

---

## 11. Locked Software Versions

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.12.3 | System Python on this VM (`python --version` confirmed) |
| flwr | 1.7.0 | Stable Flower release (`pip show flwr` confirmed) |
| torch | 2.2.2+cpu | CPU-only build; bumped from 2.1.2 by user; INT8 still supported |
| torchvision | 0.17.2+cpu | Paired with torch 2.2.2+cpu |
| stable-baselines3 | 2.2.1 | Compatible with gymnasium 0.29 |
| gymnasium | 0.29.1 | New Gym API (truncated/terminated) |
| numpy | 1.26.4 | Last NumPy 1.x release |
| pandas | 2.2.0 | For metrics aggregation |
| matplotlib | 3.8.2 | Plotting |
| psutil | 5.9.8 | CPU/RAM monitoring |
| prometheus-client | 0.19.0 | Optional metrics export |
| pyyaml | 6.0.1 | Config loading |
| tqdm | 4.66.1 | Progress bars |

> Versions reflect our implementation environment (`pip show` + `verify_env.py` output, 2026-02-28). All smoke tests pass on these exact versions.

**TensorFlow**: NOT included. PyTorch-only stack. Rationale: avoids RAM overhead of dual framework on 3.8 GB VM.

---

*End of SPEC.md*
