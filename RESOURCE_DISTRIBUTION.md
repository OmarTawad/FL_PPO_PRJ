# RESOURCE_DISTRIBUTION.md
# FL PPO Paper — Heterogeneous Client Resource Distribution

## Overview

This document describes the heterogeneous resource configuration used for the
50-client Docker-based federated learning deployment matching the paper:

> "A Reinforcement Learning Framework for Joint Client Selection and Model
> Compression in Heterogeneous Federated Learning"

---

## Client Tier Distribution

| Tier | # Clients | Client IDs | CPUs | RAM | Profile Name |
|------|-----------|------------|------|-----|--------------|
| **Strong** | 10 (20%) | 0 – 9 | 2.0 | 2048 MB | `strong` |
| **Medium** | 20 (40%) | 10 – 29 | 1.0 | 1024 MB | `medium` |
| **Weak** | 15 (30%) | 30 – 44 | 0.5 | 512 MB | `weak` |
| **Extreme-weak** | 5 (10%) | 45 – 49 | 0.25 | 256 MB | `extreme_weak` |
| **Total** | **50** | 0 – 49 | — | — | — |
| **Server** | 1 | — | 8.0 | 12288 MB | — |

---

## Rationale

### Distribution shape (20% / 40% / 30% / 10%)

This pyramid mirrors real-world federated learning deployments:

- **20% strong**: Represents high-end smartphones or moderate edge servers that
  reliably complete FP32 training. These are the backbone of reliable updates.
- **40% medium**: Represents typical IoT devices and mid-range smartphones.
  Complete FP32 in most rounds; may benefit from FP16 assignment to avoid edge OOM.
- **30% weak**: Represents constrained IoT sensors and low-end embedded boards.
  MobileNetV2 FP32 + large batch causes OOM; assigned FP16/INT8 by PPO.
- **10% extreme-weak**: Represents minimal resource nodes (256 MB RAM).
  MobileNetV2 FP32 will always OOM; INT8 is the only viable option.

This skew toward weaker clients is intentional — it stresses the PPO adaptive
selection mechanism and demonstrates the paper's core contribution.

---

## Profile Behavioral Properties (from `src/heterogeneity/profiles.py`)

| Profile | Reliability | Dropout @ FP32 | Dropout @ FP16 | Dropout @ INT8 |
|---------|-------------|----------------|----------------|----------------|
| `strong` | 0.99 | 1% | 2% | 3% |
| `medium` | 0.95 | 5% | 8% | 15% |
| `weak` | 0.70 | 10% | 20% | 35% |
| `extreme_weak` | 0.50 | 30% | 45% | 60% |

Dropout probabilities increase with quantization memory overhead:
- FP32: baseline (model weights at full precision)
- FP16: ~1.5× dropout multiplier (intermediate tensors still at FP16)
- INT8: ~3× dropout multiplier (calibration overhead adds peak memory)

---

## Memory Safety on 22 GB KVM Host

### Theoretical peak
- Server: 12 GB
- 50 clients: 10×2 + 20×1 + 15×0.5 + 5×0.25 = **~49 GB** (impossible on 22 GB)

### Why it works in practice
Memory limits in Docker are **cgroup enforcement limits**, not reservations.
In stochastic dropout mode:
- Weak/extreme-weak clients frequently dropout, releasing their memory
- Training is sequential per client in each round (Flower gRPC)
- Peak actual memory ≪ sum of declared limits

### Safe operating guidelines

| Scenario | Clients | Recommended RAM on host |
|----------|---------|------------------------|
| Smoke test | 3–5 | ≥ 4 GB free |
| Partial run | 10–15 | ≥ 8 GB free |
| Full 50-client (stochastic) | 50 | ≥ 16 GB free (our VM: 22 GB ✓) |

> ⚠️ **NEVER run 50 clients in docker mode** (real OOM) on a 22 GB host.
> Use `dropout_mode: stochastic` for the full 50-client experiment.

---

## Data Distribution

Each of the 50 clients receives:
- **IID mode**: exactly 1,000 training samples (50,000 / 50)
- **Non-IID mode**: samples following Dirichlet(α=0.1) distribution
- **Eval**: 200 test set samples for local evaluation

Partition files:
- `data/partitions_50_iid.json` — IID partition
- `data/partitions_50_dirichlet0.1.json` — Dirichlet non-IID partition

---

## Docker Resource Enforcement

Resources are enforced via Docker cgroup limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: "0.5"      # maps to --cpus=0.5 → 50% of 1 physical core
      memory: "512m"   # maps to --memory=512m → hard OOM kill at 512 MB
```

The `--cpus` limit uses Linux CFS bandwidth control (cpu.cfs_quota_us).
The `--memory` limit triggers OOM killer when exceeded (exit code 137).

---

## Smoke Test Resource Footprint

The smoke test (`docker-compose.smoke.yml`) uses only 3 clients:

| Container | CPUs | RAM |
|-----------|------|-----|
| Server | 8.0 | 12288 MB |
| client_00 (strong) | 2.0 | 2048 MB |
| client_01 (medium) | 1.0 | 1024 MB |
| client_02 (weak) | 0.5 | 512 MB |
| **Total** | **11.5** | **~16 GB** |

This is safe for the 22 GB KVM host (≥ 6 GB OS headroom).
