# RUNSHEET.md — Experiment Run Configurations

**Version:** 0.1.0  
**Date:** 2026-02-28  
**Paper:** Adaptive FL with PPO-based Client and Quantization Selection

> **Stage**: This RUNSHEET is defined at Stage 0. Command templates use placeholders `<...>` that are filled in by Stages 2–9. Configs under `configs/` are created in Stage 5+.

---

## Global Defaults (all experiments)

```
Rounds: 50
Model: MobileNetV2 (CIFAR-10, 10 classes)
Local epochs: 2
Batch size: 32
Optimizer: SGD lr=0.01 mom=0.9
Seed: 42
Dropout mode: stochastic (Exp1-4), docker (Exp5+)
```

---

## Experiment 1 — Homogeneous IID Baseline

**Purpose**: Establish a clean upper-bound baseline. All 3 clients are identical (strong profile), IID data, FP32. No PPO adaptation yet. PPO must learn to always select all 3 clients with FP32.

**Config**: `configs/exp1_homogeneous_iid.yaml`

```yaml
experiment:
  name: exp1_homogeneous_iid
  rounds: 50
  seed: 42
  dropout_mode: stochastic

clients:
  count: 3
  profiles:
    - {id: 0, profile: strong, mem_limit_mb: 2048, cpu_cores: 1.5, data_fraction: 0.333}
    - {id: 1, profile: strong, mem_limit_mb: 2048, cpu_cores: 1.5, data_fraction: 0.333}
    - {id: 2, profile: strong, mem_limit_mb: 2048, cpu_cores: 1.5, data_fraction: 0.334}

data:
  partition: iid
  reduced_fraction: 1.0

quantization:
  mode: fixed_fp32
```

**Command template** (Stage 9+):
```bash
python src/experiments/exp1.py --config configs/exp1_homogeneous_iid.yaml \
  --output-dir outputs/metrics/exp1 --seed 42
```

**Expected qualitative outcome**:
- Steady accuracy improvement over 50 rounds; reaches ~70–75% by round 50 (CIFAR-10 with MobileNetV2, 2 local epochs).
- Zero dropouts.
- Reward increases monotonically.

**Success criteria**:
- Final accuracy ≥ 60% after 50 rounds.
- Zero dropouts across all rounds.
- `summary.json` written with `status: success`.

---

## Experiment 2 — Heterogeneous IID (Weak Client OOM Early)

**Purpose**: Introduce one weak client (512 MB). With FP32, the weak client OOMs and drops frequently. Shows that resource heterogeneity degrades FL without adaptation.

**Config**: `configs/exp2_heterogeneous_iid.yaml`

```yaml
experiment:
  name: exp2_heterogeneous_iid
  rounds: 50
  dropout_mode: stochastic

clients:
  count: 3
  profiles:
    - {id: 0, profile: strong,  mem_limit_mb: 2048, cpu_cores: 1.5, data_fraction: 0.33}
    - {id: 1, profile: medium,  mem_limit_mb: 1024, cpu_cores: 1.0, data_fraction: 0.33}
    - {id: 2, profile: weak,    mem_limit_mb:  512, cpu_cores: 0.5, data_fraction: 0.34}

data:
  partition: iid

quantization:
  mode: fixed_fp32
```

**Command template**:
```bash
python src/experiments/exp2.py --config configs/exp2_heterogeneous_iid.yaml \
  --output-dir outputs/metrics/exp2 --seed 42
```

**Expected qualitative outcome**:
- Client 2 (weak, 512 MB) drops frequently when using FP32 batch_size=32.
- Overall accuracy lower than Exp1 due to lost updates.
- Dropout fraction ~0.3–0.7 for client 2.

**Success criteria**:
- Client 2 dropout rate > 20% (demonstrates the problem).
- Final accuracy noticeably lower than Exp1 (≥5% gap expected).
- `dropout_clients` field populated in round JSONs.

---

## Experiment 3 — Heterogeneous + Non-IID Dirichlet (α=0.1)

**Purpose**: Compound challenge — resource heterogeneity AND data heterogeneity. Weak client fails AND non-IID data causes accuracy instability (higher variance, possible accuracy drops).

**Config**: `configs/exp3_heterogeneous_noniid.yaml`

```yaml
experiment:
  name: exp3_heterogeneous_noniid
  rounds: 50
  dropout_mode: stochastic

clients:
  count: 3
  profiles:
    - {id: 0, profile: strong,  mem_limit_mb: 2048, data_fraction: 0.33}
    - {id: 1, profile: medium,  mem_limit_mb: 1024, data_fraction: 0.33}
    - {id: 2, profile: weak,    mem_limit_mb:  512, data_fraction: 0.34}

data:
  partition: dirichlet
  dirichlet_alpha: 0.1

quantization:
  mode: fixed_fp32
```

**Command template**:
```bash
python src/experiments/exp3.py --config configs/exp3_heterogeneous_noniid.yaml \
  --output-dir outputs/metrics/exp3 --seed 42
```

**Expected qualitative outcome**:
- Higher accuracy variance compared to Exp2 (non-IID causes client drift).
- Client 2 drops frequently; when it does, model misses some label partitions.
- Accuracy curve shows instability / plateaus or drops mid-training.

**Success criteria**:
- Accuracy variance (std over last 10 rounds) > Exp2 variance.
- Dropout rate for client 2 > 20%.
- `label_distribution.json` shows Dirichlet skew (no client has uniform distribution).

---

## Experiment 4 — Reduced Sampling (60% Data)

**Purpose**: Test FL robustness when clients only use 60% of their local data per round. Models the case of bandwidth/storage-limited clients selecting subsets.

**Config**: `configs/exp4_reduced_sampling.yaml`

```yaml
experiment:
  name: exp4_reduced_sampling
  rounds: 50
  dropout_mode: stochastic

clients:
  count: 3
  profiles:
    - {id: 0, profile: strong,  mem_limit_mb: 2048, data_fraction: 0.33}
    - {id: 1, profile: medium,  mem_limit_mb: 1024, data_fraction: 0.33}
    - {id: 2, profile: weak,    mem_limit_mb:  512, data_fraction: 0.34}

data:
  partition: iid
  reduced_fraction: 0.6

quantization:
  mode: fixed_fp32
```

**Command template**:
```bash
python src/experiments/exp4.py --config configs/exp4_reduced_sampling.yaml \
  --output-dir outputs/metrics/exp4 --seed 42
```

**Expected qualitative outcome**:
- Slower convergence than Exp2 (less data per round).
- Comparable dropout behavior to Exp2.
- Final accuracy lower than Exp1 and Exp2.

**Success criteria**:
- Convergence speed (rounds to 50% accuracy) > Exp1.
- Effective samples per round ≈ 60% of Exp2 (verifiable from logs).

---

## Experiment 5 — Extreme Memory Limits (128–512 MB)

**Purpose**: Test the failure regime. All clients use extreme_weak or weak profiles. Most fail at model load time. Demonstrates the need for quantization + adaptive selection.

**Config**: `configs/exp5_extreme_memory.yaml`

```yaml
experiment:
  name: exp5_extreme_memory
  rounds: 20            # fewer rounds, most will fail quickly
  dropout_mode: docker  # real OOM enforcement

clients:
  count: 3
  profiles:
    - {id: 0, profile: extreme_weak, mem_limit_mb: 512, cpu_cores: 0.5}
    - {id: 1, profile: extreme_weak, mem_limit_mb: 256, cpu_cores: 0.25}
    - {id: 2, profile: extreme_weak, mem_limit_mb: 128, cpu_cores: 0.25}

data:
  partition: iid

quantization:
  mode: fixed_fp32
```

**Command template**:
```bash
python src/experiments/exp5.py --config configs/exp5_extreme_memory.yaml \
  --output-dir outputs/metrics/exp5 --seed 42
# NOTE: Requires Docker. Run: docker build -f docker/Dockerfile.client -t fl_client .
```

**Expected qualitative outcome**:
- Client 2 (128 MB) almost always fails at MobileNetV2 load.
- Client 1 (256 MB) fails frequently (FP32 MobileNetV2 ~14 MB weights + activations exceed limit).
- Client 0 (512 MB) marginal; may survive some rounds.
- Very poor or no convergence.

**Success criteria** (this is a "failure demonstration"):
- Total dropout fraction > 60% across all rounds.
- Global accuracy stays low (< 25%) or NaN.
- All dropout reasons logged as `OOM` or `load_failure`.

---

## Experiment 6 — FP16 on Weakest Client (Rescued Survival)

**Purpose**: Show that assigning FP16 to the weakest client rescues its survival (memory halved). Compare survival rate vs. Exp5.

**Config**: `configs/exp6_fp16_weakest.yaml`

```yaml
experiment:
  name: exp6_fp16_weakest
  rounds: 50
  dropout_mode: stochastic

clients:
  count: 3
  profiles:
    - {id: 0, profile: strong,  mem_limit_mb: 2048}
    - {id: 1, profile: medium,  mem_limit_mb: 1024}
    - {id: 2, profile: weak,    mem_limit_mb:  512}

data:
  partition: iid

quantization:
  mode: mixed
  per_client:
    0: 32
    1: 32
    2: 16
```

**Command template**:
```bash
python src/experiments/exp6.py --config configs/exp6_fp16_weakest.yaml \
  --output-dir outputs/metrics/exp6 --seed 42
```

**Expected qualitative outcome**:
- Client 2 survival rate improves significantly vs. Exp2 (FP16 cuts activation memory).
- Slight accuracy penalty vs. Exp1 (FP16 minor precision loss).
- Overall accuracy higher than Exp2 (fewer dropouts = better aggregation).

**Success criteria**:
- Client 2 dropout rate < 15% (vs. >20% in Exp2).
- Final accuracy within 3% of Exp1.

---

## Experiment 7 — INT8 on Weakest Client (Survival + Instability)

**Purpose**: Show that INT8 rescues the weakest client but introduces more gradient noise/instability compared to FP16. Trade-off between survival and accuracy quality.

**Config**: `configs/exp7_int8_weakest.yaml`

```yaml
experiment:
  name: exp7_int8_weakest
  rounds: 50
  dropout_mode: stochastic

clients:
  count: 3
  profiles:
    - {id: 0, profile: strong, mem_limit_mb: 2048}
    - {id: 1, profile: medium, mem_limit_mb: 1024}
    - {id: 2, profile: weak,   mem_limit_mb: 512}

data:
  partition: iid

quantization:
  mode: mixed
  per_client:
    0: 32
    1: 32
    2: 8
  calibration_samples: 128
```

**Command template**:
```bash
python src/experiments/exp7.py --config configs/exp7_int8_weakest.yaml \
  --output-dir outputs/metrics/exp7 --seed 42
```

**Expected qualitative outcome**:
- Client 2 survival rate similar to Exp6 (INT8 reduces memory more than FP16).
- Higher accuracy variance than Exp6 (quantization noise from INT8 is larger).
- Possible accuracy oscillations in later rounds.

**Success criteria**:
- Client 2 dropout rate < 15%.
- Accuracy variance (std over last 10 rounds) > Exp6 variance.
- `quant_fallback` field checked: log whether static INT8 was achieved or fell back.

---

## Experiment 9 — Uniform 8-bit Across All Clients

**Purpose**: Baseline for uniform quantization. All clients use INT8. Shows accuracy degradation vs. FP32 baseline but improved throughput/memory.

**Config**: `configs/exp9_uniform_int8.yaml`

```yaml
experiment:
  name: exp9_uniform_int8
  rounds: 50
  dropout_mode: stochastic

clients:
  count: 3
  profiles:
    - {id: 0, profile: strong,  mem_limit_mb: 2048}
    - {id: 1, profile: medium,  mem_limit_mb: 1024}
    - {id: 2, profile: weak,    mem_limit_mb:  512}

data:
  partition: iid

quantization:
  mode: fixed_int8
  calibration_samples: 128
```

**Command template**:
```bash
python src/experiments/exp9.py --config configs/exp9_uniform_int8.yaml \
  --output-dir outputs/metrics/exp9 --seed 42
```

**Expected qualitative outcome**:
- Zero or very few dropouts (INT8 fits in all profiles).
- Lower final accuracy than Exp1 (FP32) due to quantization noise across all clients.
- Training faster per round (INT8 arithmetic).

**Success criteria**:
- Zero dropouts.
- Final accuracy < Exp1 by at least 2%.
- All rounds log `actual_quant_method: static_int8` (or explicit fallback flag).

---

## Experiment 10 — Mixed 8/16/32 Per-Client (Best Trade-off)

**Purpose**: Manually-tuned mixed quantization. Strong clients get FP32, medium gets FP16, weak gets INT8. This is the handcrafted "oracle" that the PPO agent should learn to replicate automatically.

**Config**: `configs/exp10_mixed_quant.yaml`

```yaml
experiment:
  name: exp10_mixed_quant
  rounds: 50
  dropout_mode: stochastic

clients:
  count: 3
  profiles:
    - {id: 0, profile: strong,  mem_limit_mb: 2048}
    - {id: 1, profile: medium,  mem_limit_mb: 1024}
    - {id: 2, profile: weak,    mem_limit_mb:  512}

data:
  partition: iid

quantization:
  mode: mixed
  per_client:
    0: 32
    1: 16
    2: 8
```

**Command template**:
```bash
python src/experiments/exp10.py --config configs/exp10_mixed_quant.yaml \
  --output-dir outputs/metrics/exp10 --seed 42
```

**Expected qualitative outcome**:
- Best balance: strong client guides accuracy, weak client survives with INT8.
- Accuracy close to Exp1 (FP32 baseline) with near-zero dropouts.
- This result is the "oracle" benchmark for the PPO experiment.

**Success criteria**:
- Client 2 dropout rate < 10%.
- Final accuracy within 5% of Exp1.
- Quantization variance `Var(Q_t)` is non-zero but stable (logs a small γ penalty).

---

## Experiment PPO — Adaptive Policy Learns Mixed Quant

**Purpose**: The PPO agent, trained over 50 FL rounds, learns to assign quantization levels and select clients adaptively. Should converge to a policy similar to Exp10 oracle.

**Config**: `configs/exp_ppo_adaptive.yaml`

```yaml
experiment:
  name: exp_ppo_adaptive
  rounds: 50
  dropout_mode: stochastic

clients:
  count: 3
  profiles:
    - {id: 0, profile: strong,  mem_limit_mb: 2048}
    - {id: 1, profile: medium,  mem_limit_mb: 1024}
    - {id: 2, profile: weak,    mem_limit_mb:  512}

data:
  partition: iid

quantization:
  mode: adaptive

rl:
  algorithm: ppo
  total_timesteps: 50
  reward_alpha: 1.0
  reward_beta: 0.5
  reward_gamma_var: 0.1
```

**Command template**:
```bash
python src/experiments/exp_ppo.py --config configs/exp_ppo_adaptive.yaml \
  --output-dir outputs/metrics/exp_ppo --seed 42
```

**Expected qualitative outcome**:
- Early rounds: PPO explores randomly (high variance actions).
- Mid rounds: PPO discovers that weak client needs lower bits to survive.
- Late rounds: PPO converges to approximately {FP32, FP16, INT8} assignment (Exp10 pattern).
- Reward curve increases over training.

**Success criteria**:
- Reward (r_t) shows positive trend over 50 rounds (linear regression slope > 0).
- Final accuracy comparable to Exp10 (within 5%).
- PPO checkpoint saved to `outputs/checkpoints/exp_ppo/`.
- `ppo_action` evolves visibly in round JSONs (early randomness → later consistency).

---

## Summary Comparison Table

| Experiment | Data | Quant | Profiles | Expected Accuracy | Expected Dropout |
|-----------|------|-------|---------|-------------------|-----------------|
| Exp1 | IID | FP32 all | All strong | ~70-75% | ~0% |
| Exp2 | IID | FP32 all | Strong/Med/Weak | Lower than Exp1 | >20% (client 2) |
| Exp3 | Non-IID α=0.1 | FP32 all | Strong/Med/Weak | Unstable | >20% (client 2) |
| Exp4 | IID 60% | FP32 all | Strong/Med/Weak | Slower convergence | >20% (client 2) |
| Exp5 | IID | FP32 all | All extreme_weak | Near 0% | >60% |
| Exp6 | IID | FP32/FP32/FP16 | Strong/Med/Weak | Close to Exp1 | <15% (client 2) |
| Exp7 | IID | FP32/FP32/INT8 | Strong/Med/Weak | Close to Exp6 ± | <15% (client 2) |
| Exp9 | IID | INT8 all | Strong/Med/Weak | Lower than Exp1 | ~0% |
| Exp10 | IID | FP32/FP16/INT8 | Strong/Med/Weak | Close to Exp1 | <10% |
| PPO | IID | Adaptive | Strong/Med/Weak | ≈ Exp10 (learned) | <10% |

---

## Smoke Test Protocol (Pre-Experiment)

Before running full 50-round experiments, validate environment with a 3-round smoke test:

```bash
# Smoke test: 3 rounds, IID, FP32, all strong clients
python src/experiments/exp1.py --config configs/exp1_homogeneous_iid.yaml \
  --rounds 3 --output-dir outputs/metrics/smoke_test --seed 42
```

**Smoke test success criteria**:
- Completes 3 rounds without crash.
- `outputs/metrics/smoke_test/round_1.json` through `round_3.json` exist.
- Global accuracy present and numeric (not NaN).
- Peak RAM usage < 2.5 GB (leave buffer for OS).

---

*End of RUNSHEET.md*
