# RESOURCE_DISTRIBUTION.md
# FL PPO Paper — Heterogeneous Client Resource Distribution

## Overview

This document describes the heterogeneous resource configuration used for the
10-client Docker-based federated learning deployment matching the paper:

> "A Reinforcement Learning Framework for Joint Client Selection and Model
> Compression in Heterogeneous Federated Learning"

---

## Experiment 1

Experiment 1 (`exp01_baseline`) configuration summary:

| Item | Value |
|------|-------|
| Number of clients | 10 |
| Number of rounds | 50 |
| Time per round | 900s |
| Resource per client | 1 CPU, 1536 MB RAM |

Client-level resources for Experiment 1:

| Client IDs | Profile | CPU limit | Memory limit |
|------------|---------|-----------|--------------|
| 0-9 | `strong` | 1.0 | 1536 MB |

---

## Experiment 2

Experiment 2 (`exp02_baseline_b`) configuration summary:

| Item | Value |
|------|-------|
| Number of clients | 10 |
| Number of rounds | 50 |
| Time per round | 1000s |
| Data partitioning | IID (heterogeneous clients) |

Client-level resources for Experiment 2:

| Client IDs | Profile | CPU limit | Memory limit |
|------------|---------|-----------|--------------|
| 0-1 | `strong` | 1.0 | 2560 MB |
| 2-5 | `medium` | 0.75 | 1536 MB |
| 6-8 | `weak` | 0.5 | 1024 MB |
| 9 | `extreme_weak` | 0.25 | 768 MB |

---

## Experiment 3

Experiment 3 (`exp03_heterogeneous_noniid`) configuration summary:

| Item | Value |
|------|-------|
| Number of clients | 10 |
| Number of rounds | 50 |
| Time per client (per round, approx.) | 1500s |
| Data partitioning | Non-IID (Dirichlet alpha=0.1, heterogeneous clients) |

Client-level resources for Experiment 3:

| Client IDs | Profile | CPU limit | Memory limit |
|------------|---------|-----------|--------------|
| 0-1 | `strong` | 1.0 | 2560 MB |
| 2-5 | `medium` | 0.75 | 1536 MB |
| 6-8 | `weak` | 0.5 | 1024 MB |
| 9 | `extreme_weak` | 0.25 | 768 MB |

---

## Experiment 4

Experiment 4 (`exp04_reduced_sampling_iid`) configuration summary:

| Item | Value |
|------|-------|
| Number of clients | 10 |
| Number of rounds | 50 |
| Avg time per client (per round, approx.) | 700s |
| Data partitioning | IID (reduced sampling 60%, heterogeneous clients) |

Client-level resources for Experiment 4:

| Client IDs | Profile | CPU limit | Memory limit |
|------------|---------|-----------|--------------|
| 0-1 | `strong` | 1.0 | 2560 MB |
| 2-5 | `medium` | 0.75 | 1536 MB |
| 6-8 | `weak` | 0.5 | 1024 MB |
| 9 | `extreme_weak` | 0.25 | 768 MB |

---
