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

