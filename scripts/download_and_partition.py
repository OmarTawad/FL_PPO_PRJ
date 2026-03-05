"""
scripts/download_and_partition.py
Stage 12 — One-time setup: download CIFAR-10 and pre-compute client partitions.

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Usage:
    python scripts/download_and_partition.py --n-clients 50 --data-root data/ \
        --partition iid --seed 42 --output data/partitions_50.json

    python scripts/download_and_partition.py --n-clients 50 --data-root data/ \
        --partition dirichlet --seed 42 --output data/partitions_50_noniid.json

Outputs:
    <output>           — train partition: {client_id: [index, ...]}
    <output>.meta.json — metadata: sizes, label histograms, config used

Designed for shared-volume Docker deployments:
    Host pre-runs this script → mounts data/ as read-only volume in all containers.
    Each container loads its own indices from the JSON using CLIENT_ID env var.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in path when run from any CWD
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.cifar import get_cifar10_train, get_cifar10_test
from src.data.partitioner import (
    build_partitions,
    compute_partition_metadata,
    save_partition_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Resource tier assignment (matches RESOURCE_DISTRIBUTION.md) ───────────────

def assign_resource_tiers(n_clients: int) -> Dict[str, str]:
    """
    Assign heterogeneous resource tiers to client IDs.

    Distribution (matches docker-compose.yml and RESOURCE_DISTRIBUTION.md):
        Strong (20%):       IDs 0..9         2.0 CPUs, 2048 MB
        Medium (40%):       IDs 10..29       1.0 CPU,  1024 MB
        Weak (30%):         IDs 30..44       0.5 CPU,   512 MB
        Extreme-weak (10%): IDs 45..49       0.25 CPU,  256 MB

    For n_clients != 50, tiers scale proportionally. Always at least 1 of each tier.
    """
    if n_clients == 50:
        tiers: Dict[str, str] = {}
        for i in range(0, 10):
            tiers[str(i)] = "strong"
        for i in range(10, 30):
            tiers[str(i)] = "medium"
        for i in range(30, 45):
            tiers[str(i)] = "weak"
        for i in range(45, 50):
            tiers[str(i)] = "extreme_weak"
        return tiers

    # Generic proportional assignment
    tiers = {}
    bounds = [
        (0.00, 0.20, "strong"),
        (0.20, 0.60, "medium"),
        (0.60, 0.90, "weak"),
        (0.90, 1.00, "extreme_weak"),
    ]
    for i in range(n_clients):
        frac = i / n_clients
        for lo, hi, name in bounds:
            if lo <= frac < hi or (frac >= hi and name == "extreme_weak"):
                tiers[str(i)] = name
                break
    return tiers


# ── Resource profile specs (mirrors profiles.py) ─────────────────────────────

_PROFILE_SPECS = {
    "strong":       {"cpu_cores": 2.0,  "mem_limit_mb": 2048},
    "medium":       {"cpu_cores": 1.0,  "mem_limit_mb": 1024},
    "weak":         {"cpu_cores": 0.5,  "mem_limit_mb":  512},
    "extreme_weak": {"cpu_cores": 0.25, "mem_limit_mb":  256},
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CIFAR-10 and generate federated client partitions."
    )
    parser.add_argument(
        "--n-clients", type=int, default=50,
        help="Number of FL clients (default: 50)",
    )
    parser.add_argument(
        "--data-root", type=str, default="data/",
        help="Root directory for CIFAR-10 download (default: data/)",
    )
    parser.add_argument(
        "--partition", type=str, default="iid", choices=["iid", "dirichlet"],
        help="Partition mode: iid or dirichlet (default: iid)",
    )
    parser.add_argument(
        "--dirichlet-alpha", type=float, default=0.1,
        help="Dirichlet alpha for non-IID partition (default: 0.1, paper-aligned)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Output JSON path (default: data/partitions_<n>_<mode>.json)"
        ),
    )
    args = parser.parse_args()

    n_clients: int = args.n_clients
    data_root: str = args.data_root
    partition_mode: str = args.partition
    alpha: float = args.dirichlet_alpha
    seed: int = args.seed

    # Determine output path
    if args.output is None:
        suffix = "iid" if partition_mode == "iid" else f"dirichlet{alpha}"
        output_path = Path(data_root) / f"partitions_{n_clients}_{suffix}.json"
    else:
        output_path = Path(args.output)

    meta_path = output_path.with_suffix(".meta.json")

    # ── 1. Download CIFAR-10 ─────────────────────────────────────────────────
    log.info(f"=== Stage 12: CIFAR-10 Download & Partition (n_clients={n_clients}) ===")
    log.info(f"Data root : {data_root}")
    log.info(f"Partition : {partition_mode}" + (
        f" (alpha={alpha})" if partition_mode == "dirichlet" else ""
    ))
    log.info(f"Seed      : {seed}")
    log.info(f"Output    : {output_path}")

    Path(data_root).mkdir(parents=True, exist_ok=True)

    log.info("Downloading CIFAR-10 training set (50,000 samples)...")
    train_ds = get_cifar10_train(root=data_root, download=True, use_32=False)
    log.info(f"  Train set: {len(train_ds)} samples")

    log.info("Downloading CIFAR-10 test set (10,000 samples — for eval partition)...")
    test_ds = get_cifar10_test(root=data_root, download=True, use_32=False)
    log.info(f"  Test set : {len(test_ds)} samples")

    # ── 2. Build train partitions ────────────────────────────────────────────
    log.info(f"Building {partition_mode.upper()} train partitions for {n_clients} clients...")
    train_partitions = build_partitions(
        dataset=train_ds,
        n_clients=n_clients,
        partition_mode=partition_mode,
        data_fractions=None,       # equal fractions for IID
        dirichlet_alpha=alpha,
        reduced_fraction=1.0,      # full data; smoke tests use reduced_fraction in config
        seed=seed,
    )

    # Sanity checks
    total_train = sum(len(p) for p in train_partitions)
    assert total_train == len(train_ds), (
        f"Partition size mismatch: {total_train} != {len(train_ds)}"
    )
    for i, p in enumerate(train_partitions):
        assert len(p) > 0, f"Client {i} has 0 training samples!"

    log.info(f"  Total partitioned: {total_train} samples across {n_clients} clients")
    log.info(f"  Min per client: {min(len(p) for p in train_partitions)}")
    log.info(f"  Max per client: {max(len(p) for p in train_partitions)}")
    log.info(f"  Avg per client: {total_train / n_clients:.1f}")

    # ── 3. Build eval partitions (from test set, equal split) ────────────────
    # Each client gets floor(10000 / n_clients) test samples for local eval.
    # Server reserves its own 10% of test set (1000 samples) separately at runtime.
    n_test = len(test_ds)
    n_eval_per_client = n_test // n_clients
    eval_partitions: List[List[int]] = []
    for i in range(n_clients):
        start = i * n_eval_per_client
        end = start + n_eval_per_client
        eval_partitions.append(list(range(start, min(end, n_test))))

    log.info(f"  Eval per client: {n_eval_per_client} samples")

    # ── 4. Assign resource tiers ─────────────────────────────────────────────
    tiers = assign_resource_tiers(n_clients)

    # ── 5. Serialize partition file ───────────────────────────────────────────
    # Format: {
    #   "config": {...},
    #   "train": {"0": [idx, ...], "1": [idx, ...], ...},
    #   "eval":  {"0": [idx, ...], ...},
    #   "tiers": {"0": "strong", ...}
    # }
    partition_data = {
        "config": {
            "n_clients": n_clients,
            "partition_mode": partition_mode,
            "dirichlet_alpha": alpha if partition_mode == "dirichlet" else None,
            "seed": seed,
            "n_train_total": total_train,
            "n_test_total": n_test,
            "n_eval_per_client": n_eval_per_client,
        },
        "train": {str(i): train_partitions[i] for i in range(n_clients)},
        "eval":  {str(i): eval_partitions[i]  for i in range(n_clients)},
        "tiers": tiers,
        "profile_specs": _PROFILE_SPECS,
    }

    log.info(f"Writing partition file: {output_path} ...")
    with open(output_path, "w") as f:
        json.dump(partition_data, f, separators=(",", ":"))   # compact for large arrays

    size_mb = output_path.stat().st_size / 1024 / 1024
    log.info(f"  Written: {size_mb:.2f} MB")

    # ── 6. Compute and save metadata ─────────────────────────────────────────
    log.info(f"Computing partition metadata (label histograms) → {meta_path} ...")
    meta = compute_partition_metadata(train_ds, train_partitions, n_classes=10)
    meta["config"] = partition_data["config"]
    meta["resource_tiers"] = {
        cid: {"profile": tier, **_PROFILE_SPECS[tier]}
        for cid, tier in tiers.items()
    }

    # Per-tier summary
    tier_counts = {}
    for t in tiers.values():
        tier_counts[t] = tier_counts.get(t, 0) + 1
    meta["tier_summary"] = tier_counts

    save_partition_metadata(meta, str(meta_path))
    log.info(f"  Metadata written: {meta_path}")

    # ── 7. Summary ───────────────────────────────────────────────────────────
    log.info("=== Partition Summary ===")
    for tier, count in sorted(tier_counts.items()):
        spec = _PROFILE_SPECS[tier]
        ids = [cid for cid, t in tiers.items() if t == tier]
        log.info(
            f"  {tier:14s}: {count:2d} clients "
            f"(IDs {ids[0]}–{ids[-1]}, "
            f"{spec['cpu_cores']} CPUs, {spec['mem_limit_mb']} MB RAM)"
        )
    log.info("=== Stage 12 Complete ===")
    log.info(f"  Partition file : {output_path}")
    log.info(f"  Metadata file  : {meta_path}")
    log.info("  CIFAR-10 is ready for shared-volume Docker deployment.")


if __name__ == "__main__":
    main()
