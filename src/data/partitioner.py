"""
src/data/partitioner.py — Dataset partitioning for federated learning

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Implements three partition strategies:
  1. iid_partition        — equal-size, uniform label distribution
  2. dirichlet_partition  — non-IID via Dirichlet(α) label skew (α=0.1 for paper)
  3. reduced_partition    — sub-sample each client's indices (Exp4: 60%)

All functions operate on index arrays (not the dataset directly), making them
compatible with any torch Dataset. Partition metadata (sizes, label histograms)
is returned for logging to outputs/metrics/partitions.json.

Usage:
    from src.data.partitioner import iid_partition, dirichlet_partition, reduced_partition
    from src.data.cifar import get_cifar10_train

    train_ds = get_cifar10_train(root="~/.cifar10", download=True)
    partitions = iid_partition(train_ds, n_clients=3, seed=42)
    # partitions[i] = list of sample indices for client i
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ─── Type alias ───────────────────────────────────────────────────────────────
# Partition: per-client list of dataset indices
Partition = List[List[int]]


# ─── Label extraction helper ─────────────────────────────────────────────────

def _get_targets(dataset: Dataset) -> np.ndarray:
    """
    Extract target labels from a dataset as a numpy array.
    Supports torchvision datasets (`.targets`) and generic datasets.
    """
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.numpy()
        return np.array(targets)
    # Fallback: iterate (slow, use only if necessary)
    return np.array([dataset[i][1] for i in range(len(dataset))])


# ─── Partition metadata ───────────────────────────────────────────────────────

def compute_partition_metadata(
    dataset: Dataset,
    partitions: Partition,
    n_classes: int = 10,
) -> dict:
    """
    Compute and return partition metadata for logging.

    Returns a dict with per-client sizes and label histograms.
    Suitable for JSON serialization to outputs/metrics/partitions.json.
    """
    targets = _get_targets(dataset)
    metadata: dict = {
        "n_clients": len(partitions),
        "total_samples": int(sum(len(p) for p in partitions)),
        "clients": [],
    }
    for i, indices in enumerate(partitions):
        if len(indices) == 0:
            hist = [0] * n_classes
        else:
            client_targets = targets[indices]
            hist = np.bincount(client_targets, minlength=n_classes).tolist()
        metadata["clients"].append({
            "client_id": i,
            "n_samples": len(indices),
            "label_histogram": hist,
            "data_fraction": len(indices) / len(targets) if len(targets) > 0 else 0.0,
        })
    return metadata


def save_partition_metadata(metadata: dict, path: str) -> None:
    """Save partition metadata to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


# ─── 1. IID Partition ─────────────────────────────────────────────────────────

def iid_partition(
    dataset: Dataset,
    n_clients: int,
    seed: int = 42,
    data_fractions: Optional[List[float]] = None,
) -> Partition:
    """
    Partition dataset indices IID (uniform label distribution) across clients.

    Args:
        dataset: Source dataset (e.g., CIFAR-10 training set).
        n_clients: Number of FL clients.
        seed: RNG seed for reproducibility.
        data_fractions: Optional list of per-client data fractions (must sum to 1.0).
            If None, data is split equally.

    Returns:
        List of index lists, one per client.

    Note:
        Indices are shuffled before splitting to ensure uniform label distribution
        within each client's shard.
    """
    n = len(dataset)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n).tolist()

    if data_fractions is None:
        # Equal split
        data_fractions = [1.0 / n_clients] * n_clients

    # Validate fractions
    if len(data_fractions) != n_clients:
        raise ValueError(
            f"data_fractions length ({len(data_fractions)}) must match n_clients ({n_clients})"
        )
    total = sum(data_fractions)
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"data_fractions must sum to 1.0, got {total:.6f}")

    # Compute integer split sizes (last client absorbs rounding remainder)
    sizes = [int(f * n) for f in data_fractions]
    # Correct rounding: assign remainder to last client
    sizes[-1] = n - sum(sizes[:-1])

    partitions: Partition = []
    start = 0
    for size in sizes:
        partitions.append(shuffled[start : start + size])
        start += size

    return partitions


# ─── 2. Dirichlet Non-IID Partition ──────────────────────────────────────────

def dirichlet_partition(
    dataset: Dataset,
    n_clients: int,
    alpha: float = 0.1,
    seed: int = 42,
    min_samples_per_client: int = 10,
) -> Partition:
    """
    Partition dataset with Dirichlet(α) label skew (non-IID).

    Each client receives samples drawn from a class distribution sampled from
    Dirichlet(α). Lower α → more skewed (α=0.1 is strongly non-IID per paper).

    Algorithm:
        For each class c:
            Sample proportions p ~ Dirichlet(α, ..., α) of length n_clients.
            Assign ceil(p_i * |class_c|) samples of class c to client i.

    Args:
        dataset: Source dataset (must have `.targets` attribute).
        n_clients: Number of FL clients.
        alpha: Dirichlet concentration parameter (paper: α=0.1).
        seed: RNG seed.
        min_samples_per_client: Minimum samples per client (safety floor).
            Clients below this threshold absorb extra from largest clients.

    Returns:
        List of index lists, one per client.
    """
    targets = _get_targets(dataset)
    n_classes = int(targets.max()) + 1
    rng = np.random.default_rng(seed)

    # Group indices by class
    class_indices: List[List[int]] = [
        np.where(targets == c)[0].tolist() for c in range(n_classes)
    ]
    # Shuffle within each class for randomness
    for c in range(n_classes):
        rng.shuffle(class_indices[c])

    # Initialize empty partitions
    partitions: Partition = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idxs = class_indices[c]
        n_c = len(idxs)
        if n_c == 0:
            continue

        # Sample Dirichlet proportions
        proportions = rng.dirichlet(alpha=[alpha] * n_clients)

        # Convert to integer counts; last client absorbs remainder
        counts = (proportions * n_c).astype(int)
        counts[-1] = n_c - counts[:-1].sum()
        counts = np.maximum(counts, 0)  # no negative counts

        # Distribute class indices to clients
        start = 0
        for i, count in enumerate(counts):
            partitions[i].extend(idxs[start : start + count])
            start += count

    # Safety check: ensure no client is starved
    n_total = sum(len(p) for p in partitions)
    for i, p in enumerate(partitions):
        if len(p) < min_samples_per_client:
            # Borrow from the largest client
            largest = max(range(n_clients), key=lambda j: len(partitions[j]))
            if len(partitions[largest]) > min_samples_per_client * 2:
                needed = min_samples_per_client - len(p)
                partitions[i].extend(partitions[largest][-needed:])
                partitions[largest] = partitions[largest][:-needed]

    # Shuffle each client's indices (Dirichlet groups by class; shuffle for diversity)
    for i in range(n_clients):
        rng.shuffle(partitions[i])

    return partitions


# ─── 3. Reduced Partition ─────────────────────────────────────────────────────

def reduced_partition(
    indices: List[int],
    fraction: float,
    seed: int = 42,
) -> List[int]:
    """
    Sub-sample a fraction of a client's index list (Exp4: 60% data).

    Args:
        indices: Client's current index list.
        fraction: Fraction to keep (e.g., 0.6 for 60%).
        seed: RNG seed.

    Returns:
        Reduced index list.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        return indices

    n = len(indices)
    n_keep = max(1, int(n * fraction))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=n_keep, replace=False)
    return [indices[i] for i in sorted(chosen)]


def apply_reduced_fraction(
    partitions: Partition,
    fraction: float,
    seed: int = 42,
) -> Partition:
    """
    Apply reduced_partition to all clients in a Partition.

    Args:
        partitions: Full partitions list.
        fraction: Data fraction to keep per client.
        seed: RNG seed (per-client seed derived from this).

    Returns:
        New partition list with reduced indices per client.
    """
    return [
        reduced_partition(p, fraction=fraction, seed=seed + i)
        for i, p in enumerate(partitions)
    ]


# ─── Convenience: build full partition from Config ────────────────────────────

def build_partitions(
    dataset: Dataset,
    n_clients: int,
    partition_mode: str,
    data_fractions: Optional[List[float]] = None,
    dirichlet_alpha: float = 0.1,
    reduced_fraction: float = 1.0,
    seed: int = 42,
) -> Partition:
    """
    Build partitions from config parameters.

    Args:
        dataset: CIFAR-10 training dataset.
        n_clients: Number of clients.
        partition_mode: "iid" or "dirichlet".
        data_fractions: Per-client fractions (IID only). None = equal.
        dirichlet_alpha: α for Dirichlet non-IID.
        reduced_fraction: Sub-sampling fraction (Exp4).
        seed: RNG seed.

    Returns:
        Partition (list of index lists per client).
    """
    if partition_mode == "iid":
        partitions = iid_partition(
            dataset, n_clients=n_clients, seed=seed,
            data_fractions=data_fractions,
        )
    elif partition_mode == "dirichlet":
        partitions = dirichlet_partition(
            dataset, n_clients=n_clients, alpha=dirichlet_alpha, seed=seed,
        )
    else:
        raise ValueError(f"partition_mode must be 'iid' or 'dirichlet', got: {partition_mode}")

    if reduced_fraction < 1.0:
        partitions = apply_reduced_fraction(partitions, fraction=reduced_fraction, seed=seed)

    return partitions
