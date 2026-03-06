"""
src/data/cifar.py — CIFAR-10 dataset loader

Paper: Adaptive FL with PPO-based Client and Quantization Selection
Input resolution: 224×224 (paper-aligned, MobileNetV2 default).
    A `use_32` flag is provided for lightweight smoke tests only.
    All real experiments must use 224×224.

Usage:
    from src.data.cifar import get_cifar10_train, get_cifar10_test, get_server_test_loader
    train_ds = get_cifar10_train(root="~/.cifar10")
    test_ds  = get_cifar10_test(root="~/.cifar10")
    server_loader = get_server_test_loader(root="~/.cifar10", batch_size=32)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# CIFAR-10 per-channel mean and std (paper/common FL baseline values)
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

# ImageNet stats used when MobileNetV2 pretrained weights are loaded
# (we use randomly-initialized MobileNetV2, so CIFAR stats are preferred)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

NUM_CLASSES = 10
DEFAULT_ROOT = os.path.expanduser("~/.cifar10")


def get_train_transform(use_32: bool = False, augment: bool = True) -> transforms.Compose:
    """
    Return training transforms.

    Args:
        use_32: If True, keep native 32×32 (smoke tests only).
                If False (default), resize to 224×224 (paper-aligned).
    """
    if use_32:
        if augment:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ])
    # Paper-aligned: MobileNetV2 expects 224×224
    if augment:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])


def get_eval_transform(use_32: bool = False) -> transforms.Compose:
    """Return eval/test transforms (no augmentation)."""
    if use_32:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])


def get_cifar10_train(
    root: str = DEFAULT_ROOT,
    download: bool = False,
    use_32: bool = False,
    augment: bool = True,
) -> datasets.CIFAR10:
    """
    Return raw CIFAR-10 training dataset (50,000 samples).

    Args:
        root: Directory to store/read CIFAR-10 data.
        download: If True, download if not cached. Default False.
        use_32: If True, use 32×32 transforms (smoke tests only).
    """
    return datasets.CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=get_train_transform(use_32=use_32, augment=augment),
    )


def get_cifar10_test(
    root: str = DEFAULT_ROOT,
    download: bool = False,
    use_32: bool = False,
) -> datasets.CIFAR10:
    """
    Return raw CIFAR-10 test dataset (10,000 samples).

    Args:
        root: Directory to store/read CIFAR-10 data.
        download: If True, download if not cached. Default False.
        use_32: If True, use 32×32 transforms (smoke tests only).
    """
    return datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=get_eval_transform(use_32=use_32),
    )


def split_server_test(
    test_dataset: datasets.CIFAR10,
    server_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """
    Split test set into:
    - server_set: held-out evaluation set (default 10%). Server-side only.
    - remainder:  remaining test samples (available for other use if needed).

    The split is seeded for reproducibility.

    Returns:
        (server_subset, remainder_subset)
    """
    n = len(test_dataset)
    n_server = max(1, int(n * server_fraction))

    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(n, generator=rng).tolist()

    server_indices    = indices[:n_server]
    remainder_indices = indices[n_server:]

    return Subset(test_dataset, server_indices), Subset(test_dataset, remainder_indices)


def get_server_test_loader(
    root: str = DEFAULT_ROOT,
    batch_size: int = 32,
    server_fraction: float = 0.1,
    seed: int = 42,
    download: bool = False,
    use_32: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Build and return the server-side evaluation DataLoader.

    Uses 10% of CIFAR-10 test set (paper-aligned). The server holds this
    fixed subset across all rounds to measure global model accuracy.

    Args:
        root: CIFAR-10 data directory.
        batch_size: Evaluation batch size.
        server_fraction: Fraction of test set reserved for server (default 0.1).
        seed: RNG seed for split reproducibility.
        download: Download CIFAR-10 if not present.
        use_32: Use 32×32 inputs (smoke tests only).
        num_workers: DataLoader workers (0 = VM-safe default).
        pin_memory: False for CPU-only VMs.
    """
    test_ds = get_cifar10_test(root=root, download=download, use_32=use_32)
    server_subset, _ = split_server_test(test_ds, server_fraction=server_fraction, seed=seed)
    return DataLoader(
        server_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def make_client_loader(
    dataset: Dataset,
    indices: list,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for a single client's local dataset.

    Args:
        dataset: The full CIFAR-10 training dataset.
        indices: Client's assigned sample indices.
        batch_size: Training batch size (use profile-appropriate value).
        shuffle: Whether to shuffle each epoch.
        seed: RNG seed for reproducibility.
        num_workers: 0 = VM-safe.
        pin_memory: False for CPU-only.
    """
    subset = Subset(dataset, indices)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def get_dataset_info(dataset: datasets.CIFAR10) -> dict:
    """Return basic info dict about a CIFAR-10 dataset (for logging)."""
    return {
        "n_samples": len(dataset),
        "n_classes": NUM_CLASSES,
        "classes": dataset.classes,
        "input_resolution": "224x224 (paper-aligned) or 32x32 (smoke test)",
    }
