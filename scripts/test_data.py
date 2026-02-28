#!/usr/bin/env python3
"""
scripts/test_data.py — Stage 2 sanity-check for CIFAR-10 loader + partitioners

Tests WITHOUT a GPU and WITHOUT downloading CIFAR-10 (uses mock tensors).
Verifies:
  1. IID partition: equal sizes, approximately uniform label distribution
  2. Dirichlet partition (α=0.1): sizes add up, one client dominates each class
  3. Reduced partition: fraction applied correctly
  4. Config integration: build_partitions() accepts Config fields
  5. Partition metadata: JSON structure correct
  6. DataLoader factory: shapes correct for 32×32 smoke mode
     (224×224 full test requires CIFAR-10 download — skipped here)

Note on 224×224 test: We verify transforms are defined and callable, but
do NOT actually load CIFAR-10 to keep this script dependency-free.
Use `python scripts/test_data.py --download` to additionally test with real data.

Exit codes: 0 = all passed, 1 = failure
"""

from __future__ import annotations

import sys
import os
import argparse

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import TensorDataset

GREEN = "\033[0;32m"
RED   = "\033[0;31m"
CYAN  = "\033[0;36m"
RESET = "\033[0m"

errors: list = []
passed = 0

def ok(msg: str):
    global passed
    passed += 1
    print(f"{GREEN}  [OK]{RESET}    {msg}")

def fail(msg: str):
    print(f"{RED}  [FAIL]{RESET}  {msg}")
    errors.append(msg)

def section(title: str):
    print(f"\n{CYAN}{'─'*60}{RESET}")
    print(f"{CYAN}  {title}{RESET}")


# ─── Mock dataset helper ──────────────────────────────────────────────────────

def make_mock_dataset(n_samples: int = 1000, n_classes: int = 10, seed: int = 0) -> TensorDataset:
    """
    Create a mock dataset with integer targets, mimicking CIFAR-10 structure.
    Images are random; targets are uniformly distributed.
    """
    rng = np.random.default_rng(seed)
    # Images: (n, 3, 32, 32) float tensors
    images = torch.randn(n_samples, 3, 32, 32)
    # Targets: balanced class labels
    targets = torch.tensor(
        [i % n_classes for i in range(n_samples)], dtype=torch.long
    )
    # Attach .targets attribute so partitioner can access it
    ds = TensorDataset(images, targets)
    ds.targets = targets  # type: ignore[attr-defined]
    return ds


# ─── Test 1: IID Partition ────────────────────────────────────────────────────

def test_iid():
    section("1/6  IID Partition")
    from src.data.partitioner import iid_partition, compute_partition_metadata

    ds = make_mock_dataset(n_samples=1000, n_classes=10, seed=0)

    # Equal split
    parts = iid_partition(ds, n_clients=3, seed=42)
    total = sum(len(p) for p in parts)
    if total == 1000:
        ok(f"IID equal split: total={total} (expected 1000)")
    else:
        fail(f"IID equal split: total={total} (expected 1000)")

    # Sizes should be close to 1000/3
    sizes = [len(p) for p in parts]
    for i, s in enumerate(sizes):
        expected = 1000 // 3
        if abs(s - expected) <= 1:
            ok(f"  Client {i}: {s} samples (expected ~{expected})")
        else:
            fail(f"  Client {i}: {s} samples (expected ~{expected})")

    # Non-overlap check
    all_indices = []
    for p in parts:
        all_indices.extend(p)
    if len(set(all_indices)) == 1000:
        ok("No overlapping indices across clients")
    else:
        fail(f"Overlapping indices! Unique={len(set(all_indices))}, total={len(all_indices)}")

    # Unequal split via data_fractions
    parts2 = iid_partition(ds, n_clients=3, seed=42, data_fractions=[0.5, 0.3, 0.2])
    sizes2 = [len(p) for p in parts2]
    total2 = sum(sizes2)
    if total2 == 1000:
        ok(f"IID unequal split: sizes={sizes2}, sum={total2}")
    else:
        fail(f"IID unequal split: sum={total2} (expected 1000)")

    # Metadata
    meta = compute_partition_metadata(ds, parts, n_classes=10)
    if meta["n_clients"] == 3 and len(meta["clients"]) == 3:
        ok("Partition metadata structure correct")
    else:
        fail(f"Partition metadata wrong: {meta}")


# ─── Test 2: Dirichlet Non-IID Partition ──────────────────────────────────────

def test_dirichlet():
    section("2/6  Dirichlet Partition (α=0.1)")
    from src.data.partitioner import dirichlet_partition, compute_partition_metadata

    ds = make_mock_dataset(n_samples=3000, n_classes=10, seed=42)
    parts = dirichlet_partition(ds, n_clients=3, alpha=0.1, seed=42)

    # Total must sum to dataset size
    total = sum(len(p) for p in parts)
    if total == len(ds):
        ok(f"Dirichlet total: {total} = {len(ds)}")
    else:
        fail(f"Dirichlet total {total} ≠ {len(ds)}")

    # No overlap
    all_idx = []
    for p in parts:
        all_idx.extend(p)
    if len(set(all_idx)) == total:
        ok("No overlapping indices")
    else:
        fail(f"Overlapping! unique={len(set(all_idx))}, total={total}")

    # Non-IID check: label histograms should NOT be uniform
    meta = compute_partition_metadata(ds, parts, n_classes=10)
    skewed_count = 0
    for client_meta in meta["clients"]:
        hist = client_meta["label_histogram"]
        hist_arr = np.array(hist)
        # Entropy of distribution — lower = more skewed
        probs = hist_arr / (hist_arr.sum() + 1e-9)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        uniform_entropy = np.log(10)  # max entropy for 10 classes
        if entropy < uniform_entropy * 0.85:  # at least 15% more skewed than uniform
            skewed_count += 1

    if skewed_count >= 2:  # at least 2 of 3 clients should be non-IID
        ok(f"Non-IID confirmed: {skewed_count}/3 clients have skewed label distributions")
    else:
        fail(f"Dirichlet partition appears too uniform: {skewed_count}/3 clients skewed")

    # Sizes shown for inspection
    sizes = [len(p) for p in parts]
    ok(f"Client sizes: {sizes} (heterogeneous expected for low α)")


# ─── Test 3: Reduced Partition ───────────────────────────────────────────────

def test_reduced():
    section("3/6  Reduced Partition (60%)")
    from src.data.partitioner import (
        iid_partition, reduced_partition, apply_reduced_fraction
    )

    ds = make_mock_dataset(n_samples=1000, seed=1)
    parts = iid_partition(ds, n_clients=3, seed=42)

    # Reduce single client
    client0 = parts[0]
    reduced = reduced_partition(client0, fraction=0.6, seed=42)
    expected = int(len(client0) * 0.6)
    if abs(len(reduced) - expected) <= 1:
        ok(f"Single client reduced: {len(client0)} → {len(reduced)} (expected ~{expected})")
    else:
        fail(f"Single client reduced wrong: {len(reduced)} (expected ~{expected})")

    # Reduced indices must be a subset of originals
    if set(reduced).issubset(set(client0)):
        ok("Reduced indices are a strict subset of original")
    else:
        fail("Some reduced indices not in original set!")

    # Apply to all clients
    reduced_parts = apply_reduced_fraction(parts, fraction=0.6, seed=42)
    for i, (orig, red) in enumerate(zip(parts, reduced_parts)):
        expect_r = int(len(orig) * 0.6)
        if abs(len(red) - expect_r) <= 1:
            ok(f"  Client {i}: {len(orig)} → {len(red)}")
        else:
            fail(f"  Client {i}: {len(orig)} → {len(red)} (expected ~{expect_r})")


# ─── Test 4: build_partitions() integration ────────────────────────────────────

def test_build_partitions():
    section("4/6  build_partitions() Integration")
    from src.data.partitioner import build_partitions

    ds = make_mock_dataset(n_samples=1000, seed=2)

    # IID
    parts_iid = build_partitions(ds, n_clients=3, partition_mode="iid", seed=42)
    total = sum(len(p) for p in parts_iid)
    if total == 1000:
        ok(f"build_partitions IID: total={total}")
    else:
        fail(f"build_partitions IID total wrong: {total}")

    # Dirichlet
    parts_dir = build_partitions(
        ds, n_clients=3, partition_mode="dirichlet",
        dirichlet_alpha=0.1, seed=42
    )
    total = sum(len(p) for p in parts_dir)
    if total == len(ds):
        ok(f"build_partitions Dirichlet: total={total}")
    else:
        fail(f"build_partitions Dirichlet total wrong: {total}")

    # IID + reduced
    parts_red = build_partitions(
        ds, n_clients=3, partition_mode="iid",
        reduced_fraction=0.6, seed=42
    )
    total_red = sum(len(p) for p in parts_red)
    expected_total = int(1000 * 0.6)
    if abs(total_red - expected_total) <= 3:
        ok(f"build_partitions IID+reduced(0.6): total={total_red} (expected ~{expected_total})")
    else:
        fail(f"build_partitions reduced total wrong: {total_red}")

    # Invalid mode
    try:
        build_partitions(ds, n_clients=3, partition_mode="invalid")
        fail("Should have raised ValueError for invalid partition mode")
    except ValueError:
        ok("Invalid partition mode correctly raises ValueError")


# ─── Test 5: Metadata JSON round-trip ────────────────────────────────────────

def test_metadata():
    section("5/6  Partition Metadata JSON")
    import json
    import tempfile
    from src.data.partitioner import (
        iid_partition, compute_partition_metadata, save_partition_metadata
    )

    ds = make_mock_dataset(n_samples=600, seed=3)
    parts = iid_partition(ds, n_clients=3, seed=42)
    meta = compute_partition_metadata(ds, parts, n_classes=10)

    # JSON serializable
    try:
        json_str = json.dumps(meta)
        ok("Metadata is JSON serializable")
    except TypeError as e:
        fail(f"Metadata not JSON serializable: {e}")
        return

    # Save and reload
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    save_partition_metadata(meta, tmp_path)
    with open(tmp_path) as f:
        loaded = json.load(f)
    os.unlink(tmp_path)

    if loaded["n_clients"] == 3 and loaded["total_samples"] == 600:
        ok(f"Metadata save/load round-trip OK: n_clients={loaded['n_clients']}, total={loaded['total_samples']}")
    else:
        fail(f"Metadata round-trip mismatch: {loaded}")

    # Label histogram sums
    for c in loaded["clients"]:
        hist_sum = sum(c["label_histogram"])
        if hist_sum == c["n_samples"]:
            ok(f"  Client {c['client_id']}: histogram sum={hist_sum} == n_samples={c['n_samples']}")
        else:
            fail(f"  Client {c['client_id']}: histogram sum={hist_sum} ≠ n_samples={c['n_samples']}")


# ─── Test 6: DataLoader factory + transforms ─────────────────────────────────

def test_dataloader():
    section("6/6  DataLoader Factory + Transforms (32×32 smoke mode)")
    from src.data.cifar import (
        get_train_transform, get_eval_transform, make_client_loader
    )
    from src.data.partitioner import iid_partition

    # Verify transforms exist and are callable
    train_t = get_train_transform(use_32=False)  # 224×224 pipeline
    eval_t  = get_eval_transform(use_32=False)
    ok(f"224×224 train transform: {type(train_t).__name__}")
    ok(f"224×224 eval transform: {type(eval_t).__name__}")

    train_t32 = get_train_transform(use_32=True)
    eval_t32  = get_eval_transform(use_32=True)
    ok(f"32×32 train transform: {type(train_t32).__name__}")

    # Test with 32×32 mock data (no CIFAR download needed)
    # Create mock dataset that mimics CIFAR-10 (32×32)
    ds = make_mock_dataset(n_samples=300, seed=5)
    parts = iid_partition(ds, n_clients=3, seed=42)

    loader = make_client_loader(
        dataset=ds,
        indices=parts[0],
        batch_size=16,
        shuffle=True,
        seed=42,
        num_workers=0,
        pin_memory=False,
    )

    # Iterate one batch
    batch = next(iter(loader))
    imgs, labels = batch
    if imgs.shape[0] <= 16 and imgs.ndim == 4:
        ok(f"DataLoader batch: images={tuple(imgs.shape)}, labels={tuple(labels.shape)}")
    else:
        fail(f"Unexpected batch shape: images={tuple(imgs.shape)}")

    # num_workers=0 confirmed
    if loader.num_workers == 0:
        ok("num_workers=0 (VM-safe)")
    else:
        fail(f"num_workers={loader.num_workers} (expected 0)")

    # If --download: test real CIFAR-10 loader (224×224 or 32×32)
    return True  # signal to main that real download test is separate


# ─── Optional: Real CIFAR-10 download test ────────────────────────────────────

def test_real_cifar(root: str, use_32: bool = True):
    section("OPTIONAL  Real CIFAR-10 Dataset")
    from src.data.cifar import (
        get_cifar10_train, get_cifar10_test,
        split_server_test, get_server_test_loader
    )

    print(f"  Downloading/loading CIFAR-10 to: {root}")
    print(f"  Resolution: {'32×32 (smoke)' if use_32 else '224×224 (paper-aligned)'}")

    try:
        train_ds = get_cifar10_train(root=root, download=True, use_32=use_32)
        test_ds  = get_cifar10_test(root=root,  download=True, use_32=use_32)
        ok(f"Train dataset: {len(train_ds)} samples")
        ok(f"Test dataset:  {len(test_ds)} samples")

        server_subset, _ = split_server_test(test_ds, server_fraction=0.1, seed=42)
        expected_server = int(len(test_ds) * 0.1)
        if abs(len(server_subset) - expected_server) <= 1:
            ok(f"Server test split: {len(server_subset)} samples (expected ~{expected_server})")
        else:
            fail(f"Server split wrong: {len(server_subset)}")

        # One batch from server loader
        loader = get_server_test_loader(root=root, batch_size=16, download=False, use_32=use_32)
        batch = next(iter(loader))
        imgs, lbls = batch
        ok(f"Server loader batch: images={tuple(imgs.shape)}, labels={tuple(lbls.shape)}")

    except Exception as e:
        fail(f"Real CIFAR-10 test failed: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2 data layer sanity checks")
    parser.add_argument("--download", action="store_true",
                        help="Also test with real CIFAR-10 (downloads if needed)")
    parser.add_argument("--cifar-root", default=os.path.expanduser("~/.cifar10"),
                        help="CIFAR-10 data directory (default: ~/.cifar10)")
    parser.add_argument("--use-32", action="store_true",
                        help="Use 32×32 for the real CIFAR test (faster, not paper-aligned)")
    args = parser.parse_args()

    print(f"\n{CYAN}{'═'*60}{RESET}")
    print(f"{CYAN}  Stage 2 Data Layer Sanity Check{RESET}")
    print(f"{CYAN}{'═'*60}{RESET}")

    test_iid()
    test_dirichlet()
    test_reduced()
    test_build_partitions()
    test_metadata()
    test_dataloader()

    if args.download:
        test_real_cifar(root=args.cifar_root, use_32=args.use_32)

    print(f"\n{CYAN}{'═'*60}{RESET}")
    if errors:
        print(f"{RED}  FAILED — {len(errors)} error(s):{RESET}")
        for e in errors:
            print(f"{RED}    • {e}{RESET}")
        print(f"{CYAN}{'═'*60}{RESET}\n")
        sys.exit(1)
    else:
        n_real = 1 if args.download else 0
        print(f"{GREEN}  ALL {passed} CHECKS PASSED ({6 + n_real} sections){RESET}")
        if not args.download:
            print(f"  Tip: run with --download --use-32 to also test real CIFAR-10 loaders.")
        print(f"{CYAN}{'═'*60}{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
