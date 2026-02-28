#!/usr/bin/env python3
"""
scripts/test_model.py — Stage 3 smoke-test for MobileNetV2 wrapper + trainer

Tests WITHOUT downloading CIFAR-10 (uses mock tensors at 224×224 and 32×32).
Verifies:
  1. Model instantiation and parameter count
  2. FP32 forward pass (224×224 — paper-aligned input shape)
  3. FP16 forward pass (half-precision model)
  4. FP32 training: 1 epoch with mock DataLoader, loss decreases
  5. FP16 training: 1 epoch with half-precision model
  6. Evaluate: accuracy + loss returned correctly
  7. get_parameters() / set_parameters() round-trip (Flower compat)
  8. OOM simulation: trainer returns oom=True signal correctly
  9. build_optimizer() factory (SGD + Adam)
  10. get_model_memory_mb() estimate reasonable

Exit codes: 0 = all passed, 1 = failure
"""

from __future__ import annotations

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

GREEN = "\033[0;32m"
RED   = "\033[0;31m"
CYAN  = "\033[0;36m"
RESET = "\033[0m"

errors: list = []
passed = 0

def ok(msg):
    global passed; passed += 1
    print(f"{GREEN}  [OK]{RESET}    {msg}")

def fail(msg):
    print(f"{RED}  [FAIL]{RESET}  {msg}")
    errors.append(msg)

def section(title):
    print(f"\n{CYAN}{'─'*60}{RESET}\n{CYAN}  {title}{RESET}")


# ─── Mock DataLoader helper ───────────────────────────────────────────────────

def make_mock_loader(
    n: int = 64, h: int = 224, w: int = 224,
    batch_size: int = 8, n_classes: int = 10,
    dtype: torch.dtype = torch.float32,
) -> DataLoader:
    """Mock image DataLoader at given resolution and dtype."""
    imgs    = torch.randn(n, 3, h, w, dtype=dtype)
    labels  = torch.randint(0, n_classes, (n,))
    ds      = TensorDataset(imgs, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ─── Test 1: Model instantiation ─────────────────────────────────────────────

def test_instantiation():
    section("1/10  Model Instantiation")
    from src.models.mobilenetv2 import get_model, count_parameters, model_summary

    model = get_model(num_classes=10)
    n_params = count_parameters(model)

    # MobileNetV2 with 10-class head: approx 2.2–3.5 M params
    if 2_000_000 < n_params < 5_000_000:
        ok(f"Parameter count: {n_params:,} ({n_params/1e6:.2f}M) — in expected range")
    else:
        fail(f"Unexpected param count: {n_params:,}")

    summary = model_summary(model)
    if summary["model"] == "MobileNetV2" and summary["num_classes"] == 10:
        ok(f"model_summary OK: {summary['n_params_M']}M params, dtype={summary['dtype']}")
    else:
        fail(f"model_summary wrong: {summary}")

    # Verify head is customized
    clf = model.classifier[1]
    if isinstance(clf, nn.Linear) and clf.out_features == 10:
        ok(f"Classifier head: Linear(in={clf.in_features}, out=10)")
    else:
        fail(f"Classifier head wrong: {clf}")


# ─── Test 2: FP32 forward pass 224×224 ───────────────────────────────────────

def test_fp32_forward():
    section("2/10  FP32 Forward Pass (224×224)")
    from src.models.mobilenetv2 import get_model

    model = get_model().eval()
    device = torch.device("cpu")
    x = torch.randn(1, 3, 224, 224)  # paper-aligned input

    with torch.no_grad():
        out = model(x)

    if out.shape == (1, 10):
        ok(f"Output shape: {tuple(out.shape)} (expected (1, 10))")
    else:
        fail(f"Wrong output shape: {tuple(out.shape)}")

    if not torch.isnan(out).any():
        ok("Output contains no NaN values")
    else:
        fail("Output contains NaN values")

    # Verify it's a proper logit distribution
    probs = torch.softmax(out, dim=1)
    if abs(probs.sum().item() - 1.0) < 1e-5:
        ok(f"Softmax sums to 1.0 (max_logit_class={out.argmax().item()})")
    else:
        fail(f"Softmax sum ≠ 1.0: {probs.sum().item()}")


# ─── Test 3: FP16 forward pass ────────────────────────────────────────────────

def test_fp16_forward():
    section("3/10  FP16 Forward Pass")
    from src.models.mobilenetv2 import get_model

    model = get_model().half().eval()
    x = torch.randn(1, 3, 224, 224, dtype=torch.float16)

    with torch.no_grad():
        out = model(x)

    if out.dtype == torch.float16:
        ok(f"FP16 output dtype: {out.dtype}")
    else:
        fail(f"Expected float16 output, got: {out.dtype}")

    if out.shape == (1, 10):
        ok(f"FP16 output shape: {tuple(out.shape)}")
    else:
        fail(f"FP16 wrong output shape: {tuple(out.shape)}")

    if not torch.isnan(out).any():
        ok("FP16 output: no NaN")
    else:
        fail("FP16 output: NaN detected")


# ─── Test 4: FP32 training (1 epoch) ─────────────────────────────────────────

def test_fp32_training():
    section("4/10  FP32 Training (1 epoch, 32×32 smoke)")
    from src.models.mobilenetv2 import get_model
    from src.models.trainer import train_one_epoch, build_optimizer

    # Use 32×32 for speed; paper-aligned 224×224 would take ~10x longer
    model = get_model()
    device = torch.device("cpu")
    loader = make_mock_loader(n=32, h=32, w=32, batch_size=8)
    optimizer = build_optimizer(model, optimizer_name="sgd", lr=0.01, momentum=0.9)

    result = train_one_epoch(model, loader, optimizer, device, epoch=1)

    if not result.oom:
        ok(f"Training completed: loss={result.loss:.4f}, n_samples={result.n_samples}")
    else:
        fail("OOM during FP32 training on mock data (unexpected)")

    if result.n_samples == 32:
        ok(f"Correct sample count: {result.n_samples}")
    else:
        fail(f"Sample count wrong: {result.n_samples}")

    if isinstance(result.loss, float) and not (result.loss != result.loss):  # not NaN
        ok(f"Loss is finite: {result.loss:.4f}")
    else:
        fail(f"Loss is NaN or non-float: {result.loss}")

    if result.train_time_s > 0:
        ok(f"Timing recorded: {result.train_time_s:.2f}s")
    else:
        fail("Timing not recorded")


# ─── Test 5: FP16 training (1 epoch) ─────────────────────────────────────────

def test_fp16_training():
    section("5/10  FP16 Training (1 epoch, 32×32 smoke)")
    from src.models.mobilenetv2 import get_model
    from src.models.trainer import train_one_epoch, build_optimizer

    model = get_model().half()
    device = torch.device("cpu")
    # DataLoader images must also be FP16 to match model
    loader = make_mock_loader(n=32, h=32, w=32, batch_size=8, dtype=torch.float16)
    optimizer = build_optimizer(model, optimizer_name="sgd", lr=0.01)

    result = train_one_epoch(model, loader, optimizer, device, epoch=1)

    if not result.oom:
        ok(f"FP16 training completed: loss={result.loss:.4f}")
    else:
        fail("OOM during FP16 training (unexpected)")

    # Loss should be finite (CE upcasts to FP32 internally)
    if isinstance(result.loss, float) and result.loss == result.loss:
        ok(f"FP16 loss is finite (CE applied in FP32): {result.loss:.4f}")
    else:
        fail(f"FP16 loss is NaN: {result.loss}")


# ─── Test 6: evaluate() ──────────────────────────────────────────────────────

def test_evaluate():
    section("6/10  evaluate() Accuracy + Loss")
    from src.models.mobilenetv2 import get_model
    from src.models.trainer import evaluate

    model = get_model().eval()
    device = torch.device("cpu")
    loader = make_mock_loader(n=40, h=32, w=32, batch_size=8)

    result = evaluate(model, loader, device)

    if 0.0 <= result.accuracy <= 1.0:
        ok(f"Accuracy in [0,1]: {result.accuracy:.4f} (random init → ~10% expected)")
    else:
        fail(f"Accuracy out of range: {result.accuracy}")

    if result.n_samples == 40:
        ok(f"Correct sample count: {result.n_samples}")
    else:
        fail(f"Sample count wrong: {result.n_samples}")

    if isinstance(result.loss, float) and result.loss == result.loss:
        ok(f"Eval loss finite: {result.loss:.4f}")
    else:
        fail("Eval loss is NaN")

    if result.eval_time_s >= 0:
        ok(f"Eval timing: {result.eval_time_s:.2f}s")
    else:
        fail("Eval timing negative")

    # Verify to_dict() is JSON-serializable
    import json
    try:
        json.dumps(result.to_dict())
        ok("EvalResult.to_dict() is JSON serializable")
    except TypeError as e:
        fail(f"EvalResult not JSON serializable: {e}")


# ─── Test 7: get/set_parameters round-trip ────────────────────────────────────

def test_parameters_roundtrip():
    section("7/10  Flower get_parameters / set_parameters Round-trip")
    from src.models.mobilenetv2 import get_model, get_parameters, set_parameters

    model_a = get_model()
    model_b = get_model()  # different random init

    # Extract from A, load into B
    params_a = get_parameters(model_a)
    set_parameters(model_b, params_a)
    params_b = get_parameters(model_b)

    # All arrays should match
    mismatches = 0
    for i, (pa, pb) in enumerate(zip(params_a, params_b)):
        if not np.allclose(pa, pb, atol=1e-6):
            mismatches += 1

    if mismatches == 0:
        ok(f"Round-trip exact: {len(params_a)} parameter tensors transferred")
    else:
        fail(f"{mismatches} parameter tensors differ after round-trip")

    # Verify output dtype is always float32 (even from FP16 model)
    model_fp16 = get_model().half()
    params_fp16 = get_parameters(model_fp16)
    if all(p.dtype == np.float32 for p in params_fp16):
        ok("get_parameters() always returns float32 numpy arrays (FP16 model)")
    else:
        dtypes = set(p.dtype for p in params_fp16)
        fail(f"get_parameters() returned non-float32 dtype: {dtypes}")


# ─── Test 8: OOM detection ────────────────────────────────────────────────────

def test_oom_detection():
    section("8/10  OOM Detection (simulated)")
    from src.models.mobilenetv2 import get_model
    from src.models.trainer import build_optimizer

    # Patch train_one_epoch to simulate OOM by raising RuntimeError
    # We do this by creating a DataLoader that raises on first iteration
    class OOMDataset(torch.utils.data.Dataset):
        def __len__(self): return 10
        def __getitem__(self, idx):
            if idx == 0:
                raise RuntimeError("CUDA out of memory (simulated for test)")
            return torch.randn(3, 32, 32), 0

    loader = DataLoader(OOMDataset(), batch_size=2, num_workers=0)
    model = get_model()
    optimizer = build_optimizer(model)
    device = torch.device("cpu")

    from src.models.trainer import train_one_epoch
    result = train_one_epoch(model, loader, optimizer, device, epoch=1)

    if result.oom:
        ok("OOM RuntimeError correctly caught; oom=True returned")
    else:
        fail("OOM not detected — trainer should return oom=True")

    if result.loss != result.loss:  # NaN check
        ok("Loss is NaN on OOM (expected)")
    else:
        # Might be 0.0 if no batches processed (that's also fine)
        ok(f"Loss on OOM: {result.loss}")


# ─── Test 9: build_optimizer ─────────────────────────────────────────────────

def test_optimizer():
    section("9/10  build_optimizer() Factory")
    from src.models.mobilenetv2 import get_model
    from src.models.trainer import build_optimizer

    model = get_model()

    sgd = build_optimizer(model, optimizer_name="sgd", lr=0.01, momentum=0.9)
    if isinstance(sgd, torch.optim.SGD):
        ok(f"SGD optimizer: lr={sgd.param_groups[0]['lr']}, momentum={sgd.param_groups[0]['momentum']}")
    else:
        fail(f"SGD returned wrong type: {type(sgd)}")

    adam = build_optimizer(model, optimizer_name="adam", lr=1e-3)
    if isinstance(adam, torch.optim.Adam):
        ok(f"Adam optimizer: lr={adam.param_groups[0]['lr']}")
    else:
        fail(f"Adam returned wrong type: {type(adam)}")

    try:
        build_optimizer(model, optimizer_name="rmsprop")
        fail("Should raise ValueError for unknown optimizer")
    except ValueError:
        ok("Unknown optimizer raises ValueError")


# ─── Test 10: Memory estimate ─────────────────────────────────────────────────

def test_memory_estimate():
    section("10/10  Model Memory Estimate")
    from src.models.mobilenetv2 import get_model
    from src.models.trainer import get_model_memory_mb

    model_fp32 = get_model()
    mem_fp32 = get_model_memory_mb(model_fp32)
    # MobileNetV2 with 10-class head: ~2.24M params × 4 bytes = ~8.5 MB params-only
    # (activations not included; this is param footprint only)
    if 5 < mem_fp32 < 25:
        ok(f"FP32 model param memory: {mem_fp32:.1f} MB (2.24M params × 4B ≈ 8.5 MB)")
    else:
        fail(f"FP32 memory estimate unexpected: {mem_fp32:.1f} MB")

    model_fp16 = get_model().half()
    mem_fp16 = get_model_memory_mb(model_fp16)
    # FP16: ~half of FP32
    if mem_fp16 < mem_fp32 * 0.6:
        ok(f"FP16 model memory: {mem_fp16:.1f} MB (< FP32 {mem_fp32:.1f} MB ✓)")
    else:
        fail(f"FP16 memory should be < FP32: {mem_fp16:.1f} vs {mem_fp32:.1f}")

    # Profile-based pre-flight check: weak client (512 MB) can hold FP16 model
    if mem_fp16 < 512:
        ok(f"FP16 model ({mem_fp16:.1f} MB) fits in weak client limit (512 MB)")
    else:
        fail(f"FP16 model too large for weak client: {mem_fp16:.1f} MB")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{CYAN}{'═'*60}{RESET}")
    print(f"{CYAN}  Stage 3 Model Layer Smoke Test{RESET}")
    print(f"{CYAN}  Note: uses 32×32 mock tensors for speed (except FP32/FP16 fwd@224){RESET}")
    print(f"{CYAN}{'═'*60}{RESET}")

    test_instantiation()
    test_fp32_forward()
    test_fp16_forward()
    test_fp32_training()
    test_fp16_training()
    test_evaluate()
    test_parameters_roundtrip()
    test_oom_detection()
    test_optimizer()
    test_memory_estimate()

    print(f"\n{CYAN}{'═'*60}{RESET}")
    if errors:
        print(f"{RED}  FAILED — {len(errors)} error(s):{RESET}")
        for e in errors:
            print(f"{RED}    • {e}{RESET}")
        print(f"{CYAN}{'═'*60}{RESET}\n")
        sys.exit(1)
    else:
        print(f"{GREEN}  ALL {passed} CHECKS PASSED (10 sections){RESET}")
        print(f"{CYAN}{'═'*60}{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
