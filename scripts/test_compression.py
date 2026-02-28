#!/usr/bin/env python3
"""
scripts/test_compression.py — Phase 4 smoke test for all quantization modes

Tests WITHOUT downloading CIFAR-10 (mock tensors only).
Verifies:
    1. FP32 passthrough: model dtype unchanged, deep copy
    2. FP16 conversion: dtype=float16, independent copy
    3. Static INT8: forward pass succeeds on small mock calibration data
    4. Fallback chain: if INT8 fails (injected error), returns fp16_fallback
    5. quantize_action() mapping: actions 0/1/2/3 behave correctly
    6. quantize() raises on invalid bits and missing calib_loader
    7. Caller model is NEVER mutated by any quantize call
    8. method strings: exactly match expected constants
    9. INT8 model produces correct output shape (inference)
   10. FP32 fallback: if both INT8 + FP16 fail, FP32 returned

Exit codes: 0 = all passed, 1 = failure
"""

from __future__ import annotations

import sys
import os
import copy
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set up INFO logging so QUANT_UNSUPPORTED messages are visible in test output
logging.basicConfig(level=logging.INFO, format="  [LOG] %(name)s: %(message)s")

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_model() -> nn.Module:
    from src.models.mobilenetv2 import get_model
    return get_model().eval()


def make_calib_loader(n: int = 32, h: int = 32, w: int = 32, batch: int = 8) -> DataLoader:
    """Mock calibration loader — 32×32 tensors for speed (full 224 would take minutes)."""
    imgs   = torch.randn(n, 3, h, w)
    labels = torch.zeros(n, dtype=torch.long)
    return DataLoader(TensorDataset(imgs, labels), batch_size=batch, num_workers=0)


def param_snapshot(model: nn.Module) -> list:
    """Return list of first-param values as a mutation-detection snapshot."""
    for p in model.parameters():
        return p.detach().cpu().float().flatten()[:8].tolist()
    return []


# ── Test 1: FP32 passthrough ──────────────────────────────────────────────────

def test_fp32():
    section("1/10  FP32 Passthrough")
    from src.compression.quantizer import quantize, METHOD_FP32

    model = make_model()
    snap_before = param_snapshot(model)

    q_model, method = quantize(model, bits=32)

    if method == METHOD_FP32:
        ok(f"method='{method}' correct")
    else:
        fail(f"method='{method}' (expected '{METHOD_FP32}')")

    if q_model is not model:
        ok("Returns deep copy (not same object)")
    else:
        fail("Returned same object — should be a copy")

    # Original unchanged
    if param_snapshot(model) == snap_before:
        ok("Original model NOT mutated")
    else:
        fail("Original model was mutated by fp32.apply!")

    # dtype is float32
    first_param = next(q_model.parameters())
    if first_param.dtype == torch.float32:
        ok(f"Output model dtype: {first_param.dtype}")
    else:
        fail(f"Expected float32, got: {first_param.dtype}")

    # Forward pass still works
    with torch.no_grad():
        out = q_model(torch.randn(1, 3, 32, 32))
    if out.shape == (1, 10):
        ok("FP32 forward pass: shape (1,10)")
    else:
        fail(f"FP32 forward pass wrong shape: {out.shape}")


# ── Test 2: FP16 conversion ───────────────────────────────────────────────────

def test_fp16():
    section("2/10  FP16 Conversion")
    from src.compression.quantizer import quantize, METHOD_FP16

    model = make_model()
    snap_before = param_snapshot(model)

    q_model, method = quantize(model, bits=16)

    if method == METHOD_FP16:
        ok(f"method='{method}' correct")
    else:
        fail(f"method='{method}' (expected '{METHOD_FP16}')")

    if q_model is not model:
        ok("Returns deep copy")
    else:
        fail("Returned same object")

    first_param = next(q_model.parameters())
    if first_param.dtype == torch.float16:
        ok(f"Output dtype: {first_param.dtype}")
    else:
        fail(f"Expected float16, got: {first_param.dtype}")

    if param_snapshot(model) == snap_before:
        ok("Original model NOT mutated")
    else:
        fail("Original mutated!")

    with torch.no_grad():
        out = q_model(torch.randn(1, 3, 32, 32, dtype=torch.float16))
    if out.dtype == torch.float16 and out.shape == (1, 10):
        ok(f"FP16 forward pass: shape {tuple(out.shape)}, dtype {out.dtype}")
    else:
        fail(f"FP16 forward pass: shape {tuple(out.shape)}, dtype {out.dtype}")


# ── Test 3: Static INT8 — per-backend probe ───────────────────────────────────

def test_int8_static():
    section("3/10  Static INT8 — Per-Backend Probe + Auto-Select")
    from src.compression import int8 as int8_mod
    from src.compression.quantizer import quantize, METHOD_STATIC_INT8, METHOD_FP16_FALLBACK, METHOD_FP32_FALLBACK

    # Print supported engines (this is what Omar wants logged)
    engines = list(torch.backends.quantized.supported_engines)
    print(f"\n  supported_engines = {engines}")
    real_engines = [e for e in engines if e != "none"]
    print(f"  Probing: {real_engines}")

    # 1) Per-backend probe using _probe_backend_works directly
    model = make_model()
    calib = make_calib_loader(n=8, h=32, w=32)
    backend_results = {}
    for backend in real_engines:
        worked = int8_mod._probe_backend_works(model, calib, backend)
        backend_results[backend] = worked
        status = "✓ INT8 OK" if worked else "✗ fallback"
        print(f"    backend='{backend}': {status}")
        ok(f"backend='{backend}' probe completed (result={'static_int8' if worked else 'fallback'})")

    any_int8_works = any(backend_results.values())

    # 2) Auto-select path via quantize(backend=None)
    model2 = make_model()
    calib2 = make_calib_loader(n=8, h=32, w=32)
    q_model, method = quantize(model2, bits=8, calib_loader=calib2, calibration_samples=8)

    if any_int8_works:
        if method == METHOD_STATIC_INT8:
            ok(f"Auto-select: static_int8 succeeded (backend found ✓)")
        else:
            fail(f"At least one backend worked in probe but auto-select returned '{method}'")
    else:
        # All backends failed — fp16_fallback or fp32_fallback is correct
        if method in (METHOD_FP16_FALLBACK, METHOD_FP32_FALLBACK):
            ok(f"Auto-select: all backends failed → '{method}' (correct QUANT_UNSUPPORTED path)")
        else:
            fail(f"All backends failed but got unexpected method='{method}'")

    # Forward pass with dtype-matched input
    first_param = next(q_model.parameters())
    input_dtype = torch.float32 if method == METHOD_STATIC_INT8 else first_param.dtype
    x = torch.randn(1, 3, 32, 32, dtype=input_dtype)
    with torch.no_grad():
        out = q_model(x)
    if out.shape == (1, 10):
        ok(f"Forward pass: shape {tuple(out.shape)} (method={method})")
    else:
        fail(f"Wrong shape: {tuple(out.shape)}")

    if not torch.isnan(out).any():
        ok("Output has no NaN values")
    else:
        fail("Output contains NaN!")


# ── Test 4: Fallback chain (injected INT8 error) ──────────────────────────────

def test_int8_fallback_injected():
    section("4/10  Fallback Chain (injected INT8 failure)")
    from src.compression import int8 as int8_mod
    from src.compression.quantizer import METHOD_FP16_FALLBACK

    model = make_model()
    calib = make_calib_loader(n=8)

    # Pass backend='qnnpack' explicitly to skip auto_select_backend() probing.
    # Patch _fuse_mobilenetv2 to raise immediately inside the try block.
    orig_fuse = int8_mod._fuse_mobilenetv2
    def _bad_fuse(m):
        raise RuntimeError("Simulated QUANT_UNSUPPORTED: unsupported op DeConv2d")
    int8_mod._fuse_mobilenetv2 = _bad_fuse

    try:
        from src.compression.int8 import try_static_int8
        q_model, method = try_static_int8(model, calib, calibration_samples=8, backend='qnnpack')
    finally:
        int8_mod._fuse_mobilenetv2 = orig_fuse  # restore

    if method == METHOD_FP16_FALLBACK:
        ok(f"Fallback: method='{method}' (INT8 failed → FP16)")
    else:
        fail(f"Expected '{METHOD_FP16_FALLBACK}', got '{method}'")

    first_param = next(q_model.parameters())
    if first_param.dtype == torch.float16:
        ok("Fallback model dtype: float16")
    else:
        fail(f"Fallback model dtype: {first_param.dtype} (expected float16)")


# ── Test 5: quantize_action() mapping ────────────────────────────────────────

def test_quantize_action():
    section("5/10  quantize_action() Action Mapping")
    from src.compression.quantizer import quantize_action, METHOD_FP32, METHOD_FP16

    model = make_model()

    # action=1 → bits=32
    _, method = quantize_action(model, action=1)
    if method == METHOD_FP32:
        ok(f"action=1 → method='{method}' ✓")
    else:
        fail(f"action=1 → method='{method}' (expected '{METHOD_FP32}')")

    # action=2 → bits=16
    _, method = quantize_action(model, action=2)
    if method == METHOD_FP16:
        ok(f"action=2 → method='{method}' ✓")
    else:
        fail(f"action=2 → method='{method}' (expected '{METHOD_FP16}')")

    # action=0 → skip → must raise
    try:
        quantize_action(model, action=0)
        fail("action=0 (skip) should raise ValueError")
    except ValueError as e:
        ok(f"action=0 raises ValueError: {str(e)[:60]}")

    # invalid action
    try:
        quantize_action(model, action=9)
        fail("action=9 should raise ValueError")
    except ValueError:
        ok("action=9 raises ValueError")


# ── Test 6: Input validation ───────────────────────────────────────────────────

def test_input_validation():
    section("6/10  Input Validation (ValueError guards)")
    from src.compression.quantizer import quantize

    model = make_model()

    # Invalid bits
    try:
        quantize(model, bits=4)
        fail("bits=4 should raise ValueError")
    except ValueError as e:
        ok(f"bits=4 raises ValueError: {str(e)[:60]}")

    # bits=8 without calib_loader
    try:
        quantize(model, bits=8, calib_loader=None)
        fail("bits=8 with calib_loader=None should raise ValueError")
    except ValueError as e:
        ok(f"bits=8, calib_loader=None raises ValueError: {str(e)[:60]}")


# ── Test 7: No mutation of caller's model ────────────────────────────────────

def test_no_mutation():
    section("7/10  No Mutation of Caller's Model")
    from src.compression.quantizer import quantize

    model = make_model()
    original_id = id(model)
    snap = param_snapshot(model)
    original_dtype = next(model.parameters()).dtype

    # FP16
    _, _ = quantize(model, bits=16)
    if id(model) == original_id and next(model.parameters()).dtype == original_dtype:
        ok("FP16 quantize: caller model not mutated")
    else:
        fail("FP16 quantize mutated caller model!")

    # FP32
    _, _ = quantize(model, bits=32)
    if param_snapshot(model) == snap:
        ok("FP32 quantize: caller model not mutated")
    else:
        fail("FP32 quantize mutated caller model!")


# ── Test 8: Method string constants ───────────────────────────────────────────

def test_method_strings():
    section("8/10  Method String Constants")
    from src.compression.quantizer import (
        METHOD_FP32, METHOD_FP16, METHOD_STATIC_INT8,
        METHOD_FP16_FALLBACK, METHOD_FP32_FALLBACK
    )

    expected = {
        "METHOD_FP32":          ("fp32",          METHOD_FP32),
        "METHOD_FP16":          ("fp16",          METHOD_FP16),
        "METHOD_STATIC_INT8":   ("static_int8",   METHOD_STATIC_INT8),
        "METHOD_FP16_FALLBACK": ("fp16_fallback", METHOD_FP16_FALLBACK),
        "METHOD_FP32_FALLBACK": ("fp32_fallback", METHOD_FP32_FALLBACK),
    }
    for name, (expected_val, actual_val) in expected.items():
        if actual_val == expected_val:
            ok(f"{name} = '{actual_val}'")
        else:
            fail(f"{name}: expected '{expected_val}', got '{actual_val}'")


# ── Test 9: INT8 output shape ─────────────────────────────────────────────────

def test_int8_output_shape():
    section("9/10  INT8 (or fallback) Forward Pass Output Shape")
    from src.compression.quantizer import quantize

    model = make_model()
    calib = make_calib_loader(n=16, h=32, w=32)
    q_model, method = quantize(model, bits=8, calib_loader=calib, calibration_samples=16)

    # Input dtype must match fallback model dtype (fp16_fallback → float16 params)
    first_param = next(q_model.parameters())
    x = torch.randn(4, 3, 32, 32, dtype=first_param.dtype)
    with torch.no_grad():
        out = q_model(x)

    if out.shape == (4, 10):
        ok(f"Batch output shape: {tuple(out.shape)} (method={method})")
    else:
        fail(f"Wrong batch output shape: {tuple(out.shape)}")

    if not torch.isnan(out).any():
        ok("No NaN in batch output")
    else:
        fail("Batch output contains NaN")


# ── Test 10: FP32 fallback (both INT8 + FP16 injected to fail) ───────────────

def test_fp32_fallback():
    section("10/10  FP32 Fallback (INT8 + FP16 both injected to fail)")
    from src.compression import int8 as int8_mod
    from src.compression import fp16 as fp16_mod
    from src.compression.quantizer import METHOD_FP32_FALLBACK

    model = make_model()
    calib = make_calib_loader(n=4)

    # Pass backend='qnnpack' to skip auto_select_backend; patch both
    # _fuse_mobilenetv2 (INT8 injection) and _fp16_module.apply (FP16 injection)
    orig_fuse = int8_mod._fuse_mobilenetv2
    orig_fp16_apply = int8_mod._fp16_module.apply

    def _bad_fuse(m): raise RuntimeError("Injected INT8 failure")
    def _bad_fp16(m, inplace=False): raise RuntimeError("Injected FP16 failure")

    int8_mod._fuse_mobilenetv2 = _bad_fuse
    int8_mod._fp16_module.apply = _bad_fp16

    try:
        from src.compression.int8 import try_static_int8
        q_model, method = try_static_int8(model, calib, calibration_samples=4, backend='qnnpack')
    finally:
        int8_mod._fuse_mobilenetv2 = orig_fuse
        int8_mod._fp16_module.apply = orig_fp16_apply

    if method == METHOD_FP32_FALLBACK:
        ok(f"Double-fallback: method='{method}' (INT8+FP16 failed → FP32)")
    else:
        fail(f"Expected '{METHOD_FP32_FALLBACK}', got '{method}'")

    first_param = next(q_model.parameters())
    if first_param.dtype == torch.float32:
        ok("FP32 fallback model dtype: float32")
    else:
        fail(f"FP32 fallback dtype wrong: {first_param.dtype}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{CYAN}{'═'*60}{RESET}")
    print(f"{CYAN}  Phase 4 Compression Layer Smoke Test{RESET}")
    print(f"{CYAN}  Note: INT8 uses 32×32 mock tensors for speed{RESET}")
    print(f"{CYAN}{'═'*60}{RESET}")

    test_fp32()
    test_fp16()
    test_int8_static()
    test_int8_fallback_injected()
    test_quantize_action()
    test_input_validation()
    test_no_mutation()
    test_method_strings()
    test_int8_output_shape()
    test_fp32_fallback()

    print(f"\n{CYAN}{'═'*60}{RESET}")
    if errors:
        print(f"{RED}  FAILED — {len(errors)} error(s):{RESET}")
        for e in errors:
            print(f"{RED}    • {e}{RESET}")
        sys.exit(1)
    else:
        print(f"{GREEN}  ALL {passed} CHECKS PASSED (10 sections){RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
