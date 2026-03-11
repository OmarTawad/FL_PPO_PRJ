"""
src/compression/int8.py — Static INT8 quantization for MobileNetV2

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Pipeline: fuse → prepare (insert observers) → calibrate (128 samples) → convert → verify_inference

Fallback chain (NEVER silent):
    static INT8 fails  →  log QUANT_UNSUPPORTED  →  return lowp_dtype model
    lowp also fails    →  log QUANT_UNSUPPORTED  →  return FP32 model

Dynamic INT8 (quantize_dynamic) is NEVER used as a fallback (SPEC.md §4.2).

Backend auto-selection (SPEC.md §4.2):
    try_static_int8() probes torch.backends.quantized.supported_engines at runtime
    and picks the first working backend in order: fbgemm → x86 → onednn → qnnpack.
    Pass backend='qnnpack' (or any other) to override.

Usage:
    from src.compression.int8 import try_static_int8, QUANT_UNSUPPORTED
    model, method = try_static_int8(model, calib_loader)
    # method in {"static_int8", "bf16_fallback", "fp16_fallback", "fp32_fallback"}
"""

from __future__ import annotations

import copy
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.compression import fp16 as _fp16_module
from src.compression import bf16 as _bf16_module
from src.compression.lowp import (
    fallback_method_for_lowp_dtype,
    resolve_lowp_dtype,
)

# Log key emitted on every fallback (SPEC.md §4.2)
QUANT_UNSUPPORTED = "QUANT_UNSUPPORTED"

# Backend preference order — outer try goes through these in order
_BACKEND_PREFERENCE = ["fbgemm", "x86", "onednn", "qnnpack"]

log = logging.getLogger(__name__)


# ─── Backend helpers ──────────────────────────────────────────────────────────

def _log_supported_engines() -> List[str]:
    """Log available quantized engines and return the list."""
    engines = list(torch.backends.quantized.supported_engines)
    log.info(f"torch.backends.quantized.supported_engines = {engines}")
    log.info(f"default engine = {torch.backends.quantized.engine}")
    return engines


def _probe_backend_works(model: nn.Module, calib_loader: DataLoader, backend: str) -> bool:
    """
    Quick 1-batch probe: fuse → prepare → calibrate → convert → infer.
    Returns True if the backend runs INT8 end-to-end. Does NOT mutate model.
    """
    try:
        m = copy.deepcopy(model).float().eval()
        m = _fuse_mobilenetv2(m)
        torch.backends.quantized.engine = backend
        m.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(m, inplace=True)
        with torch.no_grad():
            for imgs, _ in calib_loader:
                m(imgs.float()); break
        torch.quantization.convert(m, inplace=True)
        with torch.no_grad():
            for imgs, _ in calib_loader:
                m(imgs.float()); break
        return True
    except Exception as e:
        log.debug(f"  backend='{backend}' probe: {type(e).__name__}: {str(e)[:80]}")
        return False


def auto_select_backend(model: nn.Module, calib_loader: DataLoader) -> Optional[str]:
    """
    Probe supported_engines in preference order (fbgemm→x86→onednn→qnnpack).
    Returns the first working backend name, or None if none work.
    'none' is skipped — not a real quantization backend.
    """
    available = set(torch.backends.quantized.supported_engines) - {"none"}
    log.info(f"Auto-selecting INT8 backend from: {_BACKEND_PREFERENCE}")
    for backend in _BACKEND_PREFERENCE:
        if backend not in available:
            continue
        log.info(f"  Probing backend='{backend}'...")
        if _probe_backend_works(model, calib_loader, backend):
            log.info(f"  Backend='{backend}' succeeded ✓")
            return backend
        log.info(f"  Backend='{backend}' failed ✗")
    log.warning(f"{QUANT_UNSUPPORTED}: all backends failed: {sorted(available)}")
    return None


# ─── MobileNetV2-specific fusion ──────────────────────────────────────────────

def _fuse_mobilenetv2(model: nn.Module) -> nn.Module:
    """
    Fuse Conv-BN-ReLU6 patterns in MobileNetV2 for quantization.

    PyTorch's eager-mode quantization requires fused modules.
    MobileNetV2 has two patterns:
      1. features[0]: ConvBNActivation  → [0, 1, 2] (Conv2d, BN, ReLU6)
      2. features[i].conv: nested ConvBNActivation blocks

    We use a safe recursive fusion that skips any block where fusion fails.
    """
    model.eval()  # BN must be in eval mode before fusion

    # features[0] is the initial ConvBNActivation (stride=2 block)
    try:
        torch.quantization.fuse_modules(
            model.features[0], ["0", "1", "2"], inplace=True
        )
    except Exception as e:
        log.debug(f"features[0] fusion skipped: {e}")

    # features[1..N-1]: InvertedResidual blocks
    for i in range(1, len(model.features) - 1):
        block = model.features[i]
        if not hasattr(block, "conv"):
            continue
        sub = block.conv
        # Each InvertedResidual.conv is a Sequential with sub-blocks
        # Typical layout for expansion_ratio > 1: [0]=expand ConvBN, [1]=dw ConvBN, [2]=project ConvBN
        # For expansion_ratio == 1: [0]=dw ConvBN, [1]=project ConvBN
        n = len(sub)
        for j in range(n):
            child = sub[j]
            # ConvBNActivation has children [0]=Conv2d, [1]=BN, [2]=ReLU6
            children = list(child.children())
            if len(children) >= 3:
                try:
                    torch.quantization.fuse_modules(
                        child, ["0", "1", "2"], inplace=True
                    )
                except Exception:
                    pass
            elif len(children) == 2:
                try:
                    torch.quantization.fuse_modules(
                        child, ["0", "1"], inplace=True
                    )
                except Exception:
                    pass

    # features[-1]: last ConvBNActivation (1x1 expansion)
    try:
        torch.quantization.fuse_modules(
            model.features[-1], ["0", "1", "2"], inplace=True
        )
    except Exception as e:
        log.debug(f"features[-1] fusion skipped: {e}")

    return model


def _calibrate(
    model: nn.Module,
    calib_loader: DataLoader,
    n_samples: int = 128,
) -> int:
    """
    Run calibration forward passes to collect observer statistics.

    Args:
        model: Prepared model (observers already attached).
        calib_loader: DataLoader providing (image, label) batches.
        n_samples: Stop after this many samples.

    Returns:
        Actual number of samples seen.
    """
    model.eval()
    seen = 0
    with torch.no_grad():
        for images, _ in calib_loader:
            # Always feed FP32 to the prepared model
            if images.dtype != torch.float32:
                images = images.float()
            model(images)
            seen += len(images)
            if seen >= n_samples:
                break
    return seen


def _verify_inference(model: nn.Module, calib_loader: DataLoader) -> None:
    """
    Run one batch through the converted model to confirm the quantized
    kernels are available at runtime.

    This catches the case where torch.quantization.convert() succeeds but
    the QuantizedCPU kernel is absent (e.g. torch+cpu PyPI wheel does not
    ship the quantized::conv2d.new operator). Raises RuntimeError if inference
    fails so the outer try/except triggers the QUANT_UNSUPPORTED fallback.
    """
    model.eval()
    with torch.no_grad():
        for images, _ in calib_loader:
            if images.dtype != torch.float32:
                images = images.float()
            model(images)   # raises NotImplementedError / RuntimeError if kernel missing
            break            # one batch is enough


def try_static_int8(
    model: nn.Module,
    calib_loader: DataLoader,
    calibration_samples: int = 128,
    backend: Optional[str] = None,
    inplace: bool = False,
    lowp_dtype: str = "bf16",
) -> Tuple[nn.Module, str]:
    """
    Attempt static INT8 quantization with full fallback chain.

    Backend selection (SPEC.md §4.2):
        - backend=None (default): probe supported_engines in order
          fbgemm → x86 → onednn → qnnpack and use the first that works.
        - backend='fbgemm' (or any string): force that backend; skip probing.

    Steps:
        1. Log supported_engines
        2. Select backend (auto or explicit)
        3. Deep-copy model
        4. Fuse Conv-BN-ReLU6
        5. Set qconfig → prepare (insert observers)
        6. Calibrate (forward pass over n_samples images)
        7. Convert (replaces float ops with quantized ops)
        8. Verify inference (1 batch) — catches missing QuantizedCPU kernels

    On any failure in steps 4–8:
        → logs QUANT_UNSUPPORTED with error detail
        → falls back to lowp_dtype (bf16/fp16), then FP32 if lowp also fails

    Args:
        model: Source FP32 model (eval or train mode).
        calib_loader: DataLoader for calibration (128 images sufficient).
        calibration_samples: Number of images to run through observers.
        backend: None = auto-select; string = force that backend.
        inplace: Ignored — we always deep-copy for safety.

    Returns:
        Tuple (quantized_model, method_string) where method_string is one of:
            "static_int8"    — full static INT8 conversion succeeded
            "bf16_fallback"  — INT8 failed; returned BF16 model
            "fp16_fallback"  — INT8 failed; returned FP16 model
            "fp32_fallback"  — INT8 + lowp failed; returned FP32 model
    """
    resolved_lowp = resolve_lowp_dtype(lowp_dtype)
    lowp_apply = _bf16_module.apply if resolved_lowp == "bf16" else _fp16_module.apply
    lowp_method = fallback_method_for_lowp_dtype(resolved_lowp)

    # Step 1: Log supported engines
    _log_supported_engines()

    # Step 2: Select backend
    if backend is None:
        selected = auto_select_backend(model, calib_loader)
        if selected is None:
            # All backends failed probe — invoke fallback INLINE (never crash).
            # RuntimeError here would be outside the try/except below, breaking
            # the "never silent fallback" contract (SPEC.md §4.2).
            log.warning(
                f"{QUANT_UNSUPPORTED}: no working INT8 backend found in "
                f"{list(torch.backends.quantized.supported_engines)}. "
                f"int8_disabled=true. Triggering {resolved_lowp.upper()} fallback."
            )
            try:
                lowp_model = lowp_apply(model, inplace=False)
                log.info(
                    f"Fallback to {resolved_lowp.upper()} successful "
                    f"→ quant_method={lowp_method}"
                )
                return lowp_model, lowp_method
            except Exception as lowp_err:
                log.warning(
                    f"{QUANT_UNSUPPORTED}: {resolved_lowp.upper()} fallback also failed "
                    f"({type(lowp_err).__name__}). Falling back to FP32. "
                    f"quant_method=fp32_fallback"
                )
                return copy.deepcopy(model).float().eval(), "fp32_fallback"
        backend = selected
        log.info(f"Auto-selected backend='{backend}'")
    else:
        log.info(f"Using forced backend='{backend}'")

    # Always work on a deep copy — never mutate the caller's model
    work_model = copy.deepcopy(model).float().eval()

    try:
        # Step 3: Fuse
        work_model = _fuse_mobilenetv2(work_model)

        # Step 4: Set qconfig
        torch.backends.quantized.engine = backend
        work_model.qconfig = torch.quantization.get_default_qconfig(backend)

        # Step 3: Prepare (insert observers)
        torch.quantization.prepare(work_model, inplace=True)

        # Step 4: Calibrate
        seen = _calibrate(work_model, calib_loader, n_samples=calibration_samples)
        log.info(f"INT8 calibration: {seen} samples processed (backend={backend})")

        # Step 5: Convert
        torch.quantization.convert(work_model, inplace=True)

        # Step 6: Verify inference works — conversion may succeed but inference
        # can fail if the QuantizedCPU kernel is missing (e.g. torch+cpu wheel).
        # Better to detect this now than fail mid-FL-round at the client.
        _verify_inference(work_model, calib_loader)

        log.info("Static INT8 conversion successful → quant_method=static_int8")
        return work_model, "static_int8"

    except Exception as int8_err:
        log.warning(
            f"{QUANT_UNSUPPORTED}: static INT8 failed ({type(int8_err).__name__}: {int8_err}). "
            f"Falling back to {resolved_lowp.upper()}. int8_disabled=true"
        )
        # lowp fallback — using lowp module apply() so it can be intercepted in tests
        try:
            lowp_model = lowp_apply(model, inplace=False)
            log.info(
                f"Fallback to {resolved_lowp.upper()} successful "
                f"→ quant_method={lowp_method}"
            )
            return lowp_model, lowp_method

        except Exception as lowp_err:
            log.warning(
                f"{QUANT_UNSUPPORTED}: {resolved_lowp.upper()} fallback also failed "
                f"({type(lowp_err).__name__}: {lowp_err}). "
                f"Falling back to FP32. quant_method=fp32_fallback"
            )
            fp32_model = copy.deepcopy(model).float().eval()
            log.info("Fallback to FP32 → quant_method=fp32_fallback")
            return fp32_model, "fp32_fallback"
