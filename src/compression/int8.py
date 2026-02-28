"""
src/compression/int8.py — Static INT8 quantization for MobileNetV2

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Pipeline: fuse → prepare (insert observers) → calibrate (128 samples) → convert

Fallback chain (NEVER silent):
    static INT8 fails  →  log QUANT_UNSUPPORTED  →  return FP16 model
    FP16 also fails    →  log QUANT_UNSUPPORTED  →  return FP32 model

Dynamic INT8 (quantize_dynamic) is NEVER used as a fallback (SPEC.md §4.2).

Backend: qnnpack (recommended for ARM/CPU; also works on x86 in PyTorch).

Usage:
    from src.compression.int8 import try_static_int8, QUANT_UNSUPPORTED
    model, method = try_static_int8(model, calib_loader)
    # method in {"static_int8", "fp16_fallback", "fp32_fallback"}
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Log key emitted on every fallback (SPEC.md §4.2)
QUANT_UNSUPPORTED = "QUANT_UNSUPPORTED"

log = logging.getLogger(__name__)

# MobileNetV2 ConvBNActivation triplets to fuse.
# format: list of [parent_attr_path, [conv, bn, act]] per InvertedResidual block.
# We discover them dynamically instead of hardcoding to handle model variants.


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


def try_static_int8(
    model: nn.Module,
    calib_loader: DataLoader,
    calibration_samples: int = 128,
    backend: str = "qnnpack",
    inplace: bool = False,
) -> Tuple[nn.Module, str]:
    """
    Attempt static INT8 quantization with full fallback chain.

    Steps:
        1. Deep-copy model (always — original is never mutated)
        2. fuse Conv-BN-ReLU6
        3. set qconfig  (get_default_qconfig(backend))
        4. prepare  (inserts MinMax/Histogram observers)
        5. calibrate  (forward pass over n_samples images)
        6. convert  (replaces float ops with quantized ops)

    On any failure in steps 2–6:
        → logs QUANT_UNSUPPORTED with error detail
        → falls back to FP16, then FP32 if FP16 also fails

    Args:
        model: Source FP32 model (eval or train mode).
        calib_loader: DataLoader for calibration (128 images sufficient).
        calibration_samples: Number of images to run through observers.
        backend: "qnnpack" (default, CPU) or "fbgemm" (x86 server).
        inplace: Ignored — we always deep-copy for safety.

    Returns:
        Tuple (quantized_model, method_string) where method_string is one of:
            "static_int8"    — full static INT8 conversion succeeded
            "fp16_fallback"  — INT8 failed; returned FP16 model
            "fp32_fallback"  — INT8 + FP16 failed; returned FP32 model
    """
    # Always work on a deep copy — never mutate the caller's model
    work_model = copy.deepcopy(model).float().eval()

    try:
        # Step 1: Fuse
        work_model = _fuse_mobilenetv2(work_model)

        # Step 2: Set qconfig
        torch.backends.quantized.engine = backend
        work_model.qconfig = torch.quantization.get_default_qconfig(backend)

        # Step 3: Prepare (insert observers)
        torch.quantization.prepare(work_model, inplace=True)

        # Step 4: Calibrate
        seen = _calibrate(work_model, calib_loader, n_samples=calibration_samples)
        log.info(f"INT8 calibration: {seen} samples processed (backend={backend})")

        # Step 5: Convert
        torch.quantization.convert(work_model, inplace=True)

        log.info("Static INT8 conversion successful → quant_method=static_int8")
        return work_model, "static_int8"

    except Exception as int8_err:
        log.warning(
            f"{QUANT_UNSUPPORTED}: static INT8 failed ({type(int8_err).__name__}: {int8_err}). "
            f"Falling back to FP16. int8_disabled=true"
        )
        # FP16 fallback
        try:
            fp16_model = copy.deepcopy(model).half().eval()
            log.info("Fallback to FP16 successful → quant_method=fp16_fallback")
            return fp16_model, "fp16_fallback"

        except Exception as fp16_err:
            log.warning(
                f"{QUANT_UNSUPPORTED}: FP16 fallback also failed ({type(fp16_err).__name__}: {fp16_err}). "
                f"Falling back to FP32. quant_method=fp32_fallback"
            )
            fp32_model = copy.deepcopy(model).float().eval()
            log.info("Fallback to FP32 → quant_method=fp32_fallback")
            return fp32_model, "fp32_fallback"
