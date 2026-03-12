"""
src/compression/quantizer.py — Unified quantization interface

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Single entry point for all quantization modes. Called by:
  - FL client (Stage 5) when applying server-assigned quant to local model
  - Experiments (Stage 9) for fixed-quant baseline runs

Usage:
    from src.compression.quantizer import quantize
    model, method = quantize(model, bits=8, calib_loader=loader)
    # method: "fp32" | "fp16" | "bf16" | "static_int8" |
    #         "fp16_fallback" | "bf16_fallback" | "fp32_fallback"

Return value contract:
    model  — a COPY of the input model at the requested precision
    method — string key for per-round JSON logging ("quant_method" field)
    The caller's original model is NEVER mutated.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch.nn as nn
from torch.utils.data import DataLoader

from src.compression import fp32, fp16, bf16
from src.compression.int8 import try_static_int8
from src.compression.qat import prepare_qat_model
from src.compression.lowp import (
    resolve_lowp_dtype,
)

log = logging.getLogger(__name__)

# Valid quantization bit widths
VALID_BITS = (32, 16, 8)

# Method strings written to per-round JSON
METHOD_FP32           = "fp32"
METHOD_FP16           = "fp16"
METHOD_BF16           = "bf16"
METHOD_STATIC_INT8    = "static_int8"
METHOD_QAT_INT8_TRAIN = "qat_int8_train"
METHOD_FP16_FALLBACK  = "fp16_fallback"
METHOD_BF16_FALLBACK  = "bf16_fallback"
METHOD_FP32_FALLBACK  = "fp32_fallback"


def quantize(
    model: nn.Module,
    bits: int,
    calib_loader: Optional[DataLoader] = None,
    calibration_samples: int = 128,
    backend: str = "auto",
    lowp_dtype: str = "bf16",
    int8_impl: str = "static",
    qat_scope: str = "full",
    qat_inplace: bool = False,
) -> Tuple[nn.Module, str]:
    """
    Apply quantization to a model and return (quantized_model, method_string).

    Args:
        model: Source model (must be FP32, eval mode preferred but not required).
        bits: Target bit width — must be 32, 16, or 8.
        calib_loader: Required when bits=8 (INT8 calibration). Ignored for 32/16.
        calibration_samples: Number of calibration images (default 128).
        backend: Quantization backend ("auto" | "fbgemm" | "x86" | "onednn" | "qnnpack").
        int8_impl: "static" (post-training static conversion path) or
            "qat" (train-time fake-quant path).

    Returns:
        (quantized_model, method) — model is a deep copy; method is one of
        {fp32, fp16, bf16, static_int8, qat_int8_train,
         fp16_fallback, bf16_fallback, fp32_fallback}.

    Raises:
        ValueError: If bits not in {32, 16, 8}.
        ValueError: If bits=8 but calib_loader is None.
    """
    if bits not in VALID_BITS:
        raise ValueError(f"bits must be one of {VALID_BITS}, got: {bits}")

    if bits == 32:
        return fp32.apply(model, inplace=False), METHOD_FP32

    if bits == 16:
        resolved = resolve_lowp_dtype(lowp_dtype)
        if resolved == "bf16":
            return bf16.apply(model, inplace=False), METHOD_BF16
        return fp16.apply(model, inplace=False), METHOD_FP16

    # bits == 8: static INT8 or QAT INT8 train path
    int8_impl_norm = str(int8_impl).strip().lower()
    if int8_impl_norm == "qat":
        qat_model, _ = prepare_qat_model(
            model,
            backend=backend,
            inplace=qat_inplace,
            fuse_model=True,
            scope=qat_scope,
            prepare_if_needed=qat_inplace,
        )
        return qat_model, METHOD_QAT_INT8_TRAIN

    if int8_impl_norm != "static":
        raise ValueError(
            f"int8_impl must be 'static' or 'qat', got: {int8_impl}"
        )

    if calib_loader is None:
        raise ValueError(
            "calib_loader is required for bits=8 when int8_impl=static. "
            "Pass a DataLoader with ≥128 samples."
        )
    resolved_backend = None if str(backend).strip().lower() == "auto" else str(backend).strip().lower()
    return try_static_int8(
        model,
        calib_loader=calib_loader,
        calibration_samples=calibration_samples,
        backend=resolved_backend,
        inplace=False,
        lowp_dtype=lowp_dtype,
    )


def quantize_action(
    model: nn.Module,
    action: int,
    calib_loader: Optional[DataLoader] = None,
    calibration_samples: int = 128,
    backend: str = "auto",
    lowp_dtype: str = "bf16",
    int8_impl: str = "static",
    qat_scope: str = "full",
    qat_inplace: bool = False,
) -> Tuple[nn.Module, str]:
    """
    Convenience wrapper: convert PPO action integer to bits and call quantize().

    PPO action encoding (SPEC.md §3.2):
        0 → skip (caller should not call quantize; raises if called)
        1 → fp32 (32 bits)
        2 → lowp 16-bit (dtype from lowp_dtype)
        3 → int8  (8 bits — requires calib_loader)

    Args:
        model: Source model.
        action: Integer in {0, 1, 2, 3}.
        calib_loader: Required when action=3.
        calibration_samples: Calibration sample count.

    Returns:
        (quantized_model, method_string)
    """
    _ACTION_TO_BITS = {1: 32, 2: 16, 3: 8}
    if action == 0:
        raise ValueError(
            "action=0 means 'skip' — client should not be assigned a model. "
            "Do not call quantize_action for skipped clients."
        )
    if action not in _ACTION_TO_BITS:
        raise ValueError(f"action must be 0/1/2/3, got: {action}")
    bits = _ACTION_TO_BITS[action]
    return quantize(model, bits=bits, calib_loader=calib_loader,
                    calibration_samples=calibration_samples,
                    backend=backend,
                    lowp_dtype=lowp_dtype,
                    int8_impl=int8_impl,
                    qat_scope=qat_scope,
                    qat_inplace=qat_inplace)
