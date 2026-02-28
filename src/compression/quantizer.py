"""
src/compression/quantizer.py — Unified quantization interface

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Single entry point for all quantization modes. Called by:
  - FL client (Stage 5) when applying server-assigned quant to local model
  - Experiments (Stage 9) for fixed-quant baseline runs

Usage:
    from src.compression.quantizer import quantize
    model, method = quantize(model, bits=8, calib_loader=loader)
    # method: "fp32" | "fp16" | "static_int8" | "fp16_fallback" | "fp32_fallback"

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

from src.compression import fp32, fp16
from src.compression.int8 import try_static_int8, QUANT_UNSUPPORTED

log = logging.getLogger(__name__)

# Valid quantization bit widths
VALID_BITS = (32, 16, 8)

# Method strings written to per-round JSON
METHOD_FP32           = "fp32"
METHOD_FP16           = "fp16"
METHOD_STATIC_INT8    = "static_int8"
METHOD_FP16_FALLBACK  = "fp16_fallback"
METHOD_FP32_FALLBACK  = "fp32_fallback"


def quantize(
    model: nn.Module,
    bits: int,
    calib_loader: Optional[DataLoader] = None,
    calibration_samples: int = 128,
    backend: str = "qnnpack",
) -> Tuple[nn.Module, str]:
    """
    Apply quantization to a model and return (quantized_model, method_string).

    Args:
        model: Source model (must be FP32, eval mode preferred but not required).
        bits: Target bit width — must be 32, 16, or 8.
        calib_loader: Required when bits=8 (INT8 calibration). Ignored for 32/16.
        calibration_samples: Number of calibration images (default 128).
        backend: Quantization backend ("qnnpack" for CPU).

    Returns:
        (quantized_model, method) — model is a deep copy; method is one of
        {fp32, fp16, static_int8, fp16_fallback, fp32_fallback}.

    Raises:
        ValueError: If bits not in {32, 16, 8}.
        ValueError: If bits=8 but calib_loader is None.
    """
    if bits not in VALID_BITS:
        raise ValueError(f"bits must be one of {VALID_BITS}, got: {bits}")

    if bits == 32:
        return fp32.apply(model, inplace=False), METHOD_FP32

    if bits == 16:
        return fp16.apply(model, inplace=False), METHOD_FP16

    # bits == 8: attempt static INT8
    if calib_loader is None:
        raise ValueError(
            "calib_loader is required for bits=8 (static INT8 calibration). "
            "Pass a DataLoader with ≥128 samples."
        )
    return try_static_int8(
        model,
        calib_loader=calib_loader,
        calibration_samples=calibration_samples,
        backend=backend,
        inplace=False,
    )


def quantize_action(
    model: nn.Module,
    action: int,
    calib_loader: Optional[DataLoader] = None,
    calibration_samples: int = 128,
) -> Tuple[nn.Module, str]:
    """
    Convenience wrapper: convert PPO action integer to bits and call quantize().

    PPO action encoding (SPEC.md §3.2):
        0 → skip (caller should not call quantize; raises if called)
        1 → fp32 (32 bits)
        2 → fp16 (16 bits)
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
                    calibration_samples=calibration_samples)
