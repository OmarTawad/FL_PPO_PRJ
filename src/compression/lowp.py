"""
src/compression/lowp.py — Low-precision dtype helpers.

Centralizes 16-bit precision selection so bits=16 always means:
    use the configured low-precision dtype (bf16 or fp16).
"""

from __future__ import annotations

from typing import Optional

VALID_LOWP_DTYPES = ("bf16", "fp16")


def resolve_lowp_dtype(lowp_dtype: Optional[str]) -> str:
    """
    Resolve and validate the configured low-precision dtype.

    Defaults to bf16 when unset.
    """
    if lowp_dtype is None:
        return "bf16"
    value = str(lowp_dtype).strip().lower()
    if value not in VALID_LOWP_DTYPES:
        raise ValueError(
            f"lowp_dtype must be one of {VALID_LOWP_DTYPES}, got: {lowp_dtype}"
        )
    return value


def method_for_lowp_dtype(lowp_dtype: Optional[str]) -> str:
    """Return quant method string for 16-bit low precision."""
    return resolve_lowp_dtype(lowp_dtype)


def fallback_method_for_lowp_dtype(lowp_dtype: Optional[str]) -> str:
    """Return fallback quant method string for 16-bit low precision."""
    return f"{resolve_lowp_dtype(lowp_dtype)}_fallback"


def precision_from_quant_method(method: str) -> str:
    """
    Map quant method string to canonical precision label.

    Returned labels: fp32 | fp16 | bf16 | int8 | unknown
    """
    value = str(method).lower()
    if value in ("fp32", "fp32_fallback"):
        return "fp32"
    if value in ("fp16", "fp16_fallback"):
        return "fp16"
    if value in ("bf16", "bf16_fallback"):
        return "bf16"
    if value in ("static_int8", "qat_int8_train"):
        return "int8"
    return "unknown"
