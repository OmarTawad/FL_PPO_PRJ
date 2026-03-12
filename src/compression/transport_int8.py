"""
src/compression/transport_int8.py — INT8 transport codec for model deltas.

This module provides deterministic symmetric per-tensor INT8 quantization for
client->server delta transport, plus JSON metadata helpers.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


_SCHEME = "symm_per_tensor_v1"


def quantize_delta_int8_per_tensor(
    delta_ndarrays: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Quantize FP32 deltas to INT8 with per-tensor symmetric scaling.

    Returns:
      (int8_arrays, metadata_dict)
    """
    q_arrays: List[np.ndarray] = []
    scales: List[float] = []
    shapes: List[List[int]] = []

    for arr in delta_ndarrays:
        fp = np.asarray(arr, dtype=np.float32)
        max_abs = float(np.max(np.abs(fp))) if fp.size > 0 else 0.0
        scale = max(max_abs / 127.0, 1e-12)
        q = np.clip(np.round(fp / scale), -127, 127).astype(np.int8, copy=False)
        q_arrays.append(q)
        scales.append(scale)
        shapes.append(list(fp.shape))

    meta: Dict[str, Any] = {
        "scheme": _SCHEME,
        "tensor_count": len(q_arrays),
        "scales": scales,
        "zero_points": [0] * len(q_arrays),
        "shapes": shapes,
    }
    return q_arrays, meta


def dequantize_delta_int8_per_tensor(
    int8_ndarrays: Sequence[np.ndarray],
    meta: Dict[str, Any],
) -> List[np.ndarray]:
    """
    Dequantize INT8 deltas to FP32 using per-tensor metadata.
    """
    scheme = str(meta.get("scheme", ""))
    if scheme != _SCHEME:
        raise ValueError(f"unsupported_transport_scheme:{scheme}")

    scales = meta.get("scales")
    shapes = meta.get("shapes")
    tensor_count = int(meta.get("tensor_count", -1))
    if not isinstance(scales, list) or not isinstance(shapes, list):
        raise ValueError("invalid_transport_meta:missing_scales_or_shapes")
    if tensor_count != len(int8_ndarrays) or len(scales) != len(int8_ndarrays):
        raise ValueError("invalid_transport_meta:tensor_count_mismatch")

    out: List[np.ndarray] = []
    for idx, arr in enumerate(int8_ndarrays):
        q = np.asarray(arr)
        if q.dtype != np.int8:
            raise ValueError(f"invalid_transport_dtype:tensor_{idx}:{q.dtype}")
        target_shape = tuple(int(x) for x in shapes[idx])
        if q.shape != target_shape:
            raise ValueError(
                f"invalid_transport_shape:tensor_{idx}:{q.shape}!={target_shape}"
            )
        scale = float(scales[idx])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid_transport_scale:tensor_{idx}:{scale}")
        out.append((q.astype(np.float32) * scale).astype(np.float32, copy=False))
    return out


def encode_transport_meta_json(meta: Dict[str, Any]) -> str:
    """Serialize transport metadata as compact JSON."""
    return json.dumps(meta, separators=(",", ":"), ensure_ascii=True)


def decode_transport_meta_json(meta_json: str) -> Dict[str, Any]:
    """Deserialize transport metadata JSON."""
    if not meta_json:
        raise ValueError("missing_transport_meta_json")
    parsed = json.loads(meta_json)
    if not isinstance(parsed, dict):
        raise ValueError("invalid_transport_meta_json:not_object")
    return parsed

