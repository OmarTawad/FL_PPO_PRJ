"""
src/compression/fp32.py — FP32 passthrough (no-op quantization)

Returns the model unchanged. Used as the baseline / "no compression" path.
"""
import copy
import torch.nn as nn


def apply(model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Return the model in FP32, unchanged.

    Args:
        model: Source model.
        inplace: If False (default), return a deep copy so the caller's
                 model is not aliased. Set True only when the caller
                 explicitly owns the model and doesn't need it again.
    Returns:
        model in float32, train-mode preserved.
    """
    if not inplace:
        model = copy.deepcopy(model)
    # Ensure FP32 (noop if already float32; no-op guard)
    model = model.float()
    return model
