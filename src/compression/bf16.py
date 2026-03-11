"""
src/compression/bf16.py — BF16 model conversion.

Converts model weights/buffers to bfloat16.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn


def apply(model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Return a BF16 (bfloat16) copy of the model.

    Args:
        model: Source model (any dtype).
        inplace: If False (default), deep-copy before converting.

    Returns:
        Model with parameters and buffers in bfloat16.
    """
    if not inplace:
        model = copy.deepcopy(model)
    return model.to(dtype=torch.bfloat16)
