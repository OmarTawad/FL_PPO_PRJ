"""
src/compression/fp16.py — FP16 (half-precision) model conversion

Converts model weights to float16.

Server→client use: the server sends a FP16 copy of the global model to
selected clients whose PPO action assigned bits=16. Clients train locally
in FP16, then return FP32 parameters (get_parameters() always upcasts).

Design notes:
  - CrossEntropyLoss requires float32 logits → trainer.py upcasts automatically.
  - BatchNorm running stats stay float32 even after .half() on CPU
    (PyTorch behavior); this is correct.
  - On CPU, FP16 arithmetic falls back to float32 for most ops via PyTorch's
    autocast simulation. Memory savings still apply for model weights storage.
"""
import copy
import torch.nn as nn


def apply(model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Return a FP16 (half-precision) copy of the model.

    Args:
        model: Source model (any dtype).
        inplace: If False (default), deep-copy before converting.

    Returns:
        Model with all parameters and buffers in float16.
    """
    if not inplace:
        model = copy.deepcopy(model)
    model = model.half()
    return model
