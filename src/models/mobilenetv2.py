"""
src/models/mobilenetv2.py — MobileNetV2 wrapper for CIFAR-10 FL experiments

Paper: Adaptive FL with PPO-based Client and Quantization Selection
Input: 224×224 (paper-aligned). Do NOT use 32×32 for real experiments.
Classes: 10 (CIFAR-10)

Design notes:
  - Randomly initialized (no ImageNet pretrained weights) for fair FL comparison.
  - Classifier head replaced to match num_classes=10.
  - Conv-BN-ReLU patterns are preserved for compatibility with PyTorch static INT8
    quantization (prepare → fuse → calibrate → convert) in Stage 4.
  - get_parameters() / set_parameters() helpers for Flower parameter exchange.
  - FP32 by default; FP16 via .half(); INT8 quantization handled by compression layer.
"""

from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

NUM_CLASSES = 10
# Approximate unquantized MobileNetV2 param count (for logging)
_EXPECTED_PARAMS_M = 3.5  # ~3.5 M parameters (num_classes=10 head)


def get_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Return a randomly-initialized MobileNetV2 for CIFAR-10.

    Args:
        num_classes: Output classes. Default 10 for CIFAR-10.

    Returns:
        nn.Module — MobileNetV2 with modified classifier head.

    Note:
        weights=None → no pretrained weights (paper-aligned: train from scratch in FL).
        classifier[1] replaced to match num_classes.
    """
    model = mobilenet_v2(weights=None)
    # Replace the final classifier to match num_classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Extract model parameters as a list of numpy arrays.
    Used by Flower client's get_parameters() method.

    Returns parameters in FP32 regardless of model dtype
    to ensure safe FedAvg aggregation on the server.
    """
    return [
        val.detach().cpu().float().numpy()
        for val in model.state_dict().values()
    ]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Load parameter arrays into model in-place.
    Used by Flower client's set_parameters() method.

    Args:
        model: Target model to update.
        parameters: List of numpy arrays matching model.state_dict() order.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    # Use strict=True so any shape mismatch raises immediately
    model.load_state_dict(state_dict, strict=True)


def clone_model(model: nn.Module) -> nn.Module:
    """Return a deep copy of the model (for safe per-client use)."""
    return copy.deepcopy(model)


def model_summary(model: nn.Module) -> dict:
    """Return a compact summary dict for logging."""
    n_params = count_parameters(model)
    dtype = next(model.parameters()).dtype
    return {
        "model": "MobileNetV2",
        "num_classes": NUM_CLASSES,
        "n_trainable_params": n_params,
        "n_params_M": round(n_params / 1e6, 3),
        "dtype": str(dtype),
        "input_resolution": "224x224",
    }
