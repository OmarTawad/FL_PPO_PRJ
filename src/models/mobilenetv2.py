"""
src/models/mobilenetv2.py — MobileNetV2 wrapper for CIFAR-10 FL experiments

Paper: Adaptive FL with PPO-based Client and Quantization Selection
Input: 224×224 (paper-aligned). Do NOT use 32×32 for real experiments.
Classes: 10 (CIFAR-10)

Design notes:
  - Randomly initialized (no ImageNet pretrained weights) for fair FL comparison.
  - Classifier head is explicitly 10-way and checked with an assertion.
  - AdaptiveAvgPool2d((1,1)) is explicitly present before the classifier.
  - Feature extractor can be frozen during initial training.
  - get_parameters() / set_parameters() helpers for Flower parameter exchange.
  - FP32 by default; FP16 via .half(); INT8 quantization handled by compression layer.
"""

from __future__ import annotations

import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

NUM_CLASSES = 10
# Approximate unquantized MobileNetV2 param count (for logging)
_EXPECTED_PARAMS_M = 3.5  # ~3.5 M parameters (num_classes=10 head)


class MobileNetV2CIFAR(nn.Module):
    """MobileNetV2 with explicit avgpool and CIFAR-10 classifier head."""

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_features: bool = True):
        super().__init__()
        if num_classes != 10:
            raise ValueError(
                f"MobileNetV2CIFAR supports only 10 classes, got {num_classes}"
            )
        model = mobilenet_v2(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 10)

        self.features = model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = model.classifier
        assert self.classifier[-1].out_features == 10, (
            f"MobileNetV2 head must output 10 classes, got "
            f"{self.classifier[-1].out_features}"
        )

        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model(num_classes: int = NUM_CLASSES, freeze_features: bool = True) -> nn.Module:
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
    model = MobileNetV2CIFAR(
        num_classes=num_classes,
        freeze_features=freeze_features,
    )
    if model.classifier[-1].out_features != 10:
        raise AssertionError(
            f"Expected MobileNetV2 classifier out_features=10, got "
            f"{model.classifier[-1].out_features}"
        )
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
