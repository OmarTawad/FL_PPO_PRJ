"""
src/models/trainer.py — Train and evaluate MobileNetV2 for FL clients

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Functions:
  train_one_epoch(model, loader, optimizer, device) → TrainResult
  evaluate(model, loader, device) → EvalResult
  train_local(model, loader, optimizer, epochs, device) → List[TrainResult]

Design notes:
  - All train/eval results are plain dataclasses (JSON-serializable via .to_dict()).
  - FP16 model: CrossEntropyLoss uses FP32 upcast (inputs auto-cast to FP32 for CE).
  - OOM detection: catches RuntimeError containing "out of memory"; returns dropout signal.
  - num_workers=0 and pin_memory=False throughout (passed in via DataLoader).
  - Gradient clipping (max_norm=1.0) for training stability.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ─── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class TrainResult:
    """Result of one local training epoch."""
    epoch: int
    loss: float
    n_samples: int
    train_time_s: float
    oom: bool = False            # True if OOM was caught mid-epoch

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalResult:
    """Result of one evaluation pass."""
    accuracy: float              # top-1 accuracy in [0, 1]
    loss: float
    n_samples: int
    eval_time_s: float

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Core training function ───────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    grad_clip_norm: float = 1.0,
) -> TrainResult:
    """
    Run one epoch of local training.

    Handles FP16 models by keeping CrossEntropyLoss in FP32:
    inputs are cast to float32 before loss computation (CE requires float).

    Args:
        model: Model to train (may be FP16 or FP32).
        loader: DataLoader providing (image, label) batches.
        optimizer: Optimizer instance (already constructed for this model).
        device: Target computation device (CPU for this project).
        epoch: Epoch index for logging.
        grad_clip_norm: Max gradient norm for clipping (0.0 = disabled).

    Returns:
        TrainResult with per-epoch metrics.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_samples = 0
    t0 = time.perf_counter()

    try:
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass — model may be FP16
            outputs = model(images)

            # CE loss requires float32; upcast logits if model is FP16
            if outputs.dtype != torch.float32:
                outputs = outputs.float()

            loss = criterion(outputs, labels)
            loss.backward()

            if grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            total_loss += loss.item() * len(labels)
            n_samples += len(labels)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # OOM: free memory and signal dropout
            _try_free_memory()
            return TrainResult(
                epoch=epoch,
                loss=float("nan"),
                n_samples=n_samples,
                train_time_s=time.perf_counter() - t0,
                oom=True,
            )
        raise  # Non-OOM RuntimeError: re-raise

    avg_loss = total_loss / max(n_samples, 1)
    return TrainResult(
        epoch=epoch,
        loss=avg_loss,
        n_samples=n_samples,
        train_time_s=time.perf_counter() - t0,
        oom=False,
    )


# ─── Evaluation function ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EvalResult:
    """
    Evaluate model top-1 accuracy and average loss.

    Args:
        model: Model to evaluate (FP32 or FP16; logits upcasted for CE).
        loader: DataLoader (typically server test loader).
        device: CPU for this project.

    Returns:
        EvalResult with accuracy, loss, sample count, and timing.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    n_samples = 0
    t0 = time.perf_counter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        if outputs.dtype != torch.float32:
            outputs = outputs.float()

        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * len(labels)
        correct += preds.eq(labels).sum().item()
        n_samples += len(labels)

    accuracy = correct / max(n_samples, 1)
    avg_loss = total_loss / max(n_samples, 1)

    return EvalResult(
        accuracy=accuracy,
        loss=avg_loss,
        n_samples=n_samples,
        eval_time_s=time.perf_counter() - t0,
    )


# ─── Multi-epoch local training ───────────────────────────────────────────────

def train_local(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    grad_clip_norm: float = 1.0,
) -> List[TrainResult]:
    """
    Run multiple local epochs (called by Flower client's fit()).

    Stops early on OOM to prevent repeated failure.

    Args:
        model: Model to train.
        loader: Client DataLoader.
        optimizer: Optimizer.
        epochs: Number of local epochs (paper default: 2).
        device: Compute device.
        grad_clip_norm: Gradient clipping norm.

    Returns:
        List of TrainResult, one per epoch. On OOM, list has 1 element with oom=True.
    """
    results = []
    for epoch in range(1, epochs + 1):
        result = train_one_epoch(
            model, loader, optimizer, device,
            epoch=epoch, grad_clip_norm=grad_clip_norm,
        )
        results.append(result)
        if result.oom:
            break  # Don't retry after OOM
    return results


# ─── Optimizer factory ────────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    optimizer_name: str = "sgd",
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """
    Build an optimizer for local training.

    Args:
        model: Model whose parameters to optimize.
        optimizer_name: "sgd" or "adam".
        lr: Learning rate.
        momentum: Momentum (SGD only).
        weight_decay: L2 regularization.

    Returns:
        Configured optimizer.
    """
    if optimizer_name.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer_name.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Use 'sgd' or 'adam'.")


# ─── Memory helpers ───────────────────────────────────────────────────────────

def _try_free_memory() -> None:
    """Attempt to free Python/torch memory after OOM."""
    gc.collect()
    # No CUDA cache on CPU-only build; noop for safety
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model_memory_mb(model: nn.Module) -> float:
    """
    Estimate memory footprint of model parameters in MB.
    Useful for pre-flight checks before assigning to a resource-limited client.
    """
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)
