"""
src/compression/qat.py — QAT helpers for trainable INT8 simulation.

This module provides the train-time-compatible INT8 path based on fake-quant
observers (QAT), plus a diagnostic post-training conversion check.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

_QAT_BACKEND_PREFERENCE = ("x86", "fbgemm", "onednn", "qnnpack")


def resolve_qat_backend(backend: Optional[str]) -> str:
    """
    Resolve QAT backend name from requested value and runtime availability.

    backend='auto' picks the first available backend in this order:
    x86 -> fbgemm -> onednn -> qnnpack.
    """
    requested = "auto" if backend is None else str(backend).strip().lower()
    supported = [
        b for b in torch.backends.quantized.supported_engines
        if b != "none"
    ]
    if requested == "auto":
        for candidate in _QAT_BACKEND_PREFERENCE:
            if candidate in supported:
                return candidate
        raise RuntimeError(
            "No supported QAT backend found in "
            f"{list(torch.backends.quantized.supported_engines)}"
        )

    if requested not in supported:
        raise RuntimeError(
            f"Requested QAT backend '{requested}' is unavailable; "
            f"supported={supported}"
        )
    return requested


def _maybe_fuse_for_qat(model: nn.Module) -> nn.Module:
    """
    Best-effort MobileNetV2 fusion before QAT prepare.

    If fusion fails for any reason we keep the unfused model to preserve
    trainability and avoid blocking QAT path activation.
    """
    try:
        from src.compression.int8 import _fuse_mobilenetv2  # local import
        return _fuse_mobilenetv2(model)
    except Exception as err:
        log.debug(f"QAT fusion skipped: {type(err).__name__}: {err}")
        return model


def _apply_qat_scope(
    model: nn.Module,
    qconfig: torch.ao.quantization.QConfig,
    scope: str,
) -> None:
    """Apply QAT qconfig by scope without changing model topology."""
    scope_norm = str(scope).strip().lower()
    if scope_norm == "full":
        model.qconfig = qconfig
        return

    if scope_norm != "classifier_only":
        raise ValueError(f"Unsupported QAT scope: {scope}")

    # Narrow QAT to the classifier head only to reduce weak-client overhead.
    model.qconfig = None
    for module in model.modules():
        module.qconfig = None

    classifier = getattr(model, "classifier", None)
    if classifier is None:
        raise RuntimeError("QAT classifier_only scope requires model.classifier")
    # MobileNetV2 classifier[-1] is the trainable Linear head.
    classifier[-1].qconfig = qconfig


def prepare_qat_model(
    model: nn.Module,
    backend: str = "auto",
    inplace: bool = False,
    fuse_model: bool = True,
    scope: str = "full",
    prepare_if_needed: bool = False,
) -> Tuple[nn.Module, str]:
    """
    Prepare a model for QAT (fake quantization), preserving SGD trainability.
    """
    selected_backend = resolve_qat_backend(backend)
    work_model = model if inplace else copy.deepcopy(model)

    if prepare_if_needed and bool(getattr(work_model, "_qat_prepared", False)):
        scope_used = getattr(work_model, "_qat_scope_used", scope)
        setattr(work_model, "_qat_backend_used", selected_backend)
        setattr(work_model, "_qat_scope_used", str(scope_used))
        return work_model, selected_backend

    scope_norm = str(scope).strip().lower()
    if scope_norm == "classifier_only":
        fuse_model = False

    if fuse_model and not bool(getattr(work_model, "_qat_fused", False)):
        work_model = _maybe_fuse_for_qat(work_model)
        setattr(work_model, "_qat_fused", True)

    work_model.train()
    torch.backends.quantized.engine = selected_backend
    qat_qconfig = torch.ao.quantization.get_default_qat_qconfig(
        selected_backend
    )
    _apply_qat_scope(work_model, qat_qconfig, scope=scope_norm)
    torch.ao.quantization.prepare_qat(work_model, inplace=True)
    setattr(work_model, "_qat_backend_used", selected_backend)
    setattr(work_model, "_qat_scope_used", scope_norm)
    setattr(work_model, "_qat_prepared", True)
    return work_model, selected_backend


def get_qat_backend_used(model: nn.Module, fallback: str = "") -> str:
    """Return recorded backend from a QAT-prepared model when available."""
    value = getattr(model, "_qat_backend_used", fallback)
    return str(value) if value is not None else ""


def get_qat_scope_used(model: nn.Module, fallback: str = "full") -> str:
    """Return recorded QAT scope used on a prepared model."""
    value = getattr(model, "_qat_scope_used", fallback)
    return str(value) if value is not None else "full"


def convert_qat_model_for_check(
    model: nn.Module,
    backend: str,
    eval_loader: Optional[DataLoader],
    n_batches: int = 1,
    device: torch.device = torch.device("cpu"),
) -> Tuple[bool, str]:
    """
    Diagnostic-only conversion check: QAT model -> static INT8 + forward pass.

    Returns:
      (success, error_string). error_string is empty when success=True.
    """
    if eval_loader is None:
        return False, "missing_eval_loader"
    if n_batches < 1:
        return False, "invalid_n_batches"

    try:
        selected_backend = resolve_qat_backend(backend)
    except Exception as err:
        return False, f"backend_resolve_failed:{type(err).__name__}:{err}"

    try:
        work_model = copy.deepcopy(model).eval()
        torch.backends.quantized.engine = selected_backend
        quantized_model = torch.ao.quantization.convert(work_model, inplace=False)

        seen = 0
        with torch.no_grad():
            for images, _ in eval_loader:
                images = images.to(device=device, dtype=torch.float32)
                quantized_model(images)
                seen += 1
                if seen >= n_batches:
                    break
        if seen < 1:
            return False, "empty_eval_loader"
        return True, ""
    except Exception as err:
        return False, f"{type(err).__name__}:{str(err)[:500]}"
