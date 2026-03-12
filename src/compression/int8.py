"""
src/compression/int8.py — Static INT8 quantization for MobileNetV2

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Pipeline: fuse → prepare (insert observers) → calibrate (128 samples) → convert → verify_inference

Fallback chain (NEVER silent):
    static INT8 fails  →  log QUANT_UNSUPPORTED  →  return lowp_dtype model
    lowp also fails    →  log QUANT_UNSUPPORTED  →  return FP32 model

Dynamic INT8 (quantize_dynamic) is NEVER used as a fallback (SPEC.md §4.2).

Backend auto-selection (SPEC.md §4.2):
    try_static_int8() probes torch.backends.quantized.supported_engines at runtime
    and picks the first working backend in order: fbgemm → x86 → onednn → qnnpack.
    Pass backend='qnnpack' (or any other) to override.

Usage:
    from src.compression.int8 import try_static_int8, QUANT_UNSUPPORTED
    model, method = try_static_int8(model, calib_loader)
    # method in {"static_int8", "bf16_fallback", "fp16_fallback", "fp32_fallback"}
"""

from __future__ import annotations

import copy
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from src.compression import fp16 as _fp16_module
from src.compression import bf16 as _bf16_module
from src.compression.lowp import (
    fallback_method_for_lowp_dtype,
    resolve_lowp_dtype,
)

# Log key emitted on every fallback (SPEC.md §4.2)
QUANT_UNSUPPORTED = "QUANT_UNSUPPORTED"

# Backend preference order — outer try goes through these in order
_BACKEND_PREFERENCE = ["fbgemm", "x86", "onednn", "qnnpack"]

log = logging.getLogger(__name__)


class _QuantIOWrapper(nn.Module):
    """
    Wrap a float model with quant/dequant stubs for eager static quantization.

    This keeps the inner model unchanged while ensuring converted quantized ops
    receive quantized inputs at runtime.
    """

    def __init__(self, core: nn.Module):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.core = core
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.core(x)
        x = self.dequant(x)
        return x


def _build_static_int8_candidate(model: nn.Module) -> nn.Module:
    """
    Build a quantization-ready candidate model:
      QuantStub -> fused MobileNetV2 core -> DeQuantStub.
    """
    core_model = copy.deepcopy(model).float().eval()
    core_model = _fuse_mobilenetv2(core_model)
    return _QuantIOWrapper(core_model).float().eval()


def _example_inputs_from_loader(calib_loader: DataLoader) -> Tuple[torch.Tensor]:
    """Return one float32 example input tensor tuple for FX prepare."""
    for images, _ in calib_loader:
        if images.dtype != torch.float32:
            images = images.float()
        return (images[:1],)
    raise RuntimeError("empty_calibration_loader")


def _convert_static_int8_eager(
    model: nn.Module,
    calib_loader: DataLoader,
    calibration_samples: int,
    backend: str,
) -> nn.Module:
    """Eager-mode static INT8 conversion with quantized IO wrapper."""
    work_model = _build_static_int8_candidate(model)
    torch.backends.quantized.engine = backend
    work_model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(work_model, inplace=True)
    seen = _calibrate(work_model, calib_loader, n_samples=calibration_samples)
    if seen < 1:
        raise RuntimeError("empty_calibration_loader")
    torch.quantization.convert(work_model, inplace=True)
    _verify_inference(work_model, calib_loader)
    return work_model


def _convert_static_int8_fx(
    model: nn.Module,
    calib_loader: DataLoader,
    calibration_samples: int,
    backend: str,
) -> nn.Module:
    """FX-graph static INT8 conversion fallback for non-quantizable eager paths."""
    work_model = copy.deepcopy(model).float().eval()
    torch.backends.quantized.engine = backend
    qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping(backend)
    example_inputs = _example_inputs_from_loader(calib_loader)
    prepared = prepare_fx(work_model, qconfig_mapping, example_inputs)
    seen = _calibrate(prepared, calib_loader, n_samples=calibration_samples)
    if seen < 1:
        raise RuntimeError("empty_calibration_loader")
    quantized = convert_fx(prepared)
    _verify_inference(quantized, calib_loader)
    return quantized


def _convert_static_int8_with_backend(
    model: nn.Module,
    calib_loader: DataLoader,
    calibration_samples: int,
    backend: str,
) -> Tuple[nn.Module, str]:
    """
    Try eager static INT8 first, then FX static INT8 for compatibility.

    Returns:
      (quantized_model, path_label) where path_label in {"eager", "fx"}.
    Raises:
      RuntimeError with combined eager/fx error summary on total failure.
    """
    eager_err: Optional[Exception] = None
    try:
        return _convert_static_int8_eager(
            model, calib_loader, calibration_samples, backend
        ), "eager"
    except Exception as err:  # noqa: PERF203 - retain explicit exception for diagnostics
        eager_err = err
        log.debug(
            "Eager static INT8 path failed (backend=%s): %s: %s",
            backend,
            type(err).__name__,
            str(err)[:180],
        )

    try:
        return _convert_static_int8_fx(
            model, calib_loader, calibration_samples, backend
        ), "fx"
    except Exception as fx_err:
        raise RuntimeError(
            "static_int8_conversion_failed "
            f"(eager={type(eager_err).__name__}:{str(eager_err)[:180]}; "
            f"fx={type(fx_err).__name__}:{str(fx_err)[:180]})"
        ) from fx_err


# ─── Backend helpers ──────────────────────────────────────────────────────────

def _log_supported_engines() -> List[str]:
    """Log available quantized engines and return the list."""
    engines = list(torch.backends.quantized.supported_engines)
    log.info(f"torch.backends.quantized.supported_engines = {engines}")
    log.info(f"default engine = {torch.backends.quantized.engine}")
    return engines


def _probe_backend_works(model: nn.Module, calib_loader: DataLoader, backend: str) -> bool:
    """
    Quick 1-batch probe: fuse → prepare → calibrate → convert → infer.
    Returns True if the backend runs INT8 end-to-end. Does NOT mutate model.
    """
    try:
        _convert_static_int8_with_backend(
            model,
            calib_loader=calib_loader,
            calibration_samples=1,
            backend=backend,
        )
        return True
    except Exception as e:
        log.debug(f"  backend='{backend}' probe: {type(e).__name__}: {str(e)[:80]}")
        return False


def auto_select_backend(model: nn.Module, calib_loader: DataLoader) -> Optional[str]:
    """
    Probe supported_engines in preference order (fbgemm→x86→onednn→qnnpack).
    Returns the first working backend name, or None if none work.
    'none' is skipped — not a real quantization backend.
    """
    available = set(torch.backends.quantized.supported_engines) - {"none"}
    log.info(f"Auto-selecting INT8 backend from: {_BACKEND_PREFERENCE}")
    for backend in _BACKEND_PREFERENCE:
        if backend not in available:
            continue
        log.info(f"  Probing backend='{backend}'...")
        if _probe_backend_works(model, calib_loader, backend):
            log.info(f"  Backend='{backend}' succeeded ✓")
            return backend
        log.info(f"  Backend='{backend}' failed ✗")
    log.warning(f"{QUANT_UNSUPPORTED}: all backends failed: {sorted(available)}")
    return None


# ─── MobileNetV2-specific fusion ──────────────────────────────────────────────

def _fuse_mobilenetv2(model: nn.Module) -> nn.Module:
    """
    Fuse Conv-BN-ReLU6 patterns in MobileNetV2 for quantization.

    PyTorch's eager-mode quantization requires fused modules.
    MobileNetV2 has two patterns:
      1. features[0]: ConvBNActivation  → [0, 1, 2] (Conv2d, BN, ReLU6)
      2. features[i].conv: nested ConvBNActivation blocks

    We use a safe recursive fusion that skips any block where fusion fails.
    """
    model.eval()  # BN must be in eval mode before fusion

    # features[0] is the initial ConvBNActivation (stride=2 block)
    try:
        torch.quantization.fuse_modules(
            model.features[0], ["0", "1", "2"], inplace=True
        )
    except Exception as e:
        log.debug(f"features[0] fusion skipped: {e}")

    # features[1..N-1]: InvertedResidual blocks
    for i in range(1, len(model.features) - 1):
        block = model.features[i]
        if not hasattr(block, "conv"):
            continue
        sub = block.conv
        # Each InvertedResidual.conv is a Sequential with sub-blocks
        # Typical layout for expansion_ratio > 1: [0]=expand ConvBN, [1]=dw ConvBN, [2]=project ConvBN
        # For expansion_ratio == 1: [0]=dw ConvBN, [1]=project ConvBN
        n = len(sub)
        for j in range(n):
            child = sub[j]
            # ConvBNActivation has children [0]=Conv2d, [1]=BN, [2]=ReLU6
            children = list(child.children())
            if len(children) >= 3:
                try:
                    torch.quantization.fuse_modules(
                        child, ["0", "1", "2"], inplace=True
                    )
                except Exception:
                    pass
            elif len(children) == 2:
                try:
                    torch.quantization.fuse_modules(
                        child, ["0", "1"], inplace=True
                    )
                except Exception:
                    pass

    # features[-1]: last ConvBNActivation (1x1 expansion)
    try:
        torch.quantization.fuse_modules(
            model.features[-1], ["0", "1", "2"], inplace=True
        )
    except Exception as e:
        log.debug(f"features[-1] fusion skipped: {e}")

    return model


def _calibrate(
    model: nn.Module,
    calib_loader: DataLoader,
    n_samples: int = 128,
) -> int:
    """
    Run calibration forward passes to collect observer statistics.

    Args:
        model: Prepared model (observers already attached).
        calib_loader: DataLoader providing (image, label) batches.
        n_samples: Stop after this many samples.

    Returns:
        Actual number of samples seen.
    """
    model.eval()
    seen = 0
    with torch.no_grad():
        for images, _ in calib_loader:
            # Always feed FP32 to the prepared model
            if images.dtype != torch.float32:
                images = images.float()
            model(images)
            seen += len(images)
            if seen >= n_samples:
                break
    return seen


def _verify_inference(model: nn.Module, calib_loader: DataLoader) -> None:
    """
    Run one batch through the converted model to confirm the quantized
    kernels are available at runtime.

    This catches the case where torch.quantization.convert() succeeds but
    the QuantizedCPU kernel is absent (e.g. torch+cpu PyPI wheel does not
    ship the quantized::conv2d.new operator). Raises RuntimeError if inference
    fails so the outer try/except triggers the QUANT_UNSUPPORTED fallback.
    """
    model.eval()
    with torch.no_grad():
        for images, _ in calib_loader:
            if images.dtype != torch.float32:
                images = images.float()
            model(images)   # raises NotImplementedError / RuntimeError if kernel missing
            break            # one batch is enough


def try_static_int8(
    model: nn.Module,
    calib_loader: DataLoader,
    calibration_samples: int = 128,
    backend: Optional[str] = None,
    inplace: bool = False,
    lowp_dtype: str = "bf16",
) -> Tuple[nn.Module, str]:
    """
    Attempt static INT8 quantization with full fallback chain.

    Backend selection (SPEC.md §4.2):
        - backend=None (default): probe supported_engines in order
          fbgemm → x86 → onednn → qnnpack and use the first that works.
        - backend='fbgemm' (or any string): force that backend; skip probing.

    Steps:
        1. Log supported_engines
        2. Select backend (auto or explicit)
        3. Deep-copy model
        4. Fuse Conv-BN-ReLU6
        5. Set qconfig → prepare (insert observers)
        6. Calibrate (forward pass over n_samples images)
        7. Convert (replaces float ops with quantized ops)
        8. Verify inference (1 batch) — catches missing QuantizedCPU kernels

    On any failure in steps 4–8:
        → logs QUANT_UNSUPPORTED with error detail
        → falls back to lowp_dtype (bf16/fp16), then FP32 if lowp also fails

    Args:
        model: Source FP32 model (eval or train mode).
        calib_loader: DataLoader for calibration (128 images sufficient).
        calibration_samples: Number of images to run through observers.
        backend: None = auto-select; string = force that backend.
        inplace: Ignored — we always deep-copy for safety.

    Returns:
        Tuple (quantized_model, method_string) where method_string is one of:
            "static_int8"    — full static INT8 conversion succeeded
            "bf16_fallback"  — INT8 failed; returned BF16 model
            "fp16_fallback"  — INT8 failed; returned FP16 model
            "fp32_fallback"  — INT8 + lowp failed; returned FP32 model
    """
    resolved_lowp = resolve_lowp_dtype(lowp_dtype)
    lowp_apply = _bf16_module.apply if resolved_lowp == "bf16" else _fp16_module.apply
    lowp_method = fallback_method_for_lowp_dtype(resolved_lowp)

    # Step 1: Log supported engines
    _log_supported_engines()

    # Step 2: Select backend
    if backend is None:
        selected = auto_select_backend(model, calib_loader)
        if selected is None:
            # All backends failed probe — invoke fallback INLINE (never crash).
            # RuntimeError here would be outside the try/except below, breaking
            # the "never silent fallback" contract (SPEC.md §4.2).
            log.warning(
                f"{QUANT_UNSUPPORTED}: no working INT8 backend found in "
                f"{list(torch.backends.quantized.supported_engines)}. "
                f"int8_disabled=true. Triggering {resolved_lowp.upper()} fallback."
            )
            try:
                lowp_model = lowp_apply(model, inplace=False)
                log.info(
                    f"Fallback to {resolved_lowp.upper()} successful "
                    f"→ quant_method={lowp_method}"
                )
                return lowp_model, lowp_method
            except Exception as lowp_err:
                log.warning(
                    f"{QUANT_UNSUPPORTED}: {resolved_lowp.upper()} fallback also failed "
                    f"({type(lowp_err).__name__}). Falling back to FP32. "
                    f"quant_method=fp32_fallback"
                )
                return copy.deepcopy(model).float().eval(), "fp32_fallback"
        backend = selected
        log.info(f"Auto-selected backend='{backend}'")
    else:
        log.info(f"Using forced backend='{backend}'")

    try:
        work_model, path = _convert_static_int8_with_backend(
            model,
            calib_loader=calib_loader,
            calibration_samples=calibration_samples,
            backend=backend,
        )
        log.info(
            "Static INT8 conversion successful via %s path → quant_method=static_int8",
            path,
        )
        return work_model, "static_int8"

    except Exception as int8_err:
        log.warning(
            f"{QUANT_UNSUPPORTED}: static INT8 failed ({type(int8_err).__name__}: {int8_err}). "
            f"Falling back to {resolved_lowp.upper()}. int8_disabled=true"
        )
        # lowp fallback — using lowp module apply() so it can be intercepted in tests
        try:
            lowp_model = lowp_apply(model, inplace=False)
            log.info(
                f"Fallback to {resolved_lowp.upper()} successful "
                f"→ quant_method={lowp_method}"
            )
            return lowp_model, lowp_method

        except Exception as lowp_err:
            log.warning(
                f"{QUANT_UNSUPPORTED}: {resolved_lowp.upper()} fallback also failed "
                f"({type(lowp_err).__name__}: {lowp_err}). "
                f"Falling back to FP32. quant_method=fp32_fallback"
            )
            fp32_model = copy.deepcopy(model).float().eval()
            log.info("Fallback to FP32 → quant_method=fp32_fallback")
            return fp32_model, "fp32_fallback"


def check_static_int8_convert_and_infer(
    model: nn.Module,
    calib_loader: DataLoader,
    calibration_samples: int = 128,
    backend: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """
    Diagnostic-only static INT8 conversion+inference check without fallback.

    Returns:
        (success, method, error)
          success=True  -> ("static_int8", "")
          success=False -> ("int8_postcheck_failed", "<explicit_error>")
    """
    try:
        _log_supported_engines()
        selected_backend = backend
        if selected_backend is None:
            selected_backend = auto_select_backend(model, calib_loader)
            if selected_backend is None:
                return False, "int8_postcheck_failed", "no_working_backend"

        _, path = _convert_static_int8_with_backend(
            model,
            calib_loader=calib_loader,
            calibration_samples=calibration_samples,
            backend=str(selected_backend),
        )
        log.info(
            "INT8 postcheck conversion succeeded via %s path (backend=%s)",
            path,
            selected_backend,
        )
        return True, "static_int8", ""
    except Exception as err:
        return (
            False,
            "int8_postcheck_failed",
            f"{type(err).__name__}:{str(err)[:500]}",
        )
