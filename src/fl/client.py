"""
src/fl/client.py — Flower NumPyClient for Adaptive FL

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Protocol per round:
  1. Server sends global model weights + config (quant_bits, round, etc.)
  2. Client receives weights → set_parameters()
  3. Client applies quantization (server-assigned bits) via quantizer.quantize()
  4. Client trains locally for cfg.fl.local_epochs epochs
  5. Client returns UPDATED FP32 parameters + metrics dict

Resolution path is config-controlled via cfg.data.use_32:
  - False: 224×224 paper-aligned path (default)
  - True: native 32×32 lightweight validation path

Metrics returned per round (always included):
  train_loss, train_time_s, quant_bits_requested, quant_method_actual,
  n_samples, dropped, client_id
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

import flwr as fl
from flwr.common import NDArrays, Scalar

from src.common.config import Config, ClientProfileConfig
from src.models.mobilenetv2 import get_model, get_parameters, set_parameters
from src.models.trainer import train_local, evaluate, build_optimizer
from src.data.cifar import get_cifar10_train, get_cifar10_test
from src.compression.lowp import precision_from_quant_method, resolve_lowp_dtype
from src.compression.quantizer import quantize

log = logging.getLogger(__name__)

# CPU device — this project runs CPU-only
_DEVICE = torch.device("cpu")


def _requested_precision_label(quant_bits: int, lowp_dtype: str) -> str:
    """Canonical requested precision label for reporting."""
    if quant_bits == 32:
        return "fp32"
    if quant_bits == 16:
        return resolve_lowp_dtype(lowp_dtype)
    if quant_bits == 8:
        return "int8"
    return "unknown"


# ─── Data helpers ─────────────────────────────────────────────────────────────

def _make_train_loader(
    train_indices: List[int],
    cfg: Config,
    profile: ClientProfileConfig,
    data_root: str = "data/",
) -> DataLoader:
    """Build a training DataLoader from the client's partition indices."""
    full_train_ds = get_cifar10_train(
        root=data_root,
        download=True,
        use_32=cfg.data.use_32,
        augment=cfg.data.train_augment,
    )
    subset = Subset(full_train_ds, train_indices)
    batch_size = cfg.fl.batch_size_for(profile.profile)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **cfg.fl.dataloader_kwargs(),
    )


def _make_eval_loader(
    eval_indices: List[int],
    cfg: Config,
    data_root: str = "data/",
) -> DataLoader:
    """Build a local evaluation DataLoader from the client's test subset."""
    full_test_ds = get_cifar10_test(
        root=data_root,
        download=True,
        use_32=cfg.data.use_32,
    )
    subset = Subset(full_test_ds, eval_indices)
    return DataLoader(
        subset,
        batch_size=2,
        shuffle=False,
        **cfg.fl.dataloader_kwargs(),
    )


def _make_calib_loader(
    train_indices: List[int],
    cfg: Config,
    n_samples: int = 16,
    data_root: str = "data/",
) -> DataLoader:
    """Build a tiny calibration DataLoader for INT8 static quantization."""
    full_train_ds = get_cifar10_train(
        root=data_root,
        download=True,
        use_32=cfg.data.use_32,
        augment=False,
    )
    calib_indices = train_indices[:min(n_samples, len(train_indices))]
    subset = Subset(full_train_ds, calib_indices)
    return DataLoader(
        subset, batch_size=2, shuffle=False,
        **cfg.fl.dataloader_kwargs(),
    )


# ─── FlowerClient ─────────────────────────────────────────────────────────────

class FlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for federated training.

    One instance is created per client per simulation round. The model is
    initialised randomly and overwritten with the server's global weights
    at the start of every round via set_parameters().
    """

    def __init__(
        self,
        client_id: int,
        train_indices: List[int],
        eval_indices: List[int],
        cfg: Config,
        profile: ClientProfileConfig,
        data_root: str = "data/",
    ):
        self.client_id = client_id
        self.train_indices = train_indices
        self.eval_indices = eval_indices
        self.cfg = cfg
        self.profile = profile
        self.data_root = data_root
        # Server-side model — always FP32; quantized copy trained per round
        self.model = get_model(freeze_features=self.cfg.fl.freeze_features)

    # ── Flower interface ───────────────────────────────────────────────────────

    def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        """Expose logical client identity for strategy-side UUID mapping."""
        return {"client_id": str(self.client_id)}

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current model parameters as a list of FP32 numpy arrays."""
        return get_parameters(self.model)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Receive global weights → apply quant → train locally → return FP32.

        config keys expected from server:
            quant_bits (int): 32 | 16 | 8
            round      (int): current FL round number
        """
        # 1. Load global weights into model
        set_parameters(self.model, parameters)

        # 2. Extract server-side config
        quant_bits = int(config.get("quant_bits", 32))
        fl_round = int(config.get("round", 0))
        if self.cfg.quantization.mode == "mixed" and self.cfg.quantization.per_client:
            quant_bits = int(self.cfg.quantization.per_client.get(self.client_id, quant_bits))
        requested_precision = _requested_precision_label(
            quant_bits, self.cfg.quantization.lowp_dtype
        )
        log.info(
            f"[Client {self.client_id}] round={fl_round} "
            f"quant_bits={quant_bits} requested_precision={requested_precision} "
            f"n_train={len(self.train_indices)}"
        )

        # 3. Build calibration loader if INT8 requested
        calib_loader: Optional[DataLoader] = None
        if quant_bits == 8:
            calib_loader = _make_calib_loader(
                self.train_indices, self.cfg,
                n_samples=self.cfg.quantization.calibration_samples,
                data_root=self.data_root,
            )

        # 4. Apply quantization — returns a deep copy; self.model unchanged
        q_model, quant_method = quantize(
            self.model,
            bits=quant_bits,
            calib_loader=calib_loader,
            calibration_samples=self.cfg.quantization.calibration_samples,
            lowp_dtype=self.cfg.quantization.lowp_dtype,
        )

        # 5. Build train loader
        train_loader = _make_train_loader(
            self.train_indices, self.cfg, self.profile,
            data_root=self.data_root,
        )

        # 6. Build optimizer on the quantized model
        optimizer = build_optimizer(
            q_model,
            optimizer_name=self.cfg.fl.optimizer,
            lr=self.cfg.fl.lr,
            momentum=self.cfg.fl.momentum,
            weight_decay=self.cfg.fl.weight_decay,
        )

        # 7. Train locally — returns List[TrainResult], one per epoch
        t0 = time.time()
        try:
            train_results = train_local(
                q_model,
                train_loader,
                optimizer=optimizer,
                epochs=self.cfg.fl.local_epochs,
                device=_DEVICE,
                grad_clip_norm=1.0,
            )
        except RuntimeError as err:
            # Some CPU kernels can reject BF16 for specific ops. Fallback is explicit.
            if (
                requested_precision == "bf16"
                and ("bfloat16" in str(err).lower() or "bf16" in str(err).lower())
            ):
                log.warning(
                    f"[Client {self.client_id}] BF16 runtime unsupported in round={fl_round} "
                    f"({type(err).__name__}: {err}). Retrying in FP32."
                )
                q_model, quant_method = quantize(
                    self.model,
                    bits=32,
                    calibration_samples=self.cfg.quantization.calibration_samples,
                    lowp_dtype=self.cfg.quantization.lowp_dtype,
                )
                optimizer = build_optimizer(
                    q_model,
                    optimizer_name=self.cfg.fl.optimizer,
                    lr=self.cfg.fl.lr,
                    momentum=self.cfg.fl.momentum,
                    weight_decay=self.cfg.fl.weight_decay,
                )
                train_results = train_local(
                    q_model,
                    train_loader,
                    optimizer=optimizer,
                    epochs=self.cfg.fl.local_epochs,
                    device=_DEVICE,
                    grad_clip_norm=1.0,
                )
                quant_method = "fp32_fallback"
            else:
                raise
        train_time = time.time() - t0

        # Final epoch result (or the OOM result if it happened mid-training)
        result = train_results[-1]
        dropped = result.oom
        actual_precision = precision_from_quant_method(quant_method)
        if actual_precision == "unknown":
            actual_precision = _requested_precision_label(
                quant_bits, self.cfg.quantization.lowp_dtype
            )

        if dropped:
            log.warning(
                f"[Client {self.client_id}] OOM during training "
                f"(requested={requested_precision}, actual={quant_method}) "
                f"— returning unmodified global weights"
            )
            return (
                get_parameters(self.model),
                len(self.train_indices),
                {
                    "train_loss": -1.0,  # sentinel for NaN (Flower Scalar must be float)
                    "train_time_s": float(train_time),
                    "quant_bits_requested": quant_bits,
                    "quant_precision_requested": requested_precision,
                    "quant_precision_actual": actual_precision,
                    "quant_method_actual": quant_method,
                    "n_samples": len(self.train_indices),
                    "dropped": 1,
                    "client_id": self.client_id,
                },
            )

        # 8. Extract updated weights (always FP32 via get_parameters)
        updated_params = get_parameters(q_model)

        # 9. Sync self.model to the updated weights for continuity
        set_parameters(self.model, updated_params)

        # result.loss is the final epoch's average loss
        final_loss = result.loss

        metrics: Dict[str, Scalar] = {
            "train_loss": float(final_loss),
            "train_time_s": float(train_time),
            "quant_bits_requested": quant_bits,
            "quant_precision_requested": requested_precision,
            "quant_precision_actual": actual_precision,
            "quant_method_actual": quant_method,
            "n_samples": len(self.train_indices),
            "dropped": 0,
            "client_id": self.client_id,
        }

        log.info(
            f"[Client {self.client_id}] fit done: loss={final_loss:.4f} "
            f"time={train_time:.1f}s requested={requested_precision} actual={quant_method}"
        )
        return updated_params, len(self.train_indices), metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate global model on this client's local test subset."""
        set_parameters(self.model, parameters)
        self.model.eval()
        eval_loader = _make_eval_loader(
            self.eval_indices, self.cfg, data_root=self.data_root
        )
        result = evaluate(self.model, eval_loader, device=_DEVICE)
        log.info(
            f"[Client {self.client_id}] evaluate: "
            f"loss={result.loss:.4f} acc={result.accuracy:.4f}"
        )
        return (
            float(result.loss),
            len(self.eval_indices),
            {
                "accuracy": float(result.accuracy),
                "client_id": self.client_id,
            },
        )
