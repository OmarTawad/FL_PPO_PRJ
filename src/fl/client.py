"""
src/fl/client.py — Flower NumPyClient for Adaptive FL

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Protocol per round:
  1. Server sends global model weights + config (quant_bits, round, etc.)
  2. Client receives weights → set_parameters()
  3. Client applies quantization (server-assigned bits) via quantizer.quantize()
  4. Client trains locally for cfg.fl.local_epochs epochs
  5. Client returns transport payload (FP32 model or delta payload) + metrics dict

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
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

import flwr as fl
from flwr.common import NDArrays, Scalar

from src.common.config import Config, ClientProfileConfig
from src.models.mobilenetv2 import get_model, get_parameters, set_parameters
from src.models.trainer import train_local, evaluate, build_optimizer
from src.data.cifar import get_cifar10_train, get_cifar10_test
from src.compression.lowp import precision_from_quant_method, resolve_lowp_dtype
from src.compression.quantizer import quantize
from src.compression.int8 import check_static_int8_convert_and_infer
from src.compression.transport_int8 import (
    quantize_delta_int8_per_tensor,
    encode_transport_meta_json,
)
from src.compression.qat import (
    convert_qat_model_for_check,
    get_qat_backend_used,
    get_qat_scope_used,
)
from src.compression.pruning import (
    apply_magnitude_unstructured_pruning,
    resolve_pruning_amount,
)

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


def _transport_dtype_for_client(cfg: Config, client_id: int) -> str:
    return str(cfg.quantization.transport_per_client.get(client_id, "fp32"))


def _encode_transport_payload(
    cfg: Config,
    client_id: int,
    global_params_fp32: List[np.ndarray],
    updated_params_fp32: List[np.ndarray],
    transport_dtype_override: str = "",
) -> Tuple[NDArrays, str, str, str, str, int, int]:
    """
    Encode client return payload per transport policy.

    Returns:
      (payload, requested_dtype, actual_dtype, payload_kind,
       transport_meta_json, transport_meta_present, tensor_count)
    """
    override = str(transport_dtype_override).strip().lower()
    if override in ("fp32", "int8"):
        requested_dtype = override
    else:
        requested_dtype = _transport_dtype_for_client(cfg, client_id)
    if cfg.quantization.transport_mode != "delta":
        return (
            updated_params_fp32,
            requested_dtype,
            "fp32",
            "model_fp32",
            "",
            0,
            len(updated_params_fp32),
        )

    if len(global_params_fp32) != len(updated_params_fp32):
        raise ValueError(
            f"transport_delta_length_mismatch:{len(global_params_fp32)}!={len(updated_params_fp32)}"
        )

    deltas = [
        (np.asarray(u, dtype=np.float32) - np.asarray(g, dtype=np.float32)).astype(
            np.float32, copy=False
        )
        for u, g in zip(updated_params_fp32, global_params_fp32)
    ]
    if requested_dtype == "int8":
        q_deltas, meta = quantize_delta_int8_per_tensor(deltas)
        meta_json = encode_transport_meta_json(meta)
        return (
            q_deltas,
            requested_dtype,
            "int8",
            "delta_int8",
            meta_json,
            1,
            len(q_deltas),
        )

    return (
        deltas,
        requested_dtype,
        "fp32",
        "delta_fp32",
        "",
        0,
        len(deltas),
    )


def _validate_static_int8_trainability(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
) -> Tuple[bool, str]:
    """
    Lightweight guard for train-time static INT8 viability.

    Static eager INT8 in PyTorch is primarily inference-oriented. We explicitly
    check whether a converted model still exposes trainable parameters and can
    run one backward pass. This prevents ambiguous strict-INT8 reporting.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return False, "static_int8_train_unsupported:no_trainable_parameters"

    try:
        images, labels = next(iter(train_loader))
    except StopIteration:
        return False, "static_int8_train_unsupported:empty_train_loader"

    images = images.to(device=device, dtype=torch.float32)
    labels = labels.to(device).view(-1).long()
    criterion = torch.nn.CrossEntropyLoss()

    try:
        model.train()
        for p in params:
            if p.grad is not None:
                p.grad = None
        outputs = model(images)
        if outputs.dtype != torch.float32:
            outputs = outputs.float()
        loss = criterion(outputs, labels)
        loss.backward()
        return True, ""
    except Exception as err:
        return (
            False,
            f"static_int8_train_unsupported:{type(err).__name__}",
        )
    finally:
        model.eval()
        for p in params:
            if p.grad is not None:
                p.grad = None


def _copy_trainable_weights_to_base_model(
    base_model: torch.nn.Module,
    qat_model: torch.nn.Module,
) -> Tuple[int, int]:
    """
    Copy overlapping trainable parameter tensors from QAT model into base FP32 model.

    Returns:
      (copied_count, total_trainable_count)
    """
    base_state = base_model.state_dict()
    qat_state = qat_model.state_dict()
    trainable_names = [
        name for name, param in base_model.named_parameters()
        if param.requires_grad
    ]
    copied = 0
    for name in trainable_names:
        if name not in base_state or name not in qat_state:
            continue
        src = qat_state[name]
        dst = base_state[name]
        if src.shape != dst.shape:
            continue
        base_state[name] = src.detach().cpu().float()
        copied += 1
    base_model.load_state_dict(base_state, strict=False)
    return copied, len(trainable_names)


# ─── Data helpers ─────────────────────────────────────────────────────────────

def _make_train_loader(
    train_indices: List[int],
    cfg: Config,
    profile: ClientProfileConfig,
    full_train_ds: Optional[Dataset] = None,
    data_root: str = "data/",
) -> DataLoader:
    """Build a training DataLoader from the client's partition indices."""
    if full_train_ds is None:
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
    full_eval_ds: Optional[Dataset] = None,
    data_root: str = "data/",
) -> DataLoader:
    """Build a local evaluation DataLoader from the client's test subset."""
    full_test_ds = full_eval_ds
    if full_test_ds is None:
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
    full_train_ds: Optional[Dataset] = None,
    data_root: str = "data/",
) -> DataLoader:
    """Build a tiny calibration DataLoader for INT8 static quantization."""
    if full_train_ds is None:
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
        self._base_state_keys = list(self.model.state_dict().keys())
        self._train_ds_aug: Optional[Dataset] = None
        self._train_ds_noaug: Optional[Dataset] = None
        self._eval_ds: Optional[Dataset] = None
        self._qat_work_model: Optional[torch.nn.Module] = None
        self._pruning_applied_once: bool = False

    def _get_train_dataset(self, augment: bool) -> Dataset:
        if augment:
            if self._train_ds_aug is None:
                self._train_ds_aug = get_cifar10_train(
                    root=self.data_root,
                    download=True,
                    use_32=self.cfg.data.use_32,
                    augment=True,
                )
            return self._train_ds_aug
        if self._train_ds_noaug is None:
            self._train_ds_noaug = get_cifar10_train(
                root=self.data_root,
                download=True,
                use_32=self.cfg.data.use_32,
                augment=False,
            )
        return self._train_ds_noaug

    def _get_eval_dataset(self) -> Dataset:
        if self._eval_ds is None:
            self._eval_ds = get_cifar10_test(
                root=self.data_root,
                download=True,
                use_32=self.cfg.data.use_32,
            )
        return self._eval_ds

    def _sync_base_to_qat_work_model(self) -> None:
        if self._qat_work_model is None:
            raise RuntimeError("QAT work model not initialized")
        base_state = self.model.state_dict()
        sync_dict: Dict[str, torch.Tensor] = {}
        for key in self._base_state_keys:
            if key in base_state:
                sync_dict[key] = base_state[key].detach().cpu()
        self._qat_work_model.load_state_dict(sync_dict, strict=False)

    def _get_or_prepare_qat_work_model(self) -> torch.nn.Module:
        if self._qat_work_model is None:
            self._qat_work_model = get_model(
                freeze_features=self.cfg.fl.freeze_features
            )
            self._sync_base_to_qat_work_model()
            self._qat_work_model, _ = quantize(
                self._qat_work_model,
                bits=8,
                backend=self.cfg.quantization.int8_backend,
                lowp_dtype=self.cfg.quantization.lowp_dtype,
                int8_impl="qat",
                qat_scope=self.cfg.quantization.qat_scope,
                qat_inplace=True,
            )
            return self._qat_work_model
        self._sync_base_to_qat_work_model()
        return self._qat_work_model

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
        Receive global weights → apply quant → train locally → return transport payload.

        config keys expected from server:
            quant_bits (int): 32 | 16 | 8
            round      (int): current FL round number
        """
        # 1. Load global weights into model
        set_parameters(self.model, parameters)
        global_params_fp32 = [
            np.asarray(p, dtype=np.float32).copy() for p in parameters
        ]

        # 2. Extract server-side config
        quant_bits = int(config.get("quant_bits", 32))
        fl_round = int(config.get("round", 0))
        transport_dtype_override = str(
            config.get("transport_dtype", "")
        ).strip().lower()
        policy_action_requested = int(config.get("policy_action", -1))
        requested_precision = _requested_precision_label(
            quant_bits, self.cfg.quantization.lowp_dtype
        )
        training_precision_requested = requested_precision
        int8_impl = self.cfg.quantization.int8_impl
        log.info(
            f"[Client {self.client_id}] round={fl_round} "
            f"quant_bits={quant_bits} requested_precision={requested_precision} "
            f"n_train={len(self.train_indices)} int8_impl={int8_impl} "
            f"int8_backend={self.cfg.quantization.int8_backend} "
            f"transport_override={transport_dtype_override or 'none'} "
            f"policy_action={policy_action_requested}"
        )

        # 3. Build calibration loader only for static INT8
        calib_loader: Optional[DataLoader] = None
        if quant_bits == 8 and int8_impl == "static":
            train_ds_noaug = self._get_train_dataset(augment=False)
            calib_loader = _make_calib_loader(
                self.train_indices, self.cfg,
                n_samples=self.cfg.quantization.calibration_samples,
                full_train_ds=train_ds_noaug,
                data_root=self.data_root,
            )
            log.info(
                f"[Client {self.client_id}] round={fl_round} built static-int8 calibration loader"
            )

        # 4. Build train loader from cached dataset
        train_ds = self._get_train_dataset(augment=self.cfg.data.train_augment)
        train_loader = _make_train_loader(
            self.train_indices, self.cfg, self.profile,
            full_train_ds=train_ds,
            data_root=self.data_root,
        )

        # 5. Apply quantization — returns a deep copy; self.model unchanged
        quant_fallback_reason = ""
        qat_enabled = int(quant_bits == 8 and int8_impl == "qat")
        qat_backend_used = ""
        qat_scope_used = ""
        qat_convert_attempted = 0
        qat_convert_success = 0
        qat_convert_error = ""
        qat_convert_probe_policy = (
            f"client_id={self.cfg.quantization.qat_convert_probe_client_id},"
            f"round={self.cfg.quantization.qat_convert_probe_round}"
        )
        int8_convert_attempted = 0
        int8_convert_success = 0
        int8_convert_error = ""
        int8_inference_check_success = 0
        posttrain_inference_method_actual = "not_attempted"
        pruning_method = str(self.cfg.pruning.method)
        pruning_amount_requested = 0.0
        pruning_amount_applied = 0.0
        pruning_requested = 0
        pruning_applied = 0
        pruning_active_during_training = 0
        pruning_skip_reason = ""

        if quant_bits == 8 and int8_impl == "qat":
            q_model = self._get_or_prepare_qat_work_model()
            quant_method = "qat_int8_train"
            qat_backend_used = get_qat_backend_used(
                q_model, fallback=self.cfg.quantization.int8_backend
            )
            qat_scope_used = get_qat_scope_used(
                q_model, fallback=self.cfg.quantization.qat_scope
            )
        else:
            q_model, quant_method = quantize(
                self.model,
                bits=quant_bits,
                calib_loader=calib_loader,
                calibration_samples=self.cfg.quantization.calibration_samples,
                backend=self.cfg.quantization.int8_backend,
                lowp_dtype=self.cfg.quantization.lowp_dtype,
                int8_impl=int8_impl,
                qat_scope=self.cfg.quantization.qat_scope,
                qat_inplace=False,
            )

        if quant_bits == 8 and quant_method == "static_int8":
            ok_train_int8, reason = _validate_static_int8_trainability(
                q_model, train_loader, _DEVICE
            )
            if not ok_train_int8:
                quant_fallback_reason = reason
                log.warning(
                    f"[Client {self.client_id}] {reason}; "
                    f"falling back from static_int8 to configured lowp dtype."
                )
                q_model, quant_method = quantize(
                    self.model,
                    bits=16,
                    calibration_samples=self.cfg.quantization.calibration_samples,
                    backend=self.cfg.quantization.int8_backend,
                    lowp_dtype=self.cfg.quantization.lowp_dtype,
                )

        if self.cfg.pruning.enabled:
            pruning_amount_requested = resolve_pruning_amount(
                default_amount=self.cfg.pruning.amount,
                per_client=self.cfg.pruning.per_client,
                client_id=self.client_id,
            )
            pruning_requested = int(pruning_amount_requested > 0.0)
            should_apply_pruning = pruning_requested and (
                self.cfg.pruning.apply_per_round or (not self._pruning_applied_once)
            )
            if should_apply_pruning:
                try:
                    (
                        pruning_applied_bool,
                        pruning_skip_reason,
                        pruned_module_count,
                        pruning_amount_applied,
                    ) = apply_magnitude_unstructured_pruning(
                        q_model,
                        amount=pruning_amount_requested,
                        target_modules=self.cfg.pruning.target_modules,
                    )
                    pruning_applied = int(pruning_applied_bool)
                    pruning_active_during_training = int(pruning_applied_bool)
                    if pruning_applied_bool:
                        self._pruning_applied_once = True
                    log.info(
                        f"[Client {self.client_id}] pruning round={fl_round} "
                        f"requested={pruning_requested} applied={pruning_applied} "
                        f"method={pruning_method} amount_requested={pruning_amount_requested:.4f} "
                        f"amount_applied={pruning_amount_applied:.4f} "
                        f"targets={','.join(self.cfg.pruning.target_modules)} "
                        f"target_module_count={pruned_module_count} "
                        f"skip_reason={pruning_skip_reason or 'none'}"
                    )
                except Exception as err:
                    pruning_applied = 0
                    pruning_active_during_training = 0
                    pruning_amount_applied = 0.0
                    pruning_skip_reason = f"error:{type(err).__name__}"
                    log.warning(
                        f"[Client {self.client_id}] pruning failed: {type(err).__name__}: {err}"
                    )
            elif pruning_requested and not self.cfg.pruning.apply_per_round:
                pruning_skip_reason = "apply_per_round_false_already_applied"
            elif not pruning_requested:
                pruning_skip_reason = "not_requested_for_client"

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

        if quant_method == "qat_int8_train":
            probe_client = self.cfg.quantization.qat_convert_probe_client_id
            probe_round = self.cfg.quantization.qat_convert_probe_round
            should_probe_client = probe_client < 0 or self.client_id == probe_client
            should_probe_round = probe_round <= 0 or fl_round == probe_round
            should_probe = (
                self.cfg.quantization.qat_convert_after_fit
                and should_probe_client
                and should_probe_round
            )
            if should_probe:
                qat_convert_attempted = 1
                eval_ds = self._get_eval_dataset()
                qat_eval_loader = _make_eval_loader(
                    self.eval_indices,
                    self.cfg,
                    full_eval_ds=eval_ds,
                    data_root=self.data_root,
                )
                qat_convert_success_bool, qat_convert_error = convert_qat_model_for_check(
                    q_model,
                    backend=qat_backend_used or self.cfg.quantization.int8_backend,
                    eval_loader=qat_eval_loader,
                    n_batches=self.cfg.quantization.qat_eval_batches_for_convert_check,
                    device=_DEVICE,
                )
                qat_convert_success = int(qat_convert_success_bool)
                if qat_convert_success_bool:
                    log.info(
                        f"[Client {self.client_id}] QAT convert-check succeeded "
                        f"(backend={qat_backend_used}, scope={qat_scope_used})"
                    )
                else:
                    log.warning(
                        f"[Client {self.client_id}] QAT convert-check failed "
                        f"(backend={qat_backend_used}, scope={qat_scope_used}): {qat_convert_error}"
                    )
            else:
                log.info(
                    f"[Client {self.client_id}] QAT convert-check skipped "
                    f"(policy={qat_convert_probe_policy})"
                )

        # Optional diagnostic-only INT8 post-training conversion/inference check.
        int8_probe_client = self.cfg.quantization.int8_postcheck_probe_client_id
        int8_probe_round = self.cfg.quantization.int8_postcheck_probe_round
        should_probe_client = int8_probe_client < 0 or self.client_id == int8_probe_client
        should_probe_round = int8_probe_round <= 0 or fl_round == int8_probe_round
        should_run_postcheck = (
            self.cfg.quantization.int8_postcheck_enabled
            and self.cfg.quantization.int8_postcheck_after_fit
            and should_probe_client
            and should_probe_round
        )
        if should_run_postcheck:
            int8_convert_attempted = 1
            # Reuse the already-built train loader to avoid creating an extra
            # dataset/loader copy on constrained clients during postcheck.
            postcheck_loader = train_loader
            selected_backend = self.cfg.quantization.int8_postcheck_backend
            backend_arg = None if selected_backend == "auto" else selected_backend
            (
                check_ok,
                check_method,
                check_err,
            ) = check_static_int8_convert_and_infer(
                q_model,
                calib_loader=postcheck_loader,
                calibration_samples=self.cfg.quantization.calibration_samples,
                backend=backend_arg,
            )
            posttrain_inference_method_actual = check_method
            int8_convert_success = int(check_ok)
            int8_inference_check_success = int(check_ok)
            int8_convert_error = check_err
            if check_ok:
                log.info(
                    f"[Client {self.client_id}] INT8 postcheck succeeded "
                    f"(backend={selected_backend}, method={check_method})"
                )
            else:
                log.warning(
                    f"[Client {self.client_id}] INT8 postcheck failed "
                    f"(backend={selected_backend}, method={check_method}): {check_err}"
                )
        else:
            log.info(
                f"[Client {self.client_id}] INT8 postcheck skipped "
                f"(enabled={self.cfg.quantization.int8_postcheck_enabled}, "
                f"after_fit={self.cfg.quantization.int8_postcheck_after_fit}, "
                f"probe_client={int8_probe_client}, probe_round={int8_probe_round})"
            )

        # Final epoch result (or the OOM result if it happened mid-training)
        result = train_results[-1]
        dropped = result.oom
        actual_precision = precision_from_quant_method(quant_method)
        if actual_precision == "unknown":
            actual_precision = _requested_precision_label(
                quant_bits, self.cfg.quantization.lowp_dtype
            )
        training_precision_actual = actual_precision
        training_method_actual = quant_method

        if dropped:
            dropped_params_fp32 = [arr.copy() for arr in global_params_fp32]
            (
                transport_payload,
                transport_dtype_requested,
                transport_dtype_actual,
                transport_payload_kind,
                transport_quant_meta_json,
                transport_quant_meta_present,
                transport_tensor_count,
            ) = _encode_transport_payload(
                self.cfg,
                self.client_id,
                global_params_fp32=global_params_fp32,
                updated_params_fp32=dropped_params_fp32,
                transport_dtype_override=transport_dtype_override,
            )
            log.warning(
                f"[Client {self.client_id}] OOM during training "
                f"(requested={requested_precision}, actual={quant_method}) "
                f"— returning no-op transport payload kind={transport_payload_kind}"
            )
            return (
                transport_payload,
                len(self.train_indices),
                {
                    "train_loss": -1.0,  # sentinel for NaN (Flower Scalar must be float)
                    "train_time_s": float(train_time),
                    "quant_bits_requested": quant_bits,
                    "quant_precision_requested": requested_precision,
                    "quant_precision_actual": actual_precision,
                    "quant_method_actual": quant_method,
                    "training_precision_requested": training_precision_requested,
                    "training_precision_actual": training_precision_actual,
                    "training_method_actual": training_method_actual,
                    "quant_fallback_reason": quant_fallback_reason,
                    "qat_enabled": qat_enabled,
                    "qat_backend_used": qat_backend_used,
                    "qat_scope_used": qat_scope_used,
                    "qat_convert_attempted": qat_convert_attempted,
                    "qat_convert_success": qat_convert_success,
                    "qat_convert_error": qat_convert_error,
                    "qat_convert_probe_policy": qat_convert_probe_policy,
                    "int8_convert_attempted": int8_convert_attempted,
                    "int8_convert_success": int8_convert_success,
                    "int8_convert_error": int8_convert_error,
                    "int8_inference_check_success": int8_inference_check_success,
                    "posttrain_inference_method_actual": posttrain_inference_method_actual,
                    "pruning_requested": pruning_requested,
                    "pruning_applied": pruning_applied,
                    "pruning_amount_requested": float(pruning_amount_requested),
                    "pruning_amount_applied": float(pruning_amount_applied),
                    "pruning_method": pruning_method,
                    "pruning_active_during_training": pruning_active_during_training,
                    "pruning_skip_reason": pruning_skip_reason,
                    "transport_mode": self.cfg.quantization.transport_mode,
                    "policy_action_requested": policy_action_requested,
                    "transport_dtype_requested": transport_dtype_requested,
                    "transport_dtype_actual": transport_dtype_actual,
                    "transport_payload_kind": transport_payload_kind,
                    "transport_quant_meta_present": transport_quant_meta_present,
                    "transport_quant_meta_json": transport_quant_meta_json,
                    "transport_tensor_count": transport_tensor_count,
                    "n_samples": len(self.train_indices),
                    "dropped": 1,
                    "client_id": self.client_id,
                },
            )

        # 8. Extract updated weights (always FP32 for transport)
        if quant_method == "qat_int8_train":
            copied, total = _copy_trainable_weights_to_base_model(
                self.model, q_model
            )
            log.info(
                f"[Client {self.client_id}] QAT weight sync: "
                f"copied_trainable={copied}/{total}"
            )
            updated_params = get_parameters(self.model)
        else:
            updated_params = get_parameters(q_model)
            # 9. Sync self.model to the updated weights for continuity
            set_parameters(self.model, updated_params)

        (
            transport_payload,
            transport_dtype_requested,
            transport_dtype_actual,
            transport_payload_kind,
            transport_quant_meta_json,
            transport_quant_meta_present,
            transport_tensor_count,
        ) = _encode_transport_payload(
            self.cfg,
            self.client_id,
            global_params_fp32=global_params_fp32,
            updated_params_fp32=updated_params,
            transport_dtype_override=transport_dtype_override,
        )

        # result.loss is the final epoch's average loss
        final_loss = result.loss

        metrics: Dict[str, Scalar] = {
            "train_loss": float(final_loss),
            "train_time_s": float(train_time),
            "quant_bits_requested": quant_bits,
            "quant_precision_requested": requested_precision,
            "quant_precision_actual": actual_precision,
            "quant_method_actual": quant_method,
            "training_precision_requested": training_precision_requested,
            "training_precision_actual": training_precision_actual,
            "training_method_actual": training_method_actual,
            "quant_fallback_reason": quant_fallback_reason,
            "qat_enabled": qat_enabled,
            "qat_backend_used": qat_backend_used,
            "qat_scope_used": qat_scope_used,
            "qat_convert_attempted": qat_convert_attempted,
            "qat_convert_success": qat_convert_success,
            "qat_convert_error": qat_convert_error,
            "qat_convert_probe_policy": qat_convert_probe_policy,
            "int8_convert_attempted": int8_convert_attempted,
            "int8_convert_success": int8_convert_success,
            "int8_convert_error": int8_convert_error,
            "int8_inference_check_success": int8_inference_check_success,
            "posttrain_inference_method_actual": posttrain_inference_method_actual,
            "pruning_requested": pruning_requested,
            "pruning_applied": pruning_applied,
            "pruning_amount_requested": float(pruning_amount_requested),
            "pruning_amount_applied": float(pruning_amount_applied),
            "pruning_method": pruning_method,
            "pruning_active_during_training": pruning_active_during_training,
            "pruning_skip_reason": pruning_skip_reason,
            "transport_mode": self.cfg.quantization.transport_mode,
            "policy_action_requested": policy_action_requested,
            "transport_dtype_requested": transport_dtype_requested,
            "transport_dtype_actual": transport_dtype_actual,
            "transport_payload_kind": transport_payload_kind,
            "transport_quant_meta_present": transport_quant_meta_present,
            "transport_quant_meta_json": transport_quant_meta_json,
            "transport_tensor_count": transport_tensor_count,
            "n_samples": len(self.train_indices),
            "dropped": 0,
            "client_id": self.client_id,
        }

        log.info(
            f"[Client {self.client_id}] fit done: loss={final_loss:.4f} "
            f"time={train_time:.1f}s requested={requested_precision} actual={quant_method} "
            f"train_precision={training_precision_actual} "
            f"transport={transport_payload_kind}:{transport_dtype_actual} "
            f"int8_postcheck={posttrain_inference_method_actual} "
            f"qat_scope={qat_scope_used or 'n/a'}"
        )
        return transport_payload, len(self.train_indices), metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate global model on this client's local test subset."""
        set_parameters(self.model, parameters)
        self.model.eval()
        eval_ds = self._get_eval_dataset()
        eval_loader = _make_eval_loader(
            self.eval_indices,
            self.cfg,
            full_eval_ds=eval_ds,
            data_root=self.data_root,
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
