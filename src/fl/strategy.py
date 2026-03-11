"""
src/fl/strategy.py — FedAvgQuant: custom Flower strategy

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Extends FedAvg with:
  1. configure_fit(): assigns per-client quant_bits via FitIns.config
     Phase 6 (fixed_*): uniform bits from cfg.quantization
     Phase 7 (adaptive): per-client bits from self.current_quant_assignments
       set by FLEnv before each env.step().
  2. aggregate_fit(): FedAvg weights + server eval + per-round JSON log
     Sets self.last_round_log for run_one_round() to return to the env.
  3. Global model evaluation on server-held test set after each aggregation
  4. DropoutTracker integration
  5. reset_state(): called by FLEnv on env.reset()

Per-round JSON schema (SPEC.md §8):
  round, selected_clients, quant_assignments, actual_quant_method,
  dropout_clients, dropout_fraction, mean_train_loss,
  global_accuracy, accuracy_delta, timestamp_utc
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import flwr as fl
from flwr.common import (
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    GetPropertiesIns,
    NDArrays, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

from src.common.config import Config
from src.models.mobilenetv2 import get_model, set_parameters, get_parameters
from src.models.trainer import evaluate
from src.compression.lowp import precision_from_quant_method
from src.heterogeneity.dropout import DropoutTracker

log = logging.getLogger(__name__)


class FedAvgQuant(FedAvg):
    """
    Custom FedAvg strategy with per-client quantization assignment and JSON logging.

    Phase 6 (fixed modes): configure_fit assigns same quant_bits to all clients.
    Phase 7 (adaptive):    configure_fit uses self.current_quant_assignments
                           {cid_str: bits} set externally by FLEnv.step().
                           Only clients in current_quant_assignments are selected.
    """

    def __init__(
        self,
        server_test_loader: DataLoader,
        dropout_tracker: DropoutTracker,
        cfg: Config,
        output_dir: Path,
        initial_parameters: Parameters,
    ):
        min_round_clients = max(
            1, min(cfg.fl.min_clients_per_round, cfg.clients.count)
        )
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=min_round_clients,
            min_evaluate_clients=min_round_clients,
            min_available_clients=min_round_clients,
            initial_parameters=initial_parameters,
        )
        self.server_test_loader = server_test_loader
        self.dropout_tracker = dropout_tracker
        self.cfg = cfg
        self.output_dir = output_dir

        # Server-side global model for evaluation
        self.global_model = get_model(
            freeze_features=self.cfg.fl.freeze_features
        )
        self._state_keys = list(self.global_model.state_dict().keys())
        self._global_params_nd = parameters_to_ndarrays(initial_parameters)
        set_parameters(self.global_model, self._global_params_nd)

        # State reset between env episodes
        self._prev_accuracy: Optional[float] = None
        self.last_round_log: dict = {}

        # Phase 7: PPO sets this before each env.step()
        # Dict[str, int] mapping cid → quant_bits for selected clients only.
        # None in Phase 6 (fixed) mode.
        self.current_quant_assignments: Optional[Dict[str, int]] = None
        # Last configure_fit() assignment snapshot: logical client id -> quant bits.
        self._last_fit_quant_assignment: Dict[str, int] = {}
        # Last configure_fit() proxy-to-logical mapping for observability.
        self._last_fit_proxy_map: Dict[str, str] = {}
        # gRPC mode uses opaque proxy IDs; map them to logical client IDs 0..N-1.
        self._proxy_to_logical: Dict[str, int] = {}

    # ── Public: env.reset() ────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Reset per-episode state. Called by FLEnv.reset()."""
        self._prev_accuracy = None
        self.last_round_log = {}
        self.current_quant_assignments = None
        self._last_fit_quant_assignment = {}
        self._last_fit_proxy_map = {}
        self._proxy_to_logical = {}

    # ── configure_fit ──────────────────────────────────────────────────────────

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Select clients and build FitIns with per-client quant_bits.

        Adaptive mode (Phase 7):
            Uses self.current_quant_assignments {cid_str: bits}.
            Only selected clients (those with entries) receive FitIns.

        Mixed mode:
            Recomputes per-client assignments every round, preferring
            cfg.quantization.per_client with optional runtime override.

        Fixed mode (Phase 6):
            All clients selected; uniform bits from cfg.quantization.
        """
        mode = self.cfg.quantization.mode

        if mode == "mixed":
            # Round-scoped mixed assignment (always recomputed at fit-time).
            client_instructions = super().configure_fit(
                server_round, parameters, client_manager
            )
            effective_assignments = self._effective_mixed_assignments()
            updated: List[Tuple[ClientProxy, FitIns]] = []
            used_ids: set[int] = set()
            round_assignment: Dict[str, int] = {}
            proxy_map: Dict[str, str] = {}

            for client_proxy, fit_ins in client_instructions:
                proxy_cid = str(client_proxy.cid)
                logical_id = self._resolve_logical_client_id_for_fit(
                    proxy_cid=proxy_cid,
                    client_proxy=client_proxy,
                    used_ids=used_ids,
                )
                used_ids.add(logical_id)
                logical_cid = str(logical_id)
                bits = int(effective_assignments.get(logical_cid, 32))

                config = dict(fit_ins.config)
                config["quant_bits"] = bits
                config["round"] = server_round
                updated.append((client_proxy, FitIns(fit_ins.parameters, config)))

                round_assignment[logical_cid] = bits
                proxy_map[proxy_cid] = logical_cid

            self._last_fit_quant_assignment = round_assignment
            self._last_fit_proxy_map = proxy_map
            details = [
                {
                    "proxy_cid": proxy_cid,
                    "client_id": proxy_map[proxy_cid],
                    "quant_bits": round_assignment[proxy_map[proxy_cid]],
                }
                for proxy_cid in sorted(proxy_map.keys())
            ]
            log.info(
                f"[Strategy] round={server_round} MIXED fit assignments: {details}"
            )
            return updated

        if mode == "adaptive" and self.current_quant_assignments is not None:
            # Per-client assignment from PPO env
            all_clients = client_manager.all()   # Dict[str, ClientProxy]
            instructions = []
            round_assignment: Dict[str, int] = {}
            proxy_map: Dict[str, str] = {}
            for cid_str, bits in self.current_quant_assignments.items():
                if cid_str in all_clients:
                    config: Dict[str, Scalar] = {
                        "quant_bits": bits,
                        "round": server_round,
                    }
                    proxy = all_clients[cid_str]
                    instructions.append((proxy, FitIns(parameters, config)))
                    round_assignment[str(cid_str)] = int(bits)
                    proxy_map[str(proxy.cid)] = str(cid_str)
            if not instructions:
                log.warning(
                    f"[Strategy] round={server_round} adaptive mode: "
                    f"no valid client CIDs in current_quant_assignments. "
                    f"Falling back to all clients FP32."
                )
                # Safety fallback: select all clients FP32
                for cid_str, proxy in all_clients.items():
                    instructions.append(
                        (proxy, FitIns(parameters, {"quant_bits": 32, "round": server_round}))
                    )
                    round_assignment[str(cid_str)] = 32
                    proxy_map[str(proxy.cid)] = str(cid_str)
            self._last_fit_quant_assignment = round_assignment
            self._last_fit_proxy_map = proxy_map
            log.info(
                f"[Strategy] round={server_round} ADAPTIVE fit: "
                f"selected={list(self.current_quant_assignments.keys())} "
                f"bits={list(self.current_quant_assignments.values())}"
            )
            return instructions

        else:
            # Fixed mode: use FedAvg selection + uniform bits
            client_instructions = super().configure_fit(
                server_round, parameters, client_manager
            )
            quant_bits = self._get_quant_bits(server_round)
            updated = []
            round_assignment: Dict[str, int] = {}
            proxy_map: Dict[str, str] = {}
            used_ids: set[int] = set()
            for client_proxy, fit_ins in client_instructions:
                config = dict(fit_ins.config)
                config["quant_bits"] = quant_bits
                config["round"] = server_round
                updated.append((client_proxy, FitIns(fit_ins.parameters, config)))
                proxy_cid = str(client_proxy.cid)
                logical_id = self._resolve_logical_client_id_for_fit(
                    proxy_cid=proxy_cid,
                    client_proxy=client_proxy,
                    used_ids=used_ids,
                )
                used_ids.add(logical_id)
                logical_cid = str(logical_id)
                round_assignment[logical_cid] = int(quant_bits)
                proxy_map[proxy_cid] = logical_cid
            self._last_fit_quant_assignment = round_assignment
            self._last_fit_proxy_map = proxy_map

            cids = [str(cp.cid) for cp, _ in updated]
            log.info(
                f"[Strategy] round={server_round} FIXED fit: "
                f"clients={cids} quant_bits={quant_bits}"
            )
            return updated

    # ── aggregate_fit ──────────────────────────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Standard FedAvg aggregation + server-side eval + JSON log.
        Sets self.last_round_log for run_one_round() to retrieve.
        """
        if not results:
            log.warning(f"[Strategy] round={server_round}: no fit results")
            self.last_round_log = {}
            return None, {}

        # 1. FedAvg aggregation
        aggregated_params, _ = super().aggregate_fit(server_round, results, failures)
        if aggregated_params is None:
            self.last_round_log = {}
            return None, {}

        # Keep BN running statistics/counters from previous global state when disabled.
        params_ndarrays = parameters_to_ndarrays(aggregated_params)
        if not self.cfg.fl.aggregate_bn_buffers:
            if len(params_ndarrays) == len(self._state_keys) == len(self._global_params_nd):
                for idx, name in enumerate(self._state_keys):
                    if (
                        name.endswith("running_mean")
                        or name.endswith("running_var")
                        or name.endswith("num_batches_tracked")
                    ):
                        params_ndarrays[idx] = self._global_params_nd[idx]
            else:
                log.warning(
                    "[Strategy] aggregate_bn_buffers=False but state length mismatch; "
                    "falling back to aggregated tensors for this round."
                )
        aggregated_params = ndarrays_to_parameters(params_ndarrays)

        # 2. Per-client metrics
        selected_clients: List[str] = []
        quant_assignments: Dict[str, int] = {}
        actual_quant_methods: Dict[str, str] = {}
        quant_precision_requested: Dict[str, str] = {}
        quant_precision_actual: Dict[str, str] = {}
        train_losses: List[float] = []
        selected_ids_int: set[int] = set()
        failed_ids_int: set[int] = set()
        used_ids: set[int] = set()
        # Use a set to deduplicate dropout clients
        dropout_set: set = set()
        fit_failures_log: List[Dict[str, Optional[str]]] = []

        for client_proxy, fit_res in results:
            proxy_cid = str(client_proxy.cid)
            m = fit_res.metrics or {}
            cid_int = self._resolve_logical_client_id(
                proxy_cid=proxy_cid,
                metrics=m,
                used_ids=used_ids,
            )
            used_ids.add(cid_int)
            selected_ids_int.add(cid_int)
            cid = str(cid_int)
            selected_clients.append(cid)

            bits = int(m.get("quant_bits_requested", 32))
            method = str(m.get("quant_method_actual", "fp32"))
            requested_precision_raw = m.get("quant_precision_requested")
            if requested_precision_raw is None:
                if bits == 32:
                    requested_precision = "fp32"
                elif bits == 16:
                    requested_precision = str(self.cfg.quantization.lowp_dtype)
                elif bits == 8:
                    requested_precision = "int8"
                else:
                    requested_precision = "unknown"
            else:
                requested_precision = str(requested_precision_raw)
            actual_precision_raw = m.get("quant_precision_actual")
            if actual_precision_raw is None:
                actual_precision = precision_from_quant_method(method)
                if actual_precision == "unknown":
                    actual_precision = "fp32"
            else:
                actual_precision = str(actual_precision_raw)
            dropped = int(m.get("dropped", 0))

            quant_assignments[cid] = bits
            actual_quant_methods[cid] = method
            quant_precision_requested[cid] = requested_precision
            quant_precision_actual[cid] = actual_precision

            loss_val = m.get("train_loss")
            if (loss_val is not None
                    and not (isinstance(loss_val, float) and loss_val != loss_val)
                    and float(loss_val) >= 0.0):
                train_losses.append(float(loss_val))

            if dropped:
                dropout_set.add(cid)
                self.dropout_tracker.record(cid_int, dropped=True)
            else:
                self.dropout_tracker.record(cid_int, dropped=False)

        # Framework-level failures (deduplicated into set)
        for f in failures:
            if isinstance(f, tuple) and hasattr(f[0], "cid"):
                proxy_cid = str(f[0].cid)
                maybe_metrics = None
                if isinstance(f[1], FitRes):
                    maybe_metrics = f[1].metrics
                cid_int = self._resolve_logical_client_id(
                    proxy_cid=proxy_cid,
                    metrics=maybe_metrics,
                    used_ids=used_ids,
                )
                used_ids.add(cid_int)
                failed_ids_int.add(cid_int)
                dropout_set.add(str(cid_int))
                self.dropout_tracker.record(cid_int, dropped=True)

                err_type = type(f[1]).__name__
                err_msg = str(f[1])
                if isinstance(f[1], FitRes):
                    err_type = "FitResFailure"
                    err_msg = "FitRes returned in failures list"
                fit_failures_log.append(
                    {
                        "client_id": str(cid_int),
                        "stage": "fit",
                        "error_type": err_type,
                        "error_message": err_msg[:500],
                    }
                )
            elif isinstance(f, BaseException):
                fit_failures_log.append(
                    {
                        "client_id": None,
                        "stage": "fit",
                        "error_type": type(f).__name__,
                        "error_message": str(f)[:500],
                    }
                )

        # Record skipped clients
        all_cids = set(range(self.cfg.clients.count))
        attempted_ids = selected_ids_int | failed_ids_int
        for cid_int in all_cids - attempted_ids:
            self.dropout_tracker.record_not_selected(cid_int)

        dropout_client_ids = sorted(dropout_set)
        dropout_fraction = len(dropout_set) / max(1, len(attempted_ids))

        # 3. Global model update + (optional) BN recalibration + server evaluation
        set_parameters(self.global_model, params_ndarrays)
        if not self.cfg.fl.aggregate_bn_buffers:
            n_bn, n_batches = self._recalibrate_batchnorm(
                self.global_model,
                self.server_test_loader,
                device=torch.device("cpu"),
            )
            log.info(
                f"[Strategy] round={server_round} BN recalibration: "
                f"bn_layers={n_bn} batches={n_batches}"
            )
            # Push recalibrated BN buffers into next-round broadcast parameters.
            params_ndarrays = get_parameters(self.global_model)
            aggregated_params = ndarrays_to_parameters(params_ndarrays)
        self._global_params_nd = [arr.copy() for arr in params_ndarrays]
        self.global_model.eval()
        eval_result = evaluate(
            self.global_model, self.server_test_loader,
            device=torch.device("cpu"),
        )
        global_acc = float(eval_result.accuracy)
        global_loss = float(eval_result.loss)

        accuracy_delta = global_acc - (
            self._prev_accuracy if self._prev_accuracy is not None else global_acc
        )
        self._prev_accuracy = global_acc

        log.info(
            f"[Strategy] round={server_round} global_acc={global_acc:.4f} "
            f"delta={accuracy_delta:+.4f} dropout_frac={dropout_fraction:.2f}"
        )

        # 4. Build + write JSON log
        mean_loss = (
            float(sum(train_losses) / len(train_losses))
            if train_losses else None
        )
        fit_assignment_used = self._last_fit_quant_assignment
        if self.cfg.quantization.mode == "mixed" and quant_assignments:
            # Client metrics are the source of truth for effective quant bits used.
            fit_assignment_used = dict(quant_assignments)
            log.info(
                f"[Strategy] round={server_round} mixed actual quant (from clients): "
                f"{fit_assignment_used}"
            )
        round_log = {
            "round": server_round,
            "selected_clients": selected_clients,
            "fit_quant_assignment_used": fit_assignment_used,
            "fit_proxy_to_client_id": self._last_fit_proxy_map,
            "quant_assignments": quant_assignments,
            "actual_quant_method": actual_quant_methods,
            "quant_precision_requested": quant_precision_requested,
            "quant_precision_actual": quant_precision_actual,
            "dropout_clients": dropout_client_ids,
            "dropout_fraction": dropout_fraction,
            "mean_train_loss": mean_loss,
            "global_accuracy": global_acc,
            "global_loss": global_loss,
            "accuracy_delta": accuracy_delta,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if fit_failures_log:
            round_log["fit_failures"] = fit_failures_log
        log_path = self.output_dir / f"round_{server_round:03d}.json"
        with open(log_path, "w") as f:
            json.dump(round_log, f, indent=2)
        log.info(f"[Strategy] Written log: {log_path}")

        # 5. Store for run_one_round() retrieval
        self.last_round_log = round_log

        return aggregated_params, {
            "global_accuracy": global_acc,
            "accuracy_delta": accuracy_delta,
        }

    # ── configure_evaluate ─────────────────────────────────────────────────────

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Evaluate ALL registered clients (standard FedAvg behaviour)."""
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate client evaluate results."""
        if not results:
            return None, {}
        total = sum(r.num_examples for _, r in results)
        acc = (
            sum(r.num_examples * (r.metrics or {}).get("accuracy", 0.0) for _, r in results)
            / max(1, total)
        )
        log.info(f"[Strategy] round={server_round} client_agg_accuracy={acc:.4f}")
        return super().aggregate_evaluate(server_round, results, failures)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_quant_bits(self, server_round: int) -> int:
        """Return uniform quant bits from cfg for fixed-mode rounds."""
        mode = self.cfg.quantization.mode
        if mode == "fixed_fp32":
            return 32
        elif mode == "fixed_fp16":
            return 16
        elif mode == "fixed_int8":
            return 8
        else:
            return self.cfg.quantization.fixed_bits

    def _recalibrate_batchnorm(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> Tuple[int, int]:
        """
        Recompute BN running stats using forward-only passes over server data.

        This is used when aggregate_bn_buffers=False to avoid stale BN buffers
        after excluding cross-client BN-stat averaging.
        """
        bn_layers = [
            m for m in model.modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        ]
        if not bn_layers:
            return 0, 0

        model.eval()
        for bn in bn_layers:
            bn.reset_running_stats()
            bn.train()

        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32

        n_batches = 0
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device=device, dtype=model_dtype)
                model(images)
                n_batches += 1

        model.eval()
        return len(bn_layers), n_batches

    def _parse_logical_cid(self, raw_cid: object) -> Optional[int]:
        """Parse and validate a logical client ID in [0, n_clients)."""
        try:
            cid_int = int(raw_cid)
        except (TypeError, ValueError):
            return None
        if 0 <= cid_int < self.cfg.clients.count:
            return cid_int
        return None

    def _resolve_logical_client_id(
        self,
        proxy_cid: str,
        metrics: Optional[Dict[str, Scalar]],
        used_ids: set[int],
    ) -> int:
        """
        Resolve logical client ID for this round using a safe fallback chain.

        Priority:
          1) fit_res.metrics["client_id"]
          2) cached proxy_cid -> logical_id map from prior rounds
          3) numeric proxy_cid (when not opaque gRPC UUID)
          4) first unused logical ID in [0, n_clients-1], then 0 as final fallback
        """
        if metrics is not None:
            metric_cid = self._parse_logical_cid(metrics.get("client_id"))
            if metric_cid is not None:
                self._proxy_to_logical[proxy_cid] = metric_cid
                return metric_cid

        cached_cid = self._proxy_to_logical.get(proxy_cid)
        if cached_cid is not None and 0 <= cached_cid < self.cfg.clients.count:
            return cached_cid

        parsed_proxy_cid = self._parse_logical_cid(proxy_cid)
        if parsed_proxy_cid is not None:
            self._proxy_to_logical[proxy_cid] = parsed_proxy_cid
            return parsed_proxy_cid

        for cid_int in range(self.cfg.clients.count):
            if cid_int not in used_ids:
                self._proxy_to_logical[proxy_cid] = cid_int
                log.warning(
                    f"[Strategy] Could not resolve logical client_id from proxy cid '{proxy_cid}'. "
                    f"Assigned fallback logical id={cid_int}."
                )
                return cid_int

        log.warning(
            f"[Strategy] Exhausted logical client IDs while resolving proxy cid '{proxy_cid}'. "
            "Falling back to logical id=0."
        )
        return 0

    def _effective_mixed_assignments(self) -> Dict[str, int]:
        """
        Build mixed quantization mapping for this round (logical client id -> bits).

        Priority:
          1) cfg.quantization.per_client (canonical config)
          2) current_quant_assignments if provided and parseable
        """
        assignments: Dict[str, int] = {}

        cfg_map = self.cfg.quantization.per_client or {}
        for raw_cid, raw_bits in cfg_map.items():
            cid_int = self._parse_logical_cid(raw_cid)
            if cid_int is None:
                continue
            assignments[str(cid_int)] = int(raw_bits)

        if assignments:
            return assignments

        runtime_map = self.current_quant_assignments or {}
        for raw_cid, raw_bits in runtime_map.items():
            cid_int = self._parse_logical_cid(raw_cid)
            if cid_int is None:
                continue
            assignments[str(cid_int)] = int(raw_bits)
        return assignments

    def _resolve_logical_client_id_for_fit(
        self,
        proxy_cid: str,
        client_proxy: ClientProxy,
        used_ids: set[int],
    ) -> int:
        """
        Resolve logical ID before fit metrics exist.

        Priority:
          1) cached proxy_cid -> logical_id
          2) numeric proxy_cid
          3) client_proxy.get_properties(...)[\"client_id\"]
          4) first unused logical ID in [0, n_clients-1], then 0
        """
        cached_cid = self._proxy_to_logical.get(proxy_cid)
        if cached_cid is not None and 0 <= cached_cid < self.cfg.clients.count:
            return cached_cid

        parsed_proxy_cid = self._parse_logical_cid(proxy_cid)
        if parsed_proxy_cid is not None:
            self._proxy_to_logical[proxy_cid] = parsed_proxy_cid
            return parsed_proxy_cid

        try:
            props_res = client_proxy.get_properties(
                GetPropertiesIns(config={}),
                timeout=5.0,
                group_id=None,
            )
            if props_res is not None:
                props = getattr(props_res, "properties", {}) or {}
                prop_cid = self._parse_logical_cid(props.get("client_id"))
                if prop_cid is not None:
                    self._proxy_to_logical[proxy_cid] = prop_cid
                    return prop_cid
        except Exception:
            # Some execution paths (e.g., in-process mocks) do not implement
            # get_properties; fall through to deterministic fallback.
            pass

        for cid_int in range(self.cfg.clients.count):
            if cid_int not in used_ids:
                self._proxy_to_logical[proxy_cid] = cid_int
                log.warning(
                    f"[Strategy] Fit-time logical ID fallback for proxy cid '{proxy_cid}' -> {cid_int}"
                )
                return cid_int

        log.warning(
            f"[Strategy] Exhausted fit-time logical IDs for proxy cid '{proxy_cid}'. "
            "Falling back to logical id=0."
        )
        return 0
