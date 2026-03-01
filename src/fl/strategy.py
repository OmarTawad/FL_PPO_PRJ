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
    NDArrays, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

from src.common.config import Config
from src.models.mobilenetv2 import get_model, set_parameters
from src.models.trainer import evaluate
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
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=cfg.clients.count,
            min_evaluate_clients=cfg.clients.count,
            min_available_clients=cfg.clients.count,
            initial_parameters=initial_parameters,
        )
        self.server_test_loader = server_test_loader
        self.dropout_tracker = dropout_tracker
        self.cfg = cfg
        self.output_dir = output_dir

        # Server-side global model for evaluation
        self.global_model = get_model()

        # State reset between env episodes
        self._prev_accuracy: Optional[float] = None
        self.last_round_log: dict = {}

        # Phase 7: PPO sets this before each env.step()
        # Dict[str, int] mapping cid → quant_bits for selected clients only.
        # None in Phase 6 (fixed) mode.
        self.current_quant_assignments: Optional[Dict[str, int]] = None

    # ── Public: env.reset() ────────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Reset per-episode state. Called by FLEnv.reset()."""
        self._prev_accuracy = None
        self.last_round_log = {}
        self.current_quant_assignments = None

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

        Fixed mode (Phase 6):
            All clients selected; uniform bits from cfg.quantization.
        """
        mode = self.cfg.quantization.mode

        if mode == "adaptive" and self.current_quant_assignments is not None:
            # Per-client assignment from PPO env
            all_clients = client_manager.all()   # Dict[str, ClientProxy]
            instructions = []
            for cid_str, bits in self.current_quant_assignments.items():
                if cid_str in all_clients:
                    config: Dict[str, Scalar] = {
                        "quant_bits": bits,
                        "round": server_round,
                    }
                    instructions.append(
                        (all_clients[cid_str], FitIns(parameters, config))
                    )
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
            for client_proxy, fit_ins in client_instructions:
                config = dict(fit_ins.config)
                config["quant_bits"] = quant_bits
                config["round"] = server_round
                updated.append((client_proxy, FitIns(fit_ins.parameters, config)))

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

        # 2. Per-client metrics
        selected_clients: List[str] = []
        quant_assignments: Dict[str, int] = {}
        actual_quant_methods: Dict[str, str] = {}
        train_losses: List[float] = []
        # Use a set to deduplicate dropout clients
        dropout_set: set = set()

        for client_proxy, fit_res in results:
            cid = str(client_proxy.cid)
            selected_clients.append(cid)
            m = fit_res.metrics or {}

            bits = int(m.get("quant_bits_requested", 32))
            method = str(m.get("quant_method_actual", "fp32"))
            dropped = int(m.get("dropped", 0))

            quant_assignments[cid] = bits
            actual_quant_methods[cid] = method

            loss_val = m.get("train_loss")
            if (loss_val is not None
                    and not (isinstance(loss_val, float) and loss_val != loss_val)
                    and float(loss_val) >= 0.0):
                train_losses.append(float(loss_val))

            cid_int = int(cid)
            if dropped:
                dropout_set.add(cid)
                self.dropout_tracker.record(cid_int, dropped=True)
            else:
                self.dropout_tracker.record(cid_int, dropped=False)

        # Record skipped clients
        all_cids = set(range(self.cfg.clients.count))
        result_cids = {int(c) for c in selected_clients}
        for cid_int in all_cids - result_cids:
            self.dropout_tracker.record_not_selected(cid_int)

        # Framework-level failures (deduplicated into set)
        for f in failures:
            if isinstance(f, tuple) and hasattr(f[0], "cid"):
                dropout_set.add(str(f[0].cid))

        dropout_client_ids = sorted(dropout_set)
        n_total = len(selected_clients) + len(failures)
        dropout_fraction = len(dropout_set & set(selected_clients)) / max(1, len(selected_clients))

        # 3. Global model evaluation on server test set
        params_ndarrays = parameters_to_ndarrays(aggregated_params)
        set_parameters(self.global_model, params_ndarrays)
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
        round_log = {
            "round": server_round,
            "selected_clients": selected_clients,
            "quant_assignments": quant_assignments,
            "actual_quant_method": actual_quant_methods,
            "dropout_clients": dropout_client_ids,
            "dropout_fraction": dropout_fraction,
            "mean_train_loss": mean_loss,
            "global_accuracy": global_acc,
            "global_loss": global_loss,
            "accuracy_delta": accuracy_delta,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
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
