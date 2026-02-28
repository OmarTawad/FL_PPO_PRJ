"""
src/fl/strategy.py — FedAvgQuant: custom Flower strategy

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Extends FedAvg with:
  1. configure_fit(): assigns per-client quant_bits via FitIns.config
     (Phase 6 smoke: fixed_fp32=32 for all clients from cfg)
  2. aggregate_fit(): standard FedAvg on FP32 weights + per-round JSON log
  3. Global model evaluation on server-held test set after each aggregation
  4. DropoutTracker integration (tracks per-client dropout history)

Per-round JSON schema (SPEC.md §8):
  {
    "round": int,
    "selected_clients": [str, ...],
    "quant_assignments": {cid: bits},
    "actual_quant_method": {cid: method_str},
    "dropout_clients": [str, ...],
    "dropout_fraction": float,
    "mean_train_loss": float | null,
    "global_accuracy": float,
    "accuracy_delta": float,
    "timestamp_utc": str
  }
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
    Custom FedAvg strategy with:
      - Per-client quantization assignment in configure_fit()
      - Per-round JSON logging
      - Global model evaluation on server test set
      - DropoutTracker integration
    """

    def __init__(
        self,
        server_test_loader: DataLoader,
        dropout_tracker: DropoutTracker,
        cfg: Config,
        output_dir: Path,
        initial_parameters: Parameters,
    ):
        # Phase 6 smoke: select ALL clients every round
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

        # Track previous round accuracy for delta computation
        self._prev_accuracy: Optional[float] = None

    # ── configure_fit ──────────────────────────────────────────────────────────

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Select clients and build FitIns with per-client quant_bits config.

        Phase 6 smoke: assign fixed_bits from cfg.quantization to all clients.
        Phase 8 (PPO): replace with PPO action → per-client bits assignment.
        """
        # Get standard FedAvg client selection + parameters
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Determine quant_bits for this round
        quant_bits = self._get_quant_bits(server_round)

        # Inject quant_bits + round into every client's config
        updated_instructions = []
        for client_proxy, fit_ins in client_instructions:
            config = dict(fit_ins.config)
            config["quant_bits"] = quant_bits
            config["round"] = server_round
            updated_instructions.append(
                (client_proxy, FitIns(fit_ins.parameters, config))
            )

        cids = [str(cp.cid) for cp, _ in updated_instructions]
        log.info(
            f"[Strategy] round={server_round} configure_fit: "
            f"clients={cids} quant_bits={quant_bits}"
        )
        return updated_instructions

    # ── aggregate_fit ──────────────────────────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results (FedAvg), evaluate global model, write JSON log.
        """
        if not results:
            log.warning(f"[Strategy] round={server_round}: no fit results — skipping aggregation")
            return None, {}

        # ── 1. Standard FedAvg aggregation ────────────────────────────────────
        aggregated_params, _ = super().aggregate_fit(server_round, results, failures)
        if aggregated_params is None:
            return None, {}

        # ── 2. Extract per-client metrics ─────────────────────────────────────
        selected_clients: List[str] = []
        quant_assignments: Dict[str, int] = {}
        actual_quant_methods: Dict[str, str] = {}
        train_losses: List[float] = []
        dropout_client_ids: List[str] = []

        for client_proxy, fit_res in results:
            cid = str(client_proxy.cid)
            selected_clients.append(cid)
            m = fit_res.metrics or {}
            bits = int(m.get("quant_bits_requested", 32))
            method = str(m.get("quant_method_actual", "fp32"))
            dropped = int(m.get("dropped", 0))

            quant_assignments[cid] = bits
            actual_quant_methods[cid] = method

            loss = m.get("train_loss")
            if loss is not None and not (isinstance(loss, float) and loss != loss):  # not NaN
                train_losses.append(float(loss))

            cid_int = int(cid)
            if dropped:
                dropout_client_ids.append(cid)
                self.dropout_tracker.record(cid_int, dropped=True)
            else:
                self.dropout_tracker.record(cid_int, dropped=False)

        # Record skipped clients (not in results)
        all_cids = set(range(self.cfg.clients.count))
        result_cids = {int(c) for c in selected_clients}
        for cid_int in all_cids - result_cids:
            self.dropout_tracker.record_not_selected(cid_int)

        # Failures from the framework (connection errors, timeouts, etc.)
        for f in failures:
            if isinstance(f, tuple) and hasattr(f[0], 'cid'):
                dropout_client_ids.append(str(f[0].cid))

        n_total = len(selected_clients) + len(failures)
        dropout_fraction = len(dropout_client_ids) / max(1, n_total)

        # ── 3. Evaluate global model on server test set ────────────────────────
        params_ndarrays = parameters_to_ndarrays(aggregated_params)
        set_parameters(self.global_model, params_ndarrays)
        self.global_model.eval()

        eval_result = evaluate(self.global_model, self.server_test_loader, device=torch.device("cpu"))
        global_acc = float(eval_result.accuracy)
        global_loss = float(eval_result.loss)

        accuracy_delta = global_acc - (self._prev_accuracy if self._prev_accuracy is not None else global_acc)
        self._prev_accuracy = global_acc

        log.info(
            f"[Strategy] round={server_round} global_acc={global_acc:.4f} "
            f"delta={accuracy_delta:+.4f} dropout={dropout_fraction:.2f}"
        )

        # ── 4. Write per-round JSON log ────────────────────────────────────────
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
        """Use standard FedAvg evaluate configuration."""
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate client evaluate results (weighted average accuracy)."""
        if not results:
            return None, {}
        total = sum(r.num_examples for _, r in results)
        acc = sum(
            r.num_examples * (r.metrics or {}).get("accuracy", 0.0)
            for _, r in results
        ) / max(1, total)
        log.info(f"[Strategy] round={server_round} client_agg_accuracy={acc:.4f}")
        return super().aggregate_evaluate(server_round, results, failures)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_quant_bits(self, server_round: int) -> int:
        """
        Determine quant_bits to assign all clients this round.
        Phase 6: fixed from cfg.quantization.
        Phase 8 (PPO): replaced by PPO action selection.
        """
        mode = self.cfg.quantization.mode
        if mode == "fixed_fp32":
            return 32
        elif mode == "fixed_fp16":
            return 16
        elif mode == "fixed_int8":
            return 8
        elif mode == "adaptive":
            return 32   # placeholder until PPO implemented
        else:
            return self.cfg.quantization.fixed_bits
