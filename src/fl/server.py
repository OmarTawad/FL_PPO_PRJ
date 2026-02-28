"""
src/fl/server.py — Adaptive FL simulation entrypoint (no Ray required)

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Usage:
    python src/fl/server.py --config configs/exp1_smoke.yaml

Runs a lightweight in-process FL simulation (no Ray, no sockets).
The simulation loop follows the standard Flower protocol:
  For each round t:
    1. strategy.configure_fit() → per-client FitIns
    2. All clients run fit() locally (sequential, CPU-only VM)
    3. strategy.aggregate_fit() → aggregated parameters + per-round JSON log
    4. strategy.configure_evaluate() → per-client EvaluateIns
    5. All clients run evaluate() locally
    6. strategy.aggregate_evaluate() → aggregated metrics

This avoids the Ray dependency required by fl.simulation.start_simulation().
Compatible with flwr 1.7.0 on CPU-only VM.

Outputs:
    outputs/metrics/run_<YYYYMMDD_HHMMSS>/round_<t>.json
    outputs/metrics/run_<YYYYMMDD_HHMMSS>/run_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import flwr as fl
from flwr.common import (
    FitIns, EvaluateIns, FitRes, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
    GetParametersIns,
)

from src.common.config import load_config, Config
from src.models.mobilenetv2 import get_model, get_parameters
from src.data.cifar import get_server_test_loader, get_cifar10_train, get_cifar10_test
from src.data.partitioner import build_partitions
from src.heterogeneity.dropout import DropoutTracker
from src.fl.client import FlowerClient
from src.fl.strategy import FedAvgQuant

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("flwr").setLevel(logging.WARNING)


# ─── Lightweight simulation loop ──────────────────────────────────────────────

class _MockClientProxy(fl.server.client_proxy.ClientProxy):
    """Minimal proxy that calls a FlowerClient directly (no network)."""

    def __init__(self, cid: str, client: fl.client.Client):
        super().__init__(cid)
        self._client = client

    def get_properties(self, ins, timeout, group_id):
        raise NotImplementedError

    def get_parameters(self, ins, timeout, group_id):
        res = self._client.get_parameters(ins)
        return res

    def fit(self, ins, timeout, group_id):
        return self._client.fit(ins)

    def evaluate(self, ins, timeout, group_id):
        return self._client.evaluate(ins)

    def reconnect(self, ins, timeout, group_id):
        raise NotImplementedError


def _run_simulation(
    client_fn: Callable[[str], fl.client.Client],
    n_clients: int,
    n_rounds: int,
    strategy: FedAvgQuant,
) -> None:
    """
    In-process FL simulation loop. No Ray, no sockets.

    For each round:
      configure_fit → client.fit → aggregate_fit
      configure_evaluate → client.evaluate → aggregate_evaluate
    """
    # Build client instances (one per client, reused across rounds)
    clients = {str(i): client_fn(str(i)) for i in range(n_clients)}
    proxies = {cid: _MockClientProxy(cid, c) for cid, c in clients.items()}

    from flwr.server.client_manager import SimpleClientManager
    client_manager = SimpleClientManager()
    for proxy in proxies.values():
        client_manager.register(proxy)

    # Initial parameters from strategy
    parameters = strategy.initial_parameters  # already set as FedAvg attribute

    for server_round in range(1, n_rounds + 1):
        log.info(f"\n{'─'*50}\n  ROUND {server_round}/{n_rounds}\n{'─'*50}")

        # ── Fit phase ─────────────────────────────────────────────────────────
        fit_instructions = strategy.configure_fit(
            server_round, parameters, client_manager
        )

        fit_results = []
        fit_failures = []
        for proxy, fit_ins in fit_instructions:
            try:
                fit_res = proxy.fit(fit_ins, timeout=None, group_id=None)
                fit_results.append((proxy, fit_res))
            except Exception as e:
                log.warning(f"Client {proxy.cid} fit failed: {e}")
                fit_failures.append((proxy, e))

        # ── Aggregate fit ─────────────────────────────────────────────────────
        agg_params, agg_metrics = strategy.aggregate_fit(
            server_round, fit_results, fit_failures
        )
        if agg_params is not None:
            parameters = agg_params
            log.info(f"  aggregate_fit metrics: {agg_metrics}")

        # ── Evaluate phase ─────────────────────────────────────────────────────
        eval_instructions = strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

        eval_results = []
        eval_failures = []
        for proxy, eval_ins in eval_instructions:
            try:
                eval_res = proxy.evaluate(eval_ins, timeout=None, group_id=None)
                eval_results.append((proxy, eval_res))
            except Exception as e:
                log.warning(f"Client {proxy.cid} evaluate failed: {e}")
                eval_failures.append((proxy, e))

        # ── Aggregate evaluate ─────────────────────────────────────────────────
        agg_loss, eval_agg_metrics = strategy.aggregate_evaluate(
            server_round, eval_results, eval_failures
        )
        log.info(f"  aggregate_evaluate loss={agg_loss} metrics={eval_agg_metrics}")

    log.info("\nSimulation complete.")


# ─── Setup helpers ────────────────────────────────────────────────────────────

def _make_server_test_loader(cfg: Config):
    return get_server_test_loader(
        root="data/",
        batch_size=2,
        server_fraction=0.10,
        seed=cfg.experiment.seed,
        download=True,
        use_32=False,
        num_workers=cfg.fl.num_workers,
        pin_memory=cfg.fl.pin_memory,
    )


def _make_partitions(cfg: Config):
    train_ds = get_cifar10_train(root="data/", download=True, use_32=False)
    data_fractions = [p.data_fraction for p in cfg.clients.profiles]
    train_partitions = build_partitions(
        dataset=train_ds,
        n_clients=cfg.clients.count,
        partition_mode=cfg.data.partition,
        data_fractions=data_fractions,
        dirichlet_alpha=cfg.data.dirichlet_alpha,
        reduced_fraction=cfg.data.reduced_fraction,
        seed=cfg.experiment.seed,
    )
    test_ds = get_cifar10_test(root="data/", download=False, use_32=False)
    n_test = len(test_ds)
    n_per = max(1, n_test // cfg.clients.count)
    eval_partitions = [
        list(range(i * n_per, min((i + 1) * n_per, n_test)))
        for i in range(cfg.clients.count)
    ]
    return train_partitions, eval_partitions


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(config_path: str) -> None:
    cfg = load_config(config_path)
    log.info(f"Config: {cfg.experiment.name}  rounds={cfg.experiment.rounds}  clients={cfg.n_clients}")

    # 1. Output directory
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.logging.metrics_dir) / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.plots_dir).mkdir(parents=True, exist_ok=True)
    Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir: {output_dir}")

    # 2. Server test loader
    log.info("Building server test loader...")
    server_test_loader = _make_server_test_loader(cfg)
    log.info(f"Server test set: {len(server_test_loader.dataset)} samples")

    # 3. Partitions
    log.info("Building data partitions...")
    train_partitions, eval_partitions = _make_partitions(cfg)
    for i, p in enumerate(train_partitions):
        log.info(f"  Client {i}: {len(p)} train samples")

    # 4. Initial model parameters
    log.info("Initialising global model...")
    init_model = get_model()
    initial_parameters = ndarrays_to_parameters(get_parameters(init_model))

    # 5. Dropout tracker
    dropout_tracker = DropoutTracker(n_clients=cfg.clients.count)

    # 6. Strategy
    strategy = FedAvgQuant(
        server_test_loader=server_test_loader,
        dropout_tracker=dropout_tracker,
        cfg=cfg,
        output_dir=output_dir,
        initial_parameters=initial_parameters,
    )

    # 7. Client factory
    def client_fn(cid: str) -> fl.client.Client:
        cid_int = int(cid)
        profile = cfg.clients.profile_for(cid_int)
        return FlowerClient(
            client_id=cid_int,
            train_indices=train_partitions[cid_int],
            eval_indices=eval_partitions[cid_int],
            cfg=cfg,
            profile=profile,
            data_root="data/",
        ).to_client()

    # 8. Run in-process simulation
    log.info(f"Starting in-process simulation ({cfg.clients.count} clients, {cfg.experiment.rounds} rounds)")
    _run_simulation(
        client_fn=client_fn,
        n_clients=cfg.clients.count,
        n_rounds=cfg.experiment.rounds,
        strategy=strategy,
    )

    # 9. Summary + verification
    json_logs = sorted(output_dir.glob("round_*.json"))
    required_keys = {
        "round", "selected_clients", "quant_assignments",
        "actual_quant_method", "dropout_clients", "dropout_fraction",
        "global_accuracy", "accuracy_delta",
    }
    log.info(f"\n{'='*60}")
    log.info(f"Simulation complete. {len(json_logs)} round log(s) written:")
    for p in json_logs:
        with open(p) as f:
            d = json.load(f)
        missing = required_keys - set(d.keys())
        if missing:
            log.warning(f"  [INCOMPLETE] {p.name}: missing={missing}")
        else:
            log.info(
                f"  {p.name}: acc={d['global_accuracy']:.4f}  "
                f"delta={d['accuracy_delta']:+.4f}  "
                f"dropout={d['dropout_fraction']:.2f}"
            )

    summary = {
        "run_id": run_id, "config": config_path,
        "experiment": cfg.experiment.name,
        "rounds_completed": len(json_logs),
        "n_clients": cfg.clients.count,
        "quantization_mode": cfg.quantization.mode,
    }
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary: {output_dir / 'run_summary.json'}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
