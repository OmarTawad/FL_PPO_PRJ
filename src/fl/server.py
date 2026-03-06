"""
src/fl/server.py — Adaptive FL simulation entrypoint (no Ray required)

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Usage:
    python src/fl/server.py --config configs/exp1_smoke.yaml

Public interface for Phase 7 env:
    run_one_round(strategy, client_manager, parameters, server_round)
        → (new_parameters, round_log_dict)

Internal:
    _run_simulation(client_fn, n_clients, n_rounds, strategy) — full loop
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import flwr as fl
from flwr.common import (
    FitIns, FitRes, EvaluateIns, EvaluateRes, Parameters,
    ndarrays_to_parameters,
)
from flwr.server.client_manager import SimpleClientManager
from torch.utils.data import DataLoader

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


# ─── _MockClientProxy ─────────────────────────────────────────────────────────

class _MockClientProxy(fl.server.client_proxy.ClientProxy):
    """Minimal proxy that calls a FlowerClient directly (no network)."""

    def __init__(self, cid: str, client: fl.client.Client):
        super().__init__(cid)
        self._client = client

    def get_properties(self, ins, timeout, group_id):
        raise NotImplementedError

    def get_parameters(self, ins, timeout, group_id):
        return self._client.get_parameters(ins)

    def fit(self, ins, timeout, group_id):
        return self._client.fit(ins)

    def evaluate(self, ins, timeout, group_id):
        return self._client.evaluate(ins)

    def reconnect(self, ins, timeout, group_id):
        raise NotImplementedError


# ─── Public: run_one_round ────────────────────────────────────────────────────

def run_one_round(
    strategy: FedAvgQuant,
    client_manager: SimpleClientManager,
    parameters: Parameters,
    server_round: int,
) -> Tuple[Parameters, dict]:
    """
    Execute one FL round (fit + evaluate) in-process.

    Called by FLEnv.step() once per env step.

    Protocol:
        configure_fit → client.fit (all selected) → aggregate_fit
        configure_evaluate → client.evaluate (all) → aggregate_evaluate

    Returns:
        new_parameters:  Aggregated global parameters after this round.
        round_log_dict:  The per-round JSON log dict (from strategy.last_round_log)
                         with all SPEC.md §8 required keys.
    """
    # ── Fit phase ──────────────────────────────────────────────────────────────
    fit_instructions = strategy.configure_fit(server_round, parameters, client_manager)

    fit_results: List[Tuple] = []
    fit_failures: List = []
    for proxy, fit_ins in fit_instructions:
        try:
            fit_res = proxy.fit(fit_ins, timeout=None, group_id=None)
            fit_results.append((proxy, fit_res))
        except Exception as e:
            log.warning(f"Client {proxy.cid} fit failed: {type(e).__name__}: {e}")
            fit_failures.append((proxy, e))

    agg_params, _ = strategy.aggregate_fit(server_round, fit_results, fit_failures)
    new_params = agg_params if agg_params is not None else parameters

    # ── Evaluate phase ─────────────────────────────────────────────────────────
    eval_instructions = strategy.configure_evaluate(server_round, new_params, client_manager)

    eval_results: List[Tuple] = []
    eval_failures: List = []
    for proxy, eval_ins in eval_instructions:
        try:
            eval_res = proxy.evaluate(eval_ins, timeout=None, group_id=None)
            eval_results.append((proxy, eval_res))
        except Exception as e:
            log.warning(f"Client {proxy.cid} evaluate failed: {type(e).__name__}: {e}")
            eval_failures.append((proxy, e))

    strategy.aggregate_evaluate(server_round, eval_results, eval_failures)

    return new_params, strategy.last_round_log


# ─── Setup helpers ────────────────────────────────────────────────────────────

def make_server_test_loader(cfg: Config) -> DataLoader:
    """Build the server-side held-out evaluation DataLoader (10%, seed, 224×224)."""
    return get_server_test_loader(
        root="data/",
        batch_size=2,
        server_fraction=0.10,
        seed=cfg.experiment.seed,
        download=True,
        use_32=cfg.data.use_32,
        num_workers=cfg.fl.num_workers,
        pin_memory=cfg.fl.pin_memory,
    )


def make_partitions(cfg: Config):
    """
    Build per-client train + eval index lists.

    Returns:
        train_partitions: List[List[int]] — one per client
        eval_partitions:  List[List[int]] — one per client (capped by config)
    """
    train_ds = get_cifar10_train(
        root="data/",
        download=True,
        use_32=cfg.data.use_32,
        augment=False,
    )
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

    test_ds = get_cifar10_test(
        root="data/",
        download=False,
        use_32=cfg.data.use_32,
    )
    n_test = len(test_ds)
    n_per_base = max(1, int(n_test * cfg.data.eval_fraction))
    cap = cfg.data.max_eval_samples_per_client
    if cap > 0:
        n_per = min(n_per_base, cap)
    else:
        n_per = n_per_base

    eval_partitions = []
    for i in range(cfg.clients.count):
        start = i * (n_test // cfg.clients.count)
        eval_partitions.append(list(range(start, min(start + n_per, n_test))))

    return train_partitions, eval_partitions


def make_client_manager(
    cfg: Config,
    train_partitions: List[List[int]],
    eval_partitions: List[List[int]],
    data_root: str = "data/",
) -> Tuple[SimpleClientManager, List[FlowerClient]]:
    """
    Create FlowerClient instances and register _MockClientProxy in a SimpleClientManager.

    Returns:
        (client_manager, client_list)
    """
    client_manager = SimpleClientManager()
    raw_clients = []
    for i in range(cfg.clients.count):
        profile = cfg.clients.profile_for(i)
        fc = FlowerClient(
            client_id=i,
            train_indices=train_partitions[i],
            eval_indices=eval_partitions[i],
            cfg=cfg,
            profile=profile,
            data_root=data_root,
        )
        raw_clients.append(fc)
        proxy = _MockClientProxy(str(i), fc.to_client())
        client_manager.register(proxy)
    return client_manager, raw_clients


# ─── In-process simulation loop ───────────────────────────────────────────────

def _run_simulation(
    client_fn: Callable[[str], fl.client.Client],
    n_clients: int,
    n_rounds: int,
    strategy: FedAvgQuant,
) -> None:
    """Full multi-round simulation loop (used by server.py CLI)."""
    clients = {str(i): client_fn(str(i)) for i in range(n_clients)}
    client_manager = SimpleClientManager()
    for cid, c in clients.items():
        client_manager.register(_MockClientProxy(cid, c))

    parameters = strategy.initial_parameters

    for server_round in range(1, n_rounds + 1):
        log.info(f"\n{'─'*50}\n  ROUND {server_round}/{n_rounds}\n{'─'*50}")
        parameters, round_log = run_one_round(
            strategy, client_manager, parameters, server_round
        )
        log.info(
            f"  global_acc={round_log.get('global_accuracy', '?'):.4f} "
            f"delta={round_log.get('accuracy_delta', 0.0):+.4f}"
        )

    log.info("\nSimulation complete.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(config_path: str) -> None:
    cfg = load_config(config_path)
    log.info(f"Config: {cfg.experiment.name}  rounds={cfg.experiment.rounds}  clients={cfg.n_clients}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.logging.metrics_dir) / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.plots_dir).mkdir(parents=True, exist_ok=True)
    Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
    log.info(f"Output dir: {output_dir}")

    log.info("Building server test loader...")
    server_test_loader = make_server_test_loader(cfg)
    log.info(f"Server test set: {len(server_test_loader.dataset)} samples")

    log.info("Building data partitions...")
    train_partitions, eval_partitions = make_partitions(cfg)
    for i, p in enumerate(train_partitions):
        log.info(f"  Client {i}: {len(p)} train / {len(eval_partitions[i])} eval samples")

    log.info("Initialising global model...")
    init_model = get_model(freeze_features=cfg.fl.freeze_features)
    initial_parameters = ndarrays_to_parameters(get_parameters(init_model))

    dropout_tracker = DropoutTracker(n_clients=cfg.clients.count)

    strategy = FedAvgQuant(
        server_test_loader=server_test_loader,
        dropout_tracker=dropout_tracker,
        cfg=cfg,
        output_dir=output_dir,
        initial_parameters=initial_parameters,
    )

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

    log.info(f"Starting simulation ({cfg.clients.count} clients, {cfg.experiment.rounds} rounds)")
    _run_simulation(
        client_fn=client_fn,
        n_clients=cfg.clients.count,
        n_rounds=cfg.experiment.rounds,
        strategy=strategy,
    )

    json_logs = sorted(output_dir.glob("round_*.json"))
    required_keys = {
        "round", "selected_clients", "quant_assignments",
        "actual_quant_method", "dropout_clients", "dropout_fraction",
        "global_accuracy", "accuracy_delta",
    }
    log.info(f"\n{'='*60}")
    log.info(f"Complete. {len(json_logs)} log(s) at {output_dir}:")
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
    log.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
