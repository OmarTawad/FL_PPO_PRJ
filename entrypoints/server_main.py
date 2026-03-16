"""
server_main.py — Flower gRPC server entrypoint for Docker deployment
Stage 13: True multi-container FL server using fl.server.start_server()

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Environment variables:
    CONFIG_PATH      Path to YAML config (default: configs/exp_docker_smoke.yaml)
    PARTITION_FILE   Path to pre-computed partition JSON (default: data/partitions_50_iid.json)
    SERVER_PORT      gRPC port (default: 8080)
    NUM_ROUNDS       Override experiment rounds (optional; config value used if not set)
    OUTPUTS_DIR      Base output directory (default: outputs)

Usage (non-Docker):
    CONFIG_PATH=configs/exp_docker_smoke.yaml \\
    PARTITION_FILE=data/partitions_50_iid.json \\
    python3 server_main.py

Usage (Docker — via Dockerfile.server):
    docker run --env CONFIG_PATH=... fl_server

Mode selection (from config.quantization.mode):
    fixed_fp32 / fixed_fp16 / fixed_int8 / mixed:
        Server uses FedAvgQuant in fixed/mixed mode.
    adaptive:
        Server still runs Flower gRPC, but strategy-side PPO controls
        per-round client selection and client policy at runtime.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig
from torch.utils.data import DataLoader

from src.common.config import load_config, Config
from src.models.mobilenetv2 import get_model, get_parameters
from src.data.cifar import get_server_test_loader
from src.heterogeneity.dropout import DropoutTracker
from src.fl.strategy import FedAvgQuant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server_main")
logging.getLogger("flwr").setLevel(logging.WARNING)


# ── Environment variables ─────────────────────────────────────────────────────

CONFIG_PATH    = os.environ.get("CONFIG_PATH",    "configs/exp_docker_smoke.yaml")
PARTITION_FILE = os.environ.get("PARTITION_FILE", "data/partitions_50_iid.json")
SERVER_PORT    = int(os.environ.get("SERVER_PORT", "8080"))
OUTPUTS_DIR    = os.environ.get("OUTPUTS_DIR",    "outputs")
NUM_ROUNDS_ENV = os.environ.get("NUM_ROUNDS",     None)   # optional override


# ── Utilities ─────────────────────────────────────────────────────────────────

def _load_partition_file(path: str) -> dict:
    """Load and return the pre-computed partition JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Partition file not found: {path}\n"
            f"Run: python3 scripts/download_and_partition.py --n-clients <N>"
        )
    with open(p) as f:
        return json.load(f)


def _build_server_test_loader(cfg: Config, data_root: str) -> DataLoader:
    return get_server_test_loader(
        root=data_root,
        batch_size=2,
        server_fraction=0.10,
        seed=cfg.experiment.seed,
        download=True,     # safe: PyTorch skips download if data already exists
        use_32=cfg.data.use_32,
        num_workers=cfg.fl.num_workers,
        pin_memory=cfg.fl.pin_memory,
    )


def _make_output_dir(cfg: Config, outputs_base: str) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(outputs_base) / "metrics" / f"run_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    Path(outputs_base) / "plots" / f"run_{run_id}"
    (Path(outputs_base) / "plots" / f"run_{run_id}").mkdir(parents=True, exist_ok=True)
    (Path(outputs_base) / "checkpoints").mkdir(parents=True, exist_ok=True)
    return out


# ── Strategy factory ──────────────────────────────────────────────────────────

def _build_strategy(
    cfg: Config,
    server_test_loader: DataLoader,
    dropout_tracker: DropoutTracker,
    output_dir: Path,
) -> FedAvgQuant:
    """Instantiate FedAvgQuant with fresh global model parameters."""
    init_model = get_model(freeze_features=cfg.fl.freeze_features)
    initial_parameters = ndarrays_to_parameters(get_parameters(init_model))

    return FedAvgQuant(
        server_test_loader=server_test_loader,
        dropout_tracker=dropout_tracker,
        cfg=cfg,
        output_dir=output_dir,
        initial_parameters=initial_parameters,
    )


def _get_data_root() -> str:
    """Return data root directory (env var DATA_ROOT or 'data/')."""
    return os.environ.get("DATA_ROOT", "data/")


# ── Main: gRPC start_server ───────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("  FL PPO Paper — Flower gRPC Server")
    log.info("=" * 60)
    log.info(f"  Config      : {CONFIG_PATH}")
    log.info(f"  Partition   : {PARTITION_FILE}")
    log.info(f"  Port        : {SERVER_PORT}")
    log.info(f"  Outputs dir : {OUTPUTS_DIR}")

    # 1. Load config
    cfg = load_config(CONFIG_PATH)
    n_rounds = int(NUM_ROUNDS_ENV) if NUM_ROUNDS_ENV else cfg.experiment.rounds
    cfg.experiment.rounds = int(n_rounds)
    log.info(f"  Experiment  : {cfg.experiment.name}")
    log.info(f"  Rounds      : {n_rounds}")
    log.info(f"  Clients     : {cfg.n_clients}")
    log.info(f"  Quant mode  : {cfg.quantization.mode}")
    log.info(f"  Lowp dtype  : {cfg.quantization.lowp_dtype}")
    log.info(f"  INT8 impl   : {cfg.quantization.int8_impl}")
    log.info(f"  INT8 backend: {cfg.quantization.int8_backend}")
    log.info(f"  QAT scope   : {cfg.quantization.qat_scope}")

    # 2. Verify partition file exists
    _load_partition_file(PARTITION_FILE)  # validates file exists; clients read it

    # 3. Build output directories
    output_dir = _make_output_dir(cfg, OUTPUTS_DIR)
    log.info(f"  Output dir  : {output_dir}")

    # 4. Build server-side test loader (server evaluation dataset)
    data_root = _get_data_root()
    log.info(f"  Data root   : {data_root}")
    server_test_loader = _build_server_test_loader(cfg, data_root)
    log.info(f"  Server test : {len(server_test_loader.dataset)} samples")

    # 5. Dropout tracker
    dropout_tracker = DropoutTracker(n_clients=cfg.n_clients)

    # 6. Strategy
    strategy = _build_strategy(cfg, server_test_loader, dropout_tracker, output_dir)

    # 7. Override quant assignments if mode=mixed (non-PPO static per-client assignment)
    if cfg.quantization.mode == "mixed" and cfg.quantization.per_client:
        strategy.current_quant_assignments = {
            str(k): v for k, v in cfg.quantization.per_client.items()
        }
        log.info(f"[Server] Mixed mode: {strategy.current_quant_assignments}")

    # 8. Start gRPC server (adaptive PPO is strategy-side in live runtime)
    server_address = f"0.0.0.0:{SERVER_PORT}"
    if cfg.quantization.mode == "adaptive":
        log.info("[Server] Mode: ADAPTIVE (strategy-side PPO runtime)")
    else:
        log.info(f"[Server] Mode: FIXED ({cfg.quantization.mode})")
    log.info(f"[Server] Listening on {server_address}, waiting for {cfg.n_clients} clients")

    fl.server.start_server(
        server_address=server_address,
        config=ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
    )

    # 9. Post-run: write summary.json
    json_logs = sorted(output_dir.glob("round_*.json"))
    summary = {
        "run_id": output_dir.name,
        "config": CONFIG_PATH,
        "experiment": cfg.experiment.name,
        "num_rounds": n_rounds,
        "rounds_completed": len(json_logs),
        "n_clients": cfg.n_clients,
        "quantization_mode": cfg.quantization.mode,
        "int8_impl": cfg.quantization.int8_impl,
        "qat_scope": cfg.quantization.qat_scope,
        "partition_file": PARTITION_FILE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Read accuracy from last round log if available
    if json_logs:
        with open(json_logs[-1]) as f:
            last = json.load(f)
        summary["final_accuracy"] = last.get("global_accuracy")
        summary["final_loss"] = last.get("global_loss")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"[Server] Summary written: {summary_path}")
    log.info(f"[Server] Rounds completed: {len(json_logs)}/{n_rounds}")
    log.info("[Server] Done.")


if __name__ == "__main__":
    main()
