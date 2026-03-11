"""
client_main.py — Flower gRPC client entrypoint for Docker deployment
Stage 13: True multi-container FL client using fl.client.start_client()

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Environment variables (required in Docker containers):
    CLIENT_ID        Integer 0..N-1 identifying this client
    SERVER_ADDR      gRPC server address (default: server:8080)
    CONFIG_PATH      Path to YAML config (default: configs/exp_docker_smoke.yaml)
    PARTITION_FILE   Path to pre-computed partition JSON (default: data/partitions_50_iid.json)
    DATA_ROOT        CIFAR-10 data directory (default: data/)

Usage (non-Docker / local test):
    CLIENT_ID=0 SERVER_ADDR=localhost:8080 \\
    CONFIG_PATH=configs/exp_docker_smoke.yaml \\
    PARTITION_FILE=data/partitions_50_iid.json \\
    python3 client_main.py

PAPER-ALIGNED:
    - Client applies quantization assigned by server per round (quant_bits in FitIns.config)
    - Client always returns FP32 parameters (quantization is for distribution only)
    - Stochastic dropout simulation matches profile-based probabilities
    - 224×224 resolution enforced (use_32=False hardcoded in FlowerClient)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.common.config import load_config, ClientProfileConfig
from src.fl.client import FlowerClient

import flwr as fl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("client_main")
logging.getLogger("flwr").setLevel(logging.WARNING)


# ── Environment variables ─────────────────────────────────────────────────────

CLIENT_ID      = int(os.environ.get("CLIENT_ID", "0"))
SERVER_ADDR    = os.environ.get("SERVER_ADDR",    "server:8080")
CONFIG_PATH    = os.environ.get("CONFIG_PATH",    "configs/exp_docker_smoke.yaml")
PARTITION_FILE = os.environ.get("PARTITION_FILE", "data/partitions_50_iid.json")
DATA_ROOT      = os.environ.get("DATA_ROOT",      "data/")


# ── Partition loading ──────────────────────────────────────────────────────────

def _load_client_partition(partition_file: str, client_id: int):
    """
    Load this client's train + eval index lists from the shared partition JSON.

    Returns:
        (train_indices: List[int], eval_indices: List[int])
    """
    import json
    p = Path(partition_file)
    if not p.exists():
        raise FileNotFoundError(
            f"[Client {client_id}] Partition file not found: {partition_file}\n"
            f"Run: python3 scripts/download_and_partition.py --n-clients <N>\n"
            f"Then mount the data/ directory to the container."
        )

    with open(p) as f:
        data = json.load(f)

    cid = str(client_id)
    if cid not in data.get("train", {}):
        raise KeyError(
            f"[Client {client_id}] CLIENT_ID={client_id} not in partition file "
            f"(file has {len(data.get('train', {}))} entries). "
            f"Regenerate with --n-clients >= {client_id + 1}."
        )

    train_indices = data["train"][cid]
    eval_indices  = data["eval"].get(cid, [])
    return train_indices, eval_indices


def _apply_reduced_fraction(
    train_indices: list[int],
    reduced_fraction: float,
    seed: int,
    client_id: int,
) -> list[int]:
    """
    Apply config.data.reduced_fraction to partition-file train indices.

    The partition JSON stores full train shards, so we sub-sample here to keep
    Docker gRPC runs consistent with config-based reduced_fraction experiments.
    """
    if reduced_fraction >= 1.0:
        return train_indices

    import random

    n_keep = max(1, int(len(train_indices) * reduced_fraction))
    rng = random.Random(seed + client_id)
    sampled = rng.sample(train_indices, n_keep)
    sampled.sort()
    return sampled


# ── Profile from partition tier ───────────────────────────────────────────────

def _build_profile_from_partition(
    partition_file: str,
    client_id: int,
    cfg,
) -> ClientProfileConfig:
    """
    Build a ClientProfileConfig for this client using the tier stored in the
    partition file. This ensures the profile matches the Docker resource limits
    specified in docker-compose.yml, regardless of what the config YAML says.

    Falls back to cfg.clients.profile_for(client_id) if client_id is within
    the config's declared client count.
    """
    import json
    try:
        with open(partition_file) as f:
            data = json.load(f)
        tier = data.get("tiers", {}).get(str(client_id))
        specs = data.get("profile_specs", {}).get(tier, {})

        if tier and specs:
            return ClientProfileConfig(
                id=client_id,
                profile=tier,
                mem_limit_mb=specs.get("mem_limit_mb", 1024),
                cpu_cores=float(specs.get("cpu_cores", 1.0)),
                data_fraction=0.0,
            )
    except Exception:
        pass  # Fall through to config-based lookup

    # Fallback: look up by config if within declared range
    if client_id < cfg.n_clients:
        return cfg.clients.profile_for(client_id)

    # Default: medium profile
    return ClientProfileConfig(
        id=client_id,
        profile="medium",
        mem_limit_mb=1024,
        cpu_cores=1.0,
        data_fraction=0.0,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info(f"  FL PPO Paper — Flower gRPC Client (ID={CLIENT_ID})")
    log.info("=" * 60)
    log.info(f"  Server      : {SERVER_ADDR}")
    log.info(f"  Config      : {CONFIG_PATH}")
    log.info(f"  Partition   : {PARTITION_FILE}")
    log.info(f"  Data root   : {DATA_ROOT}")

    # 1. Load config
    cfg = load_config(CONFIG_PATH)
    log.info(f"  Experiment  : {cfg.experiment.name}")
    log.info(f"  Quant mode  : {cfg.quantization.mode}")
    log.info(f"  Lowp dtype  : {cfg.quantization.lowp_dtype}")

    # 2. Load this client's data partition
    train_indices, eval_indices = _load_client_partition(PARTITION_FILE, CLIENT_ID)
    train_indices = _apply_reduced_fraction(
        train_indices=train_indices,
        reduced_fraction=cfg.data.reduced_fraction,
        seed=cfg.experiment.seed,
        client_id=CLIENT_ID,
    )
    log.info(f"  Train idx   : {len(train_indices)} samples")
    log.info(f"  Eval idx    : {len(eval_indices)} samples")

    # 3. Build profile
    profile = _build_profile_from_partition(PARTITION_FILE, CLIENT_ID, cfg)
    log.info(f"  Profile     : {profile.profile} "
             f"(cpu={profile.cpu_cores}, mem={profile.mem_limit_mb}MB)")

    # 4. Build FlowerClient
    flower_client = FlowerClient(
        client_id=CLIENT_ID,
        train_indices=train_indices,
        eval_indices=eval_indices,
        cfg=cfg,
        profile=profile,
        data_root=DATA_ROOT,
    )

    # 5. Connect to server and start the gRPC client loop
    # Retry logic: server may not be ready immediately after starting.
    # Server loads CIFAR-10 and builds strategy before opening gRPC port (~10-30s).
    server_address = SERVER_ADDR
    max_retries = int(os.environ.get("CLIENT_MAX_RETRIES", "20"))
    retry_delay  = float(os.environ.get("CLIENT_RETRY_DELAY", "5.0"))

    import time
    for attempt in range(max_retries):
        try:
            log.info(f"[Client {CLIENT_ID}] Connecting to {server_address} "
                     f"(attempt {attempt + 1}/{max_retries}) ...")
            fl.client.start_client(
                server_address=server_address,
                client=flower_client.to_client(),
            )
            log.info(f"[Client {CLIENT_ID}] Disconnected from server. Done.")
            break
        except Exception as e:
            err_str = str(e)
            if "UNAVAILABLE" in err_str or "Connection refused" in err_str:
                if attempt < max_retries - 1:
                    log.warning(
                        f"[Client {CLIENT_ID}] Server not ready "
                        f"(attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {retry_delay}s ..."
                    )
                    time.sleep(retry_delay)
                    continue
            log.error(f"[Client {CLIENT_ID}] Connection failed: {e}")
            raise


if __name__ == "__main__":
    main()
