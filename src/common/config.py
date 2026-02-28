"""
src/common/config.py — YAML config loader and schema for FL + PPO project
Paper: Adaptive FL with PPO-based Client and Quantization Selection

Usage:
    from src.common.config import load_config
    cfg = load_config("configs/exp1_homogeneous_iid.yaml")
    print(cfg.experiment.name)
    print(cfg.fl.batch_size_for("weak"))  # per-profile batch size

All src modules import FLConfig from here; never read YAML directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml


# ─── Sub-configs ──────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    name: str
    rounds: int = 50
    seed: int = 42
    dropout_mode: str = "stochastic"   # stochastic | docker

    def __post_init__(self):
        if self.dropout_mode not in ("stochastic", "docker"):
            raise ValueError(f"dropout_mode must be 'stochastic' or 'docker', got: {self.dropout_mode}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got: {self.rounds}")


@dataclass
class ClientProfileConfig:
    id: int
    profile: str                       # strong | medium | weak | extreme_weak
    mem_limit_mb: int
    cpu_cores: float
    data_fraction: float = 0.0         # 0.0 = auto-split equally

    def __post_init__(self):
        valid_profiles = ("strong", "medium", "weak", "extreme_weak")
        if self.profile not in valid_profiles:
            raise ValueError(f"Client {self.id}: profile must be one of {valid_profiles}, got: {self.profile}")
        if not (0.0 <= self.data_fraction <= 1.0):
            raise ValueError(f"Client {self.id}: data_fraction must be in [0,1], got: {self.data_fraction}")


@dataclass
class ClientsConfig:
    count: int
    profiles: List[ClientProfileConfig]

    def __post_init__(self):
        if len(self.profiles) != self.count:
            raise ValueError(
                f"clients.count={self.count} but {len(self.profiles)} profiles defined"
            )
        ids = [p.id for p in self.profiles]
        if sorted(ids) != list(range(self.count)):
            raise ValueError(f"Client IDs must be 0..{self.count-1}, got: {ids}")

    def profile_for(self, client_id: int) -> ClientProfileConfig:
        return self.profiles[client_id]


@dataclass
class DataConfig:
    dataset: str = "cifar10"
    partition: str = "iid"             # iid | dirichlet
    dirichlet_alpha: float = 0.1
    reduced_fraction: float = 1.0

    def __post_init__(self):
        if self.partition not in ("iid", "dirichlet"):
            raise ValueError(f"partition must be 'iid' or 'dirichlet', got: {self.partition}")
        if not (0.0 < self.reduced_fraction <= 1.0):
            raise ValueError(f"reduced_fraction must be in (0,1], got: {self.reduced_fraction}")


@dataclass
class QuantizationConfig:
    mode: str = "adaptive"             # adaptive | fixed_fp32 | fixed_fp16 | fixed_int8 | mixed
    fixed_bits: int = 32               # used only if mode=fixed_*
    calibration_samples: int = 128
    per_client: Optional[Dict[int, int]] = None   # used if mode=mixed

    def __post_init__(self):
        valid_modes = ("adaptive", "fixed_fp32", "fixed_fp16", "fixed_int8", "mixed")
        if self.mode not in valid_modes:
            raise ValueError(f"quantization.mode must be one of {valid_modes}, got: {self.mode}")
        # Fix H: only validate fixed_bits when mode explicitly uses it
        if self.mode.startswith("fixed_") and self.fixed_bits not in (32, 16, 8):
            raise ValueError(f"fixed_bits must be 32, 16, or 8 (for fixed_* modes), got: {self.fixed_bits}")
        if self.mode == "mixed" and self.per_client is None:
            raise ValueError("quantization.mode=mixed requires per_client mapping")
        # NOTE: No dynamic INT8 fallback. Fallback chain: static_int8 -> FP16 -> FP32,
        # logged as QUANT_UNSUPPORTED. Dynamic INT8 (quantize_dynamic) is never used.


# Per-profile batch size defaults (can be overridden per config)
_DEFAULT_BATCH_PER_PROFILE: Dict[str, int] = {
    "strong":       32,
    "medium":       16,
    "weak":          8,
    "extreme_weak":  8,
}


@dataclass
class FLConfig:
    local_epochs: int = 2
    batch_size: int = 32               # global default fallback
    per_profile_batch: Dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_BATCH_PER_PROFILE)
    )
    num_workers: int = 0               # 0 = VM-safe (no shared mem issues)
    pin_memory: bool = False           # CPU-only: pin_memory is wasteful
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    min_clients_per_round: int = 1

    def batch_size_for(self, profile: str) -> int:
        """Return the batch size appropriate for a given resource profile."""
        return self.per_profile_batch.get(profile, self.batch_size)

    def dataloader_kwargs(self) -> dict:
        """Return safe DataLoader kwargs for this VM configuration."""
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }


@dataclass
class RLConfig:
    algorithm: str = "ppo"
    total_timesteps: int = 50
    gamma_discount: float = 0.99
    reward_alpha: float = 1.0
    reward_beta: float = 0.5
    reward_gamma_var: float = 0.1
    policy: str = "MlpPolicy"
    checkpoint_dir: str = "outputs/checkpoints"
    n_steps: int = 1                   # PPO n_steps per update (1 = every FL round)
    learning_rate: float = 3e-4


@dataclass
class LoggingConfig:
    level: str = "info"
    metrics_dir: str = "outputs/metrics"
    plots_dir: str = "outputs/plots"


# ─── Top-level config ─────────────────────────────────────────────────────────

@dataclass
class Config:
    experiment: ExperimentConfig
    clients: ClientsConfig
    data: DataConfig
    quantization: QuantizationConfig
    fl: FLConfig
    rl: RLConfig
    logging: LoggingConfig

    @property
    def n_clients(self) -> int:
        return self.clients.count

    @property
    def state_dim(self) -> int:
        """
        Observation space dimension: (7 per-client features × N_clients) + 3 global features.

        Feature counts are defined in SPEC.md §3.1 and MUST be kept in sync with
        StateBuilder (src/rl/state.py). If the feature set changes, update:
          1. SPEC.md §3.1  2. StateBuilder  3. This constant.
        """
        PER_CLIENT_FEATURES = 7   # matches SPEC.md §3.1 per-client table
        GLOBAL_FEATURES = 3       # matches SPEC.md §3.1 global table
        return self.n_clients * PER_CLIENT_FEATURES + GLOBAL_FEATURES

    @property
    def action_nvec(self) -> List[int]:
        """MultiDiscrete action space shape: [4] * n_clients."""
        return [4] * self.n_clients


# ─── Loader ───────────────────────────────────────────────────────────────────

def _parse_clients(raw: dict) -> ClientsConfig:
    count = int(raw["count"])
    profiles = []
    for p in raw.get("profiles", []):
        profiles.append(ClientProfileConfig(
            id=int(p["id"]),
            profile=str(p["profile"]),
            mem_limit_mb=int(p["mem_limit_mb"]),
            cpu_cores=float(p["cpu_cores"]),
            data_fraction=float(p.get("data_fraction", 0.0)),
        ))
    # Auto-distribute data fractions if all are unset (all 0.0)
    # Ensures fractions sum to exactly 1.0: first count-1 clients get equal_share,
    # last client gets the exact remainder (no rounding).
    if all(p.data_fraction == 0.0 for p in profiles):
        equal_share = 1.0 / count
        allocated = 0.0
        for p in profiles[:-1]:
            p.data_fraction = equal_share
            allocated += equal_share
        profiles[-1].data_fraction = 1.0 - allocated

    return ClientsConfig(count=count, profiles=profiles)


def _parse_fl(raw: dict) -> FLConfig:
    per_profile_batch = dict(_DEFAULT_BATCH_PER_PROFILE)
    if "per_profile_batch" in raw:
        per_profile_batch.update(raw["per_profile_batch"])
    return FLConfig(
        local_epochs=int(raw.get("local_epochs", 2)),
        batch_size=int(raw.get("batch_size", 32)),
        per_profile_batch=per_profile_batch,
        num_workers=int(raw.get("num_workers", 0)),
        pin_memory=bool(raw.get("pin_memory", False)),
        optimizer=str(raw.get("optimizer", "sgd")),
        lr=float(raw.get("lr", 0.01)),
        momentum=float(raw.get("momentum", 0.9)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        min_clients_per_round=int(raw.get("min_clients_per_round", 1)),
    )


def _parse_quant(raw: dict) -> QuantizationConfig:
    per_client = None
    if "per_client" in raw:
        per_client = {int(k): int(v) for k, v in raw["per_client"].items()}
    return QuantizationConfig(
        mode=str(raw.get("mode", "adaptive")),
        fixed_bits=int(raw.get("fixed_bits", 32)),
        calibration_samples=int(raw.get("calibration_samples", 128)),
        per_client=per_client,
    )


def load_config(path: str) -> Config:
    """
    Load and validate an experiment YAML config file.

    Args:
        path: Absolute or relative path to the YAML file.

    Returns:
        A validated Config dataclass instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If any required field is missing or has an invalid value.
        yaml.YAMLError: If the YAML is malformed.
    """
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")

    with open(abs_path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got: {type(raw)}")

    # Required top-level keys
    for required_key in ("experiment", "clients", "data", "quantization", "fl"):
        if required_key not in raw:
            raise ValueError(f"Config missing required top-level key: '{required_key}'")

    exp_raw = raw["experiment"]
    experiment = ExperimentConfig(
        name=str(exp_raw["name"]),
        rounds=int(exp_raw.get("rounds", 50)),
        seed=int(exp_raw.get("seed", 42)),
        dropout_mode=str(exp_raw.get("dropout_mode", "stochastic")),
    )

    clients = _parse_clients(raw["clients"])

    d = raw["data"]
    data = DataConfig(
        dataset=str(d.get("dataset", "cifar10")),
        partition=str(d.get("partition", "iid")),
        dirichlet_alpha=float(d.get("dirichlet_alpha", 0.1)),
        reduced_fraction=float(d.get("reduced_fraction", 1.0)),
    )

    quantization = _parse_quant(raw["quantization"])
    fl = _parse_fl(raw["fl"])

    rl_raw = raw.get("rl", {})
    rl = RLConfig(
        algorithm=str(rl_raw.get("algorithm", "ppo")),
        total_timesteps=int(rl_raw.get("total_timesteps", experiment.rounds)),
        gamma_discount=float(rl_raw.get("gamma_discount", 0.99)),
        reward_alpha=float(rl_raw.get("reward_alpha", 1.0)),
        reward_beta=float(rl_raw.get("reward_beta", 0.5)),
        reward_gamma_var=float(rl_raw.get("reward_gamma_var", 0.1)),
        policy=str(rl_raw.get("policy", "MlpPolicy")),
        checkpoint_dir=str(rl_raw.get("checkpoint_dir", "outputs/checkpoints")),
        n_steps=int(rl_raw.get("n_steps", 1)),
        learning_rate=float(rl_raw.get("learning_rate", 3e-4)),
    )

    log_raw = raw.get("logging", {})
    logging_cfg = LoggingConfig(
        level=str(log_raw.get("level", "info")),
        metrics_dir=str(log_raw.get("metrics_dir", "outputs/metrics")),
        plots_dir=str(log_raw.get("plots_dir", "outputs/plots")),
    )

    return Config(
        experiment=experiment,
        clients=clients,
        data=data,
        quantization=quantization,
        fl=fl,
        rl=rl,
        logging=logging_cfg,
    )


# ─── CLI: validate a config file ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"Usage: python -m src.common.config <config.yaml>")
        sys.exit(1)

    try:
        cfg = load_config(sys.argv[1])
        print(f"[OK] Config valid: {cfg.experiment.name}")
        print(f"     n_clients:  {cfg.n_clients}")
        print(f"     state_dim:  {cfg.state_dim}")
        print(f"     action_nv:  {cfg.action_nvec}")
        print(f"     rounds:     {cfg.experiment.rounds}")
        print(f"     quant mode: {cfg.quantization.mode}")
        print(f"     batch (weak): {cfg.fl.batch_size_for('weak')}")
        print(f"     num_workers:  {cfg.fl.num_workers}")
        sys.exit(0)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        print(f"[FAIL] Config invalid: {e}")
        sys.exit(1)
