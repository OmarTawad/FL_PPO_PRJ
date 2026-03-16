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
    profile: str                       # strong | medium | medium_low | weak | extreme_weak | ultra_extreme
    mem_limit_mb: int
    cpu_cores: float
    data_fraction: float = 0.0         # 0.0 = auto-split equally

    def __post_init__(self):
        valid_profiles = (
            "strong",
            "medium",
            "medium_low",
            "weak",
            "extreme_weak",
            "ultra_extreme",
        )
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
    use_32: bool = False               # False=224x224 paper path, True=native 32x32
    train_augment: bool = True
    eval_fraction: float = 0.10        # fraction of test set per client (for evaluate phase)
    max_eval_samples_per_client: int = 0  # 0 = use eval_fraction; >0 caps samples

    def __post_init__(self):
        if self.partition not in ("iid", "dirichlet"):
            raise ValueError(f"partition must be 'iid' or 'dirichlet', got: {self.partition}")
        if not (0.0 < self.reduced_fraction <= 1.0):
            raise ValueError(f"reduced_fraction must be in (0,1], got: {self.reduced_fraction}")
        if not (0.0 < self.eval_fraction <= 1.0):
            raise ValueError(f"eval_fraction must be in (0,1], got: {self.eval_fraction}")


@dataclass
class QuantizationConfig:
    mode: str = "adaptive"             # adaptive | fixed_fp32 | fixed_fp16 | fixed_int8 | mixed
    fixed_bits: int = 32               # used only if mode=fixed_*
    lowp_dtype: str = "bf16"           # dtype used when bits=16: bf16 | fp16
    int8_impl: str = "static"          # static | qat
    int8_backend: str = "auto"         # auto | fbgemm | x86 | onednn | qnnpack
    calibration_samples: int = 128
    qat_scope: str = "full"            # full | classifier_only
    qat_convert_after_fit: bool = False
    qat_eval_batches_for_convert_check: int = 1
    qat_convert_probe_client_id: int = -1  # -1 = all clients
    qat_convert_probe_round: int = 0        # 0 = all rounds
    int8_postcheck_enabled: bool = False
    int8_postcheck_backend: str = "auto"    # auto | fbgemm | x86 | onednn | qnnpack
    int8_postcheck_after_fit: bool = False
    int8_postcheck_probe_client_id: int = -1  # -1 = all clients
    int8_postcheck_probe_round: int = 0        # 0 = all rounds
    int8_postcheck_eval_batches: int = 1
    transport_mode: str = "model_fp32"   # model_fp32 | delta
    transport_per_client: Dict[int, str] = field(default_factory=dict)  # fp32 | int8
    transport_int8_scheme: str = "symm_per_tensor_v1"
    transport_require_decode_success: bool = True
    adaptive_bf16_warmup_rounds: int = 0
    per_client: Optional[Dict[int, int]] = None   # used if mode=mixed

    def __post_init__(self):
        valid_modes = ("adaptive", "fixed_fp32", "fixed_fp16", "fixed_int8", "mixed")
        if self.mode not in valid_modes:
            raise ValueError(f"quantization.mode must be one of {valid_modes}, got: {self.mode}")
        # Fix H: only validate fixed_bits when mode explicitly uses it
        if self.mode.startswith("fixed_") and self.fixed_bits not in (32, 16, 8):
            raise ValueError(f"fixed_bits must be 32, 16, or 8 (for fixed_* modes), got: {self.fixed_bits}")
        self.lowp_dtype = str(self.lowp_dtype).strip().lower()
        if self.lowp_dtype not in ("bf16", "fp16"):
            raise ValueError(
                f"quantization.lowp_dtype must be 'bf16' or 'fp16', got: {self.lowp_dtype}"
            )
        self.int8_impl = str(self.int8_impl).strip().lower()
        if self.int8_impl not in ("static", "qat"):
            raise ValueError(
                f"quantization.int8_impl must be 'static' or 'qat', got: {self.int8_impl}"
            )
        self.int8_backend = str(self.int8_backend).strip().lower()
        valid_int8_backends = ("auto", "fbgemm", "x86", "onednn", "qnnpack")
        if self.int8_backend not in valid_int8_backends:
            raise ValueError(
                f"quantization.int8_backend must be one of {valid_int8_backends}, got: {self.int8_backend}"
            )
        self.int8_postcheck_backend = str(self.int8_postcheck_backend).strip().lower()
        if self.int8_postcheck_backend not in valid_int8_backends:
            raise ValueError(
                "quantization.int8_postcheck_backend must be one of "
                f"{valid_int8_backends}, got: {self.int8_postcheck_backend}"
            )
        self.qat_scope = str(self.qat_scope).strip().lower()
        if self.qat_scope not in ("full", "classifier_only"):
            raise ValueError(
                "quantization.qat_scope must be 'full' or 'classifier_only', "
                f"got: {self.qat_scope}"
            )
        if self.qat_eval_batches_for_convert_check < 1:
            raise ValueError(
                "quantization.qat_eval_batches_for_convert_check must be >= 1"
            )
        if self.qat_convert_probe_round < 0:
            raise ValueError(
                "quantization.qat_convert_probe_round must be >= 0"
            )
        if self.int8_postcheck_probe_round < 0:
            raise ValueError(
                "quantization.int8_postcheck_probe_round must be >= 0"
            )
        if self.int8_postcheck_eval_batches < 1:
            raise ValueError(
                "quantization.int8_postcheck_eval_batches must be >= 1"
            )
        self.transport_mode = str(self.transport_mode).strip().lower()
        if self.transport_mode not in ("model_fp32", "delta"):
            raise ValueError(
                "quantization.transport_mode must be 'model_fp32' or 'delta', "
                f"got: {self.transport_mode}"
            )
        if int(self.adaptive_bf16_warmup_rounds) < 0:
            raise ValueError(
                "quantization.adaptive_bf16_warmup_rounds must be >= 0"
            )
        self.adaptive_bf16_warmup_rounds = int(self.adaptive_bf16_warmup_rounds)
        self.transport_int8_scheme = str(self.transport_int8_scheme).strip().lower()
        if self.transport_int8_scheme != "symm_per_tensor_v1":
            raise ValueError(
                "quantization.transport_int8_scheme must be 'symm_per_tensor_v1', "
                f"got: {self.transport_int8_scheme}"
            )
        normalized_transport: Dict[int, str] = {}
        for raw_cid, raw_dtype in (self.transport_per_client or {}).items():
            cid = int(raw_cid)
            dtype = str(raw_dtype).strip().lower()
            if dtype not in ("fp32", "int8"):
                raise ValueError(
                    "quantization.transport_per_client values must be 'fp32' or 'int8', "
                    f"got client {cid}: {raw_dtype}"
                )
            normalized_transport[cid] = dtype
        self.transport_per_client = normalized_transport
        if self.mode == "mixed" and self.per_client is None:
            raise ValueError("quantization.mode=mixed requires per_client mapping")
        # NOTE: No dynamic INT8 fallback. Fallback chain: static_int8 -> FP16 -> FP32,
        # logged as QUANT_UNSUPPORTED. Dynamic INT8 (quantize_dynamic) is never used.


@dataclass
class PruningConfig:
    enabled: bool = False
    method: str = "magnitude_unstructured"   # currently supported: magnitude_unstructured
    amount: float = 0.0
    apply_per_round: bool = True
    per_client: Dict[int, float] = field(default_factory=dict)
    target_modules: List[str] = field(default_factory=lambda: ["conv2d", "linear"])

    def __post_init__(self):
        self.method = str(self.method).strip().lower()
        if self.method != "magnitude_unstructured":
            raise ValueError(
                "pruning.method must be 'magnitude_unstructured', "
                f"got: {self.method}"
            )
        if not (0.0 <= float(self.amount) < 1.0):
            raise ValueError(
                f"pruning.amount must be in [0,1), got: {self.amount}"
            )
        normalized_targets: List[str] = []
        valid_targets = {"conv2d", "linear"}
        for target in self.target_modules:
            t = str(target).strip().lower()
            if t not in valid_targets:
                raise ValueError(
                    f"pruning.target_modules entries must be in {sorted(valid_targets)}, got: {target}"
                )
            if t not in normalized_targets:
                normalized_targets.append(t)
        if not normalized_targets:
            raise ValueError("pruning.target_modules cannot be empty")
        self.target_modules = normalized_targets

        normalized_per_client: Dict[int, float] = {}
        for raw_cid, raw_amount in (self.per_client or {}).items():
            cid = int(raw_cid)
            amt = float(raw_amount)
            if not (0.0 <= amt < 1.0):
                raise ValueError(
                    f"pruning.per_client[{cid}] must be in [0,1), got: {raw_amount}"
                )
            normalized_per_client[cid] = amt
        self.per_client = normalized_per_client


# Per-profile batch size defaults (can be overridden per config)
_DEFAULT_BATCH_PER_PROFILE: Dict[str, int] = {
    "strong":       32,
    "medium":       16,
    "medium_low":   16,
    "weak":          8,
    "extreme_weak":  8,
    "ultra_extreme": 8,
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
    freeze_features: bool = True
    min_clients_per_round: int = 1
    aggregate_bn_buffers: bool = True

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
    batch_size: int = 64               # PPO minibatch size
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
    pruning: PruningConfig
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
        freeze_features=bool(raw.get("freeze_features", True)),
        min_clients_per_round=int(raw.get("min_clients_per_round", 1)),
        aggregate_bn_buffers=bool(raw.get("aggregate_bn_buffers", True)),
    )


def _parse_quant(raw: dict) -> QuantizationConfig:
    per_client = None
    if "per_client" in raw:
        per_client = {int(k): int(v) for k, v in raw["per_client"].items()}
    transport_per_client = {}
    if "transport_per_client" in raw:
        transport_per_client = {
            int(k): str(v) for k, v in raw["transport_per_client"].items()
        }
    return QuantizationConfig(
        mode=str(raw.get("mode", "adaptive")),
        fixed_bits=int(raw.get("fixed_bits", 32)),
        lowp_dtype=str(raw.get("lowp_dtype", "bf16")),
        int8_impl=str(raw.get("int8_impl", "static")),
        int8_backend=str(raw.get("int8_backend", "auto")),
        calibration_samples=int(raw.get("calibration_samples", 128)),
        qat_scope=str(raw.get("qat_scope", "full")),
        qat_convert_after_fit=bool(raw.get("qat_convert_after_fit", False)),
        qat_eval_batches_for_convert_check=int(
            raw.get("qat_eval_batches_for_convert_check", 1)
        ),
        qat_convert_probe_client_id=int(
            raw.get("qat_convert_probe_client_id", -1)
        ),
        qat_convert_probe_round=int(
            raw.get("qat_convert_probe_round", 0)
        ),
        int8_postcheck_enabled=bool(raw.get("int8_postcheck_enabled", False)),
        int8_postcheck_backend=str(raw.get("int8_postcheck_backend", "auto")),
        int8_postcheck_after_fit=bool(raw.get("int8_postcheck_after_fit", False)),
        int8_postcheck_probe_client_id=int(
            raw.get("int8_postcheck_probe_client_id", -1)
        ),
        int8_postcheck_probe_round=int(
            raw.get("int8_postcheck_probe_round", 0)
        ),
        int8_postcheck_eval_batches=int(
            raw.get("int8_postcheck_eval_batches", 1)
        ),
        transport_mode=str(raw.get("transport_mode", "model_fp32")),
        transport_per_client=transport_per_client,
        transport_int8_scheme=str(
            raw.get("transport_int8_scheme", "symm_per_tensor_v1")
        ),
        transport_require_decode_success=bool(
            raw.get("transport_require_decode_success", True)
        ),
        adaptive_bf16_warmup_rounds=int(
            raw.get("adaptive_bf16_warmup_rounds", 0)
        ),
        per_client=per_client,
    )


def _parse_pruning(raw: dict) -> PruningConfig:
    per_client = {}
    if "per_client" in raw:
        per_client = {int(k): float(v) for k, v in raw["per_client"].items()}
    return PruningConfig(
        enabled=bool(raw.get("enabled", False)),
        method=str(raw.get("method", "magnitude_unstructured")),
        amount=float(raw.get("amount", 0.0)),
        apply_per_round=bool(raw.get("apply_per_round", True)),
        per_client=per_client,
        target_modules=list(raw.get("target_modules", ["conv2d", "linear"])),
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
        use_32=bool(d.get("use_32", False)),
        train_augment=bool(d.get("train_augment", True)),
        eval_fraction=float(d.get("eval_fraction", 0.10)),
        max_eval_samples_per_client=int(d.get("max_eval_samples_per_client", 0)),
    )

    quantization = _parse_quant(raw["quantization"])
    pruning = _parse_pruning(raw.get("pruning", {}))
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
        batch_size=int(rl_raw.get("batch_size", 64)),
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
        pruning=pruning,
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
        print(f"     lowp dtype: {cfg.quantization.lowp_dtype}")
        print(f"     int8 impl: {cfg.quantization.int8_impl}")
        print(f"     int8 backend: {cfg.quantization.int8_backend}")
        print(f"     int8 postcheck: {cfg.quantization.int8_postcheck_enabled}")
        print(f"     int8 postcheck backend: {cfg.quantization.int8_postcheck_backend}")
        print(f"     qat scope: {cfg.quantization.qat_scope}")
        print(f"     pruning enabled: {cfg.pruning.enabled}")
        print(f"     pruning method: {cfg.pruning.method}")
        print(f"     pruning amount: {cfg.pruning.amount}")
        print(f"     batch (weak): {cfg.fl.batch_size_for('weak')}")
        print(f"     num_workers:  {cfg.fl.num_workers}")
        sys.exit(0)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        print(f"[FAIL] Config invalid: {e}")
        sys.exit(1)
