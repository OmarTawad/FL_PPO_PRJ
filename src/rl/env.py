"""
src/rl/env.py — FLEnv: Gymnasium environment wrapping one FL round per step

Paper: Adaptive FL with PPO-based Client and Quantization Selection

ONE env.step() = ONE FL ROUND:
  - PPO provides MultiDiscrete action [a_0, a_1, ..., a_{N-1}]
  - Action mapping: 0=skip, 1=fp32(32 bits), 2=fp16(16 bits), 3=int8(8 bits)
  - At least 1 client must be selected (HARD REQUIREMENT)
  - Strategy runs configure_fit (per-client) → fit → aggregate_fit
    then configure_evaluate → evaluate → aggregate_evaluate
  - Reward: r_t = α·ΔAcc - β·Dropout - γ·Var(Q_t)
  - Observation: N*7+3 vector per SPEC.md §3.1

Episode lifecycle:
  reset() → initial obs (round=0, acc=0, no history)
  step(action) × T rounds → terminated=True after round T

Heavy FL infrastructure (CIFAR datasets, partitions, models) is built once
in __init__ and REUSED across env.reset() calls (multiple PPO episodes).
Only mutable state is reset: model parameters, dropout tracker, accuracy history.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from flwr.common import ndarrays_to_parameters

from src.common.config import Config
from src.models.mobilenetv2 import get_model, get_parameters, set_parameters
from src.heterogeneity.dropout import DropoutTracker
from src.heterogeneity.profiles import get_profile
from src.fl.server import (
    make_server_test_loader, make_partitions, make_client_manager, run_one_round,
    _MockClientProxy,
)
from src.fl.strategy import FedAvgQuant
from src.rl.state import StateBuilder
from src.rl.reward import compute_reward

log = logging.getLogger(__name__)

# Action → quant_bits mapping (HARD REQUIREMENT: 0=skip, 1=fp32, 2=fp16, 3=int8)
_ACTION_TO_BITS: Dict[int, int] = {0: 0, 1: 32, 2: 16, 3: 8}


class FLEnv(gym.Env):
    """
    Gymnasium environment for PPO-based FL client and quantization selection.

    Spaces:
        action_space:      MultiDiscrete([4] * N_clients)
                           0=skip, 1=fp32, 2=fp16, 3=int8
        observation_space: Box(-inf, inf, shape=(N*7+3,), dtype=float32)

    Episode:
        Each step runs one FL round.
        Terminated after cfg.experiment.rounds steps.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Config, data_root: str = "data/"):
        super().__init__()
        self.cfg = cfg
        self.data_root = data_root
        self.n = cfg.n_clients

        # Spaces
        self.action_space = spaces.MultiDiscrete([4] * self.n)
        obs_dim = self.n * 7 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Profile objects (for "force highest reliability" rule)
        self._profile_cfgs = cfg.clients.profiles  # List[ClientProfileConfig]
        self._reliabilities = []
        for prof_cfg in self._profile_cfgs:
            try:
                hp = get_profile(prof_cfg.profile)
                self._reliabilities.append(hp.reliability)
            except KeyError:
                self._reliabilities.append(1.0)

        # ── Heavy one-time setup (shared across all episodes) ─────────────────
        log.info("[FLEnv] One-time setup: building data loaders and partitions...")
        self._server_test_loader = make_server_test_loader(cfg)
        self._train_partitions, self._eval_partitions = make_partitions(cfg)
        for i, p in enumerate(self._train_partitions):
            log.info(
                f"  Client {i}: {len(p)} train / "
                f"{len(self._eval_partitions[i])} eval samples"
            )

        # Episode-level state (reset per episode)
        self.dropout_tracker: Optional[DropoutTracker] = None
        self.state_builder: Optional[StateBuilder] = None
        self.strategy: Optional[FedAvgQuant] = None
        self.client_manager = None
        self.parameters = None
        self._output_dir: Optional[Path] = None
        self._episode_count = 0
        self.round_idx = 0
        self.prev_accuracy = 0.0
        self.prev_delta = 0.0
        self.last_quant_bits: Dict[int, int] = {i: 0 for i in range(self.n)}
        self._terminated = True  # needs reset before first step

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Start a new FL episode.
        - Reinitialises model parameters (fresh random weights).
        - Resets DropoutTracker and accuracy history.
        - Creates a new output directory for this episode's JSON logs.
        """
        super().reset(seed=seed)

        self._episode_count += 1
        self.round_idx = 0
        self.prev_accuracy = 0.0
        self.prev_delta = 0.0
        self.last_quant_bits = {i: 0 for i in range(self.n)}
        self._terminated = False

        # New output directory per episode
        run_id = datetime.now(timezone.utc).strftime(f"%Y%m%d_%H%M%S_ep{self._episode_count:03d}")
        self._output_dir = Path(self.cfg.logging.metrics_dir) / f"run_{run_id}"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Fresh dropout tracker
        self.dropout_tracker = DropoutTracker(n_clients=self.n)

        # Fresh state builder
        self.state_builder = StateBuilder(
            n_clients=self.n,
            profiles=self._profile_cfgs,
            dropout_tracker=self.dropout_tracker,
            total_rounds=self.cfg.experiment.rounds,
        )

        # Fresh global model parameters (random init)
        init_model = get_model()
        self.parameters = ndarrays_to_parameters(get_parameters(init_model))

        # Fresh client manager (reuses same data partitions)
        self.client_manager, _ = make_client_manager(
            self.cfg,
            self._train_partitions,
            self._eval_partitions,
            data_root=self.data_root,
        )

        # Fresh strategy (reuses same server_test_loader)
        self.strategy = FedAvgQuant(
            server_test_loader=self._server_test_loader,
            dropout_tracker=self.dropout_tracker,
            cfg=self.cfg,
            output_dir=self._output_dir,
            initial_parameters=self.parameters,
        )

        obs = self.state_builder.build(
            round_idx=0,
            prev_global_acc=0.0,
            prev_acc_delta=0.0,
            last_quant_bits=self.last_quant_bits,
        )
        log.info(f"[FLEnv] Episode {self._episode_count} reset. obs shape={obs.shape}")
        return obs.astype(np.float32), {}

    # ── step ──────────────────────────────────────────────────────────────────

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one FL round and return (obs, reward, terminated, truncated, info).

        Action: array of shape (N,), values in {0,1,2,3}.
          0 = skip client
          1 = select with FP32
          2 = select with FP16
          3 = select with INT8

        Raises:
            RuntimeError: If called before reset().
        """
        if self._terminated:
            raise RuntimeError(
                "FLEnv.step() called on a terminated episode. Call reset() first."
            )

        # 1. Decode action → per-client bits
        action_arr = np.asarray(action, dtype=int)
        action_bits = {i: _ACTION_TO_BITS[int(action_arr[i])] for i in range(self.n)}

        # 2. Selected clients (bits > 0)
        selected = {i: b for i, b in action_bits.items() if b > 0}

        # 3. HARD REQUIREMENT: at least 1 client selected
        if not selected:
            best_cid = int(np.argmax(self._reliabilities))
            selected = {best_cid: 32}
            log.info(
                f"[FLEnv] All-skip action: forced client {best_cid} (FP32, "
                f"highest reliability={self._reliabilities[best_cid]:.2f})"
            )

        # 4. Inject into strategy (string cid keys required by configure_fit)
        self.strategy.current_quant_assignments = {
            str(cid): bits for cid, bits in selected.items()
        }

        # 5. Run one FL round
        self.round_idx += 1
        self.parameters, round_log = run_one_round(
            self.strategy,
            self.client_manager,
            self.parameters,
            self.round_idx,
        )

        # 6. Extract accuracy from round_log
        acc_t = float(round_log.get("global_accuracy", self.prev_accuracy))

        # 7. Compute reward
        reward, reward_components = compute_reward(
            acc_t=acc_t,
            acc_prev=self.prev_accuracy,
            selected_client_ids=[str(c) for c in selected],
            dropout_client_ids=round_log.get("dropout_clients", []),
            quant_assignments_bits={
                str(c): b for c, b in selected.items()
            },
            alpha=self.cfg.rl.reward_alpha,
            beta=self.cfg.rl.reward_beta,
            gamma=self.cfg.rl.reward_gamma_var,
        )

        # 8. Update tracking state
        self.prev_delta = acc_t - self.prev_accuracy
        self.prev_accuracy = acc_t
        for i in range(self.n):
            self.last_quant_bits[i] = selected.get(i, 0)

        # 9. Build next observation
        obs = self.state_builder.build(
            round_idx=self.round_idx,
            prev_global_acc=self.prev_accuracy,
            prev_acc_delta=self.prev_delta,
            last_quant_bits=self.last_quant_bits,
        )

        # 10. Termination
        terminated = self.round_idx >= self.cfg.experiment.rounds
        if terminated:
            self._terminated = True

        info: dict = {
            "round": self.round_idx,
            "global_accuracy": acc_t,
            "accuracy_delta": self.prev_delta,
            "selected_clients": list(selected.keys()),
            "reward_components": reward_components,
            "round_log": round_log,
        }
        return obs.astype(np.float32), float(reward), terminated, False, info

    def render(self):
        pass

    def close(self):
        pass
