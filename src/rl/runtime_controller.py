"""
src/rl/runtime_controller.py — PPO runtime controller for live Flower rounds.

This controller enables strategy-side PPO decisions in gRPC runs:
  - predict per-client actions from current observation
  - record (s, a, r, s') transitions
  - perform lightweight online PPO updates (no gym step loop)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure as sb3_configure_logger
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.common.config import Config

log = logging.getLogger(__name__)


class _StaticShapeEnv(gym.Env):
    """Minimal env exposing only spaces for PPO policy construction."""

    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int, n_clients: int):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([4] * n_clients)
        self._obs = np.zeros((obs_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs.copy(), {}

    def step(self, action):
        # Not used for transition rollout; strategy drives transitions directly.
        return self._obs.copy(), 0.0, False, False, {}


@dataclass
class PPOUpdateInfo:
    update_applied: int
    updates_total: int
    transitions_total: int
    buffer_size: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "update_applied": int(self.update_applied),
            "updates_total": int(self.updates_total),
            "transitions_total": int(self.transitions_total),
            "buffer_size": int(self.buffer_size),
        }


class PPORuntimeController:
    """
    PPO controller used directly from strategy.configure_fit/aggregate_fit.

    One controller instance is bound to one server run.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n_clients = cfg.n_clients
        self.obs_dim = cfg.state_dim

        requested_n_steps = int(cfg.rl.n_steps)
        requested_batch_size = int(cfg.rl.batch_size)
        # SB3 PPO requires batch_size > 1 and valid rollout horizon.
        # Keep configs backward-compatible by normalizing invalid tiny values.
        n_steps = max(2, requested_n_steps)
        batch_size = max(2, min(requested_batch_size, n_steps))
        if n_steps != requested_n_steps or batch_size != requested_batch_size:
            log.warning(
                "[PPOController] adjusted invalid PPO settings: "
                "requested(n_steps=%d,batch_size=%d) -> effective(n_steps=%d,batch_size=%d)",
                requested_n_steps,
                requested_batch_size,
                n_steps,
                batch_size,
            )

        vec_env = DummyVecEnv([lambda: _StaticShapeEnv(self.obs_dim, self.n_clients)])
        self.model = PPO(
            policy=cfg.rl.policy,
            env=vec_env,
            verbose=0,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=cfg.rl.gamma_discount,
            learning_rate=cfg.rl.learning_rate,
        )
        # We call PPO.train() directly (without model.learn()), so initialize logger.
        self.model.set_logger(sb3_configure_logger(folder=None, format_strings=["stdout"]))
        self.model.rollout_buffer.reset()

        self._steps_in_buffer = 0
        self._episode_start = True
        self._updates_total = 0
        self._transitions_total = 0

        log.info(
            "[PPOController] initialized (n_clients=%d, obs_dim=%d, n_steps=%d, batch_size=%d)",
            self.n_clients,
            self.obs_dim,
            n_steps,
            batch_size,
        )

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action, _ = self.model.predict(obs_arr, deterministic=deterministic)
        return np.asarray(action, dtype=np.int64).reshape(self.n_clients)

    def evaluate_action(self, obs: np.ndarray, action: np.ndarray) -> tuple[float, float]:
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action_arr = np.asarray(action, dtype=np.int64).reshape(1, -1)
        obs_tensor = obs_as_tensor(obs_arr, self.model.device)
        action_tensor = torch.as_tensor(action_arr, device=self.model.device)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            log_prob = distribution.log_prob(action_tensor)
            values = self.model.policy.predict_values(obs_tensor)
        value = float(values.detach().cpu().numpy().reshape(-1)[0])
        log_prob_val = float(log_prob.detach().cpu().numpy().reshape(-1)[0])
        return value, log_prob_val

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        value: float,
        log_prob: float,
    ) -> PPOUpdateInfo:
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        action_arr = np.asarray(action, dtype=np.int64).reshape(1, -1)
        reward_arr = np.array([float(reward)], dtype=np.float32)
        episode_start_arr = np.array([self._episode_start], dtype=np.float32)

        value_tensor = torch.tensor([value], dtype=torch.float32, device=self.model.device)
        log_prob_tensor = torch.tensor([log_prob], dtype=torch.float32, device=self.model.device)

        self.model.rollout_buffer.add(
            obs=obs_arr,
            action=action_arr,
            reward=reward_arr,
            episode_start=episode_start_arr,
            value=value_tensor,
            log_prob=log_prob_tensor,
        )
        self._steps_in_buffer += 1
        self._transitions_total += 1
        self._episode_start = bool(done)

        update_applied = 0
        if self._steps_in_buffer >= self.model.n_steps:
            next_obs_arr = np.asarray(next_obs, dtype=np.float32).reshape(1, -1)
            next_obs_tensor = obs_as_tensor(next_obs_arr, self.model.device)
            with torch.no_grad():
                next_values = self.model.policy.predict_values(next_obs_tensor)

            dones_arr = np.array([float(done)], dtype=np.float32)
            self.model.rollout_buffer.compute_returns_and_advantage(
                last_values=next_values,
                dones=dones_arr,
            )
            self.model.train()
            self.model.rollout_buffer.reset()
            self._steps_in_buffer = 0
            self._updates_total += 1
            update_applied = 1
        elif done and self._steps_in_buffer > 0:
            # Buffer tail is smaller than n_steps at terminal round; drop safely.
            self.model.rollout_buffer.reset()
            self._steps_in_buffer = 0

        return PPOUpdateInfo(
            update_applied=update_applied,
            updates_total=self._updates_total,
            transitions_total=self._transitions_total,
            buffer_size=self._steps_in_buffer,
        )
