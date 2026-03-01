"""
src/rl/agent.py — PPO training entrypoint

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Trains a PPO agent (stable-baselines3) where ONE FL ROUND = ONE env.step().

Usage:
    python -m src.rl.agent --config configs/exp1_ppo_smoke.yaml

Saves checkpoint: outputs/checkpoints/ppo_<runid>.zip
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.common.config import load_config

log = logging.getLogger(__name__)


def train_ppo(config_path: str) -> str:
    """
    Train PPO on FLEnv and save checkpoint.

    Args:
        config_path: Path to YAML config.

    Returns:
        Path to saved checkpoint .zip file.
    """
    from stable_baselines3 import PPO
    from src.rl.env import FLEnv

    cfg = load_config(config_path)
    log.info(
        f"[Agent] PPO training: {cfg.experiment.name}  "
        f"rounds/episode={cfg.experiment.rounds}  "
        f"total_timesteps={cfg.rl.total_timesteps}"
    )

    env = FLEnv(cfg)

    # n_steps must be >= 1 and <= episode_length (cfg.experiment.rounds)
    n_steps = max(1, cfg.rl.n_steps)
    batch_size = min(cfg.rl.batch_size, n_steps)

    model = PPO(
        policy=cfg.rl.policy,
        env=env,
        verbose=1,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=cfg.rl.gamma_discount,
        learning_rate=cfg.rl.learning_rate,
    )

    log.info(f"[Agent] PPO model created. n_steps={n_steps} batch_size={batch_size}")
    model.learn(total_timesteps=cfg.rl.total_timesteps)

    # Save checkpoint
    Path(cfg.rl.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    save_path = str(Path(cfg.rl.checkpoint_dir) / f"ppo_{run_id}")
    model.save(save_path)
    log.info(f"[Agent] PPO checkpoint saved: {save_path}.zip")
    return f"{save_path}.zip"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Train PPO agent for Adaptive FL")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    train_ppo(args.config)
