"""
src/rl/state.py — PPO state vector builder

Paper: Adaptive FL with PPO-based Client and Quantization Selection

State vector (SPEC.md §3.1):
  Per-client features (7 × N_clients):
    cpu_cores_i, mem_limit_mb_i, mem_used_frac_i,
    reliability_i, last_dropout_i, consecutive_drops_i, last_quant_i
  Global features (3):
    global_accuracy_{t-1}, ΔAcc_{t-1}, round_fraction = t / T

Total dimension: N * 7 + 3
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.heterogeneity.dropout import DropoutTracker

# Feature indices within the per-client block (7 features each)
_F_CPU          = 0
_F_MEM          = 1
_F_MEM_USED     = 2
_F_RELIABILITY  = 3
_F_LAST_DROP    = 4
_F_CONSEC_DROPS = 5
_F_LAST_QUANT   = 6
_N_PER_CLIENT   = 7

# Global feature indices (appended after all per-client blocks)
_G_PREV_ACC     = 0
_G_PREV_DELTA   = 1
_G_ROUND_FRAC   = 2
_N_GLOBAL       = 3


class StateBuilder:
    """
    Assembles the flat observation vector for the PPO agent each FL round.

    Dimension: N * 7 + 3 (matches Config.state_dim).
    All values are FP32. Features are NOT normalised here — the PPO policy
    network (MlpPolicy) handles normalisation via its architecture.
    """

    def __init__(
        self,
        n_clients: int,
        profiles: List,            # List[ClientProfileConfig]
        dropout_tracker: DropoutTracker,
        total_rounds: int,
    ):
        self.n_clients = n_clients
        self.profiles = profiles
        self.dropout_tracker = dropout_tracker
        self.total_rounds = max(1, total_rounds)
        self.obs_dim = n_clients * _N_PER_CLIENT + _N_GLOBAL

        # Pre-compute numeric reliability from heterogeneity profiles once.
        # profiles[i].profile is a string like "strong"; get_profile returns the
        # ClientProfile dataclass that has .reliability as a float.
        from src.heterogeneity.profiles import get_profile as _get_hp
        self._reliabilities: List[float] = []
        for prof_cfg in profiles:
            try:
                hp = _get_hp(prof_cfg.profile)
                self._reliabilities.append(float(hp.reliability))
            except (KeyError, AttributeError):
                self._reliabilities.append(1.0)

    def build(
        self,
        round_idx: int,
        prev_global_acc: float,
        prev_acc_delta: float,
        last_quant_bits: Dict[int, int],
        mem_used_frac: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """
        Build and return the flat observation vector.

        Args:
            round_idx:       Current FL round index (0 = before round 1, ...).
            prev_global_acc: Global accuracy from the previous round (0.0 at start).
            prev_acc_delta:  Accuracy delta from the previous round (0.0 at start).
            last_quant_bits: {client_id: bits_used_last_round}; 0 = never selected.
            mem_used_frac:   {client_id: fraction_used} in [0,1]. Default 0.5.

        Returns:
            np.ndarray of shape (N*7+3,), dtype=float32.
        """
        obs = np.zeros(self.obs_dim, dtype=np.float32)

        for i, prof_cfg in enumerate(self.profiles):
            dropout_state = self.dropout_tracker.get_state(i)
            mem_frac = (mem_used_frac or {}).get(i, 0.5)
            base = i * _N_PER_CLIENT

            obs[base + _F_CPU]          = float(prof_cfg.cpu_cores)
            obs[base + _F_MEM]          = float(prof_cfg.mem_limit_mb)
            obs[base + _F_MEM_USED]     = float(mem_frac)
            obs[base + _F_RELIABILITY]  = self._reliabilities[i]   # float, not string
            obs[base + _F_LAST_DROP]    = float(dropout_state["last_dropout"])
            obs[base + _F_CONSEC_DROPS] = float(dropout_state["consecutive_drops"])
            obs[base + _F_LAST_QUANT]   = float(last_quant_bits.get(i, 0))

        # Global features
        global_base = self.n_clients * _N_PER_CLIENT
        obs[global_base + _G_PREV_ACC]   = float(prev_global_acc)
        obs[global_base + _G_PREV_DELTA] = float(prev_acc_delta)
        obs[global_base + _G_ROUND_FRAC] = float(round_idx) / float(self.total_rounds)

        return obs
