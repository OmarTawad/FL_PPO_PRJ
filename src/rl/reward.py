"""
src/rl/reward.py — Reward function for PPO agent

Paper reward (SPEC.md §6):
    r_t = α · ΔAcc_t − β · Dropout_t − γ · Var(Q_t)

Where:
    ΔAcc_t     = Acc_t − Acc_{t-1}  (single-round accuracy delta)
    Dropout_t  = (#dropped in selected) / max(1, #selected)
    Var(Q_t)   = numpy variance of quant bits for selected clients ONLY;
                 if 0 or 1 selected client → 0.0

Args (to compute_reward):
    acc_t:                   Global accuracy this round
    acc_prev:                Global accuracy previous round (0.0 at round 0)
    selected_client_ids:     List of CID strings that were selected (action != skip)
    dropout_client_ids:      List of CID strings that dropped this round
    quant_assignments_bits:  Dict {cid_str: bits} for selected clients
    alpha, beta, gamma:      Reward hyperparameters from cfg.rl
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def compute_reward(
    acc_t: float,
    acc_prev: float,
    selected_client_ids: List[str],
    dropout_client_ids: List[str],
    quant_assignments_bits: Dict[str, int],
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
) -> Tuple[float, dict]:
    """
    Compute the scalar PPO reward and a diagnostics dict.

    Returns:
        (reward: float, components: dict)
        components keys: delta_acc, dropout_rate, quant_variance, reward
    """
    # ── ΔAcc ──────────────────────────────────────────────────────────────────
    delta_acc = float(acc_t) - float(acc_prev)

    # ── Dropout rate ──────────────────────────────────────────────────────────
    n_selected = max(1, len(selected_client_ids))
    # Deduplicate dropout_client_ids (a client can't drop twice)
    dropped_set = set(str(d) for d in dropout_client_ids)
    selected_set = set(str(s) for s in selected_client_ids)
    # Only count as dropout if the client was actually selected
    effective_drops = len(dropped_set & selected_set)
    dropout_rate = float(effective_drops) / float(n_selected)

    # ── Var(Q_t) ──────────────────────────────────────────────────────────────
    selected_bits = [
        int(quant_assignments_bits[cid])
        for cid in selected_client_ids
        if cid in quant_assignments_bits
    ]
    if len(selected_bits) <= 1:
        quant_variance = 0.0
    else:
        quant_variance = float(np.var(selected_bits, ddof=0))

    # ── Combined reward ────────────────────────────────────────────────────────
    reward = (
        alpha * delta_acc
        - beta  * dropout_rate
        - gamma * quant_variance
    )

    components = {
        "delta_acc":      delta_acc,
        "dropout_rate":   dropout_rate,
        "quant_variance": quant_variance,
        "reward":         reward,
        "alpha":          alpha,
        "beta":           beta,
        "gamma":          gamma,
    }
    return reward, components
