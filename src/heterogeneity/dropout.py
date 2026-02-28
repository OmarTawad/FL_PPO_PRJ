"""
src/heterogeneity/dropout.py — Stochastic dropout simulation for FL clients

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Implements:
    - DropoutTracker: per-client history tracking (consecutive_drops, last_dropout)
    - simulate_dropout(): returns whether a client drops this round, based on
      the stochastic dropout probability from its profile and assigned quant level

"Dropout" in this context means the client fails to return a model update:
    - OOM during local training
    - Crash or timeout due to resource pressure
    - Explicitly denied due to safety budget (future)

The dropout outcome is sampled from a Bernoulli distribution with probability
p = stochastic_dropout_probability(profile, quant_bits) (SPEC.md §5).

Usage:
    from src.heterogeneity.dropout import DropoutTracker, simulate_dropout
    from src.heterogeneity.profiles import get_profile

    tracker = DropoutTracker(n_clients=3)
    profile = get_profile("weak")

    dropped = simulate_dropout(client_id=1, profile=profile, quant_bits=8)
    tracker.record(client_id=1, dropped=dropped)
    state = tracker.get_state(client_id=1)
    # state = {"last_dropout": 1, "consecutive_drops": 1}
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.heterogeneity.profiles import (
    ClientProfile,
    stochastic_dropout_probability,
)

log = logging.getLogger(__name__)


# ─── DropoutTracker ───────────────────────────────────────────────────────────

@dataclass
class DropoutTracker:
    """
    Tracks per-client dropout history across FL rounds.

    Maintains:
        last_dropout[i]       — 1 if client i dropped last round, else 0
        consecutive_drops[i]  — count of consecutive rounds dropped

    These feed directly into the PPO state vector's per-client features
    (SPEC.md §2.1: last_dropout_i, consecutive_drops_i).
    """
    n_clients: int
    last_dropout: List[int] = field(default_factory=list)
    consecutive_drops: List[int] = field(default_factory=list)
    total_drops: List[int] = field(default_factory=list)
    total_rounds: List[int] = field(default_factory=list)

    def __post_init__(self):
        if self.n_clients < 1:
            raise ValueError(f"n_clients must be >= 1, got {self.n_clients}")
        if not self.last_dropout:
            self.last_dropout = [0] * self.n_clients
        if not self.consecutive_drops:
            self.consecutive_drops = [0] * self.n_clients
        if not self.total_drops:
            self.total_drops = [0] * self.n_clients
        if not self.total_rounds:
            self.total_rounds = [0] * self.n_clients

    def record(self, client_id: int, dropped: bool) -> None:
        """
        Record the outcome of one round for a client.

        Args:
            client_id: Integer index in [0, n_clients).
            dropped:   True if the client dropped this round.
        """
        self._validate_id(client_id)
        self.total_rounds[client_id] += 1
        if dropped:
            self.last_dropout[client_id] = 1
            self.consecutive_drops[client_id] += 1
            self.total_drops[client_id] += 1
        else:
            self.last_dropout[client_id] = 0
            self.consecutive_drops[client_id] = 0

    def record_not_selected(self, client_id: int) -> None:
        """
        Record that a client was NOT selected this round (action=skip).
        Does NOT reset consecutive_drops (only actual completion resets it).
        Does NOT increment total_rounds (client was not assigned work).
        """
        self._validate_id(client_id)
        # Per SPEC.md: last_quant_i = 0 for unselected clients; dropout state unchanged
        # We leave last_dropout as-is — if they dropped last round, they're still "dropped"

    def get_state(self, client_id: int) -> dict:
        """
        Return the dropout-related state features for a client.

        Returns:
            Dict with keys: last_dropout, consecutive_drops, dropout_rate
        """
        self._validate_id(client_id)
        total = self.total_rounds[client_id]
        dropout_rate = (
            self.total_drops[client_id] / total if total > 0 else 0.0
        )
        return {
            "last_dropout":      self.last_dropout[client_id],
            "consecutive_drops": self.consecutive_drops[client_id],
            "dropout_rate":      dropout_rate,
        }

    def reset(self) -> None:
        """Reset all per-client tracking to zero (new experiment)."""
        self.last_dropout      = [0] * self.n_clients
        self.consecutive_drops = [0] * self.n_clients
        self.total_drops       = [0] * self.n_clients
        self.total_rounds      = [0] * self.n_clients

    def _validate_id(self, client_id: int) -> None:
        if not (0 <= client_id < self.n_clients):
            raise IndexError(
                f"client_id={client_id} out of range [0, {self.n_clients})"
            )


# ─── simulate_dropout ─────────────────────────────────────────────────────────

def simulate_dropout(
    client_id: int,
    profile: ClientProfile,
    quant_bits: int,
    rng: Optional[random.Random] = None,
) -> bool:
    """
    Sample whether a client drops out this round.

    Dropout probability = stochastic_dropout_probability(profile, quant_bits).
    This is a Bernoulli sample: True = dropped, False = completed.

    Args:
        client_id:  Client index (used for logging only).
        profile:    ClientProfile determining the dropout probability.
        quant_bits: Quantization bits assigned this round (32, 16, or 8).
        rng:        Optional random.Random instance for reproducibility.
                    If None, uses the global random module.

    Returns:
        True if the client drops this round, False if it completes.
    """
    p = stochastic_dropout_probability(profile, quant_bits)
    u = (rng.random() if rng is not None else random.random())
    dropped = u < p
    log.debug(
        f"client={client_id} profile={profile.name} quant={quant_bits} "
        f"p={p:.3f} u={u:.4f} dropped={dropped}"
    )
    return dropped


def simulate_round_dropout(
    client_ids: List[int],
    profiles: List[ClientProfile],
    quant_assignments: Dict[int, int],
    tracker: Optional[DropoutTracker] = None,
    rng: Optional[random.Random] = None,
) -> Dict[int, bool]:
    """
    Simulate dropout for all selected clients in one FL round.

    Args:
        client_ids:       List of selected client IDs.
        profiles:         Parallel list of their hardware profiles.
        quant_assignments: Dict {client_id: quant_bits} for selected clients.
        tracker:          Optional DropoutTracker to record outcomes.
        rng:              Optional RNG for reproducibility.

    Returns:
        Dict {client_id: dropped} — True if client dropped this round.
    """
    if len(client_ids) != len(profiles):
        raise ValueError(
            f"client_ids length ({len(client_ids)}) != "
            f"profiles length ({len(profiles)})"
        )
    results: Dict[int, bool] = {}
    for cid, profile in zip(client_ids, profiles):
        bits = quant_assignments.get(cid, 32)
        dropped = simulate_dropout(cid, profile, bits, rng=rng)
        results[cid] = dropped
        if tracker is not None:
            tracker.record(cid, dropped)
    n_dropped = sum(results.values())
    log.info(
        f"Round dropout: {n_dropped}/{len(client_ids)} clients dropped "
        f"(ids={[cid for cid, d in results.items() if d]})"
    )
    return results
