"""
src/heterogeneity/profiles.py — Client hardware profiles for FL heterogeneity simulation

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Defines four hardware tiers that represent the VM Docker client profiles:
    strong       — capable client, rarely drops
    medium       — typical client, occasionally strains on INT8
    weak         — resource-constrained, struggles with large batches
    extreme_weak — minimal resources, high dropout probability

Each profile exposes:
    - Resource limits (cpu_cores, mem_limit_mb)
    - reliability: base probability of completing a round (FP32 reference)
    - dropout_p: per-quant-level dropout probability dict {32: p, 16: p, 8: p}
      Higher quant level (INT8) → higher memory pressure → higher dropout prob
    - state_features(): returns dict of 7 per-client state features (SPEC.md §2.1)

Usage:
    from src.heterogeneity.profiles import get_profile, stochastic_dropout_probability
    profile = get_profile("weak")
    p = stochastic_dropout_probability(profile, quant_bits=8)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal

# Valid profile names
ProfileName = Literal["strong", "medium", "weak", "extreme_weak"]

# Valid quant bit widths (matches PPO action space, SPEC.md §3.2)
VALID_BITS = (32, 16, 8)


@dataclass(frozen=True)
class ClientProfile:
    """
    Immutable hardware profile for a simulated FL client.

    Attributes:
        name:          Identifier string (one of PROFILE_NAMES)
        cpu_cores:     Virtual CPU cores available
        mem_limit_mb:  RAM limit in MB (matches Docker --memory)
        reliability:   Base probability of completing a round without OOM/crash
                       at FP32 with the default batch size (SPEC.md §2.1 feature).
        dropout_p:     Dropout probability per quant level {32: p, 16: p, 8: p}.
                       Probabilities increase with memory pressure of INT8.
    """
    name: str
    cpu_cores: int
    mem_limit_mb: int
    reliability: float
    dropout_p: Dict[int, float]

    def __post_init__(self):
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(f"reliability must be in [0,1], got {self.reliability}")
        for bits in VALID_BITS:
            if bits not in self.dropout_p:
                raise ValueError(f"dropout_p missing key {bits}")
            p = self.dropout_p[bits]
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"dropout_p[{bits}]={p} not in [0,1]")

    def state_features(
        self,
        mem_used_frac: float = 0.5,
        last_dropout: int = 0,
        consecutive_drops: int = 0,
        last_quant: int = 32,
    ) -> dict:
        """
        Return the 7 per-client state features for the PPO state vector.
        (SPEC.md §2.1 — Per-client features)

        Args:
            mem_used_frac:      Current memory used fraction [0,1]
            last_dropout:       1 if client dropped last round, else 0
            consecutive_drops:  Number of consecutive rounds dropped
            last_quant:         Quant bits used last round (0=not selected)

        Returns:
            Dict mapping feature name → value (float or int)
        """
        return {
            "cpu_cores_i":         self.cpu_cores,
            "mem_limit_mb_i":      self.mem_limit_mb,
            "mem_used_frac_i":     mem_used_frac,
            "reliability_i":       self.reliability,
            "last_dropout_i":      last_dropout,
            "consecutive_drops_i": consecutive_drops,
            "last_quant_i":        last_quant,
        }


# ─── Profile definitions ──────────────────────────────────────────────────────
# Dropout probabilities are derived from reliability + quant memory pressure:
#   p(32) ≈ 1 - reliability              (baseline; failure is mostly HW noise)
#   p(16) ≈ p(32) * 1.5                  (slight memory overhead for conversion)
#   p(8)  ≈ p(32) * 3.0                  (static INT8 adds calibration overhead)
# Values are rounded to 2 decimal places for reproducibility.

_PROFILES: Dict[str, ClientProfile] = {
    "strong": ClientProfile(
        name="strong",
        cpu_cores=4,
        mem_limit_mb=2048,
        reliability=0.99,
        dropout_p={32: 0.01, 16: 0.02, 8: 0.03},
    ),
    "medium": ClientProfile(
        name="medium",
        cpu_cores=2,
        mem_limit_mb=1024,
        reliability=0.95,
        dropout_p={32: 0.05, 16: 0.08, 8: 0.15},
    ),
    "weak": ClientProfile(
        name="weak",
        cpu_cores=1,
        mem_limit_mb=512,
        reliability=0.70,
        dropout_p={32: 0.10, 16: 0.20, 8: 0.35},
    ),
    "extreme_weak": ClientProfile(
        name="extreme_weak",
        cpu_cores=1,
        mem_limit_mb=256,
        reliability=0.50,
        dropout_p={32: 0.30, 16: 0.45, 8: 0.60},
    ),
}

PROFILE_NAMES = tuple(_PROFILES.keys())


def get_profile(name: str) -> ClientProfile:
    """
    Retrieve a ClientProfile by name.

    Args:
        name: One of "strong", "medium", "weak", "extreme_weak".

    Returns:
        Corresponding ClientProfile (immutable).

    Raises:
        KeyError: If name is not a recognised profile.
    """
    if name not in _PROFILES:
        raise KeyError(
            f"Unknown profile '{name}'. Valid: {list(_PROFILES.keys())}"
        )
    return _PROFILES[name]


def stochastic_dropout_probability(profile: ClientProfile, quant_bits: int) -> float:
    """
    Return the per-round dropout probability for this client profile and
    quantization level.

    Used by simulate_dropout() and the heterogeneity config validator.

    Args:
        profile:    ClientProfile instance.
        quant_bits: Quantization bits (32, 16, or 8).

    Returns:
        Float in [0, 1].

    Raises:
        ValueError: If quant_bits not in {32, 16, 8}.
    """
    if quant_bits not in VALID_BITS:
        raise ValueError(f"quant_bits must be in {VALID_BITS}, got {quant_bits}")
    return profile.dropout_p[quant_bits]


def list_all_profiles() -> Dict[str, ClientProfile]:
    """Return a copy of all registered profiles (name → ClientProfile)."""
    return dict(_PROFILES)
