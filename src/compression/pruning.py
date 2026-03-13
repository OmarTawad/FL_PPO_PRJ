"""
src/compression/pruning.py — Magnitude pruning helpers for client-local training.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch.nn as nn
import torch.nn.utils.prune as prune


def resolve_pruning_amount(
    default_amount: float,
    per_client: Dict[int, float],
    client_id: int,
) -> float:
    """Resolve pruning amount for a client from global default + per-client overrides."""
    if client_id in per_client:
        return float(per_client[client_id])
    return float(default_amount)


def apply_magnitude_unstructured_pruning(
    model: nn.Module,
    amount: float,
    target_modules: List[str],
) -> Tuple[bool, str, int, float]:
    """
    Apply L1 unstructured pruning to model weights.

    Returns:
      (applied, skip_reason, target_module_count, effective_pruned_fraction)
    """
    if amount <= 0.0:
        return False, "amount_zero", 0, 0.0

    target_set = {str(t).strip().lower() for t in target_modules}
    allow_conv = "conv2d" in target_set
    allow_linear = "linear" in target_set

    candidates: List[nn.Module] = []
    for module in model.modules():
        if allow_conv and isinstance(module, nn.Conv2d):
            candidates.append(module)
        elif allow_linear and isinstance(module, nn.Linear):
            candidates.append(module)

    if not candidates:
        return False, "no_target_modules", 0, 0.0

    for module in candidates:
        if hasattr(module, "weight_mask"):
            # Already pruned in this model instance.
            continue
        prune.l1_unstructured(module, name="weight", amount=float(amount))

    total_weights = 0
    total_kept = 0.0
    for module in candidates:
        mask = getattr(module, "weight_mask", None)
        if mask is None:
            continue
        total_weights += int(mask.numel())
        total_kept += float(mask.sum().item())

    if total_weights <= 0:
        return False, "mask_not_created", len(candidates), 0.0

    pruned_fraction = max(0.0, min(1.0, 1.0 - (total_kept / float(total_weights))))
    return True, "", len(candidates), pruned_fraction
