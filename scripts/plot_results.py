"""
scripts/plot_results.py
Stage 17 — Paper-aligned plots from per-round JSON logs.

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Generates 5 plots (matches paper figures):
    accuracy.png          — global accuracy per round
    loss.png              — mean train loss per round
    dropout.png           — dropout fraction per round
    quant_distribution.png — stacked bar of quantization methods per round
    reward.png            — PPO reward per round (if available)

Defensive design:
    - Never crashes on missing fields (all accesses use .get() with defaults)
    - Works on any subset of rounds (partial runs)
    - Handles empty or malformed JSONs gracefully

Usage:
    python3 scripts/plot_results.py outputs/metrics/run_20260305_130742
    python3 scripts/plot_results.py outputs/metrics/run_20260305_130742 --show
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("plot_results")

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for Docker/server
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    log.warning("matplotlib not installed — plots will not be generated")


# ── Colour palette (paper-aligned: professional, print-friendly) ──────────────

COLOURS = {
    "accuracy":   "#2196F3",   # blue  — accuracy curve
    "loss":       "#F44336",   # red   — loss curve
    "dropout":    "#FF9800",   # amber — dropout fraction
    "reward":     "#4CAF50",   # green — PPO reward
    "fp32":       "#4CAF50",   # green
    "fp16":       "#2196F3",   # blue
    "bf16":       "#00ACC1",   # cyan
    "static_int8": "#FF9800",  # amber
    "qat_int8_train": "#FFC107",  # yellow
    "fp16_fallback": "#9C27B0",  # purple
    "bf16_fallback": "#7E57C2",  # deep purple
    "fp32_fallback": "#F44336",  # red
    "other":      "#9E9E9E",   # grey
}

QUANT_LABELS = {
    "fp32":           "FP32",
    "fp16":           "FP16",
    "bf16":           "BF16",
    "static_int8":    "INT8 (static)",
    "qat_int8_train": "INT8 (QAT train)",
    "fp16_fallback":  "FP16 (fallback)",
    "bf16_fallback":  "BF16 (fallback)",
    "fp32_fallback":  "FP32 (fallback)",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_round_logs(run_dir: str) -> List[Dict[str, Any]]:
    """Load all round JSON logs from a run directory, sorted by round number."""
    pattern = str(Path(run_dir) / "round_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        log.warning(f"No round_*.json files found in {run_dir}")
        return []

    logs = []
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
            logs.append(d)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Skipping malformed log {p}: {e}")

    log.info(f"Loaded {len(logs)} round logs from {run_dir}")
    return logs


def _safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Get a value from a dict, returning default if missing or None."""
    v = d.get(key, default)
    return v if v is not None else default


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_accuracy(logs: List[dict], out_path: Path, show: bool = False) -> None:
    """Plot global accuracy per round."""
    rounds = [_safe_get(d, "round", i + 1) for i, d in enumerate(logs)]
    accs = [_safe_get(d, "global_accuracy", float("nan")) for d in logs]
    deltas = [_safe_get(d, "accuracy_delta", float("nan")) for d in logs]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    line1, = ax1.plot(rounds, accs,   color=COLOURS["accuracy"],
                      linewidth=2.0, marker="o", markersize=4,
                      label="Global Accuracy")
    line2, = ax2.plot(rounds, deltas, color="#90CAF9",
                      linewidth=1.0, linestyle="--", marker="s", markersize=3,
                      alpha=0.7, label="ΔAcc (round-over-round)")

    ax1.set_xlabel("FL Round", fontsize=11)
    ax1.set_ylabel("Global Accuracy", fontsize=11, color=COLOURS["accuracy"])
    ax2.set_ylabel("ΔAcc", fontsize=11, color="#90CAF9")
    ax1.set_ylim(0, 1)
    ax1.set_title("Global Accuracy per FL Round (CIFAR-10, MobileNetV2)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    lines = [line1, line2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_loss(logs: List[dict], out_path: Path, show: bool = False) -> None:
    """Plot mean training loss per round."""
    rounds = [_safe_get(d, "round", i + 1) for i, d in enumerate(logs)]
    losses = [_safe_get(d, "mean_train_loss", float("nan")) for d in logs]
    glosses = [_safe_get(d, "global_loss", float("nan")) for d in logs]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, losses,  color=COLOURS["loss"],
            linewidth=2.0, marker="o", markersize=4, label="Mean Train Loss (clients)")
    ax.plot(rounds, glosses, color="#EF9A9A",
            linewidth=1.5, linestyle="--", marker="s", markersize=3,
            alpha=0.8, label="Server Global Loss")

    ax.set_xlabel("FL Round", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training Loss per FL Round", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_dropout(logs: List[dict], out_path: Path, show: bool = False) -> None:
    """Plot dropout fraction per round."""
    rounds = [_safe_get(d, "round", i + 1) for i, d in enumerate(logs)]
    fracs = [_safe_get(d, "dropout_fraction", 0.0) for d in logs]
    # Count actual dropped clients per round
    n_dropped = [len(_safe_get(d, "dropout_clients", [])) for d in logs]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.bar(rounds, fracs, color=COLOURS["dropout"], alpha=0.7, label="Dropout Fraction")
    ax2.plot(rounds, n_dropped, color="#E65100",
             linewidth=1.5, marker="D", markersize=4, label="# Dropped Clients")

    ax1.set_xlabel("FL Round", fontsize=11)
    ax1.set_ylabel("Dropout Fraction", fontsize=11, color=COLOURS["dropout"])
    ax2.set_ylabel("# Dropped Clients", fontsize=11, color="#E65100")
    ax1.set_ylim(0, 1)
    ax1.set_title("Client Dropout per FL Round", fontsize=12)
    ax1.grid(True, alpha=0.3, axis="y")

    lines1 = [plt.Rectangle((0,0),1,1, color=COLOURS["dropout"], alpha=0.7)]
    lines2 = [plt.Line2D([0], [0], color="#E65100", linewidth=1.5, marker="D")]
    ax1.legend(lines1 + lines2, ["Dropout Fraction", "# Dropped"], loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_quant_distribution(logs: List[dict], out_path: Path, show: bool = False) -> None:
    """Stacked bar chart of quantization methods per round."""
    all_methods = [
        "fp32",
        "fp16",
        "bf16",
        "static_int8",
        "qat_int8_train",
        "fp16_fallback",
        "bf16_fallback",
        "fp32_fallback",
        "other",
    ]
    rounds = [_safe_get(d, "round", i + 1) for i, d in enumerate(logs)]

    # Count occurrences of each quant method per round
    method_counts: Dict[str, List[int]] = {m: [] for m in all_methods}
    for d in logs:
        actual = _safe_get(d, "actual_quant_method", {})
        counts = {m: 0 for m in all_methods}
        if isinstance(actual, dict):
            for method in actual.values():
                if isinstance(method, str) and method in counts:
                    counts[method] += 1
                else:
                    counts["other"] = counts.get("other", 0) + 1
        for m in all_methods:
            method_counts[m].append(counts[m])

    # Remove methods that never appear
    active = {m: v for m, v in method_counts.items() if any(c > 0 for c in v)}
    if not active:
        log.warning("  No quant method data found — skipping quant_distribution plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(rounds))
    for m, counts in active.items():
        color = COLOURS.get(m, COLOURS["other"])
        label = QUANT_LABELS.get(m, m)
        ax.bar(rounds, counts, bottom=bottom, color=color, alpha=0.85, label=label)
        bottom += np.array(counts, dtype=float)

    ax.set_xlabel("FL Round", fontsize=11)
    ax.set_ylabel("# Clients", fontsize=11)
    ax.set_title("Quantization Method Distribution per Round", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


def plot_reward(logs: List[dict], out_path: Path, show: bool = False) -> None:
    """Plot PPO reward per round (if reward data exists in logs)."""
    rounds = [_safe_get(d, "round", i + 1) for i, d in enumerate(logs)]
    rewards = [_safe_get(d, "reward", float("nan")) for d in logs]

    # Check if we have any real reward data
    has_reward = any(not (isinstance(r, float) and r != r) for r in rewards)
    if not has_reward:
        # reward may be embedded in round_log of FLEnv — check reward_components
        rewards = []
        for d in logs:
            rc = _safe_get(d, "reward_components", {})
            r = _safe_get(rc, "reward", float("nan"))
            rewards.append(r)
        has_reward = any(not (isinstance(r, float) and r != r) for r in rewards)

    fig, ax = plt.subplots(figsize=(10, 5))
    if has_reward:
        ax.plot(rounds, rewards, color=COLOURS["reward"],
                linewidth=2.0, marker="o", markersize=4, label="PPO Reward")
    else:
        ax.text(0.5, 0.5, "No reward data (non-PPO run)",
                ha="center", va="center", fontsize=13, color="grey",
                transform=ax.transAxes)

    ax.set_xlabel("FL Round", fontsize=11)
    ax.set_ylabel("Reward r_t = α·ΔAcc − β·Dropout − γ·Var(Q)", fontsize=10)
    ax.set_title("PPO Reward per FL Round", fontsize=12)
    if has_reward:
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    log.info(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-aligned plots from FL run metrics."
    )
    parser.add_argument("run_dir", help="Path to run directory (contains round_*.json)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for plots (default: <run_dir> with 'metrics' replaced by 'plots')"
    )
    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        log.error("Install matplotlib: pip install matplotlib")
        sys.exit(1)

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        log.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # Replace 'metrics' → 'plots' in path
        parts = list(run_dir.parts)
        for i, p in enumerate(parts):
            if p == "metrics":
                parts[i] = "plots"
                break
        out_dir = Path(*parts)

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Plots output: {out_dir}")

    # Load data
    logs = load_round_logs(str(run_dir))
    if not logs:
        log.error(f"No round logs found in {run_dir}")
        sys.exit(1)

    log.info(f"Generating 5 plots for {len(logs)} rounds...")

    # Generate all plots (never crash on missing data)
    try:
        plot_accuracy(logs,           out_dir / "accuracy.png",           args.show)
    except Exception as e:
        log.warning(f"  accuracy.png failed: {e}")

    try:
        plot_loss(logs,               out_dir / "loss.png",               args.show)
    except Exception as e:
        log.warning(f"  loss.png failed: {e}")

    try:
        plot_dropout(logs,            out_dir / "dropout.png",            args.show)
    except Exception as e:
        log.warning(f"  dropout.png failed: {e}")

    try:
        plot_quant_distribution(logs, out_dir / "quant_distribution.png", args.show)
    except Exception as e:
        log.warning(f"  quant_distribution.png failed: {e}")

    try:
        plot_reward(logs,             out_dir / "reward.png",             args.show)
    except Exception as e:
        log.warning(f"  reward.png failed: {e}")

    log.info("All plots generated.")


if __name__ == "__main__":
    main()
