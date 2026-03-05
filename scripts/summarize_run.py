"""
scripts/summarize_run.py
Stage 17 — Aggregate per-round JSON logs into a summary.json for a FL run.

Paper: Adaptive FL with PPO-based Client and Quantization Selection

Reads all round_*.json files in a run directory and produces summary.json with:
  - Basic metadata (run_id, config, n_rounds)
  - Final accuracy and loss
  - Per-round accuracy / loss / dropout arrays
  - Dropout statistics
  - Quantization method distribution
  - Reward statistics (if present)
  - Paper compliance check (required fields present)

Usage:
    python3 scripts/summarize_run.py outputs/metrics/run_20260305_130742
    python3 scripts/summarize_run.py outputs/metrics/run_20260305_130742 --verbose
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("summarize_run")


# Required per-round fields per SPEC.md §8
REQUIRED_ROUND_FIELDS = {
    "round",
    "selected_clients",
    "quant_assignments",
    "actual_quant_method",
    "dropout_clients",
    "dropout_fraction",
    "global_accuracy",
    "accuracy_delta",
}


def _safe_get(d: dict, key: str, default: Any = None) -> Any:
    v = d.get(key, default)
    return v if v is not None else default


def _mean(vals: list) -> Optional[float]:
    v = [x for x in vals if isinstance(x, (int, float)) and x == x]
    return float(sum(v) / len(v)) if v else None


def _nanmin(vals: list) -> Optional[float]:
    v = [x for x in vals if isinstance(x, (int, float)) and x == x]
    return float(min(v)) if v else None


def _nanmax(vals: list) -> Optional[float]:
    v = [x for x in vals if isinstance(x, (int, float)) and x == x]
    return float(max(v)) if v else None


def load_round_logs(run_dir: str) -> List[Dict[str, Any]]:
    pattern = str(Path(run_dir) / "round_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return []
    logs = []
    for p in paths:
        try:
            with open(p) as f:
                logs.append(json.load(f))
        except Exception as e:
            log.warning(f"  Skipping {p}: {e}")
    return logs


def summarize(run_dir: str, verbose: bool = False) -> dict:
    run_path = Path(run_dir).resolve()
    logs = load_round_logs(str(run_path))
    n_rounds = len(logs)

    if n_rounds == 0:
        log.warning(f"No round logs found in {run_dir}")
        return {"status": "empty", "run_dir": str(run_path), "rounds_found": 0}

    # ── Paper compliance check ────────────────────────────────────────────────
    compliance: Dict[str, list] = {"ok": [], "missing": []}
    for d in logs:
        missing = REQUIRED_ROUND_FIELDS - set(d.keys())
        rnd = _safe_get(d, "round", "?")
        if missing:
            compliance["missing"].append({"round": rnd, "missing_fields": sorted(missing)})
        else:
            compliance["ok"].append(rnd)

    # ── Metrics per round ─────────────────────────────────────────────────────
    acc_series    = [_safe_get(d, "global_accuracy",  float("nan")) for d in logs]
    loss_series   = [_safe_get(d, "mean_train_loss",  float("nan")) for d in logs]
    gloss_series  = [_safe_get(d, "global_loss",      float("nan")) for d in logs]
    drop_series   = [_safe_get(d, "dropout_fraction", 0.0)          for d in logs]
    delta_series  = [_safe_get(d, "accuracy_delta",   float("nan")) for d in logs]
    reward_series = [_safe_get(d, "reward",           float("nan")) for d in logs]

    # Final round values
    last = logs[-1]
    final_acc  = _safe_get(last, "global_accuracy")
    final_loss = _safe_get(last, "global_loss")

    # ── Quant distribution over all rounds ───────────────────────────────────
    quant_counter: Counter = Counter()
    for d in logs:
        actual = _safe_get(d, "actual_quant_method", {})
        if isinstance(actual, dict):
            for m in actual.values():
                quant_counter[str(m)] += 1

    # ── Dropout statistics ────────────────────────────────────────────────────
    total_dropout_events = sum(
        len(_safe_get(d, "dropout_clients", [])) for d in logs
    )
    dropout_clients_all: Counter = Counter()
    for d in logs:
        for cid in _safe_get(d, "dropout_clients", []):
            dropout_clients_all[str(cid)] += 1

    # ── Selected client statistics ────────────────────────────────────────────
    n_selected_per_round = [
        len(_safe_get(d, "selected_clients", [])) for d in logs
    ]

    # ── Build summary ─────────────────────────────────────────────────────────
    summary = {
        "run_dir": str(run_path),
        "run_id": run_path.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_rounds": n_rounds,
        "status": "complete" if not compliance["missing"] else "incomplete",

        # Final metrics
        "final_accuracy": final_acc,
        "final_loss": final_loss,

        # Accuracy statistics
        "accuracy": {
            "series": [a if a == a else None for a in acc_series],   # NaN → null
            "final": final_acc,
            "max":   _nanmax(acc_series),
            "mean":  _mean(acc_series),
        },
        "loss": {
            "train_loss_series": [x if x == x else None for x in loss_series],
            "global_loss_series": [x if x == x else None for x in gloss_series],
        },
        "accuracy_delta": {
            "series": [x if x == x else None for x in delta_series],
            "mean":   _mean(delta_series),
        },

        # Dropout
        "dropout": {
            "fraction_series":     drop_series,
            "mean_fraction":       _mean(drop_series),
            "max_fraction":        _nanmax(drop_series),
            "total_events":        total_dropout_events,
            "most_dropped_client": dropout_clients_all.most_common(1)[0] if dropout_clients_all else None,
            "per_client_counts":   dict(dropout_clients_all),
        },

        # Client selection
        "selection": {
            "n_selected_per_round": n_selected_per_round,
            "mean_selected": _mean(n_selected_per_round),
        },

        # Quantization distribution (cumulative over all rounds)
        "quantization": {
            "method_counts":  dict(quant_counter),
            "method_fractions": {
                m: c / max(1, sum(quant_counter.values()))
                for m, c in quant_counter.items()
            },
        },

        # Reward
        "reward": {
            "series": [x if x == x else None for x in reward_series],
            "mean":   _mean(reward_series),
        },

        # Paper compliance
        "paper_compliance": {
            "rounds_with_all_required_fields": len(compliance["ok"]),
            "rounds_missing_fields": len(compliance["missing"]),
            "missing_details": compliance["missing"] if verbose else (
                compliance["missing"][:3] if compliance["missing"] else []
            ),
            "required_fields": sorted(REQUIRED_ROUND_FIELDS),
        },
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize FL run from round JSON logs."
    )
    parser.add_argument("run_dir", help="Path to run directory (contains round_*.json)")
    parser.add_argument("--verbose", action="store_true",
                        help="Include full missing-field details in output")
    parser.add_argument("--out", default=None,
                        help="Output path for summary.json (default: <run_dir>/summary.json)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        log.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    log.info(f"Summarizing: {run_dir}")
    summary = summarize(str(run_dir), verbose=args.verbose)

    out_path = Path(args.out) if args.out else (run_dir / "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Summary written: {out_path}")
    log.info(f"  Rounds completed: {summary['num_rounds']}")
    log.info(f"  Final accuracy  : {summary.get('final_accuracy')}")
    log.info(f"  Final loss      : {summary.get('final_loss')}")
    log.info(f"  Mean dropout    : {summary['dropout']['mean_fraction']}")
    log.info(f"  Status          : {summary['status']}")

    # Paper compliance summary
    n_ok = summary["paper_compliance"]["rounds_with_all_required_fields"]
    n_bad = summary["paper_compliance"]["rounds_missing_fields"]
    if n_bad == 0:
        log.info(f"  SPEC compliance : ✅ All {n_ok} rounds have required fields")
    else:
        log.warning(
            f"  SPEC compliance : ⚠️ {n_bad} rounds missing required fields "
            f"(see paper_compliance.missing_details in summary.json)"
        )


if __name__ == "__main__":
    main()
