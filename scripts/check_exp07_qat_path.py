#!/usr/bin/env python3
"""Post-run checker for Exp07 QAT path activation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _latest_run_dir(metrics_dir: Path) -> Path | None:
    run_dirs = sorted(
        (p for p in metrics_dir.glob("run_*") if p.is_dir()),
        key=lambda p: p.name,
    )
    return run_dirs[-1] if run_dirs else None


def _load_round_logs(run_dir: Path) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("round_*.json")):
        with open(path, "r") as f:
            logs.append(json.load(f))
    return logs


def _resolve_run_dir(run_dir_arg: str, metrics_dir_arg: str) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")
        return run_dir

    metrics_dir = Path(metrics_dir_arg)
    if not metrics_dir.is_dir():
        raise FileNotFoundError(f"metrics_dir not found: {metrics_dir}")
    run_dir = _latest_run_dir(metrics_dir)
    if run_dir is None:
        raise FileNotFoundError(f"no run_* directories found in {metrics_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Exp07 QAT run artifacts")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--metrics-dir", default="outputs/exp07_qat_int8/metrics")
    parser.add_argument("--report-path", default="")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir, args.metrics_dir)
    round_logs = _load_round_logs(run_dir)

    violations: list[dict[str, Any]] = []
    int8_observations = 0

    for entry in round_logs:
        round_id = int(entry.get("round", -1))
        quant_assignments = entry.get("quant_assignments", {}) or {}
        actual_methods = entry.get("actual_quant_method", {}) or {}

        for cid_raw, bits_raw in quant_assignments.items():
            cid = str(cid_raw)
            try:
                bits = int(bits_raw)
            except (TypeError, ValueError):
                bits = -1
            if bits != 8:
                continue
            int8_observations += 1
            method = str(actual_methods.get(cid, "missing_actual_quant_method"))
            if method != "qat_int8_train":
                violations.append(
                    {
                        "round": round_id,
                        "client_id": cid,
                        "assigned_bits": 8,
                        "actual_quant_method": method,
                        "reason": "int8_assigned_but_not_qat_int8_train",
                    }
                )

    if not round_logs:
        violations.append(
            {
                "round": None,
                "client_id": None,
                "assigned_bits": None,
                "actual_quant_method": "missing",
                "reason": "no_round_logs_found",
            }
        )

    if int8_observations == 0:
        violations.append(
            {
                "round": None,
                "client_id": None,
                "assigned_bits": 8,
                "actual_quant_method": "missing",
                "reason": "no_int8_qat_observations_found",
            }
        )

    status = "passed" if not violations else "failed"
    report = {
        "status": status,
        "run_dir": str(run_dir),
        "rounds_checked": len(round_logs),
        "int8_observations": int8_observations,
        "violations": violations,
    }

    report_path = (
        Path(args.report_path)
        if args.report_path
        else run_dir / "qat_path_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if status == "passed":
        print(f"[OK] Exp07 QAT path check passed: {report_path}")
        raise SystemExit(0)

    print(f"[FAIL] Exp07 QAT path check failed: {report_path}")
    for item in violations:
        print(
            f"  round={item.get('round')} client={item.get('client_id')} "
            f"bits={item.get('assigned_bits')} actual={item.get('actual_quant_method')} "
            f"reason={item.get('reason')}"
        )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
