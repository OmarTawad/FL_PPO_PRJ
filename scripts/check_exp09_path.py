#!/usr/bin/env python3
"""Post-run checker for Exp09 uniformized Exp07-v2 path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


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


def _parse_client_ids(raw: str) -> List[str]:
    out: List[str] = []
    for chunk in str(raw).split(","):
        c = chunk.strip()
        if not c:
            continue
        out.append(str(int(c)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Exp09 uniform policy + INT8 transport/decode evidence"
    )
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--metrics-dir", default="outputs/exp09_uniform_int8_all/metrics")
    parser.add_argument("--report-path", default="")
    parser.add_argument("--clients", default="0,1,2,3,4,5,6,7,8,9")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir, args.metrics_dir)
    round_logs = _load_round_logs(run_dir)
    required_clients = _parse_client_ids(args.clients)

    violations: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    observations = 0

    for entry in round_logs:
        round_id = int(entry.get("round", -1))
        quant_assignments = entry.get("quant_assignments", {}) or {}
        train_precision_req = entry.get("training_precision_requested", {}) or {}
        train_precision_actual = entry.get("training_precision_actual", {}) or {}
        actual_methods = entry.get("actual_quant_method", {}) or {}
        transport_dtype_actual = entry.get("transport_dtype_actual", {}) or {}
        decode_success = entry.get("decode_success", {}) or {}
        agg_input_dtype = entry.get("aggregation_input_dtype_after_decode", {}) or {}

        for cid in required_clients:
            if cid not in quant_assignments:
                continue
            observations += 1
            bits = int(quant_assignments.get(cid, -1))
            train_req = str(train_precision_req.get(cid, "missing"))
            train_act = str(train_precision_actual.get(cid, "missing"))
            method = str(actual_methods.get(cid, "missing"))
            transport_actual = str(transport_dtype_actual.get(cid, "missing"))
            decode_ok = int(decode_success.get(cid, 0))
            agg_dtype = str(agg_input_dtype.get(cid, "missing"))

            if bits != 16:
                violations.append(
                    {
                        "round": round_id,
                        "client_id": cid,
                        "reason": "uniform_policy_quant_assignment_not_16",
                        "assigned_bits": bits,
                    }
                )

            if transport_actual != "int8":
                violations.append(
                    {
                        "round": round_id,
                        "client_id": cid,
                        "reason": "transport_not_int8",
                        "transport_dtype_actual": transport_actual,
                    }
                )

            if decode_ok != 1:
                violations.append(
                    {
                        "round": round_id,
                        "client_id": cid,
                        "reason": "server_decode_not_success",
                        "decode_success": decode_ok,
                    }
                )

            if agg_dtype != "fp32":
                violations.append(
                    {
                        "round": round_id,
                        "client_id": cid,
                        "reason": "aggregation_input_dtype_not_fp32_after_decode",
                        "aggregation_input_dtype_after_decode": agg_dtype,
                    }
                )

            # Honesty reporting: do not fail checker purely on fallback,
            # but emit explicit warning records.
            if train_req != "bf16":
                warnings.append(
                    {
                        "round": round_id,
                        "client_id": cid,
                        "reason": "training_precision_requested_not_bf16",
                        "training_precision_requested": train_req,
                    }
                )
            if train_act != "bf16":
                warnings.append(
                    {
                        "round": round_id,
                        "client_id": cid,
                        "reason": "training_precision_actual_not_bf16",
                        "training_precision_actual": train_act,
                        "actual_quant_method": method,
                    }
                )

    if not round_logs:
        violations.append(
            {
                "round": None,
                "client_id": None,
                "reason": "no_round_logs_found",
            }
        )

    if observations == 0:
        violations.append(
            {
                "round": None,
                "client_id": None,
                "reason": "no_client_observations_found",
            }
        )

    status = "passed" if not violations else "failed"
    report = {
        "status": status,
        "policy": "uniformized_exp07v2_all_clients",
        "run_dir": str(run_dir),
        "rounds_checked": len(round_logs),
        "clients_checked": required_clients,
        "observations": observations,
        "violations": violations,
        "warnings": warnings,
    }

    report_path = (
        Path(args.report_path)
        if args.report_path
        else run_dir / "exp09_path_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if status == "passed":
        print(f"[OK] Exp09 path check passed: {report_path}")
        if warnings:
            print(f"[WARN] Exp09 checker emitted {len(warnings)} warning(s)")
        raise SystemExit(0)

    print(f"[FAIL] Exp09 path check failed: {report_path}")
    for item in violations:
        print(
            f"  round={item.get('round')} client={item.get('client_id')} "
            f"reason={item.get('reason')}"
        )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
