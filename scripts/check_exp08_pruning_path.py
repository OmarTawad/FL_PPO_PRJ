#!/usr/bin/env python3
"""Post-run checker for Exp08 pruning path activation and artifacts."""

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


def _parse_clients(raw: str) -> List[str]:
    out: List[str] = []
    for chunk in str(raw).split(","):
        c = chunk.strip()
        if not c:
            continue
        out.append(str(int(c)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Exp08 pruning run artifacts")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--metrics-dir", default="outputs/exp08_pruning_weak10/metrics")
    parser.add_argument("--report-path", default="")
    parser.add_argument("--weak-clients", default="6,7,8,9")
    parser.add_argument("--expected-prune-amount", type=float, default=0.10)
    parser.add_argument("--amount-tol", type=float, default=1e-6)
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir, args.metrics_dir)
    round_logs = _load_round_logs(run_dir)
    weak_clients = set(_parse_clients(args.weak_clients))

    violations: List[Dict[str, Any]] = []
    weak_observations = 0
    nonweak_observations = 0
    weak_failure_report_evidence = 0

    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        violations.append(
            {
                "round": None,
                "client_id": None,
                "reason": "summary_missing",
                "path": str(summary_path),
            }
        )

    for entry in round_logs:
        round_id = int(entry.get("round", -1))
        selected_clients = [str(cid) for cid in (entry.get("selected_clients", []) or [])]

        pruning_requested = entry.get("pruning_requested", {}) or {}
        pruning_applied = entry.get("pruning_applied", {}) or {}
        pruning_amount_requested = entry.get("pruning_amount_requested", {}) or {}
        pruning_amount_applied = entry.get("pruning_amount_applied", {}) or {}
        pruning_method = entry.get("pruning_method", {}) or {}
        pruning_active = entry.get("pruning_active_during_training", {}) or {}
        pruning_skip_reason = entry.get("pruning_skip_reason", {}) or {}

        required_fields = {
            "pruning_requested": pruning_requested,
            "pruning_applied": pruning_applied,
            "pruning_amount_requested": pruning_amount_requested,
            "pruning_amount_applied": pruning_amount_applied,
            "pruning_method": pruning_method,
            "pruning_active_during_training": pruning_active,
            "pruning_skip_reason": pruning_skip_reason,
        }
        for field_name, field_value in required_fields.items():
            if not isinstance(field_value, dict):
                violations.append(
                    {
                        "round": round_id,
                        "client_id": None,
                        "reason": "round_field_not_dict",
                        "field": field_name,
                    }
                )

        for cid in selected_clients:
            if cid in weak_clients:
                if cid not in pruning_requested:
                    continue
                weak_observations += 1
                req = int(pruning_requested.get(cid, 0))
                app = int(pruning_applied.get(cid, 0))
                amt_req = float(pruning_amount_requested.get(cid, 0.0))
                method = str(pruning_method.get(cid, "missing")).strip().lower()
                if req != 1:
                    violations.append(
                        {
                            "round": round_id,
                            "client_id": cid,
                            "reason": "weak_client_pruning_not_requested",
                            "pruning_requested": req,
                        }
                    )
                if abs(amt_req - args.expected_prune_amount) > args.amount_tol:
                    violations.append(
                        {
                            "round": round_id,
                            "client_id": cid,
                            "reason": "weak_client_pruning_amount_mismatch",
                            "amount_requested": amt_req,
                            "expected": args.expected_prune_amount,
                        }
                    )
                if method != "magnitude_unstructured":
                    violations.append(
                        {
                            "round": round_id,
                            "client_id": cid,
                            "reason": "weak_client_pruning_method_mismatch",
                            "pruning_method": method,
                        }
                    )
                if app not in (0, 1):
                    violations.append(
                        {
                            "round": round_id,
                            "client_id": cid,
                            "reason": "weak_client_pruning_applied_invalid",
                            "pruning_applied": app,
                        }
                    )
            else:
                if cid not in pruning_requested:
                    continue
                nonweak_observations += 1
                req = int(pruning_requested.get(cid, 0))
                amt_req = float(pruning_amount_requested.get(cid, 0.0))
                if req != 0:
                    violations.append(
                        {
                            "round": round_id,
                            "client_id": cid,
                            "reason": "nonweak_client_unexpected_pruning_requested",
                            "pruning_requested": req,
                        }
                    )
                if abs(amt_req) > args.amount_tol:
                    violations.append(
                        {
                            "round": round_id,
                            "client_id": cid,
                            "reason": "nonweak_client_unexpected_pruning_amount",
                            "amount_requested": amt_req,
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

    if weak_observations == 0:
        failure_report_path = (
            run_dir.parent.parent / "logs" / run_dir.name / "client_exit_report.json"
        )
        if failure_report_path.is_file():
            try:
                with open(failure_report_path, "r") as f:
                    failure_report = json.load(f)
                for item in failure_report.get("clients", []):
                    service = str(item.get("service", ""))
                    cid_raw = service.split("_", 1)[1] if service.startswith("client_") else ""
                    cid = ""
                    if cid_raw:
                        try:
                            cid = str(int(cid_raw))
                        except ValueError:
                            cid = ""
                    if cid in weak_clients and (
                        item.get("pruning_active_last_seen") is not None
                        or item.get("pruning_amount_last_seen") is not None
                    ):
                        weak_failure_report_evidence += 1
            except Exception:
                weak_failure_report_evidence = 0

    if weak_observations == 0 and weak_failure_report_evidence == 0:
        violations.append(
            {
                "round": None,
                "client_id": None,
                "reason": "no_weak_pruning_observations_found",
            }
        )

    status = "passed" if not violations else "failed"
    report = {
        "status": status,
        "policy": "artifacts-first-strict",
        "run_dir": str(run_dir),
        "rounds_checked": len(round_logs),
        "weak_clients": sorted(weak_clients),
        "weak_observations": weak_observations,
        "weak_failure_report_evidence": weak_failure_report_evidence,
        "nonweak_observations": nonweak_observations,
        "expected_prune_amount": args.expected_prune_amount,
        "violations": violations,
    }

    report_path = (
        Path(args.report_path)
        if args.report_path
        else run_dir / "pruning_path_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if status == "passed":
        print(f"[OK] Exp08 pruning path check passed: {report_path}")
        raise SystemExit(0)

    print(f"[FAIL] Exp08 pruning path check failed: {report_path}")
    for item in violations:
        print(
            f"  round={item.get('round')} client={item.get('client_id')} "
            f"reason={item.get('reason')}"
        )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
