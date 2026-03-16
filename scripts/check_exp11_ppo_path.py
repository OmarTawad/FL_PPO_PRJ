#!/usr/bin/env python3
"""Post-run checker for Exp11 live PPO + BF16-train + INT8-transport path."""

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


def _expected_policy(action: int) -> tuple[int, str]:
    if action <= 0:
        return 0, "skip"
    if action == 1:
        return 32, "fp32"
    if action == 2:
        return 16, "fp32"
    if action == 3:
        return 16, "int8"
    return 32, "fp32"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Exp11 PPO runtime path evidence"
    )
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--metrics-dir", default="outputs/exp11_ppo_bf16_int8/metrics")
    parser.add_argument("--report-path", default="")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir, args.metrics_dir)
    round_logs = _load_round_logs(run_dir)

    violations: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    observed_bf16_clients = 0
    observed_int8_transport_clients = 0
    rounds_with_ppo_actions = 0
    rounds_with_reward = 0
    ppo_updates_applied = 0

    for entry in round_logs:
        round_id = int(entry.get("round", -1))
        ppo_actions = entry.get("ppo_actions", {}) or {}
        reward = entry.get("reward", None)
        ppo_update = entry.get("ppo_update", {}) or {}

        quant_assignments = entry.get("quant_assignments", {}) or {}
        train_precision_actual = entry.get("training_precision_actual", {}) or {}
        train_method_actual = entry.get("training_method_actual", {}) or {}
        transport_dtype_actual = entry.get("transport_dtype_actual", {}) or {}
        decode_success = entry.get("decode_success", {}) or {}
        agg_input_dtype = entry.get("aggregation_input_dtype_after_decode", {}) or {}
        policy_action_requested = entry.get("policy_action_requested", {}) or {}

        if isinstance(ppo_actions, dict) and ppo_actions:
            rounds_with_ppo_actions += 1
        else:
            violations.append(
                {
                    "round": round_id,
                    "reason": "missing_ppo_actions",
                }
            )

        if reward is not None:
            rounds_with_reward += 1
        else:
            violations.append(
                {
                    "round": round_id,
                    "reason": "missing_round_reward",
                }
            )

        if int(ppo_update.get("update_applied", 0)) == 1:
            ppo_updates_applied += 1

        for cid, bits_raw in quant_assignments.items():
            bits = int(bits_raw)
            train_prec = str(train_precision_actual.get(cid, "missing"))
            train_method = str(train_method_actual.get(cid, "missing"))
            transport_actual = str(transport_dtype_actual.get(cid, "missing")).lower()
            decode_ok = int(decode_success.get(cid, 0))
            agg_dtype = str(agg_input_dtype.get(cid, "missing")).lower()
            policy_action = int(policy_action_requested.get(cid, -1))

            if bits == 16 and train_prec == "bf16":
                observed_bf16_clients += 1
            if bits == 16 and train_prec != "bf16":
                warnings.append(
                    {
                        "round": round_id,
                        "client_id": cid,
                        "reason": "bf16_requested_but_actual_not_bf16",
                        "training_precision_actual": train_prec,
                        "training_method_actual": train_method,
                    }
                )

            if bits == 8:
                violations.append(
                    {
                        "round": round_id,
                        "client_id": cid,
                        "reason": "unexpected_8bit_training_assignment",
                        "assigned_bits": bits,
                    }
                )

            if train_prec == "fp16" or train_method in ("fp16", "fp16_fallback", "qat_int8_train"):
                violations.append(
                    {
                        "round": round_id,
                        "client_id": cid,
                        "reason": "forbidden_precision_or_qat_path_observed",
                        "training_precision_actual": train_prec,
                        "training_method_actual": train_method,
                    }
                )

            if transport_actual == "int8":
                observed_int8_transport_clients += 1
                if decode_ok != 1:
                    violations.append(
                        {
                            "round": round_id,
                            "client_id": cid,
                            "reason": "decode_failed_for_int8_transport",
                            "decode_success": decode_ok,
                        }
                    )
                if agg_dtype != "fp32":
                    violations.append(
                        {
                            "round": round_id,
                            "client_id": cid,
                            "reason": "aggregation_input_not_fp32_after_decode",
                            "aggregation_input_dtype_after_decode": agg_dtype,
                        }
                    )

            if policy_action >= 0:
                exp_bits, exp_transport = _expected_policy(policy_action)
                if bits != exp_bits:
                    violations.append(
                        {
                            "round": round_id,
                            "client_id": cid,
                            "reason": "policy_action_bits_mismatch",
                            "policy_action": policy_action,
                            "expected_bits": exp_bits,
                            "observed_bits": bits,
                        }
                    )
                if transport_actual != exp_transport:
                    violations.append(
                        {
                            "round": round_id,
                            "client_id": cid,
                            "reason": "policy_action_transport_mismatch",
                            "policy_action": policy_action,
                            "expected_transport": exp_transport,
                            "observed_transport": transport_actual,
                        }
                    )

    if not round_logs:
        violations.append({"round": None, "reason": "no_round_logs_found"})

    if observed_bf16_clients == 0:
        violations.append(
            {
                "round": None,
                "reason": "no_bf16_training_observed",
            }
        )

    if observed_int8_transport_clients == 0:
        violations.append(
            {
                "round": None,
                "reason": "no_int8_transport_observed",
            }
        )

    if ppo_updates_applied == 0:
        violations.append(
            {
                "round": None,
                "reason": "no_ppo_update_applied",
            }
        )

    status = "passed" if not violations else "failed"
    report = {
        "status": status,
        "policy": "exp11_live_ppo_bf16_int8_transport",
        "run_dir": str(run_dir),
        "rounds_checked": len(round_logs),
        "rounds_with_ppo_actions": rounds_with_ppo_actions,
        "rounds_with_reward": rounds_with_reward,
        "ppo_updates_applied": ppo_updates_applied,
        "observed_bf16_clients": observed_bf16_clients,
        "observed_int8_transport_clients": observed_int8_transport_clients,
        "violations": violations,
        "warnings": warnings,
    }

    report_path = (
        Path(args.report_path)
        if args.report_path
        else run_dir / "exp11_ppo_path_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if status == "passed":
        print(f"[OK] Exp11 PPO path check passed: {report_path}")
        if warnings:
            print(f"[WARN] Exp11 checker emitted {len(warnings)} warning(s)")
        raise SystemExit(0)

    print(f"[FAIL] Exp11 PPO path check failed: {report_path}")
    for item in violations:
        print(
            f"  round={item.get('round')} client={item.get('client_id')} "
            f"reason={item.get('reason')}"
        )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
