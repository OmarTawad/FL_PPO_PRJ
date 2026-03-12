#!/usr/bin/env python3
"""Collect and classify exp07_qat_v2 client container exits from Docker."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROUND_RE = re.compile(r"round=(\d+)")
EXIT_RE = re.compile(r"(fl_client_\d{2}_[A-Za-z0-9_]+)\s+exited with code\s+(\d+)")
LOG_PREFIX_RE = re.compile(r"^(fl_client_\d{2}_[A-Za-z0-9_]+)\s+\|\s+(.*)$")


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def latest_run_id(metrics_dir: Path) -> str:
    run_dirs = sorted(p.name for p in metrics_dir.glob("run_*") if p.is_dir())
    if not run_dirs:
        return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    return run_dirs[-1]


def classify_failure(exit_code: int | None, oom_killed: bool, status: str, state_error: str, logs_tail: str) -> str:
    hay = f"{state_error}\n{logs_tail}".lower()
    if oom_killed or exit_code == 137:
        return "likely_oom_or_resource_kill"
    if exit_code in (None, 0):
        if status == "running":
            return "running"
        return "normal_exit"
    timeout_markers = ("timeout", "timed out", "deadline exceeded", "unavailable", "connection refused")
    if any(m in hay for m in timeout_markers):
        return "timeout_or_connectivity_failure"
    if "out of memory" in hay or "oom" in hay or "killed" in hay:
        return "likely_oom_or_resource_kill"
    return "runtime_failure_nonzero_exit"


def parse_latest_round(logs_tail: str) -> int | None:
    rounds = [int(m.group(1)) for m in ROUND_RE.finditer(logs_tail)]
    return max(rounds) if rounds else None


def load_compose(compose_file: Path) -> dict[str, Any]:
    with open(compose_file) as f:
        return yaml.safe_load(f)


def parse_compose_log(log_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    exit_codes: dict[str, int] = {}
    latest_round: dict[str, int] = {}
    if not log_path.is_file():
        return exit_codes, latest_round

    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            clean = line.replace("\x1b[K", "").rstrip("\n")
            m_exit = EXIT_RE.search(clean)
            if m_exit:
                exit_codes[m_exit.group(1)] = int(m_exit.group(2))
            m_prefix = LOG_PREFIX_RE.match(clean)
            if m_prefix:
                cname, payload = m_prefix.groups()
                rounds = [int(m.group(1)) for m in ROUND_RE.finditer(payload)]
                if rounds:
                    latest_round[cname] = max(latest_round.get(cname, 0), max(rounds))
    return exit_codes, latest_round


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect exp07_qat_v2 client failures from Docker state/logs")
    parser.add_argument("--compose-file", default="docker/docker-compose.exp07_qat_v2_validation.yml")
    parser.add_argument("--outputs-dir", default="outputs/exp07_qat_int8_v2")
    parser.add_argument("--run-id", default="", help="Optional run_id (e.g., run_20260308_123456)")
    parser.add_argument("--log-tail", type=int, default=300)
    parser.add_argument("--compose-log", default="", help="Optional docker compose up log path")
    args = parser.parse_args()

    compose_file = Path(args.compose_file)
    outputs_dir = Path(args.outputs_dir)
    metrics_dir = outputs_dir / "metrics"

    run_id = args.run_id or latest_run_id(metrics_dir)
    out_dir = outputs_dir / "logs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    compose = load_compose(compose_file)
    services = compose.get("services", {})
    log_exit_codes: dict[str, int] = {}
    log_latest_round: dict[str, int] = {}
    if args.compose_log:
        log_exit_codes, log_latest_round = parse_compose_log(Path(args.compose_log))

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "compose_file": str(compose_file),
        "run_id": run_id,
        "clients": [],
        "summary": {},
    }

    cause_counts: dict[str, int] = {}

    for service_name, service_cfg in services.items():
        if not service_name.startswith("client_"):
            continue

        container_name = service_cfg.get("container_name", "")
        inspect_proc = run_cmd(["docker", "inspect", container_name])

        if inspect_proc.returncode != 0:
            log_exit = log_exit_codes.get(container_name)
            log_round = log_latest_round.get(container_name)
            likely_cause = classify_failure(
                exit_code=log_exit,
                oom_killed=(log_exit == 137),
                status="from_log_only",
                state_error=inspect_proc.stderr.strip(),
                logs_tail="",
            ) if log_exit is not None else "not_found_or_not_created"
            item = {
                "service": service_name,
                "container_name": container_name,
                "state_status": "not_found",
                "exit_code": log_exit,
                "oom_killed": bool(log_exit == 137),
                "state_error": inspect_proc.stderr.strip(),
                "started_at": None,
                "finished_at": None,
                "latest_round_seen": log_round,
                "likely_cause": likely_cause,
            }
            report["clients"].append(item)
            cause_counts[item["likely_cause"]] = cause_counts.get(item["likely_cause"], 0) + 1
            continue

        inspect_data = json.loads(inspect_proc.stdout)[0]
        state = inspect_data.get("State", {})

        logs_proc = run_cmd(["docker", "logs", "--tail", str(args.log_tail), container_name])
        logs_tail = (logs_proc.stdout or "") + ("\n" + logs_proc.stderr if logs_proc.stderr else "")

        exit_code = state.get("ExitCode")
        oom_killed = bool(state.get("OOMKilled", False))
        status = str(state.get("Status", "unknown"))
        state_error = str(state.get("Error", ""))
        likely_cause = classify_failure(exit_code, oom_killed, status, state_error, logs_tail)

        item = {
            "service": service_name,
            "container_name": container_name,
            "state_status": status,
            "exit_code": exit_code,
            "oom_killed": oom_killed,
            "state_error": state_error,
            "started_at": state.get("StartedAt"),
            "finished_at": state.get("FinishedAt"),
            "latest_round_seen": parse_latest_round(logs_tail),
            "likely_cause": likely_cause,
        }
        report["clients"].append(item)
        cause_counts[likely_cause] = cause_counts.get(likely_cause, 0) + 1

    report["summary"] = {
        "n_clients": len(report["clients"]),
        "cause_counts": cause_counts,
    }

    out_path = out_dir / "client_exit_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
