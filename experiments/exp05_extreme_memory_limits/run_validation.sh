#!/bin/bash
set -uo pipefail

cd /root/FL_PPO_PRJ
mkdir -p outputs/exp05_extreme_memory_limits/logs

ts=$(date -u +%Y%m%d_%H%M%S)
log="outputs/exp05_extreme_memory_limits/logs/exp05_validation_${ts}.log"
compose="docker/docker-compose.exp05_validation.yml"
observe_seconds="${OBSERVE_SECONDS:-1800}"

echo "[exp05] Validation stress log: ${log}"
echo "[exp05] Observation window: ${observe_seconds}s"

set +e
timeout --signal=INT --preserve-status "${observe_seconds}" \
  docker compose -f "${compose}" up --build 2>&1 | tee "${log}"
up_status=${PIPESTATUS[0]}

if [ ${up_status} -eq 124 ]; then
  echo "[exp05] Observation window reached; collecting failure evidence and stopping compose stack."
  up_status=0
fi

python3 scripts/collect_exp05_failures.py \
  --compose-file "${compose}" \
  --outputs-dir outputs/exp05_extreme_memory_limits \
  --compose-log "${log}"
collect_status=$?

docker compose -f "${compose}" down
down_status=$?

latest_run_dir="$(ls -1dt outputs/exp05_extreme_memory_limits/metrics/run_* 2>/dev/null | head -1)"
plot_status=0
if [ -n "${latest_run_dir}" ] && [ -d "${latest_run_dir}" ] && ls "${latest_run_dir}"/round_*.json >/dev/null 2>&1; then
  if [ -x ".venv_plot/bin/python" ]; then
    plot_python=".venv_plot/bin/python"
  else
    plot_python="python3"
  fi
  echo "[exp05] Plotting run: ${latest_run_dir}"
  "${plot_python}" scripts/plot_results.py "${latest_run_dir}"
  plot_status=$?
else
  echo "[exp05] No round metrics found; skipping plots (expected for severe failure-boundary runs)."
  plot_status=0
fi
set -e

if [ ${collect_status} -ne 0 ]; then
  echo "[exp05] Warning: failure collection returned ${collect_status}" >&2
fi
if [ ${down_status} -ne 0 ]; then
  echo "[exp05] Warning: docker compose down returned ${down_status}" >&2
fi
if [ ${plot_status} -ne 0 ]; then
  echo "[exp05] Warning: plot generation returned ${plot_status}" >&2
fi

final_status=${up_status}
if [ ${final_status} -eq 0 ] && [ ${collect_status} -ne 0 ]; then
  final_status=${collect_status}
fi
if [ ${final_status} -eq 0 ] && [ ${down_status} -ne 0 ]; then
  final_status=${down_status}
fi
if [ ${final_status} -eq 0 ] && [ ${plot_status} -ne 0 ]; then
  final_status=${plot_status}
fi

exit ${final_status}
