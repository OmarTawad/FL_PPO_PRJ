#!/bin/bash
set -uo pipefail

cd /root/FL_PPO_PRJ
mkdir -p outputs/exp11_ppo_bf16_int8/logs

ts=$(date -u +%Y%m%d_%H%M%S)
log="outputs/exp11_ppo_bf16_int8/logs/exp11_full_${ts}.log"
compose="docker/docker-compose.exp11.yml"
project="fl_ppo_exp11"
services=(
  server
  client_00 client_01 client_02 client_03 client_04
  client_05 client_06 client_07 client_08 client_09
)

echo "[exp11] Full run log: ${log}"

set +e
docker compose -p "${project}" -f "${compose}" down --remove-orphans >/dev/null 2>&1
docker compose -p "${project}" -f "${compose}" up --build --remove-orphans "${services[@]}" 2>&1 | tee "${log}"
up_status=${PIPESTATUS[0]}

python3 scripts/collect_exp11_failures.py \
  --compose-file "${compose}" \
  --outputs-dir outputs/exp11_ppo_bf16_int8 \
  --compose-log "${log}"
collect_status=$?

docker compose -p "${project}" -f "${compose}" down --remove-orphans
down_status=$?

latest_run_dir="$(ls -1dt outputs/exp11_ppo_bf16_int8/metrics/run_* 2>/dev/null | head -1)"
plot_status=0
if [ -n "${latest_run_dir}" ] && [ -d "${latest_run_dir}" ]; then
  if [ -x ".venv_plot/bin/python" ]; then
    plot_python=".venv_plot/bin/python"
  else
    plot_python="python3"
  fi
  echo "[exp11] Plotting run: ${latest_run_dir}"
  "${plot_python}" scripts/plot_results.py "${latest_run_dir}"
  plot_status=$?
else
  echo "[exp11] Warning: no metrics run directory found for plotting." >&2
  plot_status=1
fi

path_status=0
if [ -n "${latest_run_dir}" ] && [ -d "${latest_run_dir}" ]; then
  run_id="$(basename "${latest_run_dir}")"
  path_report="outputs/exp11_ppo_bf16_int8/logs/${run_id}/exp11_ppo_path_report.json"
  echo "[exp11] Path check on: ${latest_run_dir}"
  python3 scripts/check_exp11_ppo_path.py \
    --run-dir "${latest_run_dir}" \
    --report-path "${path_report}"
  path_status=$?
else
  echo "[exp11] Warning: no metrics run directory found for path check." >&2
  python3 scripts/check_exp11_ppo_path.py \
    --metrics-dir outputs/exp11_ppo_bf16_int8/metrics
  path_status=$?
fi
set -e

if [ ${collect_status} -ne 0 ]; then
  echo "[exp11] Warning: failure collection returned ${collect_status}" >&2
fi
if [ ${down_status} -ne 0 ]; then
  echo "[exp11] Warning: docker compose down returned ${down_status}" >&2
fi
if [ ${plot_status} -ne 0 ]; then
  echo "[exp11] Warning: plot generation returned ${plot_status}" >&2
fi
if [ ${path_status} -ne 0 ]; then
  echo "[exp11] Path check failed with code ${path_status}" >&2
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
if [ ${final_status} -eq 0 ] && [ ${path_status} -ne 0 ]; then
  final_status=${path_status}
fi

exit ${final_status}
