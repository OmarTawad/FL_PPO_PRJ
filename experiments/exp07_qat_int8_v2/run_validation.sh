#!/bin/bash
set -uo pipefail

cd /root/FL_PPO_PRJ
mkdir -p outputs/exp07_qat_int8_v2/logs

ts=$(date -u +%Y%m%d_%H%M%S)
log="outputs/exp07_qat_int8_v2/logs/exp07_qat_v2_validation_${ts}.log"
compose="docker/docker-compose.exp07_qat_v2_validation.yml"

echo "[exp07_qat_v2] Validation log: ${log}"

set +e
docker compose -f "${compose}" up --build 2>&1 | tee "${log}"
up_status=${PIPESTATUS[0]}

python3 scripts/collect_exp07_qat_v2_failures.py \
  --compose-file "${compose}" \
  --outputs-dir outputs/exp07_qat_int8_v2 \
  --compose-log "${log}"
collect_status=$?

docker compose -f "${compose}" down
down_status=$?

latest_run_dir="$(ls -1dt outputs/exp07_qat_int8_v2/metrics/run_* 2>/dev/null | head -1)"
plot_status=0
if [ -n "${latest_run_dir}" ] && [ -d "${latest_run_dir}" ]; then
  if [ -x ".venv_plot/bin/python" ]; then
    plot_python=".venv_plot/bin/python"
  else
    plot_python="python3"
  fi
  echo "[exp07_qat_v2] Plotting run: ${latest_run_dir}"
  "${plot_python}" scripts/plot_results.py "${latest_run_dir}"
  plot_status=$?
else
  echo "[exp07_qat_v2] Warning: no metrics run directory found for plotting." >&2
  plot_status=1
fi

qat_status=0
if [ -n "${latest_run_dir}" ] && [ -d "${latest_run_dir}" ]; then
  run_id="$(basename "${latest_run_dir}")"
  qat_report="outputs/exp07_qat_int8_v2/logs/${run_id}/qat_v2_path_report.json"
  echo "[exp07_qat_v2] QAT v2 path check on: ${latest_run_dir}"
  python3 scripts/check_exp07_qat_v2_path.py \
    --run-dir "${latest_run_dir}" \
    --report-path "${qat_report}"
  qat_status=$?
else
  echo "[exp07_qat_v2] Warning: no metrics run directory found for QAT check." >&2
  python3 scripts/check_exp07_qat_v2_path.py \
    --metrics-dir outputs/exp07_qat_int8_v2/metrics
  qat_status=$?
fi
set -e

if [ ${collect_status} -ne 0 ]; then
  echo "[exp07_qat_v2] Warning: failure collection returned ${collect_status}" >&2
fi
if [ ${down_status} -ne 0 ]; then
  echo "[exp07_qat_v2] Warning: docker compose down returned ${down_status}" >&2
fi
if [ ${plot_status} -ne 0 ]; then
  echo "[exp07_qat_v2] Warning: plot generation returned ${plot_status}" >&2
fi
if [ ${qat_status} -ne 0 ]; then
  echo "[exp07_qat_v2] QAT path check failed with code ${qat_status}" >&2
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
if [ ${final_status} -eq 0 ] && [ ${qat_status} -ne 0 ]; then
  final_status=${qat_status}
fi

exit ${final_status}
