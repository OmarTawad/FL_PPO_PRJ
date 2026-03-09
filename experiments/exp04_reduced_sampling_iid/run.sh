#!/bin/bash
set -uo pipefail

cd /root/FL_PPO_PRJ
mkdir -p outputs/exp04_reduced_sampling_iid/logs

ts=$(date -u +%Y%m%d_%H%M%S)
log="outputs/exp04_reduced_sampling_iid/logs/exp04_full_${ts}.log"
compose="docker/docker-compose.exp04.yml"

echo "[exp04] Full run log: ${log}"

set +e
docker compose -f "${compose}" up --build 2>&1 | tee "${log}"
up_status=${PIPESTATUS[0]}

python3 scripts/collect_exp04_failures.py \
  --compose-file "${compose}" \
  --outputs-dir outputs/exp04_reduced_sampling_iid \
  --compose-log "${log}"
collect_status=$?

docker compose -f "${compose}" down
down_status=$?

latest_run_dir="$(ls -1dt outputs/exp04_reduced_sampling_iid/metrics/run_* 2>/dev/null | head -1)"
plot_status=0
if [ -n "${latest_run_dir}" ] && [ -d "${latest_run_dir}" ]; then
  if [ -x ".venv_plot/bin/python" ]; then
    plot_python=".venv_plot/bin/python"
  else
    plot_python="python3"
  fi
  echo "[exp04] Plotting run: ${latest_run_dir}"
  "${plot_python}" scripts/plot_results.py "${latest_run_dir}"
  plot_status=$?
else
  echo "[exp04] Warning: no metrics run directory found for plotting." >&2
  plot_status=1
fi
set -e

if [ ${collect_status} -ne 0 ]; then
  echo "[exp04] Warning: failure collection returned ${collect_status}" >&2
fi
if [ ${down_status} -ne 0 ]; then
  echo "[exp04] Warning: docker compose down returned ${down_status}" >&2
fi
if [ ${plot_status} -ne 0 ]; then
  echo "[exp04] Warning: plot generation returned ${plot_status}" >&2
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
