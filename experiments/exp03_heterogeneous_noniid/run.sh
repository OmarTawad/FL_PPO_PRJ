#!/bin/bash
set -uo pipefail

cd /root/FL_PPO_PRJ
mkdir -p outputs/exp03_heterogeneous_noniid/logs

ts=$(date -u +%Y%m%d_%H%M%S)
log="outputs/exp03_heterogeneous_noniid/logs/exp03_full_${ts}.log"
compose="docker/docker-compose.exp03.yml"

echo "[exp03] Full run log: ${log}"

set +e
docker compose -f "${compose}" up --build 2>&1 | tee "${log}"
up_status=${PIPESTATUS[0]}

python3 scripts/collect_exp03_failures.py \
  --compose-file "${compose}" \
  --outputs-dir outputs/exp03_heterogeneous_noniid \
  --compose-log "${log}"
collect_status=$?

docker compose -f "${compose}" down
down_status=$?
set -e

if [ ${collect_status} -ne 0 ]; then
  echo "[exp03] Warning: failure collection returned ${collect_status}" >&2
fi
if [ ${down_status} -ne 0 ]; then
  echo "[exp03] Warning: docker compose down returned ${down_status}" >&2
fi

exit ${up_status}
