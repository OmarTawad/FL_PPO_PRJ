#!/bin/bash
set -e
cd /root/FL_PPO_PRJ
docker compose -f docker/docker-compose.exp01.yml up --build --abort-on-container-exit
docker compose -f docker/docker-compose.exp01.yml down
