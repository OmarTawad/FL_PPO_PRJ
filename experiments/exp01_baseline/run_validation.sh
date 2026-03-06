#!/bin/bash
set -e
cd /root/FL_PPO_PRJ
docker compose -f docker/docker-compose.exp01_validation.yml up --build --abort-on-container-exit
docker compose -f docker/docker-compose.exp01_validation.yml down
