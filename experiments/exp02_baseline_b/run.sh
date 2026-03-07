#!/bin/bash
set -e
cd /root/FL_PPO_PRJ
docker compose -f docker/docker-compose.exp02.yml up --build
docker compose -f docker/docker-compose.exp02.yml down
