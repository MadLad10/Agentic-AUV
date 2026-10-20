#!/usr/bin/env bash
# Third queue: after the other queues, rerun the perception A/B with obstacle-aware routing
# (the first run recorded gate collisions in murky water) - results go to a fresh folder.
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
while pgrep -f "scripts/run_queue2?[.]sh" >/dev/null; do sleep 30; done
[ -e eval/results/perception_study_routed/summary.md ] || python scripts/perception_study.py --seeds 5 \
  --out eval/results/perception_study_routed > eval/results/perception_study_routed.log 2>&1
echo "[queue3] finished $(date +%H:%M:%S)"
