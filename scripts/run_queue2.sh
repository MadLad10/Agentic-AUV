#!/usr/bin/env bash
# Follow-up queue: waits for run_queue.sh, then the water-current robustness check of the learned
# DVL-outage velocity model (trained in still water).
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
while pgrep -f "scripts/run_queue[.]sh" >/dev/null; do sleep 30; done
[ -e eval/results/nav_study_current/summary.md ] || python scripts/nav_study.py --seeds 20 \
  --estimators ekf,ekf+learned_vel --conditions dvl_dropout_current --out eval/results/nav_study_current
echo "[queue2] finished $(date +%H:%M:%S)"
