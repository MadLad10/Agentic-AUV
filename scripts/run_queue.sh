#!/usr/bin/env bash
# Sequential, resumable experiment queue (one job at a time: no CPU contention between measurements).
# Each step is skipped when its output already exists, so the queue can simply be restarted after a
# reboot:   setsid nohup scripts/run_queue.sh > eval/results/queue.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
source scripts/env.sh
log() { echo "[queue $(date +%H:%M:%S)] $*"; }

step() {   # step <name> <output-that-marks-done> <command...>
  local name=$1 done_file=$2; shift 2
  if [ -e "$done_file" ]; then log "skip $name (have $done_file)"; return 0; fi
  log "start $name"
  "$@"; local rc=$?
  log "end $name rc=$rc"
}

step perception_study eval/results/perception_study/DONE \
  bash -c 'python scripts/perception_study.py --seeds 5 > eval/results/perception_study.log 2>&1 && touch eval/results/perception_study/DONE'

step nav_study_20seeds eval/results/nav_study_20seeds/summary.md \
  python scripts/nav_study.py --seeds 20 --estimators ekf,ekf+learned_vel,fgo,fgo+learned_vel \
    --conditions nominal,dvl_dropout,dvl_dropout_no_landmarks,compass_disturbance --out eval/results/nav_study_20seeds

scripts/start_ollama.sh >/dev/null 2>&1 || log "ollama start failed"
step vlm_benchmark eval/results/vlm_benchmark/summary.md \
  bash -c 'python scripts/vlm_benchmark.py > eval/results/vlm_benchmark/run.log 2>&1'

step v2_final_eval eval/results/v2_final_qwen3b/summary.md \
  python eval/run_eval_v2.py --systems v2 --sets nominal,faults --provider ollama --out eval/results/v2_final_qwen3b --resume
scripts/start_ollama.sh stop >/dev/null 2>&1

step detector_study eval/results/detector_study/summary.md \
  bash -c 'python scripts/detector_study.py >> eval/results/detector_study/run.log 2>&1'

step sonar_detector data/models/yolo_sonar_eval.json \
  python scripts/train_yolo.py --data data/synthetic_sonar/data.yaml --name sonar_v1 --imgsz 448 --epochs 40 \
    --threads 12 --weights-name auv_yolov8n_sonar.pt --eval-name yolo_sonar_eval.json
log "queue finished"
