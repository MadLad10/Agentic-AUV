#!/usr/bin/env bash
# Launch the headless sim, run ONE mission (or a custom command), save everything, shut down cleanly.
#
#   scripts/run_sim_test.sh [options] "pass through the gate then surface"
#   scripts/run_sim_test.sh [options] --cmd "python3 some_probe.py"
#
# Options:
#   -a ARENA       arena name or yaml (default: default)
#   -t SECONDS     mission timeout, wall clock (default: 300)
#   -p PROVIDER    LLM provider for the mission CLI (default: from config)
#   -b PERCENT     simulated battery start (default: 100)
#   -r RTF         real-time factor (default: 1.0)
#   -d SOURCE      detection source: ground_truth | camera (default: config/sim.yaml)
#   -c             render the camera and save frames (implied by -d camera)
#   -S             also render a third-person spectator camera and save its frames (media)
#   -n             no rosbag
#   -o DIR         output dir (default: runs/<timestamp>_<slug>)
#   -s STACK       sim stack: full | minimal (default: full)
#   -B CLASSES     fault injection: comma-separated classes the detector never reports
#
# Output dir: launch.log, mission.log, result.json, referee.json, trace.json (if produced), bag/, frames/
# Exit code: the mission command's exit code (124 = timeout), or 90+ for harness failures.
set -u
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARENA=default; TIMEOUT=300; PROVIDER=""; BATTERY=100; RTF=1.0; DET=""
CAMERA=false; BAG=true; OUT=""; STACK=full; CMD=""; BLIND=""; SPECTATOR=false
while [ $# -gt 0 ]; do
  case "$1" in
    -a) ARENA="$2"; shift 2;;
    -t) TIMEOUT="$2"; shift 2;;
    -p) PROVIDER="$2"; shift 2;;
    -b) BATTERY="$2"; shift 2;;
    -r) RTF="$2"; shift 2;;
    -d) DET="$2"; shift 2;;
    -c) CAMERA=true; shift;;
    -S) SPECTATOR=true; shift;;
    -n) BAG=false; shift;;
    -o) OUT="$2"; shift 2;;
    -s) STACK="$2"; shift 2;;
    -B) BLIND="$2"; shift 2;;
    --cmd) CMD="$2"; shift 2;;
    -h|--help) sed -n 2,21p "$0"; exit 0;;
    *) break;;
  esac
done
PROMPT="${*:-}"
[ "$DET" = camera ] && CAMERA=true
if [ -z "$PROMPT" ] && [ -z "$CMD" ]; then echo "need a mission prompt or --cmd" >&2; exit 2; fi

slug=$(echo "${PROMPT:-$CMD}" | tr -cs 'a-zA-Z0-9' '_' | cut -c1-40)
OUT="${OUT:-$WS/runs/$(date +%Y%m%d_%H%M%S)_${slug}}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

set +u; source "$WS/scripts/env.sh"; set -u
"$WS/scripts/cleanup_sim.sh" > /dev/null || { echo "[run] cleanup failed" >&2; exit 91; }

log() { echo "[run $(date +%H:%M:%S)] $*" | tee -a "$OUT/harness.log"; }
SIM_PID=""; BAG_PID=""; FRAMES_PID=""
finish() {
  local code=$1
  [ -n "$FRAMES_PID" ] && kill -INT "$FRAMES_PID" 2>/dev/null
  [ -n "${SPEC_PID:-}" ] && kill -INT "$SPEC_PID" 2>/dev/null
  if [ -n "$BAG_PID" ]; then kill -INT "$BAG_PID" 2>/dev/null; wait "$BAG_PID" 2>/dev/null; fi
  if [ -n "$SIM_PID" ]; then
    kill -INT "$SIM_PID" 2>/dev/null
    for _ in $(seq 1 10); do kill -0 "$SIM_PID" 2>/dev/null || break; sleep 1; done
  fi
  "$WS/scripts/cleanup_sim.sh" >> "$OUT/harness.log" 2>&1
  python3 - "$OUT" "$code" "$START" "$ARENA" "$PROMPT" "$CMD" <<'PY'
import json, sys, time, os
out, code, start, arena, prompt, cmd = sys.argv[1:7]
res = {'exit_code': int(code), 'timed_out': int(code) == 124, 'wall_s': round(time.time() - float(start), 1),
       'arena': arena, 'prompt': prompt, 'cmd': cmd,
       'files': sorted(os.listdir(out))}
json.dump(res, open(os.path.join(out, 'result.json'), 'w'), indent=2)
PY
  log "done: exit=$code  results in $OUT"
  exit "$code"
}
START=$(date +%s.%N)
trap 'finish 130' INT TERM

log "launching sim arena=$ARENA camera=$CAMERA rtf=$RTF det=$DET battery=$BATTERY stack=$STACK"
LARGS=(arena:="$ARENA" camera:="$CAMERA" rtf:="$RTF" battery_start:="$BATTERY" stack:="$STACK"
       report_file:="$OUT/referee.json")
[ -n "$DET" ] && LARGS+=(detection_source:="$DET")
[ -n "$BLIND" ] && LARGS+=(blind_classes:="$BLIND")
$SPECTATOR && LARGS+=(spectator:=true)
setsid timeout $((TIMEOUT + 90)) ros2 launch auv_sim sim.launch.py "${LARGS[@]}" > "$OUT/launch.log" 2>&1 &
SIM_PID=$!

# wait for odometry (+ vehicle actions for the full stack); no ros2 CLI daemon involved
WAIT_ARGS=(--timeout 45); [ "$STACK" = full ] && WAIT_ARGS+=(--actions)
python3 "$WS/scripts/wait_ready.py" "${WAIT_ARGS[@]}" || { log "sim did not come up (see launch.log)"; finish 92; }
log "sim ready"

if $BAG; then
  timeout $((TIMEOUT + 60)) ros2 bag record -o "$OUT/bag" /auv/odom /auv/battery /auv/detections \
    /auv/controller/setpoint /auv/controller/mode /auv/thrusters/total_abs /auv_sim/events \
    /auv/mission_status > "$OUT/bag.log" 2>&1 &
  BAG_PID=$!
fi
SPEC_PID=""
if $SPECTATOR; then
  timeout $((TIMEOUT + 30)) ros2 run auv_sim save_camera_frame --ros-args -r __node:=auv_spectator_saver \
    -p topic:=/auv/spectator/image -p out_dir:="$OUT/frames_spectator" -p count:=200 -p every_n:=2 \
    -p timeout_s:="$TIMEOUT" > "$OUT/frames_spectator.log" 2>&1 &
  SPEC_PID=$!
fi
if $CAMERA; then
  timeout $((TIMEOUT + 30)) ros2 run auv_sim save_camera_frame --ros-args -p out_dir:="$OUT/frames" \
    -p count:=60 -p every_n:=10 -p timeout_s:="$TIMEOUT" > "$OUT/frames.log" 2>&1 &
  FRAMES_PID=$!
fi

if [ -n "$CMD" ]; then
  log "running: $CMD"
  timeout "$TIMEOUT" bash -c "$CMD" > "$OUT/mission.log" 2>&1
  code=$?
else
  log "mission: $PROMPT"
  extra=(); [ -n "$PROVIDER" ] && extra=(--provider "$PROVIDER")
  timeout "$TIMEOUT" ros2 run auv_mission_planner mission "${extra[@]}" --trace-dir "$OUT" \
    --timeout "$TIMEOUT" "$PROMPT" > "$OUT/mission.log" 2>&1
  code=$?
fi
log "mission command exit=$code"
sleep 3     # let the referee observe the final state before teardown

finish "$code"
