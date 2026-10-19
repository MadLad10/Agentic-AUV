#!/usr/bin/env bash
# Kill leftover Gazebo / ROS 2 processes from previous sim runs. Safe to run any time.
# Does NOT touch the Ollama container. Never matches its own ancestors (the calling shell's command
# line may contain the same words).
set -u
REGEX='(^|/)ign gazebo|ign-gazebo|gz sim|/ruby .*ign|parameter_bridge|/install/auv_[a-z_]+/lib/|ros2 launch auv_|ros2 bag record|_ros2_daemon'

ancestors() {
  local p=$$
  while [ -n "$p" ] && [ "$p" != "0" ] && [ "$p" != "1" ]; do
    echo "$p"
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  done
}

targets() {
  local skip
  skip=" $(ancestors | tr '\n' ' ') "
  ps -eo pid=,args= | while read -r pid args; do
    [[ "$skip" == *" $pid "* ]] && continue
    if [[ "$args" =~ $REGEX ]]; then echo "$pid"; fi
  done
}

# the ros2 CLI daemon caches graph state across runs and can wedge (xmlrpc "!rclpy.ok()")
command -v ros2 >/dev/null 2>&1 && timeout 10 ros2 daemon stop >/dev/null 2>&1
pids=$(targets)
[ -n "$pids" ] && kill -INT $pids 2>/dev/null
for _ in 1 2 3 4 5 6; do
  pids=$(targets)
  [ -z "$pids" ] && break
  sleep 1
done
if [ -n "$pids" ]; then kill -KILL $pids 2>/dev/null; sleep 1; pids=$(targets); fi
if [ -n "$pids" ]; then
  echo "[cleanup] still running:"; ps -o pid,args -p $pids; exit 1
fi
echo "[cleanup] sim processes cleared"
