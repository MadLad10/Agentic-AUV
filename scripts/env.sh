# source scripts/env.sh   -- ROS 2 Humble + this workspace overlay (if built)
_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
[ -f "$_WS/install/setup.bash" ] && source "$_WS/install/setup.bash"
export AUV_WS="$_WS"
export IGN_GAZEBO_RESOURCE_PATH="${IGN_GAZEBO_RESOURCE_PATH:-}"
# keep sim traffic off the LAN / other ROS graphs on this machine
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-43}"
# v2: uv venv (python deps) on top of ROS; MuJoCo renders headless through EGL on the iGPU
[ -d "$_WS/.venv" ] && export VIRTUAL_ENV="$_WS/.venv" && export PATH="$_WS/.venv/bin:$PATH"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
