#!/usr/bin/env bash
# Reproduce the GitHub Actions job locally in the same ros:humble container (no GPU, OSMesa rendering).
# The repo is copied into the container (read-only mount), so nothing in the working tree changes.
set -eu
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker run --rm -e ROS_TEST_FLAGS -e ROS_TEST_TAIL -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa -v "$WS":/src:ro ros:humble bash -c '
set -e
mkdir /ws && cd /src && tar --exclude=./.venv --exclude=./build --exclude=./install --exclude=./log \
  --exclude=./data --exclude=./runs --exclude=./runs_yolo --exclude=./.git -cf - . | tar -xf - -C /ws
cd /ws
apt-get update -qq && apt-get install -y -qq --no-install-recommends python3-pip libosmesa6 libgl1-mesa-dri \
  ros-humble-nav-msgs ros-humble-sensor-msgs ros-humble-std-srvs ros-humble-rosgraph-msgs >/dev/null
pip install -q --no-cache-dir "numpy==1.26.4" "pytest==7.4.4" "pydantic>=2.7" pyyaml pillow "mujoco>=3.3" \
  "langgraph>=1.0" langchain-core langgraph-checkpoint-sqlite "mcp>=1.10" httpx opentelemetry-sdk \
  opentelemetry-exporter-otlp-proto-http scipy fastapi uvicorn "gtsam==4.2" 2>&1 | tail -2
source /opt/ros/humble/setup.bash && colcon build --symlink-install 2>&1 | tail -2
source install/setup.bash
python3 -m pytest -q -m "not integration and not llm and not ros" 2>&1 | tail -5
python3 -m pytest -q -m ros tests/integration/test_ros2_vehicle.py -p no:warnings ${ROS_TEST_FLAGS:-} 2>&1 | tail -${ROS_TEST_TAIL:-2}
'
