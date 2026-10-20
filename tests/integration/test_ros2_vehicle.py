"""ROS 2: the MuJoCo vehicle node in its own process, driven by v1's unmodified RosBackend.

Marked `ros` (needs ROS 2 Humble sourced; ~30 s). Uses a private ROS_DOMAIN_ID.
"""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.ros


@pytest.fixture
def vehicle(tmp_path):
    # identical discovery settings for the vehicle and this test process (set before rclpy starts)
    os.environ.update(ROS_DOMAIN_ID='77', ROS_LOCALHOST_ONLY='1')
    log = open(tmp_path / 'vehicle.log', 'w')      # a file, not an unread pipe that could fill and block
    p = subprocess.Popen([sys.executable, '-m', 'auv_onboard.ros_node', '--ros-args', '-p', 'rtf:=6.0'],
                         env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    yield p, tmp_path / 'vehicle.log'
    os.killpg(p.pid, signal.SIGINT)
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
    log.close()


def test_v1_ros_backend_drives_mujoco_vehicle(vehicle):
    from std_srvs.srv import Trigger
    from auv_mission_planner.config import load_config
    from auv_mission_planner.tools.registry import ToolRegistry
    from auv_mission_planner.tools.ros_backend import RosBackend
    cfg = load_config()
    be = RosBackend(cfg)
    try:
        _, log = vehicle
        assert be.wait_ready(90), 'vehicle node not ready; its log:\n' + log.read_text()[-3000:]
        reg = ToolRegistry(be, cfg)
        for name, params, to in [('navigate_to', {'x': 2.0, 'y': 0.0, 'depth': 1.5}, 60),
                                 ('navigate_to', {'target': 'gate'}, 120), ('surface', {}, 90)]:
            obs = reg.call(name, params, to)
            assert obs.ok, (name, obs.error)
        cli = be.node.create_client(Trigger, '/auv_sim/referee_report')
        assert cli.wait_for_service(timeout_sec=5)
        fut = cli.call_async(Trigger.Request())
        t_end = time.time() + 10
        while not fut.done() and time.time() < t_end:
            time.sleep(0.05)
        rep = json.loads(fut.result().message)
        assert rep['gate_passed'] and rep['surfaced_at_end'] and rep['collisions'] == 0
    finally:
        be.shutdown()
