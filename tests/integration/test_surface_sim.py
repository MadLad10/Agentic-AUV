"""Gazebo: a vehicle parked inside the gate frame, under the crossbar, must still reach the surface
(backend stall recovery) - found by the sim-mode eval (M5), where the safe abort could not surface."""
import time

import pytest

from conftest import read_referee

pytestmark = pytest.mark.integration


def test_surface_from_under_the_gate_crossbar(sim_launcher, tmp_path):
    from auv_mission_planner.config import load_config
    from auv_mission_planner.tools.registry import ToolRegistry
    from auv_mission_planner.tools.ros_backend import RosBackend
    sim_launcher(timeout_s=300)
    cfg = load_config()
    backend = RosBackend(cfg)
    assert backend.wait_ready(30)
    tools = ToolRegistry(backend, cfg)
    try:
        # inside the gate opening, just under the 0.9 m deep crossbar (point move: no gate memory used)
        assert tools.call('navigate_to', {'x': 7.0, 'y': 0.0, 'depth': 1.1}, timeout_s=90).ok
        assert tools.call('navigate_to', {'x': 8.0, 'y': 0.0, 'depth': 1.1}, timeout_s=60).ok
        # backend-level recovery only: call the backend directly (no tools-layer gate clearing)
        out = backend.surface(timeout_s=120)
        assert out.ok, out.message
        assert backend.pose().depth < 0.25
    finally:
        backend.shutdown()
    time.sleep(1.5)
    assert read_referee(tmp_path)['surfaced_at_end']


def test_depth_hold_at_the_limit_does_not_overshoot(sim_launcher, tmp_path):
    """Found by the v4 sim eval (I2 reached 4.27 m with a 4.0 m limit): anti-windup + vertical speed limit."""
    from auv_mission_planner.config import load_config
    from auv_mission_planner.tools.registry import ToolRegistry
    from auv_mission_planner.tools.ros_backend import RosBackend
    sim_launcher(timeout_s=240)
    cfg = load_config()
    backend = RosBackend(cfg)
    assert backend.wait_ready(30)
    tools = ToolRegistry(backend, cfg)
    try:
        obs = tools.call('hold_depth', {'depth': cfg['limits']['max_depth_m'], 'duration_s': 8}, timeout_s=120)
        assert obs.ok and obs.data['mean_error_m'] < 0.1, obs.brief()
        assert tools.call('surface', timeout_s=90).ok
    finally:
        backend.shutdown()
    time.sleep(1.5)
    assert read_referee(tmp_path)['max_depth_m'] <= cfg['limits']['max_depth_m'] + 0.1
