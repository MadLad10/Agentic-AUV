"""M2: the tools layer drives the Gazebo vehicle with no LLM; the referee confirms the result."""
import time

import pytest

from conftest import read_referee

pytestmark = pytest.mark.integration


@pytest.fixture
def tools(sim_launcher, request):
    from auv_mission_planner.config import load_config
    from auv_mission_planner.tools.registry import ToolRegistry
    from auv_mission_planner.tools.ros_backend import RosBackend
    sim_launcher(timeout_s=300, **getattr(request, 'param', {}))
    cfg = load_config()
    backend = RosBackend(cfg)
    assert backend.wait_ready(30), 'vehicle backend not ready'
    yield ToolRegistry(backend, cfg)
    backend.shutdown()


def test_scripted_gate_buoy_surface(tools, tmp_path):
    start_batt = tools.call('get_battery').data['percent']
    seq = [('hold_depth', {'depth': 1.5, 'duration_s': 2}),
           ('get_detections', {}),
           ('navigate_to', {'target': 'gate', 'pass_through': True}),
           ('navigate_to', {'target': 'red_buoy'}),
           ('surface', {})]
    for name, params in seq:
        obs = tools.call(name, params, timeout_s=120)
        assert obs.ok, obs.brief()
    time.sleep(1.0)
    ref = read_referee(tmp_path)
    assert ref['gate_passed'], ref
    assert ref['buoys_touched'] == ['red_buoy'], ref
    assert ref['surfaced_at_end'], ref
    assert ref['boundary_contacts'] == 0
    assert tools.call('get_battery').data['percent'] < start_batt


@pytest.mark.parametrize('tools', [{'arena': 'buoy_hidden'}], indirect=True)
def test_hidden_buoy_needs_search(tools, tmp_path):
    assert tools.call('hold_depth', {'depth': 1.5, 'duration_s': 1}, timeout_s=60).ok
    obs = tools.call('navigate_to', {'target': 'red_buoy'}, timeout_s=60)
    assert not obs.ok and obs.error_code == 'target_not_detected'
    obs = tools.call('search_pattern', {'target': 'red_buoy', 'pattern': 'spiral', 'radius_m': 8}, timeout_s=200)
    assert obs.ok and obs.data['found'], obs.brief()
    obs = tools.call('navigate_to', {'target': 'red_buoy'}, timeout_s=120)
    assert obs.ok, obs.brief()
    time.sleep(1.0)
    assert 'red_buoy' in read_referee(tmp_path)['buoys_touched']
