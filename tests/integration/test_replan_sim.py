"""M6 in Gazebo: the red buoy is hidden, the first plan's yaw sweep cannot see it, the executor
reports the failure, the (scripted) LLM replans with a spiral search, and the referee confirms the touch."""
import time

import pytest

from conftest import read_referee

pytestmark = pytest.mark.integration


def P(*steps):
    return {'status': 'plan', 'reason': '', 'plan': {
        'summary': 't', 'abort_conditions': [{'type': 'battery_below', 'percent': 20}],
        'steps': [dict(id=i + 1, **s) for i, s in enumerate(steps)]}}


DIVE = {'tool': 'hold_depth', 'params': {'depth': 1.5, 'duration_s': 2}, 'timeout_s': 40}
SWEEP = {'tool': 'search_pattern', 'params': {'target': 'red_buoy', 'pattern': 'yaw_sweep'}, 'timeout_s': 90,
         'success': {'type': 'detected', 'target': 'red_buoy'}}
SPIRAL = {'tool': 'search_pattern', 'params': {'target': 'red_buoy', 'pattern': 'spiral', 'radius_m': 8},
          'timeout_s': 240, 'success': {'type': 'detected', 'target': 'red_buoy'}}
RED = {'tool': 'navigate_to', 'params': {'target': 'red_buoy', 'standoff_m': 0}, 'timeout_s': 120}
SURF = {'tool': 'surface', 'timeout_s': 60, 'success': {'type': 'surfaced'}, 'on_failure': 'abort'}


def test_hidden_buoy_replan_succeeds(sim_launcher, tmp_path):
    from auv_mission_planner.config import load_config, load_llm_config
    from auv_mission_planner.llm.providers import MockProvider
    from auv_mission_planner.mission import MissionRunner
    from auv_mission_planner.tools.registry import ToolRegistry
    from auv_mission_planner.tools.ros_backend import RosBackend
    sim_launcher(timeout_s=400, arena='buoy_hidden')
    cfg = load_config()
    backend = RosBackend(cfg)
    assert backend.wait_ready(30)
    runner = MissionRunner(cfg, load_llm_config(), MockProvider(responses=[P(DIVE, SWEEP, RED, SURF),
                                                                           P(SPIRAL, RED, SURF)]),
                           ToolRegistry(backend, cfg), trace_dir=str(tmp_path), log=print)
    try:
        res = runner.run('touch the red buoy then surface', wall_timeout_s=300)
    finally:
        backend.shutdown()
    time.sleep(1.5)
    ref = read_referee(tmp_path)
    assert res['outcome'] == 'succeeded', res['reason']
    assert res['replans'] == 1
    assert ref['buoys_touched'] == ['red_buoy'] and ref['surfaced_at_end'], ref
