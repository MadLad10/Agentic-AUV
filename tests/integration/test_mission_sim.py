"""M5: full missions (planner -> validator -> executor) in headless Gazebo, with a scripted LLM so the
test is deterministic. The referee - not the agent - decides whether the task was really done."""
import json
import os
import subprocess
import time

import pytest
import yaml

from conftest import read_referee

pytestmark = pytest.mark.integration

GOOD = {'status': 'plan', 'reason': '', 'plan': {
    'summary': 'gate, red buoy, surface',
    'abort_conditions': [{'type': 'battery_below', 'percent': 20}],
    'steps': [
        {'id': 1, 'tool': 'hold_depth', 'params': {'depth': 1.5, 'duration_s': 2}, 'timeout_s': 40,
         'success': {'type': 'depth_within', 'depth': 1.5, 'tolerance_m': 0.4}},
        {'id': 2, 'tool': 'navigate_to', 'params': {'target': 'gate', 'pass_through': True}, 'timeout_s': 120},
        {'id': 3, 'tool': 'navigate_to', 'params': {'target': 'red_buoy', 'standoff_m': 0}, 'timeout_s': 120},
        {'id': 4, 'tool': 'surface', 'timeout_s': 60, 'success': {'type': 'surfaced'}, 'on_failure': 'abort'}]}}


def _runner(tmp_path, responses):
    from auv_mission_planner.config import load_config, load_llm_config
    from auv_mission_planner.llm.providers import MockProvider
    from auv_mission_planner.mission import MissionRunner
    from auv_mission_planner.tools.registry import ToolRegistry
    from auv_mission_planner.tools.ros_backend import RosBackend
    cfg = load_config()
    backend = RosBackend(cfg)
    assert backend.wait_ready(30)
    r = MissionRunner(cfg, load_llm_config(), MockProvider(responses=responses), ToolRegistry(backend, cfg),
                      trace_dir=str(tmp_path), log=print)
    return r, backend


def test_full_mission_succeeds_and_referee_agrees(sim_launcher, tmp_path):
    sim_launcher(timeout_s=300)
    r, backend = _runner(tmp_path, [GOOD])
    try:
        res = r.run('pass through the gate, touch the red buoy, then surface', wall_timeout_s=240)
    finally:
        backend.shutdown()
    time.sleep(1.5)
    ref = read_referee(tmp_path)
    assert res['outcome'] == 'succeeded', res['reason']
    assert ref['gate_passed'] and ref['buoys_touched'] == ['red_buoy'] and ref['surfaced_at_end'], ref
    from auv_mission_planner.tracing import validate_trace
    trace = json.load(open(res['trace']))
    validate_trace(trace)
    assert len(trace['tool_calls']) == 4 and trace['outcome']['outcome'] == 'succeeded'
    assert trace['env']['vehicle_backend'] == 'RosBackend' and trace['totals']['tool_failures'] == 0
    assert all(c['sim_end'] >= c['sim_start'] for c in trace['tool_calls'])


def test_low_battery_aborts_and_surfaces(sim_launcher, tmp_path):
    sim_launcher(timeout_s=200, battery_start='22.0')
    r, backend = _runner(tmp_path, [GOOD])
    try:
        res = r.run('gate and red buoy', wall_timeout_s=120)
    finally:
        backend.shutdown()
    time.sleep(1.5)
    ref = read_referee(tmp_path)
    assert res['outcome'] == 'aborted' and 'battery' in res['reason']
    assert res['surfaced_after_abort'] and not ref['gate_passed'] and ref['max_depth_m'] < 0.5


def test_mission_service(sim_launcher, tmp_path):
    from auv_mission_planner.config import load_llm_config
    sim_launcher(timeout_s=300)
    (tmp_path / 'responses.json').write_text(json.dumps([GOOD, GOOD]))   # dry plan + execution plan
    llm = load_llm_config()
    llm['default_provider'] = 'scripted'
    llm['providers']['scripted'] = {'type': 'mock', 'model': 'scripted',
                                    'responses_file': str(tmp_path / 'responses.json')}
    (tmp_path / 'llm.yaml').write_text(yaml.safe_dump(llm))
    server = subprocess.Popen(['timeout', '250', 'ros2', 'run', 'auv_mission_planner', 'mission_server',
                               '--ros-args', '-p', f'llm_config:={tmp_path / "llm.yaml"}',
                               '-p', f'trace_dir:={tmp_path}'],
                              stdout=open(tmp_path / 'server.log', 'w'), stderr=subprocess.STDOUT,
                              start_new_session=True)
    try:
        call = subprocess.run(
            ['timeout', '60', 'ros2', 'service', 'call', '/auv/submit_mission',
             'auv_mission_interfaces/srv/SubmitMission',
             "{prompt: 'pass through the gate, touch the red buoy, then surface', provider: '', dry_run: false}"],
            capture_output=True, text=True)
        assert 'accepted=True' in call.stdout, call.stdout + call.stderr + (tmp_path / 'server.log').read_text()
        assert 'red_buoy' in call.stdout                         # plan_json in the response
        deadline = time.time() + 200
        while time.time() < deadline:
            traces = [p for p in os.listdir(tmp_path) if p.startswith('trace_') and '_plan' not in p]
            if traces:
                break
            time.sleep(2)
        assert traces, (tmp_path / 'server.log').read_text()[-3000:]
        trace = json.load(open(tmp_path / traces[0]))
        assert trace['outcome']['outcome'] == 'succeeded', trace['outcome']
        assert read_referee(tmp_path)['buoys_touched'] == ['red_buoy']
    finally:
        server.send_signal(2)
        server.wait(timeout=15)
