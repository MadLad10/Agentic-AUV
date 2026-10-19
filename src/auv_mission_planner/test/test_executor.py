"""Executor + MissionRunner end to end on the fake backend with a scripted LLM (no ROS, no network)."""
import copy
import json

import pytest

from auv_mission_planner.config import load_config, load_llm_config
from auv_mission_planner.executor import Executor, Watchdog
from auv_mission_planner.llm.providers import MockProvider
from auv_mission_planner.mission import MissionRunner
from auv_mission_planner.plan import MissionPlan, Step
from auv_mission_planner.tools.fake_backend import FakeBackend
from auv_mission_planner.tools.registry import ToolRegistry
from auv_mission_planner.tools.schemas import Observation
from auv_mission_planner.validator import SafetyValidator


def P(steps, aborts=None):
    return {'status': 'plan', 'reason': '', 'plan': {
        'summary': 't', 'abort_conditions': aborts if aborts is not None else [{'type': 'battery_below',
                                                                                 'percent': 20}],
        'steps': [dict(id=i + 1, **s) for i, s in enumerate(steps)]}}


DIVE = {'tool': 'hold_depth', 'params': {'depth': 1.5, 'duration_s': 2}, 'timeout_s': 40,
        'success': {'type': 'depth_within', 'depth': 1.5, 'tolerance_m': 0.4}}
GATE = {'tool': 'navigate_to', 'params': {'target': 'gate', 'pass_through': True}, 'timeout_s': 120}
RED = {'tool': 'navigate_to', 'params': {'target': 'red_buoy', 'standoff_m': 0}, 'timeout_s': 120}
SURF = {'tool': 'surface', 'timeout_s': 60, 'success': {'type': 'surfaced'}, 'on_failure': 'abort'}
GOOD = P([DIVE, GATE, RED, SURF])


@pytest.fixture
def cfg():
    c = load_config()
    c['executor']['watchdog_period_s'] = 0.05
    c['executor']['abort_surface_timeout_s'] = 5
    return c


def runner(cfg, responses, tmp_path, max_replans=0, **backend_kw):
    cfg = copy.deepcopy(cfg)
    cfg['executor']['max_replans'] = max_replans
    b = FakeBackend(**backend_kw)
    tools = ToolRegistry(b, cfg)
    r = MissionRunner(cfg, load_llm_config(), MockProvider(responses=responses), tools, trace_dir=str(tmp_path))
    return r, b


def test_success_end_to_end_with_trace(cfg, tmp_path):
    r, b = runner(cfg, [GOOD], tmp_path)
    res = r.run('pass through the gate, touch the red buoy, then surface')
    assert res['outcome'] == 'succeeded', res['reason']
    assert res['steps_executed'] == 4 and b.gate_passed and 'red_buoy' in b.touched and b.surfaced
    assert res['tokens_in'] > 0 and res['llm_calls'] == 1 and res['unsafe_plans_caught'] == 0
    trace = json.load(open(res['trace']))
    assert trace['prompt'].startswith('pass through') and len(trace['tool_calls']) == 4
    assert trace['plans'][0]['status'] == 'plan' and trace['validations'][0]['ok']
    assert trace['outcome']['outcome'] == 'succeeded'


def test_dry_run_moves_nothing(cfg, tmp_path):
    r, b = runner(cfg, [GOOD], tmp_path)
    res = r.run('pass the gate, touch the red buoy, surface', dry_run=True)
    assert res['outcome'] == 'planned' and res['final_plan']['steps'][0]['tool'] == 'hold_depth'
    assert b.calls == []


def test_refusal_executes_nothing(cfg, tmp_path):
    r, b = runner(cfg, [{'status': 'refuse', 'reason': 'thrusters are not a tool', 'plan': None}], tmp_path)
    res = r.run('thrusters to 100%')
    assert res['outcome'] == 'refused' and b.calls == []


def test_validator_rejection_is_fed_back_then_fixed(cfg, tmp_path):
    unsafe = P([{'tool': 'hold_depth', 'params': {'depth': 30, 'duration_s': 5}, 'timeout_s': 60}, SURF])
    r, b = runner(cfg, [unsafe, GOOD], tmp_path)
    res = r.run('dive deep then do the gate and red buoy')
    assert res['outcome'] == 'succeeded' and res['unsafe_plans_caught'] == 1 and res['plans'] == 2
    assert 'REJECTED' in r.provider.calls[1][-1]['content']
    assert not any(c[0] == 'hold_depth' and c[1] == 30 for c in b.calls)


def test_persistently_unsafe_plan_is_rejected_without_motion(cfg, tmp_path):
    unsafe = P([{'tool': 'set_thrusters', 'params': {'all': 100}}, SURF])
    r, b = runner(cfg, [unsafe] * 3, tmp_path)
    res = r.run('thrusters 100')
    assert res['outcome'] == 'rejected' and res['unsafe_plans_caught'] == 3
    assert 'tool_not_allowed' in res['reason'] and b.calls == []


def test_repairs_applied_and_counted(cfg, tmp_path):
    r, b = runner(cfg, [P([DIVE, GATE], aborts=[])], tmp_path)           # no surface, no battery abort
    res = r.run('do the gate')
    assert res['outcome'] == 'succeeded' and res['validator_repairs'] >= 2
    assert res['final_plan']['steps'][-1]['tool'] == 'surface' and b.surfaced


def test_step_failure_without_replanning_aborts_safely(cfg, tmp_path):
    hidden = [{'id': 'gate', 'class': 'gate', 'x': 8, 'y': 0, 'z': -1.5},
              {'id': 'red_buoy', 'class': 'red_buoy', 'x': 9, 'y': 7, 'z': -1.5}]
    r, b = runner(cfg, [P([DIVE, RED, SURF])], tmp_path, objects=hidden)
    res = r.run('touch the red buoy')
    assert res['outcome'] == 'aborted' and 'target_not_detected' in res['reason'] or 'not detected' in res['reason']
    assert b.emergency and res['surfaced_after_abort'] is True


def test_on_failure_continue_and_abort(cfg, tmp_path):
    bad = {'tool': 'navigate_to', 'params': {'target': 'red_buoy'}, 'timeout_s': 60, 'on_failure': 'continue'}
    hidden = [{'id': 'gate', 'class': 'gate', 'x': 8, 'y': 0, 'z': -1.5}]
    r, b = runner(cfg, [P([DIVE, bad, SURF])], tmp_path, objects=hidden)
    assert r.run('dive, touch the red buoy if you can, surface')['outcome'] == 'succeeded'
    bad['on_failure'] = 'abort'
    r, b = runner(cfg, [P([DIVE, bad, SURF])], tmp_path, objects=hidden)
    res = r.run('dive, touch the red buoy, surface')
    assert res['outcome'] == 'aborted' and 'step 2' in res['reason']


def test_plan_abort_step(cfg, tmp_path):
    r, b = runner(cfg, [P([DIVE, {'tool': 'abort', 'params': {'reason': 'operator asked'}}])], tmp_path)
    res = r.run('dive then abort')
    assert res['outcome'] == 'aborted' and 'operator asked' in res['reason'] and b.emergency


def test_low_battery_precheck_aborts_before_motion(cfg, tmp_path):
    r, b = runner(cfg, [GOOD], tmp_path, battery_start=22.0)
    res = r.run('pass the gate, touch the red buoy, surface')
    assert res['outcome'] == 'aborted' and 'battery' in res['reason']
    assert not any(c[0] in ('navigate', 'hold_depth') for c in b.calls) and b.emergency


def test_battery_drain_mid_mission_triggers_abort(cfg, tmp_path):
    # 1 %/s drain: the long gate leg drains below the 30 % threshold
    r, b = runner(cfg, [P([DIVE, GATE, RED, SURF], aborts=[{'type': 'battery_below', 'percent': 30}])], tmp_path,
                  battery_start=60.0, drain_pct_per_s=1.0)
    res = r.run('pass the gate, touch the red buoy, surface')
    assert res['outcome'] == 'aborted' and 'battery' in res['reason'] and b.surfaced


def test_watchdog_conditions(cfg):
    b = FakeBackend(battery_start=50)
    tools = ToolRegistry(b, cfg)
    wd = Watchdog(tools, 0.05, wall_deadline=None)
    wd.plan = MissionPlan.model_validate({'steps': [{'id': 1, 'tool': 'surface'}],
                                          'abort_conditions': [{'type': 'battery_below', 'percent': 40},
                                                               {'type': 'mission_time_exceeded', 'seconds': 100}]})
    assert wd.check() is None
    b.t += 150
    assert 'mission time' in wd.check()
    b.t -= 150
    b.batt = 30
    assert 'battery' in wd.check()
    wd2 = Watchdog(tools, 0.05, wall_deadline=0.0)
    assert 'wall-clock' in wd2.check()


def test_success_condition_evaluation(cfg):
    b = FakeBackend()
    tools = ToolRegistry(b, cfg)
    ex = Executor(tools, SafetyValidator(cfg['limits']), cfg)
    ok = Observation(tool='hold_depth', ok=True, data={})
    s = Step(id=1, tool='hold_depth', success={'type': 'depth_within', 'depth': 2.0, 'tolerance_m': 0.3})
    assert ex.evaluate(s, ok)[0] is False                      # vehicle still at 0.1 m
    b._pose.depth = 2.1
    assert ex.evaluate(s, ok)[0] is True
    s = Step(id=2, tool='get_detections', success={'type': 'detected', 'target': 'gate'})
    assert ex.evaluate(s, Observation(tool='get_detections', ok=True,
                                      data={'visible': [{'class': 'gate'}]}))[0] is True
    s = Step(id=3, tool='navigate_to', success={'type': 'reached', 'tolerance_m': 0.2})
    assert ex.evaluate(s, Observation(tool='navigate_to', ok=True, data={'final_error_m': 0.5}))[0] is False
    s = Step(id=4, tool='get_battery', success={'type': 'battery_above', 'percent': 50})
    assert ex.evaluate(s, Observation(tool='get_battery', ok=True))[0] is True


def test_cli_exit_codes(monkeypatch, tmp_path, capsys):
    from auv_mission_planner import cli
    responses = {'ok': [GOOD], 'refuse': [{'status': 'refuse', 'reason': 'no', 'plan': None}]}
    for key, code in (('ok', 0), ('refuse', 3)):
        monkeypatch.setattr('auv_mission_planner.llm.providers.make_provider',
                            lambda cfg, name=None, _r=responses[key]: MockProvider(responses=list(_r)))
        rc = cli.main(['--backend', 'fake', '--trace-dir', str(tmp_path), '--json', 'pass the gate, touch the red buoy, then surface'])
        out = json.loads(capsys.readouterr().out)
        assert rc == code and out['outcome'] in ('succeeded', 'refused')
