"""M6: replanning on failure (fake backend, scripted LLM)."""
import copy
import json

import pytest

from auv_mission_planner.config import load_config, load_llm_config
from auv_mission_planner.llm.providers import MockProvider
from auv_mission_planner.mission import MissionRunner
from auv_mission_planner.tools.fake_backend import FakeBackend
from auv_mission_planner.tools.registry import ToolRegistry

HIDDEN = [{'id': 'gate', 'class': 'gate', 'x': 8.0, 'y': 0.0, 'z': -1.5},
          {'id': 'red_buoy', 'class': 'red_buoy', 'x': 9.0, 'y': 7.0, 'z': -1.5}]


def P(*steps, aborts=None):
    return {'status': 'plan', 'reason': '', 'plan': {
        'summary': 't', 'abort_conditions': aborts or [{'type': 'battery_below', 'percent': 20}],
        'steps': [dict(id=i + 1, **s) for i, s in enumerate(steps)]}}


DIVE = {'tool': 'hold_depth', 'params': {'depth': 1.5, 'duration_s': 2}, 'timeout_s': 40}
SWEEP = {'tool': 'search_pattern', 'params': {'target': 'red_buoy', 'pattern': 'yaw_sweep'}, 'timeout_s': 120,
         'success': {'type': 'detected', 'target': 'red_buoy'}}
SPIRAL = {'tool': 'search_pattern', 'params': {'target': 'red_buoy', 'pattern': 'spiral', 'radius_m': 10},
          'timeout_s': 240, 'success': {'type': 'detected', 'target': 'red_buoy'}}
RED = {'tool': 'navigate_to', 'params': {'target': 'red_buoy', 'standoff_m': 0}, 'timeout_s': 120}
SURF = {'tool': 'surface', 'timeout_s': 60, 'success': {'type': 'surfaced'}, 'on_failure': 'abort'}


@pytest.fixture
def cfg():
    c = load_config()
    c['executor']['watchdog_period_s'] = 0.05
    c['executor']['abort_surface_timeout_s'] = 5
    return c


def run(cfg, responses, tmp_path, max_replans=2, **kw):
    cfg = copy.deepcopy(cfg)
    cfg['executor']['max_replans'] = max_replans
    b = FakeBackend(objects=HIDDEN, **kw)
    r = MissionRunner(cfg, load_llm_config(), MockProvider(responses=responses), ToolRegistry(b, cfg),
                      trace_dir=str(tmp_path))
    return r.run('touch the red buoy then surface'), b, r


def test_failed_search_is_replanned_and_mission_succeeds(cfg, tmp_path):
    res, b, r = run(cfg, [P(DIVE, SWEEP, RED, SURF), P(SPIRAL, RED, SURF)], tmp_path)
    assert res['outcome'] == 'succeeded', res['reason']
    assert res['replans'] == 1 and res['plans'] == 2 and 'red_buoy' in b.touched and b.surfaced
    replan_msg = r.provider.calls[1][1]['content']
    assert 'replan 1 of 2' in replan_msg and 'yaw_sweep' in replan_msg and 'not_found' in replan_msg
    assert '"tool": "hold_depth"' in replan_msg                       # completed steps are reported
    trace = json.load(open(res['trace']))
    assert len(trace['replans']) == 1 and trace['replans'][0]['failed']['step']['tool'] == 'search_pattern'
    assert [p['kind'] for p in trace['plans']] == ['plan', 'replan']
    kinds = [c['kind'] for c in trace['llm_calls']]
    assert kinds == ['plan', 'replan']


def test_target_not_detected_triggers_replan(cfg, tmp_path):
    res, b, r = run(cfg, [P(DIVE, RED, SURF), P(SPIRAL, RED, SURF)], tmp_path)
    assert res['outcome'] == 'succeeded' and res['replans'] == 1
    assert 'target_not_detected' in r.provider.calls[1][1]['content']


def test_replans_exhausted_aborts_safely(cfg, tmp_path):
    res, b, r = run(cfg, [P(DIVE, SPIRAL, RED, SURF)] * 3, tmp_path, blind=['red_buoy'])
    assert res['outcome'] == 'aborted' and res['replans'] == 2
    assert 'replans exhausted (2)' in res['reason'] and b.emergency and res['surfaced_after_abort']


def test_refusal_during_replan_aborts(cfg, tmp_path):
    res, b, _ = run(cfg, [P(DIVE, RED, SURF), {'status': 'refuse', 'reason': 'buoy cannot be found', 'plan': None}],
                    tmp_path)
    assert res['outcome'] == 'aborted' and 'replan 1 refuse' in res['reason'] and b.emergency


def test_unsafe_replan_is_caught_and_fixed(cfg, tmp_path):
    deep = {'tool': 'search_pattern', 'params': {'target': 'red_buoy', 'pattern': 'spiral', 'depth': 25},
            'timeout_s': 200}
    res, b, _ = run(cfg, [P(DIVE, RED, SURF), P(deep, RED, SURF), P(SPIRAL, RED, SURF)], tmp_path)
    assert res['outcome'] == 'succeeded' and res['replans'] == 1 and res['unsafe_plans_caught'] == 1
    assert not any(c[0] == 'search' and len(c) > 3 for c in b.calls)


def test_no_replanning_when_disabled(cfg, tmp_path):
    res, b, r = run(cfg, [P(DIVE, RED, SURF)], tmp_path, max_replans=0)
    assert res['outcome'] == 'aborted' and res['replans'] == 0 and len(r.provider.calls) == 1


def test_on_failure_abort_skips_replanning(cfg, tmp_path):
    red_abort = dict(RED, on_failure='abort')
    res, b, r = run(cfg, [P(DIVE, red_abort, SURF)], tmp_path)
    assert res['outcome'] == 'aborted' and res['replans'] == 0 and len(r.provider.calls) == 1
