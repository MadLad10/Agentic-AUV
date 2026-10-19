"""M7: every mission produces a complete, schema-valid JSON trace."""
import copy
import json
import os
import subprocess
import sys

import pytest

from auv_mission_planner.config import load_config, load_llm_config
from auv_mission_planner.llm.providers import MockProvider
from auv_mission_planner.mission import MissionRunner
from auv_mission_planner.tools.fake_backend import FakeBackend
from auv_mission_planner.tools.registry import ToolRegistry
from auv_mission_planner.tracing import SCHEMA_VERSION, validate_trace

HIDDEN = [{'id': 'gate', 'class': 'gate', 'x': 8.0, 'y': 0.0, 'z': -1.5},
          {'id': 'red_buoy', 'class': 'red_buoy', 'x': 9.0, 'y': 7.0, 'z': -1.5}]
WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def P(*steps):
    return {'status': 'plan', 'reason': '', 'plan': {'summary': 't', 'abort_conditions': [],
                                                     'steps': [dict(id=i + 1, **s) for i, s in enumerate(steps)]}}


DIVE = {'tool': 'hold_depth', 'params': {'depth': 1.5, 'duration_s': 2}, 'timeout_s': 40}
RED = {'tool': 'navigate_to', 'params': {'target': 'red_buoy'}, 'timeout_s': 120}
SPIRAL = {'tool': 'search_pattern', 'params': {'target': 'red_buoy', 'pattern': 'spiral', 'radius_m': 10},
          'timeout_s': 240}
SURF = {'tool': 'surface', 'timeout_s': 60}


def mission(responses, tmp_path, **kw):
    cfg = copy.deepcopy(load_config())
    cfg['executor']['abort_surface_timeout_s'] = 5
    r = MissionRunner(cfg, load_llm_config(), MockProvider(responses=responses),
                      ToolRegistry(FakeBackend(objects=HIDDEN, **kw), cfg), trace_dir=str(tmp_path))
    res = r.run('touch the red buoy then surface')
    return res, json.load(open(res['trace']))


def test_trace_with_replan_is_complete_and_valid(tmp_path):
    bad_json = 'not json at all'
    res, t = mission([P(DIVE, RED, SURF), bad_json, P(SPIRAL, RED, SURF)], tmp_path)
    assert res['outcome'] == 'succeeded'
    validate_trace(t)
    assert t['schema_version'] == SCHEMA_VERSION and t['prompt'].startswith('touch')
    assert t['git_commit'] and t['config']['limits']['max_depth_m'] == 4.0
    assert t['env']['vehicle_backend'] == 'FakeBackend'
    # LLM calls: initial plan, replan, replan format retry
    assert [c['kind'] for c in t['llm_calls']] == ['plan', 'replan', 'replan_format_retry']
    assert t['llm_calls'][1]['parse_error'] and t['llm_calls'][1]['response'] == bad_json
    # the system prompt is stored once and referenced
    assert len(t['prompts']) == 1
    ref = t['llm_calls'][0]['messages'][0]['content_ref']
    assert ref in t['prompts'] and 'mission planner for AUV' in t['prompts'][ref]
    # plans, validations (with repairs), tool calls, replan record
    assert [p['kind'] for p in t['plans']] == ['plan', 'replan'] and len(t['validations']) == 2
    assert any(f['rule'] == 'battery_abort' for f in t['validations'][0]['repaired'])
    failed = [c for c in t['tool_calls'] if not c['success']]
    assert len(failed) == 1 and failed[0]['observation']['error_code'] == 'target_not_detected'
    assert t['replans'][0]['failed']['step']['tool'] == 'navigate_to'
    assert t['replans'][0]['context']['pose'] is not None
    # totals agree with the outcome
    tot = t['totals']
    assert tot['replans'] == res['replans'] == 1 and tot['format_errors'] == 1
    assert tot['tokens_in'] == res['tokens_in'] > 0 and tot['tool_calls'] == len(t['tool_calls'])
    assert tot['llm_calls_by_kind'] == {'plan': 1, 'replan': 1, 'replan_format_retry': 1}


def test_trace_for_rejected_mission(tmp_path):
    unsafe = P({'tool': 'hold_depth', 'params': {'depth': 40}, 'timeout_s': 60}, SURF)
    res, t = mission([unsafe] * 3, tmp_path)
    validate_trace(t)
    assert res['outcome'] == 'rejected' and t['tool_calls'] == []
    assert t['totals']['unsafe_plans_caught'] == 3
    assert all(not v['ok'] and v['rejected'][0]['rule'] == 'depth_limit' for v in t['validations'])
    kinds = [c['kind'] for c in t['llm_calls']]
    assert kinds == ['plan', 'validation_repair', 'validation_repair']


def test_trace_for_llm_error_and_abort(tmp_path):
    res, t = mission([], tmp_path)
    validate_trace(t)
    assert res['outcome'] == 'error' and t['llm_calls'][0]['error']
    res, t = mission([P(DIVE, RED, SURF)], tmp_path, battery_start=22.0)
    validate_trace(t)
    assert res['outcome'] == 'aborted' and any(e['kind'] == 'safe_abort' for e in t['events'])


def test_validate_trace_catches_malformed(tmp_path):
    res, t = mission([P(DIVE, SPIRAL, RED, SURF)], tmp_path)
    broken = dict(t)
    del broken['outcome']
    with pytest.raises(Exception):
        validate_trace(broken)
    broken = json.loads(json.dumps(t))
    broken['llm_calls'][0]['messages'][0]['content_ref'] = 'deadbeef'
    with pytest.raises(ValueError):
        validate_trace(broken)


def test_show_trace_script(tmp_path):
    res, _ = mission([P(DIVE, RED, SURF), P(SPIRAL, RED, SURF)], tmp_path)
    out = subprocess.run([sys.executable, os.path.join(WS, 'scripts', 'show_trace.py'), res['trace']],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    for token in ('SUCCEEDED', 'PLAN #1', 'PLAN #2', 'REPLAN #1', 'TOOL step', 'VAL '):
        assert token in out.stdout, token
