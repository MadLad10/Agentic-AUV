"""v4 fixes found by the sim-mode eval (M3: search from the surface with useless timeouts;
M5: 'go through the gate' without pass_through, then stuck under the crossbar while surfacing)."""
import math

import pytest

from auv_mission_planner.config import load_config
from auv_mission_planner.plan import MissionPlan
from auv_mission_planner.prompts import system_prompt
from auv_mission_planner.tools.backend import Det
from auv_mission_planner.tools.fake_backend import FakeBackend
from auv_mission_planner.tools.registry import ToolRegistry
from auv_mission_planner.tools.schemas import NavigateToParams
from auv_mission_planner.validator import SafetyValidator


@pytest.fixture
def cfg():
    return load_config()


def test_gate_target_passes_through_by_default():
    assert NavigateToParams(target='gate').passes_gate
    assert not NavigateToParams(target='gate', standoff_m=2.0).passes_gate        # stop in front
    assert not NavigateToParams(target='gate', pass_through=False).passes_gate
    assert not NavigateToParams(target='red_buoy').passes_gate


def test_navigate_to_gate_without_flag_goes_through(cfg):
    b = FakeBackend()
    t = ToolRegistry(b, cfg)
    t.call('get_detections')
    o = t.call('navigate_to', {'target': 'gate', 'depth': 2.0}, timeout_s=120)
    assert o.ok and o.data['phase'] == 'gate_exit' and b.gate_passed
    assert b.pose().x > 9.0                                                        # past the gate, not in it


def test_search_from_surface_uses_object_depth(cfg):
    b = FakeBackend()
    t = ToolRegistry(b, cfg)
    seen = {}
    orig = b.search

    def spy(target, pattern, radius_m, depth, timeout_s=60.0):
        seen['depth'] = depth
        return orig(target, pattern, radius_m, depth, timeout_s)
    b.search = spy
    t.call('search_pattern', {'target': 'gate'})
    assert seen['depth'] == cfg['tools']['default_search_depth_m']
    b._pose.depth = 2.5
    t.call('search_pattern', {'target': 'gate'})
    assert seen['depth'] == 2.5                                                    # already deep: keep it


def test_surface_clears_gate_first(cfg):
    b = FakeBackend(start=(8.0, 0.0, 1.0, 0.0))                                    # inside the gate frame
    t = ToolRegistry(b, cfg)
    t.world.update([Det('gate', 0.9, 8.0, 0.0, 1.5, 0.5, 0.0, t=b.now())])
    o = t.call('surface', timeout_s=60)
    assert o.ok and o.data.get('cleared_gate_first') is True
    assert b.pose().x >= 9.0 and b.pose().depth < 0.2
    nav = [c for c in b.calls if c[0] == 'navigate']
    assert nav and nav[0][3] == 1.0                                                # moved out at the same depth
    # far from the gate: no detour
    b2 = FakeBackend(start=(2.0, 0.0, 1.0, 0.0))
    t2 = ToolRegistry(b2, cfg)
    t2.world.update([Det('gate', 0.9, 8.0, 0.0, 1.5, 6.0, 0.0, t=b2.now())])
    assert 'cleared_gate_first' not in t2.call('surface').data


def test_validator_raises_useless_search_timeout(cfg):
    v = SafetyValidator(cfg['limits'])
    p = MissionPlan.model_validate({'steps': [
        {'id': 1, 'tool': 'search_pattern', 'params': {'target': 'green_buoy', 'pattern': 'spiral'}, 'timeout_s': 60},
        {'id': 2, 'tool': 'surface', 'timeout_s': 60}],
        'abort_conditions': [{'type': 'battery_below', 'percent': 20}]})
    rep = v.validate(p)
    assert rep.ok and rep.plan.steps[0].timeout_s == 180
    assert any(f.rule == 'step_timeout' and 'too short' in f.message for f in rep.repaired)


def test_prompt_has_course_layout(cfg):
    t = ToolRegistry(FakeBackend(), cfg)
    s = system_prompt(t.schemas(), cfg['limits'])
    assert 'beyond it' in s and 'NOT visible from the start' in s


def test_empty_detected_target_is_inferred(cfg):
    v = SafetyValidator(cfg['limits'])
    p = MissionPlan.model_validate({'steps': [
        {'id': 1, 'tool': 'get_detections', 'success': {'type': 'detected', 'target': ''}},
        {'id': 2, 'tool': 'search_pattern', 'params': {'target': 'yellow_buoy'}, 'timeout_s': 150},
        {'id': 3, 'tool': 'surface', 'timeout_s': 60}],
        'abort_conditions': [{'type': 'battery_below', 'percent': 20}]})
    rep = v.validate(p)
    assert rep.ok and rep.plan.steps[0].success.target == 'yellow_buoy'
    p = MissionPlan.model_validate({'steps': [
        {'id': 1, 'tool': 'get_detections', 'success': {'type': 'detected', 'target': ' '}},
        {'id': 2, 'tool': 'surface', 'timeout_s': 60}],
        'abort_conditions': [{'type': 'battery_below', 'percent': 20}]})
    rep = v.validate(p)
    assert rep.ok and rep.plan.steps[0].success.type == 'tool_ok'


def test_mission_clock_starts_at_execution(tmp_path):
    """Slow planning must not eat the mission-time budget (sim eval C2)."""
    import copy
    from auv_mission_planner.config import load_llm_config
    from auv_mission_planner.llm.providers import MockProvider
    from auv_mission_planner.mission import MissionRunner
    cfg = copy.deepcopy(load_config())
    cfg['limits']['max_mission_time_s'] = 300
    b = FakeBackend()
    plan = {'status': 'plan', 'plan': {'abort_conditions': [{'type': 'battery_below', 'percent': 20}], 'steps': [
        {'id': 1, 'tool': 'hold_depth', 'params': {'depth': 1.0, 'duration_s': 2}, 'timeout_s': 40},
        {'id': 2, 'tool': 'surface', 'timeout_s': 60}]}}

    class SlowMock(MockProvider):
        def complete(self, messages, json_schema=None):
            b.t += 280.0                      # the LLM "thinks" for 280 s of vehicle time
            return super().complete(messages, json_schema)
    r = MissionRunner(cfg, load_llm_config(), SlowMock(responses=[plan]), ToolRegistry(b, cfg),
                      trace_dir=str(tmp_path))
    res = r.run('dive then surface')
    assert res['outcome'] == 'succeeded', res['reason']
