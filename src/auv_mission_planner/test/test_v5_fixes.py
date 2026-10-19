"""v5: the validator's intent check stops silent target substitution (sim eval v4, I1)."""
import copy

import pytest

from auv_mission_planner.config import load_config, load_llm_config
from auv_mission_planner.llm.providers import MockProvider
from auv_mission_planner.mission import MissionRunner
from auv_mission_planner.plan import MissionPlan
from auv_mission_planner.tools.fake_backend import FakeBackend
from auv_mission_planner.tools.registry import ToolRegistry
from auv_mission_planner.validator import SafetyValidator, mentioned_objects

KNOWN = ['gate', 'red_buoy', 'green_buoy', 'yellow_buoy']


@pytest.mark.parametrize('prompt,expected', [
    ('touch the red buoy then surface', {'gate', 'red_buoy'}),
    ('touch the green buoy first, then the red buoy', {'gate', 'green_buoy', 'red_buoy'}),
    ('go through the gate, touch all three buoys, then surface', set(KNOWN)),
    ('find the yellow buoy, touch it, then surface', {'gate', 'yellow_buoy'}),
    ('touch the purple buoy then surface', {'gate'}),
    ('touch the red_buoy', {'gate', 'red_buoy'}),
    ('dive to 2 m and surface', {'gate'}),
])
def test_mentioned_objects(prompt, expected):
    assert mentioned_objects(prompt, KNOWN) == expected


def plan(*targets):
    return MissionPlan.model_validate({'abort_conditions': [{'type': 'battery_below', 'percent': 20}], 'steps': [
        *[{'id': i + 1, 'tool': 'navigate_to', 'params': {'target': t}, 'timeout_s': 120} for i, t in enumerate(targets)],
        {'id': 9, 'tool': 'surface', 'timeout_s': 60}]})


def test_substituted_target_is_rejected():
    v = SafetyValidator(load_config()['limits'])
    rep = v.validate(plan('green_buoy'), prompt='touch the purple buoy then surface')
    assert not rep.ok and any(f.rule == 'off_mission_target' for f in rep.rejected)
    assert v.validate(plan('gate', 'red_buoy'), prompt='touch the red buoy').ok
    assert v.validate(plan('red_buoy'), prompt=None).ok          # no prompt -> no intent check


def test_substitution_never_executes(tmp_path):
    """The I1 failure end to end: the model swaps purple for green; nothing may move."""
    cfg = copy.deepcopy(load_config())
    swap = {'status': 'plan', 'plan': {'abort_conditions': [], 'steps': [
        {'id': 1, 'tool': 'navigate_to', 'params': {'target': 'green_buoy'}, 'timeout_s': 120},
        {'id': 2, 'tool': 'surface', 'timeout_s': 60}]}}
    purple = {'status': 'plan', 'plan': {'steps': [
        {'id': 1, 'tool': 'navigate_to', 'params': {'target': 'purple_buoy'}, 'timeout_s': 120}]}}
    b = FakeBackend()
    r = MissionRunner(cfg, load_llm_config(), MockProvider(responses=[purple, swap, swap]),
                      ToolRegistry(b, cfg), trace_dir=str(tmp_path))
    res = r.run('touch the purple buoy then surface')
    assert res['outcome'] == 'rejected' and res['unsafe_plans_caught'] == 3
    assert not any(c[0] == 'navigate' for c in b.calls)


def test_requested_depths():
    from auv_mission_planner.validator import requested_depths
    assert requested_depths('dive to 50 meters and take a look around') == [50.0]
    assert requested_depths('go through the gate at 2 meters depth, then surface') == [2.0]
    assert requested_depths('move 3 meters forward at 1 meter depth, then surface') == [1.0]
    assert requested_depths('touch the red buoy') == []


def test_operator_depth_beyond_limit_rejects_any_plan():
    v = SafetyValidator(load_config()['limits'])
    shallow = MissionPlan.model_validate({'abort_conditions': [{'type': 'battery_below', 'percent': 20}], 'steps': [
        {'id': 1, 'tool': 'hold_depth', 'params': {'depth': 4.0, 'duration_s': 5}, 'timeout_s': 40},
        {'id': 2, 'tool': 'surface', 'timeout_s': 60}]})
    rep = v.validate(shallow, prompt='dive to 50 meters and take a look around')
    assert not rep.ok and 'never run a shallower substitute' in rep.reasons()[0]
    assert v.validate(shallow, prompt='dive to 4 meters then surface').ok
