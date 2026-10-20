"""Safety validator: one test per rule, both reject and repair paths."""
import copy

import pytest

from auv_mission_planner.config import load_config
from auv_mission_planner.plan import MissionPlan
from auv_mission_planner.prompts import EXAMPLE_PLAN
from auv_mission_planner.validator import SafetyValidator


@pytest.fixture
def L():
    return load_config()['limits']


@pytest.fixture
def v(L):
    return SafetyValidator(L)


def plan(steps, aborts=None):
    return MissionPlan.model_validate({'summary': 't', 'steps': steps,
                                       'abort_conditions': aborts if aborts is not None else
                                       [{'type': 'battery_below', 'percent': 20}]})


def step(i, tool, params=None, **kw):
    return {'id': i, 'tool': tool, 'params': params or {}, **kw}


def rules(rep, action='reject'):
    return {f.rule for f in (rep.rejected if action == 'reject' else rep.repaired)}


SURF = step(9, 'surface', timeout_s=60)


def test_example_plan_is_clean(v):
    rep = v.validate(MissionPlan.model_validate(EXAMPLE_PLAN['plan']))
    assert rep.ok and not rep.rejected and not rep.repaired


def test_empty_plan(v):
    assert rules(v.validate(None)) == {'empty_plan'}
    assert rules(v.validate(MissionPlan(steps=[]))) == {'empty_plan'}


@pytest.mark.parametrize('tool', ['set_thrusters', 'publish_raw', 'disable_safety', 'cmd_vel'])
def test_tool_not_allowed(v, tool):
    rep = v.validate(plan([step(1, tool, {'value': 100}), SURF]))
    assert not rep.ok and rules(rep) == {'tool_not_allowed'}


@pytest.mark.parametrize('tool,params', [('hold_depth', {'depth': 30}), ('hold_depth', {'depth': -2}),
                                         ('navigate_to', {'x': 2, 'y': 0, 'depth': 4.5}),
                                         ('search_pattern', {'target': 'gate', 'depth': 12})])
def test_depth_limits(v, tool, params):
    rep = v.validate(plan([step(1, tool, params), SURF]))
    assert not rep.ok and 'depth_limit' in rules(rep)


def test_depth_in_success_condition(v):
    rep = v.validate(plan([step(1, 'hold_depth', {'depth': 1.0},
                                success={'type': 'depth_within', 'depth': 20}), SURF]))
    assert 'depth_limit' in rules(rep)


def test_geofence(v):
    rep = v.validate(plan([step(1, 'navigate_to', {'x': 100, 'y': 0, 'depth': 1}), SURF]))
    assert rules(rep) == {'geofence'}


def test_unknown_objects_are_impossible(v):
    rep = v.validate(plan([step(1, 'navigate_to', {'target': 'purple_buoy'}), SURF]))
    assert rules(rep) == {'unknown_object'}
    rep = v.validate(plan([step(1, 'search_pattern', {'target': 'shark'}), SURF]))
    assert rules(rep) == {'unknown_object'}


def test_invalid_params(v):
    rep = v.validate(plan([step(1, 'navigate_to', {'target': 'gate', 'x': 1}), SURF]))
    assert rules(rep) == {'invalid_params'}
    rep = v.validate(plan([step(1, 'hold_depth', {'depth': 1, 'thrust': 100}), SURF]))
    assert rules(rep) == {'invalid_params'}


def test_mission_time_budget_scaled_when_possible(v, L):
    # 5 x 240 s navigate = 1200 s > 900 s; scaling to 180 s each keeps the 60 s minimum -> repair
    steps = [step(i, 'navigate_to', {'target': 'gate'}, timeout_s=240) for i in range(1, 5)] + [SURF]
    rep = v.validate(plan(steps))
    assert rep.ok and 'mission_time' in rules(rep, 'repair')
    assert sum(s.timeout_s for s in rep.plan.steps) <= L['max_mission_time_s']
    assert all(s.timeout_s >= 60 for s in rep.plan.steps if s.tool == 'navigate_to')


def test_mission_time_budget_rejected_when_too_many_steps(v, L):
    # 16 navigate steps cannot fit 900 s with >= 60 s each -> reject
    steps = [step(i, 'navigate_to', {'target': 'gate'}, timeout_s=120) for i in range(1, 17)] + [SURF]
    rep = v.validate(plan(steps))
    assert not rep.ok and 'mission_time' in rules(rep)


def test_pass_through_on_buoy_is_repaired(v):
    rep = v.validate(plan([step(1, 'navigate_to', {'target': 'yellow_buoy', 'pass_through': True}), SURF]))
    assert rep.ok and 'pass_through' in rules(rep, 'repair')
    assert 'pass_through' not in rep.plan.steps[0].params


def test_relative_moves(v):
    rep = v.validate(plan([step(1, 'navigate_to', {'forward_m': 3, 'depth': 1.0}), SURF]))
    assert rep.ok, rep.reasons()
    rep = v.validate(plan([step(1, 'navigate_to', {'forward_m': 3, 'depth': 9.0}), SURF]))
    assert 'depth_limit' in rules(rep)
    rep = v.validate(plan([step(1, 'navigate_to', {'forward_m': 3, 'target': 'gate'}), SURF]))
    assert 'invalid_params' in rules(rep)


def test_too_many_steps(v, L):
    steps = [step(i, 'get_pose', timeout_s=1) for i in range(1, L['max_steps'] + 2)] + [SURF]
    assert 'too_many_steps' in rules(v.validate(plan(steps)))


def test_steps_after_abort(v):
    rep = v.validate(plan([step(1, 'abort'), step(2, 'navigate_to', {'target': 'gate'})]))
    assert 'steps_after_abort' in rules(rep)


# ------------------------------------------------------------ repairs


def test_appends_surface(v):
    rep = v.validate(plan([step(1, 'hold_depth', {'depth': 1.0, 'duration_s': 3}, timeout_s=40)]))
    assert rep.ok and rules(rep, 'repair') == {'must_surface'}
    assert rep.plan.steps[-1].tool == 'surface' and rep.plan.steps[-1].id == 2


def test_abort_as_last_step_needs_no_surface(v):
    rep = v.validate(plan([step(1, 'get_pose'), step(2, 'abort')]))
    assert rep.ok and 'must_surface' not in rules(rep, 'repair')


def test_battery_abort_inserted_and_raised(v, L):
    rep = v.validate(plan([SURF], aborts=[]))
    assert rep.ok and 'battery_abort' in rules(rep, 'repair')
    assert rep.plan.abort_conditions[0].percent == L['default_battery_abort_pct']
    rep = v.validate(plan([SURF], aborts=[{'type': 'battery_below', 'percent': 5}]))
    assert rep.ok and rep.plan.abort_conditions[0].percent == L['min_battery_abort_pct']
    rep = v.validate(plan([SURF], aborts=[{'type': 'battery_below', 'percent': 150}]))
    assert not rep.ok and 'battery_abort' in rules(rep)


def test_user_threshold_above_minimum_is_kept(v):
    rep = v.validate(plan([SURF], aborts=[{'type': 'battery_below', 'percent': 35}]))
    assert rep.ok and rep.plan.abort_conditions[0].percent == 35 and not rep.repaired


def test_clamps(v, L):
    rep = v.validate(plan([
        step(1, 'hold_depth', {'depth': 1.0, 'duration_s': 999}, timeout_s=9999),
        step(2, 'search_pattern', {'target': 'red_buoy', 'radius_m': 50}, timeout_s=0),
        step(3, 'navigate_to', {'target': 'red_buoy', 'speed': 3.0}), SURF]))
    assert rep.ok, rep.reasons()
    assert {'hold_duration', 'step_timeout', 'search_radius', 'speed_limit'} <= rules(rep, 'repair')
    s = rep.plan.steps
    assert s[0].params['duration_s'] == L['max_hold_duration_s'] and s[0].timeout_s <= L['max_step_timeout_s']
    assert s[1].params['radius_m'] == L['max_search_radius_m'] and s[1].timeout_s == 60
    assert s[2].params['speed'] == L['max_speed_mps']


def test_short_hold_timeout_is_extended(v):
    rep = v.validate(plan([step(1, 'hold_depth', {'depth': 1.0, 'duration_s': 30}, timeout_s=20), SURF]))
    assert rep.ok and rep.plan.steps[0].timeout_s >= 50


def test_renumbers_ids(v):
    rep = v.validate(plan([step(3, 'get_pose'), step(3, 'surface')]))
    assert rep.ok and [s.id for s in rep.plan.steps] == [1, 2]


def test_repair_disabled_turns_repairs_into_rejections(L):
    rep = SafetyValidator(L, repair=False).validate(plan([step(1, 'get_pose')], aborts=[]))
    assert not rep.ok and {'must_surface', 'battery_abort'} <= rules(rep)


def test_input_not_mutated_and_repair_is_idempotent(v):
    p = plan([step(1, 'hold_depth', {'depth': 1.0, 'duration_s': 999}, timeout_s=9999)], aborts=[])
    before = copy.deepcopy(p.model_dump())
    rep = v.validate(p)
    assert p.model_dump() == before
    again = v.validate(rep.plan)
    assert again.ok and not again.repaired


def test_report_is_serialisable_and_logged(v):
    import logging
    lines = []
    h = logging.Handler()
    h.emit = lambda r: lines.append(r.getMessage())
    lg = logging.getLogger('auv.validator')
    lg.addHandler(h)
    old = lg.level
    lg.setLevel(logging.INFO)
    try:
        rep = v.validate(plan([step(1, 'set_thrusters')]))
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    d = rep.as_dict()
    assert d['ok'] is False and d['rejected'][0]['rule'] == 'tool_not_allowed'
    assert any(x.startswith('REJECT') for x in lines) and any(x.startswith('REPAIR') for x in lines)


# ------------------------------------------------------------ runtime precheck


def test_precheck(v, L):
    p = plan([step(1, 'navigate_to', {'target': 'gate'}, timeout_s=120), SURF],
             aborts=[{'type': 'battery_below', 'percent': 30}])
    nav, surf = p.steps
    assert v.precheck_step(nav, 80, 0, p) is None
    assert 'abort threshold 30%' in v.precheck_step(nav, 25, 0, p)
    assert v.precheck_step(surf, 5, 0, p) is None                      # surfacing is always allowed
    assert 'unknown' in v.precheck_step(nav, None, 0, p)
    assert 'mission time' in v.precheck_step(nav, 80, L['max_mission_time_s'], p)
    p2 = plan([step(1, 'hold_depth', {'depth': 1}), SURF], aborts=[{'type': 'battery_below', 'percent': 15}])
    assert 'needed to start' in v.precheck_step(p2.steps[0], L['min_battery_to_start_pct'] - 1, 0, p2)


def test_objective_step_cannot_be_optional():
    """v2.1 eval R3: 'touch the green buoy' with on_failure=continue ended as 'succeeded', nothing touched."""
    from auv_mission_planner.config import load_config
    from auv_mission_planner.plan import MissionPlan
    from auv_mission_planner.validator import SafetyValidator
    v = SafetyValidator(load_config()['limits'])
    plan = MissionPlan.model_validate({'abort_conditions': [{'type': 'battery_below', 'percent': 20}], 'steps': [
        {'id': 1, 'tool': 'navigate_to', 'params': {'target': 'gate'}, 'on_failure': 'continue'},
        {'id': 2, 'tool': 'navigate_to', 'params': {'target': 'green_buoy'}, 'on_failure': 'continue'},
        {'id': 3, 'tool': 'surface', 'success': {'type': 'surfaced'}}]})
    rep = v.validate(plan, prompt='touch the green buoy, then surface')
    assert rep.ok
    assert rep.plan.steps[1].on_failure == 'replan'                 # the objective
    assert rep.plan.steps[0].on_failure == 'continue'               # gate not asked for: just a waypoint
    assert any(f.rule == 'objective_not_optional' for f in rep.repaired)
    rep = v.validate(plan, prompt='pass through the gate and touch the green buoy')
    assert rep.plan.steps[0].on_failure == 'replan'                 # now the gate is part of the objective
    rep = v.validate(plan, prompt='touch the green buoy if you can, then surface')
    assert rep.plan.steps[1].on_failure == 'continue'               # the operator made it optional


def test_mission_time_budget_keeps_minimums_when_uniform_scaling_would_not(v, L):
    """v2.1 eval C3: gate + 3 x (search, touch) + surface. Minimums fit 900 s, uniform scaling did not."""
    steps = [step(1, 'navigate_to', {'target': 'gate'}, timeout_s=120)]
    for i, b in enumerate(['red_buoy', 'green_buoy', 'yellow_buoy']):
        steps.append(step(2 + 2 * i, 'search_pattern', {'target': b}, timeout_s=200))
        steps.append(step(3 + 2 * i, 'navigate_to', {'target': b}, timeout_s=120))
    steps.append(SURF)
    rep = v.validate(plan(steps))
    assert rep.ok, rep.reasons()
    assert sum(s.timeout_s for s in rep.plan.steps) <= L['max_mission_time_s']
    assert all(s.timeout_s >= 180 for s in rep.plan.steps if s.tool == 'search_pattern')
    assert all(s.timeout_s >= 60 for s in rep.plan.steps if s.tool == 'navigate_to')


def test_unreachable_battery_check_is_repaired_to_the_floor(v, L):
    """Final eval E3: "check the battery level and then surface" got success battery_above 100%."""
    chk = step(1, 'get_battery', success={'type': 'battery_above', 'percent': 100.0}, on_failure='continue', timeout_s=60)
    rep = v.validate(plan([chk, SURF]), prompt='check the battery level and then surface')
    assert rep.ok and 'unreachable_success' in rules(rep, 'repair')
    assert rep.plan.steps[0].success.percent == L['min_battery_abort_pct']
    ok = step(1, 'get_battery', success={'type': 'battery_above', 'percent': 50.0}, timeout_s=60)
    rep = v.validate(plan([ok, SURF]), prompt='check the battery is above 50% and then surface')
    assert 'unreachable_success' not in rules(rep, 'repair') and rep.plan.steps[0].success.percent == 50.0
