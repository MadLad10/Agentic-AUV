"""Tools layer, tested directly against the fake backend (no LLM, no ROS)."""
import copy

import pytest

from auv_mission_planner.config import load_config
from auv_mission_planner.tools.backend import Det, WorldModel
from auv_mission_planner.tools.fake_backend import DEFAULT_OBJECTS, FakeBackend
from auv_mission_planner.tools.registry import ToolRegistry
from auv_mission_planner.tools.schemas import TOOL_PARAMS, Observation


@pytest.fixture
def cfg():
    return load_config()


def make(cfg, **kw):
    b = FakeBackend(**kw)
    return b, ToolRegistry(b, cfg)


def test_registry_exposes_exactly_the_allowed_tools(cfg):
    _, t = make(cfg)
    assert set(t.names) == set(cfg['limits']['allowed_tools']) == set(TOOL_PARAMS)
    for s in t.schemas():
        assert s['description'] and isinstance(s['parameters'], dict)


def test_allowed_tools_config_filters_registry(cfg):
    c = copy.deepcopy(cfg)
    c['limits']['allowed_tools'] = ['get_pose', 'surface']
    _, t = make(c)
    assert t.names == ['get_pose', 'surface']
    obs = t.call('navigate_to', {'x': 1, 'y': 0, 'depth': 1})
    assert not obs.ok and obs.error_code == 'unknown_tool'


def test_unknown_tool_and_bad_params_are_observations_not_exceptions(cfg):
    _, t = make(cfg)
    o = t.call('set_thrusters', {'all': 100})
    assert isinstance(o, Observation) and o.error_code == 'unknown_tool'
    o = t.call('navigate_to', {'target': 'gate', 'x': 3})
    assert o.error_code == 'invalid_params'
    o = t.call('hold_depth', {'depth': 1.0, 'bogus': 1})
    assert o.error_code == 'invalid_params'


def test_scripted_mission_gate_buoy_surface(cfg):
    b, t = make(cfg)
    assert t.call('hold_depth', {'depth': 1.5, 'duration_s': 2}).ok
    det = t.call('get_detections')
    assert any(d['class'] == 'gate' for d in det.data['visible'])
    o = t.call('navigate_to', {'target': 'gate', 'pass_through': True}, timeout_s=120)
    assert o.ok, o.brief()
    assert b.gate_passed
    o = t.call('navigate_to', {'target': 'red_buoy'}, timeout_s=120)
    assert o.ok, o.brief()
    assert 'red_buoy' in b.touched
    assert t.call('surface').ok and b.surfaced
    assert t.call('get_pose').data['depth'] == 0.0


def test_object_target_requires_prior_detection(cfg):
    hidden = [dict(o) for o in DEFAULT_OBJECTS]
    hidden[1].update(x=9.0, y=7.0)                     # red buoy off to the side, out of view
    b, t = make(cfg, objects=hidden)
    o = t.call('navigate_to', {'target': 'red_buoy'})
    assert not o.ok and o.error_code == 'target_not_detected'
    assert 'search' in o.error
    s = t.call('search_pattern', {'target': 'red_buoy', 'pattern': 'spiral', 'radius_m': 9}, timeout_s=600)
    assert s.ok and s.data['found'], s.brief()
    assert t.call('navigate_to', {'target': 'red_buoy'}, timeout_s=120).ok


def test_unknown_object_is_reported(cfg):
    _, t = make(cfg)
    o = t.call('navigate_to', {'target': 'purple_buoy'})
    assert not o.ok and o.error_code == 'target_not_detected'
    assert 'gate' in o.data['known_objects']


def test_runtime_limits_block_unsafe_points(cfg):
    b, t = make(cfg)
    for params in ({'x': 100, 'y': 0, 'depth': 1}, {'x': 2, 'y': 0, 'depth': 30}, {'x': 2, 'y': 0, 'depth': -1}):
        o = t.call('navigate_to', params)
        assert not o.ok and o.error_code == 'unsafe_target', params
    assert not t.call('hold_depth', {'depth': 9.0}).ok
    assert not t.call('search_pattern', {'target': 'gate', 'depth': 9.0}).ok
    assert not any(c[0] == 'navigate' for c in b.calls)   # never reached the vehicle


def test_speed_is_capped_by_limits(cfg):
    b, t = make(cfg)
    seen = {}
    orig = b.navigate

    def spy(x, y, depth, yaw=None, tolerance=0.35, speed=None, timeout_s=60.0):
        seen['speed'] = speed
        return orig(x, y, depth, yaw, tolerance, speed, timeout_s)
    b.navigate = spy
    t.call('navigate_to', {'x': 2, 'y': 0, 'depth': 1, 'speed': 5.0})
    assert seen['speed'] == cfg['limits']['max_speed_mps']


def test_timeout_and_backend_failure_become_observations(cfg):
    b, t = make(cfg, fail_next={'navigate': 1})
    o = t.call('navigate_to', {'x': 2, 'y': 0, 'depth': 1})
    assert not o.ok and o.error_code == 'action_failed'
    o = t.call('navigate_to', {'x': 20, 'y': 0, 'depth': 1}, timeout_s=5)
    assert not o.ok and o.error_code == 'timeout'

    def boom(*a, **k):
        raise RuntimeError('link lost')
    b.surface = boom
    o = t.call('surface')
    assert not o.ok and o.error_code == 'backend_unavailable' and 'link lost' in o.error


def test_abort_cancels_and_surfaces(cfg):
    b, t = make(cfg)
    t.call('hold_depth', {'depth': 2.0, 'duration_s': 1})
    o = t.call('abort', {'reason': 'test'})
    assert o.ok and b.emergency and b.surfaced
    assert ('cancel_all',) in b.calls


def test_battery_and_pose(cfg):
    _, t = make(cfg, battery_start=42.0)
    assert t.call('get_battery').data['percent'] == 42.0
    assert set(t.call('get_pose').data) == {'x', 'y', 'depth', 'yaw_deg'}


def test_world_model_prefers_close_sightings_and_forgets():
    w = WorldModel(memory_s=10, min_confidence=0.4)
    w.update([Det('gate', 0.9, 8.6, 0.4, 1.5, 9.0, 0, t=0.0),
              Det('gate', 0.9, 8.0, 0.0, 1.5, 1.5, 0, t=1.0),
              Det('gate', 0.2, 50, 50, 9, 1.0, 0, t=1.0)])        # low confidence ignored
    est = w.lookup('gate', now=2.0)
    assert 8.0 <= est.x < 8.2
    near = w.lookup('gate', now=2.0, near_range=3.0)
    assert near.x == pytest.approx(8.0)
    assert w.lookup('gate', now=100.0) is None


def test_relative_move_resolved_from_pose_and_limited(cfg):
    import math
    b, t = make(cfg, start=(2.0, 1.0, 0.5, math.pi / 2))      # facing +y
    o = t.call('navigate_to', {'forward_m': 3, 'left_m': 1, 'depth': 1.0})
    assert o.ok, o.brief()
    assert o.data['target'] == {'x': 1.0, 'y': 4.0, 'depth': 1.0}     # forward = +y, left = -x
    p = t.call('get_pose').data
    assert (p['x'], p['y'], p['depth']) == (1.0, 4.0, 1.0)
    o = t.call('navigate_to', {'forward_m': 50})                     # leaves the geofence -> blocked at runtime
    assert not o.ok and o.error_code == 'unsafe_target'
