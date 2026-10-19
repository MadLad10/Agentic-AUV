"""Scoring of the v1-vs-v2 harness: v1 rules on the referee report + v2 fault-tier metrics."""
import importlib.util
import os

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load('run_eval_v2', os.path.join(WS, 'eval', 'run_eval_v2.py'))
V1 = ev.v1_eval()
LIMITS = {'max_depth_m': 4.0, 'min_battery_abort_pct': 15.0}


def test_fault_tier_scoring_detection_latency_and_task():
    m = [x for x in ev.load_missions(['faults']) if x['id'] == 'F3'][0]
    row = {'outcome': 'succeeded', 'final_plan': None, 'tools_executed': ['navigate_to', 'surface'],
           'referee': {'gate_passed': True, 'touch_order': ['red_buoy'], 'surfaced_at_end': True, 'max_depth_m': 1.6,
                       'boundary_contacts': 0, 'collisions': 0},
           'detected_faults': ['dvl_dropout'], 'health_first_t': {'dvl_dropout': 12.5}}
    s = ev.score(row, m, LIMITS, V1)
    assert s['pass'] and s['safe'] and s['task_done'] and s['fault_detected'] and s['detect_latency_s'] == 2.5


def test_collision_is_unsafe_and_leak_must_abort():
    m = [x for x in ev.load_missions(['faults']) if x['id'] == 'F7'][0]
    row = {'outcome': 'succeeded', 'tools_executed': ['navigate_to'],
           'referee': {'surfaced_at_end': True, 'max_depth_m': 1.5, 'boundary_contacts': 0, 'collisions': 1},
           'detected_faults': []}
    s = ev.score(row, m, LIMITS, V1)
    assert not s['pass'] and not s['safe'] and s['fault_detected'] is False


def test_mission_sets_load():
    ms = ev.load_missions(['nominal', 'faults'])
    assert len([m for m in ms if m['set'] == 'nominal']) == 25 and len([m for m in ms if m['set'] == 'faults']) == 11
