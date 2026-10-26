"""Scoring of the v1-vs-v2 harness: v1 rules on the referee report + v2 fault-tier metrics."""
import importlib.util
import os

import pytest

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
    assert s['safety_status'] == 'unsafe'


@pytest.mark.parametrize('ref', [None, {}, {'surfaced_at_end': True},
    {'surfaced_at_end': True, 'max_depth_m': float('nan'), 'boundary_contacts': 0, 'collisions': 0}])
def test_missing_or_invalid_referee_is_unknown_safety(ref):
    m = next(x for x in ev.load_missions(['faults']) if x['id'] == 'F3')
    scored = ev.score({'outcome': 'error', 'referee': ref}, m, LIMITS, V1)
    assert not scored['pass'] and not scored['safe']
    assert scored['safety_status'] == 'unknown'


def test_mission_sets_load():
    ms = ev.load_missions(['nominal', 'faults'])
    assert len([m for m in ms if m['set'] == 'nominal']) == 25 and len([m for m in ms if m['set'] == 'faults']) == 11


def test_ablation_overrides_hit_real_config_keys():
    """A typo in an override key would silently leave the component on; every key must exist already."""
    from auv_onboard.factory import load_onboard_config
    base = load_onboard_config()

    def keys(d, pre=()):
        for k, v in d.items():
            yield from keys(v, pre + (k,)) if isinstance(v, dict) else [pre + (k,)]

    for abl in ('no_health', 'no_fusion'):
        over = ev.ablation_overrides(abl)
        assert over
        for path in keys(over):
            node = base
            for k in path:
                assert k in node, (abl, path)
                node = node[k]
        merged = ev.deep_merge(base, over)
        assert merged['backend']['nav_source'] == base['backend']['nav_source']        # siblings kept
    m = ev.deep_merge(base, ev.ablation_overrides('no_fusion'))
    assert not m['ekf']['sonar_walls']['enabled'] and not m['ekf']['learned_velocity']['enabled']
    assert not m['tracking']['enabled'] and m['ekf']['sonar_walls'].keys() == base['ekf']['sonar_walls'].keys()
    assert ev.ablation_overrides('none') == {} and ev.ablation_overrides('no_validator') == {}


def test_pass_through_validator_accepts_what_the_real_one_rejects():
    from auv_mission_planner.config import load_config
    from auv_mission_planner.plan import MissionPlan
    from auv_mission_planner.validator import SafetyValidator
    deep = MissionPlan.model_validate({'summary': 't', 'abort_conditions': [], 'steps': [
        {'id': 1, 'tool': 'hold_depth', 'params': {'depth': 25.0, 'duration_s': 5}, 'timeout_s': 60}]})
    assert not SafetyValidator(load_config()['limits']).validate(deep).ok
    rep = ev.PassThroughValidator().validate(deep)
    assert rep.ok and rep.plan is deep and ev.PassThroughValidator().precheck_step(None, 1.0, 9e9, deep) is None


def test_campaign_summary_pass_hat_k_and_wilson():
    cs = _load('campaign_summary', os.path.join(WS, 'eval', 'campaign_summary.py'))
    lo, hi = cs.wilson(10, 10)
    assert hi == 1.0 and 0.70 < lo < 0.73
    rows = [{'system': 'v2', 'ablation': 'none', 'mission': 'A', 'seed': s, 'tier': 'easy', 'pass': True, 'safe': True}
            for s in range(3)]
    rows += [{'system': 'v2', 'ablation': 'none', 'mission': 'B', 'seed': s, 'tier': 'easy', 'pass': s != 1, 'safe': True}
             for s in range(3)]
    rows += [{'system': 'v2', 'ablation': 'no_health', 'mission': 'B', 'seed': 0, 'tier': 'easy', 'pass': False,
              'safe': False}]
    assert cs.pass_hat_k([r for r in rows if r['ablation'] == 'none']) == (1, 2, 3)
    text = cs.summarise(rows)
    assert '| full | 6 | 5/6' in text and '| no_health | 1 | pass: ' not in text
    assert '| no_health | 1 | 1/1 → 0/1 | 1/1 → 0/1 |' in text and '| B | 2/3 | 1: ' in text
