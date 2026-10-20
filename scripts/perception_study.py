#!/usr/bin/env python3
"""A/B study: camera-sonar fusion tracking ON vs OFF, inside the full v2 agent graph.

The LLM is replaced by replayed plans (identical for both arms), so the only difference is the
perception stack. Scored by the ground-truth referee over several seeds.

  python3 scripts/perception_study.py --seeds 5 --out eval/results/perception_study
"""
import argparse
import json
import os
import statistics
import sys
import time


def P(steps):
    return {'status': 'plan', 'reason': '', 'plan': {'summary': 'replayed', 'abort_conditions': [
        {'type': 'battery_below', 'percent': 20}], 'steps': [dict(id=i + 1, **s) for i, s in enumerate(steps)]}}


DIVE = {'tool': 'hold_depth', 'params': {'depth': 1.5, 'duration_s': 2}, 'timeout_s': 40,
        'success': {'type': 'depth_within', 'depth': 1.5, 'tolerance_m': 0.4}}
GATE = {'tool': 'navigate_to', 'params': {'target': 'gate', 'pass_through': True}, 'timeout_s': 120}
SURF = {'tool': 'surface', 'timeout_s': 90, 'success': {'type': 'surfaced'}, 'on_failure': 'abort'}


def search(c):
    return {'tool': 'search_pattern', 'params': {'target': c, 'pattern': 'spiral', 'radius_m': 8.0}, 'timeout_s': 200,
            'success': {'type': 'detected', 'target': c}}


def touch(c):
    return {'tool': 'navigate_to', 'params': {'target': c, 'standoff_m': 0}, 'timeout_s': 150}


CONDITIONS = {
    'turbidity_touch_red': {
        'prompt': 'touch the red buoy then surface',
        'faults': [{'type': 'turbidity', 't_start': 0.0, 'params': {'visibility_m': 2.5}}],
        'plans': [P([DIVE, search('red_buoy'), touch('red_buoy'), SURF])] + [P([search('red_buoy'), touch('red_buoy'), SURF])] * 2 + [P([SURF])] * 4,
        'task': lambda r: r['touch_order'][:1] == ['red_buoy']},
    'false_target_gate': {
        'prompt': 'pass through the gate then surface',
        'faults': [{'type': 'sonar_false_target', 't_start': 0.0, 'params': {'x': 4.5, 'y': 0.6, 'z': -1.5}}],
        'plans': [P([DIVE, GATE, SURF])] + [P([GATE, SURF])] * 2 + [P([SURF])] * 4,
        'task': lambda r: r['gate_passed']},
    'nominal_gate_red_green': {
        'prompt': 'pass through the gate, touch the red buoy, then the green buoy, then surface',
        'faults': [],
        'plans': [P([DIVE, GATE, search('red_buoy'), touch('red_buoy'), search('green_buoy'), touch('green_buoy'), SURF])]
                 + [P([search('green_buoy'), touch('green_buoy'), SURF])] * 2 + [P([SURF])] * 4,
        'task': lambda r: r['gate_passed'] and r['touch_order'][:2] == ['red_buoy', 'green_buoy']},
}


def run(cond, tracking, seed):
    from auv_agents.run import build_runtime, load_agents_config, run_mission
    from auv_mission_planner.llm.providers import MockProvider
    from auv_onboard.factory import make_backend
    c = CONDITIONS[cond]
    be = make_backend('default', faults=c['faults'], seed=seed,
                      onboard_overrides={'backend': {'max_rtf': None, 'idle_rtf': None},
                                         'tracking': {'enabled': tracking}})
    try:
        rt = build_runtime(be, MockProvider(responses=json.loads(json.dumps(c['plans']))),
                           load_agents_config({'diagnostics': {'use_llm': False}}), None, None)
        t0 = time.time()
        res = run_mission(rt, c['prompt'])
        t_end = time.time() + 120
        while time.time() < t_end and any(t.is_alive() for t in be._threads):
            time.sleep(0.05)
        r = be.sim.referee.report()
        faults = sorted({f['type'] for rep in be.health_log for f in rep.faults})
        wrong = [b for b in r['touch_order'] if b not in ('red_buoy', 'green_buoy')] if cond != 'false_target_gate' else []
        return {'condition': cond, 'tracking': tracking, 'seed': seed, 'outcome': res.get('outcome'),
                'task': bool(c['task'](r)), 'wrong_touch': wrong, 'surfaced': r['surfaced_at_end'],
                'collisions': r['collisions'], 'sim_s': r['sim_time_s'], 'wall_s': round(time.time() - t0, 1),
                'replans': res.get('replans', 0), 'inspections': (res.get('stats') or {}).get('inspections', 0),
                'false_target_flagged': 'sonar_false_target' in faults, 'health_faults': faults}
    finally:
        be.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--out', default='eval/results/perception_study')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    conds = [c for c in CONDITIONS if not a.only or c in a.only.split(',')]
    rows = []
    for cond in conds:
        for tracking in (False, True):
            for s in range(a.seeds):
                r = run(cond, tracking, s)
                rows.append(r)
                print(json.dumps(r), flush=True)
    json.dump(rows, open(os.path.join(a.out, 'runs.json'), 'w'), indent=1)
    lines = ['# Perception A/B: camera-sonar fusion tracking OFF vs ON (v2 agent graph, replayed plans)', '',
             f'{a.seeds} seeds per arm; referee-scored. "task" = mission objective met per ground truth.', '',
             '| Condition | Tracking | task | surfaced | runs with collisions | wrong-buoy touches | false target flagged | inspections (mean) | sim s (mean) |',
             '|---|---|---|---|---|---|---|---|---|']
    for cond in conds:
        for tracking in (False, True):
            rs = [r for r in rows if r['condition'] == cond and r['tracking'] == tracking]
            lines.append(f'| {cond} | {"ON" if tracking else "off"} | {sum(r["task"] for r in rs)}/{len(rs)} | '
                         f'{sum(r["surfaced"] for r in rs)}/{len(rs)} | {sum(r["collisions"] > 0 for r in rs)}/{len(rs)} | '
                         f'{sum(len(r["wrong_touch"]) for r in rs)} | '
                         f'{sum(r["false_target_flagged"] for r in rs)}/{len(rs)} | '
                         f'{statistics.mean(r["inspections"] for r in rs):.1f} | {statistics.mean(r["sim_s"] for r in rs):.0f} |')
    open(os.path.join(a.out, 'summary.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    sys.exit(main())
