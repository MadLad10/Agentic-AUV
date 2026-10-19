#!/usr/bin/env python3
"""Localisation study: scripted gate -> red buoy -> surface mission (no LLM), several seeds per
navigation condition; EKF error is measured against simulator truth by the referee.

  python3 scripts/nav_study.py --seeds 5 --out eval/results/nav_study
"""
import argparse
import csv
import json
import os
import statistics
import sys

CONDITIONS = {
    'nominal': [],
    'dvl_dropout': [{'type': 'dvl_dropout', 't_start': 10.0}],
    'imu_bias': [{'type': 'imu_bias', 't_start': 10.0, 'params': {'gyro_bias_z': 0.02}}],
    'compass_disturbance': [{'type': 'compass_disturbance', 't_start': 10.0, 'params': {'offset_deg': 30}}],
    'dvl_dropout_no_landmarks': [{'type': 'dvl_dropout', 't_start': 10.0}],
}
STEPS = [('navigate_to', {'x': 2.0, 'y': 0.0, 'depth': 1.5}, 60), ('navigate_to', {'target': 'gate'}, 120),
         ('search_pattern', {'target': 'red_buoy'}, 120), ('navigate_to', {'target': 'red_buoy'}, 120),
         ('surface', {}, 90)]


def run(cond, seed):
    from auv_onboard.factory import make_backend, make_registry
    ob = {'backend': {'max_rtf': None, 'idle_rtf': None}}
    if cond.endswith('no_landmarks'):
        ob['ekf'] = {'landmarks': {'gate_prior': False}}
    be = make_backend('default', faults=CONDITIONS[cond], seed=seed, onboard_overrides=ob)
    reg, _ = make_registry(be)
    ok = True
    try:
        for name, p, to in STEPS:
            o = reg.call(name, p, to)
            if not o.ok:
                ok = False
                break
        r = be.sim.referee.report()
    finally:
        be.shutdown()
    L = r['localisation']
    return {'condition': cond, 'seed': seed, 'mission_ok': ok, 'gate_passed': r['gate_passed'],
            'buoy_touched': 'red_buoy' in r['buoys_touched'], 'collisions': r['collisions'],
            'rmse_xy_m': L['rmse_xy_m'], 'max_xy_m': L['max_xy_m'], 'final_xy_m': L['final_xy_m'],
            'rmse_depth_m': L['rmse_depth_m']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--out', default='eval/results/nav_study')
    ap.add_argument('--summary-only', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.summary_only:
        rows = json.load(open(os.path.join(a.out, 'runs.json')))
    else:
        rows = [run(c, s) for c in CONDITIONS for s in range(a.seeds)]
    with open(os.path.join(a.out, 'runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ['# Localisation study (EKF vs ground truth)', '',
             f'Scripted gate -> red buoy -> surface, no LLM; {a.seeds} seeds per condition; faults injected at t = 10 s.', '',
             '"tools ok" = every tool call reported success (what the vehicle believes); "referee: task" = gate '
             'passed AND red buoy touched according to simulator ground truth.', '',
             '| Condition | tools ok | referee: task | RMSE xy (m) mean ± sd | max xy (m) mean | worst max xy (m) | RMSE depth (m) |',
             '|---|---|---|---|---|---|---|']
    for c in CONDITIONS:
        rs = [r for r in rows if r['condition'] == c]
        rm = [r['rmse_xy_m'] for r in rs]
        mx = [r['max_xy_m'] for r in rs]
        task = sum(bool(r['gate_passed']) and bool(r['buoy_touched']) for r in rs)
        lines.append(f'| {c} | {sum(r["mission_ok"] for r in rs)}/{len(rs)} | {task}/{len(rs)} | {statistics.mean(rm):.3f} ± '
                     f'{statistics.pstdev(rm):.3f} | {statistics.mean(mx):.3f} | {max(mx):.3f} | '
                     f'{statistics.mean(r["rmse_depth_m"] for r in rs):.3f} |')
    with open(os.path.join(a.out, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    json.dump(rows, open(os.path.join(a.out, 'runs.json'), 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
