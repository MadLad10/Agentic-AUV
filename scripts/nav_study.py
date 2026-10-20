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
    'dvl_dropout_current': [{'type': 'dvl_dropout', 't_start': 10.0}],
    # the F10 failure mode: heading without the compass drifts while the vehicle waits for the LLM
    'compass_dvl_long_holds': [{'type': 'compass_disturbance', 't_start': 10.0, 'params': {'offset_deg': 30}},
                               {'type': 'dvl_dropout', 't_start': 10.0}],
}
HOLD = ('hold_depth', {'depth': 1.5, 'duration_s': 60}, 120)
# simulator changes per condition: a 0.1 m/s water current the learned velocity model never saw in training
SIM_OVERRIDES = {'dvl_dropout_current': {'physics': {'current': [0.08, 0.06, 0.0]}}}
STEPS = [('navigate_to', {'x': 2.0, 'y': 0.0, 'depth': 1.5}, 60), ('navigate_to', {'target': 'gate'}, 120),
         ('search_pattern', {'target': 'red_buoy'}, 120), ('navigate_to', {'target': 'red_buoy'}, 120),
         ('surface', {}, 90)]


_OFF = {'sonar_walls': {'enabled': False}, 'learned_velocity': {'enabled': False, 'current_estimation': False}}
ESTIMATORS = {   # name -> ekf config overrides (every fusion switch explicit: independent of the defaults)
    'ekf': {'estimator': 'ekf'},
    'fgo': {'estimator': 'fgo'},
    'ekf+learned_vel': {'estimator': 'ekf', 'learned_velocity': {'enabled': True}},
    'fgo+learned_vel': {'estimator': 'fgo', 'learned_velocity': {'enabled': True}},
    'ekf+learned_vel+current': {'estimator': 'ekf', 'learned_velocity': {'enabled': True, 'current_estimation': True}},
    'ekf+sonar_walls': {'estimator': 'ekf', 'sonar_walls': {'enabled': True}},
    'ekf+sonar_walls+learned_vel+current': {'estimator': 'ekf', 'sonar_walls': {'enabled': True},
                                            'learned_velocity': {'enabled': True, 'current_estimation': True}},
    'ekf+sonar_walls+learned_vel+current+floor': {'estimator': 'ekf', 'sonar_walls': {'enabled': True},
                                                  'learned_velocity': {'enabled': True, 'current_estimation': True,
                                                                       'std_floor_mps': 0.03}},
    'ekf+sonar_landmarks': {'estimator': 'ekf', 'landmarks': {'sonar_posts': True}},
    'ekf+sonar_landmarks+learned_vel': {'estimator': 'ekf', 'landmarks': {'sonar_posts': True},
                                        'learned_velocity': {'enabled': True}},
}


def run(cond, seed, estimator='ekf'):
    from auv_marine_sim.world import deep_update
    from auv_onboard.factory import make_backend, make_registry
    import copy
    ob = {'backend': {'max_rtf': None, 'idle_rtf': None},
          'ekf': deep_update(copy.deepcopy(_OFF), copy.deepcopy(ESTIMATORS[estimator]))}
    if cond.endswith('no_landmarks'):
        ob['ekf']['landmarks'] = {'gate_prior': False}
    be = make_backend('default', faults=CONDITIONS[cond], seed=seed, onboard_overrides=ob,
                      sim_overrides=SIM_OVERRIDES.get(cond))
    reg, _ = make_registry(be)
    ok = True
    steps = STEPS if cond != 'compass_dvl_long_holds' else STEPS[:2] + [HOLD] + STEPS[2:4] + [HOLD] + STEPS[4:]
    try:
        for name, p, to in steps:
            o = reg.call(name, p, to)
            if not o.ok:
                ok = False
                break
        r = be.sim.referee.report()
    finally:
        be.shutdown()
    L = r['localisation']
    return {'condition': cond, 'estimator': estimator, 'seed': seed, 'mission_ok': ok, 'gate_passed': r['gate_passed'],
            'buoy_touched': 'red_buoy' in r['buoys_touched'], 'collisions': r['collisions'],
            'rmse_xy_m': L['rmse_xy_m'], 'max_xy_m': L['max_xy_m'], 'final_xy_m': L['final_xy_m'],
            'rmse_depth_m': L['rmse_depth_m']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--out', default='eval/results/nav_study')
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--estimators', default='ekf')
    ap.add_argument('--conditions', default=','.join(CONDITIONS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ests = a.estimators.split(',')
    conds = a.conditions.split(',')
    if a.summary_only:
        rows = json.load(open(os.path.join(a.out, 'runs.json')))
    else:
        rows = []
        for c in conds:
            for e in ests:
                for sd in range(a.seeds):
                    rows.append(run(c, sd, e))
                    print(json.dumps(rows[-1]), flush=True)
    for r in rows:
        r.setdefault('estimator', 'ekf')
    with open(os.path.join(a.out, 'runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    keys = []
    for r in rows:
        if (r['condition'], r['estimator']) not in keys:
            keys.append((r['condition'], r['estimator']))
    lines = ['# Localisation study (navigation estimate vs ground truth)', '',
             f'Scripted gate -> red buoy -> surface, no LLM; {len(rows) // max(1, len(keys))} seeds per row; faults injected at t = 10 s.', '',
             '"tools ok" = every tool call reported success (what the vehicle believes); "referee: task" = gate '
             'passed AND red buoy touched according to simulator ground truth.', '',
             '| Condition | Estimator | tools ok | referee: task | RMSE xy (m) mean ± sd | max xy (m) mean | worst max xy (m) | RMSE depth (m) |',
             '|---|---|---|---|---|---|---|---|']
    for c, e in keys:
        rs = [r for r in rows if r['condition'] == c and r['estimator'] == e]
        rm = [r['rmse_xy_m'] for r in rs]
        mx = [r['max_xy_m'] for r in rs]
        task = sum(bool(r['gate_passed']) and bool(r['buoy_touched']) for r in rs)
        lines.append(f'| {c} | {e} | {sum(r["mission_ok"] for r in rs)}/{len(rs)} | {task}/{len(rs)} | {statistics.mean(rm):.3f} ± '
                     f'{statistics.pstdev(rm):.3f} | {statistics.mean(mx):.3f} | {max(mx):.3f} | '
                     f'{statistics.mean(r["rmse_depth_m"] for r in rs):.3f} |')
    with open(os.path.join(a.out, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    json.dump(rows, open(os.path.join(a.out, 'runs.json'), 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
