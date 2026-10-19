#!/usr/bin/env python3
"""Scripted (no LLM) mission on the MuJoCo sim through v1's tool registry: the M2/M3 smoke test.

  scripts/scripted_mission.py [--nav ekf|ground_truth] [--arena default] [--fault TYPE[:t_start[:k=v,...]]]
"""
import argparse
import json
import sys
import time


def parse_fault(s):
    parts = s.split(':')
    f = {'type': parts[0], 't_start': float(parts[1]) if len(parts) > 1 else 0.0, 'params': {}}
    if len(parts) > 2:
        for kv in parts[2].split(','):
            k, v = kv.split('=')
            try:
                v = float(v)
            except ValueError:
                pass
            f['params'][k] = v
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nav', default='ekf')
    ap.add_argument('--arena', default='default')
    ap.add_argument('--fault', action='append', default=[])
    ap.add_argument('--no-health', action='store_true')
    ap.add_argument('--rtf', type=float, default=0.0, help='max real-time factor (0 = unthrottled)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--detector', default='ground_truth', choices=['ground_truth', 'yolo'])
    a = ap.parse_args()
    from auv_onboard.factory import make_backend, make_registry
    be = make_backend(a.arena, faults=[parse_fault(f) for f in a.fault], seed=a.seed,
                      onboard_overrides={'backend': {'nav_source': a.nav, 'max_rtf': a.rtf or None,
                                                     'detection_source': a.detector,
                                                     'health_monitor': not a.no_health, 'idle_rtf': None}})
    reg, _ = make_registry(be)
    steps = [('navigate_to', {'x': 2.0, 'y': 0.0, 'depth': 1.5}, 60),
             ('navigate_to', {'target': 'gate'}, 120),
             ('search_pattern', {'target': 'red_buoy', 'pattern': 'yaw_sweep', 'radius_m': 6.0}, 120),
             ('navigate_to', {'target': 'red_buoy'}, 120),
             ('surface', {}, 90)]
    w0 = time.time()
    try:
        for name, p, to in steps:
            obs = reg.call(name, p, to)
            print(f'[t={be.now():6.1f}] {name:15s} ok={obs.ok} {obs.error or obs.data.get("message", "")}', flush=True)
            if not obs.ok and name != 'search_pattern':
                if obs.error_code == 'health_veto':      # the backend is already surfacing: let it finish
                    deadline = be.now() + 120
                    while be.now() < deadline and any(t.is_alive() for t in be._threads):
                        time.sleep(0.05)
                break
    except BaseException:
        be.shutdown()
        raise
    rep = be.sim.referee.report()
    rep.pop('events')
    out = {'nav': a.nav, 'faults': a.fault, 'referee': rep, 'wall_s': round(time.time() - w0, 1),
           'ekf': be.ekf.summary(), 'health_faults': sorted({f['type'] for r in be.health_log for f in r.faults}),
           'backend_events': be.events, 'injected_log': be.sim.faults.log}
    print(json.dumps(out, indent=1))
    be.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
