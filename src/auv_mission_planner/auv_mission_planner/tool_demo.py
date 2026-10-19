"""Drive the vehicle through the tools layer with a fixed script (no LLM).

ros2 run auv_mission_planner tool_demo [--backend ros|fake] [--steps JSON]
Prints one JSON observation per line; exit 0 only if every step succeeded.
"""
import argparse
import json
import sys

from auv_mission_planner.config import load_config
from auv_mission_planner.tools.registry import ToolRegistry

DEMO = [
    ['get_pose', {}],
    ['get_battery', {}],
    ['hold_depth', {'depth': 1.5, 'duration_s': 3}],
    ['get_detections', {}],
    ['navigate_to', {'target': 'gate', 'pass_through': True}],
    ['navigate_to', {'target': 'red_buoy', 'standoff_m': 0.0}],
    ['surface', {}],
    ['get_pose', {}],
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='ros', choices=['ros', 'fake'])
    ap.add_argument('--steps', help='JSON list of [tool, params]', default=None)
    ap.add_argument('--step-timeout', type=float, default=120.0)
    ap.add_argument('--config', default=None)
    a, _ = ap.parse_known_args(argv)
    cfg = load_config(a.config)
    steps = json.loads(a.steps) if a.steps else DEMO
    if a.backend == 'fake':
        from auv_mission_planner.tools.fake_backend import FakeBackend
        backend = FakeBackend()
    else:
        from auv_mission_planner.tools.ros_backend import RosBackend
        backend = RosBackend(cfg)
        if not backend.wait_ready(30.0):
            print(json.dumps({'error': 'vehicle backend not ready'}), flush=True)
            backend.shutdown()
            return 3
    tools = ToolRegistry(backend, cfg)
    ok_all = True
    try:
        for name, params in steps:
            obs = tools.call(name, params, timeout_s=a.step_timeout)
            print(json.dumps(obs.model_dump()), flush=True)
            if not obs.ok:
                ok_all = False
                break
    finally:
        backend.shutdown()
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
