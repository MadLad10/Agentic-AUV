"""mission CLI.

  mission "pass through the gate, touch the red buoy, then surface; abort if battery is below 20%"
  mission --dry-run --provider groq "touch the green buoy"
  mission --backend fake "..."          # kinematic fake vehicle, no Gazebo

Exit codes: 0 succeeded/planned, 1 aborted, 3 refused, 4 rejected by validator, 5 LLM error, 6 no vehicle.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from auv_mission_planner.config import find_workspace_root, load_config, load_dotenv, load_llm_config

EXIT = {'succeeded': 0, 'planned': 0, 'aborted': 1, 'refused': 3, 'rejected': 4, 'error': 5}


def build_backend(kind: str, cfg: dict, log):
    if kind == 'fake':
        from auv_mission_planner.tools.fake_backend import FakeBackend
        return FakeBackend()
    from auv_mission_planner.tools.ros_backend import RosBackend
    b = RosBackend(cfg)
    if not b.wait_ready(20.0 if kind == 'ros' else 5.0):
        b.shutdown()
        if kind == 'auto':
            log('[cli] no vehicle backend answering; using the fake backend for context')
            from auv_mission_planner.tools.fake_backend import FakeBackend
            return FakeBackend()
        return None
    return b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='mission', description='Plan and run a AUV mission from plain language.')
    ap.add_argument('prompt', nargs='+')
    ap.add_argument('--provider', default=None, help='LLM provider name from config/llm.yaml')
    ap.add_argument('--dry-run', action='store_true', help='plan + validate only, do not move the vehicle')
    ap.add_argument('--backend', default=None, choices=['ros', 'fake', 'auto'],
                    help='vehicle backend (default: ros, or auto for --dry-run)')
    ap.add_argument('--timeout', type=float, default=None, help='wall-clock mission timeout (s)')
    ap.add_argument('--trace-dir', default=None, help='where to write trace JSON (default <ws>/traces)')
    ap.add_argument('--config', default=None)
    ap.add_argument('--llm-config', default=None)
    ap.add_argument('--json', action='store_true', help='print the final result as JSON only')
    ap.add_argument('--quiet', action='store_true')
    a, _ = ap.parse_known_args(argv)        # tolerate --ros-args from ros2 run
    prompt = ' '.join(a.prompt)
    log = (lambda m: None) if (a.quiet or a.json) else (lambda m: print(m, flush=True))

    load_dotenv()
    cfg = load_config(a.config)
    llm_cfg = load_llm_config(a.llm_config)
    from auv_mission_planner.llm.providers import make_provider
    from auv_mission_planner.mission import MissionRunner
    from auv_mission_planner.tools.registry import ToolRegistry

    provider = make_provider(llm_cfg, a.provider)
    backend_kind = a.backend or ('auto' if a.dry_run else 'ros')
    backend = build_backend(backend_kind, cfg, log)
    if backend is None:
        print('[cli] vehicle backend not ready (is the sim running? scripts/run_sim_test.sh)', file=sys.stderr)
        return 6
    trace_dir = a.trace_dir or os.path.join(find_workspace_root() or '.', 'traces')
    try:
        runner = MissionRunner(cfg, llm_cfg, provider, ToolRegistry(backend, cfg), trace_dir=trace_dir, log=log)
        res = runner.run(prompt, dry_run=a.dry_run, wall_timeout_s=a.timeout)
    finally:
        backend.shutdown()
    if a.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        log(f'[result] {res["outcome"].upper()}' + (f': {res["reason"]}' if res['reason'] else ''))
        log(f'[result] plans={res["plans"]} replans={res["replans"]} unsafe_caught={res["unsafe_plans_caught"]} '
            f'repairs={res["validator_repairs"]} tokens={res["tokens_in"]}+{res["tokens_out"]} '
            f'llm={res["llm_latency_s"]}s sim={res["sim_time_s"]}s wall={res["wall_s"]}s '
            f'cost=${res["cost_usd"]} (list ${res["list_cost_usd"]})')
        if a.dry_run and res['final_plan']:
            log(json.dumps(res['final_plan'], indent=2))
        log(f'[result] trace: {res["trace"]}')
    return EXIT.get(res['outcome'], 5)


if __name__ == '__main__':
    sys.exit(main())
