#!/usr/bin/env python3
"""v1 single agent vs v2 multi-agent graph on the MuJoCo sim: 25 v1 missions + the sensor-fault tier.

Both systems get the same simulator, onboard stack (EKF navigation), tools, validator and LLM.
Differences are exactly the v2 additions: the multi-agent graph (perception/navigation/diagnostics
agents, sonar use, VLM), and the health & safety monitor with veto and thrust re-allocation.

Each run is a separate process (a crash cannot take down the batch); results are appended to
runs.jsonl so `--resume` continues an interrupted evaluation. Scoring reuses v1's rules
(eval/run_eval.py) on the ground-truth referee report.

  setsid nohup python eval/run_eval_v2.py --systems v1,v2 --sets nominal,faults --provider ollama \\
      --out eval/results/v2_ollama --resume > eval/results/v2_ollama.log 2>&1 &

Repeats and ablations (PLAN.md §8): `--seed N` sets the simulator seed and the LLM sampling seed;
`--ablation` switches one component off for this evaluation only (the switch exists only here, never in
the vehicle, MCP server or dashboard):
  no_validator  plans get the schema check only: no rule checks, repairs or runtime prechecks
  no_health     no health monitor: no veto, fault detection or thrust re-allocation
  no_fusion     plain EKF (no sonar walls, learned velocity, current estimate) and no camera-sonar tracker
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SETS = {'nominal': os.path.join(WS, 'eval', 'missions.yaml'), 'faults': os.path.join(WS, 'eval', 'faults.yaml')}
MOTION = {'navigate_to', 'search_pattern', 'hold_depth'}
ABLATIONS = ('none', 'no_validator', 'no_health', 'no_fusion')


class PassThroughValidator:
    """`no_validator` ablation: accept every schema-valid plan unchanged and never block a step."""

    def validate(self, plan, prompt=None):
        from auv_mission_planner.validator import ValidationReport
        return ValidationReport(ok=plan is not None and bool(plan.steps), plan=plan)

    def precheck_step(self, *a, **kw):
        return None


def ablation_overrides(ablation: str) -> dict:
    if ablation == 'no_health':
        return {'backend': {'health_monitor': False}}
    if ablation == 'no_fusion':
        return {'ekf': {'sonar_walls': {'enabled': False},
                        'learned_velocity': {'enabled': False, 'current_estimation': False}},
                'tracking': {'enabled': False}}
    return {}


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def v1_eval():
    spec = importlib.util.spec_from_file_location('v1_run_eval', os.path.join(WS, 'eval', 'run_eval.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_missions(sets):
    import yaml
    out = []
    for s in sets:
        with open(SETS[s]) as f:
            out += [dict(m, set=s) for m in yaml.safe_load(f)['missions']]
    return out


# ====================================================================== one run (child process)
def run_one(spec: dict) -> dict:
    from auv_mission_planner.config import load_config, load_llm_config
    from auv_mission_planner.llm.providers import make_provider
    from auv_onboard.factory import make_backend
    m, system = spec['mission'], spec['system']
    seed, ablation = int(spec.get('seed', 0)), spec.get('ablation', 'none')
    onboard = {'backend': {'health_monitor': system == 'v2', 'nav_source': 'ekf',
                           'max_rtf': spec.get('max_rtf', 20.0), 'idle_rtf': 1.0},
               'perception': {'blind_classes': [c.strip() for c in str(m.get('blind_classes', '')).split(',') if c.strip()]}}
    onboard = deep_merge(onboard, ablation_overrides(ablation))
    sim_over = {'sensors': {'battery': {'initial_percent': float(m['battery_start'])}}} if m.get('battery_start') else None
    be = make_backend(m.get('arena', 'default'), faults=m.get('faults', []), seed=seed,
                      onboard_overrides=onboard, sim_overrides=sim_over)
    cfg, llm_cfg = load_config(), load_llm_config()
    llm_cfg['providers'][spec['provider']]['seed'] = seed          # LLM sampling seed (used by ollama)
    provider = make_provider(llm_cfg, spec['provider'])
    t0 = time.time()
    row = {'system': system, 'mission': m['id'], 'tier': m['tier'], 'set': m['set'], 'provider': spec['provider'],
           'model': provider.model, 'seed': seed, 'ablation': ablation}
    try:
        if system == 'v1':
            from auv_mission_planner.mission import MissionRunner
            from auv_mission_planner.tools.registry import ToolRegistry
            runner = MissionRunner(cfg, llm_cfg, provider, ToolRegistry(be, cfg),
                                   trace_dir=os.path.join(spec['out'], 'traces'))
            res = runner.run(m['prompt'])
            tools = []
            if res.get('trace') and os.path.isfile(res['trace']):
                with open(res['trace']) as f:
                    tools = [c['step']['tool'] for c in json.load(f).get('tool_calls', [])]
            row.update(outcome=res['outcome'], reason=res.get('reason', ''), final_plan=res.get('final_plan'),
                       replans=res.get('replans', 0), llm_calls=res.get('llm_calls', 0),
                       tokens_in=res.get('tokens_in', 0), tokens_out=res.get('tokens_out', 0),
                       llm_latency_s=res.get('llm_latency_s', 0.0), unsafe_plans_caught=res.get('unsafe_plans_caught', 0),
                       tools_executed=tools, detected_faults=[], diagnosis=None)
        else:
            from auv_agents.run import build_runtime, load_agents_config, run_mission
            from auv_agents.vlm import make_vlm
            acfg = load_agents_config()
            diag = make_provider(llm_cfg, spec['provider'])
            rt = build_runtime(be, provider, acfg, diag, make_vlm(acfg['perception']['vlm']), auto_approve=True)
            if ablation == 'no_validator':
                rt.validator = rt.executor.validator = PassThroughValidator()
            res = run_mission(rt, m['prompt'])
            st = res.get('stats') or {}
            calls = rt.llm_calls + [c for c in rt.diagnostics.calls if c.get('usage')]
            us = [c['usage'] for c in calls if c.get('usage')]
            row.update(outcome=res.get('outcome'), reason=res.get('reason', ''), final_plan=res.get('plan'),
                       replans=res.get('replans', 0), llm_calls=len(rt.llm_calls) + len(rt.diagnostics.calls),
                       tokens_in=sum(u['input_tokens'] for u in us), tokens_out=sum(u['output_tokens'] for u in us),
                       llm_latency_s=round(sum(u['latency_s'] for u in us), 2),
                       unsafe_plans_caught=st.get('unsafe_plans_caught', 0),
                       tools_executed=[r['step']['tool'] for r in res.get('records', [])],
                       diagnosis=res.get('diagnosis'), inspections=st.get('inspections', 0),
                       vlm_queries=len(rt.perception.vlm_log))
        # let an abort finish surfacing, then give the referee its surfaced window
        t_end = time.time() + 150
        while time.time() < t_end and any(t.is_alive() for t in be._threads):
            time.sleep(0.1)
        time.sleep(2.5)
    except Exception as e:
        row.update(outcome='error', reason=f'{type(e).__name__}: {e}')
    finally:
        with be._lock:
            ref = be.sim.referee.report()
        first = {}
        for rep in be.health_log:
            for f in rep.faults:
                first.setdefault(f['type'], rep.t)
        row.update(referee={k: v for k, v in ref.items() if k != 'events'}, health_first_t=first,
                   detected_faults=sorted(first), injected=m.get('faults', []),
                   sim_time_s=round(be.now(), 1), wall_s=round(time.time() - t0, 1),
                   backend_events=be.events[:10])
        be.shutdown()
    return row


# ====================================================================== scoring
def score(row: dict, m: dict, limits: dict, v1) -> dict:
    ref = row.get('referee') or {}
    outcome = {'outcome': row.get('outcome'), 'final_plan': row.get('final_plan')}
    trace = {'tool_calls': [{'step': {'tool': t}} for t in row.get('tools_executed', [])]}
    failed = v1.check_sim(m, outcome, trace, ref, limits)
    # A crashed/timed-out worker may have no final referee report. Absence of
    # evidence is unknown safety, never a successful safety trial.
    numeric = ('max_depth_m', 'boundary_contacts', 'collisions')
    complete = isinstance(ref.get('surfaced_at_end'), bool) and all(
        isinstance(ref.get(k), (int, float)) and not isinstance(ref[k], bool)
        and math.isfinite(ref[k]) and ref[k] >= 0 for k in numeric)
    safe = complete and v1.sim_safe(ref, trace, limits) and ref['collisions'] == 0
    safety_status = ('verified_safe' if safe else 'unsafe') if complete else 'unknown'
    task = None
    if m.get('task_checks'):
        tf = v1.check_sim(dict(m, accept=[row.get('outcome')], checks=m['task_checks']), outcome, trace, ref, limits)
        task = not tf
    det = lat = None
    if m.get('detect'):
        diag = (row.get('diagnosis') or {}).get('fault_types', [])
        det = m['detect'] in set(row.get('detected_faults', [])) | set(diag)
        t0 = (m.get('faults') or [{}])[0].get('t_start', 0.0)
        if m['detect'] in (row.get('health_first_t') or {}):
            lat = round(row['health_first_t'][m['detect']] - t0, 2)
    return {'pass': not failed, 'safe': safe, 'safety_status': safety_status,
            'failed_checks': failed, 'task_done': task,
            'fault_detected': det, 'detect_latency_s': lat}


# ====================================================================== summary
def summarise(rows, missions, out_dir):
    by = {m['id']: m for m in missions}
    systems = sorted({r['system'] for r in rows})
    lines = ['# v1 single agent vs v2 multi-agent (MuJoCo sim, referee-scored)', '',
             f'Provider/model: {", ".join(sorted({r["provider"] + "/" + r.get("model", "?") for r in rows}))}  ',
             f'Runs: {len(rows)}  ', '']
    lines += ['| Tier | ' + ' | '.join(f'{s} pass | {s} safe' for s in systems) + ' |',
              '|---|' + '---|---|' * len(systems)]
    tiers = []
    for m in missions:
        if m['tier'] not in tiers:
            tiers.append(m['tier'])
    for tier in tiers + ['ALL']:
        cells = []
        for s in systems:
            rs = [r for r in rows if r['system'] == s and (tier == 'ALL' or by[r['mission']]['tier'] == tier)]
            cells.append(f'{sum(r["pass"] for r in rs)}/{len(rs)} | {sum(r["safe"] for r in rs)}/{len(rs)}')
        lines.append(f'| {tier} | ' + ' | '.join(cells) + ' |')
    ft = [r for r in rows if r['set'] == 'faults']
    if ft:
        lines += ['', '## Fault tier', '',
                  '| Mission | Fault | ' + ' | '.join(f'{s} outcome | {s} task | {s} detected (latency s)' for s in systems) + ' |',
                  '|---|---|' + '---|---|---|' * len(systems)]
        for mid in [m['id'] for m in missions if m['set'] == 'faults']:
            cells = []
            for s in systems:
                r = next((x for x in ft if x['mission'] == mid and x['system'] == s), None)
                if r is None:
                    cells.append('– | – | –')
                    continue
                det = '–' if r['fault_detected'] is None else ('yes' if r['fault_detected'] else 'no')
                if r['detect_latency_s'] is not None:
                    det += f' ({r["detect_latency_s"]})'
                task = '–' if r['task_done'] is None else ('yes' if r['task_done'] else 'no')
                cells.append(f'{r["outcome"]}{"" if r["pass"] else " ✗"} | {task} | {det}')
            lines.append(f'| {mid} | {by[mid]["faults"][0]["type"]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Cost and latency', '', '| System | LLM calls | tokens in | tokens out | LLM s | sim s | wall s | replans |',
              '|---|---|---|---|---|---|---|---|']
    for s in systems:
        rs = [r for r in rows if r['system'] == s]
        lines.append(f'| {s} | {sum(r.get("llm_calls", 0) for r in rs)} | {sum(r.get("tokens_in", 0) for r in rs)} | '
                     f'{sum(r.get("tokens_out", 0) for r in rs)} | {round(sum(r.get("llm_latency_s", 0) for r in rs))} | '
                     f'{round(sum(r.get("sim_time_s", 0) for r in rs))} | {round(sum(r.get("wall_s", 0) for r in rs))} | '
                     f'{sum(r.get("replans", 0) for r in rs)} |')
    lines += ['', 'Safety requires a complete final referee report. Unknown safety is counted as not verified safe.',
              '', '## Per run', '', '| System | Mission | Outcome | Pass | Safe | Failed checks | loc RMSE m |',
              '|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: (r['mission'], r['system'])):
        loc = (r.get('referee') or {}).get('localisation', {}).get('rmse_xy_m', '–')
        lines.append(f'| {r["system"]} | {r["mission"]} | {r["outcome"]} | {"✓" if r["pass"] else "✗"} | '
                     f'{"unknown" if r.get("safety_status") == "unknown" else "✓" if r["safe"] else "✗"} | '
                     f'{"; ".join(r["failed_checks"])[:120]} | {loc} |')
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    fields = ['system', 'mission', 'tier', 'set', 'provider', 'model', 'outcome', 'pass', 'safe', 'safety_status', 'task_done',
              'fault_detected', 'detect_latency_s', 'replans', 'llm_calls', 'tokens_in', 'tokens_out',
              'llm_latency_s', 'sim_time_s', 'wall_s', 'reason']
    with open(os.path.join(out_dir, 'runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({**r, 'reason': str(r.get('reason', ''))[:200]})
    return '\n'.join(lines)


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--one', help=argparse.SUPPRESS)
    ap.add_argument('--systems', default='v1,v2')
    ap.add_argument('--sets', default='nominal,faults')
    ap.add_argument('--provider', default='ollama')
    ap.add_argument('--out', default='eval/results/v2_eval')
    ap.add_argument('--only', default='')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--timeout', type=int, default=1200, help='wall seconds per run')
    ap.add_argument('--max-rtf', type=float, default=20.0)
    ap.add_argument('--summary-only', action='store_true')
    ap.add_argument('--seed', type=int, default=0, help='simulator + LLM sampling seed')
    ap.add_argument('--ablation', default='none', choices=ABLATIONS)
    a = ap.parse_args()
    if a.one:
        print('RESULT ' + json.dumps(run_one(json.loads(a.one)), default=str), flush=True)
        return 0
    os.makedirs(a.out, exist_ok=True)
    from auv_mission_planner.config import load_config
    limits = load_config()['limits']
    v1 = v1_eval()
    missions = load_missions(a.sets.split(','))
    if a.only:
        keep = set(a.only.split(','))
        missions = [m for m in missions if m['id'] in keep]
    path = os.path.join(a.out, 'runs.jsonl')
    done = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done[(r['system'], r['mission'])] = r
    if not a.resume and not a.summary_only and done:
        sys.exit(f'{path} exists: use --resume or a new --out')
    if not a.summary_only:
        for m in missions:
            for system in a.systems.split(','):
                if (system, m['id']) in done:
                    continue
                spec = {'system': system, 'mission': m, 'provider': a.provider, 'out': a.out, 'max_rtf': a.max_rtf,
                        'seed': a.seed, 'ablation': a.ablation}
                t0 = time.time()
                print(f'[eval] {system} {m["id"]}: {m["prompt"]}', flush=True)
                try:
                    p = subprocess.run([sys.executable, __file__, '--one', json.dumps(spec)], capture_output=True,
                                       text=True, timeout=a.timeout, cwd=WS)
                    res = [ln for ln in p.stdout.splitlines() if ln.startswith('RESULT ')]
                    row = json.loads(res[-1][7:]) if res else {
                        'system': system, 'mission': m['id'], 'tier': m['tier'], 'set': m['set'], 'provider': a.provider,
                        'outcome': 'error', 'reason': (p.stderr or '')[-400:]}
                except subprocess.TimeoutExpired:
                    row = {'system': system, 'mission': m['id'], 'tier': m['tier'], 'set': m['set'],
                           'provider': a.provider, 'outcome': 'error', 'reason': f'run timed out after {a.timeout}s'}
                row.setdefault('seed', a.seed)
                row.setdefault('ablation', a.ablation)
                row.update(score(row, m, limits, v1))
                with open(path, 'a') as f:
                    f.write(json.dumps(row, default=str) + '\n')
                done[(system, m['id'])] = row
                print(f'[eval] {system} {m["id"]}: {row["outcome"]} pass={row["pass"]} safe={row["safe"]} '
                      f'{"; ".join(row["failed_checks"])} ({time.time() - t0:.0f}s)', flush=True)
    rows = [r for r in done.values() if r['mission'] in {m['id'] for m in missions}]
    print(summarise(rows, missions, a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
