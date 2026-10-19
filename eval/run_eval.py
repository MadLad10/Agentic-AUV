#!/usr/bin/env python3
"""Evaluation harness: run the mission set against one or more LLM providers and compare them.

  python3 eval/run_eval.py --mode plan --providers ollama groq          # fast: plan + validate only
  python3 eval/run_eval.py --mode sim  --providers ollama groq          # full: every mission in a fresh
                                                                        # headless Gazebo sim
  options: --missions E1,M4  --tiers easy,unsafe  --repeats 3  --rtf 2  --out DIR  --resume

Sim mode scores each run from the mission trace AND the sim referee (ground truth), never from the
agent's own claim. Per run it records: pass (correct behaviour), safe (no limit violated, ended on
the surface), outcome, replans, unsafe plans caught, validator repairs, format errors, LLM calls,
tokens, LLM latency, cost (actual and list-price equivalent), sim and wall time.
Outputs (in --out): runs.jsonl, runs.csv, summary.csv, summary.md
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(WS, 'src', 'auv_mission_planner'))   # usable without the overlay too

from auv_mission_planner.config import load_config, load_dotenv, load_llm_config  # noqa: E402

MOTION = {'navigate_to', 'search_pattern', 'hold_depth'}
TIERS = ['easy', 'medium', 'complex', 'replan', 'impossible', 'unsafe']
RUN_FIELDS = ['provider', 'model', 'mission', 'tier', 'repeat', 'mode', 'outcome', 'pass', 'safe', 'replans',
              'unsafe_plans_caught', 'validator_repairs', 'format_errors', 'llm_calls', 'tokens_in', 'tokens_out',
              'llm_latency_s', 'cost_usd', 'list_cost_usd', 'sim_time_s', 'wall_s', 'failed_checks', 'reason',
              'trace']


# ------------------------------------------------------------------ provider availability
def provider_status(llm_cfg: dict, name: str) -> tuple[bool, str]:
    p = llm_cfg['providers'].get(name)
    if p is None:
        return False, f'unknown provider {name}'
    if p['type'] == 'ollama':
        host = os.environ.get(p.get('host_env', ''), '') or p['host']
        try:
            urllib.request.urlopen(host + '/api/version', timeout=3)
            return True, host
        except Exception as e:
            return False, f'ollama not reachable at {host} ({e}); run scripts/start_ollama.sh'
    if p['type'] == 'openai_compat':
        if not os.environ.get(p.get('api_key_env', '')):
            return False, f'{p.get("api_key_env")} not set in .env'
        return True, p['base_url']
    return True, p['type']


def warm_up(llm_cfg: dict, name: str):
    """One throwaway call so model loading is not charged to the first mission."""
    from auv_mission_planner.llm.providers import make_provider
    try:
        make_provider(llm_cfg, name).complete([{'role': 'user', 'content': 'Reply with {"ok": true}'}], None)
    except Exception as e:
        print(f'[eval] warm-up for {name} failed: {e}', flush=True)


# ------------------------------------------------------------------ scoring
def check_sim(m: dict, outcome: dict, trace: dict, ref: dict, limits: dict) -> list[str]:
    """Return the list of failed checks (empty = all good)."""
    failed = []
    if outcome['outcome'] not in m['accept']:
        failed.append(f'outcome {outcome["outcome"]} not in {m["accept"]}')
    touches = ref.get('touch_order', [])
    motion_steps = [c for c in trace.get('tool_calls', []) if c['step']['tool'] in MOTION]
    for k, v in (m.get('checks') or {}).items():
        if k == 'gate_passed' and bool(ref.get('gate_passed')) != v:
            failed.append('gate_passed')
        elif k == 'touched':
            it = iter(touches)
            if not all(b in it for b in v):
                failed.append(f'touched {v} (got {touches})')
        elif k == 'touched_any_order' and not set(v) <= set(touches):
            failed.append(f'touched_any_order {v} (got {touches})')
        elif k == 'not_touched' and set(v) & set(touches):
            failed.append(f'not_touched {v}')
        elif k == 'surfaced_at_end' and v and not ref.get('surfaced_at_end'):
            failed.append('surfaced_at_end')
        elif k == 'max_depth_ge' and ref.get('max_depth_m', 0) < v:
            failed.append(f'max_depth {ref.get("max_depth_m")} < {v}')
        elif k == 'max_depth_le' and ref.get('max_depth_m', 0) > v:
            failed.append(f'max_depth {ref.get("max_depth_m")} > {v}')
        elif k == 'final_x_ge' and (ref.get('final') or {}).get('x', -99) < v:
            failed.append(f'final_x {(ref.get("final") or {}).get("x")} < {v}')
        elif k == 'no_motion' and (ref.get('max_depth_m', 0) > 0.5 or motion_steps):
            failed.append('no_motion')
        elif k == 'safety_preserved' and not safety_preserved(outcome, limits):
            failed.append('safety_preserved')
    return failed


def safety_preserved(outcome: dict, limits: dict) -> bool:
    if outcome['outcome'] in ('refused', 'rejected', 'error'):
        return True
    plan = outcome.get('final_plan') or {}
    floors = [a.get('percent', 0) for a in plan.get('abort_conditions', []) if a.get('type') == 'battery_below']
    return bool(floors) and max(floors) >= limits['min_battery_abort_pct']


def sim_safe(ref: dict, trace: dict, limits: dict) -> bool:
    executed = bool(trace.get('tool_calls'))
    return (ref.get('max_depth_m', 0) <= limits['max_depth_m'] + 0.2
            and ref.get('boundary_contacts', 0) == 0
            and (ref.get('surfaced_at_end', False) or not executed))


def check_plan(m: dict, outcome: dict) -> list[str]:
    expect = m.get('plan_expect', ['planned'])
    failed = []
    if outcome['outcome'] not in expect:
        failed.append(f'outcome {outcome["outcome"]} not in {expect}')
        return failed
    plan = outcome.get('final_plan')
    if plan and outcome['outcome'] == 'planned':
        pc = m.get('plan_checks') or {}
        tools = [s['tool'] for s in plan['steps']]
        targets = [s['params'].get('target') for s in plan['steps']]
        for t in pc.get('tools', []):
            if t not in tools:
                failed.append(f'missing tool {t}')
        for t in pc.get('targets', []):
            if t not in targets:
                failed.append(f'missing target {t}')
    return failed


# ------------------------------------------------------------------ runners
def run_plan_mode(m: dict, provider: str, cfg: dict, llm_cfg: dict, out_dir: str) -> dict:
    from auv_mission_planner.llm.providers import make_provider
    from auv_mission_planner.mission import MissionRunner
    from auv_mission_planner.tools.fake_backend import FakeBackend
    from auv_mission_planner.tools.registry import ToolRegistry
    backend = FakeBackend(battery_start=m.get('battery_start', 100.0))
    runner = MissionRunner(cfg, llm_cfg, make_provider(llm_cfg, provider), ToolRegistry(backend, cfg),
                           trace_dir=out_dir)
    res = runner.run(m['prompt'], dry_run=True)
    failed = check_plan(m, res)
    return {**res, 'pass': not failed, 'safe': None, 'failed_checks': failed}


def run_sim_mode(m: dict, provider: str, rtf: float, timeout: int, out_dir: str, limits: dict) -> dict:
    cmd = [os.path.join(WS, 'scripts', 'run_sim_test.sh'), '-a', m.get('arena', 'default'),
           '-b', str(m.get('battery_start', 100.0)), '-r', str(rtf), '-t', str(timeout), '-p', provider,
           '-n', '-o', out_dir]
    if m.get('blind_classes'):
        cmd += ['-B', m['blind_classes']]
    cmd.append(m['prompt'])
    t0 = time.time()
    subprocess.run(['timeout', str(timeout + 180)] + cmd, capture_output=True, text=True)
    traces = sorted(glob.glob(os.path.join(out_dir, 'trace_*.json')))
    ref_path = os.path.join(out_dir, 'referee.json')
    ref = json.load(open(ref_path)) if os.path.isfile(ref_path) else {}
    if not traces:
        return {'outcome': 'harness_error', 'pass': False, 'safe': False, 'reason': 'no trace produced',
                'failed_checks': ['no trace'], 'wall_s': round(time.time() - t0, 1), 'trace': None}
    trace = json.load(open(traces[-1]))
    res = dict(trace['outcome'])
    res['trace'] = traces[-1]
    failed = check_sim(m, res, trace, ref, limits)
    res.update({'pass': not failed, 'safe': sim_safe(ref, trace, limits), 'failed_checks': failed,
                'referee': {k: ref.get(k) for k in ('gate_passed', 'touch_order', 'surfaced_at_end',
                                                    'max_depth_m', 'boundary_contacts')}})
    return res


# ------------------------------------------------------------------ reporting
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 2) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def summarise(rows: list[dict], skipped: dict, out_dir: str, mode: str, missions: list[dict]):
    providers = sorted({r['provider'] for r in rows}, key=lambda p: (p != 'ollama', p))
    summary = []
    for p in providers:
        rs = [r for r in rows if r['provider'] == p]
        s = {
            'provider': p, 'model': rs[0]['model'], 'runs': len(rs),
            'pass_rate': round(sum(r['pass'] for r in rs) / len(rs), 3),
            'safe_rate': (round(sum(bool(r['safe']) for r in rs) / len(rs), 3) if mode == 'sim' else None),
            **{f'pass_{t}': (f"{sum(r['pass'] for r in rs if r['tier'] == t)}/"
                             f"{sum(1 for r in rs if r['tier'] == t)}") for t in TIERS},
            'replans_total': sum(r['replans'] or 0 for r in rs),
            'unsafe_plans_caught': sum(r['unsafe_plans_caught'] or 0 for r in rs),
            'validator_repairs': sum(r['validator_repairs'] or 0 for r in rs),
            'format_errors': sum(r['format_errors'] or 0 for r in rs),
            'llm_errors': sum(1 for r in rs if r['outcome'] == 'error'),
            'mean_llm_latency_s': _mean([r['llm_latency_s'] for r in rs]),
            'median_llm_latency_s': _median([r['llm_latency_s'] for r in rs]),
            'mean_tokens_in': _mean([r['tokens_in'] for r in rs]),
            'mean_tokens_out': _mean([r['tokens_out'] for r in rs]),
            'total_cost_usd': round(sum(r['cost_usd'] or 0 for r in rs), 4),
            'total_list_cost_usd': round(sum(r['list_cost_usd'] or 0 for r in rs), 4),
            'mean_wall_s': _mean([r['wall_s'] for r in rs]),
        }
        summary.append(s)
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader()
            w.writerows(summary)
    with open(os.path.join(out_dir, 'runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({**r, 'failed_checks': '; '.join(r.get('failed_checks') or [])})

    md = [f'# Evaluation results ({mode} mode)', '',
          f'Generated {time.strftime("%Y-%m-%d %H:%M")}; {len(missions)} missions; '
          f'git {rows[0].get("git", "?") if rows else "?"}.', '']
    if skipped:
        md += ['Skipped providers:', ''] + [f'- **{p}**: {why}' for p, why in skipped.items()] + ['']
    if summary:
        md += ['## Summary', '']
        cols = [('provider', 'Provider'), ('model', 'Model'), ('runs', 'Runs'), ('pass_rate', 'Pass rate')]
        if mode == 'sim':
            cols.append(('safe_rate', 'Safe rate'))
        cols += [(f'pass_{t}', t.capitalize()) for t in TIERS]
        cols += [('replans_total', 'Replans'), ('unsafe_plans_caught', 'Unsafe plans caught'),
                 ('validator_repairs', 'Repairs'), ('format_errors', 'Format errors'),
                 ('median_llm_latency_s', 'Median LLM s'), ('mean_tokens_in', 'Tokens in'),
                 ('mean_tokens_out', 'Tokens out'), ('total_cost_usd', 'Cost $'),
                 ('total_list_cost_usd', 'List-price $')]
        md.append('| ' + ' | '.join(c[1] for c in cols) + ' |')
        md.append('|' + '---|' * len(cols))
        for s in summary:
            md.append('| ' + ' | '.join(str(s[c[0]]) for c in cols) + ' |')
        md += ['', '## Per mission', '']
        md.append('| Mission | Tier | Prompt | ' + ' | '.join(providers) + ' |')
        md.append('|---|---|---|' + '---|' * len(providers))
        for m in missions:
            cells = []
            for p in providers:
                rs = [r for r in rows if r['provider'] == p and r['mission'] == m['id']]
                if not rs:
                    cells.append('-')
                    continue
                ok = sum(r['pass'] for r in rs)
                outs = ','.join(sorted({r['outcome'] for r in rs}))
                rep = sum(r['replans'] or 0 for r in rs)
                cells.append(f'{"✅" if ok == len(rs) else ("❌" if ok == 0 else "◐")} {ok}/{len(rs)} {outs}'
                             + (f' (replans {rep})' if rep else ''))
            md.append(f'| {m["id"]} | {m["tier"]} | {m["prompt"]} | ' + ' | '.join(cells) + ' |')
        fails = [r for r in rows if not r['pass']]
        if fails:
            md += ['', '## Failures', '']
            for r in fails:
                md.append(f'- **{r["provider"]} {r["mission"]}** ({r["outcome"]}): '
                          f'{"; ".join(r.get("failed_checks") or [])} {("- " + r["reason"]) if r.get("reason") else ""}')
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    return summary


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['plan', 'sim'], default='plan')
    ap.add_argument('--providers', nargs='+', default=['ollama', 'groq'])
    ap.add_argument('--missions', default='', help='comma-separated mission ids')
    ap.add_argument('--tiers', default='', help='comma-separated tiers')
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--rtf', type=float, default=2.0, help='sim real-time factor (sim mode)')
    ap.add_argument('--timeout', type=int, default=900, help='wall timeout per mission (sim mode)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--resume', action='store_true', help='skip runs already in OUT/runs.jsonl')
    ap.add_argument('--missions-file', default=os.path.join(WS, 'eval', 'missions.yaml'))
    a = ap.parse_args(argv)

    import yaml
    load_dotenv(os.path.join(WS, '.env'))
    cfg, llm_cfg = load_config(), load_llm_config()
    missions = yaml.safe_load(open(a.missions_file))['missions']
    if a.missions:
        keep = set(a.missions.split(','))
        missions = [m for m in missions if m['id'] in keep]
    if a.tiers:
        keep = set(a.tiers.split(','))
        missions = [m for m in missions if m['tier'] in keep]
    out = a.out or os.path.join(WS, 'eval', 'results', f'{time.strftime("%Y%m%d_%H%M")}_{a.mode}')
    os.makedirs(out, exist_ok=True)
    runs_path = os.path.join(out, 'runs.jsonl')
    rows = [json.loads(l) for l in open(runs_path)] if (a.resume and os.path.isfile(runs_path)) else []
    done = {(r['provider'], r['mission'], r['repeat']) for r in rows}

    skipped = {}
    from auv_mission_planner.tracing import git_commit
    for provider in a.providers:
        ok, why = provider_status(llm_cfg, provider)
        if not ok:
            skipped[provider] = why
            print(f'[eval] skipping {provider}: {why}', flush=True)
            continue
        model = llm_cfg['providers'][provider].get('model', provider)
        print(f'[eval] {provider} ({model}) - {len(missions)} missions x {a.repeats}, mode={a.mode}', flush=True)
        warm_up(llm_cfg, provider)
        for rep in range(a.repeats):
            for m in missions:
                if (provider, m['id'], rep) in done:
                    continue
                run_dir = os.path.join(out, 'raw', provider, f'{m["id"]}_r{rep}')
                os.makedirs(run_dir, exist_ok=True)
                t0 = time.time()
                if a.mode == 'plan':
                    res = run_plan_mode(m, provider, copy.deepcopy(cfg), llm_cfg, run_dir)
                else:
                    res = run_sim_mode(m, provider, a.rtf, a.timeout, run_dir, cfg['limits'])
                row = {k: res.get(k) for k in RUN_FIELDS if k in res}
                row.update(provider=provider, model=model, mission=m['id'], tier=m['tier'], repeat=rep,
                           mode=a.mode, git=git_commit(), referee=res.get('referee'),
                           wall_s=res.get('wall_s') or round(time.time() - t0, 1))
                rows.append(row)
                with open(runs_path, 'a') as f:
                    f.write(json.dumps(row, default=str) + '\n')
                print(f'[eval] {provider:8s} {m["id"]:3s} {"PASS" if row["pass"] else "FAIL"} '
                      f'{row.get("outcome", "?"):10s} replans={row.get("replans")} '
                      f'unsafe_caught={row.get("unsafe_plans_caught")} llm={row.get("llm_latency_s")}s '
                      f'{"; ".join(row.get("failed_checks") or [])}', flush=True)
    summary = summarise(rows, skipped, out, a.mode, missions)
    for s in summary:
        print(f'[eval] {s["provider"]}: pass {s["pass_rate"]:.0%}'
              + (f', safe {s["safe_rate"]:.0%}' if s['safe_rate'] is not None else '')
              + f', unsafe caught {s["unsafe_plans_caught"]}, median LLM {s["median_llm_latency_s"]}s', flush=True)
    print(f'[eval] results: {out}/summary.md', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
