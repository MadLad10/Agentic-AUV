#!/usr/bin/env python3
"""Print a mission trace as a readable timeline.

  scripts/show_trace.py traces/trace_<id>.json [--messages]
  scripts/show_trace.py --latest [dir]
"""
import argparse
import glob
import json
import os
import sys


def short(obj, n=110):
    s = json.dumps(obj, separators=(',', ':'), default=str) if not isinstance(obj, str) else obj
    return s if len(s) <= n else s[:n - 3] + '...'


def show(path, messages=False):
    t = json.load(open(path))
    out = t.get('outcome') or {}
    tot = t.get('totals') or {}
    print(f"mission  {t['mission_id']}  ({t.get('schema_version', '?')}, git {t.get('git_commit', '?')})")
    print(f"prompt   {t['prompt']!r}")
    print(f"provider {t['provider'].get('provider')} / {t['provider'].get('model')}")
    print(f"outcome  {out.get('outcome', '?').upper()}  {out.get('reason') or ''}")
    if tot:
        print(f"totals   llm_calls={tot.get('llm_calls')} tokens={tot.get('tokens_in')}+{tot.get('tokens_out')} "
              f"llm={tot.get('llm_latency_s')}s cost=${tot.get('cost_usd')} (list ${tot.get('list_cost_usd')}) "
              f"replans={tot.get('replans')} unsafe_caught={tot.get('unsafe_plans_caught')} "
              f"wall={t.get('wall_s')}s")
    print('-' * 100)
    rows = []
    for c in t.get('llm_calls', []):
        u = c.get('usage') or {}
        line = f"LLM  {c['kind']:<28} {u.get('input_tokens', 0):>5}+{u.get('output_tokens', 0):<4} tok " \
               f"{u.get('latency_s', 0):6.1f}s"
        if c.get('parse_error'):
            line += f"  PARSE ERROR: {short(c['parse_error'], 60)}"
        if c.get('error'):
            line += f"  ERROR: {short(c['error'], 60)}"
        rows.append((c.get('t', 0), line))
        if messages:
            for m in c.get('messages', []):
                text = m.get('content') if 'content' in m else f"<system prompt {m.get('content_ref')}>"
                rows.append((c.get('t', 0), f"       {m['role']:>9}: {short(text, 90)}"))
            rows.append((c.get('t', 0), f"       {'response':>9}: {short(c.get('response', ''), 90)}"))
    for p in t.get('plans', []):
        steps = (p.get('plan') or {}).get('steps', [])
        desc = ' -> '.join(s['tool'] + (f"({s['params']['target']})" if s['params'].get('target') else '')
                           for s in steps)
        rows.append((p['t'], f"PLAN #{p['index']} {p['kind']:<18} {p['status']:<7} {desc or p.get('reason', '')}"))
    for v in t.get('validations', []):
        tag = 'OK' if v['ok'] else 'REJECTED'
        extra = '; '.join(f"[{f['rule']}] {f['message']}" for f in v['rejected'] + v['repaired'])
        rows.append((v['t'], f"VAL  plan #{v['plan_index']} {tag}  {short(extra, 80)}"))
    for r in t.get('tool_calls', []):
        obs = r.get('observation') or {}
        rows.append((r['t'], f"TOOL step {r['step']['id']:<2} {r['step']['tool']:<15} "
                             f"{'ok  ' if r['success'] else 'FAIL'} sim {r['sim_end'] - r['sim_start']:6.1f}s  "
                             f"{short(r['step']['params'], 45)}  {obs.get('error_code') or ''} {r.get('detail', '')}"))
    for r in t.get('replans', []):
        rows.append((r['t'], f"REPLAN #{r['attempt']} after step {r['failed']['step']['id']} "
                             f"({r['failed']['step']['tool']})"))
    for e in t.get('events', []):
        if e['kind'] in ('safe_abort', 'abort'):
            rows.append((e['t'], f"ABORT {short(e.get('reason', ''), 80)}"))
    for ts, line in sorted(rows, key=lambda r: r[0]):
        print(f'{ts:8.1f}s  {line}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?')
    ap.add_argument('--latest', action='store_true')
    ap.add_argument('--messages', action='store_true', help='also print LLM messages/responses')
    a = ap.parse_args()
    path = a.path
    if a.latest:
        d = a.path or os.path.join(os.path.dirname(__file__), '..', 'traces')
        files = sorted(glob.glob(os.path.join(d, '**', 'trace_*.json'), recursive=True), key=os.path.getmtime)
        if not files:
            sys.exit(f'no traces in {d}')
        path = files[-1]
    if not path:
        ap.error('give a trace path or --latest')
    show(path, a.messages)


if __name__ == '__main__':
    main()
