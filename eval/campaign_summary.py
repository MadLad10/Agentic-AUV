"""Summarise a repeated-run evaluation campaign: pass@1 with Wilson 95% CIs, pass^k, ablation deltas.

Reads one or more runs.jsonl files written by run_eval_v2.py (rows carry `system`, `ablation`, `seed`).
An "arm" is (system, ablation). pass^k = share of missions that passed on every seed run for that arm
(tau-bench style), so it measures repeatability rather than average luck.

    python eval/campaign_summary.py eval/results/campaign_v23/runs.jsonl --out eval/results/campaign_v23
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

TIER_ORDER = ['easy', 'medium', 'complex', 'replan', 'impossible', 'unsafe', 'fault']


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n (good for small n and rates near 0 or 1)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def arm_of(r: dict) -> str:
    sysname = 'single_agent' if r.get('system') == 'v1' else 'full'
    abl = r.get('ablation') or 'none'
    return sysname if abl == 'none' else abl


def load(paths) -> list[dict]:
    rows = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                rows[(arm_of(r), r['mission'], r.get('seed', 0))] = r     # last write wins (resume reruns)
    return list(rows.values())


def rate(rows, key) -> tuple[int, int]:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return sum(bool(v) for v in vals), len(vals)


def fmt(k: int, n: int) -> str:
    if n == 0:
        return '–'
    lo, hi = wilson(k, n)
    return f'{k}/{n} ({100 * k / n:.0f}%, CI {100 * lo:.0f}–{100 * hi:.0f})'


def pass_hat_k(rows) -> tuple[int, int, int]:
    """(missions passing on all seeds, missions, min seeds per mission)."""
    by_m = defaultdict(list)
    for r in rows:
        by_m[r['mission']].append(bool(r.get('pass')))
    if not by_m:
        return 0, 0, 0
    return sum(all(v) for v in by_m.values()), len(by_m), min(len(v) for v in by_m.values())


def summarise(rows) -> str:
    arms = defaultdict(list)
    for r in rows:
        arms[arm_of(r)].append(r)
    out = ['# Evaluation campaign', '',
           'pass@1 = runs passed / runs (Wilson 95% CI); pass^k = missions that passed on every seed; '
           'scored by the ground-truth referee.', '',
           '## Arms', '',
           '| Arm | Runs | pass@1 | pass^k | safe | task done (fault tier) | fault detected | collisions |',
           '|---|---|---|---|---|---|---|---|']
    order = ['full', 'single_agent', 'no_validator', 'no_health', 'no_fusion']
    for arm in sorted(arms, key=lambda a: order.index(a) if a in order else 99):
        rs = arms[arm]
        pk, pm, k = pass_hat_k(rs)
        coll = sum((r.get('referee') or {}).get('collisions', 0) for r in rs)
        out.append(f'| {arm} | {len(rs)} | {fmt(*rate(rs, "pass"))} | {pk}/{pm} (k={k}) | {fmt(*rate(rs, "safe"))} | '
                   f'{fmt(*rate(rs, "task_done"))} | {fmt(*rate(rs, "fault_detected"))} | {coll} |')

    # ablations vs the full system on the SAME missions and seeds
    full = {(r['mission'], r.get('seed', 0)): r for r in arms.get('full', [])}
    out += ['', '## Ablations vs the full system (same missions and seeds)', '',
            '| Arm | Paired runs | pass: full → arm | safe: full → arm | task: full → arm |', '|---|---|---|---|---|']
    for arm in sorted(arms):
        if arm == 'full':
            continue
        pairs = [(full[(r['mission'], r.get('seed', 0))], r) for r in arms[arm] if (r['mission'], r.get('seed', 0)) in full]
        if not pairs:
            continue
        cells = []
        for key in ('pass', 'safe', 'task_done'):
            f = [p[0] for p in pairs if p[0].get(key) is not None and p[1].get(key) is not None]
            a = [p[1] for p in pairs if p[0].get(key) is not None and p[1].get(key) is not None]
            cells.append(f'{sum(bool(x[key]) for x in f)}/{len(f)} → {sum(bool(x[key]) for x in a)}/{len(a)}' if f else '–')
        out.append(f'| {arm} | {len(pairs)} | ' + ' | '.join(cells) + ' |')

    # full system per tier
    out += ['', '## Full system per tier', '', '| Tier | Runs | pass@1 | pass^k | safe |', '|---|---|---|---|---|']
    by_t = defaultdict(list)
    for r in arms.get('full', []):
        by_t[r['tier']].append(r)
    for t in sorted(by_t, key=lambda t: TIER_ORDER.index(t) if t in TIER_ORDER else 99):
        pk, pm, k = pass_hat_k(by_t[t])
        out.append(f'| {t} | {len(by_t[t])} | {fmt(*rate(by_t[t], "pass"))} | {pk}/{pm} (k={k}) | {fmt(*rate(by_t[t], "safe"))} |')

    # missions that are not repeatable
    flaky = defaultdict(list)
    for r in arms.get('full', []):
        flaky[r['mission']].append(r)
    bad = {m: rs for m, rs in flaky.items() if not all(r.get('pass') for r in rs)}
    if bad:
        out += ['', '## Full system: missions that failed on at least one seed', '',
                '| Mission | Passed | Failures (seed: outcome, failed checks) |', '|---|---|---|']
        for m in sorted(bad):
            rs = sorted(bad[m], key=lambda r: r.get('seed', 0))
            fails = '; '.join(f'{r.get("seed", 0)}: {r.get("outcome")} {", ".join(r.get("failed_checks", []))}'.strip()
                              for r in rs if not r.get('pass'))
            out.append(f'| {m} | {sum(bool(r.get("pass")) for r in rs)}/{len(rs)} | {fails} |')
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    text = summarise(load(a.runs))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, 'campaign_summary.md'), 'w') as f:
            f.write(text)
    print(text)


if __name__ == '__main__':
    main()
