"""Build and run the v2 multi-agent mission graph on the MuJoCo sim.

  python3 -m auv_agents.run "pass through the gate, touch the red buoy, then surface"
      [--provider ollama|groq|mock] [--approval auto|interrupt] [--fault TYPE:t[:k=v]] [--arena default]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

import yaml


def agents_config_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory('auv_agents'), 'config', 'agents.yaml')
        if os.path.isfile(p):
            return p
    except Exception:
        pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'agents.yaml')


def load_agents_config(overrides: dict | None = None) -> dict:
    from auv_marine_sim.world import deep_update
    with open(agents_config_path()) as f:
        return deep_update(yaml.safe_load(f), overrides or {})


def build_runtime(backend, provider, agents_cfg: dict, diag_provider=None, vlm=None, auto_approve=True, log=None):
    from auv_agents.graph import Runtime
    from auv_agents.specialists import DiagnosticsAgent
    from auv_mission_planner.config import load_config, load_llm_config
    from auv_mission_planner.planner import Planner
    from auv_mission_planner.tools.registry import ToolRegistry
    from auv_agents import telemetry
    cfg = load_config()
    llm_cfg = load_llm_config()
    reg = ToolRegistry(backend, cfg)
    if telemetry.enabled():                       # OTel spans for every LLM and tool call
        provider = telemetry.TracedProvider(provider, 'planner')
        diag_provider = telemetry.TracedProvider(diag_provider, 'diagnostics') if diag_provider else None
        reg = telemetry.TracedRegistry(reg)
    pcfg = llm_cfg.get('planner', {})
    planner = Planner(provider, reg.schemas(), cfg['limits'], pcfg.get('max_format_retries', 2))
    diag = DiagnosticsAgent(diag_provider, use_llm=agents_cfg['diagnostics'].get('use_llm', True))
    return Runtime(backend, reg, cfg, agents_cfg, planner, diag, vlm=vlm, auto_approve=auto_approve, log=log)


def run_mission(rt, prompt: str, thread_id: str | None = None, checkpointer=None, on_event=None,
                approve_fn=None) -> dict:
    """Stream the graph to completion. approve_fn(payload) -> resume value when the graph interrupts."""
    from langgraph.types import Command
    from auv_agents.graph import build_graph
    if checkpointer is None:                      # state inspection / HITL resume need one
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = rt.__dict__.setdefault('_memory_saver', InMemorySaver())
    graph = build_graph(rt, checkpointer)
    tid = thread_id or uuid.uuid4().hex[:12]
    config = {'configurable': {'thread_id': tid}, 'recursion_limit': 200}
    inp = {'prompt': prompt}
    snap = graph.get_state(config)
    pending = [i for t in (snap.tasks or ()) for i in (t.interrupts or ())] if snap.next else []
    if pending:                                   # a paused thread (HITL): resume it from the checkpoint
        if approve_fn is None:
            out = dict(snap.values)
            out.update(outcome='awaiting_approval', thread_id=tid)
            return out
        inp = Command(resume=approve_fn(pending[0].value))
    final = {}
    t0 = time.time()
    from auv_agents import telemetry
    with telemetry.span('mission', **{'auv.prompt': prompt, 'auv.thread_id': tid}) as mspan:
        final = _stream(graph, inp, config, on_event, approve_fn, tid)
        if mspan is not None:
            telemetry._set(mspan, 'auv.outcome', final.get('outcome'))
            telemetry._set(mspan, 'auv.reason', final.get('reason'))
    if final.get('outcome') != 'awaiting_approval':
        final['wall_s'] = round(time.time() - t0, 1)
    return final


def _stream(graph, inp, config, on_event, approve_fn, tid):
    from langgraph.types import Command
    while True:
        interrupted = None
        for chunk in graph.stream(inp, config, stream_mode='updates'):
            for node, upd in chunk.items():
                if node == '__interrupt__':
                    interrupted = upd[0].value if upd else {}
                    continue
                for ev in (upd or {}).get('events', []) if isinstance(upd, dict) else []:
                    if on_event:
                        on_event(node, ev)
        if interrupted is None:
            break
        if approve_fn is None:
            final = dict(graph.get_state(config).values)
            final['outcome'] = 'awaiting_approval'
            final['thread_id'] = tid
            return final
        inp = Command(resume=approve_fn(interrupted))
    final = dict(graph.get_state(config).values)
    final['thread_id'] = tid
    return final


def _parse_fault(s):
    parts = s.split(':')
    f = {'type': parts[0], 't_start': float(parts[1]) if len(parts) > 1 else 0.0, 'params': {}}
    for kv in (parts[2].split(',') if len(parts) > 2 else []):
        k, v = kv.split('=')
        try:
            v = float(v)
        except ValueError:
            pass
        f['params'][k] = v
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prompt')
    ap.add_argument('--provider', default=None)
    ap.add_argument('--approval', default=None, choices=['auto', 'interrupt'])
    ap.add_argument('--fault', action='append', default=[])
    ap.add_argument('--arena', default='default')
    ap.add_argument('--nav', default='ekf')
    ap.add_argument('--detector', default='ground_truth')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    from auv_agents import telemetry
    telemetry.maybe_init_from_env()
    from auv_mission_planner.config import load_llm_config
    from auv_mission_planner.llm.providers import make_provider
    from auv_onboard.factory import make_backend
    from auv_agents.vlm import make_vlm
    acfg = load_agents_config({'graph': {'approval': a.approval}} if a.approval else None)
    llm_cfg = load_llm_config()
    prov = make_provider(llm_cfg, a.provider or acfg['supervisor']['provider'])
    diag = make_provider(llm_cfg, a.provider or acfg['diagnostics']['provider'])
    be = make_backend(a.arena, faults=[_parse_fault(f) for f in a.fault], seed=a.seed,
                      onboard_overrides={'backend': {'nav_source': a.nav, 'detection_source': a.detector}})
    rt = build_runtime(be, prov, acfg, diag, make_vlm(acfg['perception']['vlm']),
                       auto_approve=acfg['graph']['approval'] == 'auto')

    def on_event(node, ev):
        print(f'[{node:12s}] {ev["agent"]:16s} {ev["msg"]}', flush=True)

    def approve(payload):
        print(json.dumps(payload['plan'], indent=1))
        ans = input('approve this plan? [y/N] ').strip().lower()
        return {'decision': 'approve' if ans == 'y' else 'reject'}
    try:
        res = run_mission(rt, a.prompt, on_event=on_event, approve_fn=approve)
        rep = be.sim.referee.report()
        rep.pop('events')
        print(json.dumps({'outcome': res.get('outcome'), 'reason': res.get('reason'), 'stats': res.get('stats'),
                          'referee': rep}, indent=1, default=str))
    finally:
        be.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
