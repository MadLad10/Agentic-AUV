"""Mission tracing: one JSON file per mission with everything needed to audit, debug or replay it.

Contents (schema `auv-trace/1`, validated by `validate_trace`):
  prompt, provider/model, config snapshot (limits, executor, backend), git commit, environment
  llm_calls    every LLM request: kind (plan / *_format_retry / validation_repair / replan),
               messages (system prompt stored once under `prompts` and referenced by hash),
               raw response, parse error, tokens in/out, latency, cost (actual + list price)
  plans        every plan / refusal the LLM produced, with its index and kind
  validations  validator report per plan (rejected + repaired findings)
  tool_calls   every executed step: params, observation, success-condition detail, sim/wall time
  replans      failure that triggered each replan + the vehicle state sent to the LLM
  events       context snapshots, aborts
  outcome      final result (same dict the CLI / eval report) and `totals` (aggregates)
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = 'auv-trace/1'
_GIT = None


def new_mission_id() -> str:
    return time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]


def git_commit() -> str:
    global _GIT
    if _GIT is None:
        try:
            here = os.path.dirname(os.path.realpath(__file__))
            _GIT = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=here, capture_output=True,
                                  text=True, timeout=3).stdout.strip() or 'unknown'
        except Exception:
            _GIT = 'unknown'
    return _GIT


class MissionTrace:
    def __init__(self, prompt: str, provider: dict, out_dir: Optional[str] = None, mission_id: Optional[str] = None,
                 config: Optional[dict] = None, env: Optional[dict] = None):
        self.mission_id = mission_id or new_mission_id()
        self.out_dir = out_dir
        self.t0 = time.time()
        self.data: dict[str, Any] = {
            'schema_version': SCHEMA_VERSION,
            'mission_id': self.mission_id,
            'prompt': prompt,
            'provider': provider,
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'git_commit': git_commit(),
            'config': config or {},
            'env': {'host': platform.node(), 'python': platform.python_version(), **(env or {})},
            'prompts': {},
            'llm_calls': [],
            'plans': [],
            'validations': [],
            'tool_calls': [],
            'replans': [],
            'events': [],
            'outcome': None,
            'totals': None,
        }

    def _t(self):
        return round(time.time() - self.t0, 3)

    def event(self, kind: str, **data):
        self.data['events'].append({'t': self._t(), 'kind': kind, **data})

    def _dedupe(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m.get('role') == 'system' and isinstance(m.get('content'), str):
                h = hashlib.sha256(m['content'].encode()).hexdigest()[:12]
                self.data['prompts'].setdefault(h, m['content'])
                out.append({'role': 'system', 'content_ref': h})
            else:
                out.append(m)
        return out

    def llm_calls(self, calls: list[dict]):
        for c in calls:
            c = dict(c)
            c['t'] = round(c.pop('t_wall', time.time()) - self.t0, 3)
            c['messages'] = self._dedupe(c.get('messages', []))
            self.data['llm_calls'].append(c)

    def plan(self, plan_index: int, kind: str, plan: Optional[dict], status: str, reason: str = ''):
        self.data['plans'].append({'t': self._t(), 'index': plan_index, 'kind': kind, 'status': status,
                                   'reason': reason, 'plan': plan})

    def validation(self, plan_index: int, report: dict):
        self.data['validations'].append({'t': self._t(), 'plan_index': plan_index, **report})

    def tool_call(self, record: dict):
        self.data['tool_calls'].append({'t': self._t(), **record})

    def totals(self) -> dict:
        calls = self.data['llm_calls']
        u = [c['usage'] for c in calls if c.get('usage')]
        kinds: dict[str, int] = {}
        for c in calls:
            kinds[c['kind']] = kinds.get(c['kind'], 0) + 1
        out = self.data.get('outcome') or {}
        return {
            'llm_calls': len(calls),
            'llm_calls_by_kind': kinds,
            'llm_errors': sum(1 for c in calls if c.get('error')),
            'format_errors': sum(1 for c in calls if c.get('parse_error')),
            'tokens_in': sum(x['input_tokens'] for x in u),
            'tokens_out': sum(x['output_tokens'] for x in u),
            'llm_latency_s': round(sum(x['latency_s'] for x in u), 2),
            'cost_usd': round(sum(x['cost_usd'] for x in u), 6),
            'list_cost_usd': round(sum(x['list_cost_usd'] for x in u), 6),
            'plans': len(self.data['plans']),
            'replans': len(self.data['replans']),
            'unsafe_plans_caught': sum(1 for v in self.data['validations'] if not v['ok']),
            'validator_repairs': sum(len(v['repaired']) for v in self.data['validations']),
            'tool_calls': len(self.data['tool_calls']),
            'tool_failures': sum(1 for r in self.data['tool_calls'] if not r.get('success')),
            'sim_time_s': out.get('sim_time_s'),
        }

    def finish(self, outcome: dict) -> dict:
        self.data['outcome'] = outcome
        self.data['wall_s'] = self._t()
        self.data['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        self.data['totals'] = self.totals()
        return self.data

    def save(self) -> Optional[str]:
        if not self.out_dir:
            return None
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f'trace_{self.mission_id}.json')
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
        return path


# ------------------------------------------------------------------ schema (for validation / tooling)
class _Loose(BaseModel):
    model_config = ConfigDict(extra='allow')


class _Usage(_Loose):
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    list_cost_usd: float


class _LLMCall(_Loose):
    kind: str
    provider: str
    model: str
    t: float
    messages: list[dict]
    usage: Optional[_Usage] = None
    response: Optional[str] = None
    parse_error: Optional[str] = None
    error: Optional[str] = None


class _Plan(_Loose):
    t: float
    index: int
    kind: str
    status: str
    plan: Optional[dict] = None


class _Validation(_Loose):
    t: float
    plan_index: int
    ok: bool
    rejected: list[dict]
    repaired: list[dict]


class _ToolCall(_Loose):
    t: float
    plan_index: int
    step: dict
    observation: Optional[dict]
    success: bool
    sim_start: float
    sim_end: float
    wall_s: float


class _Outcome(_Loose):
    mission_id: str
    outcome: str
    reason: str
    replans: int
    unsafe_plans_caught: int
    tokens_in: int
    tokens_out: int
    llm_latency_s: float
    cost_usd: float


class TraceModel(_Loose):
    schema_version: str
    mission_id: str
    prompt: str
    provider: dict
    started_at: str
    finished_at: str
    git_commit: str
    config: dict
    env: dict
    prompts: dict[str, str]
    llm_calls: list[_LLMCall]
    plans: list[_Plan]
    validations: list[_Validation]
    tool_calls: list[_ToolCall]
    replans: list[dict]
    events: list[dict]
    outcome: _Outcome
    totals: dict
    wall_s: float


def validate_trace(data: dict) -> TraceModel:
    """Raise pydantic.ValidationError if the trace is malformed; also checks prompt references."""
    m = TraceModel.model_validate(data)
    for c in m.llm_calls:
        for msg in c.messages:
            ref = msg.get('content_ref')
            if ref is not None and ref not in m.prompts:
                raise ValueError(f'llm call references unknown prompt {ref}')
    return m
