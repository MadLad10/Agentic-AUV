"""LLM planner: prompt -> validated-structure MissionPlan (or refusal).

Every LLM call is recorded in PlanResult.calls (messages, raw text, usage, parse error) for tracing.
Structure errors are fed back to the model up to max_format_retries times.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import ValidationError

from auv_mission_planner import prompts
from auv_mission_planner.llm.base import LLMError, LLMProvider
from auv_mission_planner.plan import MissionPlan, PlannerOutput, output_json_schema


@dataclass
class PlanResult:
    status: str                        # plan | refuse | error
    plan: Optional[MissionPlan] = None
    reason: str = ''
    calls: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)   # conversation, so a repair can continue it

    @property
    def tokens(self) -> tuple[int, int]:
        return (sum(c['usage']['input_tokens'] for c in self.calls if c.get('usage')),
                sum(c['usage']['output_tokens'] for c in self.calls if c.get('usage')))


def extract_json(text: str) -> Any:
    """Parse the first JSON object in text (tolerates ``` fences and prose around it)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                esc = (c == '\\') and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find('{', start + 1)
    raise ValueError('no JSON object found in the output')


def parse_output(text: str) -> PlannerOutput:
    data = extract_json(text)
    if isinstance(data, dict) and 'status' not in data:
        if 'steps' in data:                              # bare plan
            data = {'status': 'plan', 'reason': '', 'plan': data}
        elif 'plan' in data:
            data = {'status': 'plan', **data}
    return PlannerOutput.model_validate(data)


def _err(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return '; '.join(f'{".".join(map(str, x["loc"])) or "root"}: {x["msg"]}' for x in e.errors()[:6])
    return str(e)


class Planner:
    def __init__(self, provider: LLMProvider, tool_specs: list[dict], limits: dict, max_format_retries=2):
        self.provider = provider
        self.limits = limits
        self.system = prompts.system_prompt(tool_specs, limits)
        self.schema = output_json_schema()
        self.max_format_retries = max_format_retries

    # ------------------------------------------------------------ entry points
    def plan(self, prompt: str, context: dict) -> PlanResult:
        msgs = [{'role': 'system', 'content': self.system},
                {'role': 'user', 'content': prompts.plan_request(prompt, context)}]
        return self._ask(msgs, kind='plan')

    def repair(self, previous: PlanResult, reasons: list[str]) -> PlanResult:
        """Continue the conversation after the safety validator rejected the plan."""
        msgs = previous.messages + [{'role': 'user', 'content': prompts.validation_retry(reasons)}]
        return self._ask(msgs, kind='validation_repair')

    def replan(self, prompt: str, done: list[dict], failed: dict, context: dict, attempt: int,
               max_replans: int) -> PlanResult:
        msgs = [{'role': 'system', 'content': self.system},
                {'role': 'user', 'content': prompts.replan_request(prompt, done, failed, context, attempt,
                                                                   max_replans)}]
        return self._ask(msgs, kind='replan')

    # ------------------------------------------------------------ core loop
    def _ask(self, msgs: list[dict], kind: str) -> PlanResult:
        calls = []
        msgs = list(msgs)
        for attempt in range(self.max_format_retries + 1):
            call = {'kind': kind if attempt == 0 else f'{kind}_format_retry', 'provider': self.provider.name,
                    'model': self.provider.model, 'messages': [dict(m) for m in msgs], 't_wall': time.time()}
            try:
                resp = self.provider.complete(msgs, self.schema)
            except LLMError as e:
                call.update(error=str(e), usage=None)
                calls.append(call)
                return PlanResult('error', reason=f'LLM call failed: {e}', calls=calls, messages=msgs)
            call.update(response=resp.text, usage=resp.usage.as_dict())
            msgs.append({'role': 'assistant', 'content': resp.text})
            try:
                out = parse_output(resp.text)
            except (ValueError, ValidationError) as e:
                call['parse_error'] = _err(e)
                calls.append(call)
                msgs.append({'role': 'user', 'content': prompts.format_retry(_err(e))})
                continue
            calls.append(call)
            if out.status == 'refuse':
                return PlanResult('refuse', reason=out.reason, calls=calls, messages=msgs)
            return PlanResult('plan', plan=out.plan, reason=out.reason, calls=calls, messages=msgs)
        return PlanResult('error', reason=f'no valid plan after {self.max_format_retries + 1} attempts: '
                                          f'{calls[-1].get("parse_error")}', calls=calls, messages=msgs)
