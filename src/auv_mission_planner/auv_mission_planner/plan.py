"""Structured mission plan produced by the LLM (Pydantic). No ROS imports.

Parsing checks STRUCTURE only. Whether a plan is allowed (tools, depths, battery, time) is the
safety validator's job, so unsafe-but-well-formed plans are caught, logged and counted there.
Conditions are flat objects with a `type` enum - easier for small models and for constrained
JSON decoding than nested unions.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SuccessType = Literal['tool_ok', 'reached', 'detected', 'depth_within', 'surfaced', 'battery_above']
AbortType = Literal['battery_below', 'mission_time_exceeded']
OnFailure = Literal['replan', 'abort', 'continue']


class _M(BaseModel):
    model_config = ConfigDict(extra='forbid')


class SuccessCondition(_M):
    type: SuccessType = 'tool_ok'
    target: Optional[str] = None          # detected
    depth: Optional[float] = None         # depth_within
    tolerance_m: Optional[float] = None   # reached / depth_within
    percent: Optional[float] = None       # battery_above

    @model_validator(mode='after')
    def _fields(self):
        need = {'detected': 'target', 'depth_within': 'depth', 'battery_above': 'percent'}.get(self.type)
        if need and getattr(self, need) is None:
            raise ValueError(f'success type {self.type!r} needs field {need!r}')
        return self


class AbortCondition(_M):
    type: AbortType
    percent: Optional[float] = None       # battery_below
    seconds: Optional[float] = None       # mission_time_exceeded

    @model_validator(mode='after')
    def _fields(self):
        need = {'battery_below': 'percent', 'mission_time_exceeded': 'seconds'}[self.type]
        if getattr(self, need) is None:
            raise ValueError(f'abort type {self.type!r} needs field {need!r}')
        return self


class Step(_M):
    id: int
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    success: SuccessCondition = Field(default_factory=SuccessCondition)
    timeout_s: float = 60.0
    on_failure: OnFailure = 'replan'
    note: str = ''


class MissionPlan(_M):
    summary: str = ''
    abort_conditions: list[AbortCondition] = Field(default_factory=list)
    steps: list[Step]


class PlannerOutput(_M):
    """Top-level LLM answer: a plan, or a refusal for unsafe / impossible requests."""
    status: Literal['plan', 'refuse']
    reason: str = ''
    plan: Optional[MissionPlan] = None

    @model_validator(mode='after')
    def _consistent(self):
        if self.status == 'plan' and (self.plan is None or not self.plan.steps):
            raise ValueError('status "plan" requires a plan with at least one step')
        if self.status == 'refuse' and not self.reason:
            raise ValueError('status "refuse" requires a reason')
        return self


def output_json_schema() -> dict:
    """JSON schema handed to providers that support constrained decoding (e.g. Ollama `format`)."""
    return PlannerOutput.model_json_schema()
