"""Executor: runs a VALIDATED plan step by step through the tools layer.

- Before each step: validator.precheck_step (battery, remaining mission time).
- Each step: one tool call (ROS action under the hood) bounded by the step timeout.
- After each step: the step's success condition is evaluated from the observation + vehicle state.
- A watchdog thread enforces the plan's abort conditions (battery_below, mission_time_exceeded) and
  the wall-clock mission deadline; when it trips it cancels the running action.
- Failures return to the caller (MissionRunner) which decides to replan or abort safely.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from auv_mission_planner.plan import MissionPlan, Step
from auv_mission_planner.tools.registry import ToolRegistry
from auv_mission_planner.tools.schemas import Observation
from auv_mission_planner.validator import SafetyValidator

OBSERVE_TOOLS = {'get_detections', 'get_battery', 'get_pose'}


def _fmt(pose):
    return 'unknown' if pose is None else f'{pose.depth:.2f}'


@dataclass
class StepRecord:
    plan_index: int                 # which plan (0 = initial, k = k-th replan)
    step: dict
    observation: Optional[dict]
    success: bool
    detail: str = ''
    sim_start: float = 0.0
    sim_end: float = 0.0
    wall_s: float = 0.0

    def brief(self) -> dict:
        obs = self.observation or {}
        return {'id': self.step['id'], 'tool': self.step['tool'], 'params': self.step['params'],
                'ok': self.success, 'detail': self.detail,
                'observation': {k: obs.get(k) for k in ('ok', 'error_code', 'error', 'data') if obs.get(k)}}


@dataclass
class ExecResult:
    status: str                     # completed | step_failed | abort
    records: list[StepRecord] = field(default_factory=list)
    failed_step: Optional[Step] = None
    failed_record: Optional[StepRecord] = None
    abort_reason: str = ''


class Watchdog:
    def __init__(self, registry: ToolRegistry, period_s: float, wall_deadline: Optional[float]):
        self.registry = registry
        self.period = period_s
        self.wall_deadline = wall_deadline
        self.plan: Optional[MissionPlan] = None
        self.mission_t0 = registry.backend.now()
        self.reason: Optional[str] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def elapsed(self) -> float:
        return self.registry.backend.now() - self.mission_t0

    def check(self) -> Optional[str]:
        b = self.registry.backend.battery_percent()
        if self.plan is not None:
            for a in self.plan.abort_conditions:
                if a.type == 'battery_below' and b is not None and b < a.percent:
                    return f'battery {b:.1f}% below abort threshold {a.percent:.0f}%'
                if a.type == 'mission_time_exceeded' and self.elapsed() > a.seconds:
                    return f'mission time {self.elapsed():.0f}s exceeded limit {a.seconds:.0f}s'
        if self.wall_deadline is not None and time.time() > self.wall_deadline:
            return 'mission wall-clock timeout'
        return None

    def _loop(self):
        while not self._stop.wait(self.period):
            if self.reason is None:
                r = self.check()
                if r:
                    self.reason = r
                    self.registry.backend.cancel_all()      # interrupt the running action


class Executor:
    def __init__(self, registry: ToolRegistry, validator: SafetyValidator, cfg: dict,
                 on_event: Optional[Callable[[str, dict], None]] = None):
        self.tools = registry
        self.validator = validator
        self.cfg = cfg.get('executor', {})
        self.on_event = on_event or (lambda kind, data: None)

    # ------------------------------------------------------------ success conditions
    def evaluate(self, step: Step, obs: Observation) -> tuple[bool, str]:
        c = step.success
        backend = self.tools.backend
        action_ok = obs.ok or step.tool in OBSERVE_TOOLS
        if c.type == 'tool_ok':
            return obs.ok, 'tool reported ok' if obs.ok else f'tool failed: {obs.error_code}'
        if c.type == 'reached':
            err = obs.data.get('final_error_m')
            tol = c.tolerance_m or 0.5
            ok = obs.ok and (err is None or err <= tol)
            return ok, f'final error {err} m (tolerance {tol})'
        if c.type == 'detected':
            vis = {d['class'] for d in obs.data.get('visible', [])}
            det = obs.data.get('detection') or {}
            seen = c.target in vis or det.get('class') == c.target or \
                self.tools.world.lookup(c.target, backend.now()) is not None
            return action_ok and seen, f'{c.target} {"detected" if seen else "not detected"}'
        if c.type == 'depth_within':
            pose = backend.pose()
            tol = c.tolerance_m or 0.4
            ok = pose is not None and abs(pose.depth - c.depth) <= tol
            return action_ok and ok, f'depth {_fmt(pose)} vs {c.depth} ± {tol}'
        if c.type == 'surfaced':
            pose = backend.pose()
            ok = pose is not None and pose.depth < 0.3
            return action_ok and ok, f'depth {_fmt(pose)}'
        if c.type == 'battery_above':
            b = backend.battery_percent()
            ok = b is not None and b >= c.percent
            return action_ok and ok, f'battery {b}% vs {c.percent}%'
        return False, f'unknown condition {c.type}'

    # ------------------------------------------------------------ run
    def run(self, plan: MissionPlan, watchdog: Watchdog, plan_index: int = 0) -> ExecResult:
        watchdog.plan = plan
        res = ExecResult('completed')
        for step in plan.steps:
            if watchdog.reason:
                res.status, res.abort_reason = 'abort', watchdog.reason
                return res
            why = self.validator.precheck_step(step, self.tools.backend.battery_percent(),
                                               watchdog.elapsed(), plan)
            if why:
                res.status, res.abort_reason = 'abort', f'precheck before step {step.id}: {why}'
                return res
            self.on_event('step_start', {'plan_index': plan_index, 'step': step.model_dump()})
            t0_sim, t0 = self.tools.backend.now(), time.time()
            obs = self.tools.call(step.tool, step.params, step.timeout_s)
            ok, detail = self.evaluate(step, obs)
            rec = StepRecord(plan_index, step.model_dump(), obs.model_dump(), ok, detail, round(t0_sim, 2),
                             round(self.tools.backend.now(), 2), round(time.time() - t0, 2))
            res.records.append(rec)
            self.on_event('step_end', {'plan_index': plan_index, 'record': rec.__dict__})
            # synchronous check too: a threshold crossed during the step must stop the next one,
            # even if the watchdog thread has not polled yet
            watchdog.reason = watchdog.reason or watchdog.check()
            if watchdog.reason:
                res.status, res.abort_reason = 'abort', watchdog.reason
                return res
            if step.tool == 'abort' and obs.ok:
                res.status, res.abort_reason = 'abort', f'plan requested abort: {step.params.get("reason", "")}'
                return res
            if ok or step.on_failure == 'continue':
                continue
            if step.on_failure == 'abort':
                res.status, res.abort_reason = 'abort', f'step {step.id} ({step.tool}) failed: {detail}'
                return res
            res.status, res.failed_step, res.failed_record = 'step_failed', step, rec
            return res
        return res

    def safe_abort(self, reason: str) -> dict:
        """Cancel motion and surface. Never depends on the LLM."""
        self.on_event('abort', {'reason': reason})
        obs = self.tools.call('abort', {'reason': reason}, timeout_s=10)
        timeout = self.cfg.get('abort_surface_timeout_s', 90.0)
        t0 = self.tools.backend.now()
        w0 = time.time()
        surfaced = False
        since = None
        while self.tools.backend.now() - t0 < timeout and time.time() - w0 < 3 * timeout:
            pose = self.tools.backend.pose()
            if pose is not None and pose.depth < 0.3:      # must STAY up (it can still be rising/bobbing)
                since = since if since is not None else (self.tools.backend.now(), time.time())
                held = max(self.tools.backend.now() - since[0], time.time() - since[1])   # sim clock may be static
                if held >= 2.0:
                    surfaced = True
                    break
            else:
                since = None
            time.sleep(0.2)
        return {'reason': reason, 'abort_call': obs.model_dump(), 'surfaced': surfaced}
