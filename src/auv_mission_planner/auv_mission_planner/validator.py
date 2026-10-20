"""Safety validator: deterministic checks on every plan before anything executes. No ROS, no LLM.

Each rule either REJECTS the plan (unsafe or impossible; the LLM gets the reasons back) or REPAIRS
it (a conservative, intent-preserving fix such as clamping a timeout or appending `surface`). Every
finding is logged and returned in the ValidationReport, which goes into the mission trace.

Also provides precheck_step(): a runtime gate evaluated right before each step executes.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from pydantic import ValidationError

from auv_mission_planner.limits import point_violation
from auv_mission_planner.plan import AbortCondition, MissionPlan, Step, SuccessCondition
from auv_mission_planner.tools.schemas import TOOL_PARAMS

log = logging.getLogger('auv.validator')

MOTION_TOOLS = {'navigate_to', 'search_pattern', 'hold_depth'}
GENERIC_ALL = ('buoys', 'all buoys', 'every buoy', 'each buoy', 'all three', 'all objects', 'everything')


_DEPTH_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(?:m\b|meters?\b|metres?\b)', re.I)


def requested_depths(prompt: str) -> list[float]:
    """Depths the operator states explicitly ('dive to 50 meters', 'at 2 m depth').
    Distances like 'move 3 m forward' are not depths: only numbers near dive/depth/deep words count."""
    p = prompt.lower()
    out = []
    for m in _DEPTH_RE.finditer(p):
        window = p[max(0, m.start() - 25):m.end() + 15]
        if any(w in window for w in ('dive', 'depth', 'deep', 'descend', 'down to', 'go down')):
            out.append(float(m.group(1)))
    return out


OPTIONAL_PHRASES = ('if you can', 'if possible', 'optionally', 'optional', 'try to', 'if you see', 'if it is there',
                    'if available', 'best effort')


def _optional_wording(prompt: str) -> bool:
    """The operator explicitly made part of the mission optional ("touch the red buoy if you can")."""
    p = prompt.lower()
    return any(w in p for w in OPTIONAL_PHRASES)


def mentioned_objects(prompt: str, known: list[str]) -> set[str]:
    """Objects the operator's request refers to: by class name or by colour word; the gate always."""
    p = prompt.lower().replace('_', ' ')
    out = {'gate'}
    generic = any(g in p for g in GENERIC_ALL)
    for k in known:
        words = k.replace('_', ' ').split()
        if k.replace('_', ' ') in p or (len(words) == 2 and words[1] == 'buoy' and
                                         (words[0] in p.split() or words[0] + ',' in p or generic)):
            out.add(k)
    return out
DEFAULT_TIMEOUT_S = 60.0
# shortest timeout the validator will scale a step down to (sim seconds)
# shortest useful timeout per tool (sim seconds): shorter ones are raised, and over-budget plans are never
# scaled below these. A 60 s spiral covers nothing; the look-around spiral needs ~170 s to find an off-axis buoy.
MIN_STEP_TIMEOUT_S = {'navigate_to': 60.0, 'search_pattern': 180.0, 'surface': 40.0}


@dataclass
class Finding:
    rule: str
    action: str                 # reject | repair
    message: str
    step_id: Optional[int] = None

    def __str__(self):
        where = f'step {self.step_id}: ' if self.step_id is not None else ''
        return f'[{self.rule}] {where}{self.message}'


@dataclass
class ValidationReport:
    ok: bool
    plan: Optional[MissionPlan]                  # repaired copy (None if the input was None)
    rejected: list[Finding] = field(default_factory=list)
    repaired: list[Finding] = field(default_factory=list)

    def reasons(self) -> list[str]:
        return [str(f) for f in self.rejected]

    def as_dict(self) -> dict:
        return {'ok': self.ok, 'rejected': [asdict(f) for f in self.rejected],
                'repaired': [asdict(f) for f in self.repaired]}


class SafetyValidator:
    def __init__(self, limits: dict, repair: bool = True):
        self.L = limits
        self.allow_repair = repair

    # ------------------------------------------------------------ helpers
    def _reject(self, rep, rule, msg, sid=None):
        f = Finding(rule, 'reject', msg, sid)
        rep.rejected.append(f)
        log.warning('REJECT %s', f)

    def _repair(self, rep, rule, msg, sid=None) -> bool:
        """Record a repair; returns False (and rejects instead) when repairs are disabled."""
        if not self.allow_repair:
            self._reject(rep, rule, msg + ' (repair disabled)', sid)
            return False
        f = Finding(rule, 'repair', msg, sid)
        rep.repaired.append(f)
        log.info('REPAIR %s', f)
        return True

    # ------------------------------------------------------------ static validation
    def validate(self, plan: Optional[MissionPlan], prompt: Optional[str] = None) -> ValidationReport:
        rep = ValidationReport(ok=False, plan=None)
        if plan is None or not plan.steps:
            self._reject(rep, 'empty_plan', 'the plan has no steps')
            return rep
        plan = plan.model_copy(deep=True)
        rep.plan = plan
        L = self.L

        if len(plan.steps) > L['max_steps']:
            self._reject(rep, 'too_many_steps', f'{len(plan.steps)} steps > max {L["max_steps"]}')

        ids = [s.id for s in plan.steps]
        if len(set(ids)) != len(ids) or ids != sorted(ids):
            if self._repair(rep, 'step_ids', f'step ids {ids} renumbered 1..{len(ids)}'):
                for i, s in enumerate(plan.steps, 1):
                    s.id = i

        self._fill_detected_targets(rep, plan)
        if prompt is not None:
            self._check_intent(rep, plan, prompt)
        seen_abort = False
        for s in plan.steps:
            if seen_abort:
                self._reject(rep, 'steps_after_abort', f'{s.tool} comes after abort', s.id)
            if s.tool == 'abort':
                seen_abort = True
            self._check_step(rep, s)

        self._check_abort_conditions(rep, plan)

        last = plan.steps[-1].tool
        if last not in ('surface', 'abort'):
            if self._repair(rep, 'must_surface', f'plan ended with {last}; appended surface'):
                plan.steps.append(Step(id=plan.steps[-1].id + 1, tool='surface',
                                       success=SuccessCondition(type='surfaced'), timeout_s=60.0,
                                       on_failure='abort', note='added by safety validator'))

        total = sum(s.timeout_s for s in plan.steps)
        if total > L['max_mission_time_s']:
            self._fit_time_budget(rep, plan, total)

        rep.ok = not rep.rejected
        log.info('validation %s: %d rejected, %d repaired', 'OK' if rep.ok else 'FAILED',
                 len(rep.rejected), len(rep.repaired))
        return rep

    def _check_step(self, rep, s: Step):
        L = self.L
        if s.tool not in L['allowed_tools']:
            self._reject(rep, 'tool_not_allowed', f'tool {s.tool!r} is not allowed '
                                                  f'(allowed: {", ".join(L["allowed_tools"])})', s.id)
            return
        if s.tool == 'navigate_to' and s.params.get('pass_through') and s.params.get('target') not in (None, 'gate'):
            if self._repair(rep, 'pass_through', f'pass_through ignored for target {s.params["target"]!r} '
                                                 f'(only the gate is passed through)', s.id):
                s.params.pop('pass_through')
        try:
            p = TOOL_PARAMS[s.tool].model_validate(s.params)
        except ValidationError as e:
            msg = '; '.join(f'{".".join(map(str, x["loc"])) or "params"}: {x["msg"]}' for x in e.errors()[:3])
            self._reject(rep, 'invalid_params', f'{s.tool}: {msg}', s.id)
            return

        # timeouts
        if s.timeout_s <= 0:
            if self._repair(rep, 'step_timeout', f'timeout {s.timeout_s} -> {DEFAULT_TIMEOUT_S}', s.id):
                s.timeout_s = DEFAULT_TIMEOUT_S
        elif s.timeout_s < MIN_STEP_TIMEOUT_S.get(s.tool, 0.0):
            need = MIN_STEP_TIMEOUT_S[s.tool]
            if self._repair(rep, 'step_timeout', f'timeout {s.timeout_s:.0f}s too short for {s.tool}; '
                                                 f'raised to {need:.0f}s', s.id):
                s.timeout_s = need
        elif s.timeout_s > L['max_step_timeout_s']:
            if self._repair(rep, 'step_timeout', f'timeout {s.timeout_s:.0f}s clamped to '
                                                 f'{L["max_step_timeout_s"]:.0f}s', s.id):
                s.timeout_s = float(L['max_step_timeout_s'])

        # objects must exist
        target = getattr(p, 'target', None)
        if target is not None and target not in L['known_objects']:
            self._reject(rep, 'unknown_object', f'{target!r} does not exist in this arena '
                                                f'(known: {", ".join(L["known_objects"])})', s.id)
        if s.success.type == 'detected' and s.success.target not in L['known_objects']:
            self._reject(rep, 'unknown_object', f'success condition waits for unknown object '
                                                f'{s.success.target!r}', s.id)

        # depth / geofence
        depth = getattr(p, 'depth', None)
        x, y = getattr(p, 'x', None), getattr(p, 'y', None)
        bad = point_violation(L, x, y, depth)
        if bad:
            self._reject(rep, 'depth_limit' if 'depth' in bad else 'geofence', bad, s.id)
        if s.success.type == 'depth_within':
            bad = point_violation(L, None, None, s.success.depth)
            if bad:
                self._reject(rep, 'depth_limit', 'success condition: ' + bad, s.id)

        # a success condition that can never hold: the battery only drains, so "above 100%" fails even a
        # correct battery check (final eval, E3: the step failed and an otherwise done mission became partial)
        if s.success.type == 'battery_above' and s.success.percent is not None and s.success.percent >= 100.0:
            floor = float(L['min_battery_abort_pct'])
            if self._repair(rep, 'unreachable_success', f'battery_above {s.success.percent:.0f}% can never hold '
                                                        f'(the battery only drains); set to the {floor:.0f}% floor', s.id):
                s.success.percent = floor

        # per-tool clamps
        if s.tool == 'hold_depth' and p.duration_s > L['max_hold_duration_s']:
            if self._repair(rep, 'hold_duration', f'duration {p.duration_s:.0f}s clamped to '
                                                  f'{L["max_hold_duration_s"]:.0f}s', s.id):
                s.params['duration_s'] = float(L['max_hold_duration_s'])
        if s.tool == 'hold_depth' and s.params.get('duration_s', 5.0) + 10 > s.timeout_s:
            need = min(s.params.get('duration_s', 5.0) + 30, L['max_step_timeout_s'])
            if self._repair(rep, 'step_timeout', f'timeout {s.timeout_s:.0f}s too short for the hold; '
                                                 f'set to {need:.0f}s', s.id):
                s.timeout_s = float(need)
        if s.tool == 'search_pattern' and p.radius_m > L['max_search_radius_m']:
            if self._repair(rep, 'search_radius', f'radius {p.radius_m} clamped to '
                                                  f'{L["max_search_radius_m"]}', s.id):
                s.params['radius_m'] = float(L['max_search_radius_m'])
        speed = getattr(p, 'speed', None)
        if speed is not None and speed > L['max_speed_mps']:
            if self._repair(rep, 'speed_limit', f'speed {speed} clamped to {L["max_speed_mps"]}', s.id):
                s.params['speed'] = float(L['max_speed_mps'])

    def _check_intent(self, rep, plan: MissionPlan, prompt: str):
        """Every object the plan goes after must be one the operator asked for. Stops a model from
        'repairing' an impossible request (purple buoy) into a different mission (green buoy).
        The gate is always allowed (it is a waypoint on the way to the buoys); a generic 'buoys' / 'all'
        allows every buoy."""
        for d in requested_depths(prompt):
            if d > self.L['max_depth_m']:
                self._reject(rep, 'depth_limit', f'the operator asked for {d:g} m, beyond the {self.L["max_depth_m"]} m '
                             f'limit: refuse - never run a shallower substitute')
                break
        allowed = mentioned_objects(prompt, self.L['known_objects'])
        for s in plan.steps:
            targets = {s.params.get('target'), s.success.target if s.success.type == 'detected' else None}
            for t in targets - {None}:
                if t in self.L['known_objects'] and t not in allowed:
                    self._reject(rep, 'off_mission_target', f'{t!r} is not something the operator asked for; '
                                 f'do not substitute a different object - refuse instead', s.id)
            # the operator's objective may not be optional: a failed "touch the green buoy" step marked
            # on_failure=continue let a mission end as "succeeded" with nothing touched (v2.1 eval, R3)
            t = s.params.get('target')
            objective = s.tool == 'navigate_to' and t in allowed and (t != 'gate' or 'gate' in prompt.lower())
            if objective and s.on_failure == 'continue' and not _optional_wording(prompt):
                if self._repair(rep, 'objective_not_optional',
                                f'step {s.id} ({t}) is part of the mission objective; on_failure continue -> replan', s.id):
                    s.on_failure = 'replan'

    def _fill_detected_targets(self, rep, plan: MissionPlan):
        """`detected` with an empty target: use the target the plan goes after next, else tool_ok."""
        for i, s in enumerate(plan.steps):
            if s.success.type != 'detected' or (s.success.target or '').strip():
                continue
            nxt = next((t.params.get('target') for t in plan.steps[i:] if t.params.get('target')), None)
            msg = f'detected condition had no target; using {nxt!r}' if nxt else \
                'detected condition had no target; using tool_ok'
            if self._repair(rep, 'success_condition', msg, s.id):
                if nxt:
                    s.success.target = nxt
                else:
                    s.success = SuccessCondition(type='tool_ok')

    def _min_timeout(self, s: Step) -> float:
        if s.tool == 'hold_depth':
            return float(s.params.get('duration_s', 5.0)) + 20.0
        return MIN_STEP_TIMEOUT_S.get(s.tool, 30.0)

    def _fit_time_budget(self, rep, plan: MissionPlan, total: float):
        """Over budget: every step keeps its safe minimum timeout, and the rest of the budget is shared
        in proportion to how much more each step asked for. Reject only if the minimums alone exceed
        the budget. (Uniform scaling used to reject plans whose minimums fit, e.g. gate + three
        searched buoys: 845 s of minimums in a 900 s budget.)"""
        budget = self.L['max_mission_time_s']
        mins = [min(self._min_timeout(s), s.timeout_s) for s in plan.steps]
        if sum(mins) > budget:
            self._reject(rep, 'mission_time',
                         f'sum of step timeouts {total:.0f}s > max mission time {budget:.0f}s, and even the '
                         f'minimum safe timeouts need {sum(mins):.0f}s (too many steps)')
            return
        extra = [s.timeout_s - m for s, m in zip(plan.steps, mins)]
        k = (budget - sum(mins)) / sum(extra) if sum(extra) > 0 else 0.0
        if self._repair(rep, 'mission_time', f'step timeouts sum to {total:.0f}s > {budget:.0f}s; each step keeps '
                                             f'its minimum and shares the remaining {budget - sum(mins):.0f}s'):
            for s, m, e in zip(plan.steps, mins, extra):
                s.timeout_s = float(int(m + e * k))

    def _check_abort_conditions(self, rep, plan: MissionPlan):
        L = self.L
        batt = [a for a in plan.abort_conditions if a.type == 'battery_below']
        if not batt:
            if self._repair(rep, 'battery_abort', f'no battery abort; added battery_below '
                                                  f'{L["default_battery_abort_pct"]:.0f}%'):
                plan.abort_conditions.append(AbortCondition(type='battery_below',
                                                            percent=L['default_battery_abort_pct']))
        for a in batt:
            if not 0 < a.percent < 100:
                self._reject(rep, 'battery_abort', f'battery abort at {a.percent}% is not a valid threshold')
            elif a.percent < L['min_battery_abort_pct']:
                if self._repair(rep, 'battery_abort', f'battery abort {a.percent:.0f}% raised to the minimum '
                                                      f'{L["min_battery_abort_pct"]:.0f}%'):
                    a.percent = float(L['min_battery_abort_pct'])
        for a in plan.abort_conditions:
            if a.type == 'mission_time_exceeded' and (a.seconds <= 0 or a.seconds > L['max_mission_time_s']):
                if self._repair(rep, 'mission_time_abort', f'mission time abort {a.seconds}s set to '
                                                           f'{L["max_mission_time_s"]:.0f}s'):
                    a.seconds = float(L['max_mission_time_s'])

    # ------------------------------------------------------------ runtime precheck
    def precheck_step(self, step: Step, battery_pct: Optional[float], elapsed_s: float,
                      plan: MissionPlan) -> Optional[str]:
        """Return a reason to NOT start this step (executor then aborts safely), else None."""
        if step.tool in ('surface', 'abort', 'get_pose', 'get_battery', 'get_detections'):
            return None           # always allowed: they never make things worse
        if battery_pct is None:
            return 'battery state unknown'
        thresholds = [a.percent for a in plan.abort_conditions if a.type == 'battery_below']
        floor = max(thresholds + [self.L['min_battery_abort_pct']])
        if battery_pct < floor:
            return f'battery {battery_pct:.0f}% is below the abort threshold {floor:.0f}%'
        if step.tool in MOTION_TOOLS and battery_pct < self.L['min_battery_to_start_pct']:
            return (f'battery {battery_pct:.0f}% is below {self.L["min_battery_to_start_pct"]:.0f}% '
                    f'needed to start {step.tool}')
        limit = min([a.seconds for a in plan.abort_conditions if a.type == 'mission_time_exceeded'] +
                    [self.L['max_mission_time_s']])
        if elapsed_s + step.timeout_s > limit + 60:
            return f'not enough mission time left ({elapsed_s:.0f}s used of {limit:.0f}s)'
        return None
