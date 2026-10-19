"""Tool registry: the only way the executor (and therefore the LLM's plan) can act on the vehicle.

call(name, params, timeout_s) -> Observation. Parameters are parsed with the typed schemas; object
targets are resolved from the detection memory; every resolved point is re-checked against the
safety limits at runtime (object positions are only known at execution time).
"""
from __future__ import annotations

import math
import time
from typing import Any, Optional

from pydantic import ValidationError

from auv_mission_planner.limits import point_violation
from auv_mission_planner.tools.backend import Outcome, VehicleBackend, WorldModel
from auv_mission_planner.tools.schemas import TOOL_DESCRIPTIONS, TOOL_PARAMS, Observation


class ToolRegistry:
    def __init__(self, backend: VehicleBackend, cfg: dict):
        self.backend = backend
        self.limits = cfg['limits']
        self.tcfg = cfg.get('tools', {})
        self.world = WorldModel(self.tcfg.get('memory_s', 120.0), self.tcfg.get('min_confidence', 0.4))
        backend.add_detection_listener(self.world.update)
        self._impl = {
            'get_detections': self._get_detections,
            'navigate_to': self._navigate_to,
            'search_pattern': self._search_pattern,
            'hold_depth': self._hold_depth,
            'get_battery': self._get_battery,
            'get_pose': self._get_pose,
            'surface': self._surface,
            'abort': self._abort,
        }
        allowed = set(self.limits.get('allowed_tools', self._impl))
        self._impl = {k: v for k, v in self._impl.items() if k in allowed}

    # ------------------------------------------------------------ public API
    @property
    def names(self) -> list[str]:
        return list(self._impl)

    def schemas(self) -> list[dict[str, Any]]:
        """Tool descriptions + JSON schemas, for the LLM prompt."""
        out = []
        for name in self._impl:
            s = TOOL_PARAMS[name].model_json_schema()
            props = {k: {kk: vv for kk, vv in v.items() if kk in ('type', 'description', 'enum', 'default', 'anyOf')}
                     for k, v in s.get('properties', {}).items()}
            out.append({'name': name, 'description': TOOL_DESCRIPTIONS[name], 'parameters': props,
                        'required': s.get('required', [])})
        return out

    def call(self, name: str, params: Optional[dict] = None, timeout_s: float = 60.0) -> Observation:
        t0 = time.time()
        if name not in self._impl:
            return self._obs(name, t0, False, error=f'tool {name!r} is not allowed', code='unknown_tool')
        try:
            p = TOOL_PARAMS[name].model_validate(params or {})
        except ValidationError as e:
            return self._obs(name, t0, False, error=_short(e), code='invalid_params')
        try:
            ok, data, err, code = self._impl[name](p, timeout_s)
        except Exception as e:   # backend crashed: report, never raise into the executor
            ok, data, err, code = False, {}, f'{type(e).__name__}: {e}', 'backend_unavailable'
        return self._obs(name, t0, ok, data, err, code)

    # ------------------------------------------------------------ helpers
    def _obs(self, name, t0, ok, data=None, error=None, code=None):
        return Observation(tool=name, ok=ok, data=data or {}, error=error, error_code=code,
                           duration_s=round(time.time() - t0, 2), sim_time=round(self.backend.now(), 2))

    def _from_outcome(self, o: Outcome, data=None):
        return o.ok, {**(data or {}), **o.data, 'message': o.message}, None if o.ok else o.message, \
            None if o.ok else (o.code or 'action_failed')

    def _visible(self):
        return [d.as_dict() for d in self.backend.detections()]

    def _go(self, x, y, depth, yaw=None, tol=None, speed=None, timeout_s=60.0):
        bad = point_violation(self.limits, x, y, depth)
        if bad:
            return Outcome(False, bad, 'unsafe_target')
        speed = min(speed or self.limits['max_speed_mps'], self.limits['max_speed_mps'])
        return self.backend.navigate(x, y, depth, yaw, tol or self.tcfg.get('default_tolerance_m', 0.35),
                                     speed, timeout_s)

    # ------------------------------------------------------------ tools
    def _get_detections(self, p, timeout_s):
        now = self.backend.now()
        return True, {'visible': self._visible(), 'remembered': self.world.summary(now)}, None, None

    def _get_battery(self, p, timeout_s):
        b = self.backend.battery_percent()
        if b is None:
            return False, {}, 'no battery data', 'backend_unavailable'
        return True, {'percent': round(b, 1)}, None, None

    def _get_pose(self, p, timeout_s):
        pose = self.backend.pose()
        if pose is None:
            return False, {}, 'no pose data', 'backend_unavailable'
        return True, pose.as_dict(), None, None

    def _navigate_to(self, p, timeout_s):
        deadline = self.backend.now() + timeout_s
        remaining = lambda: max(1.0, deadline - self.backend.now())
        if p.mode == 'point':
            o = self._go(p.x, p.y, p.depth, speed=p.speed, timeout_s=timeout_s)
            return self._from_outcome(o, {'target': {'x': p.x, 'y': p.y, 'depth': p.depth}})
        if p.mode == 'relative':
            pose = self.backend.pose()
            if pose is None:
                return False, {}, 'no pose data for a relative move', 'backend_unavailable'
            f, l = p.forward_m or 0.0, p.left_m or 0.0
            x = pose.x + math.cos(pose.yaw) * f - math.sin(pose.yaw) * l
            y = pose.y + math.sin(pose.yaw) * f + math.cos(pose.yaw) * l
            depth = p.depth if p.depth is not None else pose.depth
            o = self._go(x, y, depth, speed=p.speed, timeout_s=timeout_s)   # limits checked on the resolved point
            return self._from_outcome(o, {'target': {'x': round(x, 2), 'y': round(y, 2), 'depth': round(depth, 2)}})

        if p.target not in self.limits.get('known_objects', []):
            return False, {'known_objects': self.limits.get('known_objects', [])}, \
                f'unknown object {p.target!r}', 'target_not_detected'
        est = self.world.lookup(p.target, self.backend.now())
        if est is None:
            return False, {'visible_now': [d['class'] for d in self._visible()],
                           'remembered': list(self.world.summary(self.backend.now()))}, \
                f'{p.target} has not been detected yet (search for it first)', 'target_not_detected'
        pose = self.backend.pose()
        depth = p.depth if p.depth is not None else est.depth
        dx, dy = est.x - pose.x, est.y - pose.y
        dist = math.hypot(dx, dy) or 1e-6
        ux, uy = dx / dist, dy / dist
        heading = math.atan2(uy, ux)

        if p.passes_gate:
            a = self.tcfg.get('gate_approach_m', 1.5)
            o = self._go(est.x - a * ux, est.y - a * uy, depth, heading, 0.4, p.speed, remaining())
            if not o.ok:
                return self._from_outcome(o, {'phase': 'gate_approach'})
            est2 = self.world.lookup('gate', self.backend.now(), near_range=4.0) or est
            o = self._go(est2.x + a * ux, est2.y + a * uy, depth, heading, 0.4, p.speed, remaining())
            return self._from_outcome(o, {'phase': 'gate_exit', 'gate_estimate': est2.as_dict()['position']})

        standoff = max(0.0, p.standoff_m)
        approach = self.tcfg.get('touch_approach_m', 1.5)
        if dist > standoff + approach + 0.5:
            # stop short, look again from close range, then finish
            o = self._go(est.x - (standoff + approach) * ux, est.y - (standoff + approach) * uy, depth,
                         heading, 0.5, p.speed, remaining())
            if not o.ok:
                return self._from_outcome(o, {'phase': 'approach'})
            est = self.world.lookup(p.target, self.backend.now(), near_range=approach + 1.5) or est
            if p.depth is None:
                depth = est.depth
        tol = 0.25 if standoff == 0 else self.tcfg.get('default_tolerance_m', 0.35)
        o = self._go(est.x - standoff * ux, est.y - standoff * uy, depth, heading, tol, p.speed, remaining())
        return self._from_outcome(o, {'target': p.target, 'estimate': est.as_dict()['position'],
                                      'standoff_m': standoff})

    def _search_pattern(self, p, timeout_s):
        depth = p.depth
        if depth is None:
            pose = self.backend.pose()
            # objects sit at ~1.5-2 m: searching from the surface sees almost nothing
            default = self.tcfg.get('default_search_depth_m', 1.5)
            depth = pose.depth if (pose and pose.depth >= 1.0) else default
        bad = point_violation(self.limits, None, None, depth)
        if bad:
            return False, {}, bad, 'unsafe_target'
        radius = min(p.radius_m, self.limits.get('max_search_radius_m', 12.0))
        o = self.backend.search(p.target, p.pattern, radius, depth, timeout_s)
        if not o.ok and o.code in (None, 'action_failed') and o.data.get('found') is False:
            o.code = 'not_found'          # pattern finished without seeing the target
        return self._from_outcome(o, {'target': p.target, 'pattern': p.pattern})

    def _hold_depth(self, p, timeout_s):
        bad = point_violation(self.limits, None, None, p.depth)
        if bad:
            return False, {}, bad, 'unsafe_target'
        dur = min(p.duration_s, self.limits.get('max_hold_duration_s', 120.0))
        return self._from_outcome(self.backend.hold_depth(p.depth, dur, 0.3, timeout_s), {'depth': p.depth})

    def _surface(self, p, timeout_s):
        cleared = self._clear_gate(timeout_s / 3)
        o = self.backend.surface(timeout_s)
        return self._from_outcome(o, {'cleared_gate_first': cleared} if cleared else None)

    def _clear_gate(self, timeout_s) -> bool:
        """If the vehicle is inside the remembered gate frame, drive out along its heading first
        (ascending straight up would hit the crossbar)."""
        pose = self.backend.pose()
        gate = self.world.lookup('gate', self.backend.now()) if pose else None
        if gate is None or math.hypot(pose.x - gate.x, pose.y - gate.y) > self.tcfg.get('gate_clearance_m', 1.0):
            return False
        a = self.tcfg.get('gate_approach_m', 1.5)
        x, y = pose.x + a * math.cos(pose.yaw), pose.y + a * math.sin(pose.yaw)
        return self._go(x, y, pose.depth, pose.yaw, 0.5, None, max(10.0, timeout_s)).ok

    def _abort(self, p, timeout_s):
        self.backend.cancel_all()
        o = self.backend.emergency_surface()
        return self._from_outcome(o, {'reason': p.reason})


def _short(e: ValidationError) -> str:
    return '; '.join(f'{".".join(map(str, er["loc"])) or "params"}: {er["msg"]}' for er in e.errors())
