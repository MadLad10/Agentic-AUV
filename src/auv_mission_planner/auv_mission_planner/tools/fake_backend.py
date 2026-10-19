"""Deterministic kinematic backend for unit tests and plan-only evaluation (no ROS, no Gazebo).

Motion is instantaneous but advances a virtual clock by distance / speed, so timeouts behave
realistically. Detections use a simple camera model (FOV + range). Fault injection:
  blind      - classes never detected
  fail_next  - {tool_key: n} make the next n backend calls of that kind fail
  battery_start, drain_pct_per_s
"""
from __future__ import annotations

import math
from typing import Optional

from auv_mission_planner.tools.backend import Det, Outcome, Pose, VehicleBackend

DEFAULT_OBJECTS = [
    {'id': 'gate', 'class': 'gate', 'x': 8.0, 'y': 0.0, 'z': -1.5},
    {'id': 'red_buoy', 'class': 'red_buoy', 'x': 14.0, 'y': -2.0, 'z': -1.5},
    {'id': 'green_buoy', 'class': 'green_buoy', 'x': 14.0, 'y': 2.0, 'z': -1.5},
    {'id': 'yellow_buoy', 'class': 'yellow_buoy', 'x': 18.0, 'y': 0.0, 'z': -2.0},
]


class FakeBackend(VehicleBackend):
    HFOV = math.radians(80)

    def __init__(self, objects=None, start=(0.0, 0.0, 0.1, 0.0), battery_start=100.0,
                 drain_pct_per_s=0.05, blind=(), fail_next=None, speed=0.5, max_range=8.0, gate_range=14.0):
        super().__init__()
        self.objects = objects if objects is not None else DEFAULT_OBJECTS
        self._pose = Pose(*start)
        self.t = 0.0
        self.batt = battery_start
        self.drain = drain_pct_per_s
        self.blind = set(blind)
        self.fail_next = dict(fail_next or {})
        self.speed = speed
        self.max_range, self.gate_range = max_range, gate_range
        self.calls: list[tuple] = []
        self.surfaced = False
        self.emergency = False
        self.touched: list[str] = []
        self.gate_passed = False

    # ------------------------------------------------------------ helpers
    def _advance(self, dt):
        self.t += dt
        self.batt = max(0.0, self.batt - self.drain * dt)

    def _see(self):
        p = self._pose
        out = []
        for o in self.objects:
            if o['class'] in self.blind:
                continue
            dx, dy, dz = o['x'] - p.x, o['y'] - p.y, -o['z'] - p.depth
            rng = math.sqrt(dx * dx + dy * dy + dz * dz)
            bearing = math.atan2(dy, dx) - p.yaw
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            lim = self.gate_range if o['class'] == 'gate' else self.max_range
            if rng <= lim and abs(bearing) <= self.HFOV / 2:
                out.append(Det(o['class'], 0.9, o['x'], o['y'], -o['z'], rng, math.degrees(bearing), self.t))
        if out:
            self._notify(out)
        return out

    def _fail(self, key) -> Optional[Outcome]:
        n = self.fail_next.get(key, 0)
        if n > 0:
            self.fail_next[key] = n - 1
            return Outcome(False, f'injected {key} failure', 'action_failed')
        return None

    def _move(self, x, y, depth, yaw, timeout_s):
        p = self._pose
        d = math.dist((p.x, p.y, p.depth), (x, y, depth))
        need = d / self.speed
        heading = math.atan2(y - p.y, x - p.x) if math.hypot(x - p.x, y - p.y) > 0.5 else p.yaw
        if need > timeout_s:
            f = timeout_s / need
            self._pose = Pose(p.x + (x - p.x) * f, p.y + (y - p.y) * f, p.depth + (depth - p.depth) * f, heading)
            self._advance(timeout_s)
            return False
        # referee-like bookkeeping
        for o in self.objects:
            if o['class'] == 'gate':
                if (p.x - o['x']) * (x - o['x']) < 0 and abs((y + p.y) / 2 - o['y']) < 1.0:
                    self.gate_passed = True
            elif math.dist((x, y, depth), (o['x'], o['y'], -o['z'])) < 0.55 and o['id'] not in self.touched:
                self.touched.append(o['id'])
        self._pose = Pose(x, y, depth, yaw if yaw is not None else heading)
        self._advance(need + 1.0)
        self.surfaced = depth < 0.2
        self._see()
        return True

    # ------------------------------------------------------------ VehicleBackend
    def now(self):
        return self.t

    def pose(self):
        return self._pose

    def battery_percent(self):
        return self.batt

    def detections(self):
        return self._see()

    def navigate(self, x, y, depth, yaw=None, tolerance=0.35, speed=None, timeout_s=60.0):
        self.calls.append(('navigate', round(x, 2), round(y, 2), round(depth, 2)))
        f = self._fail('navigate')
        if f:
            return f
        if self._move(x, y, depth, yaw, timeout_s):
            return Outcome(True, 'reached', data={'final_error_m': 0.05})
        return Outcome(False, f'navigate timed out after {timeout_s:.0f}s', 'timeout')

    def search(self, target, pattern, radius_m, depth, timeout_s=60.0):
        self.calls.append(('search', target, pattern))
        f = self._fail('search')
        if f:
            return f
        p = self._pose
        t_start = self.t
        self._move(p.x, p.y, depth, p.yaw, timeout_s)
        r = int(radius_m)
        probes = [(0, 0)] if pattern == 'yaw_sweep' else sorted(   # nearest first, like a real spiral
            [(dx, dy) for dx in range(-r, r + 1, 3) for dy in range(-r, r + 1, 3)
             if math.hypot(dx, dy) <= radius_m], key=lambda q: math.hypot(*q))
        for dx, dy in probes:
            for k in range(6):
                yaw = p.yaw + k * math.pi / 3
                self._pose = Pose(p.x + dx, p.y + dy, depth, yaw)
                self._advance(1.0)
                if self.t - t_start > timeout_s:
                    self._pose = Pose(p.x, p.y, depth, p.yaw)
                    return Outcome(False, f'search timed out after {timeout_s:.0f}s', 'timeout', {'found': False})
                for d in self._see():
                    if d.cls == target:
                        return Outcome(True, f'{target} found', data={'found': True, 'detection': d.as_dict()})
        self._pose = Pose(p.x, p.y, depth, p.yaw)
        return Outcome(False, f'{target} not found with {pattern}', 'not_found', {'found': False})

    def hold_depth(self, depth, duration_s, tolerance=0.3, timeout_s=60.0):
        self.calls.append(('hold_depth', depth, duration_s))
        f = self._fail('hold_depth')
        if f:
            return f
        p = self._pose
        if not self._move(p.x, p.y, depth, p.yaw, timeout_s):
            return Outcome(False, 'hold_depth timed out', 'timeout')
        self._advance(duration_s)
        return Outcome(True, f'held {depth} m', data={'mean_error_m': 0.05})

    def surface(self, timeout_s=60.0):
        self.calls.append(('surface',))
        f = self._fail('surface')
        if f:
            return f
        p = self._pose
        self._move(p.x, p.y, 0.0, p.yaw, timeout_s)
        self.surfaced = True
        return Outcome(True, 'surfaced')

    def emergency_surface(self):
        self.calls.append(('emergency_surface',))
        self.emergency = True
        p = self._pose
        self._pose = Pose(p.x, p.y, 0.0, p.yaw)
        self.surfaced = True
        return Outcome(True, 'surfacing')

    def cancel_all(self):
        self.calls.append(('cancel_all',))
