"""Obstacle-aware leg planning around the gate (the only rigid course structure in the pool).

Guidance used to drive every move in a straight line. A straight leg from the start to a target
beyond the gate can run through a gate post, or under the crossbar too shallow: the referee
recorded repeated collisions in exactly that case (sonar-cued inspections in murky water).

Model (from the course map the vehicle already uses for navigation):
  posts     keep-out circles: post radius + hull half-diagonal + margin
  crossbar  crossing the gate line inside the frame is allowed only deeper than the bar + margin;
            shallower, the whole frame is an obstacle and the leg detours around the nearer post
Planner: repeatedly find the first obstacle a leg violates and insert a detour waypoint beside it
(tangent side chosen by the leg's closest approach), up to a few iterations. Pure functions, no ROS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Gate:
    cx: float
    cy: float
    yaw: float           # gate plane normal direction (travel direction through the gate)
    width: float
    bar_depth: float     # depth of the crossbar (m, positive down)

    def posts(self):
        # posts lie along the gate line, perpendicular to the normal (yaw)
        ux, uy = -math.sin(self.yaw), math.cos(self.yaw)
        h = self.width / 2
        return (self.cx + ux * h, self.cy + uy * h), (self.cx - ux * h, self.cy - uy * h)


def _closest(p, q, c):
    """Closest point to c on segment p-q, and its parameter t in [0, 1]."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((c[0] - p[0]) * dx + (c[1] - p[1]) * dy) / L2))
    return (p[0] + t * dx, p[1] + t * dy), t


def _cross_gate_line(p, q, g: Gate):
    """Parameter where segment p-q crosses the gate line, and the lateral offset from the centre."""
    nx, ny = math.cos(g.yaw), math.sin(g.yaw)
    sp = (p[0] - g.cx) * nx + (p[1] - g.cy) * ny
    sq = (q[0] - g.cx) * nx + (q[1] - g.cy) * ny
    if sp * sq > 0 or abs(sp - sq) < 1e-9:
        return None
    t = sp / (sp - sq)
    x, y = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
    lateral = (x - g.cx) * -math.sin(g.yaw) + (y - g.cy) * math.cos(g.yaw)
    return t, lateral


def plan_legs(start, goal, depth: float, gate: Gate | None, clearance: float = 0.65,
              vertical_clearance: float = 0.3, duck_m: float = 1.2, max_iter: int = 8) -> list[tuple]:
    """Intermediate (x, y, depth) waypoints so every leg from start to goal keeps clear of the gate.

    Posts: detour beside the post. Crossbar: if a leg crosses the gate line inside the frame
    shallower than the bar, the crossing is kept (a "pass through the gate" mission still passes)
    but done deeper: waypoints `duck_m` before and after the gate line at bar depth + clearance."""
    if gate is None:
        return []
    safe_depth = gate.bar_depth + vertical_clearance
    path = [(start[0], start[1], depth), (goal[0], goal[1], depth)]
    posts = gate.posts()
    for _ in range(max_iter):
        fixed = False
        for i in range(len(path) - 1):
            p, q = path[i], path[i + 1]
            ins = None
            for c in posts:                                   # 1. too close to a post
                cp, _ = _closest(p, q, c)
                d = math.dist(cp, c)
                if d < clearance:
                    if d > 1e-6:
                        nx, ny = (cp[0] - c[0]) / d, (cp[1] - c[1]) / d
                    else:                                     # aimed at the post centre: pass on its outer side
                        ox, oy = c[0] - gate.cx, c[1] - gate.cy
                        n = math.hypot(ox, oy) or 1.0
                        nx, ny = ox / n, oy / n
                    ins = [(c[0] + nx * (clearance + 0.25), c[1] + ny * (clearance + 0.25), q[2])]
                    break
            if ins is None:                                   # 2. under the crossbar too shallow: duck
                x = _cross_gate_line(p, q, gate)
                if x is not None and abs(x[1]) < gate.width / 2 and min(p[2], q[2]) < safe_depth:
                    t, L = x[0], math.dist(p[:2], q[:2]) or 1.0
                    dt = min(duck_m / L, 0.45)
                    d2 = max(q[2], safe_depth)
                    a = (p[0] + max(0.0, t - dt) * (q[0] - p[0]), p[1] + max(0.0, t - dt) * (q[1] - p[1]), d2)
                    b = (p[0] + min(1.0, t + dt) * (q[0] - p[0]), p[1] + min(1.0, t + dt) * (q[1] - p[1]), d2)
                    ins = [a, b]
            if ins is not None:
                path[i + 1:i + 1] = ins
                fixed = True
                break
        if not fixed:
            break
    return path[1:-1]
