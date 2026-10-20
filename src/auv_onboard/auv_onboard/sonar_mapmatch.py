"""Sonar map-matching: forward-looking-sonar ranges to the KNOWN pool walls as navigation updates.

A test pool is a perfectly known map (a rectangle of walls). Every sonar beam that hits a wall is a
range measurement to a known surface, so it constrains the vehicle's position and heading even when
the DVL has lost the floor and no landmark is in view - the situation where camera and sonar
landmarks stop helping on our route (the gate is behind the vehicle).

For each selected beam: predicted range = distance from the sonar along (yaw + beam bearing) to the
pool rectangle (2D ray casting); measurement = the beam's range. Beams are used only if
  - the measured range is close to the predicted wall range (gate), so returns from buoys, posts
    or ghosts are not mistaken for the wall,
  - the beam meets the wall at a usable angle (grazing returns are unreliable),
  - the predicted range is inside the sonar's maximum range.
The Jacobian w.r.t. (x, y, yaw) is computed by central differences. Pure functions, no ROS.
"""
from __future__ import annotations

import math

import numpy as np


def ray_to_rect(ox, oy, ang, x0, x1, y0, y1):
    """Distance from (ox, oy) along angle `ang` to the inside boundary of the rectangle, and the
    incidence angle (0 = head-on) of the hit wall. Returns (inf, None) if the origin is outside."""
    if not (x0 < ox < x1 and y0 < oy < y1):
        return math.inf, None
    dx, dy = math.cos(ang), math.sin(ang)
    best, inc = math.inf, None
    if dx > 1e-9:
        t = (x1 - ox) / dx
        if t < best:
            best, inc = t, math.acos(min(1.0, abs(dx)))
    if dx < -1e-9:
        t = (x0 - ox) / dx
        if t < best:
            best, inc = t, math.acos(min(1.0, abs(dx)))
    if dy > 1e-9:
        t = (y1 - oy) / dy
        if t < best:
            best, inc = t, math.acos(min(1.0, abs(dy)))
    if dy < -1e-9:
        t = (y0 - oy) / dy
        if t < best:
            best, inc = t, math.acos(min(1.0, abs(dy)))
    return best, inc


def predicted_range(x, y, yaw, bearing, offset, rect):
    """Range from the sonar (mounted at `offset` in the body frame) to the pool wall along a beam."""
    c, s = math.cos(yaw), math.sin(yaw)
    ox = x + c * offset[0] - s * offset[1]
    oy = y + s * offset[0] + c * offset[1]
    return ray_to_rect(ox, oy, yaw + bearing, *rect)


def select_and_linearise(state_xyyaw, scan: dict, offset, rect, cfg: dict):
    """Pick usable wall beams and return [(innovation, H_xyyaw (3,), variance)] for a scalar update each."""
    if not scan or not scan.get('valid') or not scan.get('ranges'):
        return []
    x, y, yaw = state_xyyaw
    out = []
    step = max(1, int(cfg.get('beam_stride', 4)))
    max_inc = math.radians(cfg.get('max_incidence_deg', 60.0))
    max_r = cfg.get('max_range_m', 15.0) - 0.3
    sigma = cfg.get('range_std_m', 0.15)
    gate = cfg.get('gate_m', 0.8)
    for i in range(0, len(scan['bearings']), step):
        b, r = scan['bearings'][i], scan['ranges'][i]
        if r <= 0:
            continue
        rp, inc = predicted_range(x, y, yaw, b, offset, rect)
        if not math.isfinite(rp) or rp > max_r or inc is None or inc > max_inc:
            continue
        if abs(r - rp) > gate:
            continue                              # a buoy, post or ghost in front of the wall
        eps = 1e-4
        H = np.zeros(3)
        for k in range(3):
            d = np.zeros(3)
            d[k] = eps
            hp, _ = predicted_range(x + d[0], y + d[1], yaw + d[2], b, offset, rect)
            hm, _ = predicted_range(x - d[0], y - d[1], yaw - d[2], b, offset, rect)
            H[k] = (hp - hm) / (2 * eps)
        # grazing beams have range errors that grow with 1/cos(incidence)
        var = (sigma / max(math.cos(inc), 0.3)) ** 2
        out.append((r - rp, H, var))
    return out


def pool_rect(pool: dict) -> tuple:
    x0 = pool.get('x_min', -5.0)
    return (x0, x0 + pool['length'], -pool['width'] / 2, pool['width'] / 2)
