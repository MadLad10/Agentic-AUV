"""Onboard perception: camera object detection, sonar contacts, and camera-sonar fusion.

Camera detectors produce RELATIVE measurements (range, bearing, elevation from the camera). They
become world positions only through the navigation estimate, so localisation error shows up in
object positions exactly as it would on a real vehicle.

Detector sources (`detection_source`):
  ground_truth  camera-model oracle: true geometry + FOV, visibility (turbidity), occlusion,
                range/bearing noise and dropout; blackout -> nothing. The reliable fallback.
  yolo          YOLO trained on synthetic auto-labelled frames (see auv_marine_sim.autolabel)
  hsv           v1's colour-threshold detector (kept as a baseline)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from auv_onboard.ekf import wrap


@dataclass
class RelDet:
    cls: str
    confidence: float
    range_m: float
    bearing: float          # rad, + left
    elevation: float        # rad, + up
    bbox: Optional[tuple] = None
    source: str = 'camera'


@dataclass
class SonarContact:
    range_m: float
    bearing: float
    width_rad: float
    intensity: float
    x: float = 0.0
    y: float = 0.0


@dataclass
class Track:
    cls: str
    x: float
    y: float
    depth: float
    confidence: float
    sources: list = field(default_factory=list)
    t: float = 0.0


def project_bbox(rng, bearing, elev, w_m, h_m, hfov, W, H):
    f = (W / 2) / math.tan(hfov / 2)
    u = W / 2 - f * math.tan(bearing)
    v = H / 2 - f * math.tan(elev)
    hw, hh = f * w_m / max(rng, 0.1) / 2, f * h_m / max(rng, 0.1) / 2
    cl = lambda a, hi: float(max(0, min(hi, a)))
    return cl(u - hw, W), cl(v - hh, H), cl(u + hw, W), cl(v + hh, H)


class GroundTruthCameraDetector:
    """Camera-model oracle on simulator truth (no image processing)."""

    def __init__(self, cfg: dict, sim, rng: np.random.Generator):
        self.cfg = cfg
        self.sim = sim
        self.rng = rng
        cam = sim.cfg['sensors']['camera']
        self.W, self.H = cam['width'], cam['height']
        self.hfov = math.radians(cam['hfov_deg'])
        self.vfov = 2 * math.atan(math.tan(self.hfov / 2) * self.H / self.W)
        self.offset = np.array(cam['offset'], float)

    def detect(self) -> list[RelDet]:
        sim, c = self.sim, self.cfg
        if sim.faults.get('camera_blackout', sim.t):
            return []
        vis = sim.visible_range()
        R = sim.R()
        tr = sim.truth()
        cam_pos = tr.pos + R @ self.offset
        out = []
        blind = set(c.get('blind_classes') or [])        # eval fault: classes the detector never reports
        for o in sim.arena['objects']:
            if o['class'] in blind:
                continue
            p = np.array([o['x'], o['y'], o['z']], float)
            rel = R.T @ (p - cam_pos)
            rng = float(np.linalg.norm(rel))
            if rel[0] <= 0.2:
                continue
            bearing = math.atan2(rel[1], rel[0])
            elev = math.atan2(rel[2], math.hypot(rel[0], rel[1]))
            if abs(bearing) > self.hfov / 2 or abs(elev) > self.vfov / 2:
                continue
            max_r = c['gate_max_range_m'] if o['class'] == 'gate' else c['max_range_m']
            max_r = min(max_r, vis * (c['gate_visibility_factor'] if o['class'] == 'gate' else c['visibility_factor']))
            if rng > max_r or self.rng.random() > c['p_detect']:
                continue
            if o['class'] != 'gate' and self._occluded(cam_pos, p, rng, o['id']):
                continue
            conf = float(np.clip(1.0 - 0.6 * rng / max(max_r, 1e-3) + self.rng.normal(0, 0.03), 0.05, 0.99))
            r_noisy = rng * (1 + self.rng.normal(0, c['range_noise_frac']))
            b_noisy = bearing + self.rng.normal(0, math.radians(c['bearing_noise_deg']))
            e_noisy = elev + self.rng.normal(0, math.radians(c['bearing_noise_deg']))
            w = o.get('width', 2 * o.get('radius', 0.2))
            h = o.get('height', 2 * o.get('radius', 0.2))
            out.append(RelDet(o['class'], conf, r_noisy, b_noisy, e_noisy,
                              project_bbox(rng, bearing, elev, w, h, self.hfov, self.W, self.H)))
        return out

    def _occluded(self, cam_pos, p, rng, obj_id) -> bool:
        import mujoco
        d = (p - cam_pos) / max(rng, 1e-6)
        gid = np.array([-1], np.int32)
        dist = mujoco.mj_ray(self.sim.model, self.sim.data, cam_pos, d, np.array([1, 0, 0, 0, 1, 0], np.uint8),
                             1, self.sim.body, gid)
        if gid[0] < 0 or dist < 0:
            return False
        name = self.sim.model.geom(int(gid[0])).name or ''
        return not name.startswith(obj_id + '__') and dist < rng - 0.3


def rel_to_world(det: RelDet, est: dict, cam_offset) -> tuple[float, float, float]:
    """Camera-relative detection -> world (x, y, depth) using the navigation estimate."""
    yaw = est['yaw']
    c, s = math.cos(yaw), math.sin(yaw)
    cx = est['x'] + c * cam_offset[0] - s * cam_offset[1]
    cy = est['y'] + s * cam_offset[0] + c * cam_offset[1]
    cz = est['z'] + cam_offset[2]
    horiz = det.range_m * math.cos(det.elevation)
    a = yaw + det.bearing
    return cx + horiz * math.cos(a), cy + horiz * math.sin(a), -(cz + det.range_m * math.sin(det.elevation))


def sonar_contacts(scan: dict, est: dict, pool: dict, offset, wall_margin=0.8, gap_m=0.5) -> list[SonarContact]:
    """Cluster sonar returns into contacts, dropping returns that map onto the known pool walls."""
    if not scan or not scan.get('valid') or not scan.get('ranges'):
        return []
    yaw = est['yaw']
    c, s = math.cos(yaw), math.sin(yaw)
    ox = est['x'] + c * offset[0] - s * offset[1]
    oy = est['y'] + s * offset[0] + c * offset[1]
    x0 = pool.get('x_min', -5.0)
    x1, yh = x0 + pool['length'], pool['width'] / 2
    pts = []
    for b, r, it in zip(scan['bearings'], scan['ranges'], scan['intensity']):
        if r <= 0:
            pts.append(None)
            continue
        wx, wy = ox + r * math.cos(yaw + b), oy + r * math.sin(yaw + b)
        if wx < x0 + wall_margin or wx > x1 - wall_margin or abs(wy) > yh - wall_margin:
            pts.append(None)
            continue
        pts.append((b, r, it, wx, wy))
    out, cur = [], []

    def flush():
        if cur:
            rs = [p[1] for p in cur]
            out.append(SonarContact(float(np.mean(rs)), float(np.mean([p[0] for p in cur])),
                                    float(cur[-1][0] - cur[0][0]), float(max(p[2] for p in cur)),
                                    float(np.mean([p[3] for p in cur])), float(np.mean([p[4] for p in cur]))))
    for p in pts:
        if p is None or (cur and abs(p[1] - cur[-1][1]) > gap_m):
            flush()
            cur = [] if p is None else [p]
        else:
            cur.append(p)
    flush()
    return out


def associate(dets_world: list[tuple], contacts: list[SonarContact], gate_m=1.0):
    """Pair camera detections (cls, x, y, depth, conf) with the nearest sonar contact within
    `gate_m`. Returns [(det, contact_or_None)] and unmatched sonar contacts."""
    used = set()
    pairs = []
    for d in dets_world:
        best, bd = None, gate_m
        for i, ct in enumerate(contacts):
            if i in used:
                continue
            dist = math.hypot(ct.x - d[1], ct.y - d[2])
            if dist < bd:
                best, bd = i, dist
        if best is not None:
            used.add(best)
        pairs.append((d, contacts[best] if best is not None else None))
    return pairs, [ct for i, ct in enumerate(contacts) if i not in used]


__all__ = ['RelDet', 'SonarContact', 'Track', 'GroundTruthCameraDetector', 'rel_to_world', 'sonar_contacts',
           'associate', 'project_bbox', 'wrap']
