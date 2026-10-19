"""Specialist agents. Each has one job and a measurable output.

  PerceptionAgent   camera detections + sonar contacts -> fused tracks; sonar-only gate recognition
                    (two posts a gate-width apart); flags unconfirmed sonar contacts (false targets);
                    proposes sonar-cued inspection points; asks the VLM on demand.
  NavigationAgent   reads the EKF: position uncertainty, which sensors are being fused; declares
                    degraded navigation and slows the vehicle down.
  HealthAgent       deterministic, veto power: the onboard health monitor's report plus the plan's
                    abort conditions (battery floor, mission time).
  DiagnosticsAgent  LLM: explains new faults from telemetry and picks a degraded-mode action from a
                    closed set; deterministic fallback when the LLM fails or is too slow.
"""
from __future__ import annotations

import json
import math
import time
from typing import Optional

from auv_mission_planner.tools.backend import Det
from auv_onboard.health import DEGRADED_MODE, VETO_FAULTS
from auv_onboard.perception import associate

FAULTS = sorted(set(DEGRADED_MODE) | VETO_FAULTS)
ACTIONS = ['continue', 'replan', 'abort']


# ===================================================================== perception
class PerceptionAgent:
    def __init__(self, cfg: dict, backend, registry, vlm=None):
        self.cfg = cfg
        self.be = backend
        self.reg = registry
        self.vlm = vlm
        self._unconfirmed: dict[tuple, int] = {}
        self.vlm_log: list[dict] = []

    def _gate_from_sonar(self, contacts, gate_w):
        """Gate signature: two thin posts a gate-width apart, or (at gate depth, where the crossbar
        also returns) one contact about a gate-width wide."""
        for c in contacts:
            if abs(c.range_m * c.width_rad - gate_w) < 2 * self.cfg['gate_pair_tolerance_m']:
                return (gate_w, c, c)
        best = None
        for i, a in enumerate(contacts):
            for b in contacts[i + 1:]:
                d = math.hypot(a.x - b.x, a.y - b.y)
                if abs(d - gate_w) < self.cfg['gate_pair_tolerance_m'] and max(a.width_rad, b.width_rad) < 0.25:
                    if best is None or d < best[0]:
                        best = (d, a, b)
        return best

    def update(self, camera_ok: bool = True) -> dict:
        be = self.be
        t = be.now()
        est = be.est()
        dets = be.detections()
        contacts = list(be.contacts)
        world = self.reg.world.summary(t)
        dw = [(d.cls, d.x, d.y, d.depth, d.confidence) for d in dets]
        gate_w = next((o.get('width', 2.0) for o in be.sim.arena['objects'] if o['class'] == 'gate'), 2.0)
        # a gate detection explains contacts on its posts
        matched_gate = set()
        for d in dw:
            if d[0] == 'gate':
                for i, c in enumerate(contacts):
                    if math.hypot(c.x - d[1], c.y - d[2]) < gate_w / 2 + 0.6:
                        matched_gate.add(i)
        rest = [c for i, c in enumerate(contacts) if i not in matched_gate]
        pairs, unmatched = associate([d for d in dw if d[0] != 'gate'], rest, gate_m=1.0)
        tracks = [{'class': d[0], 'x': round(d[1], 2), 'y': round(d[2], 2), 'depth': round(d[3], 2),
                   'confidence': round(d[4] if c is None else min(0.99, d[4] + 0.1), 2),   # sonar confirms
                   'sources': ['camera'] + (['sonar'] if c is not None else [])}
                  for d, c in pairs]
        tracks += [{'class': 'gate', 'x': round(d[1], 2), 'y': round(d[2], 2), 'depth': round(d[3], 2),
                    'confidence': round(d[4], 2), 'sources': ['camera'] + (['sonar'] if matched_gate else [])}
                   for d in dw if d[0] == 'gate']
        # sonar-only gate (camera blind or murky): two thin posts a gate-width apart
        sonar_gate = None
        if not any(tr['class'] == 'gate' for tr in tracks):
            g = self._gate_from_sonar(unmatched, gate_w)
            if g is not None:
                _, a, b = g
                gx, gy = (a.x + b.x) / 2, (a.y + b.y) / 2
                gd = be.landmark_depth.get('gate', -est['z'])          # course-map depth prior
                sonar_gate = {'class': 'gate', 'x': round(gx, 2), 'y': round(gy, 2), 'depth': round(gd, 2),
                              'confidence': 0.6, 'sources': ['sonar']}
                tracks.append(sonar_gate)
                # inject into the shared world model so navigate_to(gate) works without the camera
                self.reg.world.update([Det('gate', 0.6, gx, gy, gd, math.hypot(gx - est['x'], gy - est['y']),
                                           0.0, t)])
                unmatched = [c for c in unmatched if c is not a and c is not b]
        # unconfirmed sonar contacts: inside the camera's good range and FOV but not seen by it
        vis = be.sim.visible_range()
        suspects = []
        for c in unmatched:
            key = (round(c.x), round(c.y))
            inside = abs(c.bearing) < math.radians(35) and c.range_m < self.cfg['false_target_max_range_frac'] * vis
            if camera_ok and inside:
                self._unconfirmed[key] = self._unconfirmed.get(key, 0) + 1
                if self._unconfirmed[key] >= self.cfg['false_target_scans']:
                    suspects.append({'x': round(c.x, 2), 'y': round(c.y, 2), 'range_m': round(c.range_m, 2)})
        be.health.sonar_flags = {'false_target': bool(suspects),
                                 'detail': f'sonar contact at {suspects[0]} not confirmed by the camera'
                                 if suspects else ''}
        suspect_keys = {(round(s['x']), round(s['y'])) for s in suspects}
        inspect = []
        for c in unmatched:
            if (round(c.x), round(c.y)) in suspect_keys:
                continue
            inspect.append({'x': round(c.x, 2), 'y': round(c.y, 2), 'range_m': round(c.range_m, 2)})
        return {'t': round(t, 2), 'tracks': tracks, 'remembered': world,
                'unclassified_sonar_contacts': inspect[:6], 'suspected_false_targets': suspects,
                'sonar_only_gate': sonar_gate is not None, 'camera_visibility_m': round(vis, 1)}

    def inspection_points(self, report: dict, standoff: float) -> list[tuple]:
        """Waypoints `standoff` metres short of each unclassified contact (nearest first)."""
        e = self.be.est()
        pts = []
        for c in sorted(report.get('unclassified_sonar_contacts', []), key=lambda c: c['range_m']):
            dx, dy = c['x'] - e['x'], c['y'] - e['y']
            d = math.hypot(dx, dy)
            if d < 0.5:
                continue
            k = max(0.0, (d - standoff) / d)
            pts.append((round(e['x'] + dx * k, 2), round(e['y'] + dy * k, 2), math.atan2(dy, dx)))
        return pts

    def ask_vlm(self, question: str) -> Optional[dict]:
        if self.vlm is None:
            return None
        img = self.be.sim.camera_image()
        t0 = time.time()
        try:
            ans = self.vlm.describe(img, question)
        except Exception as e:                      # VLM down / too slow: perception carries on
            ans = None
            err = f'{type(e).__name__}: {e}'
        else:
            err = None
        rec = {'t_sim': round(self.be.now(), 2), 'question': question, 'answer': ans,
               'latency_s': round(time.time() - t0, 2), 'error': err, 'model': getattr(self.vlm, 'model', '?')}
        self.vlm_log.append(rec)
        return rec


# ===================================================================== navigation
class NavigationAgent:
    def __init__(self, cfg: dict, backend, guidance_cfg: dict):
        self.cfg = cfg
        self.be = backend
        self.base_speed = 1.0

    def update(self, health: dict) -> dict:
        be = self.be
        if be.nav_source != 'ekf':
            return {'mode': 'ground_truth', 'pos_std_m': 0.0}
        s = be.ekf.summary()
        degraded = 'dead_reckoning' in health.get('degraded_modes', []) or \
            s['pos_std_m'] > self.cfg['dead_reckoning_pos_std_m']
        be.speed_scale = self.cfg['degraded_speed_factor'] if degraded else 1.0
        sources = [k for k, v in s['last_update_t'].items() if be.now() - v < 3.0]
        return {'mode': 'dead_reckoning' if degraded else 'nominal', 'pos_std_m': s['pos_std_m'],
                'yaw_std_deg': s['yaw_std_deg'], 'fused_sensors': sorted(sources),
                'speed_scale': be.speed_scale}


# ===================================================================== health (deterministic, veto)
class HealthAgent:
    def __init__(self, backend):
        self.be = backend

    def update(self, plan: Optional[dict], mission_t0: Optional[float]) -> dict:
        be = self.be
        rep = be.health.last
        out = rep.as_dict() if rep else {'faults': [], 'veto': False, 'veto_reason': '', 'degraded_modes': []}
        out = dict(out)
        b = be.battery_percent()
        out['battery_percent'] = round(b, 1) if b is not None else None
        why = check_abort_conditions(plan, b, (be.now() - mission_t0) if mission_t0 is not None else 0.0)
        if why and not out.get('veto'):
            out['veto'], out['veto_reason'] = True, why
        if be._veto and not out.get('veto'):
            out['veto'], out['veto_reason'] = True, be._veto
        return out


def check_abort_conditions(plan: Optional[dict], battery: Optional[float], elapsed: float) -> Optional[str]:
    for a in (plan or {}).get('abort_conditions', []):
        if a['type'] == 'battery_below' and battery is not None and battery < a['percent']:
            return f'battery {battery:.1f}% below abort threshold {a["percent"]:.0f}%'
        if a['type'] == 'mission_time_exceeded' and elapsed > a['seconds']:
            return f'mission time {elapsed:.0f}s exceeded limit {a["seconds"]:.0f}s'
    return None


# ===================================================================== diagnostics (LLM)
DIAG_SCHEMA = {
    'type': 'object',
    'properties': {
        'fault_types': {'type': 'array', 'items': {'type': 'string', 'enum': FAULTS}},
        'explanation': {'type': 'string'},
        'degraded_mode': {'type': 'string', 'enum': sorted(set(DEGRADED_MODE.values())) + ['none', 'surface_now']},
        'action': {'type': 'string', 'enum': ACTIONS},
    },
    'required': ['fault_types', 'explanation', 'degraded_mode', 'action'],
}

DIAG_SYSTEM = (
    'You are the diagnostics agent of an autonomous underwater vehicle. You receive telemetry and the '
    'health monitor\'s findings. Identify which faults are present (only from the allowed list), explain the '
    'evidence in one or two sentences, choose ONE degraded mode, and choose an action: continue (mission can '
    'go on as planned), replan (the remaining plan should change), or abort (surface now). A leak or a frozen '
    'depth sensor always means abort. Answer with JSON only, matching the schema.')


def deterministic_diagnosis(health: dict) -> dict:
    types = sorted({f['type'] for f in health.get('faults', [])})
    action = 'abort' if health.get('veto') or set(types) & VETO_FAULTS else (
        'replan' if set(types) & {'battery_drop', 'camera_blackout', 'turbidity'} else 'continue')
    modes = [DEGRADED_MODE[t] for t in types if t in DEGRADED_MODE]
    return {'fault_types': types, 'explanation': '; '.join(f['detail'] for f in health.get('faults', [])),
            'degraded_mode': 'surface_now' if action == 'abort' else (modes[0] if modes else 'none'),
            'action': action, 'source': 'rules'}


class DiagnosticsAgent:
    def __init__(self, provider=None, use_llm: bool = True):
        self.provider = provider if use_llm else None
        self.calls: list[dict] = []

    def diagnose(self, health: dict, nav: dict, perception: dict, telemetry: dict) -> dict:
        base = deterministic_diagnosis(health)
        if self.provider is None:
            return base
        msg = {'health_findings': health.get('faults', []), 'veto': health.get('veto'),
               'navigation': nav, 'perception': {k: perception.get(k) for k in
                                                 ('camera_visibility_m', 'suspected_false_targets', 'sonar_only_gate')},
               'telemetry': telemetry, 'allowed_faults': FAULTS}
        msgs = [{'role': 'system', 'content': DIAG_SYSTEM},
                {'role': 'user', 'content': json.dumps(msg, default=str)}]
        t0 = time.time()
        try:
            r = self.provider.complete(msgs, DIAG_SCHEMA)
            out = json.loads(r.text[r.text.find('{'):r.text.rfind('}') + 1])
            out['fault_types'] = [f for f in out.get('fault_types', []) if f in FAULTS]
            if out.get('action') not in ACTIONS:
                raise ValueError(f'bad action {out.get("action")!r}')
            # safety floor: the LLM may not downgrade a veto
            if base['action'] == 'abort':
                out['action'], out['degraded_mode'] = 'abort', 'surface_now'
            out['source'] = 'llm'
            self.calls.append({'usage': r.usage.as_dict(), 'latency_s': round(time.time() - t0, 2), 'ok': True})
            return out
        except Exception as e:
            self.calls.append({'error': f'{type(e).__name__}: {e}', 'latency_s': round(time.time() - t0, 2), 'ok': False})
            base['llm_error'] = str(e)[:200]
            return base


def telemetry_snapshot(backend) -> dict:
    r = backend.sim.readings
    th = r.get('thrusters', {})
    return {
        'dvl': {k: r.get('dvl', {}).get(k) for k in ('valid', 'altitude')},
        'pressure_depth': r.get('pressure', {}).get('depth'),
        'compass_yaw_deg': round(math.degrees(r.get('compass', {}).get('yaw', 0.0)), 1),
        'gyro_z': round(r.get('imu', {}).get('gyro', [0, 0, 0])[2], 4),
        'thrusters': {n: {'cmd_n': c, 'current_a': a} for n, c, a in
                      zip(th.get('names', []), th.get('cmd_n', []), th.get('current_a', []))},
        'battery': r.get('battery'), 'leak': r.get('leak', {}).get('leak'),
        'sonar_valid': r.get('sonar', {}).get('valid'),
        'ekf': backend.ekf.summary() if backend.nav_source == 'ekf' else None,
    }
