"""Health & safety monitor: deterministic rules over sensor readings and the EKF. Has veto power.

Each check maps telemetry to a fault label from the same closed set the fault injector uses, so
detection can be scored against injected ground truth (precision / recall / latency).

Veto (stop everything and surface): leak, frozen depth sensor, battery below the absolute floor,
both heave thrusters dead. Everything else produces a degraded mode the agents can plan around.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

VETO_FAULTS = {'leak', 'pressure_stuck'}
DEGRADED_MODE = {
    'dvl_dropout': 'dead_reckoning',          # IMU-only velocity: slow down, use landmarks, short legs
    'camera_blackout': 'sonar_only',          # no visual detection: sonar contacts only
    'turbidity': 'low_visibility',            # short camera range: approach slowly, rely on sonar
    'imu_bias': 'compass_heading',            # heading from compass, distrust gyro integration
    'compass_disturbance': 'gyro_heading',    # heading from gyro, compass rejected
    'sonar_dropout': 'camera_only',
    'sonar_false_target': 'ignore_unconfirmed_sonar',
    'thruster_dead': 'reallocated_thrust',    # allocator masks the thruster; reduced authority
    'battery_drop': 'return_early',
}


@dataclass
class HealthReport:
    t: float
    faults: list = field(default_factory=list)          # [{'type', 'detail', 'since'}]
    veto: bool = False
    veto_reason: str = ''
    degraded_modes: list = field(default_factory=list)
    dead_thrusters: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)

    def types(self) -> set:
        return {f['type'] for f in self.faults}


class HealthMonitor:
    def __init__(self, cfg: dict, thruster_names: list[str]):
        self.cfg = cfg
        self.names = thruster_names
        self.dvl_last_valid = None
        self.batt_hist: deque = deque()
        self.depth_hist: deque = deque()
        self.thr_bad_since: dict[str, float] = {}
        self.compass_hist: deque = deque()       # (t, accepted, rejected) cumulative
        self.since: dict[str, float] = {}
        self.camera_stats = None                 # (t, mean, deviation from water colour)
        self.camera_hist: list = []
        self.sonar_flags: dict = {}              # from perception fusion: {'false_target': bool}
        self.latched: set[str] = set()           # faults that stay reported once seen (thruster, leak)
        self.last = HealthReport(0.0)

    def note_camera(self, t, img, water_rgb=None):
        """Per-frame camera statistics: mean brightness and mean deviation from the water colour
        (a turbid image converges to the backscatter colour everywhere)."""
        if img is None:
            return
        mean = float(img.mean())
        dev = float(np.abs(img.astype(np.float32) - np.asarray(water_rgb if water_rgb is not None else img.mean((0, 1)),
                                                                np.float32)).mean())
        self.camera_hist.append((t, mean, dev))
        del self.camera_hist[:-self.cfg['camera_low_vis_frames']]
        self.camera_stats = (t, mean, dev)

    def update(self, t: float, readings: dict, ekf=None) -> HealthReport:
        c = self.cfg
        found: dict[str, str] = {}
        # --- DVL
        dvl = readings.get('dvl')
        if dvl:
            if dvl.get('valid'):
                self.dvl_last_valid = dvl['t']
            else:
                ref = self.dvl_last_valid if self.dvl_last_valid is not None else 0.0
                if t - ref >= c['dvl_timeout_s']:
                    found['dvl_dropout'] = f'no valid DVL velocity for {t - ref:.1f}s'
        # --- thrusters: commanded but no current drawn
        th = readings.get('thrusters')
        if th:
            for n, cmd, cur in zip(th['names'], th['cmd_n'], th['current_a']):
                if abs(cmd) >= c['thruster_min_cmd_n'] and cur < c['thruster_dead_current_a']:
                    self.thr_bad_since.setdefault(n, t)
                    if t - self.thr_bad_since[n] >= c['thruster_dead_window_s']:
                        self.latched.add(f'thruster_dead:{n}')
                elif abs(cmd) >= c['thruster_min_cmd_n']:
                    self.thr_bad_since.pop(n, None)
        dead = sorted(x.split(':', 1)[1] for x in self.latched if x.startswith('thruster_dead:'))
        if dead:
            found['thruster_dead'] = 'no current drawn by ' + ', '.join(dead)
        # --- pressure stuck (real sensor noise never repeats exactly)
        p = readings.get('pressure')
        if p:
            if not self.depth_hist or self.depth_hist[-1][0] != p['t']:
                self.depth_hist.append((p['t'], p['depth']))
            while self.depth_hist and t - self.depth_hist[0][0] > c['pressure_stuck_window_s']:
                self.depth_hist.popleft()
            vals = [d for _, d in self.depth_hist]
            if len(vals) >= 5 and max(vals) - min(vals) < 1e-9:
                self.latched.add('pressure_stuck')
        if 'pressure_stuck' in self.latched:
            found['pressure_stuck'] = 'depth reading frozen'
        # --- leak
        lk = readings.get('leak')
        if lk and lk.get('leak'):
            self.latched.add('leak')
        if 'leak' in self.latched:
            found['leak'] = 'leak sensor wet'
        # --- battery
        b = readings.get('battery')
        if b:
            if not self.batt_hist or self.batt_hist[-1][0] != b['t']:
                self.batt_hist.append((b['t'], b['percent']))
            while self.batt_hist and t - self.batt_hist[0][0] > c['battery_drop_window_s']:
                self.batt_hist.popleft()
            if self.batt_hist and self.batt_hist[0][1] - self.batt_hist[-1][1] >= c['battery_drop_pct']:
                self.latched.add('battery_drop')
        if 'battery_drop' in self.latched:
            found['battery_drop'] = 'sudden battery capacity loss'
        # --- sonar
        sn = readings.get('sonar')
        if sn is not None and not sn.get('valid', True):
            found['sonar_dropout'] = 'no sonar data'
        if self.sonar_flags.get('false_target'):
            found['sonar_false_target'] = self.sonar_flags.get('detail', 'unconfirmed sonar contact')
        # --- camera
        if self.camera_stats and t - self.camera_stats[0] < 3.0:
            _, mean, dev = self.camera_stats
            n = c['camera_low_vis_frames']
            if mean < c['camera_dark_mean']:
                found['camera_blackout'] = f'camera frames dark (mean {mean:.1f})'
            elif len(self.camera_hist) >= n and all(d < c['camera_low_vis_dev'] and m >= c['camera_dark_mean']
                                                    for _, m, d in self.camera_hist[-n:]):
                found['turbidity'] = f'image washed out to the water colour for {n} frames (dev {dev:.1f})'
        # --- EKF-derived: gyro bias, compass rejections
        if ekf is not None:
            if abs(ekf.s[7]) > c['gyro_bias_limit']:
                found['imu_bias'] = f'gyro z bias estimate {ekf.s[7]:+.4f} rad/s'
            st = ekf.stats['compass']
            self.compass_hist.append((t, st.accepted, st.rejected))
            while self.compass_hist and t - self.compass_hist[0][0] > 5.0:
                self.compass_hist.popleft()
            if len(self.compass_hist) > 2:
                da = self.compass_hist[-1][1] - self.compass_hist[0][1]
                dr = self.compass_hist[-1][2] - self.compass_hist[0][2]
                if da + dr >= 10 and dr / (da + dr) >= c['compass_reject_ratio']:
                    found['compass_disturbance'] = f'{dr}/{da + dr} compass readings rejected'
        for k in list(self.since):
            if k not in found:
                del self.since[k]
        faults = []
        for k, detail in found.items():
            self.since.setdefault(k, t)
            faults.append({'type': k, 'detail': detail, 'since': round(self.since[k], 2)})
        veto, why = False, ''
        vetoed = [k for k in found if k in VETO_FAULTS]
        if vetoed:
            veto, why = True, f'{vetoed[0]}: {found[vetoed[0]]}'
        if b and b['percent'] < c['veto_battery_pct']:
            veto, why = True, f'battery {b["percent"]:.1f}% below the {c["veto_battery_pct"]}% floor'
        if {'heave_front', 'heave_back'} <= set(dead):
            veto, why = True, 'both heave thrusters dead: no depth control'
        rep = HealthReport(round(t, 2), faults, veto, why,
                           sorted({DEGRADED_MODE[k] for k in found if k in DEGRADED_MODE}), dead)
        self.last = rep
        return rep

    def thruster_mask(self) -> np.ndarray:
        dead = {x.split(':', 1)[1] for x in self.latched if x.startswith('thruster_dead:')}
        return np.array([0.0 if n in dead else 1.0 for n in self.names])


def score_detection(reports: list[HealthReport], injected: list[dict], horizon_s: float = 30.0) -> dict:
    """Compare detected fault types with injected ground truth: per-fault detected?, latency."""
    out = {}
    for f in injected:
        hits = [r.t for r in reports if r.t >= f['t_start'] and f['type'] in r.types()
                and r.t <= f['t_start'] + horizon_s]
        out[f['type']] = {'detected': bool(hits), 'latency_s': round(min(hits) - f['t_start'], 2) if hits else None}
    injected_types = {f['type'] for f in injected}
    false_pos = sorted({k for r in reports for k in r.types()} - injected_types)
    out['_false_positives'] = false_pos
    return out


