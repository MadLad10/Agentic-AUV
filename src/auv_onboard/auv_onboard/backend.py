"""MarineSimBackend: v1's VehicleBackend interface on top of the MuJoCo sim and the onboard stack.

v1's tool registry, validator, executor and single-agent MissionRunner run on it unchanged, which
makes the v1-vs-v2 comparison fair (same vehicle, same sim, same tools).

One control tick (20 Hz sim time):
  sensors -> EKF (predict IMU, update DVL / pressure / compass / landmarks)
          -> perception (camera detector + sonar contacts, at perception_hz)
          -> health monitor (at health_hz; veto preempts the running action)
          -> guidance setpoint -> PID + allocation (health mask) -> thrusters -> MuJoCo step
Clock: actions run lockstep, capped at `max_rtf`; between actions a background ticker keeps the
sim running at `idle_rtf` with the vehicle holding station (time passes while agents think).
Guidance behaviours are a port of v1's vehicle_server (carrot following, turn-first, vertical
speed limit, search patterns, surface stall recovery), driven by the ESTIMATED pose.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Callable, Optional

import numpy as np

from auv_mission_planner.tools.backend import Det, Outcome, Pose, VehicleBackend
from auv_onboard.control import Controller, wrap
from auv_onboard.ekf import NavEKF
from auv_onboard.health import HealthMonitor
from auv_onboard.perception import GroundTruthCameraDetector, rel_to_world, sonar_contacts
from auv_onboard.tracking import FusionTracker, Meas, polar_cov


class Preempted(Exception):
    pass


class MarineSimBackend(VehicleBackend):
    def __init__(self, sim, cfg: dict, seed: int = 0, overrides: Optional[dict] = None, start_ticker=True):
        super().__init__()
        from auv_marine_sim.world import deep_update
        import copy
        self.cfg = copy.deepcopy(cfg)
        if overrides:
            deep_update(self.cfg, overrides)
        self.sim = sim
        self.rng = np.random.default_rng(seed + 1000)
        b = self.cfg['backend']
        self.nav_source = b['nav_source']
        self.detection_source = b['detection_source']
        self.health_enabled = b['health_monitor']
        self.max_rtf = b.get('max_rtf')
        self.idle_rtf = b.get('idle_rtf')
        self.g = self.cfg['guidance']
        v = sim.cfg['vehicle']
        self.ctrl = Controller(self.cfg['controller'], v['thrusters'], v['max_thrust_n'])
        tr = sim.truth()
        if self.cfg['ekf'].get('estimator', 'ekf') == 'fgo':       # factor graph (GTSAM ISAM2)
            from auv_onboard.fgo import FactorGraphNav
            self.ekf = FactorGraphNav(self.cfg['ekf'], tuple(tr.pos), tr.yaw, sim.cfg['physics']['gravity'])
        else:
            self.ekf = NavEKF(self.cfg['ekf'], tuple(tr.pos), tr.yaw, sim.cfg['physics']['gravity'])
        self.health = HealthMonitor(self.cfg['health'], sim.thruster_names)
        self.learned_vel = None
        self._next_learned = 0.0
        lv = self.cfg['ekf'].get('learned_velocity', {})
        if lv.get('enabled'):
            from auv_onboard.learned_velocity import LearnedVelocity, default_path
            path = lv.get('weights') or default_path()
            if path:
                self.learned_vel = LearnedVelocity(path, lv.get('std_inflation', 1.5))
        self.detector = self._make_detector()
        tcfg = self.cfg.get('tracking', {})
        self.tracker = FusionTracker(tcfg) if tcfg.get('enabled') else None
        self.sonar_detector = None
        if self.tracker is not None and tcfg.get('sonar_detection') == 'yolo':
            from auv_onboard.sonar_detector import YoloSonarDetector
            self.sonar_detector = YoloSonarDetector(sim)
        from collections import deque
        self._cam_ranges = deque(maxlen=400)        # (t, range) of recent camera buoy detections
        self.cam_offset = sim.cfg['sensors']['camera']['offset']
        self.sonar_offset = sim.cfg['sensors']['sonar']['offset']
        p = sim.arena['pool']
        m = self.g['fence_margin_m']
        x0 = p.get('x_min', -5.0)
        self.fence = (x0 + m, x0 + p['length'] - m, -p['width'] / 2 + m, p['width'] / 2 - m)
        self.hard_max_depth = self.cfg['controller']['hard_max_depth_m']
        # landmark map prior (gate pose from the course map, with map error)
        self.landmarks = {}
        self.landmark_depth = {}             # course-map depth prior (a 2D imaging sonar has no elevation)
        lm = self.cfg['ekf']['landmarks']
        if lm.get('gate_prior'):
            for o in sim.arena['objects']:
                if o['class'] == 'gate':
                    self.landmarks['gate'] = (o['x'] + self.rng.normal(0, lm['prior_std_m']),
                                              o['y'] + self.rng.normal(0, lm['prior_std_m']))
                    self.landmark_depth['gate'] = -o['z'] + self.rng.normal(0, lm['prior_std_m'])
        self._seen_t: dict[str, float] = {}
        self._dets: list[Det] = []
        self._det_t = -1.0
        self.contacts = []
        self.tracks_raw = []                 # last perception output (for the perception agent)
        self.health_log = []
        self.events: list[dict] = []
        self.tick_hooks: list[Callable] = []
        self.speed_scale = 1.0               # set by the navigation agent in degraded modes
        self._last_read_t: dict[str, float] = {}
        self._lock = threading.RLock()
        self._busy = 0
        self._token = 0
        self._veto = None
        self._next_perc = 0.0
        self._next_health = 0.0
        self._wall0, self._sim0 = time.time(), sim.t
        self._closed = False
        self._threads: list[threading.Thread] = []
        self.ctrl.set_mode('idle')
        self._ticker = None
        if start_ticker and self.idle_rtf:
            self._ticker = threading.Thread(target=self._idle_loop, daemon=True)
            self._ticker.start()

    def _make_detector(self):
        src = self.detection_source
        if src == 'ground_truth':
            return GroundTruthCameraDetector(self.cfg['perception'], self.sim, self.rng)
        if src == 'yolo':
            from auv_onboard.yolo_detector import YoloCameraDetector
            return YoloCameraDetector(self.cfg['perception'], self.sim)
        raise ValueError(f'unknown detection_source {src!r}')

    # ================================================================ estimation / state
    def est(self) -> dict:
        if self.nav_source == 'ground_truth':
            tr = self.sim.truth()
            c, s = math.cos(tr.yaw), math.sin(tr.yaw)
            return {'x': tr.pos[0], 'y': tr.pos[1], 'z': tr.pos[2], 'yaw': tr.yaw, 'vx': tr.vel_world[0],
                    'vy': tr.vel_world[1], 'vz': tr.vel_world[2], 'r': tr.omega_body[2],
                    'u': c * tr.vel_world[0] + s * tr.vel_world[1]}
        return self.ekf.state()

    def _fresh(self, name):
        r = self.sim.readings.get(name)
        if r is None or r['t'] <= self._last_read_t.get(name, -1.0):
            return None
        self._last_read_t[name] = r['t']
        return r

    def _nav_update(self):
        imu = self._fresh('imu')
        if imu:
            self.ekf.predict(imu)
            if self.learned_vel is not None:
                p = self.sim.readings.get('pressure')
                self.learned_vel.push(imu, self.sim.cmd, p['depth'] if p else 0.0, self.sim.control_dt)
                dvl = self.sim.readings.get('dvl') or {}
                if not dvl.get('valid') and self.learned_vel.ready() and \
                        self.sim.t + 1e-9 >= self._next_learned:
                    # DVL has no bottom lock: learned pseudo-DVL bounds the IMU dead-reckoning drift
                    self._next_learned = self.sim.t + 1.0 / self.cfg['ekf']['learned_velocity']['rate_hz']
                    self.ekf.update_velocity_pseudo(self.learned_vel.predict(), self.learned_vel.std)
        for name, fn in (('dvl', self.ekf.update_dvl), ('pressure', self.ekf.update_pressure),
                         ('compass', self.ekf.update_compass)):
            r = self._fresh(name)
            if r and not (name == 'pressure' and 'pressure_stuck' in self.health.latched):
                fn(r)                           # a sensor flagged frozen is no longer fused

    def _perceive(self):
        t = self.sim.t
        est = self.est()
        rel = self.detector.detect()
        dets = []
        for d in rel:
            x, y, depth = rel_to_world(d, est, self.cam_offset)
            dets.append(Det(d.cls, d.confidence, x, y, depth, d.range_m, math.degrees(d.bearing), t))
            if d.cls in self.landmarks and self.nav_source == 'ekf':
                lm = self.cfg['ekf']['landmarks']
                horiz = d.range_m * math.cos(d.elevation)
                # landmark measured from the camera; correct to the body origin (small offset)
                horiz += self.cam_offset[0] * math.cos(d.bearing)
                self.ekf.update_landmark(self.landmarks[d.cls], horiz, d.bearing,
                                         max(0.1, lm['camera_range_std_frac'] * horiz),
                                         math.radians(lm['camera_bearing_std_deg']))
        self._dets, self._det_t = dets, t
        scan = self.sim.readings.get('sonar')
        new_scan = bool(scan and scan['t'] > self._last_read_t.get('sonar_contacts', -1))
        if new_scan:
            self._last_read_t['sonar_contacts'] = scan['t']
            self.contacts = sonar_contacts(scan, est, self.sim.arena['pool'], self.sonar_offset)
        if self.tracker is None:
            if dets:
                self._notify(dets)
            return
        self._track(t, est, rel, new_scan)

    # ---------------------------------------------------------------- camera-sonar fusion tracking
    def _camera_view(self):
        """Onboard belief of where the camera reliably sees a buoy now (None = camera unusable).
        Uses the vehicle's OWN visibility estimate (nominal, or degraded once turbidity is detected),
        never the simulator's ground truth."""
        faults = self.health.last.types() if self.health_enabled else set()
        if 'camera_blackout' in faults:
            return None
        tc = self.cfg['tracking']
        vis = tc['camera_degraded_visibility_m'] if 'turbidity' in faults else tc['camera_nominal_visibility_m']
        # evidence-based reach: only expect the camera to see as far as it has recently PROVEN it can
        # (farthest buoy detection in the last window); no recent detection -> no expectation at all
        t = self.sim.t
        proven = [r for (tt, r) in self._cam_ranges if t - tt <= tc['visibility_witness_s']]
        if not proven:
            return None
        reach = min(tc['camera_reliable_frac'] * vis, max(proven))
        half = math.radians(self.sim.cfg['sensors']['camera']['hfov_deg'] / 2 - 5)
        e = self.est()

        def view(x, y):
            dx, dy = x - e['x'], y - e['y']
            r = math.hypot(dx, dy)
            return 0.5 < r < reach and abs(wrap(math.atan2(dy, dx) - e['yaw'])) < half
        return view

    def _sonar_meas(self, est, t):
        tc = self.cfg['tracking']
        nav = self.ekf.pos_std() if self.nav_source == 'ekf' else 0.0
        if self.sonar_detector is not None:          # learned FLS detector: classes come from the network
            out = []
            for cls, conf, c in self.sonar_detector.detect(est, self.sonar_offset):
                R = polar_cov(c.range_m, est['yaw'] + c.bearing, tc['sonar_range_std_m'],
                              math.radians(tc['sonar_bearing_std_deg']), nav)
                out.append(Meas(c.x, c.y, R, 'sonar', cls, conf, None, t))
            return out
        gate_w = next((o.get('width', 2.0) for o in self.sim.arena['objects'] if o['class'] == 'gate'), 2.0)
        cs = [c for c in self.contacts if c.range_m * max(c.width_rad, 1e-3) < tc['sonar_max_width_m']]
        out = []
        for c in cs:
            post = any(o is not c and abs(math.hypot(o.x - c.x, o.y - c.y) - gate_w) < tc['gate_pair_tolerance_m']
                       for o in cs)
            R = polar_cov(c.range_m, est['yaw'] + c.bearing, tc['sonar_range_std_m'],
                          math.radians(tc['sonar_bearing_std_deg']), nav)
            out.append(Meas(c.x, c.y, R, 'sonar', 'gate_post' if post else 'buoy', 0.6, None, t))
        return out

    def _track(self, t, est, rel, new_scan):
        tc = self.cfg['tracking']
        nav = self.ekf.pos_std() if self.nav_source == 'ekf' else 0.0
        meas = []
        for d in rel:
            x, y, depth = rel_to_world(d, est, self.cam_offset)
            horiz = d.range_m * math.cos(d.elevation)
            # visibility evidence: a buoy seen at r proves reach r; the big gate is visible further, so a
            # gate seen at r proves buoy reach of r * (buoy / gate visibility factor)
            pc = self.cfg['perception']
            k = pc['visibility_factor'] / pc['gate_visibility_factor'] if d.cls == 'gate' else 1.0
            self._cam_ranges.append((t, horiz * k))
            R = polar_cov(horiz, est['yaw'] + d.bearing, tc['camera_range_std_frac'] * horiz,
                          math.radians(tc['camera_bearing_std_deg']), nav)
            meas.append(Meas(x, y, R, 'camera', d.cls, d.confidence, depth, t))
        if new_scan:
            meas += self._sonar_meas(est, t)
        self.tracker.update(t, meas, self._camera_view())
        # the shared world model (object targets for the tools) gets FUSED, classified tracks
        fused = []
        for tr in self.tracker.confirmed():
            if tr.cls is None or tr.depth is None:
                continue
            dx, dy = tr.x[0] - est['x'], tr.x[1] - est['y']
            fused.append(Det(tr.cls, tr.confidence, float(tr.x[0]), float(tr.x[1]), float(tr.depth),
                             math.hypot(dx, dy), math.degrees(wrap(math.atan2(dy, dx) - est['yaw'])), t))
        if fused:
            self._notify(fused)
        sus = self.tracker.false_target_suspects()
        self.health.sonar_flags = {'false_target': bool(sus),
                                   'detail': (f'sonar track at ({sus[0].x[0]:.1f}, {sus[0].x[1]:.1f}) seen '
                                              f'{sus[0].sonar_hits}x by sonar, never by the camera') if sus else ''}

    def _health_tick(self):
        if self.sim.t % 1.0 < self.sim.control_dt / 2 + 1e-9:         # 1 Hz camera health frame
            img = self.sim.camera_image()
            water = np.array(self.sim.cfg['sensors']['camera']['water_rgb']) * 255.0
            self.health.note_camera(self.sim.t, img, water)
        rep = self.health.update(self.sim.t, self.sim.readings, self.ekf if self.nav_source == 'ekf' else None)
        self.health_log.append(rep)
        self.ctrl.mask = self.health.thruster_mask()
        if rep.veto and self._veto is None:
            self._veto = rep.veto_reason
            self.events.append({'t': self.sim.t, 'event': 'health_veto', 'reason': rep.veto_reason})
            self.emergency_surface()          # veto power: independent of any agent or plan

    def _tick(self):
        """One control tick. Caller holds the lock."""
        sim = self.sim
        self._nav_update()
        if sim.t + 1e-9 >= self._next_perc:
            self._next_perc = sim.t + 1.0 / self.cfg['backend']['perception_hz']
            self._perceive()
        if self.health_enabled and sim.t + 1e-9 >= self._next_health:
            self._next_health = sim.t + 1.0 / self.cfg['backend']['health_hz']
            self._health_tick()
        p = sim.readings.get('pressure')
        cmd = self.ctrl.compute(self.est(), sim.control_dt, p['depth'] if p else None)
        sim.set_thrust(cmd)
        sim.step()
        tr = sim.truth()
        if self.nav_source == 'ekf':
            sim.referee.record_estimate(sim.t, self.ekf.s[:3], tr.pos)
        for h in list(self.tick_hooks):
            try:
                h(self)
            except Exception as e:           # a telemetry/logging hook must never stop the vehicle loop
                import traceback
                traceback.print_exc()
                self.events.append({'t': sim.t, 'event': 'tick_hook_error', 'error': f'{type(e).__name__}: {e}'})
                self.tick_hooks.remove(h)

    def _throttle(self, rtf):
        if not rtf:
            return
        ahead = (self.sim.t - self._sim0) / rtf - (time.time() - self._wall0)
        if ahead > 0:
            time.sleep(min(ahead, 0.5))
        elif ahead < -1.0:                       # fell behind (slow host): don't try to catch up
            self._wall0, self._sim0 = time.time(), self.sim.t

    def _idle_loop(self):
        while not self._closed:
            if self._busy:
                time.sleep(0.01)
                continue
            with self._lock:
                if self._busy or self._closed:
                    continue
                if self.ctrl.mode == 'idle' and self.ctrl.sp is not None:
                    self.ctrl.set_mode('hold')
                self._tick()
            time.sleep(self.sim.control_dt / self.idle_rtf)

    # ================================================================ VehicleBackend API
    def now(self) -> float:
        return self.sim.t

    def pose(self) -> Optional[Pose]:
        e = self.est()
        return Pose(float(e['x']), float(e['y']), float(-e['z']), float(e['yaw']))

    def battery_percent(self) -> Optional[float]:
        b = self.sim.readings.get('battery')
        return b['percent'] if b else None

    def detections(self) -> list[Det]:
        return list(self._dets) if self.sim.t - self._det_t <= 0.5 else []

    def cancel_all(self) -> None:
        self._token += 1

    def shutdown(self) -> None:
        self._closed = True
        self.cancel_all()
        for th in self._threads + ([self._ticker] if self._ticker else []):
            th.join(timeout=5.0)
        with self._lock:
            self.sim.close()

    # ================================================================ action plumbing
    def _run_action(self, fn, timeout_s, *a, **kw) -> Outcome:
        self._token += 1
        token = self._token
        self._busy += 1
        t0 = self.sim.t
        self._wall0, self._sim0 = time.time(), self.sim.t
        try:
            deadline = t0 + timeout_s
            o = fn(token, deadline, *a, **kw)
            o.ok = bool(o.ok)
            return o
        except Preempted as e:
            if token == self._token and not self._veto:     # nothing newer took over: hold station
                with self._lock:
                    self._hold_here()
            return Outcome(False, str(e), 'canceled' if 'veto' not in str(e) else 'health_veto')
        except TimeoutError as e:
            if token == self._token:
                with self._lock:
                    self._hold_here()
            return Outcome(False, str(e), 'timeout')
        finally:
            self._busy -= 1

    def step_once(self, token, deadline):
        if self._closed:
            raise Preempted('backend shut down (canceled)')
        if self._veto and token != getattr(self, '_veto_token', None):
            raise Preempted(f'health veto: {self._veto}')
        if token != self._token:
            raise Preempted('preempted by a newer command (canceled)')
        if self.sim.t >= deadline:
            raise TimeoutError(f'timed out after {deadline:.0f}s sim time')
        with self._lock:
            self._tick()
        self._throttle(self.max_rtf)

    def _set_sp(self, x, y, z, yaw):
        self.ctrl.sp = (x, y, z, yaw)

    def _hold_here(self):
        e = self.est()
        self._set_sp(e['x'], e['y'], min(e['z'], 0.0), e['yaw'])
        if self.ctrl.mode != 'surface':
            self.ctrl.set_mode('hold')

    def _in_fence(self, x, y):
        x0, x1, y0, y1 = self.fence
        return x0 <= x <= x1 and y0 <= y <= y1

    def _drive(self, token, deadline, tx, ty, tz, tyaw=None, tol=0.3, speed=None, stop_when=None):
        """Carrot-follow to (tx, ty, tz) on the estimated pose. Returns ('reached'|'stopped', err)."""
        g = self.g
        speed = min(speed or g['max_speed'], g['max_speed']) * self.speed_scale
        e = self.est()
        cx, cy, cz, cyaw = e['x'], e['y'], e['z'], e['yaw']
        dt = self.sim.control_dt
        settled = None
        self.ctrl.set_mode('hold')
        while True:
            if stop_when is not None and stop_when():
                return 'stopped', 0.0
            e = self.est()
            x, y, z, yaw = e['x'], e['y'], e['z'], e['yaw']
            dxy = math.hypot(tx - x, ty - y)
            want = math.atan2(ty - y, tx - x) if dxy > 1.0 else (tyaw if tyaw is not None else cyaw)
            dyaw = wrap(want - cyaw)
            cyaw = wrap(cyaw + max(-g['max_yaw_rate'] * dt, min(g['max_yaw_rate'] * dt, dyaw)))
            turning = abs(wrap(want - yaw)) > math.radians(g['turn_first_deg']) and dxy > 1.0
            lag = math.dist((cx, cy, cz), (x, y, z))
            vx, vy, vz = tx - cx, ty - cy, tz - cz
            vd = math.sqrt(vx * vx + vy * vy + vz * vz)
            if vd > 1e-6 and not turning and lag < g['carrot_lead_m']:
                step = min(vd, speed * dt)
                if abs(vz) > 1e-6:                 # gentle vertical approach: no depth overshoot
                    step = min(step, g['max_vertical_speed'] * dt * vd / abs(vz))
                cx, cy, cz = cx + vx / vd * step, cy + vy / vd * step, cz + vz / vd * step
            self._set_sp(cx, cy, cz, cyaw)
            self.step_once(token, deadline)
            e = self.est()
            err = math.dist((e['x'], e['y'], e['z']), (tx, ty, tz))
            yaw_ok = tyaw is None or abs(wrap(tyaw - e['yaw'])) < 0.2
            if err < tol and yaw_ok:
                settled = settled if settled is not None else self.sim.t
                if self.sim.t - settled >= g['settle_s']:
                    return 'reached', err
            else:
                settled = None

    # ================================================================ actions
    def navigate(self, x, y, depth, yaw=None, tolerance=0.35, speed=None, timeout_s=60.0) -> Outcome:
        if not (0.0 <= depth <= self.hard_max_depth) or not self._in_fence(x, y):
            return Outcome(False, f'target ({x:.1f},{y:.1f},{depth:.1f}m) outside the safe envelope', 'unsafe_target')

        def run(token, deadline):
            _, err = self._drive(token, deadline, x, y, -depth, yaw, tol=tolerance or 0.3, speed=speed)
            return Outcome(True, 'reached', data={'final_error_m': round(err, 3)})
        o = self._run_action(run, timeout_s)
        if not o.ok and 'final_error_m' not in o.data:
            e = self.est()
            o.data['final_error_m'] = round(math.dist((e['x'], e['y'], -e['z']), (x, y, depth)), 3)
        return o

    def _seen(self, cls, since):
        best = None
        for d in self._dets:
            if d.cls == cls and d.t >= since and d.confidence >= 0.4 and (best is None or d.confidence > best.confidence):
                best = d
        return best

    def _pattern(self, pattern, radius, depth):
        e = self.est()
        x, y, yaw = e['x'], e['y'], e['yaw']
        z = -depth
        wps = []

        def sweep(px, py, base):
            return [(px, py, z, wrap(base + k * math.pi / 3)) for k in range(1, 7)]
        if pattern == 'yaw_sweep':
            wps += sweep(x, y, yaw)
        elif pattern == 'spiral':
            wps += sweep(x, y, yaw)
            r, k = 2.0, 0
            dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            px, py = x, y
            while r <= radius + 1e-6:
                for _ in range(2):
                    dx, dy = dirs[k % 4]
                    px, py = px + dx * r, py + dy * r
                    wps.append((px, py, z, None))
                    if r >= 4.0:
                        wps += sweep(px, py, math.atan2(dy, dx))
                    k += 1
                r += 2.0
        elif pattern == 'lawnmower':
            c, s = math.cos(yaw), math.sin(yaw)
            rows = max(2, int(2 * radius / 3.0) + 1)
            for i in range(rows):
                lat = -radius + i * (2 * radius / (rows - 1))
                ends = [(0.0, lat), (2 * radius, lat)]
                if i % 2:
                    ends.reverse()
                for fwd, l in ends:
                    wps.append((x + c * fwd - s * l, y + s * fwd + c * l, z, None))
        else:
            raise ValueError(pattern)
        x0, x1, y0, y1 = self.fence
        return [(min(max(a, x0), x1), min(max(b, y0), y1), c_, d) for a, b, c_, d in wps]

    def search(self, target, pattern, radius_m, depth, timeout_s=60.0) -> Outcome:
        if not (0.0 <= depth <= self.hard_max_depth):
            return Outcome(False, f'depth {depth} outside the safe envelope', 'unsafe_target')
        try:
            wps = self._pattern(pattern or 'yaw_sweep', max(2.0, radius_m or 6.0), depth)
        except ValueError:
            return Outcome(False, f'unknown pattern {pattern!r} (yaw_sweep|spiral|lawnmower)', 'invalid_params')

        def run(token, deadline):
            start = self.sim.t
            hits = {'n': 0, 'det': None, 't': None}

            def found():
                d = self._seen(target, start)
                if d is None:
                    return False
                if d.t != hits['t']:
                    hits['t'], hits['n'], hits['det'] = d.t, hits['n'] + 1, d
                return hits['n'] >= 2
            for wx, wy, wz, wyaw in wps:
                status, _ = self._drive(token, deadline, wx, wy, wz, wyaw, tol=0.4 if wyaw is None else 0.5,
                                        stop_when=found)
                if status == 'stopped':
                    self._hold_here()
                    return Outcome(True, f'{target} found after {self.sim.t - start:.0f}s',
                                   data={'found': True, 'detection': hits['det'].as_dict()})
            self._hold_here()
            return Outcome(False, f'{target} not found with {pattern} pattern (radius {radius_m:.1f} m)',
                           'not_found', {'found': False})
        o = self._run_action(run, timeout_s)
        o.data.setdefault('found', o.ok)
        return o


    def hold_depth(self, depth, duration_s, tolerance=0.3, timeout_s=60.0) -> Outcome:
        if not (0.0 <= depth <= self.hard_max_depth):
            return Outcome(False, f'depth {depth} outside the safe envelope', 'unsafe_target')

        def run(token, deadline):
            e = self.est()
            x, y, yaw = e['x'], e['y'], e['yaw']
            self._drive(token, deadline, x, y, -depth, yaw, tol=tolerance)
            t0 = self.sim.t
            errs = []
            while self.sim.t - t0 < duration_s:
                self._set_sp(x, y, -depth, yaw)
                self.step_once(token, deadline)
                errs.append(abs(-self.est()['z'] - depth))
            mean = float(sum(errs) / len(errs)) if errs else 0.0
            ok = bool(mean < tolerance)
            return Outcome(ok, f'held {depth:.2f} m for {duration_s:.0f}s (mean err {mean:.2f})',
                           None if ok else 'action_failed', {'mean_error_m': round(mean, 3)})
        return self._run_action(run, timeout_s)

    def _surface_loop(self, token, deadline):
        """Ascend; if the ascent stalls (e.g. under the gate crossbar), move clear and retry."""
        g = self.g
        e = self.est()
        self._set_sp(e['x'], e['y'], 0.0, e['yaw'])
        self.ctrl.set_mode('surface')
        if 'pressure_stuck' in self.health.latched:
            # depth is untrusted: blind ascent (positive buoyancy + up-thrust) for a fixed time
            self.ctrl.blind_ascent = True
            end = self.sim.t + g['blind_surface_s']
            try:
                while self.sim.t < end:
                    self.step_once(token, deadline)
            finally:
                self.ctrl.blind_ascent = False
            return True
        since = None
        best, best_t = -e['z'], self.sim.t
        nudges = 0
        while True:
            self.step_once(token, deadline)
            d = -self.est()['z']
            if d < 0.2:
                since = since if since is not None else self.sim.t
                if self.sim.t - since > g['surface_confirm_s']:      # must STAY up (a hull bobs)
                    return True
                continue
            since = None
            if d < best - 0.05:
                best, best_t = d, self.sim.t
            elif self.sim.t - best_t > g['surface_stall_s'] and nudges < 4:
                nudges += 1
                sign = 1.0 if nudges % 2 else -1.0
                e = self.est()
                tx = e['x'] + sign * g['surface_clear_m'] * math.cos(e['yaw'])
                ty = e['y'] + sign * g['surface_clear_m'] * math.sin(e['yaw'])
                self.events.append({'t': self.sim.t, 'event': 'surface_stall', 'depth': round(d, 2)})
                self._set_sp(tx, ty, e['z'], e['yaw'])
                self.ctrl.set_mode('hold')
                end = self.sim.t + 15.0
                while self.sim.t < end and math.dist((self.est()['x'], self.est()['y']), (tx, ty)) > 0.3:
                    self.step_once(token, deadline)
                e = self.est()
                self._set_sp(e['x'], e['y'], 0.0, e['yaw'])
                self.ctrl.set_mode('surface')
                best, best_t = -e['z'], self.sim.t

    def surface(self, timeout_s=60.0) -> Outcome:
        def run(token, deadline):
            self._veto_token = token                     # surfacing is what a veto asks for
            self._surface_loop(token, deadline)
            return Outcome(True, 'surfaced')
        return self._run_action(run, timeout_s)

    def emergency_surface(self) -> Outcome:
        """Preempt everything and surface in the background (like v1's emergency service)."""
        self.cancel_all()

        def bg():
            def run(token, deadline):
                self._veto_token = token                 # the abort itself is never vetoed
                self._surface_loop(token, deadline)
                return Outcome(True, 'surfaced')
            self._run_action(run, 180.0)
        th = threading.Thread(target=bg, daemon=True)
        self._threads = [t for t in self._threads if t.is_alive()] + [th]
        th.start()
        return Outcome(True, 'surfacing')

    # ================================================================ helpers for agents / eval
    def run_until_surfaced(self, timeout_s=120.0) -> bool:
        """Blocking safe surface (used by the v2 graph's abort node and tests)."""
        o = self.surface(timeout_s)
        return o.ok

    def health_report(self):
        return self.health.last

    def clear_veto(self):
        self._veto = None
