"""High-level vehicle actions for the stand-in sim (plays the role of AUV's `control` action).

Actions:  /auv/navigate_to (NavigateTo), /auv/search (Search),
          /auv/hold_depth (HoldDepth), /auv/surface (Surface)
Service:  /auv/emergency_surface (std_srvs/Trigger) - preempt everything and surface
Outputs:  /auv/controller/setpoint, /auv/controller/mode  (consumed by controller_node only)

Only one goal runs at a time; a new goal preempts the running one.
"""
import math
import threading
import time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from auv_mission_interfaces.action import HoldDepth, NavigateTo, Search, Surface
from auv_mission_interfaces.msg import Detection, DetectionArray
from auv_sim.arena import load_arena, wrap, yaw_from_quat
from auv_sim.ros_util import spin_node

TICK = 0.05          # wall seconds per control tick
CARROT_LEAD = 1.5    # m: carrot never runs further ahead of the vehicle than this
TURN_FIRST = math.radians(45)


class Preempted(Exception):
    pass


class VehicleServer(Node):
    def __init__(self):
        super().__init__('auv_vehicle_server')
        self.max_speed = self.declare_parameter('max_speed', 0.6).value
        self.max_yaw_rate = self.declare_parameter('max_yaw_rate', 0.6).value
        self.settle_s = self.declare_parameter('settle_s', 1.0).value
        self.max_vz = self.declare_parameter('max_vertical_speed', 0.3).value
        self.stall_s = self.declare_parameter('surface_stall_s', 4.0).value       # no ascent progress -> stalled
        self.clear_m = self.declare_parameter('surface_clear_m', 1.5).value       # horizontal move to get clear
        self.hard_max_depth = self.declare_parameter('hard_max_depth_m', 4.6).value
        arena = load_arena(self.declare_parameter('arena_file', 'default').value)
        p = arena['pool']
        m = 0.8   # keep-out margin from walls
        self.fence = (p.get('x_min', -5.0) + m, p.get('x_min', -5.0) + p['length'] - m,
                      -p['width'] / 2 + m, p['width'] / 2 - m)

        self.pose = None
        self.detections = []
        self.det_stamp = 0.0
        self.lock = threading.Lock()
        self.active = 0            # token of the running goal; bump to preempt
        self.sp = None             # current setpoint (x, y, z, yaw)

        cg = ReentrantCallbackGroup()
        self.create_subscription(Odometry, '/auv/odom', self.on_odom, 20, callback_group=cg)
        self.create_subscription(DetectionArray, '/auv/detections', self.on_det, 10, callback_group=cg)
        self.sp_pub = self.create_publisher(PoseStamped, '/auv/controller/setpoint', 10)
        self.mode_pub = self.create_publisher(String, '/auv/controller/mode', 10)
        kw = dict(callback_group=cg, goal_callback=self.on_goal, cancel_callback=lambda _: CancelResponse.ACCEPT)
        ActionServer(self, NavigateTo, '/auv/navigate_to', self.exec_navigate, **kw)
        ActionServer(self, Search, '/auv/search', self.exec_search, **kw)
        ActionServer(self, HoldDepth, '/auv/hold_depth', self.exec_hold_depth, **kw)
        ActionServer(self, Surface, '/auv/surface', self.exec_surface, **kw)
        self.create_service(Trigger, '/auv/emergency_surface', self.on_emergency, callback_group=cg)
        self.mode = 'idle'
        self.create_timer(0.1, self.republish, callback_group=cg)   # keep controller fed

    # ---------------------------------------------------------------- inputs
    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = (p.x, p.y, p.z, yaw_from_quat(q))

    def on_det(self, msg):
        self.detections = list(msg.detections)
        self.det_stamp = self.now()

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_goal(self, _goal):
        return GoalResponse.ACCEPT if self.pose is not None else GoalResponse.REJECT

    # ---------------------------------------------------------------- outputs
    def set_mode(self, mode):
        self.mode = mode
        self.mode_pub.publish(String(data=mode))

    def set_sp(self, x, y, z, yaw):
        self.sp = (x, y, z, yaw)
        self._pub_sp()

    def _pub_sp(self):
        if self.sp is None:
            return
        m = PoseStamped()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y, m.pose.position.z = self.sp[:3]
        m.pose.orientation.z, m.pose.orientation.w = math.sin(self.sp[3] / 2), math.cos(self.sp[3] / 2)
        self.sp_pub.publish(m)

    def republish(self):
        if self.mode != 'idle':
            self._pub_sp()
            self.mode_pub.publish(String(data=self.mode))

    # ---------------------------------------------------------------- goal plumbing
    def _pose_str(self):
        if self.pose is None:
            return 'none'
        x, y, z, yaw = self.pose
        return f'({x:.2f}, {y:.2f}, depth {-z:.2f}, yaw {math.degrees(yaw):.0f})'

    def _claim(self):
        self.get_logger().info(f'goal start (mode {self.mode}) pose={self._pose_str()}')
        with self.lock:
            self.active += 1
            return self.active

    def _check(self, token, gh):
        if token != self.active:
            raise Preempted('preempted by a newer goal')
        if gh.is_cancel_requested:
            raise Preempted('canceled')

    def _hold_here(self):
        x, y, z, yaw = self.pose
        self.set_sp(x, y, min(z, -0.0), yaw)
        if self.mode != 'surface':
            self.set_mode('hold')

    def _in_fence(self, x, y):
        x0, x1, y0, y1 = self.fence
        return x0 <= x <= x1 and y0 <= y <= y1

    def _drive(self, token, gh, tx, ty, tz, tyaw=None, tol=0.3, speed=None, stop_when=None, feedback=None):
        """Carrot-follow to (tx, ty, tz). Returns ('reached'|'stopped', err). Raises Preempted."""
        speed = min(speed or self.max_speed, self.max_speed)
        x, y, z, yaw = self.pose
        cx, cy, cz = x, y, z
        cyaw = yaw
        last = self.now()
        settled_since = None
        self.set_mode('hold')
        while True:
            self._check(token, gh)
            if stop_when is not None and stop_when():
                return 'stopped', 0.0
            t = self.now()
            dt = max(0.0, min(0.5, t - last))
            last = t
            x, y, z, yaw = self.pose
            # heading: face travel direction while far away, else requested/current
            dxy = math.hypot(tx - x, ty - y)
            want_yaw = math.atan2(ty - y, tx - x) if dxy > 1.0 else (tyaw if tyaw is not None else cyaw)
            if tyaw is not None and dxy <= 1.0:
                want_yaw = tyaw
            dyaw = wrap(want_yaw - cyaw)
            cyaw = wrap(cyaw + max(-self.max_yaw_rate * dt, min(self.max_yaw_rate * dt, dyaw)))
            # advance carrot unless we still need to turn a lot or the vehicle lags behind
            turning = abs(wrap(want_yaw - yaw)) > TURN_FIRST and dxy > 1.0
            lag = math.dist((cx, cy, cz), (x, y, z))
            vx, vy, vz = tx - cx, ty - cy, tz - cz
            vd = math.sqrt(vx * vx + vy * vy + vz * vz)
            if vd > 1e-6 and not turning and lag < CARROT_LEAD:
                step = min(vd, speed * dt)
                if abs(vz) > 1e-6:                       # gentle vertical approach: no depth overshoot
                    step = min(step, self.max_vz * dt * vd / abs(vz))
                cx, cy, cz = cx + vx / vd * step, cy + vy / vd * step, cz + vz / vd * step
            self.set_sp(cx, cy, cz, cyaw)
            err = math.dist((x, y, z), (tx, ty, tz))
            if feedback:
                feedback(err)
            yaw_ok = tyaw is None or abs(wrap(tyaw - yaw)) < 0.2
            if err < tol and yaw_ok:
                settled_since = settled_since or t
                if t - settled_since >= self.settle_s:
                    return 'reached', err
            else:
                settled_since = None
            time.sleep(TICK)

    def _finish(self, gh, result, ok):
        self.get_logger().info(f'goal finished ok={ok}: {getattr(result, "message", "")} pose={self._pose_str()}')
        if ok:
            gh.succeed()
        elif gh.is_cancel_requested:
            gh.canceled()
        else:
            gh.abort()
        return result

    # ---------------------------------------------------------------- actions
    def exec_navigate(self, gh):
        g = gh.request
        token = self._claim()
        res = NavigateTo.Result()
        if not (0.0 <= g.depth <= self.hard_max_depth) or not self._in_fence(g.x, g.y):
            res.message = f'target ({g.x:.1f},{g.y:.1f},{g.depth:.1f}m) outside the safe envelope'
            res.final_error_m = -1.0
            return self._finish(gh, res, False)
        fb = NavigateTo.Feedback()

        def feedback(err):
            fb.distance_remaining = err
            gh.publish_feedback(fb)
        try:
            _, err = self._drive(token, gh, g.x, g.y, -g.depth, g.yaw if g.use_yaw else None,
                                 tol=g.tolerance_m or 0.3, speed=g.max_speed or None, feedback=feedback)
            res.success, res.final_error_m, res.message = True, err, 'reached'
            return self._finish(gh, res, True)
        except Preempted as e:
            res.message = str(e)
            res.final_error_m = math.dist(self.pose[:3], (g.x, g.y, -g.depth))
            if token == self.active:
                self._hold_here()
            return self._finish(gh, res, False)

    def _seen(self, cls, since):
        if self.det_stamp < since:
            return None
        best = None
        for d in self.detections:
            if d.class_name == cls and d.confidence >= 0.4 and (best is None or d.confidence > best.confidence):
                best = d
        return best

    def _pattern(self, pattern, radius, depth):
        x, y, _, yaw = self.pose
        z = -depth
        wps = []   # (x, y, z, yaw-or-None)
        def sweep(px, py, base):
            return [(px, py, z, wrap(base + k * math.pi / 3)) for k in range(1, 7)]
        if pattern == 'yaw_sweep':
            wps += sweep(x, y, yaw)
        elif pattern == 'spiral':
            # look around first, then square spiral outwards, looking around at each outer corner
            wps += sweep(x, y, yaw)
            step, r, k = 2.0, 2.0, 0
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
                r += step
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

    def exec_search(self, gh):
        g = gh.request
        token = self._claim()
        res = Search.Result()
        pattern = g.pattern or 'yaw_sweep'
        depth = g.depth if g.depth > 0 else max(0.5, -self.pose[2])
        if not (0.0 <= depth <= self.hard_max_depth):
            res.message = f'depth {depth} outside the safe envelope'
            return self._finish(gh, res, False)
        try:
            wps = self._pattern(pattern, max(2.0, g.radius_m or 6.0), depth)
        except ValueError:
            res.message = f'unknown pattern {pattern!r} (yaw_sweep|spiral|lawnmower)'
            return self._finish(gh, res, False)
        start = self.now()
        hits = {'n': 0, 'det': None}

        def found():
            d = self._seen(g.target_class, start)
            if d is None:
                hits['n'] = 0
                return False
            if self.det_stamp != hits.get('stamp'):
                hits['stamp'] = self.det_stamp
                hits['n'] += 1
                hits['det'] = d
            return hits['n'] >= 2
        fb = Search.Feedback()
        try:
            for i, (wx, wy, wz, wyaw) in enumerate(wps):
                fb.progress = i / len(wps)
                gh.publish_feedback(fb)
                tol = 0.4 if wyaw is None else 0.5
                status, _ = self._drive(token, gh, wx, wy, wz, wyaw, tol=tol, stop_when=found)
                if status == 'stopped':
                    self._hold_here()
                    res.found, res.detection = True, hits['det']
                    res.message = f'{g.target_class} found after {self.now() - start:.0f}s'
                    return self._finish(gh, res, True)
            res.message = f'{g.target_class} not found with {pattern} pattern (radius {g.radius_m:.1f} m)'
            res.detection = Detection()
            self._hold_here()
            return self._finish(gh, res, False)
        except Preempted as e:
            res.message = str(e)
            if token == self.active:
                self._hold_here()
            return self._finish(gh, res, False)

    def exec_hold_depth(self, gh):
        g = gh.request
        token = self._claim()
        res = HoldDepth.Result()
        if not (0.0 <= g.depth <= self.hard_max_depth):
            res.message = f'depth {g.depth} outside the safe envelope'
            return self._finish(gh, res, False)
        x, y, _, yaw = self.pose
        tol = g.tolerance_m or 0.3
        try:
            self._drive(token, gh, x, y, -g.depth, yaw, tol=tol)
            t0 = self.now()
            errs = []
            fb = HoldDepth.Feedback()
            while self.now() - t0 < g.duration_s:
                self._check(token, gh)
                self.set_sp(x, y, -g.depth, yaw)
                errs.append(abs(-self.pose[2] - g.depth))
                fb.current_depth = -self.pose[2]
                gh.publish_feedback(fb)
                time.sleep(TICK)
            res.mean_error_m = sum(errs) / len(errs) if errs else abs(-self.pose[2] - g.depth)
            res.success = res.mean_error_m < tol
            res.message = f'held {g.depth:.2f} m for {g.duration_s:.0f}s (mean err {res.mean_error_m:.2f})'
            return self._finish(gh, res, res.success)
        except Preempted as e:
            res.message = str(e)
            return self._finish(gh, res, False)

    def _surface_loop(self, token, gh=None, timeout_s=None):
        """Ascend; if the ascent stalls (something overhead, e.g. the gate crossbar), move clear
        horizontally and try again. Returns True when on the surface. Raises Preempted."""
        x, y, _, yaw = self.pose
        self.set_sp(x, y, 0.0, yaw)
        self.set_mode('surface')
        fb = Surface.Feedback()
        t0 = self.now()
        since = None
        best, best_t = -self.pose[2], t0
        nudges = 0
        while timeout_s is None or self.now() - t0 < timeout_s:
            if gh is not None:
                self._check(token, gh)
                fb.current_depth = -self.pose[2]
                gh.publish_feedback(fb)
            elif token != self.active:
                raise Preempted('preempted by a newer goal')
            d = -self.pose[2]
            if d < 0.2:
                since = since or self.now()
                if self.now() - since > 1.0:
                    return True
                time.sleep(TICK)
                continue
            since = None
            if d < best - 0.05:
                best, best_t = d, self.now()
            elif self.now() - best_t > self.stall_s and nudges < 4:
                nudges += 1
                sign = 1.0 if nudges % 2 else -1.0          # forward first, then backward
                px, py, pz, pyaw = self.pose
                tx = px + sign * self.clear_m * math.cos(pyaw)
                ty = py + sign * self.clear_m * math.sin(pyaw)
                self.get_logger().warn(f'ascent stalled at {d:.2f} m: moving {sign * self.clear_m:+.1f} m to clear')
                self.set_sp(tx, ty, pz, pyaw)
                self.set_mode('hold')
                end = self.now() + 15.0
                while self.now() < end and math.dist(self.pose[:2], (tx, ty)) > 0.3:
                    if gh is not None:
                        self._check(token, gh)
                    time.sleep(TICK)
                self.set_sp(self.pose[0], self.pose[1], 0.0, pyaw)
                self.set_mode('surface')
                best, best_t = -self.pose[2], self.now()
            time.sleep(TICK)
        return False

    def exec_surface(self, gh):
        token = self._claim()
        res = Surface.Result()
        try:
            ok = self._surface_loop(token, gh)
            res.success, res.message = ok, 'surfaced' if ok else 'surface timed out'
            return self._finish(gh, res, ok)
        except Preempted as e:
            res.message = str(e)
            return self._finish(gh, res, False)

    def on_emergency(self, _req, resp):
        self._claim()                     # preempt whatever runs
        if self.pose is not None:
            x, y, _, yaw = self.pose
            self.set_sp(x, y, 0.0, yaw)
        self.set_mode('surface')
        self.get_logger().warn('EMERGENCY SURFACE requested')
        token = self.active

        def run():
            try:
                self._surface_loop(token, None, timeout_s=180.0)
            except Preempted:
                pass
        threading.Thread(target=run, daemon=True).start()
        resp.success, resp.message = True, 'surfacing'
        return resp


def main():
    spin_node(VehicleServer, threads=6)
