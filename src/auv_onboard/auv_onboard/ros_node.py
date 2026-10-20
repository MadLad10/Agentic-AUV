"""ROS 2 vehicle node: the MuJoCo sim + onboard stack (EKF, fusion tracker, health monitor, guidance,
control) as a ROS 2 Humble node with v1's interfaces, so any ROS 2 client drives it.

Serves (names from auv_mission_planner/config/auv.yaml, interfaces.sim, same as v1):
  actions  /auv/navigate_to  /auv/search  /auv/hold_depth  /auv/surface   (auv_mission_interfaces)
  service  /auv/emergency_surface (std_srvs/Trigger)
           /auv_sim/referee_report (std_srvs/Trigger, ground-truth JSON - evaluation only)
Publishes
  /clock (sim time)            /auv/odom (navigation ESTIMATE)      /auv_sim/odom_truth (truth, eval only)
  /auv/imu  /auv/dvl  /auv/depth  /auv/battery  /auv/detections
  /auv/health  /auv/nav_status  /auv/tracks  /auv/telemetry   (std_msgs/String JSON)
  /auv/sonar/image (mono8 fan)  /auv/camera/image (rgb8, param camera_hz, 0 = off)
Subscribes
  /auv/speed_scale (std_msgs/Float64): degraded-mode speed factor set by the navigation agent
There is NO thruster or setpoint topic: motion only through the actions above (safety boundary).

  ros2 run auv_onboard vehicle_node --ros-args -p arena:=default -p rtf:=1.0 -p faults:="dvl_dropout:15"
"""
from __future__ import annotations

import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import BatteryState, Image, Imu
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger

from auv_mission_interfaces.action import HoldDepth, NavigateTo, Search, Surface
from auv_mission_interfaces.msg import Detection, DetectionArray


def _time(ros_time, t):
    ros_time.sec = int(t)
    ros_time.nanosec = int((t - int(t)) * 1e9)


def _stamp(msg_header, t):
    _time(msg_header.stamp, t)


def parse_faults(spec: str) -> list[dict]:
    """'type:t_start[:k=v,...]' entries separated by ';' (ROS parameters are YAML, so no JSON lists)."""
    out = []
    for item in filter(None, (x.strip() for x in (spec or '').split(';'))):
        parts = item.split(':')
        f = {'type': parts[0], 't_start': float(parts[1]) if len(parts) > 1 else 0.0, 'params': {}}
        for kv in (parts[2].split(',') if len(parts) > 2 else []):
            k, v = kv.split('=')
            try:
                v = float(v)
            except ValueError:
                pass
            f['params'][k] = v
        out.append(f)
    return out


def _quat(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                              math.cos(yaw / 2), math.sin(yaw / 2))
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class VehicleNode(Node):
    def __init__(self):
        super().__init__('auv_vehicle')
        p = self.declare_parameter
        arena = p('arena', 'default').value
        seed = p('seed', 0).value
        rtf = p('rtf', 1.0).value
        faults = json.loads(p('faults_json', '[]').value or '[]')
        faults += parse_faults(p('faults', '').value)       # compact: "dvl_dropout:15;turbidity:0:visibility_m=2.5"
        self.camera_hz = p('camera_hz', 0.0).value
        overrides = json.loads(p('onboard_overrides_json', '{}').value or '{}')
        from auv_marine_sim.world import deep_update
        from auv_onboard.factory import make_backend
        ob = deep_update({'backend': {'max_rtf': rtf, 'idle_rtf': rtf}}, overrides)
        self.be = make_backend(arena, faults=faults, seed=seed, onboard_overrides=ob)
        cg = ReentrantCallbackGroup()
        self.pub = {
            'clock': self.create_publisher(Clock, '/clock', 10),
            'odom': self.create_publisher(Odometry, '/auv/odom', 10),
            'truth': self.create_publisher(Odometry, '/auv_sim/odom_truth', 10),
            'imu': self.create_publisher(Imu, '/auv/imu', 10),
            'dvl': self.create_publisher(TwistWithCovarianceStamped, '/auv/dvl', 10),
            'depth': self.create_publisher(Float64, '/auv/depth', 10),
            'battery': self.create_publisher(BatteryState, '/auv/battery', 10),
            'dets': self.create_publisher(DetectionArray, '/auv/detections', 10),
            'health': self.create_publisher(String, '/auv/health', 10),
            'nav': self.create_publisher(String, '/auv/nav_status', 10),
            'tracks': self.create_publisher(String, '/auv/tracks', 10),
            'telemetry': self.create_publisher(String, '/auv/telemetry', 10),
            'sonar': self.create_publisher(Image, '/auv/sonar/image', 2),
            'camera': self.create_publisher(Image, '/auv/camera/image', 2),
        }
        self.create_subscription(Float64, '/auv/speed_scale', self._on_speed, 10, callback_group=cg)
        kw = dict(callback_group=cg, goal_callback=lambda _: GoalResponse.ACCEPT,
                  cancel_callback=self._on_cancel)
        ActionServer(self, NavigateTo, '/auv/navigate_to', self._navigate, **kw)
        ActionServer(self, Search, '/auv/search', self._search, **kw)
        ActionServer(self, HoldDepth, '/auv/hold_depth', self._hold, **kw)
        ActionServer(self, Surface, '/auv/surface', self._surface, **kw)
        self.create_service(Trigger, '/auv/emergency_surface', self._emergency, callback_group=cg)
        self.create_service(Trigger, '/auv_sim/referee_report', self._referee, callback_group=cg)
        self._next = {}
        self._last_det_t = -1.0
        self._last_sonar_t = -1.0
        self.be.tick_hooks.append(self._on_tick)
        self.get_logger().info(f'vehicle ready: arena={arena} rtf={rtf} faults={[f["type"] for f in faults]}')

    # ------------------------------------------------------------------ publishing (sim tick hook)
    def _due(self, key, hz, t):
        if hz <= 0:
            return False
        if t + 1e-9 >= self._next.get(key, 0.0):
            self._next[key] = t + 1.0 / hz
            return True
        return False

    def _on_tick(self, be):
        t = be.now()
        clk = Clock()
        _time(clk.clock, t)
        self.pub['clock'].publish(clk)
        r = be.sim.readings
        if self._due('odom', 20, t):
            e = be.est()
            tr = be.sim.truth()
            for key, (x, y, z, yaw, v) in (('odom', (e['x'], e['y'], e['z'], e['yaw'], (e['vx'], e['vy'], e['vz']))),
                                           ('truth', (*tr.pos, tr.yaw, tr.vel_world))):
                m = Odometry()
                _stamp(m.header, t)
                m.header.frame_id, m.child_frame_id = 'world', 'base_link'
                m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = float(x), float(y), float(z)
                q = _quat(0.0, 0.0, float(yaw))
                m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, \
                    m.pose.pose.orientation.w = q
                m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = (float(c) for c in v)
                self.pub[key].publish(m)
            imu = r.get('imu')
            if imu:
                m = Imu()
                _stamp(m.header, t)
                m.header.frame_id = 'base_link'
                m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = (float(c) for c in imu['gyro'])
                m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = (float(c) for c in imu['accel'])
                q = _quat(imu['roll'], imu['pitch'], float(r.get('compass', {}).get('yaw', 0.0)))
                m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = q
                self.pub['imu'].publish(m)
        dvl = r.get('dvl')
        if dvl and dvl.get('valid') and self._due('dvl', 5, t):
            m = TwistWithCovarianceStamped()
            _stamp(m.header, t)
            m.header.frame_id = 'base_link'
            m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = (float(c) for c in dvl['vel'])
            self.pub['dvl'].publish(m)
        if self._due('slow', 2, t):
            if 'pressure' in r:
                self.pub['depth'].publish(Float64(data=float(r['pressure']['depth'])))
            b = r.get('battery')
            if b:
                m = BatteryState()
                _stamp(m.header, t)
                m.percentage, m.voltage = float(b['percent']) / 100.0, float(b['voltage'])
                self.pub['battery'].publish(m)
            self.pub['health'].publish(String(data=json.dumps(be.health.last.as_dict(), default=float)))
            self.pub['nav'].publish(String(data=json.dumps({**be.ekf.summary(), 'speed_scale': be.speed_scale},
                                                           default=float)))
            if be.tracker is not None:
                tk = be.tracker
                self.pub['tracks'].publish(String(data=json.dumps({
                    'tracks': [x.as_dict() for x in tk.confirmed()],
                    'unclassified': [x.id for x in tk.unclassified()],
                    'false_targets': [x.id for x in tk.false_target_suspects()],
                    'landmark_depth': be.landmark_depth}, default=float)))
            th = r.get('thrusters', {})
            self.pub['telemetry'].publish(String(data=json.dumps({
                'thrusters': {n: {'cmd_n': c, 'current_a': a} for n, c, a in
                              zip(th.get('names', []), th.get('cmd_n', []), th.get('current_a', []))},
                'dvl': {k: (dvl or {}).get(k) for k in ('valid', 'altitude')}, 'leak': r.get('leak', {}).get('leak'),
                'sonar_valid': r.get('sonar', {}).get('valid'), 'veto': be._veto}, default=float)))
        if be._det_t > self._last_det_t:
            self._last_det_t = be._det_t
            m = DetectionArray()
            _stamp(m.header, t)
            for d in be._dets:
                x = Detection(class_name=d.cls, confidence=float(d.confidence), range_m=float(d.range_m),
                              bearing_rad=math.radians(d.bearing_deg), source=be.detection_source)
                x.position.x, x.position.y, x.position.z = float(d.x), float(d.y), -float(d.depth)
                m.detections.append(x)
            self.pub['dets'].publish(m)
        sc = r.get('sonar')
        if sc and sc['t'] > self._last_sonar_t:
            self._last_sonar_t = sc['t']
            img, _ = be.sim.sonar_image(sc)
            if img is not None:
                self.pub['sonar'].publish(self._image(img, 'mono8', t))
        if self._due('camera', self.camera_hz, t):
            self.pub['camera'].publish(self._image(be.sim.camera_image(), 'rgb8', t))

    def _image(self, img, enc, t):
        m = Image()
        _stamp(m.header, t)
        m.height, m.width = img.shape[:2]
        m.encoding = enc
        m.step = int(img.strides[0])
        m.data = np.ascontiguousarray(img).tobytes()
        return m

    # ------------------------------------------------------------------ actions / services
    def _on_cancel(self, _goal):
        self.be.cancel_all()
        return CancelResponse.ACCEPT

    def _on_speed(self, msg):
        self.be.speed_scale = float(min(1.0, max(0.1, msg.data)))

    def _finish(self, gh, res, ok):
        if ok:
            gh.succeed()
        elif gh.is_cancel_requested:
            gh.canceled()
        else:
            gh.abort()
        return res

    def _navigate(self, gh):
        g = gh.request
        o = self.be.navigate(g.x, g.y, g.depth, g.yaw if g.use_yaw else None, g.tolerance_m or 0.35,
                             g.max_speed or None, timeout_s=1e6)
        res = NavigateTo.Result(success=o.ok, message=o.message,
                                final_error_m=float(o.data.get('final_error_m', -1.0)))
        return self._finish(gh, res, o.ok)

    def _search(self, gh):
        g = gh.request
        o = self.be.search(g.target_class, g.pattern or 'yaw_sweep', g.radius_m or 6.0, g.depth, timeout_s=1e6)
        res = Search.Result(found=bool(o.data.get('found')), message=o.message)
        d = o.data.get('detection')
        if d:
            res.detection = Detection(class_name=d['class'], confidence=float(d['confidence']),
                                      range_m=float(d['range_m']))
            res.detection.position.x, res.detection.position.y = float(d['position']['x']), float(d['position']['y'])
            res.detection.position.z = -float(d['position']['depth'])
        return self._finish(gh, res, o.ok)

    def _hold(self, gh):
        g = gh.request
        o = self.be.hold_depth(g.depth, g.duration_s, g.tolerance_m or 0.3, timeout_s=1e6)
        return self._finish(gh, HoldDepth.Result(success=o.ok, message=o.message,
                                                 mean_error_m=float(o.data.get('mean_error_m', -1.0))), o.ok)

    def _surface(self, gh):
        o = self.be.surface(timeout_s=1e6)
        return self._finish(gh, Surface.Result(success=o.ok, message=o.message), o.ok)

    def _emergency(self, _req, resp):
        o = self.be.emergency_surface()
        resp.success, resp.message = o.ok, o.message
        return resp

    def _referee(self, _req, resp):
        rep = self.be.sim.referee.report()
        resp.success, resp.message = True, json.dumps(rep, default=float)
        return resp

    def destroy_node(self):
        self.be.shutdown()
        super().destroy_node()


def main():
    rclpy.init()
    node = VehicleNode()
    ex = MultiThreadedExecutor(num_threads=8)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

