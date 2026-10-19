"""Adapter: serve the planner's high-level vehicle interface on top of the REAL AUV stack.

Set `backend: real` in config/auv.yaml and run this node next to the AUV nodes
(`control_action_server`, `dvl_listener`, `vision_node`). It provides, under /auv_adapter/*,
exactly what the sim's vehicle_server provides (NavigateTo / Search / HoldDepth / Surface actions,
emergency_surface service, odom, battery, detections), so the tools layer, validator, planner and
executor are unchanged.

Mapping onto AUV (read from the team's existing AUV workspace, simulator/basic_action_server.py):
  Control action `control` (custom_interfaces/action/Control: string action, float32 value)
    forward/backward  value = metres along DVL x   (target_x = current_x +/- value)
    left/right        value = metres along DVL y   (target_y = current_y +/- value)
    yaw_left/right    value = absolute heading, degrees
    arm / disarm / mode_depth_hold
  /dvl/position (geometry_msgs/Vector3)      -> odom x, y (and depth if depth_from_dvl_z)
  simulator/yaw (custom_interfaces/Control)  -> heading (degrees), if published
  camera_topic (custom_interfaces/CameraMsg) -> detections (bearing from pixel offset, range from bbox)
  /simulator/control 'set_depth' (Control msg, consumed by controller_node) -> depth setpoints

UNTESTED ON HARDWARE (no vehicle / ArduSub here). Known gaps, documented for the team:
  * basic_action_server has no depth command; depth goes through controller_node's /simulator/control
    `set_depth` topic (or add a `set_depth` branch to the action server).
  * battery is not on a ROS topic in AUV (MAVLink SYS_STATUS). Without `battery_topic` the adapter
    publishes nothing and the validator's runtime precheck refuses motion ("battery state unknown")
    - the safe default.
  * DVL frame conventions (axis signs, origin) must be confirmed on the vehicle.
This is a backend (vehicle-side) component: the LLM/planner never imports it.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

# ------------------------------------------------------------------ pure translation helpers (testable)

DEFAULT_CLASS_MAP = {'gate': 'gate', 'Gate': 'gate', 'red_buoy': 'red_buoy', 'green_buoy': 'green_buoy',
                     'yellow_buoy': 'yellow_buoy', 'buoy_red': 'red_buoy', 'buoy_green': 'green_buoy',
                     'buoy_yellow': 'yellow_buoy'}
OBJECT_WIDTH_M = {'gate': 2.0, 'red_buoy': 0.4, 'green_buoy': 0.4, 'yellow_buoy': 0.4}


@dataclass
class Cmd:
    action: str
    value: float


def plan_navigation(cur_x: float, cur_y: float, tx: float, ty: float, min_step: float = 0.05) -> list[Cmd]:
    """World-frame (DVL) displacement -> AUV Control commands (x first, then y)."""
    out = []
    dx, dy = tx - cur_x, ty - cur_y
    if abs(dx) > min_step:
        out.append(Cmd('forward' if dx > 0 else 'backward', abs(dx)))
    if abs(dy) > min_step:
        out.append(Cmd('left' if dy > 0 else 'right', abs(dy)))
    return out


def plan_heading(current_deg: float, target_deg: float) -> Cmd:
    """Absolute heading command, turning the short way round."""
    diff = (target_deg - current_deg + 180.0) % 360.0 - 180.0
    return Cmd('yaw_left' if diff >= 0 else 'yaw_right', target_deg % 360.0)


def camera_to_detection(class_name: str, cx: int, cy: int, frame_cx: int, frame_cy: int, bbox_w: int,
                        hfov_deg: float, conf: float, pose: tuple, class_map: Optional[dict] = None) -> Optional[dict]:
    """AUV CameraMsg fields -> planner detection (bearing/elevation from pixels, range from bbox width).

    pose = (x, y, depth, yaw_rad) of the vehicle. Returns None for classes the planner does not know."""
    cls = (class_map or DEFAULT_CLASS_MAP).get(class_name)
    if cls is None or bbox_w <= 0 or frame_cx <= 0:
        return None
    f = frame_cx / math.tan(math.radians(hfov_deg) / 2)        # focal length in pixels
    bearing = math.atan((frame_cx - cx) / f)                   # +left
    elevation = math.atan((frame_cy - cy) / f)                 # +up
    rng = f * OBJECT_WIDTH_M[cls] / bbox_w
    x, y, depth, yaw = pose
    horiz = rng * math.cos(elevation)
    wx = x + horiz * math.cos(yaw + bearing)
    wy = y + horiz * math.sin(yaw + bearing)
    return {'class': cls, 'confidence': float(conf), 'range_m': rng, 'bearing_rad': bearing,
            'elevation_rad': elevation, 'x': wx, 'y': wy, 'depth': depth - rng * math.sin(elevation)}


# ------------------------------------------------------------------ ROS node


def main():  # pragma: no cover - needs the AUV workspace (custom_interfaces) and the vehicle
    import rclpy
    from geometry_msgs.msg import Vector3
    from nav_msgs.msg import Odometry
    from rclpy.action import ActionClient, ActionServer, CancelResponse
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import BatteryState
    from std_srvs.srv import Trigger

    from auv_mission_interfaces.action import HoldDepth, NavigateTo, Search, Surface
    from auv_mission_interfaces.msg import Detection, DetectionArray
    try:
        from custom_interfaces.action import Control as ControlAction
        from custom_interfaces.msg import CameraMsg, Control as ControlMsg
    except ImportError as e:
        raise SystemExit(f'auv_adapter needs the AUV custom_interfaces package built and sourced: {e}')

    class AUVAdapter(Node):
        def __init__(self):
            super().__init__('auv_adapter')
            p = self.declare_parameter
            self.ns = p('namespace', '/auv_adapter').value
            self.hfov = p('camera_hfov_deg', 80.0).value
            self.depth_from_dvl_z = p('depth_from_dvl_z', True).value
            self.surface_command = p('surface_command', 'disarm').value      # disarm -> positive buoyancy
            battery_in = p('battery_topic', '').value                           # optional
            self.pose = None            # x, y, depth, yaw_rad
            self.yaw_deg = 0.0
            self.dets = []
            self.lock = threading.Lock()
            cg = ReentrantCallbackGroup()
            self.create_subscription(Vector3, p('dvl_topic', '/dvl/position').value, self.on_dvl, 10,
                                     callback_group=cg)
            self.create_subscription(ControlMsg, p('yaw_topic', 'simulator/yaw').value, self.on_yaw, 10,
                                     callback_group=cg)
            self.create_subscription(CameraMsg, p('camera_topic', 'camera_topic').value, self.on_cam, 10,
                                     callback_group=cg)
            self.control = ActionClient(self, ControlAction, p('control_action', 'control').value,
                                        callback_group=cg)
            self.depth_pub = self.create_publisher(ControlMsg, p('depth_topic', '/simulator/control').value, 10)
            self.odom_pub = self.create_publisher(Odometry, f'{self.ns}/odom', 10)
            self.det_pub = self.create_publisher(DetectionArray, f'{self.ns}/detections', 10)
            self.batt_pub = self.create_publisher(BatteryState, f'{self.ns}/battery', 10)
            if battery_in:
                self.create_subscription(BatteryState, battery_in, self.batt_pub.publish, 10, callback_group=cg)
            kw = dict(callback_group=cg, cancel_callback=lambda _: CancelResponse.ACCEPT)
            ActionServer(self, NavigateTo, f'{self.ns}/navigate_to', self.navigate, **kw)
            ActionServer(self, Search, f'{self.ns}/search', self.search, **kw)
            ActionServer(self, HoldDepth, f'{self.ns}/hold_depth', self.hold_depth, **kw)
            ActionServer(self, Surface, f'{self.ns}/surface', self.surface, **kw)
            self.create_service(Trigger, f'{self.ns}/emergency_surface', self.emergency, callback_group=cg)

        # ---- inputs
        def on_dvl(self, v):
            depth = v.z if self.depth_from_dvl_z else (self.pose[2] if self.pose else 0.0)
            self.pose = (v.x, v.y, depth, math.radians(self.yaw_deg))
            o = Odometry()
            o.header.stamp = self.get_clock().now().to_msg()
            o.header.frame_id = 'dvl'
            o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = v.x, v.y, -depth
            yaw = math.radians(self.yaw_deg)
            o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            self.odom_pub.publish(o)

        def on_yaw(self, msg):
            self.yaw_deg = float(msg.value)

        def on_cam(self, m):
            if self.pose is None:
                return
            arr = DetectionArray()
            arr.header.stamp = self.get_clock().now().to_msg()
            for name in (m.class_name or []):
                d = camera_to_detection(name, m.cx, m.cy, m.frame_cx, m.frame_cy, m.width, self.hfov, m.conf,
                                        self.pose)
                if d is None:
                    continue
                det = Detection()
                det.class_name, det.confidence, det.range_m = d['class'], d['confidence'], d['range_m']
                det.bearing_rad, det.elevation_rad = d['bearing_rad'], d['elevation_rad']
                det.position.x, det.position.y, det.position.z = d['x'], d['y'], -d['depth']
                det.x1, det.y1, det.x2, det.y2 = m.x1, m.y1, m.x2, m.y2
                det.source = 'camera'
                arr.detections.append(det)
            self.dets = list(arr.detections)
            self.det_pub.publish(arr)

        # ---- helpers
        def send(self, cmd: Cmd, gh) -> bool:
            if not self.control.wait_for_server(timeout_sec=5.0):
                return False
            goal = ControlAction.Goal()
            goal.action, goal.value = cmd.action, float(cmd.value)
            fut = self.control.send_goal_async(goal)
            while not fut.done():
                time.sleep(0.05)
            handle = fut.result()
            if not handle.accepted:
                return False
            res = handle.get_result_async()
            while not res.done():
                if gh.is_cancel_requested:
                    handle.cancel_goal_async()
                    return False
                time.sleep(0.05)
            return bool(res.result().result.success)

        def set_depth(self, depth):
            m = ControlMsg()
            m.action, m.value = 'set_depth', float(-abs(depth))     # AUV uses negative-down targets
            self.depth_pub.publish(m)

        def wait_depth(self, depth, tol, timeout, gh):
            end = time.time() + timeout
            while time.time() < end and not gh.is_cancel_requested:
                if self.pose and abs(self.pose[2] - depth) <= tol:
                    return True
                time.sleep(0.1)
            return False

        # ---- actions
        def navigate(self, gh):
            g, res = gh.request, NavigateTo.Result()
            self.set_depth(g.depth)
            ok = self.wait_depth(g.depth, max(g.tolerance_m, 0.3), 60.0, gh)
            if ok and self.pose:
                for cmd in plan_navigation(self.pose[0], self.pose[1], g.x, g.y):
                    if not self.send(cmd, gh):
                        ok = False
                        break
            if ok and g.use_yaw:
                ok = self.send(plan_heading(self.yaw_deg, math.degrees(g.yaw)), gh)
            res.success = ok
            res.final_error_m = math.dist(self.pose[:2], (g.x, g.y)) if self.pose else -1.0
            res.message = 'reached' if ok else 'navigation failed or canceled'
            (gh.succeed if ok else gh.abort)()
            return res

        def search(self, gh):
            g, res = gh.request, Search.Result()
            self.set_depth(g.depth if g.depth > 0 else 1.0)
            start_heading = self.yaw_deg
            for k in range(1, 7):                    # yaw sweep in 60 degree steps (all patterns start here)
                if gh.is_cancel_requested:
                    break
                found = [d for d in self.dets if d.class_name == g.target_class and d.confidence >= 0.4]
                if found:
                    res.found, res.detection, res.message = True, found[0], f'{g.target_class} found'
                    gh.succeed()
                    return res
                self.send(plan_heading(self.yaw_deg, start_heading + 60 * k), gh)
            res.message = f'{g.target_class} not found (yaw sweep)'
            res.detection = Detection()
            gh.abort()
            return res

        def hold_depth(self, gh):
            g, res = gh.request, HoldDepth.Result()
            self.set_depth(g.depth)
            ok = self.wait_depth(g.depth, g.tolerance_m or 0.3, 60.0, gh)
            t0, errs = time.time(), []
            while ok and time.time() - t0 < g.duration_s and not gh.is_cancel_requested:
                errs.append(abs(self.pose[2] - g.depth))
                time.sleep(0.1)
            res.mean_error_m = sum(errs) / len(errs) if errs else -1.0
            res.success = ok and res.mean_error_m <= (g.tolerance_m or 0.3)
            res.message = 'held' if res.success else 'depth not held'
            (gh.succeed if res.success else gh.abort)()
            return res

        def surface(self, gh):
            res = Surface.Result()
            ok = self.send(Cmd(self.surface_command, 0.0), gh) if self.surface_command else False
            if not ok:
                self.set_depth(0.0)
            ok = self.wait_depth(0.0, 0.25, 90.0, gh)
            res.success, res.message = ok, 'surfaced' if ok else 'surface timeout'
            (gh.succeed if ok else gh.abort)()
            return res

        def emergency(self, _req, resp):
            self.set_depth(0.0)
            threading.Thread(target=lambda: self.send(Cmd(self.surface_command or 'disarm', 0.0),
                                                      type('G', (), {'is_cancel_requested': False})()),
                             daemon=True).start()
            resp.success, resp.message = True, 'surfacing'
            return resp

    rclpy.init()
    node = AUVAdapter()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
