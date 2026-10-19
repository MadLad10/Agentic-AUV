"""ROS 2 implementation of VehicleBackend.

Talks ONLY to the high-level interfaces named in config (interfaces.<backend>): read-only
subscriptions (odom, battery, detections), four action clients and one emergency service client.
It creates no publishers at all - there is no path from here to thrusters or controller setpoints.
Timeouts are in sim time (use_sim_time), with a wall-clock safety net.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import BatteryState
from std_srvs.srv import Trigger

from auv_mission_interfaces.action import HoldDepth, NavigateTo, Search, Surface
from auv_mission_interfaces.msg import DetectionArray
from auv_mission_planner.tools.backend import Det, Outcome, Pose, VehicleBackend


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RosBackend(VehicleBackend):
    def __init__(self, cfg: dict, node: Optional[Node] = None, use_sim_time: bool = True):
        super().__init__()
        self.iface = cfg['iface']
        self.wall_factor = cfg.get('tools', {}).get('wall_timeout_factor', 3.0)
        self._own = node is None
        if self._own:
            if not rclpy.ok():
                rclpy.init()
            node = Node('auv_mission_tools',
                        parameter_overrides=[Parameter('use_sim_time', value=use_sim_time)])
        self.node = node
        self._pose: Optional[Pose] = None
        self._battery: Optional[float] = None
        self._dets: list[Det] = []
        self._goals = {}     # id(handle) -> handle (ClientGoalHandle is unhashable)
        self._lock = threading.Lock()
        cg = ReentrantCallbackGroup()
        i = self.iface
        node.create_subscription(Odometry, i['odom_topic'], self._on_odom, 10, callback_group=cg)
        node.create_subscription(BatteryState, i['battery_topic'], self._on_battery, 10, callback_group=cg)
        node.create_subscription(DetectionArray, i['detections_topic'], self._on_dets, 10, callback_group=cg)
        self.clients = {
            'navigate': ActionClient(node, NavigateTo, i['navigate_action'], callback_group=cg),
            'search': ActionClient(node, Search, i['search_action'], callback_group=cg),
            'hold_depth': ActionClient(node, HoldDepth, i['hold_depth_action'], callback_group=cg),
            'surface': ActionClient(node, Surface, i['surface_action'], callback_group=cg),
        }
        self.emergency = node.create_client(Trigger, i['emergency_service'], callback_group=cg)
        self._executor = None
        if self._own:
            self._executor = MultiThreadedExecutor(num_threads=4)
            self._executor.add_node(node)
            self._thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._thread.start()

    # ------------------------------------------------------------ state
    def _on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self._pose = Pose(p.x, p.y, -p.z, _yaw(q))

    def _on_battery(self, m):
        self._battery = m.percentage * 100.0

    def _on_dets(self, m):
        t = self.now()
        dets = [Det(d.class_name, d.confidence, d.position.x, d.position.y, -d.position.z, d.range_m,
                    math.degrees(d.bearing_rad), t) for d in m.detections]
        self._dets = dets
        self._notify(dets)

    def now(self) -> float:
        t = self.node.get_clock().now().nanoseconds * 1e-9
        return t if t > 0 else time.time()

    def pose(self):
        return self._pose

    def battery_percent(self):
        return self._battery

    def detections(self):
        return list(self._dets)

    def wait_ready(self, timeout_s=30.0) -> bool:
        end = time.time() + timeout_s
        while time.time() < end:
            if self._pose is not None and self._battery is not None and \
                    all(c.server_is_ready() for c in self.clients.values()):
                return True
            time.sleep(0.2)
        return False

    # ------------------------------------------------------------ actions
    def _run(self, key, goal, timeout_s, describe) -> Outcome:
        client = self.clients[key]
        if not client.wait_for_server(timeout_sec=5.0):
            return Outcome(False, f'{key} action server unavailable', 'backend_unavailable')
        handle = None
        for _ in range(2):                  # a loaded machine can delay the ack; resend once
            fut = client.send_goal_async(goal)
            if _wait(fut, 15.0):
                handle = fut.result()
                break
        if handle is None:
            return Outcome(False, f'{key}: goal not acknowledged', 'backend_unavailable')
        if not handle.accepted:
            return Outcome(False, f'{key}: goal rejected by vehicle', 'action_failed')
        with self._lock:
            self._goals[id(handle)] = handle
        res_fut = handle.get_result_async()
        start_sim, start_wall = self.now(), time.time()
        wall_cap = self.wall_factor * timeout_s + 15.0
        try:
            while not res_fut.done():
                if self.now() - start_sim > timeout_s or time.time() - start_wall > wall_cap:
                    cf = handle.cancel_goal_async()
                    _wait(cf, 5.0)
                    _wait(res_fut, 5.0)
                    return Outcome(False, f'{key} timed out after {timeout_s:.0f}s', 'timeout',
                                   {'elapsed_s': round(self.now() - start_sim, 1)})
                time.sleep(0.05)
        finally:
            with self._lock:
                self._goals.pop(id(handle), None)
        wrapped = res_fut.result()
        r = wrapped.result
        ok = wrapped.status == GoalStatus.STATUS_SUCCEEDED
        return Outcome(ok, r.message, None if ok else 'action_failed', describe(r))

    def navigate(self, x, y, depth, yaw=None, tolerance=0.35, speed=None, timeout_s=60.0):
        g = NavigateTo.Goal()
        g.x, g.y, g.depth = float(x), float(y), float(depth)
        g.use_yaw = yaw is not None
        g.yaw = float(yaw or 0.0)
        g.tolerance_m, g.max_speed = float(tolerance), float(speed or 0.0)
        return self._run('navigate', g, timeout_s, lambda r: {'final_error_m': round(r.final_error_m, 2)})

    def search(self, target, pattern, radius_m, depth, timeout_s=60.0):
        g = Search.Goal()
        g.target_class, g.pattern, g.radius_m, g.depth = target, pattern, float(radius_m), float(depth)

        def describe(r):
            if not r.found:
                return {'found': False}
            d = r.detection
            return {'found': True, 'detection': {'class': d.class_name, 'confidence': round(d.confidence, 2),
                    'position': {'x': round(d.position.x, 2), 'y': round(d.position.y, 2),
                                 'depth': round(-d.position.z, 2)}, 'range_m': round(d.range_m, 2)}}
        return self._run('search', g, timeout_s, describe)

    def hold_depth(self, depth, duration_s, tolerance=0.3, timeout_s=60.0):
        g = HoldDepth.Goal()
        g.depth, g.duration_s, g.tolerance_m = float(depth), float(duration_s), float(tolerance)
        return self._run('hold_depth', g, timeout_s, lambda r: {'mean_error_m': round(r.mean_error_m, 2)})

    def surface(self, timeout_s=60.0):
        return self._run('surface', Surface.Goal(), timeout_s, lambda r: {})

    def cancel_all(self):
        with self._lock:
            handles = list(self._goals.values())
        for h in handles:
            _wait(h.cancel_goal_async(), 3.0)

    def emergency_surface(self):
        if not self.emergency.wait_for_service(timeout_sec=5.0):
            return Outcome(False, 'emergency service unavailable', 'backend_unavailable')
        fut = self.emergency.call_async(Trigger.Request())
        if not _wait(fut, 5.0):
            return Outcome(False, 'emergency service did not answer', 'backend_unavailable')
        return Outcome(fut.result().success, fut.result().message)

    def shutdown(self):
        if self._own:
            self.cancel_all()
            self._executor.shutdown(timeout_sec=2.0)
            self._thread.join(timeout=2.0)
            try:
                self.node.destroy_node()
            except Exception:
                pass
            rclpy.try_shutdown()


def _wait(fut, timeout_s) -> bool:
    end = time.time() + timeout_s
    while not fut.done() and time.time() < end:
        time.sleep(0.02)
    return fut.done()
