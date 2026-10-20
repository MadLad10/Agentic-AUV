"""The vehicle as seen by the v2 agents over ROS 2 (agents and vehicle in separate processes).

Extends v1's RosBackend (odometry, battery, detections, the four actions, emergency service) with
what the specialist agents read, all from topics of the vehicle node (auv_onboard/ros_node.py):
  /auv/health -> health report      /auv/nav_status -> navigation filter summary
  /auv/tracks -> camera-sonar fusion tracks (also fed to the tools' world model, as onboard)
  /auv/telemetry -> diagnostics evidence   /auv/camera/image -> VLM queries
and commands degraded-mode speed on /auv/speed_scale. Nothing here can reach thrusters.
"""
from __future__ import annotations

import json
import math
import threading

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, String

from auv_mission_planner.tools.backend import Det
from auv_mission_planner.tools.ros_backend import RosBackend
from auv_onboard.health import HealthReport


class _Track:
    def __init__(self, d: dict):
        self.d = d
        self.id = d['id']
        self.kind = d['kind']
        self.x = np.array([d['x'], d['y']], float)
        self.cls = None if d['class'].startswith('unknown_') else d['class']

    def as_dict(self):
        return dict(self.d)


class _TrackView:
    """Read-only tracker interface (same queries as auv_onboard.tracking.FusionTracker)."""

    def __init__(self):
        self.msg = {'tracks': [], 'unclassified': [], 'false_targets': [], 'landmark_depth': {}}

    def _all(self):
        return [_Track(d) for d in self.msg['tracks']]

    def confirmed(self):
        return self._all()

    def unclassified(self):
        ids = set(self.msg['unclassified'])
        return [t for t in self._all() if t.id in ids]

    def false_target_suspects(self):
        ids = set(self.msg['false_targets'])
        return [t for t in self._all() if t.id in ids]

    def sonar_gate(self, gate_w, tol):
        posts = [t for t in self._all() if t.kind == 'gate_post']
        best = None
        for i, a in enumerate(posts):
            for b in posts[i + 1:]:
                d = float(np.linalg.norm(a.x - b.x))
                if abs(d - gate_w) < tol and (best is None or abs(d - gate_w) < best[0]):
                    best = (abs(d - gate_w), (a.x + b.x) / 2)
        return None if best is None else best[1]


class _NavView:
    def __init__(self):
        self.status = {'pos_std_m': 0.0, 'yaw_std_deg': 0.0, 'last_update_t': {}, 'updates': {}}

    def summary(self):
        return dict(self.status)

    def pos_std(self):
        return float(self.status.get('pos_std_m', 0.0))


class _SimView:
    """The few simulator-side facts agents use: the course layout (known before a run) and the
    latest camera frame."""

    def __init__(self, arena: dict):
        self.arena = arena
        self.image = None

    def camera_image(self):
        return self.image


class RosVehicle(RosBackend):
    nav_source = 'ekf'
    health_enabled = True

    def __init__(self, cfg: dict, arena='default', **kw):
        super().__init__(cfg, **kw)
        from auv_marine_sim.world import load_arena
        self.sim = _SimView(load_arena(arena))
        self.health = type('H', (), {})()
        self.health.last = HealthReport(0.0)
        self.tracker = _TrackView()
        self.ekf = _NavView()
        self.telemetry = {}
        self.landmark_depth = {}
        self.tick_hooks: list = []
        self._threads: list = []
        self._veto = None
        self._speed = 1.0
        self._est = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
        self._lock2 = threading.Lock()
        cg = ReentrantCallbackGroup()
        n = self.node
        n.create_subscription(String, '/auv/health', self._on_health, 10, callback_group=cg)
        n.create_subscription(String, '/auv/nav_status', self._on_nav, 10, callback_group=cg)
        n.create_subscription(String, '/auv/tracks', self._on_tracks, 10, callback_group=cg)
        n.create_subscription(String, '/auv/telemetry', self._on_tel, 10, callback_group=cg)
        n.create_subscription(Image, '/auv/camera/image', self._on_image, 2, callback_group=cg)
        n.create_subscription(Odometry, self.iface['odom_topic'], self._on_odom2, 20, callback_group=cg)
        self._speed_pub = n.create_publisher(Float64, '/auv/speed_scale', 10)

    # ------------------------------------------------------------------ subscriptions
    def _on_health(self, m):
        d = json.loads(m.data)
        self.health.last = HealthReport(**d)
        if d.get('veto') and self._veto is None:
            self._veto = d.get('veto_reason') or 'health veto'

    def _on_nav(self, m):
        self.ekf.status = json.loads(m.data)

    def _on_tracks(self, m):
        self.tracker.msg = json.loads(m.data)
        self.landmark_depth = self.tracker.msg.get('landmark_depth', {})
        t = self.now()
        fused = [Det(d['class'], d['confidence'], d['x'], d['y'], d['depth'],
                     math.hypot(d['x'] - self._est['x'], d['y'] - self._est['y']), 0.0, t)
                 for d in self.tracker.msg['tracks'] if not d['class'].startswith('unknown_') and d.get('depth') is not None]
        if fused:
            self._notify(fused)                     # world model gets fused tracks, as onboard

    def _on_tel(self, m):
        self.telemetry = json.loads(m.data)
        if self.telemetry.get('veto') and self._veto is None:
            self._veto = self.telemetry['veto']

    def _on_image(self, m):
        self.sim.image = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)

    def _on_odom2(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self._est = {'x': p.x, 'y': p.y, 'z': p.z,
                     'yaw': math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
                     'vx': m.twist.twist.linear.x, 'vy': m.twist.twist.linear.y, 'vz': m.twist.twist.linear.z,
                     'r': 0.0}
        for h in list(self.tick_hooks):            # the graph's in-mission watchdog
            try:
                h(self)
            except Exception:
                self.tick_hooks.remove(h)

    # ------------------------------------------------------------------ onboard-like accessors
    def est(self) -> dict:
        return dict(self._est)

    @property
    def speed_scale(self):
        return self._speed

    @speed_scale.setter
    def speed_scale(self, v):
        self._speed = float(v)
        self._speed_pub.publish(Float64(data=self._speed))
