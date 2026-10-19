"""Ground-truth referee: watches the vehicle and records task events independently of the planner.

Events (std_msgs/String JSON on /auv_sim/events):
  gate_passed / gate_missed, buoy_touched (per buoy id), surfaced, boundary_contact
Report: /auv_sim/referee_report (std_srvs/Trigger, JSON in message) and, if report_file is set,
the same JSON written to disk after every event and at shutdown. The eval harness scores missions
from this report, not from what the agent claims.
"""
import json
import math
from collections import deque

from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from auv_sim.arena import load_arena, world_to_body
from auv_sim.ros_util import spin_node

TOUCH_MARGIN = 0.35   # hull half-length + slack
SURFACE_WINDOW_S = 2.0
SURFACED_DEPTH_M = 0.3   # same threshold as the executor's `surfaced` condition


class Referee(Node):
    def __init__(self):
        super().__init__('auv_referee')
        self.arena = load_arena(self.declare_parameter('arena_file', 'default').value)
        self.report_file = self.declare_parameter('report_file', '').value
        self.events = []
        self.touched = set()
        self.gate_side = {}
        self.max_depth = 0.0
        self.was_deep = False
        self.at_boundary = False
        self.path_len = 0.0
        self.last_pos = None
        self.final = None
        self.recent = deque()          # (t, depth) over the last SURFACE_WINDOW_S: a floating hull bobs
        self.create_subscription(Odometry, '/auv/odom', self.on_odom, 20)
        self.pub = self.create_publisher(String, '/auv_sim/events', 10)
        self.create_service(Trigger, '/auv_sim/referee_report', self.on_report)
        self.create_timer(1.0, self.write)      # keep the on-disk report (final pose) fresh

    def t(self):
        return round(self.get_clock().now().nanoseconds * 1e-9, 2)

    def emit(self, event, **data):
        e = {'event': event, 't': self.t(), **data}
        self.events.append(e)
        self.pub.publish(String(data=json.dumps(e)))
        self.get_logger().info(f'EVENT {e}')
        self.write()

    def report(self):
        return {
            'events': self.events,
            'gate_passed': any(e['event'] == 'gate_passed' for e in self.events),
            'buoys_touched': sorted(self.touched),
            'touch_order': [e['buoy'] for e in self.events if e['event'] == 'buoy_touched'],
            'surfaced': any(e['event'] == 'surfaced' for e in self.events),
            'surfaced_at_end': self._surfaced_now(),
            'max_depth_m': round(self.max_depth, 2),
            'boundary_contacts': sum(e['event'] == 'boundary_contact' for e in self.events),
            'path_length_m': round(self.path_len, 1),
            'final': self.final,
            'sim_time_s': self.t(),
        }

    def _surfaced_now(self) -> bool:
        """On the surface = mean depth over the last window below the threshold (robust to bobbing)."""
        if not self.recent:
            return False
        return sum(d for _, d in self.recent) / len(self.recent) < SURFACED_DEPTH_M

    def write(self):
        if self.report_file:
            with open(self.report_file, 'w') as f:
                json.dump(self.report(), f, indent=2)

    def on_report(self, _req, resp):
        resp.success, resp.message = True, json.dumps(self.report())
        return resp

    def on_odom(self, msg):
        p = msg.pose.pose.position
        x, y, z = p.x, p.y, p.z
        depth = -z
        if self.last_pos:
            self.path_len += math.dist(self.last_pos, (x, y, z))
        self.last_pos = (x, y, z)
        self.final = {'x': round(x, 2), 'y': round(y, 2), 'depth': round(depth, 2)}
        now = self.t()
        self.recent.append((now, depth))
        while self.recent and now - self.recent[0][0] > SURFACE_WINDOW_S:
            self.recent.popleft()
        self.max_depth = max(self.max_depth, depth)
        if depth > 0.5:
            self.was_deep = True
        if self.was_deep and depth < 0.2:
            self.was_deep = False
            self.emit('surfaced', x=round(x, 2), y=round(y, 2))
        for o in self.arena['objects']:
            if o['class'] == 'gate':
                self._gate(o, x, y, z)
            elif o['id'] not in self.touched:
                d = math.dist((x, y, z), (o['x'], o['y'], o['z']))
                if d < o.get('radius', 0.2) + TOUCH_MARGIN:
                    self.touched.add(o['id'])
                    self.emit('buoy_touched', buoy=o['id'], cls=o['class'], dist=round(d, 2))
        pool = self.arena['pool']
        x0 = pool.get('x_min', -5.0)
        near = (x - x0 < 0.4 or x0 + pool['length'] - x < 0.4 or pool['width'] / 2 - abs(y) < 0.4
                or pool['depth'] - depth < 0.3)
        if near and not self.at_boundary:
            self.emit('boundary_contact', x=round(x, 2), y=round(y, 2), depth=round(depth, 2))
        self.at_boundary = near

    def _gate(self, o, x, y, z):
        gx, gy = world_to_body(x - o['x'], y - o['y'], o.get('yaw', 0.0))
        gz = z - o['z']
        side = gx > 0
        prev = self.gate_side.get(o['id'])
        self.gate_side[o['id']] = side
        if prev is None or prev == side:
            return
        inside = abs(gy) < o['width'] / 2 - 0.05 and abs(gz) < o['height'] / 2
        direction = 'forward' if side else 'backward'
        self.emit('gate_passed' if inside else 'gate_missed', gate=o['id'], direction=direction,
                  lateral=round(gy, 2), vertical=round(gz, 2))

    def destroy_node(self):
        self.write()
        super().destroy_node()


def main():
    spin_node(Referee)
