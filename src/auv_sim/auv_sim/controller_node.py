"""Low-level controller: pose setpoint -> 6 thruster commands.

This is the ONLY node that publishes to thruster topics. The mission planner / LLM never talks to
it directly; it only sees the high-level actions served by vehicle_server.

Inputs:  odom (nav_msgs/Odometry), setpoint (geometry_msgs/PoseStamped, z = -depth),
         mode (std_msgs/String: idle | hold | surface)
Outputs: /auv/thrusters/<name> (std_msgs/Float64, N), /auv/thrusters/total_abs
"""
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64, String

from auv_sim.ros_util import spin_node
from auv_sim.arena import body_to_world, world_to_body, wrap, yaw_from_quat
from auv_sim.world_gen import MAX_THRUST_N, THRUSTERS

SURGE_ARM = abs(THRUSTERS['surge_left'][1])


class Pid:
    def __init__(self, kp, ki, kd, i_lim, i_zone=None):
        self.kp, self.ki, self.kd, self.i_lim = kp, ki, kd, i_lim
        self.i_zone = i_zone      # anti-windup: integrate only when |err| < i_zone
        self.i = 0.0

    def __call__(self, err, rate, dt):
        if self.i_zone is None or abs(err) < self.i_zone:
            self.i = max(-self.i_lim, min(self.i_lim, self.i + err * dt))
        return self.kp * err + self.ki * self.i - self.kd * rate

    def reset(self):
        self.i = 0.0


class Controller(Node):
    def __init__(self):
        super().__init__('auv_controller')
        p = self.declare_parameter
        self.max_thrust = p('max_thrust_n', MAX_THRUST_N).value
        self.hard_max_depth = p('hard_max_depth_m', 4.6).value
        kp_xy, kd_xy = p('kp_xy', 30.0).value, p('kd_xy', 45.0).value
        self.pid_x = Pid(kp_xy, 0.0, kd_xy, 1.0)
        self.pid_y = Pid(kp_xy, 0.0, kd_xy, 1.0)
        self.pid_z = Pid(p('kp_z', 70.0).value, p('ki_z', 15.0).value, p('kd_z', 50.0).value, 1.5, i_zone=0.25)
        self.pid_yaw = Pid(p('kp_yaw', 8.0).value, 0.0, p('kd_yaw', 5.0).value, 0.5)

        self.pose = None          # (x, y, z, yaw)
        self.vel = (0.0, 0.0, 0.0, 0.0)
        self.last_t = None
        self.setpoint = None      # (x, y, z, yaw)
        self.mode = 'idle'

        self.create_subscription(Odometry, 'odom', self.on_odom, 20)
        self.create_subscription(PoseStamped, 'setpoint', self.on_setpoint, 10)
        self.create_subscription(String, 'mode', self.on_mode, 10)
        self.pubs = {n: self.create_publisher(Float64, f'/auv/thrusters/{n}', 10) for n in THRUSTERS}
        self.total_pub = self.create_publisher(Float64, '/auv/thrusters/total_abs', 10)
        self.create_timer(0.05, self.step)

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        new = (p.x, p.y, p.z, yaw_from_quat(q))
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.pose is not None and self.last_t is not None and t > self.last_t:
            dt = t - self.last_t
            raw = [(new[i] - self.pose[i]) / dt for i in range(3)] + [wrap(new[3] - self.pose[3]) / dt]
            a = 0.3   # low-pass on finite-difference velocity
            self.vel = tuple(a * r + (1 - a) * v for r, v in zip(raw, self.vel))
        self.pose, self.last_t = new, t

    def on_setpoint(self, msg):
        q = msg.pose.orientation
        self.setpoint = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, yaw_from_quat(q))

    def on_mode(self, msg):
        if msg.data not in ('idle', 'hold', 'surface'):
            self.get_logger().warn(f'unknown mode {msg.data!r}')
            return
        if msg.data != self.mode:
            for pid in (self.pid_x, self.pid_y, self.pid_z, self.pid_yaw):
                pid.reset()
        self.mode = msg.data

    def publish(self, cmds):
        total = 0.0
        for n, v in cmds.items():
            v = max(-self.max_thrust, min(self.max_thrust, v))
            total += abs(v)
            self.pubs[n].publish(Float64(data=float(v)))
        self.total_pub.publish(Float64(data=total))

    def step(self):
        zero = {n: 0.0 for n in THRUSTERS}
        if self.pose is None or self.mode == 'idle' or self.setpoint is None:
            self.publish(zero)
            return
        dt = 0.05
        x, y, z, yaw = self.pose
        sx, sy, sz, syaw = self.setpoint
        vx, vy, vz, r = self.vel
        fx_w = self.pid_x(sx - x, vx, dt)
        fy_w = self.pid_y(sy - y, vy, dt)
        fx, fy = world_to_body(fx_w, fy_w, yaw)
        mz = self.pid_yaw(wrap(syaw - yaw), r, dt)
        if self.mode == 'surface':
            fz = 0.0 if z > -0.3 else 15.0     # gentle ascent, then let buoyancy finish
        else:
            fz = self.pid_z(sz - z, vz, dt)
        if -z > self.hard_max_depth:          # backend failsafe, independent of the planner
            fz = self.max_thrust
            self.get_logger().warn(f'hard depth limit {self.hard_max_depth} m exceeded, ascending',
                                   throttle_duration_sec=2.0)
        # allocation (see world_gen.THRUSTERS)
        cmds = {
            'surge_left': fx / 2 - mz / (2 * SURGE_ARM),
            'surge_right': fx / 2 + mz / (2 * SURGE_ARM),
            'heave_front': fz / 2, 'heave_back': fz / 2,
            'sway_front': fy / 2, 'sway_back': fy / 2,
        }
        self.publish(cmds)


def main():
    spin_node(Controller)
