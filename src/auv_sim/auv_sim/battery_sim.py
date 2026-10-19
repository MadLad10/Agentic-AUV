"""Battery simulator: drains with hotel load + thruster effort (sim time). Publishes BatteryState.

percentage follows the ROS convention (0..1). initial_percent is 0..100.
"""
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Float64

from auv_sim.ros_util import spin_node


class Battery(Node):
    def __init__(self):
        super().__init__('auv_battery')
        self.capacity_wh = self.declare_parameter('capacity_wh', 20.0).value
        self.hotel_w = self.declare_parameter('hotel_w', 20.0).value
        self.w_per_n = self.declare_parameter('w_per_newton', 1.2).value
        self.energy_wh = self.capacity_wh * self.declare_parameter('initial_percent', 100.0).value / 100.0
        self.thrust = 0.0
        self.last = None
        self.create_subscription(Float64, '/auv/thrusters/total_abs', self.on_thrust, 10)
        self.pub = self.create_publisher(BatteryState, '/auv/battery', 10)
        self.create_timer(0.5, self.step)

    def on_thrust(self, msg):
        self.thrust = msg.data

    def step(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        if self.last is not None and t > self.last:
            dt_h = (t - self.last) / 3600.0
            self.energy_wh = max(0.0, self.energy_wh - (self.hotel_w + self.w_per_n * self.thrust) * dt_h)
        self.last = t
        frac = self.energy_wh / self.capacity_wh
        m = BatteryState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.percentage = float(frac)
        m.voltage = float(14.0 + 2.8 * frac)          # 4S Li-ion-ish
        m.present = True
        m.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        self.pub.publish(m)


def main():
    spin_node(Battery)
