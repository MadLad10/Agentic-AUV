"""M1: the stand-in sim runs headless, publishes odometry, the AUV floats, and closed-loop hold works."""
import math
import time

import pytest

pytestmark = pytest.mark.integration


def _collect(node, secs, odom):
    import rclpy
    end = time.time() + secs
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)


def test_floats_idle_and_holds_setpoint(sim_launcher):
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String

    sim_launcher(stack='minimal', camera='false')
    rclpy.init()
    node = rclpy.create_node('m1_probe')
    odom = {}
    node.create_subscription(Odometry, '/auv/odom', lambda m: odom.update(p=m.pose.pose.position), 10)
    try:
        _collect(node, 6.0, odom)           # idle: positive buoyancy -> rises to the surface
        assert odom['p'].z > -0.2, f'should float near surface, z={odom["p"].z}'

        sp = node.create_publisher(PoseStamped, '/auv/controller/setpoint', 10)
        mode = node.create_publisher(String, '/auv/controller/mode', 10)
        msg = PoseStamped()
        msg.pose.position.x, msg.pose.position.z = 2.0, -1.2
        msg.pose.orientation.w = 1.0
        end = time.time() + 20
        while time.time() < end:
            sp.publish(msg)
            mode.publish(String(data='hold'))
            rclpy.spin_once(node, timeout_sec=0.1)
        p = odom['p']
        err = math.sqrt((p.x - 2.0) ** 2 + p.y ** 2 + (p.z + 1.2) ** 2)
        assert err < 0.25, f'hold error {err:.2f} m at {p}'
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
