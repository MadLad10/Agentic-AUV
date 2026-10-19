#!/usr/bin/env python3
"""Wait until the sim is up: odometry flowing (+ optionally the vehicle action servers).
Does not use the ros2 CLI daemon. Exit 0 when ready, 1 on timeout.

  scripts/wait_ready.py [--timeout 40] [--actions]
"""
import argparse
import sys
import time


def wait_ready(timeout_s=40.0, actions=True, odom_topic='/auv/odom', action='/auv/navigate_to') -> bool:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.action import ActionClient
    own = not rclpy.ok()
    if own:
        rclpy.init()
    node = rclpy.create_node('auv_wait_ready')
    got = []
    node.create_subscription(Odometry, odom_topic, lambda m: got.append(1), 10)
    client = None
    if actions:
        from auv_mission_interfaces.action import NavigateTo
        client = ActionClient(node, NavigateTo, action)
    end = time.time() + timeout_s
    ok = False
    try:
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.2)
            if got and (client is None or client.server_is_ready()):
                ok = True
                break
    finally:
        node.destroy_node()
        if own:
            rclpy.try_shutdown()
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeout', type=float, default=40.0)
    ap.add_argument('--actions', action='store_true')
    a = ap.parse_args()
    sys.exit(0 if wait_ready(a.timeout, a.actions) else 1)
