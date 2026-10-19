"""Small rclpy helpers shared by the sim nodes."""
import signal

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor


def spin_node(node_cls, threads=1):
    """Spin a node until shutdown; teardown is immune to a second SIGINT."""
    rclpy.init()
    node = node_cls()
    executor = MultiThreadedExecutor(num_threads=threads) if threads > 1 else None
    try:
        if executor:
            executor.add_node(node)
            executor.spin()
        else:
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()
