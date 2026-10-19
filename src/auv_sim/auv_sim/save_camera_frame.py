"""Save camera frames to PNG for headless verification.

ros2 run auv_sim save_camera_frame --ros-args -p out_dir:=/tmp/frames -p count:=1 -p every_n:=1
Exits after `count` frames or `timeout_s` (wall) seconds.
"""
import os
import time

import cv2
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import Image


def image_to_bgr(msg):
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ('rgb8', 'bgr8'):
        img = buf.reshape(msg.height, msg.step // 3 if msg.step else msg.width, 3)[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img.copy()
    if msg.encoding in ('rgba8', 'bgra8'):
        img = buf.reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if msg.encoding == 'rgba8' else cv2.COLOR_BGRA2BGR)
    raise ValueError(f'unsupported encoding {msg.encoding}')


class Saver(Node):
    def __init__(self):
        super().__init__('auv_frame_saver')
        self.out_dir = self.declare_parameter('out_dir', '/tmp/auv_frames').value
        self.count = self.declare_parameter('count', 1).value
        self.every_n = self.declare_parameter('every_n', 1).value
        self.topic = self.declare_parameter('topic', '/auv/camera/image').value
        os.makedirs(self.out_dir, exist_ok=True)
        self.seen = 0
        self.saved = 0
        self.create_subscription(Image, self.topic, self.cb, 5)

    def cb(self, msg):
        self.seen += 1
        if self.saved >= self.count or (self.seen - 1) % self.every_n:
            return
        path = os.path.join(self.out_dir, f'frame_{self.saved:04d}_{msg.header.stamp.sec}.png')
        cv2.imwrite(path, image_to_bgr(msg))
        self.saved += 1
        self.get_logger().info(f'saved {path}')


def main():
    rclpy.init()
    node = Saver()
    deadline = time.time() + float(node.declare_parameter(
        'timeout_s', 30.0, ParameterDescriptor(dynamic_typing=True)).value)
    while rclpy.ok() and node.saved < node.count and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
    ok = node.saved >= node.count
    node.destroy_node()
    rclpy.try_shutdown()
    raise SystemExit(0 if ok else 1)
