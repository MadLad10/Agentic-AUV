"""Simulated object detector -> /auv/detections (DetectionArray).

detection_source:
  ground_truth  project arena object poses into the camera (FOV, range limit, noise, dropout).
                Reliable fallback; the default.
  camera        OpenCV HSV colour detector on the rendered front camera (needs camera:=true).

blind_classes: comma-separated classes the detector never reports (fault injection for evals).
"""
import math
import random

from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image

from auv_mission_interfaces.msg import Detection, DetectionArray
from auv_sim.arena import body_to_world, load_arena, world_to_body, yaw_from_quat
from auv_sim.ros_util import spin_node

IMG_W, IMG_H = 640, 480
CAM_OFFSET = 0.26
BUOY_DIAM = 0.4
# HSV ranges (OpenCV H in 0..180) for the colours used in world_gen
HSV = {
    'red_buoy': [((0, 120, 80), (8, 255, 255)), ((172, 120, 80), (180, 255, 255))],
    'green_buoy': [((45, 120, 80), (80, 255, 255))],
    'yellow_buoy': [((25, 120, 215), (35, 255, 255))],  # emissive -> very bright
    'gate': [((8, 150, 40), (27, 255, 210))],           # orange posts, unlit -> darker than buoys
}


def project(obj, pose, hfov, vfov):
    """Ground-truth geometry: returns (range, bearing, elevation, body_x) of obj from the camera."""
    x, y, z, yaw = pose
    cx, cy = body_to_world(CAM_OFFSET, 0.0, yaw)
    dx, dy, dz = obj['x'] - (x + cx), obj['y'] - (y + cy), obj['z'] - z
    bx, by = world_to_body(dx, dy, yaw)
    rng = math.sqrt(dx * dx + dy * dy + dz * dz)
    return rng, math.atan2(by, bx), math.atan2(dz, math.hypot(bx, by)), bx


def bbox(rng, bearing, elev, w_m, h_m, hfov):
    f = (IMG_W / 2) / math.tan(hfov / 2)
    u = IMG_W / 2 - f * math.tan(bearing)
    v = IMG_H / 2 - f * math.tan(elev)
    hw, hh = f * w_m / max(rng, 0.1) / 2, f * h_m / max(rng, 0.1) / 2
    clamp = lambda a, hi: int(max(0, min(hi, a)))
    return clamp(u - hw, IMG_W), clamp(v - hh, IMG_H), clamp(u + hw, IMG_W), clamp(v + hh, IMG_H)


class Detector(Node):
    def __init__(self):
        super().__init__('auv_detector')
        p = self.declare_parameter
        self.arena = load_arena(p('arena_file', 'default').value)
        self.source = p('detection_source', 'ground_truth').value or 'ground_truth'
        self.hfov = math.radians(p('hfov_deg', 80.0).value)
        self.vfov = math.radians(p('vfov_deg', 62.0).value)
        self.max_range = p('max_range_m', 8.0).value
        self.gate_range = p('gate_max_range_m', 14.0).value
        self.p_detect = p('p_detect', 0.95).value
        self.r_noise = p('range_noise_frac', 0.05).value
        self.b_noise = math.radians(p('bearing_noise_deg', 1.5).value)
        self.blind = {c.strip() for c in p('blind_classes', '').value.split(',') if c.strip()}
        self.rng = random.Random(p('seed', 7).value)
        camera_enabled = p('camera_enabled', False).value
        if self.source == 'camera' and not camera_enabled:
            self.get_logger().warn('detection_source=camera but camera disabled; using ground_truth')
            self.source = 'ground_truth'
        self.pose = None
        self.create_subscription(Odometry, '/auv/odom', self.on_odom, 20)
        self.pub = self.create_publisher(DetectionArray, '/auv/detections', 10)
        if self.source == 'camera':
            self.create_subscription(Image, '/auv/camera/image', self.on_image, 2)
        else:
            self.create_timer(0.2, self.tick_gt)
        self.get_logger().info(f'detector source={self.source} blind={sorted(self.blind)}')

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = (p.x, p.y, p.z, yaw_from_quat(q))

    def _make(self, cls, rng, bearing, elev, conf, box):
        x, y, z, yaw = self.pose
        cx, cy = body_to_world(CAM_OFFSET, 0.0, yaw)
        horiz = rng * math.cos(elev)
        wx, wy = body_to_world(horiz * math.cos(bearing), horiz * math.sin(bearing), yaw)
        d = Detection()
        d.class_name, d.confidence = cls, float(conf)
        d.range_m, d.bearing_rad, d.elevation_rad = float(rng), float(bearing), float(elev)
        d.position.x, d.position.y, d.position.z = x + cx + wx, y + cy + wy, z + rng * math.sin(elev)
        d.x1, d.y1, d.x2, d.y2 = box
        d.source = self.source
        return d

    def _publish(self, dets):
        m = DetectionArray()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'world'
        m.detections = [d for d in dets if d.class_name not in self.blind]
        self.pub.publish(m)

    # ------------------------------------------------------------ ground truth
    def tick_gt(self):
        if self.pose is None:
            return
        dets = []
        for o in self.arena['objects']:
            rng, bearing, elev, bx = project(o, self.pose, self.hfov, self.vfov)
            max_r = self.gate_range if o['class'] == 'gate' else self.max_range
            if bx < 0.2 or rng > max_r or abs(bearing) > self.hfov / 2 or abs(elev) > self.vfov / 2:
                continue
            if self.rng.random() > self.p_detect * (1.0 - 0.3 * rng / max_r):
                continue
            n_rng = rng * (1.0 + self.rng.gauss(0, self.r_noise))
            n_b = bearing + self.rng.gauss(0, self.b_noise)
            n_e = elev + self.rng.gauss(0, self.b_noise)
            w, h = (o['width'], o['height']) if o['class'] == 'gate' else (BUOY_DIAM, BUOY_DIAM)
            conf = max(0.3, min(0.99, 1.0 - 0.5 * rng / max_r + self.rng.gauss(0, 0.03)))
            dets.append(self._make(o['class'], n_rng, n_b, n_e, conf, bbox(n_rng, n_b, n_e, w, h, self.hfov)))
        self._publish(dets)

    # ------------------------------------------------------------ camera (HSV)
    def on_image(self, msg):
        if self.pose is None:
            return
        import cv2
        import numpy as np
        from auv_sim.save_camera_frame import image_to_bgr
        hsv = cv2.cvtColor(image_to_bgr(msg), cv2.COLOR_BGR2HSV)
        f = (IMG_W / 2) / math.tan(self.hfov / 2)
        dets = []
        for cls, ranges in HSV.items():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
            # blobs cut by the frame border give wrong centroids/sizes -> unusable for range/elevation
            blobs = [(stats[i], cent[i]) for i in range(1, n) if stats[i][cv2.CC_STAT_AREA] >= 12
                     and stats[i][1] > 1 and stats[i][1] + stats[i][3] < IMG_H - 1
                     and stats[i][0] > 1 and stats[i][0] + stats[i][2] < IMG_W - 1]
            if not blobs:
                continue
            if cls == 'gate':
                posts = sorted(blobs, key=lambda b: -b[0][cv2.CC_STAT_HEIGHT])[:2]
                if len(posts) < 2:
                    continue
                (s1, c1), (s2, c2) = posts
                sep = abs(c1[0] - c2[0])
                gate_w = next(o['width'] for o in self.arena['objects'] if o['class'] == 'gate')
                if sep < 5:
                    continue
                rng = f * gate_w / sep
                u, v = (c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2
                box = (int(min(s1[0], s2[0])), int(min(s1[1], s2[1])),
                       int(max(s1[0] + s1[2], s2[0] + s2[2])), int(max(s1[1] + s1[3], s2[1] + s2[3])))
            else:
                s, c = max(blobs, key=lambda b: b[0][cv2.CC_STAT_AREA])
                diam_px = max(s[cv2.CC_STAT_WIDTH], s[cv2.CC_STAT_HEIGHT])
                rng = f * BUOY_DIAM / max(diam_px, 1)
                u, v = c
                box = (int(s[0]), int(s[1]), int(s[0] + s[2]), int(s[1] + s[3]))
            bearing = math.atan((IMG_W / 2 - u) / f)
            elev = math.atan((IMG_H / 2 - v) / f)
            if rng > (self.gate_range if cls == 'gate' else self.max_range):
                continue   # same underwater visibility limit as the ground-truth model
            conf = max(0.3, min(0.95, 0.5 + (box[2] - box[0]) / 200.0))
            dets.append(self._make(cls, rng, bearing, elev, conf, box))
        self._publish(dets)


def main():
    spin_node(Detector)
