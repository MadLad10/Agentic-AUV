"""Learned forward-looking-sonar detector (YOLO trained from scratch on auto-labelled FLS images).

Fan image -> boxes (buoy | gate_post) -> (bearing, range) by inverting the scan conversion ->
world position through the navigation estimate. Replaces the classical clustering of sonar returns
when `tracking.sonar_detection: yolo`.
"""
from __future__ import annotations

import math
import os

import numpy as np

from auv_onboard.perception import SonarContact

DEFAULT_WEIGHTS = 'data/models/auv_yolov8n_sonar.pt'


def resolve_weights(path=None):
    for c in (path, os.environ.get('AUV_SONAR_WEIGHTS'), DEFAULT_WEIGHTS,
              os.path.join(os.environ.get('AUV_WS', ''), DEFAULT_WEIGHTS)):
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError('sonar YOLO weights not found (scripts/train_yolo.py --data data/synthetic_sonar/...)')


class YoloSonarDetector:
    def __init__(self, sim, weights=None, conf=0.3, imgsz=448):
        os.environ.setdefault('YOLO_OFFLINE', '1')
        from ultralytics import YOLO               # imported at construction: before any EGL rendering
        self.model = YOLO(resolve_weights(weights))
        self.sim = sim
        self.conf, self.imgsz = conf, imgsz
        im = sim.cfg['sensors']['sonar'].get('image', {})
        self.H, self.W = im.get('height', 256), im.get('width', 448)
        self.scale = sim.cfg['sensors']['sonar']['max_range_m'] / self.H

    def pixel_to_polar(self, u, v):
        x = (self.W / 2 - 0.5 - u) * self.scale
        y = (self.H - 0.5 - v) * self.scale
        return math.atan2(x, y), math.hypot(x, y)

    def detect(self, est: dict, offset) -> list[tuple[str, float, SonarContact]]:
        img, _ = self.sim.sonar_image()
        if img is None:
            return []
        rgb = np.ascontiguousarray(np.repeat(img[..., None], 3, axis=2))
        res = self.model.predict(rgb, imgsz=self.imgsz, conf=self.conf, device='cpu', verbose=False)[0]
        yaw = est['yaw']
        c, s = math.cos(yaw), math.sin(yaw)
        ox = est['x'] + c * offset[0] - s * offset[1]
        oy = est['y'] + s * offset[0] + c * offset[1]
        out = []
        for b, p, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            u = float(b[0] + b[2]) / 2
            v = float(b[3])                          # nearest edge of the echo = target range
            brg, rng = self.pixel_to_polar(u, v)
            wx, wy = ox + rng * math.cos(yaw + brg), oy + rng * math.sin(yaw + brg)
            out.append((res.names[int(k)], float(p), SonarContact(rng, brg, 0.02, float(p), wx, wy)))
        return out
