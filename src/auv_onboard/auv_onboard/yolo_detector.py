"""YOLO camera detector (trained from scratch on the synthetic auto-labelled set).

Image -> boxes -> relative measurements, using a pinhole camera model:
  bearing / elevation  from the box centre and the focal length
  range                from the box size and the object's known physical size
                       (gate: 2.0 m wide; buoys: 0.4 m diameter, from the arena)
The box size gives a range error of a few percent at short range, growing with distance and with
partial occlusion, which is what a real monocular detector delivers.
"""
from __future__ import annotations

import math
import os

import numpy as np

from auv_onboard.perception import RelDet

DEFAULT_WEIGHTS = 'data/models/auv_yolov8n_synth.pt'


def resolve_weights(path: str | None = None) -> str:
    cands = [path, os.environ.get('AUV_YOLO_WEIGHTS'), DEFAULT_WEIGHTS,
             os.path.join(os.environ.get('AUV_WS', ''), DEFAULT_WEIGHTS)]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError('YOLO weights not found; train with scripts/train_yolo.py or set AUV_YOLO_WEIGHTS')


class YoloCameraDetector:
    def __init__(self, cfg: dict, sim, weights: str | None = None, conf: float = 0.35, imgsz: int = 320):
        os.environ.setdefault('YOLO_OFFLINE', '1')
        from ultralytics import YOLO
        self.cfg = cfg
        self.sim = sim
        self.model = YOLO(resolve_weights(weights or cfg.get('yolo_weights')))
        self.conf = conf
        self.imgsz = imgsz
        cam = sim.cfg['sensors']['camera']
        self.W, self.H = cam['width'], cam['height']
        self.f = (self.W / 2) / math.tan(math.radians(cam['hfov_deg']) / 2)
        self.sizes = {}
        for o in sim.arena['objects']:
            if o['class'] == 'gate':          # the box spans the posts' outer edges, plus the crossbar
                post = 0.1
                self.sizes['gate'] = (o['width'] + post, o['height'] + post)
            else:
                self.sizes[o['class']] = (2 * o.get('radius', 0.2), 2 * o.get('radius', 0.2))
        self.last_image = None
        self.last_boxes = []

    def detect_image(self, img: np.ndarray) -> list[RelDet]:
        # ultralytics treats numpy input as OpenCV BGR (training images were read with cv2); the sim
        # renders RGB, so swap channels or every colour class is confused
        bgr = np.ascontiguousarray(img[..., ::-1])
        res = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, device='cpu', verbose=False)[0]
        out = []
        self.last_boxes = []
        for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            cls = res.names[int(k)]
            x0, y0, x1, y1 = (float(v) for v in b)
            self.last_boxes.append((cls, float(c), (x0, y0, x1, y1)))
            u, v = (x0 + x1) / 2, (y0 + y1) / 2
            bearing = math.atan2(self.W / 2 - u, self.f)
            elev = math.atan2(self.H / 2 - v, self.f)
            w_m, h_m = self.sizes.get(cls, (0.4, 0.4))
            # use the less-truncated box dimension (boxes clipped at the image edge are too small)
            edge = x0 <= 1 or x1 >= self.W - 1
            px, size = ((y1 - y0), h_m) if edge or cls != 'gate' and (y1 - y0) > (x1 - x0) else ((x1 - x0), w_m)
            rng = size * self.f / max(px, 1.0)
            out.append(RelDet(cls, float(c), rng, bearing, elev, (x0, y0, x1, y1), 'yolo'))
        return out

    def detect(self) -> list[RelDet]:
        img = self.sim.camera_image()
        self.last_image = img
        if img is None or img.max() == 0:
            return []
        return self.detect_image(img)
