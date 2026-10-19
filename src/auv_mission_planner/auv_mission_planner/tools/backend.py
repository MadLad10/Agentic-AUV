"""Vehicle backend interface used by the tools layer, plus the detection memory. No ROS imports.

A backend exposes ONLY high-level, safety-checked vehicle behaviours. There is deliberately no
method for thrusters, raw velocities or controller setpoints.
"""
from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Pose:
    x: float
    y: float
    depth: float
    yaw: float   # rad, ENU (0 = +x, CCW positive)

    def as_dict(self):
        return {'x': round(self.x, 2), 'y': round(self.y, 2), 'depth': round(self.depth, 2),
                'yaw_deg': round(math.degrees(self.yaw), 1)}


@dataclass
class Det:
    cls: str
    confidence: float
    x: float
    y: float
    depth: float
    range_m: float
    bearing_deg: float
    t: float = 0.0

    def as_dict(self):
        return {'class': self.cls, 'confidence': round(self.confidence, 2),
                'position': {'x': round(self.x, 2), 'y': round(self.y, 2), 'depth': round(self.depth, 2)},
                'range_m': round(self.range_m, 2), 'bearing_deg': round(self.bearing_deg, 1)}


@dataclass
class Outcome:
    ok: bool
    message: str = ''
    code: Optional[str] = None      # timeout | action_failed | backend_unavailable | not_found
    data: dict = field(default_factory=dict)


class VehicleBackend(ABC):
    """Blocking, timeout-bounded high-level vehicle API (timeouts in sim seconds)."""

    def __init__(self):
        self._listeners = []

    def add_detection_listener(self, cb):
        self._listeners.append(cb)

    def _notify(self, dets: list[Det]):
        for cb in self._listeners:
            cb(dets)

    @abstractmethod
    def now(self) -> float: ...
    @abstractmethod
    def pose(self) -> Optional[Pose]: ...
    @abstractmethod
    def battery_percent(self) -> Optional[float]: ...
    @abstractmethod
    def detections(self) -> list[Det]: ...
    @abstractmethod
    def navigate(self, x, y, depth, yaw=None, tolerance=0.35, speed=None, timeout_s=60.0) -> Outcome: ...
    @abstractmethod
    def search(self, target, pattern, radius_m, depth, timeout_s=60.0) -> Outcome: ...
    @abstractmethod
    def hold_depth(self, depth, duration_s, tolerance=0.3, timeout_s=60.0) -> Outcome: ...
    @abstractmethod
    def surface(self, timeout_s=60.0) -> Outcome: ...
    @abstractmethod
    def emergency_surface(self) -> Outcome: ...
    @abstractmethod
    def cancel_all(self) -> None: ...

    def shutdown(self) -> None:
        pass


class WorldModel:
    """Remembers where objects were seen (confidence-weighted mean of recent sightings)."""

    def __init__(self, memory_s=120.0, min_confidence=0.4, max_samples=30):
        self.memory_s = memory_s
        self.min_conf = min_confidence
        self.max_samples = max_samples
        self._seen: dict[str, list[Det]] = {}
        self._lock = threading.Lock()

    def update(self, dets: list[Det]):
        with self._lock:
            for d in dets:
                if d.confidence < self.min_conf:
                    continue
                lst = self._seen.setdefault(d.cls, [])
                lst.append(d)
                del lst[:-self.max_samples]

    def lookup(self, cls: str, now: float, near_range: Optional[float] = None) -> Optional[Det]:
        """Estimate for cls; prefers close-range sightings (more accurate) when available."""
        with self._lock:
            lst = [d for d in self._seen.get(cls, []) if now - d.t <= self.memory_s]
        if not lst:
            return None
        if near_range is not None:
            close = [d for d in lst if d.range_m <= near_range]
            lst = close or lst
        # weight by confidence and 1/range (near sightings are more accurate)
        w = [d.confidence / max(d.range_m, 0.5) for d in lst]
        sw = sum(w)
        best = max(lst, key=lambda d: d.t)
        return Det(cls, max(d.confidence for d in lst),
                   sum(d.x * k for d, k in zip(lst, w)) / sw,
                   sum(d.y * k for d, k in zip(lst, w)) / sw,
                   sum(d.depth * k for d, k in zip(lst, w)) / sw,
                   best.range_m, best.bearing_deg, best.t)

    def summary(self, now: float) -> dict:
        out = {}
        for cls in list(self._seen):
            d = self.lookup(cls, now)
            if d:
                out[cls] = {'x': round(d.x, 2), 'y': round(d.y, 2), 'depth': round(d.depth, 2),
                            'last_seen_s_ago': round(now - d.t, 1)}
        return out

    def clear(self):
        with self._lock:
            self._seen.clear()
