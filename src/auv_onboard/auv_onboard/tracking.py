"""Camera-sonar multi-object tracker (opti-acoustic fusion) in the world frame.

Why: the camera classifies (colour) but its range is poor and it goes blind in turbid water; the
forward-looking sonar ranges precisely through murk but cannot tell a red buoy from a green one.
Fusing both in one tracker gives tracks that are accurate (sonar), labelled (camera), persistent
(survive camera dropouts) and self-checking (a strong sonar track the camera should see but never
does is a suspected false target).

  state     per track: position (x, y) with a 2x2 covariance (static objects + small process noise
            that absorbs navigation drift)
  measure   camera: (x, y) from range/bearing with polar->Cartesian covariance, class + confidence
            sonar:  (x, y) with tight range / beam-width bearing covariance, coarse class only
                    ('buoy' or 'gate_post')
  associate two stages (ByteTrack-style): confident detections first, then low-confidence ones to
            the remaining tracks; Hungarian assignment on Mahalanobis distance with a chi-square gate
            and class compatibility
  classify  Dirichlet-style evidence counts from camera labels (confidence-weighted)
  lifecycle tentative -> confirmed after `confirm_hits`; deleted after `max_age_s` unseen
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

CHI2_2_99 = 9.21
CHI2_2_999 = 13.82
BUOY_CLASSES = {'red_buoy', 'green_buoy', 'yellow_buoy', 'purple_buoy'}
_ids = itertools.count(1)


@dataclass
class Meas:
    x: float
    y: float
    R: np.ndarray                  # 2x2 covariance
    source: str                    # camera | sonar
    cls: Optional[str]             # camera: full class; sonar: 'buoy' | 'gate_post'
    conf: float
    depth: Optional[float] = None
    t: float = 0.0


@dataclass
class Track:
    x: np.ndarray
    P: np.ndarray
    t_first: float
    t_last: float
    kind: str                      # 'buoy' | 'gate' | 'gate_post'
    id: int = field(default_factory=lambda: next(_ids))
    evidence: dict = field(default_factory=dict)    # class -> confidence-weighted count (camera)
    hits: int = 0
    camera_hits: int = 0
    sonar_hits: int = 0
    in_view_unseen: int = 0        # camera should have seen it (good visibility, in FOV) but did not
    cam_t_last: float = -1.0
    sonar_t_last: float = -1.0

    def last_by(self, src: str) -> float:
        return self.cam_t_last if src == 'camera' else self.sonar_t_last
    depth: Optional[float] = None
    confirmed: bool = False

    @property
    def cls(self) -> Optional[str]:
        if not self.evidence:
            return None
        c, w = max(self.evidence.items(), key=lambda kv: kv[1])
        return c if w >= 1.0 else None

    @property
    def confidence(self) -> float:
        tot = sum(self.evidence.values())
        if not tot:
            return 0.3 if self.sonar_hits else 0.0
        return round(min(0.99, max(self.evidence.values()) / tot * (1 - math.exp(-tot / 2))), 3)

    def as_dict(self):
        return {'id': self.id, 'class': self.cls or ('unknown_' + self.kind), 'kind': self.kind,
                'x': round(float(self.x[0]), 2), 'y': round(float(self.x[1]), 2),
                'depth': None if self.depth is None else round(self.depth, 2),
                'pos_std_m': round(math.sqrt(max(np.trace(self.P), 0.0) / 2), 3),
                'confidence': self.confidence, 'camera_hits': self.camera_hits, 'sonar_hits': self.sonar_hits,
                'confirmed': self.confirmed, 'sources': [s for s, n in (('camera', self.camera_hits),
                                                                          ('sonar', self.sonar_hits)) if n]}


def polar_cov(rng: float, bearing_world: float, sr: float, sb: float, nav_std: float = 0.0) -> np.ndarray:
    """Covariance of a range/bearing measurement mapped to (x, y), plus navigation uncertainty."""
    c, s = math.cos(bearing_world), math.sin(bearing_world)
    J = np.array([[c, -rng * s], [s, rng * c]])
    return J @ np.diag([sr * sr, sb * sb]) @ J.T + np.eye(2) * nav_std ** 2


def _kind(m: Meas) -> str:
    if m.cls in ('gate', 'gate_post'):
        return m.cls
    return 'buoy'


def _compatible(t: Track, m: Meas) -> bool:
    k = _kind(m)
    if t.kind != k:
        return False
    if m.source == 'camera' and t.cls is not None and m.cls in BUOY_CLASSES and t.cls != m.cls:
        return False               # a confidently red track does not take a green detection
    return True


class FusionTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.tracks: list[Track] = []
        self.t = 0.0

    # ------------------------------------------------------------------ main update
    def update(self, t: float, meas: list[Meas], camera_view=None) -> list[Track]:
        """camera_view(x, y) -> True if the camera should reliably see a target there now."""
        dt = max(0.0, t - self.t)
        self.t = t
        q = self.cfg['process_noise_mps'] ** 2 * dt
        for tr in self.tracks:
            tr.P = tr.P + np.eye(2) * q
        # sequential per-sensor updates (each sensor sees every object at most once per cycle):
        # sonar first (precise range), then camera; within a sensor, confident detections first
        for src in ('sonar', 'camera'):
            ms = [m for m in meas if m.source == src]
            hi = [m for m in ms if m.conf >= self.cfg['high_conf']]
            lo = [m for m in ms if m.conf < self.cfg['high_conf']]
            leftovers = []
            for group in (hi, lo):
                # matching cascade (DeepSORT): established tracks get first pick, so a young uncertain
                # track cannot steal an established object's detections
                remaining = list(range(len(group)))
                for tier in (True, False):
                    # each sensor updates a track at most once per cycle
                    tids = [i for i, tr in enumerate(self.tracks)
                            if tr.confirmed == tier and tr.last_by(src) != t]
                    sub = [group[i] for i in remaining]
                    pairs, _, um = self._associate(tids, sub)
                    for ti, mi in pairs:
                        self._correct(self.tracks[ti], sub[mi], t)
                    remaining = [remaining[j] for j in um]
                leftovers += [group[i] for i in remaining if group is hi or src == 'sonar']
            for m in leftovers:        # births (low-conf camera can't start tracks); join a same-cycle birth first
                near = self._nearest(m, [tr for tr in self.tracks if tr.t_first == t])
                if near is None and m.source == 'camera' and m.cls:
                    # class-aware second chance: a labelled detection near a track already carrying that
                    # label is the same object with a large range error, not a new object
                    near = self._nearest(m, [tr for tr in self.tracks if tr.cls == m.cls], CHI2_2_999)
                if near is not None:
                    self._correct(near, m, t)
                else:
                    self._birth(m, t)
        self._merge()
        if camera_view is not None:
            for tr in self.tracks:            # the camera should see it now, but did not this cycle
                if tr.cam_t_last != t and camera_view(tr.x[0], tr.x[1]):
                    tr.in_view_unseen += 1
        self.tracks = [tr for tr in self.tracks if t - tr.t_last <= self.cfg['max_age_s']]
        return self.tracks

    def _nearest(self, m: Meas, tracks: list[Track], gate: float = CHI2_2_99) -> Optional[Track]:
        best, bd = None, gate
        for tr in tracks:
            if not _compatible(tr, m):
                continue
            d = np.array([m.x, m.y]) - tr.x
            md = float(d @ np.linalg.solve(tr.P + m.R, d))
            if md < bd:
                best, bd = tr, md
        return best

    def _merge(self):
        """Fold duplicate tracks (same kind, statistically the same place, compatible classes)."""
        keep: list[Track] = []
        for tr in sorted(self.tracks, key=lambda t: (t.t_first, t.id)):
            dup = None
            for k in keep:
                if k.kind != tr.kind or (k.cls and tr.cls and k.cls != tr.cls):
                    continue
                d = tr.x - k.x
                gate = CHI2_2_999 if (k.cls and k.cls == tr.cls) else CHI2_2_99
                too_close = tr.kind == 'buoy' and float(np.hypot(*d)) < self.cfg['min_separation_m']
                if too_close or float(d @ np.linalg.solve(k.P + tr.P, d)) < gate:
                    dup = k
                    break
            if dup is None:
                keep.append(tr)
                continue
            S = dup.P + tr.P                         # information-weighted fusion of the two estimates
            K = dup.P @ np.linalg.inv(S)
            dup.x = dup.x + K @ (tr.x - dup.x)
            dup.P = (np.eye(2) - K) @ dup.P
            for c, w in tr.evidence.items():
                dup.evidence[c] = dup.evidence.get(c, 0.0) + w
            dup.hits += tr.hits
            dup.camera_hits += tr.camera_hits
            dup.sonar_hits += tr.sonar_hits
            dup.t_last = max(dup.t_last, tr.t_last)
            dup.cam_t_last = max(dup.cam_t_last, tr.cam_t_last)
            dup.sonar_t_last = max(dup.sonar_t_last, tr.sonar_t_last)
            dup.in_view_unseen = min(dup.in_view_unseen, tr.in_view_unseen)
            dup.confirmed = dup.confirmed or tr.confirmed
        self.tracks = keep

    def _associate(self, track_idx: list[int], ms: list[Meas]):
        if not track_idx or not ms:
            return [], track_idx, list(range(len(ms)))
        C = np.full((len(track_idx), len(ms)), 1e6)
        for a, ti in enumerate(track_idx):
            tr = self.tracks[ti]
            for b, m in enumerate(ms):
                if not _compatible(tr, m):
                    continue
                S = tr.P + m.R
                d = np.array([m.x, m.y]) - tr.x
                md = float(d @ np.linalg.solve(S, d))
                if md <= CHI2_2_99:
                    C[a, b] = md
        rows, cols = linear_sum_assignment(C)
        pairs = [(track_idx[r], c) for r, c in zip(rows, cols) if C[r, c] < 1e6]
        mt = {p[0] for p in pairs}
        mm = {p[1] for p in pairs}
        return pairs, [i for i in track_idx if i not in mt], [j for j in range(len(ms)) if j not in mm]

    def _correct(self, tr: Track, m: Meas, t: float):
        S = tr.P + m.R
        K = tr.P @ np.linalg.inv(S)
        tr.x = tr.x + K @ (np.array([m.x, m.y]) - tr.x)
        tr.P = (np.eye(2) - K) @ tr.P
        self._bookkeep(tr, m, t)

    def _birth(self, m: Meas, t: float):
        tr = Track(np.array([m.x, m.y], float), m.R.copy(), t, t, _kind(m))
        self.tracks.append(tr)
        self._bookkeep(tr, m, t)

    def _bookkeep(self, tr: Track, m: Meas, t: float):
        tr.t_last = t
        tr.hits += 1
        if m.source == 'camera':
            tr.camera_hits += 1
            tr.cam_t_last = t
            tr.in_view_unseen = 0
            if m.cls:
                tr.evidence[m.cls] = tr.evidence.get(m.cls, 0.0) + m.conf
            if m.depth is not None:
                tr.depth = m.depth if tr.depth is None else 0.7 * tr.depth + 0.3 * m.depth
        else:
            tr.sonar_hits += 1
            tr.sonar_t_last = t
        tr.confirmed = tr.confirmed or tr.hits >= self.cfg['confirm_hits']

    # ------------------------------------------------------------------ queries
    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed]

    def on_gate_structure(self, tr: Track) -> bool:
        """A buoy-like track that is really part of the gate: on a post track (a post seen without its
        partner) or inside the frame of a camera-confirmed gate."""
        sep = self.cfg['min_separation_m']
        for o in self.tracks:
            if not o.confirmed or o is tr:
                continue
            d = float(np.hypot(*(tr.x - o.x)))
            if (o.kind == 'gate_post' and d <= sep) or (o.kind == 'gate' and d <= self.cfg['gate_exclusion_m']):
                return True
        return False

    def false_target_suspects(self) -> list[Track]:
        """Strong sonar tracks the camera should have seen repeatedly but never classified."""
        return [t for t in self.tracks if t.confirmed and t.kind == 'buoy' and t.camera_hits == 0
                and t.sonar_hits >= self.cfg['confirm_hits'] and t.in_view_unseen >= self.cfg['false_target_unseen']
                and not self.on_gate_structure(t)]

    def unclassified(self) -> list[Track]:
        """Confirmed buoy-like sonar tracks without a camera label (inspection candidates). Excludes
        suspected false targets and 'buoys' that sit on a gate post (a post seen without its partner)."""
        sus = {t.id for t in self.false_target_suspects()}
        return [t for t in self.tracks if t.confirmed and t.kind == 'buoy' and t.cls is None and t.id not in sus
                and not self.on_gate_structure(t)]

    def sonar_gate(self, gate_w: float, tol: float):
        """Gate centre from two confirmed post tracks about a gate-width apart."""
        posts = [t for t in self.tracks if t.kind == 'gate_post' and t.confirmed]
        best = None
        for a, b in itertools.combinations(posts, 2):
            d = float(np.linalg.norm(a.x - b.x))
            if abs(d - gate_w) < tol and (best is None or abs(d - gate_w) < best[0]):
                best = (abs(d - gate_w), (a.x + b.x) / 2)
        return None if best is None else best[1]
