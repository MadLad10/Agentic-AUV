"""Sonar map-matching against the known pool walls (sonar_mapmatch.py)."""
import math

import mujoco
import numpy as np

from auv_marine_sim.sim import MarineSim
from auv_onboard.sonar_mapmatch import pool_rect, ray_to_rect, select_and_linearise

RECT = (-5.0, 25.0, -10.0, 10.0)


def test_ray_to_rect_distances_and_incidence():
    d, inc = ray_to_rect(0.0, 0.0, 0.0, *RECT)
    assert d == 25.0 and inc == 0.0
    d, inc = ray_to_rect(0.0, 0.0, math.pi / 2, *RECT)
    assert abs(d - 10.0) < 1e-9
    d, _ = ray_to_rect(0.0, 0.0, math.pi / 4, *RECT)
    assert abs(d - 10.0 * math.sqrt(2)) < 1e-9          # hits the side wall first
    assert ray_to_rect(30.0, 0.0, 0.0, *RECT)[0] == math.inf


def _scan_at(x, y, yaw, seed=0):
    s = MarineSim('default', seed=seed)
    s.data.qpos[:3] = [x, y, -1.5]
    s.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(s.model, s.data)
    scan = s._sonar(s.truth())
    rect, off, mx = pool_rect(s.arena['pool']), s.cfg['sensors']['sonar']['offset'], s.cfg['sensors']['sonar']['max_range_m']
    s.close()
    return scan, rect, off, mx


def test_wall_beams_recover_a_wrong_pose_estimate():
    true = np.array([16.0, 6.0, 1.0])                     # near a corner: two walls in view
    scan, rect, off, mx = _scan_at(*true)
    cfg = {'beam_stride': 2, 'max_range_m': mx, 'range_std_m': 0.15, 'gate_m': 1.5, 'max_incidence_deg': 70}
    est = true + np.array([0.6, -0.4, math.radians(6)])  # drifted estimate
    assert len(select_and_linearise(est, scan, off, rect, cfg)) >= 6
    for _ in range(6):                                    # Gauss-Newton on the wall ranges alone
        rows = select_and_linearise(est, scan, off, rect, cfg)
        H = np.array([r[1] for r in rows])
        y = np.array([r[0] for r in rows])
        est = est + np.linalg.lstsq(H, y, rcond=None)[0]
    assert np.hypot(*(est[:2] - true[:2])) < 0.15 and abs(est[2] - true[2]) < math.radians(1.5)


def test_returns_from_objects_are_not_used_as_walls():
    scan, rect, off, mx = _scan_at(3.0, 0.0, 0.0)          # gate posts and buoys in front, far wall at 22 m
    rows = select_and_linearise((3.0, 0.0, 0.0), scan, off, rect,
                                {'beam_stride': 1, 'max_range_m': mx, 'gate_m': 0.8})
    for innov, _, _ in rows:
        assert abs(innov) < 0.8
