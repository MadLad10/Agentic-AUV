"""YOLO detector on a rendered frame (skipped when no trained weights are present; they are gitignored)."""
import math
import os

import mujoco
import pytest

from auv_onboard.yolo_detector import resolve_weights

try:
    WEIGHTS = resolve_weights()
except FileNotFoundError:
    WEIGHTS = None


@pytest.mark.skipif(WEIGHTS is None, reason='no trained YOLO weights (scripts/train_yolo.py)')
def test_yolo_detects_gate_with_plausible_range_and_bearing():
    from auv_marine_sim.sim import MarineSim
    from auv_onboard.factory import load_onboard_config
    from auv_onboard.yolo_detector import YoloCameraDetector
    sim = MarineSim('default', seed=0)
    try:
        sim.data.qpos[:3] = [3.0, 0.0, -1.5]
        mujoco.mj_forward(sim.model, sim.data)
        det = YoloCameraDetector(load_onboard_config()['perception'], sim, WEIGHTS)
        gates = [d for d in det.detect() if d.cls == 'gate']
        assert gates, 'gate not detected 5 m ahead in clear water'
        g = gates[0]
        true_r = math.dist((3.26, 0.0, -1.5), (8.0, 0.0, -1.5))
        assert abs(g.range_m - true_r) / true_r < 0.2
        assert abs(math.degrees(g.bearing)) < 5
    finally:
        sim.close()
