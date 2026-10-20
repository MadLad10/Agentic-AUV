"""YOLO detector on a rendered frame (skipped when no trained weights are present; they are gitignored).

Runs in a subprocess: importing torch (its triton module) AFTER MuJoCo has created EGL contexts in the
same process segfaults on this machine, and earlier tests in the session have already rendered.
"""
import json
import subprocess
import sys

import pytest

from auv_onboard.yolo_detector import resolve_weights

try:
    WEIGHTS = resolve_weights()
except FileNotFoundError:
    WEIGHTS = None

CHECK = r'''
import json, math, sys
import ultralytics  # noqa: F401  (torch first, then EGL)
import mujoco
from auv_marine_sim.sim import MarineSim
from auv_onboard.factory import load_onboard_config
from auv_onboard.yolo_detector import YoloCameraDetector
sim = MarineSim('default', seed=0)
sim.data.qpos[:3] = [3.0, 0.0, -1.5]
mujoco.mj_forward(sim.model, sim.data)
det = YoloCameraDetector(load_onboard_config()['perception'], sim, sys.argv[1])
out = [{'cls': d.cls, 'range': d.range_m, 'bearing': d.bearing} for d in det.detect()]
sim.close()
print('RESULT ' + json.dumps(out))
'''


@pytest.mark.skipif(WEIGHTS is None, reason='no trained YOLO weights (scripts/train_yolo.py)')
def test_yolo_detects_gate_with_plausible_range_and_bearing():
    p = subprocess.run([sys.executable, '-c', CHECK, WEIGHTS], capture_output=True, text=True, timeout=300)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith('RESULT ')]
    assert lines, p.stderr[-2000:]
    gates = [d for d in json.loads(lines[-1][7:]) if d['cls'] == 'gate']
    assert gates, 'gate not detected 5 m ahead in clear water'
    true_r = 8.0 - 3.26
    assert abs(gates[0]['range'] - true_r) / true_r < 0.2
    assert abs(gates[0]['bearing']) < 0.09
