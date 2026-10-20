"""Learned DVL-outage velocity estimator: GRU inference in pure numpy (no torch onboard).

Trained by scripts/train_dvl_net.py on simulated IMU + thrust-command history. While the DVL has no
bottom lock, its prediction of the body-frame velocity is fed to the EKF as a pseudo-DVL measurement
with the network's validation error (inflated) as noise, which bounds the drift of pure IMU dead
reckoning.
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class LearnedVelocity:
    def __init__(self, path: str, std_inflation: float = 1.5):
        w = np.load(path)
        self.mu, self.sd = w['mu'], w['sd']
        self.win = int(w['win'])
        self.w_ih, self.w_hh, self.b_ih, self.b_hh = w['w_ih'], w['w_hh'], w['b_ih'], w['b_hh']
        self.w_out, self.b_out = w['w_out'], w['b_out']
        self.H = self.w_hh.shape[1]
        self.std = np.maximum(np.asarray(w['val_rmse'], float) * std_inflation, 0.01)
        self.buf: deque = deque(maxlen=self.win)
        self.last_depth = None

    def push(self, imu: dict, cmd: np.ndarray, depth: float, dt: float):
        rate = 0.0 if self.last_depth is None else (depth - self.last_depth) / dt
        self.last_depth = depth
        f = np.array(list(imu['gyro']) + list(imu['accel']) + [imu['roll'], imu['pitch']] +
                     list(np.asarray(cmd, float) / 40.0) + [rate], float)
        self.buf.append((f - self.mu) / self.sd)

    def ready(self) -> bool:
        return len(self.buf) == self.win

    def predict(self) -> np.ndarray:
        h = np.zeros(self.H)
        H = self.H
        for x in self.buf:                       # PyTorch GRU equations, gate order r, z, n
            gi = self.w_ih @ x + self.b_ih
            gh = self.w_hh @ h + self.b_hh
            r = _sig(gi[:H] + gh[:H])
            z = _sig(gi[H:2 * H] + gh[H:2 * H])
            n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
            h = (1 - z) * n + z * h
        return self.w_out @ h + self.b_out


def default_path() -> str | None:
    for p in (os.environ.get('AUV_DVL_NET'), 'data/models/dvl_net.npz',
              os.path.join(os.environ.get('AUV_WS', ''), 'data/models/dvl_net.npz')):
        if p and os.path.isfile(p):
            return p
    return None
