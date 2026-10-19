"""Low-level control: PID on the (estimated) pose -> body wrench -> thruster forces.

This is the only code that produces thruster commands. Agents and tools never reach it; they go
through the guidance behaviours in the backend, which only set pose setpoints and modes.

Thrust allocation solves a weighted, regularised least-squares problem on the thruster
configuration matrix B (wrench = B @ u) with a per-thruster health mask, so a thruster flagged dead
by the health monitor is re-allocated automatically:
    u = argmin ||W (B_h u - tau)||^2 + lam ||u||^2,  B_h = B * mask
Roll is not actuated independently (both sway thrusters sit below the centre), so its weight is 0;
the hull is passively stable in roll and pitch (centre of mass below centre of buoyancy).
"""
from __future__ import annotations

import math

import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Pid:
    def __init__(self, kp, ki, kd, i_lim, i_zone=None):
        self.kp, self.ki, self.kd, self.i_lim = kp, ki, kd, i_lim
        self.i_zone = i_zone      # anti-windup: integrate only when |err| < i_zone
        self.i = 0.0

    def __call__(self, err, rate, dt):
        if self.i_zone is None or abs(err) < self.i_zone:
            self.i = max(-self.i_lim, min(self.i_lim, self.i + err * dt))
        return self.kp * err + self.ki * self.i - self.kd * rate

    def reset(self):
        self.i = 0.0


class Allocator:
    WEIGHTS = np.array([1.0, 1.0, 1.0, 0.0, 0.3, 1.0])   # Fx Fy Fz Mx My Mz

    def __init__(self, thrusters: dict, max_thrust: float, lam: float = 1e-3):
        self.names = list(thrusters)
        cols = []
        for x, y, z, dx, dy, dz in thrusters.values():
            r, d = np.array([x, y, z], float), np.array([dx, dy, dz], float)
            cols.append(np.concatenate([d, np.cross(r, d)]))
        self.B = np.array(cols).T                 # 6 x n
        self.max = max_thrust
        self.lam = lam

    def solve(self, tau, mask=None) -> np.ndarray:
        mask = np.ones(self.B.shape[1]) if mask is None else np.asarray(mask, float)
        Bh = self.B * mask
        W2 = np.diag(self.WEIGHTS ** 2)
        A = Bh.T @ W2 @ Bh + self.lam * np.eye(Bh.shape[1])
        u = np.linalg.solve(A, Bh.T @ W2 @ np.asarray(tau, float)) * mask
        peak = np.abs(u).max()
        if peak > self.max:                       # scale uniformly: keeps the wrench direction
            u *= self.max / peak
        return u


class Controller:
    """Modes: idle (no thrust), hold (track setpoint), surface (gentle ascent, hold x/y/yaw)."""

    def __init__(self, cfg: dict, thrusters: dict, max_thrust: float):
        c = cfg
        self.alloc = Allocator(thrusters, max_thrust)
        self.pid_x = Pid(c['kp_xy'], 0.0, c['kd_xy'], 1.0)
        self.pid_y = Pid(c['kp_xy'], 0.0, c['kd_xy'], 1.0)
        self.pid_z = Pid(c['kp_z'], c['ki_z'], c['kd_z'], 1.5, i_zone=0.25)
        self.pid_yaw = Pid(c['kp_yaw'], 0.0, c['kd_yaw'], 0.5)
        self.hard_max_depth = c['hard_max_depth_m']
        self.surface_thrust = c.get('surface_thrust_n', 15.0)
        self.max_thrust = max_thrust
        self.mode = 'idle'
        self.sp = None             # (x, y, z, yaw)
        self.mask = np.ones(len(thrusters))
        self.blind_ascent = False          # depth sensor untrusted: push up regardless of estimate
        self.last_wrench = np.zeros(6)

    def set_mode(self, mode: str):
        if mode not in ('idle', 'hold', 'surface'):
            raise ValueError(mode)
        if mode != self.mode:
            for p in (self.pid_x, self.pid_y, self.pid_z, self.pid_yaw):
                p.reset()
        self.mode = mode

    def compute(self, state, dt: float, failsafe_depth: float | None = None) -> np.ndarray:
        """state: dict with x, y, z, yaw, vx, vy, vz (world), r (yaw rate).
        failsafe_depth: raw pressure-sensor depth; the hard depth limit trips on either source."""
        n = self.alloc.B.shape[1]
        if self.mode == 'idle' or self.sp is None or state is None:
            self.last_wrench = np.zeros(6)
            return np.zeros(n)
        x, y, z, yaw = state['x'], state['y'], state['z'], state['yaw']
        sx, sy, sz, syaw = self.sp
        fx_w = self.pid_x(sx - x, state['vx'], dt)
        fy_w = self.pid_y(sy - y, state['vy'], dt)
        c, s = math.cos(yaw), math.sin(yaw)
        fx, fy = c * fx_w + s * fy_w, -s * fx_w + c * fy_w
        mz = self.pid_yaw(wrap(syaw - yaw), state['r'], dt)
        if self.mode == 'surface':
            fz = 0.0 if z > -0.3 else self.surface_thrust     # gentle ascent, buoyancy finishes
            if self.blind_ascent:
                fz = self.surface_thrust
        else:
            fz = self.pid_z(sz - z, state['vz'], dt)
        depth = -z if failsafe_depth is None else max(-z, failsafe_depth)   # raw pressure reading too
        if depth > self.hard_max_depth:                     # failsafe, independent of planning
            fz = self.max_thrust
        tau = np.array([fx, fy, fz, 0.0, 0.0, mz])
        self.last_wrench = tau
        return self.alloc.solve(tau, self.mask)
