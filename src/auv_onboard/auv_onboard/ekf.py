"""Extended Kalman filter for AUV navigation (visual-inertial-acoustic fusion).

State  s = [x, y, z, vx, vy, vz, yaw, b_gz]   (world position and velocity, heading, gyro-z bias)
Predict  IMU: specific force rotated to the world frame (roll/pitch from the AHRS, yaw from the
         state) plus gravity; heading rate from the gyro corrected for the estimated bias.
Updates  DVL       body-frame velocity over ground (when bottom lock is valid)
         pressure  depth
         compass   heading
         landmark  range + bearing to a mapped landmark (gate from the prior course map), seen by
                   the camera or sonar; Mahalanobis-gated so false targets are rejected
When the DVL drops out the filter keeps running on the IMU alone (dead reckoning): position
uncertainty grows, which the navigation agent reads from the covariance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CHI2_1 = 6.63     # 99% gate, 1 dof
CHI2_2 = 9.21     # 99% gate, 2 dof
CHI2_3 = 11.34    # 99% gate, 3 dof
IX, IY, IZ, IVX, IVY, IVZ, IYAW, IBG = range(8)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot(roll, pitch, yaw) -> np.ndarray:
    """Body -> world rotation (ZYX)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def drot_dyaw(roll, pitch, yaw) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return np.array([[-sy * cp, -sy * sp * sr - cy * cr, -sy * sp * cr + cy * sr],
                     [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [0.0, 0.0, 0.0]])


@dataclass
class UpdateStat:
    accepted: int = 0
    rejected: int = 0
    last_nis: float = 0.0


class NavEKF:
    def __init__(self, cfg: dict, x0=(0.0, 0.0, 0.0), yaw0=0.0, g=9.81):
        self.cfg = cfg
        self.g = g
        self.s = np.zeros(8)
        self.s[[IX, IY, IZ]] = x0
        self.s[IYAW] = yaw0
        p0 = cfg['initial_std']
        self.P = np.diag(np.array([p0['pos'], p0['pos'], p0['pos'], p0['vel'], p0['vel'], p0['vel'],
                                   math.radians(p0['yaw_deg']), p0['gyro_bias']]) ** 2)
        self.roll = 0.0
        self.pitch = 0.0
        self.t = None
        self.yaw_rate = 0.0
        self.stats = {k: UpdateStat() for k in ('dvl', 'pressure', 'compass', 'landmark')}
        self.last_update = {}

    # ----------------------------------------------------------------- predict
    def predict(self, imu: dict):
        t = imu['t']
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        if dt <= 0:
            return
        self.t = t
        self.roll, self.pitch = imu['roll'], imu['pitch']
        q = self.cfg['process_noise']
        yaw = self.s[IYAW]
        R = rot(self.roll, self.pitch, yaw)
        f_b = np.asarray(imu['accel'], float)
        a_w = R @ f_b - np.array([0.0, 0.0, self.g])
        gx, gy, gz = imu['gyro']
        # Euler-rate kinematics: yaw_dot = (q sin(roll) + r cos(roll)) / cos(pitch)
        cr, sr, cp = math.cos(self.roll), math.sin(self.roll), max(0.2, math.cos(self.pitch))
        yaw_rate = (gy * sr + (gz - self.s[IBG]) * cr) / cp
        self.yaw_rate = yaw_rate
        s = self.s
        s[IX:IZ + 1] += s[IVX:IVZ + 1] * dt + 0.5 * a_w * dt * dt
        s[IVX:IVZ + 1] += a_w * dt
        s[IYAW] = wrap(s[IYAW] + yaw_rate * dt)
        F = np.eye(8)
        F[IX:IZ + 1, IVX:IVZ + 1] = np.eye(3) * dt
        dA = drot_dyaw(self.roll, self.pitch, yaw) @ f_b
        F[IVX:IVZ + 1, IYAW] = dA * dt
        F[IX:IZ + 1, IYAW] = 0.5 * dA * dt * dt
        F[IYAW, IBG] = -cr / cp * dt
        Q = np.zeros((8, 8))
        Q[IVX:IVZ + 1, IVX:IVZ + 1] = np.eye(3) * q['accel'] ** 2 * dt
        Q[IX:IZ + 1, IX:IZ + 1] = np.eye(3) * (q['accel'] * dt) ** 2 * dt
        Q[IYAW, IYAW] = q['gyro'] ** 2 * dt
        Q[IBG, IBG] = q['gyro_bias_walk'] ** 2 * dt
        self.P = F @ self.P @ F.T + Q

    # ----------------------------------------------------------------- updates
    def _update(self, name, y, H, Rm, gate):
        S = H @ self.P @ H.T + Rm
        Si = np.linalg.inv(S)
        nis = float(y @ Si @ y)
        st = self.stats[name]
        st.last_nis = nis
        if gate is not None and nis > gate:
            st.rejected += 1
            return False
        K = self.P @ H.T @ Si
        self.s = self.s + K @ y
        self.s[IYAW] = wrap(self.s[IYAW])
        I_KH = np.eye(8) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T         # Joseph form
        st.accepted += 1
        self.last_update[name] = self.t
        return True

    def update_dvl(self, dvl: dict) -> bool:
        if not dvl.get('valid'):
            return False
        r = self.cfg['meas_noise']
        R = rot(self.roll, self.pitch, self.s[IYAW])
        v = self.s[IVX:IVZ + 1]
        h = R.T @ v
        H = np.zeros((3, 8))
        H[:, IVX:IVZ + 1] = R.T
        H[:, IYAW] = drot_dyaw(self.roll, self.pitch, self.s[IYAW]).T @ v
        y = np.asarray(dvl['vel'], float) - h
        return self._update('dvl', y, H, np.eye(3) * r['dvl_vel'] ** 2, CHI2_3 * 4)

    def update_pressure(self, p: dict) -> bool:
        r = self.cfg['meas_noise']
        H = np.zeros((1, 8))
        H[0, IZ] = -1.0
        y = np.array([p['depth'] - (-self.s[IZ])])
        return self._update('pressure', y, H, np.eye(1) * r['depth'] ** 2, CHI2_1 * 4)

    def update_compass(self, c: dict) -> bool:
        r = self.cfg['meas_noise']
        H = np.zeros((1, 8))
        H[0, IYAW] = 1.0
        y = np.array([wrap(c['yaw'] - self.s[IYAW])])
        return self._update('compass', y, H, np.eye(1) * math.radians(r['compass_deg']) ** 2, CHI2_1 * 4)

    def update_landmark(self, lm_xy, rng_m: float, bearing_rad: float, rng_std: float, brg_std: float) -> bool:
        """Horizontal range and bearing (body frame) to a landmark at known world (x, y)."""
        dx, dy = lm_xy[0] - self.s[IX], lm_xy[1] - self.s[IY]
        q = dx * dx + dy * dy
        rr = math.sqrt(max(q, 1e-6))
        h = np.array([rr, wrap(math.atan2(dy, dx) - self.s[IYAW])])
        H = np.zeros((2, 8))
        H[0, IX], H[0, IY] = -dx / rr, -dy / rr
        H[1, IX], H[1, IY], H[1, IYAW] = dy / q, -dx / q, -1.0
        y = np.array([rng_m - h[0], wrap(bearing_rad - h[1])])
        return self._update('landmark', y, H, np.diag([rng_std ** 2, brg_std ** 2]), CHI2_2)

    # ----------------------------------------------------------------- outputs
    def state(self) -> dict:
        s = self.s
        c, sn = math.cos(s[IYAW]), math.sin(s[IYAW])
        return {'x': s[IX], 'y': s[IY], 'z': s[IZ], 'yaw': s[IYAW], 'vx': s[IVX], 'vy': s[IVY],
                'vz': s[IVZ], 'u': c * s[IVX] + sn * s[IVY], 'r': self.yaw_rate}

    def pos_std(self) -> float:
        """1-sigma horizontal position uncertainty (m)."""
        return float(math.sqrt(max(self.P[IX, IX] + self.P[IY, IY], 0.0)))

    def summary(self) -> dict:
        return {'pos_std_m': round(self.pos_std(), 3),
                'depth_std_m': round(float(math.sqrt(self.P[IZ, IZ])), 3),
                'yaw_std_deg': round(math.degrees(math.sqrt(self.P[IYAW, IYAW])), 2),
                'gyro_bias_z': round(float(self.s[IBG]), 5),
                'updates': {k: {'ok': v.accepted, 'rejected': v.rejected} for k, v in self.stats.items()},
                'last_update_t': {k: round(v, 2) for k, v in self.last_update.items()}}
