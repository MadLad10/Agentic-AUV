"""Factor-graph navigation (GTSAM ISAM2): a drop-in alternative to the EKF.

State per keyframe (2 Hz): Pose3 X_k, velocity V_k (world), IMU bias B_k.
Factors
  IMU         pre-integrated accelerometer + gyro between keyframes (CombinedImuFactor: includes the
              bias random walk)
  attitude    AHRS roll/pitch + compass yaw as a rotation prior (translation unconstrained)
  DVL         body-frame velocity: R_k^T V_k = v_dvl          (custom factor)
  depth       pressure depth: -z_k = d                          (custom factor)
  landmark    horizontal range + bearing to a mapped landmark   (custom factor)
Robust Huber kernels on DVL, compass and landmarks (outliers, e.g. a magnetic disturbance or a wrong
detection, are down-weighted instead of corrupting the estimate). Solved incrementally with ISAM2;
between keyframes the estimate is propagated with the current pre-integration so control runs at
20 Hz. Same public interface as `NavEKF` (the backend switches with `ekf.estimator: fgo`).

Literature: FGO-ILNS (tightly coupled factor-graph AUV navigation), robust factor graphs for DVL
outliers (2025). Custom-factor Jacobians are numerical (central differences on the manifold).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, V, X

from auv_onboard.ekf import UpdateStat, wrap


@dataclass
class _Pending:
    dvl: list
    depth: list
    compass: list
    landmarks: list


def _num_jac(f, values: list, kinds: list, eps=1e-6):
    """Central-difference Jacobians of f(values) w.r.t. each value (Pose3 -> 6, Point/Vector3 -> 3)."""
    e0 = f(values)
    jacs = []
    for i, (v, kd) in enumerate(zip(values, kinds)):
        n = 6 if kd == 'pose' else 3
        J = np.zeros((len(e0), n))
        for j in range(n):
            d = np.zeros(n)
            d[j] = eps
            vp, vm = list(values), list(values)
            if kd == 'pose':
                vp[i], vm[i] = v.retract(d), v.retract(-d)
            else:
                vp[i], vm[i] = v + d, v - d
            J[:, j] = (f(vp) - f(vm)) / (2 * eps)
        jacs.append(J)
    return e0, jacs


def _custom(keys, kinds, fun, noise):
    def err(this, values, H):
        vals = [values.atPose3(k) if kd == 'pose' else values.atVector(k) for k, kd in zip(this.keys(), kinds)]
        e, js = _num_jac(fun, vals, kinds)
        if H is not None:
            for i, J in enumerate(js):
                H[i] = J
        return e
    return gtsam.CustomFactor(noise, keys, err)


def _robust(sigmas, k=1.345):
    return gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k),
                                          gtsam.noiseModel.Diagonal.Sigmas(np.asarray(sigmas, float)))


class FactorGraphNav:
    def __init__(self, cfg: dict, x0=(0.0, 0.0, 0.0), yaw0=0.0, g=9.81):
        self.cfg = cfg
        f = cfg.get('fgo', {})
        self.dt_key = f.get('keyframe_dt', 0.5)
        q = cfg['process_noise']
        p = gtsam.PreintegrationCombinedParams.MakeSharedU(g)
        # GTSAM wants continuous-time noise densities: sigma_density = sigma_discrete * sqrt(dt_imu)
        dt_imu = f.get('imu_dt', 0.05)
        p.setAccelerometerCovariance(np.eye(3) * (f.get('accel_noise', 0.03) ** 2) * dt_imu)
        p.setGyroscopeCovariance(np.eye(3) * (f.get('gyro_noise', 0.002) ** 2) * dt_imu)
        p.setIntegrationCovariance(np.eye(3) * 1e-6)
        p.setBiasAccCovariance(np.eye(3) * f.get('accel_bias_walk', 1e-3) ** 2)
        p.setBiasOmegaCovariance(np.eye(3) * q['gyro_bias_walk'] ** 2)
        self.params = p
        self.isam = gtsam.ISAM2(gtsam.ISAM2Params())
        self.k = 0
        pose0 = gtsam.Pose3(gtsam.Rot3.Ypr(yaw0, 0.0, 0.0), gtsam.Point3(*x0))
        self.bias = gtsam.imuBias.ConstantBias()
        i0 = cfg['initial_std']
        graph = gtsam.NonlinearFactorGraph()
        init = gtsam.Values()
        graph.add(gtsam.PriorFactorPose3(X(0), pose0, gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [0.01, 0.01, math.radians(i0['yaw_deg']), i0['pos'], i0['pos'], i0['pos']]))))
        graph.add(gtsam.PriorFactorVector(V(0), np.zeros(3), gtsam.noiseModel.Isotropic.Sigma(3, i0['vel'])))
        graph.add(gtsam.PriorFactorConstantBias(B(0), self.bias, gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, i0['gyro_bias'], i0['gyro_bias'], i0['gyro_bias']]))))
        init.insert(X(0), pose0)
        init.insert(V(0), np.zeros(3))
        init.insert(B(0), self.bias)
        self.isam.update(graph, init)
        self.nav = gtsam.NavState(pose0, np.zeros(3))
        self.pim = gtsam.PreintegratedCombinedMeasurements(p, self.bias)
        self.t = None
        self.t_key = None
        self.roll = self.pitch = 0.0
        self.yaw_rate = 0.0
        self.pend = _Pending([], [], [], [])
        self.stats = {k: UpdateStat() for k in ('dvl', 'pressure', 'compass', 'landmark')}
        self.last_update = {}
        self._pos_std = i0['pos'] * math.sqrt(2)
        self._cov = np.eye(6) * i0['pos'] ** 2

    # ------------------------------------------------------------------ inputs (EKF-compatible)
    def predict(self, imu: dict):
        t = imu['t']
        if self.t is None:
            self.t = self.t_key = t
            return
        dt = t - self.t
        if dt <= 0:
            return
        self.t = t
        self.roll, self.pitch = imu['roll'], imu['pitch']
        acc, gyr = np.asarray(imu['accel'], float), np.asarray(imu['gyro'], float)
        self.pim.integrateMeasurement(acc, gyr, dt)
        self.yaw_rate = float(gyr[2] - self.bias.gyroscope()[2])
        if t - self.t_key >= self.dt_key - 1e-9:
            self._keyframe(t)

    def update_dvl(self, dvl: dict) -> bool:
        if not dvl.get('valid'):
            return False
        self.pend.dvl.append(np.asarray(dvl['vel'], float))
        return True

    def update_velocity_pseudo(self, vel_body, std) -> bool:
        self.pend.dvl.append(np.asarray(vel_body, float))
        return True

    def update_pressure(self, p: dict) -> bool:
        self.pend.depth.append(float(p['depth']))
        return True

    def update_compass(self, c: dict) -> bool:
        self.pend.compass.append(float(c['yaw']))
        return True

    def update_landmark(self, lm_xy, rng_m, bearing_rad, rng_std, brg_std) -> bool:
        self.pend.landmarks.append((tuple(lm_xy), float(rng_m), float(bearing_rad), float(rng_std), float(brg_std)))
        return True

    # ------------------------------------------------------------------ keyframe / optimisation
    def _keyframe(self, t):
        k0, k1 = self.k, self.k + 1
        r = self.cfg['meas_noise']
        graph = gtsam.NonlinearFactorGraph()
        init = gtsam.Values()
        graph.add(gtsam.CombinedImuFactor(X(k0), V(k0), X(k1), V(k1), B(k0), B(k1), self.pim))
        pred = self.pim.predict(self.nav, self.bias)
        init.insert(X(k1), pred.pose())
        init.insert(V(k1), pred.velocity())
        init.insert(B(k1), self.bias)
        # attitude: AHRS roll/pitch + compass yaw (robust: a magnetic disturbance becomes an outlier)
        # measurements are attached to the keyframe time: use the most recent reading (a window mean
        # would lag by half a keyframe interval during motion)
        yaw = self.pend.compass[-1] if self.pend.compass else None
        if yaw is not None:
            rp = math.radians(0.5)
            prior = gtsam.Pose3(gtsam.Rot3.Ypr(yaw, self.pitch, self.roll), pred.pose().translation())
            graph.add(gtsam.PriorFactorPose3(X(k1), prior, _robust(
                [rp, rp, math.radians(r['compass_deg']), 1e3, 1e3, 1e3])))
        if self.pend.dvl:
            vb = self.pend.dvl[-1]
            sd = r['dvl_vel']

            def e_dvl(vals, vb=vb):
                return vals[0].rotation().unrotate(vals[1]) - vb
            graph.add(_custom([X(k1), V(k1)], ['pose', 'vec'], e_dvl, _robust([sd * 2] * 3)))
            self.stats['dvl'].accepted += 1
            self.last_update['dvl'] = t
        if self.pend.depth:
            d = self.pend.depth[-1]

            def e_depth(vals, d=d):
                return np.array([-vals[0].translation()[2] - d])
            graph.add(_custom([X(k1)], ['pose'], e_depth, gtsam.noiseModel.Isotropic.Sigma(1, r['depth'])))
            self.stats['pressure'].accepted += 1
            self.last_update['pressure'] = t
        for (lx, ly), rng, brg, rs, bs in self.pend.landmarks:
            def e_lm(vals, lx=lx, ly=ly, rng=rng, brg=brg):
                p = vals[0]
                tx, ty = p.translation()[0], p.translation()[1]
                yaw_ = p.rotation().yaw()
                return np.array([math.hypot(lx - tx, ly - ty) - rng, wrap(math.atan2(ly - ty, lx - tx) - yaw_ - brg)])
            graph.add(_custom([X(k1)], ['pose'], e_lm, _robust([rs, bs])))
            self.stats['landmark'].accepted += 1
            self.last_update['landmark'] = t
        self.isam.update(graph, init)
        est = self.isam.calculateEstimate()
        self.nav = gtsam.NavState(est.atPose3(X(k1)), est.atVector(V(k1)))
        if yaw is not None:
            # robust kernels down-weight instead of rejecting; report >3 sigma compass residuals as
            # "rejected" so the health monitor's compass-disturbance check works as with the EKF
            res = abs(wrap(yaw - self.nav.pose().rotation().yaw()))
            st = self.stats['compass']
            if res > 3 * math.radians(r['compass_deg']):
                st.rejected += 1
            else:
                st.accepted += 1
                self.last_update['compass'] = t
        self.bias = est.atConstantBias(B(k1))
        self.pim.resetIntegrationAndSetBias(self.bias)
        try:
            self._cov = self.isam.marginalCovariance(X(k1))
            self._pos_std = float(math.sqrt(max(self._cov[3, 3] + self._cov[4, 4], 0.0)))
        except RuntimeError:
            pass
        self.k = k1
        self.t_key = t
        self.pend = _Pending([], [], [], [])

    # ------------------------------------------------------------------ outputs (EKF-compatible)
    def _current(self) -> gtsam.NavState:
        if self.pim.deltaTij() > 0:
            return self.pim.predict(self.nav, self.bias)
        return self.nav

    @property
    def s(self) -> np.ndarray:
        n = self._current()
        p, v = n.pose().translation(), n.velocity()
        return np.array([p[0], p[1], p[2], v[0], v[1], v[2], n.pose().rotation().yaw(), self.bias.gyroscope()[2]])

    def state(self) -> dict:
        s = self.s
        c, sn = math.cos(s[6]), math.sin(s[6])
        return {'x': s[0], 'y': s[1], 'z': s[2], 'yaw': s[6], 'vx': s[3], 'vy': s[4], 'vz': s[5],
                'u': c * s[3] + sn * s[4], 'r': self.yaw_rate}

    def pos_std(self) -> float:
        return self._pos_std

    def summary(self) -> dict:
        return {'estimator': 'factor_graph_isam2', 'keyframes': self.k, 'pos_std_m': round(self._pos_std, 3),
                'depth_std_m': round(float(math.sqrt(max(self._cov[5, 5], 0.0))), 3),
                'yaw_std_deg': round(math.degrees(math.sqrt(max(self._cov[2, 2], 0.0))), 2),
                'gyro_bias_z': round(float(self.bias.gyroscope()[2]), 5),
                'updates': {k: {'ok': v.accepted, 'rejected': v.rejected} for k, v in self.stats.items()},
                'last_update_t': {k: round(v, 2) for k, v in self.last_update.items()}}
