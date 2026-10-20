#!/usr/bin/env python3
"""Training data for the learned DVL-outage velocity estimator.

Random-waypoint missions (random speeds, depths, headings, hover pauses) on the MuJoCo sim. Logged at
20 Hz: what the vehicle itself knows (IMU gyro/accel, AHRS roll/pitch, its own thruster commands,
depth rate from the pressure sensor) and the target, the true body-frame velocity over ground.
No camera rendering (health/tracking off) - nothing but physics and sensors.

  python3 scripts/gen_nav_data.py --episodes 60 --out data/nav/train.npz
"""
import argparse
import os
import sys

import numpy as np

FEATURES = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch',
            'thr_surge_l', 'thr_surge_r', 'thr_heave_f', 'thr_heave_b', 'thr_sway_f', 'thr_sway_b', 'depth_rate']


def episode(seed: int, seconds: float):
    from auv_onboard.factory import make_backend
    rng = np.random.default_rng(seed)
    be = make_backend('default', seed=seed, start_ticker=False,
                      onboard_overrides={'backend': {'nav_source': 'ground_truth', 'health_monitor': False,
                                                     'max_rtf': None, 'idle_rtf': None},
                                         'tracking': {'enabled': False}})
    X, Y = [], []
    last_depth = None

    def log(b):
        nonlocal last_depth
        r = b.sim.readings
        imu = r['imu']
        d = r['pressure']['depth']
        rate = 0.0 if last_depth is None else (d - last_depth) / b.sim.control_dt
        last_depth = d
        X.append(list(imu['gyro']) + list(imu['accel']) + [imu['roll'], imu['pitch']] +
                 [float(c) / 40.0 for c in b.sim.cmd] + [rate])
        Y.append(b.sim.truth().vel_body.tolist())
    be.tick_hooks.append(log)
    try:
        while be.now() < seconds:
            if rng.random() < 0.15:
                be.hold_depth(float(rng.uniform(0.6, 3.5)), float(rng.uniform(2, 8)), timeout_s=40)
                continue
            x, y = rng.uniform(-3, 22), rng.uniform(-8, 8)
            be.speed_scale = float(rng.uniform(0.3, 1.0))
            be.navigate(float(x), float(y), float(rng.uniform(0.5, 3.8)), timeout_s=60)
    finally:
        be.shutdown()
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=60)
    ap.add_argument('--seconds', type=float, default=180.0)
    ap.add_argument('--seed0', type=int, default=0)
    ap.add_argument('--out', default='data/nav/train.npz')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    xs, ys, ep = [], [], []
    for i in range(a.episodes):
        x, y = episode(a.seed0 + i, a.seconds)
        xs.append(x)
        ys.append(y)
        ep.append(np.full(len(x), i, np.int32))
        print(f'episode {i}: {len(x)} samples, |v| max {np.abs(y).max():.2f} m/s', flush=True)
    np.savez_compressed(a.out, X=np.concatenate(xs), Y=np.concatenate(ys), episode=np.concatenate(ep),
                        features=np.array(FEATURES))
    print(f'saved {a.out}: {sum(len(x) for x in xs)} samples', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
