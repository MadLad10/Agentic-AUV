"""Onboard stack: thrust allocation, EKF, health monitor, and closed-loop missions on the MuJoCo sim."""
import math

import numpy as np
import pytest

from auv_marine_sim.world import load_config as load_sim_config
from auv_onboard.control import Allocator
from auv_onboard.ekf import NavEKF
from auv_onboard.factory import load_onboard_config, make_backend, make_registry
from auv_onboard.health import HealthMonitor, score_detection

THR = load_sim_config()['vehicle']['thrusters']
NAMES = list(THR)


def test_allocator_reproduces_wrench_and_masks_dead_thruster():
    a = Allocator(THR, 40.0)
    tau = np.array([20.0, 5.0, -10.0, 0.0, 0.0, 2.0])
    u = a.solve(tau)
    got = a.B @ u
    assert np.allclose(got[[0, 1, 2, 5]], tau[[0, 1, 2, 5]], atol=0.1)
    mask = np.ones(6)
    mask[NAMES.index('surge_left')] = 0
    u = a.solve(tau, mask)
    got = a.B @ u
    assert u[NAMES.index('surge_left')] == 0.0
    assert abs(got[0] - 20.0) < 0.5 and abs(got[5] - 2.0) < 0.5   # yaw recovered via sway thrusters


def test_allocator_saturates_uniformly():
    a = Allocator(THR, 40.0)
    u = a.solve(np.array([200.0, 0, 0, 0, 0, 0]))
    assert np.abs(u).max() == pytest.approx(40.0)


def _ekf():
    return NavEKF(load_onboard_config()['ekf'], (0.0, 0.0, -1.0), 0.0)


def test_ekf_static_with_dvl_pressure_compass_stays_put():
    ekf = _ekf()
    rng = np.random.default_rng(0)
    for k in range(400):
        t = k * 0.05
        ekf.predict({'t': t, 'gyro': [0, 0, rng.normal(0, 0.002)], 'accel': [0, 0, 9.81 + rng.normal(0, 0.03)],
                     'roll': 0.0, 'pitch': 0.0})
        if k % 4 == 0:
            ekf.update_dvl({'valid': True, 'vel': list(rng.normal(0, 0.01, 3))})
        if k % 2 == 0:
            ekf.update_pressure({'depth': 1.0 + rng.normal(0, 0.005)})
            ekf.update_compass({'yaw': rng.normal(0, math.radians(1.5))})
    s = ekf.state()
    assert math.hypot(s['x'], s['y']) < 0.1 and abs(s['z'] + 1.0) < 0.02 and abs(s['yaw']) < 0.03


def test_ekf_dead_reckoning_uncertainty_grows_and_landmark_gate():
    ekf = _ekf()
    p0 = ekf.pos_std()
    for k in range(600):
        ekf.predict({'t': k * 0.05, 'gyro': [0, 0, 0], 'accel': [0, 0, 9.81], 'roll': 0.0, 'pitch': 0.0})
    assert ekf.pos_std() > 3 * p0                     # no DVL: uncertainty grows
    assert ekf.update_landmark((8.0, 0.0), 8.0, 0.0, 0.3, math.radians(2)) is True
    assert ekf.update_landmark((8.0, 0.0), 2.0, 1.2, 0.3, math.radians(2)) is False   # outlier rejected
    assert ekf.stats['landmark'].rejected == 1


def _readings(t, **kw):
    r = {'dvl': {'t': t, 'valid': True, 'vel': [0, 0, 0]},
         'thrusters': {'t': t, 'names': NAMES, 'cmd_n': [0.0] * 6, 'current_a': [0.0] * 6},
         'pressure': {'t': t, 'depth': 1.0 + 0.001 * math.sin(7 * t)},
         'leak': {'t': t, 'leak': False}, 'battery': {'t': t, 'percent': 90.0},
         'sonar': {'t': t, 'valid': True}}
    r.update(kw)
    return r


def test_health_monitor_labels_and_veto():
    cfg = load_onboard_config()['health']
    h = HealthMonitor(cfg, NAMES)
    rep = h.update(0.5, _readings(0.5))
    assert rep.faults == [] and not rep.veto
    reports = []
    for k in range(1, 12):
        t = k * 0.5
        cmd = [10.0, 10.0, 0, 0, 0, 0]
        cur = [0.0, 1.2, 0, 0, 0, 0]                  # surge_left commanded but draws no current
        reports.append(h.update(t, _readings(t, dvl={'t': t, 'valid': False, 'vel': None},
                                             thrusters={'t': t, 'names': NAMES, 'cmd_n': cmd, 'current_a': cur})))
    last = reports[-1]
    assert {'dvl_dropout', 'thruster_dead'} <= last.types()
    assert last.dead_thrusters == ['surge_left'] and h.thruster_mask()[0] == 0.0
    assert 'dead_reckoning' in last.degraded_modes and not last.veto
    rep = h.update(6.5, _readings(6.5, leak={'t': 6.5, 'leak': True}))
    assert rep.veto and 'leak' in rep.veto_reason
    sc = score_detection(reports, [{'type': 'dvl_dropout', 't_start': 0.5}, {'type': 'thruster_dead', 't_start': 0.5}])
    assert sc['dvl_dropout']['detected'] and sc['thruster_dead']['detected']


def test_health_pressure_stuck_veto():
    h = HealthMonitor(load_onboard_config()['health'], NAMES)
    rep = None
    for k in range(10):
        t = k * 0.5
        rep = h.update(t, _readings(t, pressure={'t': t, 'depth': 1.234}))
    assert 'pressure_stuck' in rep.types() and rep.veto


# ------------------------------------------------------------------ closed loop on the sim
def _mission(faults=None, nav='ekf'):
    be = make_backend('default', faults=faults or [], seed=0,
                      onboard_overrides={'backend': {'nav_source': nav, 'max_rtf': None, 'idle_rtf': None}})
    reg, _ = make_registry(be)
    out = []
    for name, p, to in [('navigate_to', {'x': 2.0, 'y': 0.0, 'depth': 1.5}, 60),
                        ('navigate_to', {'target': 'gate'}, 120),
                        ('search_pattern', {'target': 'red_buoy'}, 120),
                        ('navigate_to', {'target': 'red_buoy'}, 120),
                        ('surface', {}, 90)]:
        out.append(reg.call(name, p, to))
        if not out[-1].ok:
            break
    return be, out


def test_closed_loop_gate_buoy_surface_on_ekf():
    be, obs = _mission()
    try:
        assert all(o.ok for o in obs), [o.error for o in obs]
        r = be.sim.referee.report()
        assert r['gate_passed'] and r['buoys_touched'] == ['red_buoy'] and r['surfaced_at_end']
        assert r['collisions'] == 0 and r['max_depth_m'] < 2.0
        assert r['localisation']['rmse_xy_m'] < 0.3
        assert not [f for rep in be.health_log for f in rep.faults]       # no false alarms
    finally:
        be.shutdown()


def test_dvl_dropout_degrades_localisation():
    be, _ = _mission([{'type': 'dvl_dropout', 't_start': 10.0}])
    try:
        loc = be.sim.referee.report()['localisation']
        assert loc['rmse_xy_m'] > 0.2                   # dead reckoning drifts
        assert 'dvl_dropout' in {f['type'] for rep in be.health_log for f in rep.faults}
    finally:
        be.shutdown()


def test_leak_veto_surfaces_vehicle():
    import time
    be, obs = _mission([{'type': 'leak', 't_start': 12.0}])
    try:
        assert obs[-1].error_code == 'health_veto'
        t_end = time.time() + 60
        while time.time() < t_end and any(t.is_alive() for t in be._threads):
            time.sleep(0.05)
        r = be.sim.referee.report()
        assert r['surfaced_at_end'] and r['collisions'] == 0
    finally:
        be.shutdown()
