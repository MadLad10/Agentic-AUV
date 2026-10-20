"""Camera-sonar fusion tracker on synthetic measurements."""
import math

import numpy as np

from auv_onboard.factory import load_onboard_config
from auv_onboard.tracking import FusionTracker, Meas, polar_cov

CFG = load_onboard_config()['tracking']


def cam(x, y, cls, rng, conf=0.8, gen=None):
    """Camera measurement from the origin: true noise 8% range / 2 deg; covariance from the tracker config."""
    r, b = math.hypot(x, y), math.atan2(y, x)
    r_n = r * (1 + gen.normal(0, 0.08))
    b_n = b + gen.normal(0, math.radians(2))
    return Meas(r_n * math.cos(b_n), r_n * math.sin(b_n),
                polar_cov(r_n, b_n, CFG['camera_range_std_frac'] * r_n, math.radians(CFG['camera_bearing_std_deg'])),
                'camera', cls, conf, depth=1.5)


def son(x, y, cls='buoy', gen=None):
    r, b = math.hypot(x, y), math.atan2(y, x)
    r_n, b_n = r + gen.normal(0, 0.08), b + gen.normal(0, math.radians(1))
    return Meas(r_n * math.cos(b_n), r_n * math.sin(b_n), polar_cov(r_n, b_n, 0.08, math.radians(1)), 'sonar', cls, 0.6)


def test_fusion_beats_camera_only_and_keeps_class():
    errs = {'camera': [], 'fused': []}
    for trial in range(8):
        for mode in ('camera', 'fused'):
            g = np.random.default_rng(100 * trial + (mode == 'fused'))
            tk = FusionTracker(CFG)
            for k in range(20):
                ms = [cam(10, 0, 'red_buoy', 10, gen=g)]
                if mode == 'fused':
                    ms.append(son(10, 0, gen=g))
                tk.update(k * 0.5, ms)
            tr = tk.confirmed()
            assert len(tr) == 1 and tr[0].cls == 'red_buoy', (mode, trial, [t.as_dict() for t in tr])
            errs[mode].append(float(np.hypot(tr[0].x[0] - 10, tr[0].x[1])))
    assert np.mean(errs['fused']) < 0.5 * np.mean(errs['camera']) and np.mean(errs['fused']) < 0.15


def test_nearby_buoys_do_not_swap_classes():
    g = np.random.default_rng(1)
    tk = FusionTracker(CFG)
    for k in range(20):
        tk.update(k * 0.5, [cam(12, -1.0, 'red_buoy', 12, gen=g), cam(12, 1.0, 'green_buoy', 12, gen=g),
                            son(12, -1.0, gen=g), son(12, 1.0, gen=g)])
    cls = sorted((str(t.cls), round(t.x[1])) for t in tk.confirmed())
    assert cls == [('green_buoy', 1), ('red_buoy', -1)]


def test_sonar_only_target_in_camera_view_is_flagged_false():
    g = np.random.default_rng(2)
    tk = FusionTracker(CFG)
    view = lambda x, y: math.hypot(x, y) < 6.0           # camera reliably sees anything closer than 6 m
    for k in range(12):
        tk.update(k * 0.5, [son(4, 0.5, gen=g), cam(5, -1.5, 'red_buoy', 5, gen=g), son(5, -1.5, gen=g)], view)
    sus = tk.false_target_suspects()
    assert len(sus) == 1 and abs(sus[0].x[0] - 4) < 0.3
    assert not tk.unclassified()                         # the phantom is not offered for inspection


def test_far_sonar_track_is_unclassified_not_false():
    g = np.random.default_rng(3)
    tk = FusionTracker(CFG)
    view = lambda x, y: math.hypot(x, y) < 3.0           # murky water: camera only sees 3 m
    for k in range(10):
        tk.update(k * 0.5, [son(9, 2, gen=g)], view)
    assert len(tk.unclassified()) == 1 and not tk.false_target_suspects()


def test_gate_from_two_post_tracks():
    g = np.random.default_rng(4)
    tk = FusionTracker(CFG)
    for k in range(6):
        tk.update(k * 0.5, [son(8, 1.0, 'gate_post', gen=g), son(8, -1.0, 'gate_post', gen=g)])
    c = tk.sonar_gate(2.0, 0.35)
    assert c is not None and abs(c[0] - 8) < 0.2 and abs(c[1]) < 0.2
