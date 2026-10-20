"""Obstacle-aware legs around the gate (pathing.py)."""
import math

from auv_onboard.pathing import Gate, _closest, plan_legs

GATE = Gate(8.0, 0.0, 0.0, 2.0, 0.9)     # posts at (8, +1) and (8, -1); crossbar at 0.9 m depth


def _clear_of_posts(path, clearance=0.65):
    for p, q in zip(path, path[1:]):
        for c in GATE.posts():
            cp, _ = _closest(p, q, c)
            if math.dist(cp, c) < clearance - 1e-6:
                return False
    return True


def test_straight_leg_through_gate_centre_needs_no_detour():
    assert plan_legs((2.0, 0.0), (12.0, 0.0), 1.5, GATE) == []


def test_leg_through_a_post_gets_a_detour():
    wps = plan_legs((2.0, 0.0), (12.5, 1.6), 1.5, GATE)     # the leg that scraped the left post
    assert wps
    path = [(2.0, 0.0)] + [w[:2] for w in wps] + [(12.5, 1.6)]
    assert _clear_of_posts(path)


def test_shallow_crossing_ducks_under_the_crossbar_but_still_passes():
    wps = plan_legs((2.0, 0.0), (12.0, 0.0), 0.5, GATE)
    assert len(wps) == 2
    assert all(w[2] >= 0.9 + 0.3 for w in wps)               # deep enough under the bar
    assert wps[0][0] < 8.0 < wps[1][0]                         # ... on both sides of the gate line


def test_legs_away_from_the_gate_are_untouched_and_no_gate_means_no_plan():
    assert plan_legs((14.0, -2.0), (14.0, 2.0), 1.5, GATE) == []
    assert plan_legs((2.0, 0.0), (12.5, 1.6), 1.5, None) == []


def test_non_target_buoys_are_avoided():
    green = (14.0, 2.0, 0.75)
    wps = plan_legs((10.0, 2.0), (18.0, 2.0), 1.5, None, circles=[green])     # straight through the green buoy
    assert wps
    path = [(10.0, 2.0)] + [w[:2] for w in wps] + [(18.0, 2.0)]
    for p, q in zip(path, path[1:]):
        cp, _ = _closest(p, q, green[:2])
        assert math.dist(cp, green[:2]) >= green[2] - 1e-6
