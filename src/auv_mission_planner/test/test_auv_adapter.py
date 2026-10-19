"""Pure translation helpers of the real-vehicle adapter (the ROS node itself needs the AUV hardware)."""
import math

import pytest

from auv_mission_planner.backends.auv_adapter import (Cmd, camera_to_detection, plan_heading,
                                                           plan_navigation)


def test_navigation_maps_to_auv_relative_moves():
    assert plan_navigation(0, 0, 3, 0) == [Cmd('forward', 3)]
    assert plan_navigation(5, 2, 4, -1) == [Cmd('backward', 1), Cmd('right', 3)]
    assert plan_navigation(1, 1, 1.01, 3) == [Cmd('left', 2)]
    assert plan_navigation(1, 1, 1, 1) == []


def test_heading_turns_short_way_with_absolute_target():
    assert plan_heading(10, 50) == Cmd('yaw_left', 50)
    assert plan_heading(10, 330) == Cmd('yaw_right', 330)
    assert plan_heading(350, 20) == Cmd('yaw_left', 20)


def test_camera_detection_geometry():
    # buoy dead ahead, 0.4 m wide covering 40 px with f = 320 px -> 3.2 m
    hfov = math.degrees(2 * math.atan(320 / 320))   # 90 deg -> f = 320 px for a 640 px frame
    d = camera_to_detection('red_buoy', 320, 240, 320, 240, 40, hfov, 0.9, (1.0, 2.0, 1.5, 0.0))
    assert d['class'] == 'red_buoy' and d['range_m'] == pytest.approx(3.2)
    assert d['x'] == pytest.approx(4.2) and d['y'] == pytest.approx(2.0) and d['depth'] == pytest.approx(1.5)
    # object left of centre -> positive bearing -> +y when facing +x
    d = camera_to_detection('gate', 100, 240, 320, 240, 200, hfov, 0.8, (0, 0, 1.0, 0.0))
    assert d['bearing_rad'] > 0 and d['y'] > 0


def test_unknown_classes_ignored():
    assert camera_to_detection('torpedo_target', 1, 1, 320, 240, 10, 80, 0.9, (0, 0, 0, 0)) is None
    assert camera_to_detection('Gate', 320, 240, 320, 240, 100, 80, 0.9, (0, 0, 0, 0))['class'] == 'gate'
