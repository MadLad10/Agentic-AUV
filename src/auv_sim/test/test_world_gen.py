import os
import subprocess
import xml.etree.ElementTree as ET

import pytest

from auv_sim.arena import load_arena, world_to_body, body_to_world, wrap
from auv_sim.world_gen import THRUSTERS, generate_world, HULL, HULL_MASS, PROP_MASS

ARENAS = os.path.join(os.path.dirname(__file__), '..', 'config', 'arenas')


@pytest.fixture(params=['default.yaml', 'buoy_hidden.yaml'])
def arena(request):
    return load_arena(os.path.join(ARENAS, request.param))


def test_world_contains_all_objects_and_thrusters(arena):
    root = ET.fromstring(generate_world(arena))
    models = {m.get('name') for m in root.iter('model')}
    assert 'auv' in models
    for o in arena['objects']:
        assert o['id'] in models
    thr = [p for p in root.iter('plugin') if p.get('name', '').endswith('Thruster')]
    assert len(thr) == len(THRUSTERS) == 6


def test_camera_only_when_requested(arena):
    assert 'front_camera' not in generate_world(arena, camera=False)
    assert 'front_camera' in generate_world(arena, camera=True)


def test_vehicle_slightly_positively_buoyant():
    displaced_kg = HULL[0] * HULL[1] * HULL[2] * 1000.0
    total = HULL_MASS + PROP_MASS * len(THRUSTERS)
    assert 0 < displaced_kg - total < 1.0     # floats up slowly when idle


def test_sdf_validates_with_ign(tmp_path, arena):
    if subprocess.run(['which', 'ign'], capture_output=True).returncode != 0:
        pytest.skip('ign not installed')
    f = tmp_path / 'w.sdf'
    f.write_text(generate_world(arena, camera=True))
    r = subprocess.run(['ign', 'sdf', '-k', str(f)], capture_output=True, text=True, timeout=60)
    assert 'Valid' in r.stdout, r.stdout + r.stderr


def test_frame_helpers_roundtrip():
    for yaw in (0.0, 0.7, -2.5, 3.1):
        bx, by = world_to_body(1.3, -0.4, yaw)
        wx, wy = body_to_world(bx, by, yaw)
        assert abs(wx - 1.3) < 1e-9 and abs(wy + 0.4) < 1e-9
    assert abs(wrap(3.5) - (3.5 - 2 * 3.141592653589793)) < 1e-9
