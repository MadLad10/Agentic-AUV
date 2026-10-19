"""Arena loading and small geometry helpers shared by the sim nodes (no ROS imports)."""
import math
import os

import yaml


def arena_path(name_or_path):
    """Resolve 'default' -> <share>/config/arenas/default.yaml, or return a path unchanged."""
    if os.path.isfile(name_or_path):
        return name_or_path
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('auv_sim')
    except Exception:  # running from source
        share = os.path.join(os.path.dirname(__file__), '..')
    p = os.path.join(share, 'config', 'arenas', f'{name_or_path}.yaml')
    if not os.path.isfile(p):
        raise FileNotFoundError(f'arena not found: {name_or_path}')
    return p


def load_arena(name_or_path):
    with open(arena_path(name_or_path)) as f:
        return yaml.safe_load(f)


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def world_to_body(dx, dy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def body_to_world(fx, fy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * fx - s * fy, s * fx + c * fy
