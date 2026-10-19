"""HARD RULE 3: the planner side can never reach thrusters or low-level control topics."""
import ast
import os
import re

import pytest

from auv_mission_planner.config import load_config

PKG_DIR = os.path.join(os.path.dirname(__file__), '..', 'auv_mission_planner')
FORBIDDEN = re.compile(r'/[\w/]*thruster|cmd_thrust|controller/setpoint|controller/mode|cmd_vel|'
                       r'/simulator/control|arduino_control|rc_channels|RC_CHANNELS_OVERRIDE', re.I)


def _sources():
    for root, _, files in os.walk(PKG_DIR):
        for f in files:
            if f.endswith('.py') and 'auv_adapter' not in f:   # adapter is backend-side (vehicle)
                yield os.path.join(root, f)


def _code_strings(path):
    """String literals and identifiers used in code (docstrings excluded)."""
    tree = ast.parse(open(path).read())
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            docs.add(id(node.body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            yield node.lineno, node.value
        elif isinstance(node, ast.Name):
            yield node.lineno, node.id
        elif isinstance(node, ast.Attribute):
            yield node.lineno, node.attr


def test_no_planner_module_mentions_low_level_topics():
    hits = []
    for path in _sources():
        if os.path.basename(path).startswith('test_'):
            continue
        for n, text in _code_strings(path):
            if FORBIDDEN.search(text):
                hits.append(f'{os.path.basename(path)}:{n}: {text}')
    assert not hits, 'low-level control referenced from planner code:\n' + '\n'.join(hits)


def test_tools_layer_creates_no_publishers():
    for path in _sources():
        if os.sep + 'tools' + os.sep in path:
            assert 'create_publisher' not in open(path).read(), path


def test_config_interfaces_are_high_level_only():
    cfg = load_config()
    for backend, iface in cfg['interfaces'].items():
        for key, name in iface.items():
            assert not FORBIDDEN.search(name), f'{backend}.{key} -> {name}'
    assert set(cfg['interfaces']['sim']) == set(cfg['interfaces']['real'])


def test_ros_backend_node_has_no_publishers():
    rclpy = pytest.importorskip('rclpy')
    pytest.importorskip('auv_mission_interfaces')
    from auv_mission_planner.tools.ros_backend import RosBackend
    rclpy.init()
    try:
        b = RosBackend(load_config(), use_sim_time=False)
        pubs = b.node.get_publisher_names_and_types_by_node(b.node.get_name(), b.node.get_namespace())
        topics = {t for t, _ in pubs} - {'/rosout', '/parameter_events'}
        assert topics == set(), topics
        b.shutdown()
    finally:
        rclpy.try_shutdown()
