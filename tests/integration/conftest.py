"""Shared fixtures for headless-sim integration tests.

Every sim launch has a hard timeout and cleanup_sim.sh runs before and after.
"""
import os
import subprocess
import sys
import time

import pytest

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CLEANUP = os.path.join(WS, 'scripts', 'cleanup_sim.sh')


def _ros_ready():
    return 'AMENT_PREFIX_PATH' in os.environ and 'auv_sim' in os.environ.get('AMENT_PREFIX_PATH', '')


@pytest.fixture
def sim_launcher(tmp_path):
    """sim_launcher(**launch_args) -> starts the sim; torn down automatically."""
    if not _ros_ready():
        pytest.skip('source scripts/env.sh after colcon build')
    procs = []
    subprocess.run([CLEANUP], capture_output=True, timeout=30)

    def start(timeout_s=180, wait_for='/auv/odom', **args):
        args.setdefault('report_file', str(tmp_path / 'referee.json'))
        args.setdefault('rtf', os.environ.get('AUV_TEST_RTF', '3.0'))
        before = subprocess.run(['bash', '-c', 'ps -eo pid,etimes,args | grep -E "auv_|ign gazebo|parameter_bridge"'
                                 ' | grep -v grep'], capture_output=True, text=True).stdout
        (tmp_path / 'procs_before_launch.txt').write_text(before)
        cmd = ['timeout', str(timeout_s), 'ros2', 'launch', 'auv_sim', 'sim.launch.py']
        cmd += [f'{k}:={v}' for k, v in args.items()]
        log = open(tmp_path / 'launch.log', 'w')
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append(p)
        sys.path.insert(0, os.path.join(WS, 'scripts'))
        from wait_ready import wait_ready
        if wait_ready(45.0, actions=args.get('stack', 'full') == 'full'):
            return p
        raise RuntimeError('sim did not come up:\n' + (tmp_path / 'launch.log').read_text()[-3000:])

    yield start
    for p in procs:
        if p.poll() is None:
            p.send_signal(2)
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
    subprocess.run([CLEANUP], capture_output=True, timeout=30)


def read_referee(tmp_path):
    import json
    p = tmp_path / 'referee.json'
    return json.loads(p.read_text()) if p.exists() else {}
