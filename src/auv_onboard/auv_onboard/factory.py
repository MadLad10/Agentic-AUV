"""Build a MuJoCo sim + onboard backend (+ v1 tool registry) from configs."""
from __future__ import annotations

import os

import yaml

from auv_marine_sim.sim import MarineSim
from auv_marine_sim.world import deep_update


def _share(pkg):
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory(pkg)
    except Exception:
        return None


def onboard_config_path() -> str:
    for base in (_share('auv_onboard'), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')):
        if base and os.path.isfile(os.path.join(base, 'config', 'onboard.yaml')):
            return os.path.join(base, 'config', 'onboard.yaml')
    raise FileNotFoundError('onboard.yaml not found')


def load_onboard_config(overrides: dict | None = None) -> dict:
    with open(onboard_config_path()) as f:
        cfg = yaml.safe_load(f)
    return deep_update(cfg, overrides or {})


def make_backend(arena='default', faults=None, seed=0, onboard_overrides=None, sim_overrides=None,
                 start_ticker=True):
    from auv_onboard.backend import MarineSimBackend
    sim = MarineSim(arena, faults=faults, seed=seed, overrides=sim_overrides)
    return MarineSimBackend(sim, load_onboard_config(onboard_overrides), seed=seed, start_ticker=start_ticker)


def make_registry(backend, planner_cfg=None):
    from auv_mission_planner.config import load_config
    from auv_mission_planner.tools.registry import ToolRegistry
    cfg = planner_cfg or load_config()
    return ToolRegistry(backend, cfg), cfg
