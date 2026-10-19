"""Config loading (YAML) and a minimal .env loader. No ROS imports."""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

PKG = 'auv_mission_planner'


def _candidates(name: str) -> list[str]:
    here = os.path.dirname(os.path.realpath(__file__))
    out = [os.path.join(here, '..', 'config', name)]           # source tree / symlink install
    try:
        from ament_index_python.packages import get_package_share_directory
        out.insert(0, os.path.join(get_package_share_directory(PKG), 'config', name))
    except Exception:
        pass
    return out


def config_path(name: str, override: str | None = None) -> str:
    if override:
        return override
    env = os.environ.get('AUV_' + name.split('.')[0].upper() + '_CONFIG')
    if env:
        return env
    for c in _candidates(name):
        if os.path.isfile(c):
            return os.path.realpath(c)
    raise FileNotFoundError(f'config {name} not found in {_candidates(name)}')


def load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load auv.yaml and resolve the active backend's interface names into cfg['iface']."""
    cfg = load_yaml(config_path('auv.yaml', path))
    cfg = copy.deepcopy(cfg)
    backend = cfg.get('backend', 'sim')
    if backend not in cfg.get('interfaces', {}):
        raise ValueError(f'backend {backend!r} has no interfaces section')
    cfg['iface'] = cfg['interfaces'][backend]
    return cfg


def load_llm_config(path: str | None = None) -> dict[str, Any]:
    return load_yaml(config_path('llm.yaml', path))


def find_workspace_root(start: str | None = None) -> str | None:
    d = os.path.abspath(start or os.environ.get('AUV_WS') or os.getcwd())
    while d != '/':
        if os.path.isdir(os.path.join(d, 'src', 'auv_mission_planner')) and os.path.isdir(os.path.join(d, 'scripts')):
            return d
        d = os.path.dirname(d)
    return None


def load_dotenv(path: str | None = None, override: bool = False) -> dict[str, str]:
    """Parse KEY=VALUE lines from .env (workspace root by default) into os.environ."""
    if path is None:
        root = find_workspace_root()
        path = os.path.join(root, '.env') if root else '.env'
    loaded = {}
    if not os.path.isfile(path):
        return loaded
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k, v = k.strip().removeprefix('export ').strip(), v.strip().strip('"').strip("'")
            if v and (override or k not in os.environ):
                os.environ[k] = v
                loaded[k] = v
    return loaded
