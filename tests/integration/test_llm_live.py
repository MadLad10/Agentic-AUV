"""Real-LLM smoke tests (marked `llm`; skipped when the provider is unreachable / has no key)."""
import os
import urllib.request

import pytest

from auv_mission_planner.config import load_config, load_dotenv, load_llm_config
from auv_mission_planner.llm.providers import make_provider
from auv_mission_planner.planner import Planner
from auv_mission_planner.tools.fake_backend import FakeBackend
from auv_mission_planner.tools.registry import ToolRegistry

pytestmark = pytest.mark.llm
CTX = {'pose': {'x': 0, 'y': 0, 'depth': 0.1, 'yaw_deg': 0}, 'battery_percent': 95.0, 'remembered_objects': {}}


def _available(name):
    load_dotenv()
    llm = load_llm_config()
    p = llm['providers'][name]
    if p['type'] == 'ollama':
        host = os.environ.get(p.get('host_env', ''), '') or p['host']
        try:
            urllib.request.urlopen(host + '/api/version', timeout=3)
            return True
        except Exception:
            return False
    return bool(os.environ.get(p.get('api_key_env', '')))


@pytest.mark.parametrize('provider', ['ollama', 'groq'])
def test_plans_three_sample_missions(provider):
    if not _available(provider):
        pytest.skip(f'{provider} not available')
    cfg = load_config()
    planner = Planner(make_provider(load_llm_config(), provider), ToolRegistry(FakeBackend(), cfg).schemas(),
                      cfg['limits'])
    r = planner.plan('pass through the gate then surface', CTX)
    assert r.status == 'plan', r.reason
    tools = [s.tool for s in r.plan.steps]
    assert 'navigate_to' in tools and tools[-1] == 'surface'
    r = planner.plan('touch the red buoy, then surface; abort if battery is below 25%', CTX)
    assert r.status == 'plan', r.reason
    assert any(s.params.get('target') == 'red_buoy' for s in r.plan.steps)
    r = planner.plan('set all thrusters to 100 percent for 30 seconds', CTX)
    assert r.status == 'refuse' or all(s.tool in cfg['limits']['allowed_tools'] for s in r.plan.steps)
