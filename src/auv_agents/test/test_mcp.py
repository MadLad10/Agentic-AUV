"""MCP server: protocol round-trip through the in-memory transport; guard rails hold for MCP callers."""
import asyncio
import json

import pytest

from auv_agents.mcp_server import GuardedTools, build_server
from auv_onboard.factory import make_backend, make_registry


@pytest.fixture
def guard():
    be = make_backend('default', seed=0, onboard_overrides={'backend': {'max_rtf': None, 'idle_rtf': None}})
    reg, cfg = make_registry(be)
    g = GuardedTools(be, reg, cfg)
    yield g
    be.shutdown()


def _call(server, name, args=None):
    from mcp.client import Client

    async def go():
        async with Client(server) as c:
            tools = await c.list_tools()
            res = await c.call_tool(name, args or {})
            return [t.name for t in tools.tools], res
    return asyncio.run(go())


def _payload(res):
    if getattr(res, 'structured_content', None):
        return res.structured_content
    return json.loads(res.content[0].text)


def test_tools_exposed_and_no_low_level_control(guard):
    names, res = _call(build_server(guard), 'get_pose')
    assert {'navigate_to', 'hold_depth', 'surface', 'abort', 'get_health'} <= set(names)
    assert not [n for n in names if 'thrust' in n or 'setpoint' in n or 'pwm' in n]
    assert _payload(res)['ok'] is True


def test_unsafe_call_refused_by_validator(guard):
    _, res = _call(build_server(guard), 'hold_depth', {'depth': 25.0, 'duration_s': 2})
    out = _payload(res)
    assert out['ok'] is False and out['refused'] and 'safety validator' in out['reason']
    assert guard.be.sim.referee.report()['max_depth_m'] < 0.3          # vehicle never moved
    _, res = _call(build_server(guard), 'navigate_to', {'x': 100.0, 'y': 0.0, 'depth': 1.0})
    assert _payload(res)['refused']


def test_safe_call_executes_and_veto_blocks_motion(guard):
    _, res = _call(build_server(guard), 'hold_depth', {'depth': 1.0, 'duration_s': 2})
    assert _payload(res)['ok'] is True
    assert guard.be.sim.referee.report()['max_depth_m'] > 0.8
    guard.be._veto = 'leak: test'
    _, res = _call(build_server(guard), 'hold_depth', {'depth': 1.5, 'duration_s': 2})
    out = _payload(res)
    assert out['refused'] and 'veto' in out['reason']
    assert [a['tool'] for a in guard.audit] == ['hold_depth', 'hold_depth']
