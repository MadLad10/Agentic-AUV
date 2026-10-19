"""MCP (Model Context Protocol) server: the vehicle's high-level tools for any MCP client.

Any MCP-capable agent (an IDE assistant, another LangGraph app, a desktop chat client) can drive
the simulated AUV, but only through the same guard rails as the v2 graph:
  1. the tool must be one of v1's allowed high-level tools (there is NO thruster / setpoint tool),
  2. every motion call is wrapped in a one-step plan and checked by the deterministic safety
     validator (depth limits, geofence, timeouts, allowed tools); any rejection refuses the call,
  3. the runtime precheck (battery, mission time) runs before the call,
  4. a health-monitor veto blocks every motion except surface / abort,
  5. resolved object targets are re-checked against the limits inside the tool registry.

  python3 -m auv_agents.mcp_server                       # stdio transport
  python3 -m auv_agents.mcp_server --http --port 8071    # streamable HTTP on 127.0.0.1
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from auv_mission_planner.plan import MissionPlan
from auv_mission_planner.validator import SafetyValidator

MOTION = {'navigate_to', 'search_pattern', 'hold_depth', 'surface', 'abort'}
ALWAYS_ALLOWED_UNDER_VETO = {'surface', 'abort'}


class GuardedTools:
    """Validator-gated access to the tool registry (shared by the MCP server and its tests)."""

    def __init__(self, backend, registry, cfg: dict):
        self.be = backend
        self.reg = registry
        self.cfg = cfg
        self.validator = SafetyValidator(cfg['limits'], repair=True)
        self.t0 = backend.now()
        self.audit: list[dict] = []

    def call(self, tool: str, params: Optional[dict] = None, timeout_s: float = 60.0) -> dict:
        params = params or {}
        entry = {'tool': tool, 'params': params, 'sim_t': round(self.be.now(), 2)}
        if tool in MOTION:
            if self.be._veto and tool not in ALWAYS_ALLOWED_UNDER_VETO:
                return self._done(entry, {'ok': False, 'refused': True,
                                          'reason': f'health veto active: {self.be._veto}; only surface/abort allowed'})
            try:
                plan = MissionPlan.model_validate({'steps': [{'id': 1, 'tool': tool, 'params': params,
                                                              'timeout_s': timeout_s, 'on_failure': 'abort'}]})
            except Exception as e:
                return self._done(entry, {'ok': False, 'refused': True, 'reason': f'invalid call: {e}'})
            rep = self.validator.validate(plan)
            if rep.rejected:
                return self._done(entry, {'ok': False, 'refused': True,
                                          'reason': 'safety validator: ' + '; '.join(str(f) for f in rep.rejected)})
            step = rep.plan.steps[0]
            why = self.validator.precheck_step(step, self.be.battery_percent(), self.be.now() - self.t0, rep.plan)
            if why and tool not in ALWAYS_ALLOWED_UNDER_VETO:
                return self._done(entry, {'ok': False, 'refused': True, 'reason': f'runtime precheck: {why}'})
            params, timeout_s = step.params, step.timeout_s          # validator may clamp the timeout
        obs = self.reg.call(tool, params, timeout_s)
        return self._done(entry, json.loads(obs.model_dump_json()))

    def health(self) -> dict:
        rep = self.be.health.last
        return {'faults': rep.faults, 'veto': rep.veto, 'veto_reason': rep.veto_reason,
                'degraded_modes': rep.degraded_modes, 'battery_percent': self.be.battery_percent(),
                'navigation': self.be.ekf.summary() if self.be.nav_source == 'ekf' else None}

    def _done(self, entry, result):
        entry['result'] = {k: result.get(k) for k in ('ok', 'refused', 'reason', 'error', 'error_code')}
        self.audit.append(entry)
        return result


def build_server(guard: GuardedTools):
    from mcp.server import MCPServer
    srv = MCPServer('auv-tools', instructions=(
        'High-level tools of a simulated autonomous underwater vehicle in a test pool (gate, red/green/'
        'yellow buoys). Depth is metres below the surface; x forward, y left. Every motion call is '
        'checked by a deterministic safety validator and may be refused. There is no low-level control.'))

    @srv.tool(description='Vehicle position (x, y, depth) and heading from the navigation filter.')
    def get_pose() -> dict:
        return guard.call('get_pose')

    @srv.tool(description='Battery percentage.')
    def get_battery() -> dict:
        return guard.call('get_battery')

    @srv.tool(description='Objects seen by the camera now plus remembered object positions.')
    def get_detections() -> dict:
        return guard.call('get_detections')

    @srv.tool(description='Health & safety report: active faults, veto, degraded modes, navigation quality.')
    def get_health() -> dict:
        return guard.health()

    @srv.tool(description='Move to an object (target: gate|red_buoy|green_buoy|yellow_buoy; the gate is passed '
                          'through) or to a world point (x, y, depth). Validated before execution.')
    def navigate_to(target: Optional[str] = None, x: Optional[float] = None, y: Optional[float] = None,
                    depth: Optional[float] = None, standoff_m: float = 0.0, timeout_s: float = 120.0) -> dict:
        p: dict[str, Any] = {k: v for k, v in dict(target=target, x=x, y=y, depth=depth).items() if v is not None}
        if standoff_m:
            p['standoff_m'] = standoff_m
        return guard.call('navigate_to', p, timeout_s)

    @srv.tool(description='Search for an object class with a pattern: yaw_sweep | spiral | lawnmower.')
    def search_pattern(target: str, pattern: str = 'yaw_sweep', radius_m: float = 6.0,
                       depth: Optional[float] = None, timeout_s: float = 120.0) -> dict:
        p = {'target': target, 'pattern': pattern, 'radius_m': radius_m}
        if depth is not None:
            p['depth'] = depth
        return guard.call('search_pattern', p, timeout_s)

    @srv.tool(description='Hold a depth (metres) for duration_s seconds.')
    def hold_depth(depth: float, duration_s: float = 5.0, timeout_s: float = 60.0) -> dict:
        return guard.call('hold_depth', {'depth': depth, 'duration_s': duration_s}, timeout_s)

    @srv.tool(description='Ascend to the surface and stay there.')
    def surface(timeout_s: float = 90.0) -> dict:
        return guard.call('surface', {}, timeout_s)

    @srv.tool(description='Stop safely: cancel motion and surface.')
    def abort(reason: str = 'operator request') -> dict:
        return guard.call('abort', {'reason': reason}, 10.0)

    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arena', default='default')
    ap.add_argument('--http', action='store_true')
    ap.add_argument('--port', type=int, default=8071)
    a = ap.parse_args()
    from auv_onboard.factory import make_backend, make_registry
    be = make_backend(a.arena)                      # idle ticker on: the sim runs in real time between calls
    reg, cfg = make_registry(be)
    srv = build_server(GuardedTools(be, reg, cfg))
    try:
        if a.http:
            import anyio
            anyio.run(lambda: srv.run_streamable_http_async(host='127.0.0.1', port=a.port))
        else:
            srv.run()
    finally:
        be.shutdown()


if __name__ == '__main__':
    main()
