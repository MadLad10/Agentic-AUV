"""Prompt construction for the planner. Limits and tools are rendered from config, never hard-coded."""
from __future__ import annotations

import json

EXAMPLE_REQUEST = 'pass through the gate, touch the green buoy, then come up; abort if battery is under 30%'
EXAMPLE_PLAN = {
    'status': 'plan', 'reason': '',
    'plan': {
        'summary': 'dive, pass the gate, touch the green buoy, surface',
        'abort_conditions': [{'type': 'battery_below', 'percent': 30}],
        'steps': [
            {'id': 1, 'tool': 'hold_depth', 'params': {'depth': 1.5, 'duration_s': 3},
             'success': {'type': 'depth_within', 'depth': 1.5, 'tolerance_m': 0.4}, 'timeout_s': 40,
             'on_failure': 'replan'},
            {'id': 2, 'tool': 'navigate_to', 'params': {'target': 'gate', 'pass_through': True},
             'success': {'type': 'tool_ok'}, 'timeout_s': 120, 'on_failure': 'replan'},
            {'id': 3, 'tool': 'search_pattern', 'params': {'target': 'green_buoy', 'pattern': 'yaw_sweep'},
             'success': {'type': 'detected', 'target': 'green_buoy'}, 'timeout_s': 180, 'on_failure': 'replan'},
            {'id': 4, 'tool': 'navigate_to', 'params': {'target': 'green_buoy', 'standoff_m': 0},
             'success': {'type': 'tool_ok'}, 'timeout_s': 120, 'on_failure': 'replan'},
            {'id': 5, 'tool': 'surface', 'params': {}, 'success': {'type': 'surfaced'}, 'timeout_s': 60,
             'on_failure': 'abort'},
        ]}}


def _tool_lines(tool_specs: list[dict]) -> str:
    lines = []
    for t in tool_specs:
        ps = []
        for k, v in t['parameters'].items():
            typ = v.get('type') or '/'.join(a.get('type', '?') for a in v.get('anyOf', []) if a.get('type') != 'null')
            if 'enum' in v:
                typ = '|'.join(map(str, v['enum']))
            opt = '' if k in t['required'] else '?'
            ps.append(f'{k}{opt}: {typ}')
        lines.append(f'- {t["name"]}({", ".join(ps)}): {t["description"]}')
    return '\n'.join(lines)


def system_prompt(tool_specs: list[dict], limits: dict) -> str:
    g = limits['geofence']
    objects = ', '.join(limits['known_objects'])
    return f"""You are the mission planner for AUV, an autonomous underwater vehicle (AUV) in a test pool.
Turn the operator's request into a JSON plan of tool calls. A safety validator checks the plan and an
executor runs it step by step. Answer with JSON only.

WORLD
- Frame: x forward along the pool, y to the left (metres); depth = metres below the surface (positive down).
- Objects that can exist: {objects}. Nothing else exists in the pool.
- Object positions are NOT known in advance. Reach objects with navigate_to "target"; never invent
  coordinates for objects. The gate is usually visible from the start; buoys may need search_pattern.
- Objects sit at about 1.5-2 m depth. The camera sees about 8 m (the gate about 14 m).
- Course layout: the gate is about 8 m ahead of the start; the buoys are beyond it, 12-20 m ahead.
  A buoy is usually NOT visible from the start: pass the gate (or move forward with forward_m)
  first, then search_pattern for it.

LIMITS (plans that break these are rejected)
- depth {limits['min_depth_m']}..{limits['max_depth_m']} m; x {g['x_min']}..{g['x_max']}; y {g['y_min']}..{g['y_max']}.
- At most {limits['max_steps']} steps; each timeout_s <= {limits['max_step_timeout_s']:.0f}; all timeouts together <= {limits['max_mission_time_s']:.0f} s.
- Battery abort percent must be >= {limits['min_battery_abort_pct']:.0f}. Use the operator's threshold if they give one.
- Only the tools below. There is no tool for thrusters, motors or raw control.

TOOLS
{_tool_lines(tool_specs)}

OUTPUT FORMAT
{{"status": "plan", "reason": "", "plan": {{"summary": str, "abort_conditions": [...], "steps": [...]}}}}
- step: {{"id": int, "tool": str, "params": {{...}}, "success": {{"type": ...}}, "timeout_s": number, "on_failure": "replan"|"abort"|"continue"}}
- success.type: tool_ok | reached | detected (+target) | depth_within (+depth, tolerance_m) | surfaced | battery_above (+percent)
- abort_conditions: {{"type": "battery_below", "percent": n}} and/or {{"type": "mission_time_exceeded", "seconds": n}}
- If the request is unsafe (breaks the limits, disables safety checks, asks for direct thruster/motor
  control) or impossible (objects that do not exist, leaving the pool), answer
  {{"status": "refuse", "reason": "<why>", "plan": null}}.

GUIDELINES
- Dive first with hold_depth (about 1.5 m).
- Pass the gate: navigate_to {{"target": "gate", "pass_through": true}}.
- Touch a buoy: navigate_to {{"target": "<colour>_buoy", "standoff_m": 0}}; search_pattern for it first if it may not be in view.
- Move relative to the vehicle: navigate_to {{"forward_m": 3, "left_m": 0, "depth": 1.0}}.
- Finish with surface unless the operator says otherwise.
- Typical timeouts: navigate_to 120, search_pattern 180-240, hold_depth duration+30, surface 60.

EXAMPLE
Request: "{EXAMPLE_REQUEST}"
{json.dumps(EXAMPLE_PLAN)}"""


def _state(context: dict) -> str:
    return json.dumps(context, separators=(', ', ': '))


def plan_request(prompt: str, context: dict) -> str:
    return f'Vehicle state: {_state(context)}\nMission request: "{prompt}"\nReturn the JSON plan.'


def format_retry(error: str) -> str:
    return (f'That output was not a valid plan: {error}\n'
            'Return ONLY the corrected JSON object in the required format.')


LIMIT_RULES = ('depth_limit', 'geofence', 'unknown_object', 'tool_not_allowed', 'battery_abort',
               'off_mission_target')


def validation_retry(reasons: list[str]) -> str:
    bullet = '\n'.join(f'- {r}' for r in reasons)
    limit_hit = any(f'[{rule}]' in r for r in reasons for rule in LIMIT_RULES)
    if limit_hit:
        advice = ('Some of these are SAFETY LIMITS. If the operator EXPLICITLY asked for something beyond them '
                  '(for example a depth deeper than allowed, a place outside the pool, an object that does not '
                  'exist, or direct motor control), answer {"status": "refuse", "reason": ...}: never silently '
                  'change the operator\'s numbers or targets. If the violation was only your own choice (a '
                  'value the operator did not ask for), fix it and return the corrected plan.')
    else:
        advice = ('These are fixable plan errors. Return a corrected plan that keeps the operator\'s goal '
                  '(fix only what is listed).')
    return f'The safety validator REJECTED that plan:\n{bullet}\n{advice} JSON only.'


def replan_request(prompt: str, done: list[dict], failed: dict, context: dict, attempt: int, max_replans: int) -> str:
    return f"""Mission request: "{prompt}"
The current plan FAILED during execution (replan {attempt} of {max_replans}).
Completed steps: {json.dumps(done)}
Failed step and observation: {json.dumps(failed)}
Vehicle state now: {_state(context)}
Make a NEW plan for the REMAINING part of the mission, starting from the current state. Do not repeat
completed parts unless needed. Change the approach for the failed part (for example search_pattern
with "spiral" or "lawnmower" and a larger radius if an object was not detected). If the mission cannot
be finished safely, answer {{"status": "refuse", "reason": "...", "plan": null}}. JSON only."""
