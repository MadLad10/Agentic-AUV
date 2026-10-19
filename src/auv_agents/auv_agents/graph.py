"""LangGraph multi-agent mission graph (v2).

    START -> sense -> supervisor -> validate -> approval -> execute_step -> monitor --+
                        ^   ^          |            |            ^                   |
                        |   |   reject/refuse       reject       | next step         |
                        |   +--------------------- replan <------+--- step failed ---+
                        |                                        |    (after inspect)
                        +---------- diagnose <--- new faults ----+--- veto / watchdog -> abort -> finish

Nodes are agents with one job each (see specialists.py). The supervisor is the only planner; the
deterministic safety validator checks every plan; the operator approves through a LangGraph
interrupt (human-in-the-loop); every tool call goes through v1's typed tool registry and the
validator's runtime precheck; the health agent can veto at any time. State is plain JSON so it can
be checkpointed (SQLite); the live vehicle/sim objects stay in the runtime, outside the state.
"""
from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from auv_mission_planner.executor import Executor
from auv_mission_planner.plan import MissionPlan
from auv_mission_planner.planner import Planner, PlanResult
from auv_mission_planner.validator import SafetyValidator
from auv_agents import telemetry
from auv_agents.specialists import (DiagnosticsAgent, HealthAgent, NavigationAgent, PerceptionAgent,
                                    check_abort_conditions, telemetry_snapshot)

RETRY_CODES = {'target_not_detected', 'not_found'}


class MissionState(TypedDict, total=False):
    prompt: str
    phase: str
    plan: Optional[dict]
    plan_kind: str
    step_idx: int
    plan_index: int
    replans: int
    failed: Optional[dict]                  # {'step', 'result'} of the last failed step
    done: list                              # briefs of successful steps (for replanning)
    records: Annotated[list, operator.add]  # every executed step
    events: Annotated[list, operator.add]   # agent messages, streamed to the operator / trace
    health: dict
    nav: dict
    perception: dict
    diagnosis: Optional[dict]
    diagnosed: list                         # fault types already diagnosed
    inspections: int
    mission_t0: Optional[float]
    approval: Optional[dict]
    outcome: str
    reason: str
    stats: dict


@dataclass
class Runtime:
    """Live objects shared by the nodes (not checkpointed)."""
    backend: Any
    registry: Any
    cfg: dict                    # auv.yaml (limits, executor, tools)
    agents_cfg: dict
    planner: Planner
    diagnostics: DiagnosticsAgent
    vlm: Any = None
    auto_approve: bool = True
    log: Any = None
    last_plan_result: Optional[PlanResult] = None
    llm_calls: list = field(default_factory=list)

    def __post_init__(self):
        self.validator = SafetyValidator(self.cfg['limits'], repair=self.cfg.get('validator', {}).get('repair', True))
        self.executor = Executor(self.registry, self.validator, self.cfg)
        self.perception = PerceptionAgent(self.agents_cfg['perception'], self.backend, self.registry, self.vlm)
        self.navigation = NavigationAgent(self.agents_cfg['navigation'], self.backend, {})
        self.health = HealthAgent(self.backend)
        self.log = self.log or (lambda m: None)


def _plain(o):
    if hasattr(o, 'item'):                 # numpy scalar
        return o.item()
    if hasattr(o, 'tolist'):
        return o.tolist()
    raise TypeError(f'not serialisable: {type(o).__name__}')


def jsonable(fn):
    """Node wrapper: state updates must be plain JSON (checkpointing), so convert numpy values."""
    import functools
    import json

    @functools.wraps(fn)
    def wrapped(state):
        out = fn(state)
        return json.loads(json.dumps(out, default=_plain)) if out is not None else out
    return wrapped


def _ev(agent: str, msg: str, **data) -> dict:
    return {'agent': agent, 'msg': msg, 'wall': round(time.time(), 2), **data}


def build_graph(rt: Runtime, checkpointer=None):
    g = rt.agents_cfg['graph']
    max_replans = g['max_replans']

    # ------------------------------------------------------------------ helpers
    def context(state) -> dict:
        b = rt.backend
        pose = b.pose()
        batt = b.battery_percent()
        ctx = {'pose': pose.as_dict() if pose else None,
               'battery_percent': round(batt, 1) if batt is not None else None,
               'visible_objects': sorted({d.cls for d in b.detections()}),
               'remembered_objects': rt.registry.world.summary(b.now())}
        h, n, p = state.get('health') or {}, state.get('nav') or {}, state.get('perception') or {}
        if h.get('faults'):
            ctx['active_faults'] = [f['type'] for f in h['faults']]
            ctx['degraded_modes'] = h.get('degraded_modes', [])
        if n.get('mode') and n['mode'] != 'nominal':
            ctx['navigation'] = n
        if p.get('unclassified_sonar_contacts'):
            ctx['sonar_contacts'] = p['unclassified_sonar_contacts'][:4]
        if p.get('vlm'):
            ctx['camera_description'] = p['vlm'].get('answer')
        if state.get('diagnosis'):
            d = state['diagnosis']
            ctx['diagnosis'] = {k: d.get(k) for k in ('fault_types', 'explanation', 'degraded_mode', 'action')}
        return ctx

    def stats(state, **inc) -> dict:
        s = dict(state.get('stats') or {})
        for k, v in inc.items():
            s[k] = s.get(k, 0) + v
        return s

    def _record_llm(result: PlanResult):
        rt.llm_calls.extend(result.calls)

    # ------------------------------------------------------------------ nodes
    def sense(state):
        h = rt.health.update(None, None)
        p = rt.perception.update(camera_ok=not {'camera_blackout', 'turbidity'} & {f['type'] for f in h['faults']})
        n = rt.navigation.update(h)
        return {'health': h, 'perception': p, 'nav': n, 'phase': 'planning', 'step_idx': 0, 'replans': 0,
                'plan_index': 0, 'done': [], 'diagnosed': [], 'inspections': 0, 'mission_t0': None,
                'stats': {}, 'events': [_ev('perception', f'{len(p["tracks"])} tracks, '
                                                          f'{len(p["unclassified_sonar_contacts"])} unclassified sonar contacts'),
                                        _ev('navigation', f'nav mode {n["mode"]}')]}

    def supervisor(state):
        ctx = context(state)
        if state.get('failed') is None:
            res = rt.planner.plan(state['prompt'], ctx)
            kind = 'plan'
        else:
            res = rt.planner.replan(state['prompt'], state.get('done', []), state['failed'], ctx,
                                    state.get('replans', 0), max_replans)
            kind = 'replan'
        rt.last_plan_result = res
        _record_llm(res)
        upd = {'stats': stats(state, llm_calls=len(res.calls),
                              format_errors=sum(1 for c in res.calls if c.get('parse_error')))}
        if res.status == 'refuse':
            return {**upd, 'phase': 'refused', 'plan': None, 'reason': res.reason,
                    'events': [_ev('supervisor', f'refused: {res.reason}')]}
        if res.status != 'plan':
            return {**upd, 'phase': 'error', 'plan': None, 'reason': res.reason,
                    'events': [_ev('supervisor', f'planning error: {res.reason}')]}
        return {**upd, 'phase': 'validating', 'plan': res.plan.model_dump(), 'plan_kind': kind,
                'events': [_ev('supervisor', f'{kind}: ' + ' -> '.join(s.tool for s in res.plan.steps))]}

    def validate(state):
        res = rt.last_plan_result
        plan = MissionPlan.model_validate(state['plan'])
        events, st = [], state.get('stats') or {}
        for attempt in range(g['max_validation_retries'] + 1):
            rep = rt.validator.validate(plan, prompt=state['prompt'])
            st = {**st, 'plans': st.get('plans', 0) + 1, 'repairs': st.get('repairs', 0) + len(rep.repaired)}
            if rep.ok:
                events.append(_ev('safety_validator', 'plan OK' + (f' ({len(rep.repaired)} repairs)' if rep.repaired else ''),
                                  repairs=[str(f) for f in rep.repaired]))
                return {'phase': 'approval', 'plan': rep.plan.model_dump(), 'stats': st, 'events': events,
                        'plan_index': state.get('plan_index', 0) + 1}
            st['unsafe_plans_caught'] = st.get('unsafe_plans_caught', 0) + 1
            events.append(_ev('safety_validator', 'REJECTED: ' + '; '.join(rep.reasons())))
            if attempt == g['max_validation_retries']:
                break
            res = rt.planner.repair(res, rep.reasons())
            _record_llm(res)
            st['llm_calls'] = st.get('llm_calls', 0) + len(res.calls)
            if res.status == 'refuse':
                return {'phase': 'refused', 'plan': None, 'reason': res.reason, 'stats': st,
                        'events': events + [_ev('supervisor', f'refused after validator feedback: {res.reason}')]}
            if res.status != 'plan':
                break
            plan = res.plan
        return {'phase': 'rejected', 'plan': None, 'reason': 'safety validator rejected every plan',
                'stats': st, 'events': events}

    def approval(state):
        plan = state['plan']
        if rt.auto_approve:
            return {'phase': 'executing', 'approval': {'decision': 'approve', 'by': 'auto'},
                    'step_idx': 0,
                    'events': [_ev('operator', 'plan auto-approved')]}
        answer = interrupt({'question': 'Approve this plan?', 'plan': plan, 'prompt': state['prompt']})
        decision = answer.get('decision') if isinstance(answer, dict) else str(answer)
        if decision != 'approve':
            return {'phase': 'operator_rejected', 'approval': {'decision': decision, 'by': 'operator'},
                    'reason': f'operator rejected the plan: {answer}',
                    'events': [_ev('operator', f'plan rejected: {answer}')]}
        return {'phase': 'executing', 'approval': {'decision': 'approve', 'by': 'operator'}, 'step_idx': 0,
                'events': [_ev('operator', 'plan approved')]}

    def execute_step(state):
        plan = MissionPlan.model_validate(state['plan'])
        i = state.get('step_idx', 0)
        step = plan.steps[i]
        be = rt.backend
        t0 = state['mission_t0'] if state.get('mission_t0') is not None else be.now()   # clock starts here
        why = rt.validator.precheck_step(step, be.battery_percent(), be.now() - t0, plan)
        if why:
            return {'mission_t0': t0, 'phase': 'abort', 'reason': f'precheck before step {step.id}: {why}',
                    'events': [_ev('safety_validator', f'precheck failed: {why}')]}
        # in-sim watchdog: the plan's abort conditions are checked every control tick
        trip = {}

        def hook(b):
            if not trip:
                r = check_abort_conditions(state['plan'], b.battery_percent(), b.now() - t0)
                if r:
                    trip['reason'] = r
                    b.cancel_all()
        be.tick_hooks.append(hook)
        try:
            obs = rt.registry.call(step.tool, step.params, step.timeout_s)
        finally:
            be.tick_hooks.remove(hook)
        ok, detail = rt.executor.evaluate(step, obs)
        rec = {'plan_index': state.get('plan_index', 0), 'step': step.model_dump(), 'observation': obs.model_dump(),
               'success': ok, 'detail': detail, 'sim_end': round(be.now(), 2)}
        upd = {'mission_t0': t0, 'records': [rec], 'stats': stats(state, steps_executed=1),
               'events': [_ev('executor', f'step {step.id} {step.tool} {"ok" if ok else "FAILED"}: {detail}'
                              + ('' if ok else f' [{obs.error_code}] {obs.error or ""}'))]}
        if trip:
            return {**upd, 'phase': 'abort', 'reason': trip['reason']}
        if ok or step.on_failure == 'continue':
            brief = {'id': step.id, 'tool': step.tool, 'params': step.params, 'ok': True}
            return {**upd, 'phase': 'step_ok', 'done': state.get('done', []) + [brief], 'failed': None,
                    'step_idx': i + 1, 'inspections': 0}
        failed = {'step': step.model_dump(), 'result': {'ok': False, 'detail': detail, 'error': obs.error,
                                                        'error_code': obs.error_code}}
        if step.on_failure == 'abort' or step.tool == 'abort':
            return {**upd, 'phase': 'abort', 'failed': failed, 'reason': f'step {step.id} ({step.tool}) failed: {detail}'}
        return {**upd, 'phase': 'step_failed', 'failed': failed}

    def monitor(state):
        h = rt.health.update(state.get('plan'), state.get('mission_t0'))
        faults = {f['type'] for f in h['faults']}
        p = rt.perception.update(camera_ok=not faults & {'camera_blackout', 'turbidity'})
        n = rt.navigation.update(h)
        ev = []
        new = sorted(faults - set(state.get('diagnosed', [])))
        if new:
            ev.append(_ev('health', f'new faults: {new}; degraded modes {h["degraded_modes"]}'))
        if n['mode'] != (state.get('nav') or {}).get('mode'):
            ev.append(_ev('navigation', f'mode {n["mode"]} (pos std {n.get("pos_std_m")} m, speed x{n.get("speed_scale", 1)})'))
        failed = state.get('failed') or {}
        code = (failed.get('result') or {}).get('error_code')
        if state.get('phase') == 'step_failed' and code in RETRY_CODES and rt.vlm is not None:
            rec = rt.perception.ask_vlm(f'The robot is looking for: {failed["step"]["params"].get("target")}. '
                                        'Describe what is visible and whether that object is in view.')
            if rec:
                p = {**p, 'vlm': rec}
                ev.append(_ev('perception', f'VLM ({rec["latency_s"]}s): {rec["answer"] or rec["error"]}'))
        out = {'health': h, 'perception': p, 'nav': n, 'events': ev}
        if h.get('veto') and state.get('phase') != 'abort':
            ev.append(_ev('health', f'VETO: {h["veto_reason"]}'))
            out.update(phase='abort', reason=f'health veto: {h["veto_reason"]}')
        return out

    def diagnose(state):
        h, n, p = state['health'], state['nav'], state['perception']
        d = rt.diagnostics.diagnose(h, n, p, telemetry_snapshot(rt.backend))
        types = sorted(set(state.get('diagnosed', [])) | {f['type'] for f in h['faults']})
        return {'diagnosis': d, 'diagnosed': types, 'stats': stats(state, diagnoses=1),
                'events': [_ev('diagnostics', f'[{d.get("source")}] {d["fault_types"]} -> {d["degraded_mode"]}, '
                                              f'action {d["action"]}: {d["explanation"][:160]}')]}

    def inspect(state):
        """Sonar-cued inspection: drive up to unclassified sonar contacts so the camera can classify them."""
        target = state['failed']['step']['params'].get('target')
        pts = rt.perception.inspection_points(state['perception'], rt.agents_cfg['perception']['inspect_standoff_m'])
        n = state.get('inspections', 0)
        ev = []
        depth = rt.backend.pose().depth
        for x, y, _yaw in pts[:1]:
            obs = rt.registry.call('navigate_to', {'x': x, 'y': y, 'depth': max(depth, 1.0)}, 60.0)
            ev.append(_ev('perception', f'inspecting sonar contact near ({x}, {y}) for {target}: '
                                        f'{"reached" if obs.ok else obs.error}'))
        seen = rt.registry.world.lookup(target, rt.backend.now()) is not None
        upd = {'inspections': n + 1, 'stats': stats(state, inspections=1), 'events': ev}
        if seen:
            ev.append(_ev('perception', f'{target} now detected: retrying the step'))
            return {**upd, 'phase': 'executing', 'failed': None}
        return upd

    def abort(state):
        reason = state.get('reason') or 'abort'
        be = rt.backend
        o = rt.registry.call('abort', {'reason': reason}, 10.0)
        t_end = time.time() + g['abort_surface_timeout_s']
        surfaced = False
        while time.time() < t_end:
            alive = any(t.is_alive() for t in be._threads)
            if not alive and be.pose().depth < 0.3:
                surfaced = True
                break
            if not alive:          # nothing surfacing in the background: do it here
                surfaced = be.surface(60.0).ok
                break
            time.sleep(0.05)
        return {'phase': 'aborted', 'outcome': 'aborted', 'reason': reason,
                'events': [_ev('supervisor', f'safe abort ({reason}); abort call ok={o.ok}; surfaced={surfaced}')],
                'stats': stats(state, aborts=1)}

    def finish(state):
        phase = state.get('phase')
        outcome = {'step_ok': 'succeeded', 'refused': 'refused', 'rejected': 'rejected', 'error': 'error',
                   'operator_rejected': 'operator_rejected', 'aborted': 'aborted'}.get(phase, phase)
        return {'outcome': outcome, 'events': [_ev('supervisor', f'mission finished: {outcome}')]}

    # ------------------------------------------------------------------ routing
    def after_supervisor(state):
        return 'validate' if state['phase'] == 'validating' else 'finish'

    def after_validate(state):
        return 'approval' if state['phase'] == 'approval' else 'finish'

    def after_approval(state):
        return 'execute_step' if state['phase'] == 'executing' else 'finish'

    def after_monitor(state):
        h = state['health']
        if state.get('phase') == 'abort' or h.get('veto'):
            return 'abort'
        new = {f['type'] for f in h['faults']} - set(state.get('diagnosed', []))
        if new:
            return 'diagnose'
        return route_progress(state)

    def route_progress(state):
        phase = state.get('phase')
        plan = state.get('plan') or {'steps': []}
        if phase == 'step_ok':
            last = state['records'][-1]['step'] if state.get('records') else {}
            if last.get('tool') == 'abort':
                return 'finish'
            return 'execute_step' if state['step_idx'] < len(plan['steps']) else 'finish'
        if phase == 'step_failed':
            code = (state['failed'].get('result') or {}).get('error_code')
            if code in RETRY_CODES and state.get('inspections', 0) < g['max_inspections_per_step'] and \
                    rt.perception.inspection_points(state['perception'], 1.5):
                return 'inspect'
            return 'replan' if state.get('replans', 0) < max_replans else 'abort'
        if phase == 'executing':          # inspection found the target: retry the same step
            return 'execute_step'
        return 'abort'

    def after_diagnose(state):
        d = state['diagnosis']
        if d['action'] == 'abort':
            return 'abort'
        if d['action'] == 'replan' and state.get('phase') == 'step_ok':
            plan = state.get('plan') or {'steps': []}
            if state['step_idx'] < len(plan['steps']) and state.get('replans', 0) < max_replans:
                return 'replan'
        return route_progress(state)

    def replan(state):
        """Bookkeeping before asking the supervisor again (failed step or fault-driven revision)."""
        failed = state.get('failed')
        if failed is None:                 # fault-driven revision of the remaining plan
            nxt = state['plan']['steps'][state['step_idx']]
            d = state.get('diagnosis') or {}
            failed = {'step': nxt, 'result': {'ok': False, 'detail': f'not executed: faults {d.get("fault_types")} '
                                                                     f'detected; degraded mode {d.get("degraded_mode")}; '
                                                                     f'{d.get("explanation", "")[:200]}'}}
        return {'replans': state.get('replans', 0) + 1, 'failed': failed, 'phase': 'replanning',
                'events': [_ev('supervisor', f'replanning ({state.get("replans", 0) + 1}/{max_replans}): '
                                             f'{failed["result"].get("detail")}')]}

    def after_inspect(state):
        if state.get('phase') == 'executing':
            return 'execute_step'
        return 'replan' if state.get('replans', 0) < max_replans else 'abort'

    sg = StateGraph(MissionState)
    for name, fn in [('sense', sense), ('supervisor', supervisor), ('validate', validate), ('approval', approval),
                     ('execute_step', execute_step), ('monitor', monitor), ('diagnose', diagnose),
                     ('inspect', inspect), ('replan', replan), ('abort', abort), ('finish', finish)]:
        sg.add_node(name, jsonable(telemetry.node(name, fn)))
    sg.add_edge(START, 'sense')
    sg.add_edge('sense', 'supervisor')
    sg.add_conditional_edges('supervisor', after_supervisor, ['validate', 'finish'])
    sg.add_conditional_edges('validate', after_validate, ['approval', 'finish'])
    sg.add_conditional_edges('approval', after_approval, ['execute_step', 'finish'])
    sg.add_edge('execute_step', 'monitor')
    sg.add_conditional_edges('monitor', after_monitor,
                             ['abort', 'diagnose', 'execute_step', 'inspect', 'replan', 'finish'])
    sg.add_conditional_edges('diagnose', after_diagnose, ['abort', 'replan', 'execute_step', 'inspect', 'finish'])
    sg.add_conditional_edges('inspect', after_inspect, ['execute_step', 'replan', 'abort'])
    sg.add_edge('replan', 'supervisor')
    sg.add_edge('abort', 'finish')
    sg.add_edge('finish', END)
    return sg.compile(checkpointer=checkpointer)
