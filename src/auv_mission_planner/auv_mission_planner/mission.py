"""MissionRunner: prompt -> plan -> validate (-> repair) -> execute -> outcome + trace.

Shared by the CLI and the ROS 2 service node. Outcomes:
  succeeded  plan executed, all steps met their success conditions
  planned    dry run: a validated plan was produced (nothing executed)
  refused    the LLM refused (unsafe / impossible request); nothing executed
  rejected   the safety validator rejected every plan the LLM produced; nothing executed
  aborted    execution stopped and the vehicle was surfaced safely
  error      the LLM could not produce a parseable plan / provider unreachable; nothing executed
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from auv_mission_planner.executor import Executor, ExecResult, Watchdog
from auv_mission_planner.llm.base import LLMProvider
from auv_mission_planner.plan import MissionPlan
from auv_mission_planner.planner import Planner, PlanResult
from auv_mission_planner.tools.registry import ToolRegistry
from auv_mission_planner.tracing import MissionTrace
from auv_mission_planner.validator import SafetyValidator


class MissionRunner:
    def __init__(self, cfg: dict, llm_cfg: dict, provider: LLMProvider, registry: ToolRegistry,
                 trace_dir: Optional[str] = None,
                 on_status: Optional[Callable[[str, int, int, str], None]] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.tools = registry
        self.provider = provider
        pcfg = llm_cfg.get('planner', {})
        self.max_validation_retries = pcfg.get('max_validation_retries', 2)
        self.max_replans = cfg.get('executor', {}).get('max_replans', 0)
        self.validator = SafetyValidator(cfg['limits'], repair=cfg.get('validator', {}).get('repair', True))
        self.planner = Planner(provider, registry.schemas(), cfg['limits'], pcfg.get('max_format_retries', 2))
        self.trace_dir = trace_dir
        self.on_status = on_status or (lambda *a: None)
        self.log = log or (lambda msg: None)

    # ------------------------------------------------------------ helpers
    def context(self) -> dict:
        b = self.tools.backend
        pose = b.pose()
        batt = b.battery_percent()
        return {'pose': pose.as_dict() if pose else None,
                'battery_percent': round(batt, 1) if batt is not None else None,
                'visible_objects': sorted({d.cls for d in b.detections()}),
                'remembered_objects': self.tools.world.summary(b.now())}

    def _validated(self, result: PlanResult, trace: MissionTrace, stats: dict, kind: str):
        """Validate; on rejection feed reasons back to the LLM. Returns (status, plan, reason)."""
        for attempt in range(self.max_validation_retries + 1):
            stats['plan_index'] += 1
            idx = stats['plan_index']
            trace.llm_calls(result.calls)
            stats['llm_calls'].extend(result.calls)
            stats['format_errors'] += sum(1 for c in result.calls if c.get('parse_error'))
            trace.plan(idx, kind if attempt == 0 else 'validation_repair',
                       result.plan.model_dump() if result.plan else None, result.status, result.reason)
            if result.status != 'plan':
                return result.status, None, result.reason
            rep = self.validator.validate(result.plan, prompt=self._prompt)
            trace.validation(idx, rep.as_dict())
            stats['repairs'] += len(rep.repaired)
            if rep.ok:
                self.log(f'[validate] plan {idx} OK' + (f' ({len(rep.repaired)} repairs: '
                         f'{"; ".join(str(f) for f in rep.repaired)})' if rep.repaired else ''))
                return 'ok', rep.plan, ''
            stats['unsafe_plans_caught'] += 1
            self.log(f'[validate] plan {idx} REJECTED: {"; ".join(rep.reasons())}')
            if attempt == self.max_validation_retries:
                return 'rejected', None, '; '.join(rep.reasons())
            self.on_status('validating', 0, stats['replans'], 'plan rejected, asking LLM to fix it')
            result = self.planner.repair(result, rep.reasons())
        return 'rejected', None, 'validation retries exhausted'

    def _describe(self, plan: MissionPlan) -> str:
        parts = []
        for s in plan.steps:
            t = s.params.get('target')
            parts.append(f'{s.tool}({t})' if t else s.tool)
        return ' -> '.join(parts)

    # ------------------------------------------------------------ main entry
    def run(self, prompt: str, dry_run: bool = False, wall_timeout_s: Optional[float] = None,
            mission_id: Optional[str] = None) -> dict:
        self._prompt = prompt
        trace = MissionTrace(prompt, self.provider.describe(), self.trace_dir, mission_id,
                             config={'backend': self.cfg.get('backend'), 'limits': self.cfg['limits'],
                                     'executor': self.cfg.get('executor', {}), 'tools': self.cfg.get('tools', {}),
                                     'planner': {'max_validation_retries': self.max_validation_retries,
                                                 'max_format_retries': self.planner.max_format_retries}},
                             env={'vehicle_backend': type(self.tools.backend).__name__,
                                  'dry_run': dry_run})
        stats = {'plan_index': 0, 'replans': 0, 'unsafe_plans_caught': 0, 'repairs': 0, 'format_errors': 0,
                 'llm_calls': [], 'steps_executed': 0}
        sim_t0 = self.tools.backend.now()
        ctx = self.context()
        trace.event('context', context=ctx)
        self.on_status('planning', 0, 0, 'asking the LLM for a plan')
        self.log(f'[plan] asking {self.provider.name} ({self.provider.model}) ...')
        status, plan, reason = self._validated(self.planner.plan(prompt, ctx), trace, stats, 'plan')

        if status != 'ok':
            outcome = {'refuse': 'refused', 'error': 'error', 'rejected': 'rejected'}[status]
            self.log(f'[plan] {outcome}: {reason}')
            return self._finish(trace, stats, outcome, reason, None, sim_t0)
        self.log(f'[plan] {len(plan.steps)} steps: {self._describe(plan)}')
        if dry_run:
            return self._finish(trace, stats, 'planned', '', plan, sim_t0)

        deadline = time.time() + wall_timeout_s if wall_timeout_s else None
        wd = Watchdog(self.tools, self.cfg.get('executor', {}).get('watchdog_period_s', 1.0), deadline)
        # the mission-time budget starts when the vehicle starts executing (planning happens before
        # deployment, like a real run); replanning during the mission does count against it
        wd.start()
        executor = Executor(self.tools, self.validator, self.cfg, on_event=self._event_cb(trace, stats))
        done: list[dict] = []          # briefs of successful steps across all plans
        try:
            self.on_status('executing', 0, 0, plan.summary)
            res = executor.run(plan, wd, stats['plan_index'])
            done += [r.brief() for r in res.records if r.success]
            while res.status == 'step_failed' and stats['replans'] < self.max_replans:
                stats['replans'] += 1
                failed = {'step': res.failed_step.model_dump(), 'result': res.failed_record.brief()}
                ctx = self.context()
                ctx['mission_elapsed_s'] = round(wd.elapsed(), 1)
                trace.data['replans'].append({'t': trace._t(), 'attempt': stats['replans'], 'failed': failed,
                                              'context': ctx})
                self.on_status('replanning', res.failed_step.id, stats['replans'],
                               f'step {res.failed_step.id} failed: {res.failed_record.detail}')
                self.log(f'[replan {stats["replans"]}/{self.max_replans}] step {res.failed_step.id} '
                         f'({res.failed_step.tool}) failed -> asking the LLM for a new plan')
                pr = self.planner.replan(prompt, done, failed, ctx, stats['replans'], self.max_replans)
                status, new_plan, reason = self._validated(pr, trace, stats, 'replan')
                if status != 'ok':
                    return self._abort(executor, trace, stats, f'replan {stats["replans"]} {status}: {reason}',
                                       plan, sim_t0)
                plan = new_plan
                self.log(f'[replan {stats["replans"]}] {len(plan.steps)} steps: {self._describe(plan)}')
                res = executor.run(plan, wd, stats['plan_index'])
                done += [r.brief() for r in res.records if r.success]
            if res.status == 'completed':
                return self._finish(trace, stats, 'succeeded', '', plan, sim_t0)
            if res.status == 'abort':
                reason = res.abort_reason
            else:
                reason = (f'step {res.failed_step.id} ({res.failed_step.tool}) failed: {res.failed_record.detail}'
                          + (f'; replans exhausted ({self.max_replans})' if self.max_replans else ''))
            return self._abort(executor, trace, stats, reason, plan, sim_t0)
        finally:
            wd.stop()

    def _event_cb(self, trace: MissionTrace, stats: dict):
        def cb(kind, data):
            if kind == 'step_start':
                s = data['step']
                self.on_status('executing', s['id'], stats['replans'], f'{s["tool"]} {s["params"]}')
                self.log(f'[step {s["id"]}] {s["tool"]} {s["params"]} (timeout {s["timeout_s"]:.0f}s)')
            elif kind == 'step_end':
                r = data['record']
                stats['steps_executed'] += 1
                trace.tool_call(r)
                obs = r['observation'] or {}
                tail = '' if r['success'] else f' [{obs.get("error_code")}] {obs.get("error") or ""}'
                self.log(f'[step {r["step"]["id"]}] {"ok" if r["success"] else "FAILED"} '
                         f'({r["wall_s"]:.1f}s wall, sim {r["sim_end"] - r["sim_start"]:.1f}s) {r["detail"]}{tail}')
            else:
                trace.event(kind, **data)
        return cb

    def _abort(self, executor: Executor, trace, stats, reason, plan, sim_t0) -> dict:
        self.on_status('aborting', 0, stats['replans'], reason)
        self.log(f'[abort] {reason} -> surfacing')
        info = executor.safe_abort(reason)
        trace.event('safe_abort', **info)
        return self._finish(trace, stats, 'aborted', reason, plan, sim_t0, surfaced=info['surfaced'])

    def _finish(self, trace, stats, outcome, reason, plan, sim_t0, surfaced=None) -> dict:
        calls = stats['llm_calls']
        u = [c['usage'] for c in calls if c.get('usage')]
        result = {
            'mission_id': trace.mission_id,
            'outcome': outcome,
            'reason': reason,
            'provider': self.provider.name,
            'model': self.provider.model,
            'plans': stats['plan_index'],
            'replans': stats['replans'],
            'unsafe_plans_caught': stats['unsafe_plans_caught'],
            'validator_repairs': stats['repairs'],
            'format_errors': stats['format_errors'],
            'llm_calls': len(calls),
            'tokens_in': sum(x['input_tokens'] for x in u),
            'tokens_out': sum(x['output_tokens'] for x in u),
            'llm_latency_s': round(sum(x['latency_s'] for x in u), 2),
            'cost_usd': round(sum(x['cost_usd'] for x in u), 6),
            'list_cost_usd': round(sum(x['list_cost_usd'] for x in u), 6),
            'steps_executed': stats['steps_executed'],
            'sim_time_s': round(self.tools.backend.now() - sim_t0, 1),
            'surfaced_after_abort': surfaced,
            'final_plan': plan.model_dump() if plan else None,
        }
        self.on_status({'succeeded': 'succeeded', 'planned': 'succeeded', 'aborted': 'aborted'}.get(outcome, outcome),
                       0, stats['replans'], reason)
        trace.finish(result)
        result['wall_s'] = trace.data['wall_s']
        result['trace'] = trace.save()
        return result
