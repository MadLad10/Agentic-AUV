"""OpenTelemetry spans: mission -> graph nodes -> LLM calls (gen_ai.*) and tool calls."""
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from auv_agents import telemetry
from auv_agents.run import build_runtime, load_agents_config, run_mission
from auv_mission_planner.llm.providers import MockProvider
from auv_onboard.factory import make_backend

PLAN = {'status': 'plan', 'reason': '', 'plan': {'summary': 't', 'abort_conditions': [{'type': 'battery_below', 'percent': 20}],
        'steps': [{'id': 1, 'tool': 'hold_depth', 'params': {'depth': 1.0, 'duration_s': 1}, 'timeout_s': 30},
                  {'id': 2, 'tool': 'surface', 'timeout_s': 60, 'success': {'type': 'surfaced'}}]}}


def test_spans_cover_mission_nodes_llm_and_tools():
    exp = InMemorySpanExporter()
    telemetry.init(exporter=exp)
    be = make_backend('default', seed=0, onboard_overrides={'backend': {'max_rtf': None, 'idle_rtf': None}})
    try:
        rt = build_runtime(be, MockProvider(responses=[PLAN]), load_agents_config({'diagnostics': {'use_llm': False}}))
        res = run_mission(rt, 'dive to 1 m then surface')
        assert res['outcome'] == 'succeeded'
    finally:
        be.shutdown()
        telemetry._tracer = None
    spans = {s.name: s for s in exp.get_finished_spans()}
    assert {'mission', 'graph.supervisor', 'graph.validate', 'graph.execute_step', 'llm.planner',
            'tool.hold_depth', 'tool.surface'} <= set(spans)
    llm = spans['llm.planner']
    assert llm.attributes['gen_ai.request.model'] == 'mock' and llm.attributes['gen_ai.usage.output_tokens'] > 0
    assert spans['mission'].attributes['auv.outcome'] == 'succeeded'
    assert spans['tool.hold_depth'].attributes['auv.tool.ok'] is True
    # nesting: the tool span is inside the execute_step node span, inside the mission span
    by_id = {s.context.span_id: s for s in exp.get_finished_spans()}
    parent = by_id[spans['tool.hold_depth'].parent.span_id]
    assert parent.name == 'graph.execute_step' and by_id[parent.parent.span_id].name == 'mission'
