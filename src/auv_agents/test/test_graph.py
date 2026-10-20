"""LangGraph multi-agent graph on the MuJoCo sim with a scripted (mock) LLM: no network, no Ollama."""

from auv_agents.run import build_runtime, load_agents_config, run_mission
from auv_agents.specialists import DiagnosticsAgent, deterministic_diagnosis
from auv_agents.vlm import MockVLM
from auv_mission_planner.llm.providers import MockProvider
from auv_onboard.factory import make_backend


def P(steps, aborts=None):
    return {'status': 'plan', 'reason': '', 'plan': {
        'summary': 't', 'abort_conditions': aborts if aborts is not None else [{'type': 'battery_below', 'percent': 20}],
        'steps': [dict(id=i + 1, **s) for i, s in enumerate(steps)]}}


DIVE = {'tool': 'hold_depth', 'params': {'depth': 1.5, 'duration_s': 2}, 'timeout_s': 40,
        'success': {'type': 'depth_within', 'depth': 1.5, 'tolerance_m': 0.4}}
GATE = {'tool': 'navigate_to', 'params': {'target': 'gate', 'pass_through': True}, 'timeout_s': 120}
RED = {'tool': 'navigate_to', 'params': {'target': 'red_buoy', 'standoff_m': 0}, 'timeout_s': 120}
SEARCH_RED = {'tool': 'search_pattern', 'params': {'target': 'red_buoy', 'pattern': 'yaw_sweep'}, 'timeout_s': 120,
              'success': {'type': 'detected', 'target': 'red_buoy'}}
SURF = {'tool': 'surface', 'timeout_s': 90, 'success': {'type': 'surfaced'}, 'on_failure': 'abort'}
PROMPT = 'pass through the gate, touch the red buoy, then surface'


def setup(responses, faults=None, approval='auto', vlm=None, diag_llm=None, tracking_on=True, **ob):
    be = make_backend('default', faults=faults or [], seed=0,
                      onboard_overrides={'backend': {'max_rtf': None, 'idle_rtf': None, **ob},
                                         'tracking': {'enabled': tracking_on}})
    acfg = load_agents_config({'graph': {'approval': approval}, 'diagnostics': {'use_llm': diag_llm is not None}})
    rt = build_runtime(be, MockProvider(responses=responses), acfg, diag_llm, vlm, auto_approve=approval == 'auto')
    return be, rt


def finish_abort(be):
    import time
    t_end = time.time() + 60
    while time.time() < t_end and any(t.is_alive() for t in be._threads):
        time.sleep(0.05)


def test_nominal_mission_succeeds_and_streams_agent_events():
    be, rt = setup([P([DIVE, GATE, SEARCH_RED, RED, SURF])])
    events = []
    try:
        res = run_mission(rt, PROMPT, on_event=lambda n, e: events.append((n, e['agent'])))
        r = be.sim.referee.report()
        assert res['outcome'] == 'succeeded', res.get('reason')
        assert r['gate_passed'] and r['buoys_touched'] == ['red_buoy'] and r['surfaced_at_end']
        agents = {a for _, a in events}
        assert {'supervisor', 'safety_validator', 'operator', 'executor', 'perception', 'navigation'} <= agents
        assert res['stats']['steps_executed'] == 5
    finally:
        be.shutdown()


def test_human_in_the_loop_interrupt_and_checkpointed_resume(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    be, rt = setup([P([DIVE, SURF])], approval='interrupt')
    try:
        with SqliteSaver.from_conn_string(str(tmp_path / 'ck.sqlite')) as ck:
            res = run_mission(rt, 'dive to 1.5 m then surface', thread_id='t1', checkpointer=ck)
            assert res['outcome'] == 'awaiting_approval'
            assert be.sim.referee.report()['max_depth_m'] < 0.3        # nothing moved before approval
            # resume the same thread from the checkpoint with the operator's decision
            res = run_mission(rt, 'dive to 1.5 m then surface', thread_id='t1', checkpointer=ck,
                              approve_fn=lambda payload: {'decision': 'approve'})
        assert res['outcome'] == 'succeeded'
        assert res['approval']['by'] == 'operator'
        assert be.sim.referee.report()['max_depth_m'] > 1.2
    finally:
        be.shutdown()


def test_operator_rejects_plan():
    be, rt = setup([P([DIVE, SURF])], approval='interrupt')
    try:
        res = run_mission(rt, 'dive to 1.5 m then surface', approve_fn=lambda p: {'decision': 'reject'})
        assert res['outcome'] == 'operator_rejected'
        assert be.sim.referee.report()['path_length_m'] < 0.5
    finally:
        be.shutdown()


def test_refusal_and_validator_rejection_with_repair():
    be, rt = setup([{'status': 'refuse', 'reason': 'there is no purple buoy'}])
    try:
        res = run_mission(rt, 'touch the purple buoy')
        assert res['outcome'] == 'refused'
    finally:
        be.shutdown()
    deep = dict(DIVE, params={'depth': 10.0, 'duration_s': 2}, success={'type': 'depth_within', 'depth': 10.0})
    be, rt = setup([P([deep, SURF]), P([DIVE, SURF])])
    try:
        res = run_mission(rt, 'dive to 1.5 m then surface')
        assert res['outcome'] == 'succeeded'
        assert res['stats']['unsafe_plans_caught'] == 1
        assert be.sim.referee.report()['max_depth_m'] < 2.0
    finally:
        be.shutdown()


def test_leak_health_veto_aborts_and_surfaces():
    be, rt = setup([P([DIVE, GATE, SEARCH_RED, RED, SURF])], faults=[{'type': 'leak', 't_start': 12.0}])
    try:
        res = run_mission(rt, PROMPT)
        finish_abort(be)
        r = be.sim.referee.report()
        assert res['outcome'] == 'aborted' and 'leak' in res['reason']
        assert r['surfaced_at_end'] and r['collisions'] == 0
    finally:
        be.shutdown()


def test_dvl_dropout_is_diagnosed_and_mission_continues():
    be, rt = setup([P([DIVE, GATE, SEARCH_RED, RED, SURF])], faults=[{'type': 'dvl_dropout', 't_start': 5.0}])
    try:
        res = run_mission(rt, PROMPT)
        assert res['outcome'] == 'succeeded'
        assert 'dvl_dropout' in res['diagnosis']['fault_types']
        assert res['nav']['mode'] == 'dead_reckoning'
        assert res['stats']['diagnoses'] >= 1
    finally:
        be.shutdown()


def test_camera_blackout_diagnosed_replanned_gate_found_by_sonar():
    events = []
    be, rt = setup([P([DIVE, GATE, SURF]), P([GATE, SURF])], faults=[{'type': 'camera_blackout', 't_start': 0.0}])
    try:
        res = run_mission(rt, 'pass through the gate then surface', on_event=lambda n, e: events.append(e['msg']))
        r = be.sim.referee.report()
        assert res['outcome'] == 'succeeded' and r['gate_passed'] and r['surfaced_at_end'], res.get('reason')
        assert res['replans'] == 1 and 'camera_blackout' in res['diagnosis']['fault_types']
        assert not be.detections()                     # the camera never saw anything
    finally:
        be.shutdown()


def test_turbidity_sonar_cued_search_touches_red_buoy():
    """Fusion tracking on (default): the perception agent visits sonar tracks until the camera
    classifies the red buoy, instead of a blind search."""
    be, rt = setup([P([DIVE, SEARCH_RED, RED, SURF])] + [P([SEARCH_RED, RED, SURF])] * 2 + [P([SURF])] * 3,
                   faults=[{'type': 'turbidity', 't_start': 0.0, 'params': {'visibility_m': 2.5}}])
    try:
        res = run_mission(rt, 'touch the red buoy then surface')
        r = be.sim.referee.report()
        assert res['outcome'] == 'succeeded' and r['touch_order'][:1] == ['red_buoy'], (res.get('reason'), r['touch_order'])
        assert res['stats'].get('inspections', 0) >= 1 and r['surfaced_at_end']
    finally:
        be.shutdown()


def test_failed_search_queries_vlm_when_tracking_is_off():
    vlm = MockVLM()
    be, rt = setup([P([DIVE, SEARCH_RED, RED, SURF])] + [P([SEARCH_RED, RED, SURF])] * 2 + [P([SURF])] * 3,
                   faults=[{'type': 'turbidity', 't_start': 0.0, 'params': {'visibility_m': 2.0}}], vlm=vlm,
                   tracking_on=False)
    try:
        run_mission(rt, 'touch the red buoy then surface')
        assert rt.perception.vlm_log and rt.perception.vlm_log[0]['answer']
        assert be.sim.referee.report()['surfaced_at_end']
    finally:
        be.shutdown()


def test_diagnostics_llm_cannot_downgrade_a_veto():
    health = {'faults': [{'type': 'leak', 'detail': 'wet', 'since': 1.0}], 'veto': True, 'degraded_modes': []}
    assert deterministic_diagnosis(health)['action'] == 'abort'
    llm = MockProvider(responses=[{'fault_types': ['leak'], 'explanation': 'probably a false alarm',
                                   'degraded_mode': 'none', 'action': 'continue'}])
    d = DiagnosticsAgent(llm).diagnose(health, {}, {}, {})
    assert d['action'] == 'abort' and d['source'] == 'llm'
