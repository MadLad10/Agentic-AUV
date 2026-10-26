"""Dashboard API: start a mission, plan waits for operator approval, approve, mission completes."""
import json
import time

import pytest

PLAN = {'status': 'plan', 'reason': '', 'plan': {'summary': 't', 'abort_conditions': [{'type': 'battery_below', 'percent': 20}],
        'steps': [{'id': 1, 'tool': 'hold_depth', 'params': {'depth': 1.0, 'duration_s': 1}, 'timeout_s': 30},
                  {'id': 2, 'tool': 'surface', 'timeout_s': 60, 'success': {'type': 'surfaced'}}]}}


def test_approval_timeout_rejects_after_a_previous_approval():
    from auv_agents.dashboard import Session
    s = Session.__new__(Session)
    s.decision = None
    s.pending = None

    class Decisions:
        calls = 0

        def clear(self):
            assert s.decision is None

        def wait(self, timeout):
            self.calls += 1
            assert s.pending is not None
            if self.calls == 1:
                s.decision = 'approve'
                return True
            return False

    s.decided = Decisions()
    assert s._approve({'plan': {'steps': [1]}}) == {'decision': 'approve'}
    assert s.pending is None
    assert s._approve({'plan': {'steps': [2]}}) == {'decision': 'reject'}
    assert s.pending is None


def test_dashboard_hitl_flow(tmp_path, monkeypatch):
    pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from auv_agents.dashboard import Session, make_app
    f = tmp_path / 'resp.json'
    f.write_text(json.dumps([PLAN]))
    monkeypatch.setenv('AUV_MOCK_RESPONSES', str(f))
    s = Session('mock')
    try:
        c = TestClient(make_app(s))
        assert c.get('/').status_code == 200
        assert c.post('/api/mission', json={'prompt': 'dive to 1 m then surface'}).json()['started']
        t_end = time.time() + 30
        while time.time() < t_end and not c.get('/api/state').json()['pending_plan']:
            time.sleep(0.1)
        st = c.get('/api/state').json()
        assert [x['tool'] for x in st['pending_plan']['steps']] == ['hold_depth', 'surface']
        c.post('/api/approve', json={'decision': 'approve'})
        t_end = time.time() + 120
        while time.time() < t_end and (c.get('/api/state').json()['running']):
            time.sleep(0.2)
        st = c.get('/api/state').json()
        assert st['outcome'] == 'succeeded' and st['referee']['max_depth_m'] > 0.7
        assert c.get('/api/camera.jpg').headers['content-type'] == 'image/jpeg'
        assert any(e['agent'] == 'operator' for e in st['events'])
    finally:
        s.be.shutdown()
