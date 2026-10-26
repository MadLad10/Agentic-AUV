"""Mission-control dashboard: FastAPI backend + one static page (vanilla JS, no build step).

Shows the live front camera (with detector boxes), a third-person view, a top-down map (truth vs
EKF), sensor/health status, the agents' streamed messages, and the plan waiting for approval with
Approve / Reject buttons (the LangGraph human-in-the-loop interrupt). Faults can be injected live.

  python3 -m auv_agents.dashboard --provider ollama      # http://127.0.0.1:8070
"""
from __future__ import annotations

import argparse
import io
import math
import os
import threading
from collections import deque

from typing import Optional

from PIL import Image, ImageDraw
from pydantic import BaseModel

STATIC = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static')   # source tree (symlink install)


class Session:
    def __init__(self, provider_name: str, arena: str = 'default', detector: str = 'ground_truth'):
        from auv_agents.run import build_runtime, load_agents_config
        from auv_agents.vlm import make_vlm
        from auv_mission_planner.config import load_llm_config
        from auv_mission_planner.llm.providers import make_provider
        from auv_onboard.factory import make_backend
        self.be = make_backend(arena, onboard_overrides={'backend': {'detection_source': detector}})
        llm = load_llm_config()
        acfg = load_agents_config({'graph': {'approval': 'interrupt'}})
        self.rt = build_runtime(self.be, make_provider(llm, provider_name), acfg, make_provider(llm, provider_name),
                                make_vlm(acfg['perception']['vlm']), auto_approve=False)
        self.events: deque = deque(maxlen=200)
        self.track: deque = deque(maxlen=4000)
        self.pending = None
        self.decision = None
        self.decided = threading.Event()
        self.running = False
        self.result = None
        self.be.tick_hooks.append(self._on_tick)

    def _on_tick(self, be):
        if int(be.now() / be.sim.control_dt) % 5 == 0:
            tr, e = be.sim.truth(), be.est()
            self.track.append((round(float(tr.pos[0]), 2), round(float(tr.pos[1]), 2),
                               round(float(e['x']), 2), round(float(e['y']), 2)))

    def _approve(self, payload):
        # Reset before publishing the next plan. A timeout must never reuse a
        # decision from a previous approval (including a replan in this mission).
        self.decision = None
        self.decided.clear()
        self.pending = payload.get('plan')
        try:
            answered = self.decided.wait(timeout=600)
            return {'decision': (self.decision or 'reject') if answered else 'reject'}
        finally:
            self.pending = None

    def start(self, prompt: str):
        if self.running:
            return False
        from auv_agents.run import run_mission
        self.running, self.result = True, None
        self.events.append({'agent': 'operator', 'msg': f'mission: {prompt}', 't': round(self.be.now(), 1)})

        def run():
            try:
                self.result = run_mission(self.rt, prompt, approve_fn=self._approve,
                                          on_event=lambda node, ev: self.events.append(
                                              {'agent': ev['agent'], 'msg': ev['msg'], 't': round(self.be.now(), 1)}))
                self.events.append({'agent': 'supervisor', 'msg': f'outcome: {self.result.get("outcome")}',
                                    't': round(self.be.now(), 1)})
            except Exception as e:
                self.events.append({'agent': 'error', 'msg': f'{type(e).__name__}: {e}', 't': round(self.be.now(), 1)})
            finally:
                self.running = False
        threading.Thread(target=run, daemon=True).start()
        return True

    def decide(self, decision: str):
        self.decision = decision
        self.decided.set()

    def state(self) -> dict:
        be = self.be
        tr, e = be.sim.truth(), be.est()
        h = be.health.last
        r = be.sim.readings
        ref = be.sim.referee.report()
        return {
            't': round(be.now(), 1), 'running': self.running,
            'truth': {'x': round(float(tr.pos[0]), 2), 'y': round(float(tr.pos[1]), 2), 'depth': round(tr.depth, 2),
                      'yaw_deg': round(math.degrees(tr.yaw), 1)},
            'estimate': {'x': round(float(e['x']), 2), 'y': round(float(e['y']), 2), 'depth': round(-float(e['z']), 2),
                         'pos_std_m': round(be.ekf.pos_std(), 2)},
            'battery': r.get('battery', {}).get('percent'),
            'dvl_valid': r.get('dvl', {}).get('valid'), 'leak': r.get('leak', {}).get('leak'),
            'thrusters': dict(zip(r.get('thrusters', {}).get('names', []), r.get('thrusters', {}).get('thrust_n', []))),
            'health': {'faults': h.faults, 'veto': h.veto, 'veto_reason': h.veto_reason,
                       'degraded_modes': h.degraded_modes},
            'events': list(self.events)[-60:], 'pending_plan': self.pending,
            'outcome': (self.result or {}).get('outcome'),
            'referee': {k: ref[k] for k in ('gate_passed', 'buoys_touched', 'surfaced', 'max_depth_m', 'collisions')},
            'track': list(self.track)[::2][-1500:],
            'objects': [{'class': o['class'], 'x': o['x'], 'y': o['y']} for o in be.sim.arena['objects']],
            'pool': be.sim.arena['pool'],
        }

    def jpeg(self, which: str) -> bytes:
        be = self.be
        if which == 'camera':
            img = Image.fromarray(be.sim.camera_image())
            d = ImageDraw.Draw(img)
            for det in getattr(be.detector, 'detect', lambda: [])():
                if det.bbox:
                    d.rectangle(det.bbox, outline=(255, 220, 0), width=2)
                    d.text((det.bbox[0], max(0, det.bbox[1] - 11)), f'{det.cls} {det.confidence:.2f}', fill=(255, 220, 0))
        else:
            img = Image.fromarray(be.sim.render_view('spectator', 640, 360))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        return buf.getvalue()


class Mission(BaseModel):
    prompt: str


class Decision(BaseModel):
    decision: str


class FaultIn(BaseModel):
    type: str
    params: dict = {}
    duration: Optional[float] = None


def make_app(session: Session):
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse, Response

    app = FastAPI(title='AUV mission control')

    @app.get('/')
    def index():
        return FileResponse(os.path.join(STATIC, 'index.html'))

    @app.get('/api/state')
    def state():
        return JSONResponse(session.state())

    @app.get('/api/{which}.jpg')
    def image(which: str):
        return Response(session.jpeg('camera' if which == 'camera' else 'spectator'), media_type='image/jpeg')

    @app.post('/api/mission')
    def mission(m: Mission):
        return {'started': session.start(m.prompt)}

    @app.post('/api/approve')
    def approve(d: Decision):
        session.decide('approve' if d.decision == 'approve' else 'reject')
        return {'ok': True}

    @app.post('/api/fault')
    def fault(f: FaultIn):
        session.be.sim.faults.add({'type': f.type, 't_start': session.be.now(), 'duration': f.duration,
                                   'params': f.params})
        session.events.append({'agent': 'operator', 'msg': f'injected fault {f.type} {f.params}',
                               't': round(session.be.now(), 1)})
        return {'ok': True}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--provider', default='ollama')
    ap.add_argument('--port', type=int, default=8070)
    ap.add_argument('--detector', default='ground_truth')
    a = ap.parse_args()
    import uvicorn
    s = Session(a.provider, detector=a.detector)
    try:
        uvicorn.run(make_app(s), host='127.0.0.1', port=a.port, log_level='warning')
    finally:
        s.be.shutdown()


if __name__ == '__main__':
    main()
