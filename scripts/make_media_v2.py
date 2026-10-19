#!/usr/bin/env python3
"""Record README media headlessly from a MuJoCo run (no display needed).

Runs a tool sequence through v1's tool registry on the v2 onboard stack (the same tools the agents
call) and captures, every --every sim seconds:
  - a 2x2 panel GIF: third-person view | onboard camera (+ detector boxes) | sonar fan | top-down map
  - a top-down plot of the true path vs the EKF estimate (localisation error)

  python3 scripts/make_media_v2.py --name nominal
  python3 scripts/make_media_v2.py --name dvl_dropout --fault dvl_dropout:10
  python3 scripts/make_media_v2.py --name blackout --fault camera_blackout:0 --steps gate_only
"""
import argparse
import json
import math
import os
import sys


STEPS = {
    'gate_buoy': [('navigate_to', {'x': 2.0, 'y': 0.0, 'depth': 1.5}, 60), ('navigate_to', {'target': 'gate'}, 120),
                  ('search_pattern', {'target': 'red_buoy'}, 120), ('navigate_to', {'target': 'red_buoy'}, 120),
                  ('surface', {}, 90)],
    'gate_only': [('navigate_to', {'x': 2.0, 'y': 0.0, 'depth': 1.5}, 60), ('navigate_to', {'target': 'gate'}, 120),
                  ('surface', {}, 90)],
}
COLORS = {'gate': (255, 140, 0), 'red_buoy': (255, 60, 60), 'green_buoy': (60, 230, 90), 'yellow_buoy': (255, 230, 40)}


def parse_fault(s):
    parts = s.split(':')
    f = {'type': parts[0], 't_start': float(parts[1]) if len(parts) > 1 else 0.0, 'params': {}}
    for kv in (parts[2].split(',') if len(parts) > 2 else []):
        k, v = kv.split('=')
        try:
            v = float(v)
        except ValueError:
            pass
        f['params'][k] = v
    return f


def sonar_panel(scan, size=320):
    from PIL import Image, ImageDraw
    im = Image.new('RGB', (size, size), (8, 16, 24))
    d = ImageDraw.Draw(im)
    cx, cy, R = size / 2, size - 12, size - 24
    rmax = 15.0
    for r in (5, 10, 15):
        rr = R * r / rmax
        d.arc([cx - rr, cy - rr, cx + rr, cy + rr], 210, 330, fill=(40, 70, 80))
    if scan and scan.get('valid'):
        for b, r, it in zip(scan['bearings'], scan['ranges'], scan['intensity']):
            if r <= 0:
                continue
            rr = R * min(r, rmax) / rmax
            x, y = cx - rr * math.sin(b), cy - rr * math.cos(b)
            g = int(80 + 175 * min(1.0, it))
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(g, g, 60))
    else:
        d.text((cx - 40, cy - R / 2), 'NO SONAR', fill=(255, 80, 80))
    d.text((6, 6), 'forward sonar (15 m)', fill=(180, 200, 210))
    return im


def map_panel(arena, truth, est, size=(320, 220)):
    from PIL import Image, ImageDraw
    W, H = size
    p = arena['pool']
    x0, L, Wd = p.get('x_min', -5.0), p['length'], p['width']
    sx, sy = W / L, H / Wd
    to = lambda x, y: ((x - x0) * sx, H / 2 - y * sy)
    im = Image.new('RGB', size, (20, 45, 60))
    d = ImageDraw.Draw(im)
    for o in arena['objects']:
        x, y = to(o['x'], o['y'])
        c = COLORS.get(o['class'], (200, 200, 200))
        if o['class'] == 'gate':
            d.line([to(o['x'], o['y'] - 1), to(o['x'], o['y'] + 1)], fill=c, width=3)
        else:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=c)
    if len(truth) > 1:
        d.line([to(*q) for q in truth], fill=(240, 240, 240), width=2)
    if len(est) > 1:
        d.line([to(*q) for q in est], fill=(80, 200, 255), width=1)
    d.text((6, 6), 'top-down: truth (white) / EKF (blue)', fill=(200, 220, 230))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='nominal')
    ap.add_argument('--fault', action='append', default=[])
    ap.add_argument('--steps', default='gate_buoy', choices=list(STEPS))
    ap.add_argument('--detector', default='ground_truth')
    ap.add_argument('--every', type=float, default=1.0)
    ap.add_argument('--out', default='docs/media')
    ap.add_argument('--graph', default=None, help='run the v2 LangGraph agents with this mission prompt')
    ap.add_argument('--plans', default=None, help='JSON list of LLM outputs to replay (mock LLM) for --graph')
    ap.add_argument('--provider', default='ollama', help='LLM for --graph when --plans is not given')
    a = ap.parse_args()
    from PIL import Image, ImageDraw
    from auv_onboard.factory import make_backend, make_registry
    be = make_backend('default', faults=[parse_fault(f) for f in a.fault],
                      onboard_overrides={'backend': {'max_rtf': None, 'idle_rtf': None,
                                                     'detection_source': a.detector}})
    reg, _ = make_registry(be)
    frames, truth, est, errs = [], [], [], []
    state = {'next': 0.0}

    def capture(b):
        tr = b.sim.truth()
        e = b.est()
        truth.append((float(tr.pos[0]), float(tr.pos[1])))
        est.append((float(e['x']), float(e['y'])))
        errs.append((b.now(), math.hypot(e['x'] - tr.pos[0], e['y'] - tr.pos[1]), tr.depth))
        if b.now() + 1e-9 < state['next']:
            return
        state['next'] = b.now() + a.every
        spec = Image.fromarray(b.sim.render_view('spectator', 640, 360))
        cam = Image.fromarray(b.sim.camera_image()).resize((480, 360))
        dc = ImageDraw.Draw(cam)
        k = 480 / 640
        for d in b.detector.detect() if hasattr(b.detector, 'detect') else []:
            if d.bbox:
                x0, y0, x1, y1 = (v * k for v in d.bbox)
                c = COLORS.get(d.cls, (255, 255, 255))
                dc.rectangle([x0, y0, x1, y1], outline=c, width=2)
                dc.text((x0, max(0, y0 - 11)), f'{d.cls} {d.confidence:.2f}', fill=c)
        top = Image.new('RGB', (1120, 360))
        top.paste(spec, (0, 0))
        top.paste(cam, (640, 0))
        bot = Image.new('RGB', (1120, 220), (10, 20, 28))
        bot.paste(sonar_panel(b.sim.readings.get('sonar'), 220), (0, 0))
        bot.paste(map_panel(b.sim.arena, truth, est, (420, 220)), (240, 0))
        db = ImageDraw.Draw(bot)
        h = b.health.last
        faults = ', '.join(f['type'] for f in h.faults) or 'none'
        db.text((680, 12), f't = {b.now():5.1f} s   depth {tr.depth:4.2f} m', fill=(220, 230, 240))
        db.text((680, 36), f'EKF error {errs[-1][1]:.2f} m   pos std {b.ekf.pos_std():.2f} m', fill=(120, 210, 255))
        db.text((680, 60), f'battery {b.battery_percent():.0f}%', fill=(220, 230, 240))
        db.text((680, 84), f'health faults: {faults}', fill=(255, 120, 120) if h.faults else (150, 220, 150))
        if h.veto:
            db.text((680, 108), f'VETO: {h.veto_reason}'[:60], fill=(255, 60, 60))
        frame = Image.new('RGB', (1120, 580))
        frame.paste(top, (0, 0))
        frame.paste(bot, (0, 360))
        frames.append(frame.resize((720, 373)).quantize(colors=96, method=Image.Quantize.FASTOCTREE))
    be.tick_hooks.append(capture)
    log = []
    try:
        if a.graph:
            from auv_agents.run import build_runtime, load_agents_config, run_mission
            from auv_mission_planner.config import load_llm_config
            from auv_mission_planner.llm.providers import MockProvider, make_provider
            if a.plans:
                with open(a.plans) as f:
                    prov = MockProvider(responses=json.load(f))
                diag = None
            else:
                prov = make_provider(load_llm_config(), a.provider)
                diag = make_provider(load_llm_config(), a.provider)
            acfg = load_agents_config({'diagnostics': {'use_llm': diag is not None}})
            rt = build_runtime(be, prov, acfg, diag, None)
            res = run_mission(rt, a.graph, on_event=lambda n, e: log.append(f'[{e["agent"]}] {e["msg"]}'))
            log.append(f'outcome: {res.get("outcome")}')
        else:
            for name, p, to in STEPS[a.steps]:
                obs = reg.call(name, p, to)
                log.append(f'{name} {p}: {"ok" if obs.ok else obs.error}')
                if not obs.ok and name != 'search_pattern':
                    break
        import time
        t_end = time.time() + 60
        while time.time() < t_end and any(t.is_alive() for t in be._threads):
            time.sleep(0.05)
        rep = be.sim.referee.report()
    finally:
        be.tick_hooks.clear()
        be.shutdown()
    os.makedirs(a.out, exist_ok=True)
    gif = os.path.join(a.out, f'v2_{a.name}.gif')
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=150, loop=0, optimize=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    t = [e[0] for e in errs]
    ax1.plot([q[0] for q in truth], [q[1] for q in truth], 'k-', lw=2, label='true path')
    ax1.plot([q[0] for q in est], [q[1] for q in est], 'C0--', lw=1.5, label='EKF estimate')
    with open(os.path.join(os.path.dirname(__file__), '..', 'src', 'auv_marine_sim', 'config', 'arenas', 'default.yaml')) as f:
        import yaml
        arena = yaml.safe_load(f)
    for o in arena['objects']:
        if o['class'] == 'gate':
            ax1.plot([o['x'], o['x']], [o['y'] - 1, o['y'] + 1], color='darkorange', lw=4)
        else:
            ax1.plot(o['x'], o['y'], 'o', color={'red_buoy': 'red', 'green_buoy': 'green'}.get(o['class'], 'gold'), ms=9)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_title('Top-down: truth vs EKF')
    ax2.plot(t, [e[1] for e in errs], 'C0', label='horizontal EKF error (m)')
    ax2.plot(t, [e[2] for e in errs], 'C2', alpha=0.6, label='true depth (m)')
    for f in a.fault:
        ft = parse_fault(f)
        ax2.axvline(ft['t_start'], color='r', ls=':', label=f'{ft["type"]} injected')
    ax2.set_xlabel('sim time (s)')
    ax2.legend(fontsize=8)
    ax2.set_title('Localisation error over time')
    fig.tight_layout()
    png = os.path.join(a.out, f'v2_{a.name}_nav.png')
    fig.savefig(png, dpi=110)
    info = {'gif': gif, 'png': png, 'frames': len(frames), 'steps': log, 'faults': a.fault,
            'mode': ('graph+replayed LLM plans' if a.plans else f'graph+{a.provider}') if a.graph else 'scripted tools',
            'referee': {k: v for k, v in rep.items() if k != 'events'}}
    with open(os.path.join(a.out, f'v2_{a.name}.json'), 'w') as f:
        json.dump(info, f, indent=1, default=str)
    print(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
