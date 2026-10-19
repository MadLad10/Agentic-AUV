#!/usr/bin/env python3
"""Turn a run directory (from scripts/run_sim_test.sh) into README media.

  scripts/make_media.py RUN_DIR --out docs/media --name gate_red_buoy [--arena default]

Produces:
  <name>_topdown.png   pool, gate and buoys from the arena YAML, the vehicle's path from the rosbag
                       (/auv/odom), coloured by depth, with referee events marked
  <name>_depth.png     depth vs time with the same events
  <name>_onboard.gif   front-camera frames (RUN_DIR/frames, recorded with run_sim_test.sh -c)
  <name>_spectator.gif third-person frames (RUN_DIR/frames_spectator, recorded with -S)
"""
import argparse
import glob
import json
import os
import sqlite3
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
COLORS = {'red_buoy': '#e03131', 'green_buoy': '#2f9e44', 'yellow_buoy': '#f2c80f', 'purple_buoy': '#9c36b5'}


def read_odom(bag_dir):
    """(t, x, y, depth) from /auv/odom in a ROS 2 sqlite bag."""
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry
    dbs = glob.glob(os.path.join(bag_dir, '*.db3'))
    if not dbs:
        return []
    con = sqlite3.connect(dbs[0])
    tid = con.execute("select id from topics where name='/auv/odom'").fetchone()
    if tid is None:
        return []
    out = []
    for ts, data in con.execute('select timestamp, data from messages where topic_id=? order by timestamp', tid):
        m = deserialize_message(data, Odometry)
        p = m.pose.pose.position
        out.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, p.x, p.y, -p.z))
    return out


def arena_objects(name):
    path = name if os.path.isfile(name) else os.path.join(WS, 'src', 'auv_sim', 'config', 'arenas', f'{name}.yaml')
    return yaml.safe_load(open(path))


def plot_topdown(odom, arena, events, title, path):
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=110)
    p = arena['pool']
    x0 = p.get('x_min', -5.0)
    ax.add_patch(plt.Rectangle((x0, -p['width'] / 2), p['length'], p['width'], fill=True, fc='#e7f5ff', ec='#1c7ed6', lw=1.5))
    for o in arena['objects']:
        if o['class'] == 'gate':
            w = o['width'] / 2
            ax.plot([o['x'], o['x']], [o['y'] - w, o['y'] + w], color='#495057', lw=2.5, zorder=3)
            for s in (-1, 1):
                ax.plot(o['x'], o['y'] + s * w, 's', color='#fd7e14', ms=8, zorder=4)
            ax.annotate('gate', (o['x'], o['y'] + w + 0.4), ha='center', fontsize=9)
        else:
            ax.add_patch(plt.Circle((o['x'], o['y']), max(0.3, o.get('radius', 0.2)), color=COLORS.get(o['class'], 'k'),
                                    zorder=4))
            ax.annotate(o['id'].replace('_', ' '), (o['x'], o['y'] - 0.9), ha='center', fontsize=8)
    if odom:
        xs, ys, ds = [r[1] for r in odom], [r[2] for r in odom], [r[3] for r in odom]
        sc = ax.scatter(xs, ys, c=ds, cmap='viridis_r', s=4, zorder=5, vmin=0, vmax=max(2.5, max(ds)))
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, shrink=0.65, label='depth (m)')
        ax.plot(xs[0], ys[0], marker='>', color='k', ms=9, zorder=6)
        ax.annotate('start', (xs[0], ys[0] + 0.6), fontsize=8, ha='center')
    for k, e in enumerate(events):
        pos = (e.get('x'), e.get('y'))
        if None in pos and odom:
            near = min(odom, key=lambda r: abs(r[0] - e['t']))
            pos = (near[1], near[2])
        if None in pos:
            continue
        label = {'gate_passed': 'gate passed', 'buoy_touched': f'touched {e.get("buoy", "")}'.replace('_', ' '),
                 'surfaced': 'surfaced'}.get(e['event'])
        if label:
            ax.plot(*pos, marker='*', color='#d6336c', ms=13, zorder=7, mec='white')
            ax.annotate(label, pos, xytext=(8, 10 - 14 * (k % 3)), textcoords='offset points', fontsize=8,
                        color='#a61e4d', bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax.set_aspect('equal')
    xs_all = [o['x'] for o in arena['objects']] + ([r[1] for r in odom] if odom else [0.0])
    ys_all = [o['y'] for o in arena['objects']] + ([r[2] for r in odom] if odom else [0.0])
    ax.set_xlim(max(x0 - 0.5, min(xs_all) - 2.5), min(x0 + p['length'] + 0.5, max(xs_all) + 2.5))
    ax.set_ylim(max(-p['width'] / 2 - 0.5, min(ys_all) - 2.5), min(p['width'] / 2 + 0.5, max(ys_all) + 2.5))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_depth(odom, events, path, motion_start=None, llm_label='LLM planning'):
    if not odom:
        return
    t0 = odom[0][0]
    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=110)
    if motion_start is not None and motion_start > t0:
        ax.axvspan(0, motion_start - t0, color='#fff3bf', zorder=0)
        ax.annotate(llm_label, ((motion_start - t0) / 2, 0.6), ha='center', fontsize=8, color='#e67700')
    ax.plot([r[0] - t0 for r in odom], [r[3] for r in odom], color='#1c7ed6', lw=1.4)
    for e in events:
        if e['event'] in ('gate_passed', 'buoy_touched', 'surfaced'):
            ax.axvline(e['t'] - t0, color='#d6336c', ls='--', lw=0.8)
            ax.annotate(e['event'].replace('_', ' '), (e['t'] - t0, 0.1), rotation=90, fontsize=7, va='top')
    ax.invert_yaxis()
    ax.set_xlabel('sim time (s)')
    ax.set_ylabel('depth (m)')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_gif(frame_dir, path, width=480, every=1, fps=4, max_frames=80, crop=None):
    from PIL import Image
    files = sorted(glob.glob(os.path.join(frame_dir, '*.png')))[::every][:max_frames]
    if not files:
        return False
    frames = []
    for f in files:
        im = Image.open(f).convert('RGB')
        if crop:
            im = im.crop(crop)
        im = im.resize((width, int(im.height * width / im.width)))
        frames.append(im.convert('P', palette=Image.ADAPTIVE, colors=128))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=os.path.join(WS, 'docs', 'media'))
    ap.add_argument('--name', required=True)
    ap.add_argument('--arena', default=None, help='arena name/yaml (default: from result.json)')
    ap.add_argument('--title', default=None)
    ap.add_argument('--gif-every', type=int, default=1)
    ap.add_argument('--spectator-crop', default='120,150,760,330',
                    help='x0,y0,x1,y1 crop of the spectator view (zoom on the course); empty = none')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    result = {}
    if os.path.isfile(os.path.join(a.run_dir, 'result.json')):
        result = json.load(open(os.path.join(a.run_dir, 'result.json')))
    arena = arena_objects(a.arena or result.get('arena') or 'default')
    ref = json.load(open(os.path.join(a.run_dir, 'referee.json'))) if os.path.isfile(
        os.path.join(a.run_dir, 'referee.json')) else {'events': []}
    odom = read_odom(os.path.join(a.run_dir, 'bag'))
    title = a.title or (f'"{result.get("prompt")}"' if result.get('prompt') else a.name)
    made = []
    plot_topdown(odom, arena, ref.get('events', []), title, os.path.join(a.out, f'{a.name}_topdown.png'))
    made.append(f'{a.name}_topdown.png')
    motion_start, model = None, 'LLM'
    traces = sorted(glob.glob(os.path.join(a.run_dir, 'trace_*.json')))
    if traces:
        tr = json.load(open(traces[-1]))
        model = tr.get('provider', {}).get('model', model)
        if tr.get('tool_calls'):
            motion_start = tr['tool_calls'][0]['sim_start']
    if odom:
        plot_depth(odom, ref.get('events', []), os.path.join(a.out, f'{a.name}_depth.png'), motion_start,
                   f'LLM planning ({model})')
        made.append(f'{a.name}_depth.png')
    crop = tuple(int(v) for v in a.spectator_crop.split(',')) if a.spectator_crop else None
    for sub, suffix, kw in (('frames', 'onboard', {'every': a.gif_every}),
                            ('frames_spectator', 'spectator', {'every': 2, 'width': 640, 'crop': crop, 'fps': 5})):
        if make_gif(os.path.join(a.run_dir, sub), os.path.join(a.out, f'{a.name}_{suffix}.gif'), **kw):
            made.append(f'{a.name}_{suffix}.gif')
    for m in made:
        print(os.path.join(a.out, m), f'{os.path.getsize(os.path.join(a.out, m)) / 1e3:.0f} kB')
    return 0 if odom else 1


if __name__ == '__main__':
    sys.exit(main())
