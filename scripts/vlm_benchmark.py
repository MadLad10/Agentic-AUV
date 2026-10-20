#!/usr/bin/env python3
"""Benchmark a real vision-language model (local moondream via Ollama) on labelled simulator frames.

Questions (short yes/no, which small VLMs answer far more reliably than open lists):
  objects    "Is there a red ball ...?" (red/green/yellow buoy) and "... an orange gate ...?"
  condition  "Is this image clear, murky, or completely black?"
Frames: clear-water held-out frames (visibility-aware labels), murky frames (2.5 m visibility, own
labels), and black frames (camera blackout). The YOLO detector is scored on the same frames.

  python3 scripts/vlm_benchmark.py --out eval/results/vlm_benchmark
"""
import argparse
import glob
import json
import os
import random
import statistics
import time

import numpy as np

CLASSES = ['gate', 'red_buoy', 'green_buoy', 'yellow_buoy']
# moondream (Ollama build) returns EMPTY answers to yes/no and counting questions but answers open
# "what" questions, so the benchmark asks open questions and scores them by keyword matching.
Q_COLOURS = 'What colors are the balls in this image?'
Q_OBJECTS = 'What objects are in this image?'
Q_DESCRIBE = 'Describe this image.'
GATE_WORDS = ('gate', 'arch', 'frame', 'goal', 'post', 'pole', 'doorway', 'entrance')
DARK_WORDS = ('black', 'dark')


def labels(path):
    present = set()
    if os.path.isfile(path):
        for line in open(path):
            if line.strip():
                present.add(CLASSES[int(line.split()[0])])
    return present


def frames(a):
    from PIL import Image
    rnd = random.Random(0)
    out = []
    clear = sorted(glob.glob('data/synthetic/images/val/*.jpg'))
    for f in rnd.sample(clear, a.clear):
        out.append(('clear', np.array(Image.open(f).convert('RGB')), labels(f.replace('images', 'labels')[:-4] + '.txt')))
    murky = sorted(glob.glob('data/vlm_bench/murky/images/val/*.jpg'))
    for f in rnd.sample(murky, min(a.murky, len(murky))):
        out.append(('murky', np.array(Image.open(f).convert('RGB')), labels(f.replace('images', 'labels')[:-4] + '.txt')))
    for _ in range(a.black):
        out.append(('black', np.zeros((480, 640, 3), np.uint8), set()))
    return out


def yolo_presence(imgs, weights):
    import subprocess
    import sys
    np.save('/tmp/_vlm_bench_imgs.npy', np.stack(imgs))
    code = r'''
import json, sys, numpy as np
from ultralytics import YOLO
m = YOLO(sys.argv[1]); imgs = np.load('/tmp/_vlm_bench_imgs.npy'); out = []
for im in imgs:
    r = m.predict(np.ascontiguousarray(im[..., ::-1]), imgsz=320, conf=0.35, verbose=False)[0]
    out.append(sorted({r.names[int(k)] for k in r.boxes.cls.cpu().numpy()}))
print('RESULT ' + json.dumps(out))
'''
    p = subprocess.run([sys.executable, '-c', code, weights], capture_output=True, text=True, timeout=900)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith('RESULT ')]
    return [set(x) for x in json.loads(line[-1][7:])] if line else None


def prf(pred, truth, cls):
    tp = sum(cls in p and cls in t for p, t in zip(pred, truth))
    fp = sum(cls in p and cls not in t for p, t in zip(pred, truth))
    fn = sum(cls not in p and cls in t for p, t in zip(pred, truth))
    return {'precision': round(tp / (tp + fp), 3) if tp + fp else None,
            'recall': round(tp / (tp + fn), 3) if tp + fn else None, 'support': tp + fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clear', type=int, default=40)
    ap.add_argument('--murky', type=int, default=10)
    ap.add_argument('--black', type=int, default=10)
    ap.add_argument('--model', default='moondream')
    ap.add_argument('--host', default='http://127.0.0.1:11437')
    ap.add_argument('--out', default='eval/results/vlm_benchmark')
    a = ap.parse_args()
    from auv_agents.vlm import OllamaVLM
    os.makedirs(a.out, exist_ok=True)
    vlm = OllamaVLM(a.model, 180, a.host)
    vlm.describe(np.zeros((64, 64, 3), np.uint8), 'warm up')          # model load not charged to frames
    fr = frames(a)
    rows, lat = [], []
    for i, (cond, img, truth) in enumerate(fr):
        answers = {}
        for key, q in (('colours', Q_COLOURS), ('objects', Q_OBJECTS), ('describe', Q_DESCRIBE)):
            t0 = time.time()
            answers[key] = (vlm.describe(img, q) or '').strip()
            lat.append(time.time() - t0)
        col = answers['colours'].lower()
        negated = any(w in col for w in ('no ball', 'no balls', 'there are no', 'not any'))
        pred = set() if negated else {f'{c}_buoy' for c in ('red', 'green', 'yellow') if c in col}
        if any(w in answers['objects'].lower() for w in GATE_WORDS):
            pred.add('gate')
        c_pred = 'black' if any(w in answers['describe'].lower() for w in DARK_WORDS) else 'not_black'
        rows.append({'i': i, 'condition': cond, 'truth': sorted(truth), 'vlm': sorted(pred), 'answers': answers,
                     'condition_pred': c_pred})
        print(json.dumps(rows[-1]), flush=True)
    yolo = yolo_presence([img for _, img, _ in fr], 'data/models/auv_yolov8n_synth.pt')
    truth = [set(r['truth']) for r in rows]
    vpred = [set(r['vlm']) for r in rows]
    res = {'model': a.model, 'frames': {c: sum(r['condition'] == c for r in rows) for c in ('clear', 'murky', 'black')},
           'vlm_objects': {c: prf(vpred, truth, c) for c in CLASSES},
           'yolo_objects': {c: prf(yolo, truth, c) for c in CLASSES} if yolo else None,
           'blackout_detection': {
               'black_frames_called_dark': round(statistics.mean(r['condition_pred'] == 'black' for r in rows if r['condition'] == 'black'), 3) if any(r['condition'] == 'black' for r in rows) else None,
               'normal_frames_called_dark': round(statistics.mean(r['condition_pred'] == 'black' for r in rows if r['condition'] != 'black'), 3)},
           'vlm_empty_answers': sum(1 for r in rows for v in r['answers'].values() if not (v or '').strip()),
           'latency_s': {'median': round(statistics.median(lat), 2), 'p90': round(sorted(lat)[int(0.9 * len(lat))], 2),
                         'mean': round(statistics.mean(lat), 2), 'calls': len(lat)}}
    json.dump({'summary': res, 'rows': rows}, open(os.path.join(a.out, 'results.json'), 'w'), indent=1, default=list)
    L = [f'# VLM benchmark: {a.model} (local, CPU) vs YOLOv8n on the same simulator frames', '',
         f'Frames: {res["frames"]}. Open questions scored by keyword matching (moondream returns empty answers to '
         'yes/no questions); YOLO = our detector trained from scratch (conf 0.35). Raw answers in results.json.', '',
         '| Object | VLM precision | VLM recall | YOLO precision | YOLO recall | support |', '|---|---|---|---|---|---|']
    for c in CLASSES:
        v, y = res['vlm_objects'][c], (res['yolo_objects'] or {}).get(c, {})
        L.append(f'| {c} | {v["precision"]} | {v["recall"]} | {y.get("precision")} | {y.get("recall")} | {v["support"]} |')
    L += ['', f'Blackout detection ("black"/"dark" in the description): {res["blackout_detection"]}',
          f'Empty VLM answers: {res["vlm_empty_answers"]} of {3 * len(rows)} questions',
          f'Latency per VLM call: median {res["latency_s"]["median"]} s, p90 {res["latency_s"]["p90"]} s '
          f'({res["latency_s"]["calls"]} calls, CPU, machine shared with other jobs)']
    open(os.path.join(a.out, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
