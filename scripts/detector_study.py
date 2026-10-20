#!/usr/bin/env python3
"""Detector study: architecture (YOLOv8n / YOLO11n / YOLO12n) and physics-informed turbidity
augmentation, all trained FROM SCRATCH on simulator-auto-labelled images (no pretrained weights).

Evaluation:
  - held-out scenes of the base synthetic set (data/synthetic, val split = unseen layouts)
  - robustness curve: the same scenes/poses rendered at fixed visibility 2..12 m, labelled with the
    clear-water labels (data/vis_test/README.md), so recall shows how far each detector still sees.

Resumable: a configuration whose weights already exist is not retrained.

  setsid nohup .venv/bin/python scripts/detector_study.py > eval/results/detector_study/run.log 2>&1 &
"""
import argparse
import json
import os
import shutil
import time

os.environ.setdefault('YOLO_OFFLINE', '1')

CONFIGS = [  # name, architecture yaml, training data
    ('yolov8n_base', 'yolov8n.yaml', 'data/synthetic/data.yaml'),
    ('yolo11n_base', 'yolo11n.yaml', 'data/synthetic/data.yaml'),
    ('yolo12n_base', 'yolo12n.yaml', 'data/synthetic/data.yaml'),
    ('yolov8n_turbid', 'yolov8n.yaml', 'data/synthetic_turbid/data.yaml'),
]
VIS = [2, 3, 4, 6, 9, 12]


def train(name, arch, data, a):
    from ultralytics import YOLO
    dst = os.path.join(a.models, f'{name}.pt')
    if os.path.isfile(dst):
        return dst, None
    if name == 'yolov8n_base' and os.path.isfile(os.path.join(a.models, 'auv_yolov8n_synth.pt')):
        shutil.copy(os.path.join(a.models, 'auv_yolov8n_synth.pt'), dst)     # the M4 model is this config
        return dst, None
    m = YOLO(arch)
    t0 = time.time()
    m.train(data=data, epochs=a.epochs, imgsz=a.imgsz, batch=32, device='cpu', workers=4, plots=False, amp=False,
            pretrained=False, project='runs_yolo', name=name, exist_ok=True, patience=15, verbose=False, seed=0,
            deterministic=True, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, fliplr=0.5, mosaic=1.0, close_mosaic=5)
    shutil.copy(str(m.trainer.best), dst)
    return dst, round(time.time() - t0)


def evaluate(weights, data):
    from ultralytics import YOLO
    m = YOLO(weights).val(data=data, imgsz=320, batch=32, device='cpu', plots=False, verbose=False,
                          project='runs_yolo', name='study_val', exist_ok=True)
    return {'mAP50': round(float(m.box.map50), 4), 'mAP50_95': round(float(m.box.map), 4),
            'precision': round(float(m.box.mp), 4), 'recall': round(float(m.box.mr), 4),
            'ms_per_img': round(m.speed['inference'], 2)}


def params(weights):
    from ultralytics import YOLO
    return round(sum(p.numel() for p in YOLO(weights).model.parameters()) / 1e6, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--models', default='data/models')
    ap.add_argument('--out', default='eval/results/detector_study')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, 'results.json')
    results = json.load(open(res_path)) if os.path.isfile(res_path) else {}
    for name, arch, data in CONFIGS:
        print(f'[study] {name}: training {arch} on {data}', flush=True)
        w, secs = train(name, arch, data, a)
        r = results.get(name, {})
        r.update(arch=arch, train_data=data, params_M=params(w))
        if secs is not None:
            r['train_wall_s'] = secs
        r['held_out_scenes'] = evaluate(w, 'data/synthetic/data.yaml')
        r['visibility'] = {str(v): evaluate(w, f'data/vis_test/vis_{v}/data.yaml') for v in VIS}
        results[name] = r
        json.dump(results, open(res_path, 'w'), indent=1)
        print(f'[study] {name}: {json.dumps(r)}', flush=True)
    lines = ['# Detector study (all trained from scratch on synthetic, auto-labelled frames; CPU)', '',
             '| Model | Params (M) | Train data | mAP@0.5 | mAP@0.5:0.95 | CPU ms/img | ' +
             ' | '.join(f'recall @ {v} m' for v in VIS) + ' |',
             '|---|---|---|---|---|---|' + '---|' * len(VIS)]
    for name, r in results.items():
        h = r['held_out_scenes']
        lines.append(f'| {name} | {r["params_M"]} | {os.path.basename(os.path.dirname(r["train_data"]))} | '
                     f'{h["mAP50"]:.3f} | {h["mAP50_95"]:.3f} | {h["ms_per_img"]} | ' +
                     ' | '.join(f'{r["visibility"][str(v)]["recall"]:.2f}' for v in VIS) + ' |')
    lines += ['', 'Recall @ visibility: same scenes and poses at each water visibility, labelled with the clear-water '
              'labels (objects visible at 12 m), so lower visibility hides far objects for every model alike.']
    open(os.path.join(a.out, 'summary.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
