#!/usr/bin/env python3
"""Train a YOLOv8n detector FROM SCRATCH (random init; no pretrained weights are downloaded) on the
synthetic, simulator-auto-labelled dataset, then evaluate mAP on the held-out scenes.

  setsid nohup .venv/bin/python scripts/train_yolo.py --epochs 40 > data/models/train.log 2>&1 &
"""
import argparse
import json
import os
import shutil
import time

os.environ.setdefault('YOLO_OFFLINE', '1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/synthetic/data.yaml')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--name', default='synth_v1')
    ap.add_argument('--out', default='data/models')
    ap.add_argument('--weights-name', default='auv_yolov8n_synth.pt')
    ap.add_argument('--arch', default='yolov8n.yaml')
    ap.add_argument('--eval-name', default='yolo_eval.json')
    ap.add_argument('--eval-only', default=None, help='evaluate these weights on the val split, no training')
    a = ap.parse_args()
    if a.eval_only:
        dst = os.path.join(a.out, a.weights_name)
        if os.path.abspath(a.eval_only) != os.path.abspath(dst):
            shutil.copy(a.eval_only, dst)
        return evaluate(dst, a)
    import torch
    torch.set_num_threads(a.threads)
    from ultralytics import YOLO
    os.makedirs(a.out, exist_ok=True)
    model = YOLO(a.arch)                         # architecture only: random init, nothing downloaded
    t0 = time.time()
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device='cpu', workers=4,
                plots=False, amp=False, pretrained=False, project='runs_yolo', name=a.name, exist_ok=True,
                patience=15, verbose=True, seed=0, deterministic=True,
                hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, fliplr=0.5, mosaic=1.0, close_mosaic=5)
    train_s = time.time() - t0
    best = str(model.trainer.best)               # ultralytics decides the run dir (runs/detect/...)
    dst = os.path.join(a.out, a.weights_name)
    shutil.copy(best, dst)
    evaluate(dst, a, train_s)


def evaluate(dst, a, train_s=None):
    from ultralytics import YOLO
    m = YOLO(dst).val(data=a.data, imgsz=a.imgsz, batch=a.batch, device='cpu', plots=False, split='val',
                      project='runs_yolo', name=a.name + '_val', exist_ok=True)
    names = m.names
    per_class = {names[int(c)]: round(float(m.box.maps[int(c)]), 4) for c in range(len(names))}
    res = {'weights': dst, 'epochs': a.epochs, 'imgsz': a.imgsz,
           'train_wall_s': round(train_s) if train_s else None,
           'mAP50': round(float(m.box.map50), 4), 'mAP50_95': round(float(m.box.map), 4),
           'precision': round(float(m.box.mp), 4), 'recall': round(float(m.box.mr), 4),
           'per_class_mAP50_95': per_class,
           'speed_ms_per_img': {k: round(v, 2) for k, v in m.speed.items()}}
    with open(os.path.join(a.out, a.eval_name), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
