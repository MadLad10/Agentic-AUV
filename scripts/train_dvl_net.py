#!/usr/bin/env python3
"""Train the learned DVL-outage velocity estimator (GRU over a 2 s window of IMU + thrust history).

Literature: BeamsNet / Set-transformer DVL forecasting (learned velocity during DVL outages). Here the
vehicle's own thruster commands are inputs too: velocity is mostly thrust vs. drag, so the network
learns the vehicle's dynamics instead of integrating noisy accelerations.

Exports pure-numpy weights (data/models/dvl_net.npz) for onboard inference without torch, plus
validation metrics against two baselines (zero velocity and IMU integration).

  python3 scripts/train_dvl_net.py --epochs 20
"""
import argparse
import json
import time

import numpy as np

WIN = 40          # 2 s at 20 Hz


def windows(X, Y, ep, stride):
    idx = []
    for e in np.unique(ep):
        ii = np.nonzero(ep == e)[0]
        for k in range(WIN, len(ii), stride):
            idx.append(ii[k - WIN:k])
    idx = np.array(idx)
    return X[idx], Y[idx[:, -1]]


def imu_integration_baseline(X, Y, ep, horizon=200):
    """Dead-reckoned body velocity from accelerometer integration, restarted from truth every
    `horizon` samples (a 10 s outage), gravity removed with the AHRS roll/pitch."""
    errs = []
    for e in np.unique(ep):
        ii = np.nonzero(ep == e)[0]
        for s0 in range(0, len(ii) - horizon, horizon):
            v = Y[ii[s0]].astype(np.float64).copy()
            for k in ii[s0 + 1:s0 + horizon]:
                ax, ay, az = X[k, 3:6]
                r, p = X[k, 6], X[k, 7]
                g = 9.81 * np.array([-np.sin(p), np.cos(p) * np.sin(r), np.cos(p) * np.cos(r)])
                v += (np.array([ax, ay, az]) - g) * 0.05
                errs.append(v - Y[k])
    errs = np.array(errs)
    return np.sqrt((errs ** 2).mean(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/nav/train.npz')
    ap.add_argument('--val', default='data/nav/val.npz')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--out', default='data/models/dvl_net.npz')
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    tr, va = np.load(a.train), np.load(a.val)
    mu, sd = tr['X'].mean(0), tr['X'].std(0) + 1e-6
    Xtr, Ytr = windows((tr['X'] - mu) / sd, tr['Y'], tr['episode'], 2)
    Xva, Yva = windows((va['X'] - mu) / sd, va['Y'], va['episode'], 5)
    print(f'train windows {len(Xtr)}, val windows {len(Xva)}', flush=True)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(Xtr.shape[2], a.hidden, batch_first=True)
            self.head = torch.nn.Linear(a.hidden, 3)

        def forward(self, x):
            _, h = self.gru(x)
            return self.head(h[-1])
    net = Net()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    xt, yt = torch.tensor(Xtr), torch.tensor(Ytr)
    xv, yv = torch.tensor(Xva), torch.tensor(Yva)
    t0 = time.time()
    for epoch in range(a.epochs):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = perm[i:i + 256]
            loss = torch.nn.functional.mse_loss(net(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            rmse = torch.sqrt(((net(xv) - yv) ** 2).mean(0)).numpy()
        print(f'epoch {epoch + 1}: val RMSE u/v/w = {np.round(rmse, 4).tolist()}', flush=True)
    sd_ = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    np.savez(a.out, mu=mu, sd=sd, win=WIN, w_ih=sd_['gru.weight_ih_l0'], w_hh=sd_['gru.weight_hh_l0'],
             b_ih=sd_['gru.bias_ih_l0'], b_hh=sd_['gru.bias_hh_l0'], w_out=sd_['head.weight'], b_out=sd_['head.bias'],
             val_rmse=rmse)
    base_zero = np.sqrt((Yva ** 2).mean(0))
    base_imu = imu_integration_baseline(va['X'], va['Y'], va['episode'])
    res = {'val_rmse_mps': {'gru': np.round(rmse, 4).tolist(), 'zero_velocity': np.round(base_zero, 4).tolist(),
                            'imu_integration_10s': np.round(base_imu, 4).tolist()},
           'train_windows': len(Xtr), 'val_windows': len(Xva), 'hidden': a.hidden, 'epochs': a.epochs,
           'train_wall_s': round(time.time() - t0), 'params': int(sum(v.size for v in sd_.values()))}
    json.dump(res, open(a.out.replace('.npz', '_eval.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
