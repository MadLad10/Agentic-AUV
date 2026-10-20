"""Factor-graph navigation (GTSAM) closed loop, and the numpy GRU against PyTorch's GRU."""
import subprocess
import sys

import numpy as np
import pytest

from auv_onboard.factory import make_backend, make_registry

STEPS = [('navigate_to', {'x': 2.0, 'y': 0.0, 'depth': 1.5}, 60), ('navigate_to', {'target': 'gate'}, 120),
         ('surface', {}, 90)]


def test_factor_graph_navigation_closed_loop():
    pytest.importorskip('gtsam')
    be = make_backend('default', seed=0, onboard_overrides={'backend': {'max_rtf': None, 'idle_rtf': None},
                                                            'ekf': {'estimator': 'fgo'}})
    try:
        reg, _ = make_registry(be)
        assert all(reg.call(n, p, t).ok for n, p, t in STEPS)
        r = be.sim.referee.report()
        assert r['gate_passed'] and r['surfaced_at_end'] and r['collisions'] == 0
        assert r['localisation']['rmse_xy_m'] < 0.4 and r['localisation']['rmse_depth_m'] < 0.06
        s = be.ekf.summary()
        assert s['estimator'] == 'factor_graph_isam2' and s['keyframes'] > 50 and s['updates']['dvl']['ok'] > 50
    finally:
        be.shutdown()


GRU_CHECK = r'''
import sys, numpy as np, torch
torch.manual_seed(0)
gru, head = torch.nn.GRU(15, 8, batch_first=True), torch.nn.Linear(8, 3)
sd = {k: v.detach().numpy() for k, v in gru.state_dict().items()}
np.savez(sys.argv[1], mu=np.zeros(15), sd=np.ones(15), win=5, w_ih=sd['weight_ih_l0'], w_hh=sd['weight_hh_l0'],
         b_ih=sd['bias_ih_l0'], b_hh=sd['bias_hh_l0'], w_out=head.weight.detach().numpy(),
         b_out=head.bias.detach().numpy(), val_rmse=np.array([0.02, 0.02, 0.02]))
x = torch.randn(1, 5, 15)
_, h = gru(x)
np.save(sys.argv[1] + '.x.npy', x.numpy()[0])
np.save(sys.argv[1] + '.y.npy', head(h[-1]).detach().numpy()[0])
'''


def test_numpy_gru_matches_torch(tmp_path):
    pytest.importorskip('torch')
    w = str(tmp_path / 'net.npz')
    p = subprocess.run([sys.executable, '-c', GRU_CHECK, w], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-1500:]
    from auv_onboard.learned_velocity import LearnedVelocity
    lv = LearnedVelocity(w)
    for row in np.load(w + '.x.npy'):
        lv.buf.append(row)
    assert np.allclose(lv.predict(), np.load(w + '.y.npy'), atol=1e-5)
