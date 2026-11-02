"""Validate a Phase 1 export and generate review plots without rerunning RT."""
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
args = parser.parse_args()
root = args.output.resolve()
data = root / 'dataset/140GHz'
figures = root / 'figures'
manifest = json.loads((root/'RUN_CONFIG.json').read_text())
assert manifest['status'] == 'completed', 'Run did not complete'
a, tau = (np.load(data/f) for f in ['CIR_Gain.npy', 'CIR_Delay.npy'])
metrics = json.loads((root/'logs/validation.json').read_text())
checks = {}
meta = np.load(data/'Metadata.npz', allow_pickle=False)
nsc = int(meta['NumSubcarriers'])
shape = (int(meta['NumRxAntennas']), int(meta['NumTxAntennas']), nsc)
f = (np.arange(nsc)-nsc//2)*float(meta['SubcarrierSpacingMHz'])*1e6
mat = sio.loadmat(data/'CSI_Dataset_140GHz.mat')
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, name in enumerate(['Bob','Eve']):
    h = np.load(data/f'H_{name}.npy')
    assert h.shape == shape and np.isfinite(h).all() and np.any(h != 0)
    np.testing.assert_array_equal(h, mat[f'H_{name}'])
    # Independent Fourier sum for one antenna pair, including absolute delay.
    valid = tau[i,0] >= 0
    expected = (a[i,0,0,0,valid,None] * np.exp(-2j*np.pi*tau[i,0,valid,None]*f)).sum(axis=0)
    error = float(np.max(np.abs(h[0,0]-expected)) / np.max(np.abs(expected)))
    assert error < 0.002, (name, error)
    np.testing.assert_allclose(h[:,:,nsc//2], a[i,0].sum(axis=-1), rtol=2e-5, atol=1e-12)
    checks[name] = dict(shape=list(h.shape), fourier_relative_max_error=error, matlab_matches_numpy=True)
    power = np.mean(np.abs(a[i,0])**2,axis=(0,1))
    mask = valid & (power>0)
    axes[0].scatter(tau[i,0,mask]*1e9,10*np.log10(power[mask]),label=name,s=20)
    axes[1].plot(f/1e9,10*np.log10(np.mean(np.abs(h)**2,axis=(0,1))),label=name,lw=1)
axes[0].set(xlabel='Absolute delay (ns)',ylabel='Antenna-averaged path power (dB)',title='140 GHz CIR: Bob and Eve')
axes[1].set(xlabel='Baseband frequency offset (GHz)',ylabel='Mean channel power (dB)',title='Unnormalized CSI over 2 GHz bandwidth')
for ax in axes:
    ax.grid(alpha=.25);ax.legend()
fig.tight_layout();fig.savefig(figures/'Bob_Eve_Channel_Review.png',dpi=180);fig.savefig(figures/'Bob_Eve_Channel_Review.pdf');plt.close(fig)
(root/'logs/export_checks.json').write_text(json.dumps(checks,indent=2))
summary = f"""# Phase 1: kiểm tra dữ liệu đã xuất

- Trạng thái: {manifest['status']}; mode: {manifest['mode']}.
- Layout: {manifest['layout']}; BS: {manifest['BS']}; Bob: {manifest['Bob']}; Eve: {manifest['Eve']}.
- Seed: {manifest['seed']}; mẫu/nguồn: {manifest['samples_per_src']}.
- CSI [Rx, Tx, subcarrier]: {shape}.
- Tần số: {float(meta['CarrierFrequencyGHz'])} GHz; băng thông: {float(meta['BandwidthGHz'])} GHz.
- MATLAB khớp NumPy; tổng Fourier CIR khớp CSI trên cặp anten được kiểm tra.
- Hình: figures/Bob_Eve_Channel_Review.png; thống kê: logs/validation.json.
- Sai số kiểm tra: logs/export_checks.json. Đây chưa phải đánh giá BER/bảo mật.
"""
(root/'REVIEW.md').write_text(summary, encoding='utf-8')
print(json.dumps(checks, indent=2))
