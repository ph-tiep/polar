"""Plots of measured values only; zeros shown as conservative upper bounds."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out/f'{name}.png', dpi=220)
    fig.savefig(out/f'{name}.pdf')
    plt.close(fig)


def line(ax, rows, metric, label):
    rows = sorted(rows, key=lambda r: r['ebno_db'])
    if not rows:
        return
    x = np.array([r['ebno_db'] for r in rows])
    y = np.array([r[metric] for r in rows])
    positive = y > 0
    plotted = ax.semilogy(x, np.where(positive, y, np.nan), 'o-', label=label)[0]
    if (~positive).any():
        bound = [r['zero_error_upper95'] for r in rows if r[metric] == 0]
        ax.scatter(x[~positive], bound, marker='v', color=plotted.get_color())
    ax.set_xlabel(r'$E_b/N_0$ (dB), payload-bit reference')
    ax.set_ylabel(metric.upper())
    ax.grid(True, which='both', alpha=.3)


def decorate(fig, title):
    fig.suptitle(title+'\nTriangles: zero observed errors, conservative 95% upper bound', fontsize=11)
    for ax in fig.axes:
        ax.legend(fontsize=7)


def plot_channel(ch, out):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(ch.g_bob_db, label='Bob: SVD transmit/receive beamforming')
    axes[0].plot(ch.g_eve_db, label='Eve: Bob transmit beam + Eve MRC')
    axes[0].set_ylabel('Effective power gain (dB)')
    axes[0].legend(fontsize=8)
    axes[0].set_title(f'Fig1: raw ray-traced CSI, Tx={ch.n_tx}, Rx={ch.n_rx}, carriers={ch.n_sc}')
    axes[1].plot(ch.spatial_corr, label='Absolute inner product of dominant TX vectors')
    axes[1].axhline(ch.spatial_corr.mean(), linestyle=':', label=f'Mean={ch.spatial_corr.mean():.4f}')
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Spatial correlation')
    axes[1].set_xlabel('Subcarrier index')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.3)
    save(fig, out, 'Fig1_RayTraced_Channel')


def plot_case(rows, diag, out, c):
    case = rows[0]
    title = f'N={case["n"]}, K={case["k"]} (incl. CRC), payload={case["payload_bits"]}'
    for metric, number in [('ber', 2), ('bler', 3)]:
        fig, axes = plt.subplots(1, len(c['list_sizes']), figsize=(5*len(c['list_sizes']), 4), squeeze=False)
        for ax, size in zip(axes[0], c['list_sizes']):
            for method in ('standard', 'crypto'):
                line(ax, [r for r in rows if r['curve'] == f'{method}_L{size}'], metric, f'{method} CA-SCL L={size}')
            ax.set_title(f'Bob: same decoder, L={size}')
        decorate(fig, title)
        save(fig, out, f'Fig{number}_Bob_{metric.upper()}')
        if c['modified']['enabled']:
            fig, ax = plt.subplots(figsize=(8, 5))
            for size in sorted(set([c['modified']['base_list'], min(2*c['modified']['base_list'], 32)])):
                line(ax, [r for r in rows if r['curve'] == f'crypto_L{size}'], metric, f'Crypto CA-SCL L={size}')
            line(ax, [r for r in rows if r['curve'] == 'modified'], metric, 'Crypto Modified CA-SCL (adaptive budget)')
            decorate(fig, title)
            save(fig, out, f'Decoder_{metric.upper()}')
        for scope in c['mismatch_scopes']:
            fig, ax = plt.subplots(figsize=(9, 5))
            line(ax, [r for r in rows if r['curve'] == f'crypto_L{c["eve_list_size"]}'], metric, 'Bob: correct key')
            for frac in sorted(set([0.] + c['mismatch_fractions'])):
                data = [r for r in rows if r['curve'] == f'eve_{scope}_{frac:g}']
                if data:
                    r = data[0]
                    line(ax, data, metric, f'Eve: requested {100*frac:g}%, actual {r["flipped_key_bits"]}/{r["eligible_key_bits"]} bits')
            decorate(fig, title+f' | {scope} derived-key mismatch; Eve={c["eve_snr"]}')
            save(fig, out, f'Fig5_Eve_{scope}_{metric.upper()}')
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, label in [('eve_standard', 'Eve: Standard CA-SCL'), ('eve_full_0', 'Eve: Crypto CA-SCL, correct key'), ('frozen_only_inverse', 'Eve: frozen-only, hard decision + inverse transform')]:
            line(ax, [r for r in rows if r['curve'] == name], metric, label)
        decorate(fig, title+' | controls, different decoding rules explicitly labeled')
        save(fig, out, f'Eve_Controls_{metric.upper()}')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(diag['output_bit_flip_probability'])
    axes[0].set(xlabel='Codeword bit index', ylabel='Observed flip fraction', ylim=(0, 1), title='One master-key bit changed; paired context')
    axes[1].hist([diag['standard_weights'], diag['crypto_weights']], bins=20, label=['Standard', 'Crypto'])
    axes[1].set(xlabel='Codeword Hamming weight', ylabel='Count', title='Same payload and CRC')
    axes[1].legend()
    save(fig, out, 'Key_Perturbation_and_Weight')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for curve in dict.fromkeys(r['curve'] for r in rows if r['group'] in ('bob', 'modified')):
        data = sorted([r for r in rows if r['curve'] == curve], key=lambda r: r['ebno_db'])
        axes[0].plot([r['ebno_db'] for r in data], [r['batch_amortized_ms'] for r in data], 'o-', label=curve)
        axes[1].plot([r['ebno_db'] for r in data], [r['payload_goodput'] for r in data], 'o-', label=curve)
    axes[0].set(ylabel='Batch-amortized decoding time (ms/frame)', xlabel='Eb/N0 (dB)')
    axes[1].set(ylabel='Payload goodput (bits/channel use)', xlabel='Eb/N0 (dB)')
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
    save(fig, out, 'Decoding_Cost_and_Payload_Goodput')
