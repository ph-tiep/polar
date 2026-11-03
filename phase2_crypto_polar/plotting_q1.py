"""Publication-style measured curves; never substitute upper bounds for rates."""
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                     'axes.titlesize': 11, 'axes.labelsize': 11,
                     'legend.fontsize': 8, 'lines.linewidth': 1.7})


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(Path(out)/f'{name}.png', dpi=220, bbox_inches='tight')
    fig.savefig(Path(out)/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def validate_plot_data(rows):
    if not rows or len({r['ebno_db'] for r in rows}) < 2:
        raise ValueError('A curve needs at least two SNR points. Smoke/single-SNR data cannot make performance curves.')
    for r in rows:
        for rate, count, total in [('ber', 'bit_errors', 'bits'), ('bler', 'block_errors', 'frames')]:
            if r[total] <= 0 or not 0 <= r[count] <= r[total] or not np.isclose(r[rate], r[count]/r[total], rtol=1e-10, atol=1e-14):
                raise ValueError(f'Inconsistent {rate} and error counts for {r["curve"]}')


def line(ax, rows, metric, label, *, color=None, marker='o', linestyle='-', hollow=False):
    """Connect available measured points; NaN gaps preserve zero-rate observations."""
    rows = sorted(rows, key=lambda r: r['ebno_db'])
    if not rows:
        return
    x = np.asarray([r['ebno_db'] for r in rows], dtype=float)
    y = np.asarray([r[metric] for r in rows], dtype=float)
    ax.semilogy(x, np.where(y > 0, y, np.nan), label=label, color=color,
                marker=marker, linestyle=linestyle, markersize=4.5,
                markerfacecolor='none' if hollow else color)
    if not hasattr(ax, '_zero_groups'):
        ax._zero_groups = {}
    zeros = tuple((float(r['ebno_db']), int(r['frames'])) for r in rows if r[metric] == 0)
    if zeros:
        ax._zero_groups.setdefault(zeros, []).append(label)
    ax.set_xlabel(r'$E_b/N_0$ (dB), payload-bit reference')
    ax.set_ylabel(metric.upper())
    ax.grid(True, which='major', alpha=.35)
    ax.grid(True, which='minor', alpha=.15, linestyle=':')
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_minor_formatter(NullFormatter())


def decorate(fig, title, rows, metric):
    """Common axes for fair visual comparison; zeros are text, never fake rates."""
    snrs = sorted({float(r['ebno_db']) for r in rows})
    positive = [float(r[metric]) for r in rows if r[metric] > 0]
    lower = 10 ** math.floor(math.log10(min(positive))) if positive else 1e-3
    if lower >= 1:
        lower = .1
    margin = max(.15, (snrs[-1]-snrs[0])*.025)
    fig.suptitle(title, fontsize=11)
    for ax in fig.axes:
        ax.set(xlim=(snrs[0]-margin, snrs[-1]+margin), ylim=(lower, 1.25))
        if len(snrs) <= 15:
            ax.set_xticks(snrs)
        ax.legend(fontsize=7.5, framealpha=.9)
        notes = []
        for points, labels in getattr(ax, '_zero_groups', {}).items():
            values = ', '.join(f'{x:g}' for x, _ in points)
            frames = sorted({n for _, n in points})
            sample = str(frames[0]) if len(frames) == 1 else 'see CSV'
            notes.append(f'{"; ".join(labels)}: {values} dB ({sample} frames/point)')
        if notes:
            ax.text(0, -.25, 'Zero observed errors; omitted on log axis:\n'+'\n'.join(notes),
                    transform=ax.transAxes, va='top', fontsize=7)


def plot_channel(ch, out):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(ch.g_bob_db, label='Bob: SVD transmit/receive beamforming')
    axes[0].plot(ch.g_eve_db, '--', label='Eve: Bob transmit beam + Eve MRC')
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
    validate_plot_data(rows)
    out = Path(out)
    case = rows[0]
    title = f'N={case["n"]}, K={case["k"]}, R=K/N={case["k"]/case["n"]:g}, payload={case["payload_bits"]} ({c["crc"]})'
    for metric, number in [('ber', 2), ('bler', 3)]:
        bob_rows = [r for r in rows if r['group'] == 'bob']
        fig, axes = plt.subplots(1, len(c['list_sizes']), figsize=(5*len(c['list_sizes']), 5), squeeze=False, sharey=True)
        for ax, size in zip(axes[0], c['list_sizes']):
            for method, style in [('standard', dict(color='tab:blue', marker='o', linestyle='-')),
                                  ('crypto', dict(color='tab:orange', marker='s', linestyle='--', hollow=True))]:
                line(ax, [r for r in rows if r['curve'] == f'{method}_L{size}'], metric, f'{method.title()} L={size}', **style)
            ax.set_title(f'Bob: CA-SCL, L={size}')
        decorate(fig, title, bob_rows, metric)
        save(fig, out, f'Fig{number}_Bob_{metric.upper()}')
        if c['modified']['enabled']:
            fig, ax = plt.subplots(figsize=(9, 5.7))
            sizes = [c['modified']['base_list']]
            if c['modified']['expand_list']:
                sizes.append(min(2*sizes[0], 32))
            names = [f'crypto_L{s}' for s in sorted(set(sizes))] + list(dict.fromkeys(r['curve'] for r in rows if r['group'] == 'modified'))
            for name, color, marker, style in zip(names, ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown'], ['o', 's', '^', 'D', 'v', 'P'], ['-', '--', '-.', ':', '--', '-.']):
                label = f'Crypto CA-SCL L={name.split("L")[1]}' if name.startswith('crypto_L') else name.replace('_', ' ').title()
                line(ax, [r for r in rows if r['curve'] == name], metric, label, color=color, marker=marker, linestyle=style, hollow=True)
            decorate(fig, title+' | decoder comparison', [r for r in rows if r['curve'] in names], metric)
            save(fig, out, f'Decoder_{metric.upper()}')
        for scope in c['mismatch_scopes']:
            fig, ax = plt.subplots(figsize=(10, 6.2))
            bob_name = f'crypto_L{c["eve_list_size"]}'
            selected = [r for r in rows if r['curve'] == bob_name or (r['group'] == 'eve' and r['scope'] == scope and r['fraction'] > 0)]
            line(ax, [r for r in selected if r['curve'] == bob_name], metric, 'Bob: key identical to Alice', color='black', marker='*')
            for i, frac in enumerate(sorted(set(p for p in c['mismatch_fractions'] if p > 0))):
                data = [r for r in selected if r['curve'] == f'eve_{scope}_{frac:g}']
                if data:
                    r = data[0]
                    label = f'Eve: p={100*frac:g}% ({r["flipped_key_bits"]}/{r["eligible_key_bits"]} bits)'
                    line(ax, data, metric, label, color=['black','gray','tab:blue','tab:red'][i % 4],
                         marker=['s','o','v','+'][i % 4], linestyle=['--',':','--','-.'][i % 4], hollow=True)
            decorate(fig, title+f' | {scope} keystream bit flips, CA-SCL L={c["eve_list_size"]}\n'
                     f'Eve SNR: {c["eve_snr"]}; 100% = complement, not an independent random key', selected, metric)
            save(fig, out, f'Fig5_Eve_{scope}_{metric.upper()}')
        fig, ax = plt.subplots(figsize=(10, 5.7))
        correct_name = f'eve_{c["mismatch_scopes"][0]}_0'
        controls = [('eve_standard', 'Eve: Standard CA-SCL'), (correct_name, 'Eve: Crypto, correct key'), ('frozen_only_inverse', 'Eve: frozen-only, hard decision + inverse')]
        for i, (name, label) in enumerate(controls):
            line(ax, [r for r in rows if r['curve'] == name], metric, label, marker=['o','s','^'][i], linestyle=['-','--','-.'][i])
        decorate(fig, title+' | distinct decoding rules', [r for r in rows if r['curve'] in [x[0] for x in controls]], metric)
        save(fig, out, f'Eve_Controls_{metric.upper()}')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(diag['output_bit_flip_probability'])
    axes[0].axhline(.5, linestyle='--', color='gray')
    axes[0].set(xlabel='Codeword bit index', ylabel='Observed flip fraction', ylim=(0, 1), title='One master-key bit changed; paired context')
    axes[1].hist([diag['standard_weights'], diag['crypto_weights']], bins=20, label=['Standard', 'Crypto'])
    axes[1].set(xlabel='Codeword Hamming weight', ylabel='Count', title='Same payload and CRC')
    axes[1].legend()
    save(fig, out, 'Key_Perturbation_and_Weight')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for i, curve in enumerate(dict.fromkeys(r['curve'] for r in rows if r['group'] in ('bob', 'modified'))):
        data = sorted([r for r in rows if r['curve'] == curve], key=lambda r: r['ebno_db'])
        for ax, field in zip(axes, ['batch_amortized_ms', 'payload_goodput']):
            ax.plot([r['ebno_db'] for r in data], [r[field] for r in data],
                    marker=['o','s','^','D','v','P','X'][i % 7], linestyle='--' if i % 2 else '-', label=curve, markersize=4)
    axes[0].set(ylabel='Batch-amortized decoding time (ms/frame)', xlabel='Eb/N0 (dB)')
    axes[1].set(ylabel='Payload goodput (bits/channel use)', xlabel='Eb/N0 (dB)')
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
    save(fig, out, 'Decoding_Cost_and_Payload_Goodput')
    fields = ['curve','ebno_db','frames','bits','bit_errors','block_errors','zero_error_upper95']
    with (out/'zero_error_observations.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows if r['bit_errors'] == 0)
    (out/'FIGURE_NOTES.md').write_text(
        '# Reading the curves\n\nLines connect measured nonzero rates at the sampled SNR values; no smoothing or extrapolation. '
        'Zero observed errors are listed below each plot and in zero_error_observations.csv, not plotted as an error floor. '
        'The conservative zero-error bound is a separate statistic, not a measured BER.\n\n'
        'Fig5 flips bits of the derived keystream. 100% means its complement; it does not mean an independent random master key. '
        'A near-one BER can indicate complemented plaintext and is not a secrecy proof.\n', encoding='utf-8')
