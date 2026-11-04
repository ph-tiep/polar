"""Matched-decoder construction plots; measurements are never smoothed."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from .construction_experiments import curve_name
from .plotting_q1 import line, decorate, save

LABELS = {'fixed': 'Fixed ranking', 'dt': 'DT construction', 'robust': 'Robust DT'}
STYLES = {
    ('fixed', 'standard'): dict(color='gray', marker='v', linestyle='--', hollow=True),
    ('fixed', 'crypto'): dict(color='tab:blue', marker='o', linestyle='-'),
    ('dt', 'standard'): dict(color='tab:orange', marker='^', linestyle='--', hollow=True),
    ('dt', 'crypto'): dict(color='#003366', marker='*', linestyle='-'),
    ('robust', 'standard'): dict(color='tab:purple', marker='s', linestyle='-.', hollow=True),
    ('robust', 'crypto'): dict(color='tab:green', marker='D', linestyle='-'),
}


def plot_construction_case(rows, diag, out, c):
    out = Path(out)
    case = rows[0]
    methods = c['construction']['methods']
    title = f'N={case["n"]}, K={case["k"]}, R=K/N={case["k"]/case["n"]:g}, payload={case["payload_bits"]} ({c["crc"]})'
    by_name = {}
    for row in rows:
        by_name.setdefault(row['curve'], []).append(row)

    def draw_bob(ax, size, metric, crypto_only=False):
        selected = []
        for method in methods:
            for coding in (('crypto',) if crypto_only else ('standard', 'crypto')):
                name = curve_name(method, f'{coding}_L{size}')
                data = by_name[name]
                selected.extend(data)
                line(ax, data, metric, f'{LABELS[method]}: {coding.title()}', **STYLES[(method, coding)])
        ax.set_title(f'Bob: CA-SCL L={size}')
        return selected

    for metric, number in [('ber', 2), ('bler', 3)]:
        fig, axes = plt.subplots(1, len(c['list_sizes']), figsize=(6*len(c['list_sizes']), 5.8), squeeze=False, sharey=True)
        selected = []
        for ax, size in zip(axes[0], c['list_sizes']):
            selected.extend(draw_bob(ax, size, metric))
        decorate(fig, title, selected, metric)
        save(fig, out, f'Fig{number}_Bob_{metric.upper()}')
        # Standalone panels avoid tiny labels when pasted into the report.
        for size in c['list_sizes']:
            fig, ax = plt.subplots(figsize=(9, 6.2))
            selected = draw_bob(ax, size, metric)
            decorate(fig, title, selected, metric)
            save(fig, out, f'Fig{number}_Bob_L{size}_{metric.upper()}')
        fig, axes = plt.subplots(1, len(c['list_sizes']), figsize=(6*len(c['list_sizes']), 5.8), squeeze=False, sharey=True)
        selected = []
        for ax, size in zip(axes[0], c['list_sizes']):
            selected.extend(draw_bob(ax, size, metric, crypto_only=True))
        decorate(fig, title+' | Crypto-Polar construction comparison', selected, metric)
        save(fig, out, f'Construction_{metric.upper()}')
        for method in methods:
            for scope in c['mismatch_scopes']:
                fig, ax = plt.subplots(figsize=(10, 6.5))
                data = by_name[curve_name(method, f'crypto_L{c["eve_list_size"]}')]
                selected = list(data)
                line(ax, data, metric, 'Bob: key identical to Alice', color='#003366', marker='*')
                for i, fraction in enumerate(sorted(set(p for p in c['mismatch_fractions'] if p > 0))):
                    data = by_name[curve_name(method, f'eve_{scope}_{fraction:g}')]
                    selected.extend(data)
                    r = data[0]
                    label = f'Eve: p={100*fraction:g}% ({r["flipped_key_bits"]}/{r["eligible_key_bits"]} key bits)'
                    line(ax, data, metric, label, color=['black','gray','tab:blue','tab:red'][i % 4],
                         marker=['s','o','v','+'][i % 4], linestyle=['--',':','--','-.'][i % 4], hollow=True)
                decorate(fig, title+f' | {LABELS[method]}, CA-SCL L={c["eve_list_size"]}\n'
                         f'{scope} keystream mismatch; Eve SNR: {c["eve_snr"]}', selected, metric)
                name = f'Fig5_Eve_{scope}_{metric.upper()}' if method == 'dt' else f'Fig5_Eve_{method}_{scope}_{metric.upper()}'
                save(fig, out, name)
            fig, ax = plt.subplots(figsize=(10, 6.2))
            controls = [('eve_standard', 'Eve: Standard CA-SCL'),
                        (f'eve_{c["mismatch_scopes"][0]}_0', 'Eve: Crypto, correct key'),
                        ('frozen_only_inverse', 'Eve: frozen-only, hard decision + inverse'),
                        ('eve_full_1_inverted', 'Eve: full 100%, output bits inverted')]
            selected = []
            for i, (suffix, label) in enumerate(controls):
                data = by_name.get(curve_name(method, suffix), [])
                if data:
                    selected.extend(data)
                    line(ax, data, metric, label, marker=['o','s','^','+'][i], linestyle=['-','--','-.',':'][i])
            decorate(fig, title+f' | {LABELS[method]}: separate attack controls', selected, metric)
            save(fig, out, f'Eve_Controls_{method}_{metric.upper()}')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(diag['output_bit_flip_probability'])
    axes[0].axhline(.5, linestyle='--', color='gray')
    axes[0].set(xlabel='Codeword bit index', ylabel='Observed flip fraction', ylim=(0, 1), title='One shared-key bit changed; DT group 0')
    axes[1].hist([diag['standard_weights'], diag['crypto_weights']], bins=20, label=['Standard', 'Crypto'])
    axes[1].set(xlabel='Codeword Hamming weight', ylabel='Count', title='Same payload and CRC')
    axes[1].legend()
    save(fig, out, 'Key_Perturbation_and_Weight')
    for size in c['list_sizes']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for method in methods:
            for coding in ('standard', 'crypto'):
                data = sorted(by_name[curve_name(method, f'{coding}_L{size}')], key=lambda r: r['ebno_db'])
                style = {k:v for k,v in STYLES[(method, coding)].items() if k != 'hollow'}
                for ax, field in zip(axes, ['batch_amortized_ms', 'payload_goodput']):
                    ax.plot([r['ebno_db'] for r in data], [r[field] for r in data], label=f'{LABELS[method]}: {coding}', **style)
        axes[0].set(ylabel='Batch-amortized decoding time (ms/frame)', xlabel='Eb/N0 (dB)')
        axes[1].set(ylabel='Payload goodput (bits/channel use)', xlabel='Eb/N0 (dB)')
        for ax in axes:
            ax.legend(fontsize=7)
            ax.grid(alpha=.3)
        fig.suptitle(f'CA-SCL L={size}; design/update cost recorded separately in construction.json')
        save(fig, out, f'Decoding_Cost_and_Payload_Goodput_L{size}')
    fields = ['curve','ebno_db','frames','bits','bit_errors','block_errors','zero_error_upper95']
    with (out/'zero_error_observations.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:r[key] for key in fields} for r in rows if r['bit_errors'] == 0)
    (out/'FIGURE_NOTES.md').write_text(
        '# Reading the construction comparisons\n\n'
        'Lines connect measured nonzero points. Zeros remain in CSV and plot annotations; no fake floor or forced monotonicity.\n'
        'Within a panel all curves share N/K, payload, CRC, L, carrier groups and noise. Fixed vs DT changes the information set, not the decoder.\n'
        'DT alone uses nominal design gains; Robust DT uses the supplied scenarios, not generated uncertainty. Consult RUN_CONFIG for perfect-profile vs external design assumptions.\n'
        'Fig5_Eve_full is the nominal DT construction. Each Fig5 has Bob plus the four configured wrong-key cases; fixed/robust comparisons are separate.\n'
        'The percentages count actual derived-keystream bit differences, not unknown master-key bits. 100% is a complement. '
        'BER is measured after decoding/unmasking and is not forced to one. Eve output inversion is a separate control.\n'
        'Key/weight diagnostics describe DT carrier group 0; neither these statistics nor BER proves secrecy.\n', encoding='utf-8')
