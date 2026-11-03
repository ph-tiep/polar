"""Reproducible, paired CA-SCL comparisons on externally supplied Phase 1 CSI."""
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from fec.polar.utils import generate_5g_ranking
from fec.polar.crypto_polar import ChannelBoundKeyGenerator, CryptoPolarEncoder, ImprovedCryptoPolarSCLDecoder
from .digital_twin_channel import DigitalTwinTHzChannel
from .security_metrics import mismatch_key, inverse_polar_payload, paired_diagnostics


def validate_config(c):
    allowed = {'seed', 'threads', 'frames_per_snr', 'batch_size', 'ebno_db', 'list_sizes',
               'cases', 'crc', 'eve_list_size', 'eve_snr', 'mismatch_scopes',
               'mismatch_fractions', 'modified', 'diagnostic_samples'}
    if set(c) - allowed:
        raise ValueError(f'Unknown config fields: {set(c) - allowed}')
    for name in ('frames_per_snr', 'batch_size', 'threads', 'diagnostic_samples'):
        if not isinstance(c[name], int) or c[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for case in c['cases']:
        n, k = case['n'], case['k']
        if set(case) != {'n', 'k'} or not isinstance(n, int) or not isinstance(k, int):
            raise ValueError('Cases require integer n and k')
        if n < 32 or n > 1024 or n & (n-1) or not 0 < k < n:
            raise ValueError('Require power-of-two 32 <= N <= 1024 and 0 < K < N')
        if k <= CryptoPolarEncoder._get_crc_length(c['crc']):
            raise ValueError('K includes CRC and must leave a nonempty payload')
    if not c['cases'] or len({(x['n'], x['k']) for x in c['cases']}) != len(c['cases']):
        raise ValueError('Require nonempty unique cases')
    if not c['list_sizes'] or any(not isinstance(l, int) or l < 2 or l > 32 or l & (l-1) for l in c['list_sizes'] + [c['eve_list_size']]):
        raise ValueError('List sizes must be powers of two from 2 to 32')
    if len(set(c['list_sizes'])) != len(c['list_sizes']):
        raise ValueError('Duplicate list sizes')
    if c['eve_list_size'] not in c['list_sizes']:
        raise ValueError('Include eve_list_size among Bob baselines')
    if c['eve_snr'] not in ('physical', 'equal-average'):
        raise ValueError('Unknown Eve SNR normalization')
    if not c['mismatch_scopes'] or set(c['mismatch_scopes']) - {'full', 'frozen'}:
        raise ValueError('Unknown mismatch scope')
    if not c['mismatch_fractions'] or any(not 0 <= p <= 1 for p in c['mismatch_fractions']):
        raise ValueError('Mismatch fractions must be in [0,1]')
    if not c['ebno_db'] or not np.isfinite(c['ebno_db']).all() or len(set(c['ebno_db'])) != len(c['ebno_db']):
        raise ValueError('Require unique finite SNR values')
    m = c['modified']
    if set(m) != {'enabled', 'base_list', 'max_flip_trials', 'damping', 'expand_list', 'retry_damping'}:
        raise ValueError('Invalid modified-decoder settings')
    if m['base_list'] not in c['list_sizes'] or m['max_flip_trials'] < 0 or m['damping'] <= 0 or m['retry_damping'] <= 0:
        raise ValueError('Invalid modified-decoder budget')
    if m['enabled'] and m['expand_list'] and min(2*m['base_list'], 32) not in c['list_sizes']:
        raise ValueError('Include the expanded-list baseline in list_sizes')


def summarize_errors(frame_errors, payload_bits):
    """Wilson BLER interval; BER SE treats frames, not bits, as independent."""
    counts = np.asarray(frame_errors)
    frames, errors = len(counts), int(counts.sum())
    blocks = int(np.count_nonzero(counts))
    p, z = blocks / frames, 1.959963984540054
    den = 1 + z*z/frames
    center = (p + z*z/(2*frames))/den
    half = z*np.sqrt(p*(1-p)/frames + z*z/(4*frames*frames))/den
    per_frame = counts / payload_bits
    # If all frames have zero errors, BER <= BLER gives a conservative upper
    # bound without assuming independent errors within a codeword.
    upper_zero = float(1 - .05**(1/frames)) if blocks == 0 else None
    return dict(frames=frames, bits=frames*payload_bits, bit_errors=errors,
                block_errors=blocks, ber=errors/(frames*payload_bits), bler=p,
                bler_low95=max(0., center-half), bler_high95=min(1., center+half),
                ber_frame_se=float(per_frame.std(ddof=1)/np.sqrt(frames)) if frames > 1 else None,
                zero_error_upper95=upper_zero)


def carrier_indices(batch, n, carriers, generator):
    # Each full cycle covers all carriers once; remainder samples without replacement.
    return torch.stack([torch.cat([torch.randperm(carriers, generator=generator)
                                  for _ in range((n+carriers-1)//carriers)])[:n]
                        for _ in range(batch)])


def make_decoder(frozen, n, crc, size, crypto=True, modified=None):
    m = modified or dict(damping=1., max_flip_trials=0, expand_list=False, retry_damping=1.)
    return ImprovedCryptoPolarSCLDecoder(frozen, n, list_size=size, crc_degree=crc,
        enable_info_masking=crypto, enable_frozen_crypto=crypto,
        llr_damping=m['damping'], max_flip_trials=m['max_flip_trials'],
        expand_list=m['expand_list'], retry_damping=m['retry_damping'])


def run_case(channel, c, case, out, *, render_plots=True):
    n, k = case['n'], case['k']
    frozen, info = generate_5g_ranking(k, n)
    common = dict(frozen_pos=frozen, n=n, crc_degree=c['crc'])
    enc = CryptoPolarEncoder(**common)
    std = CryptoPolarEncoder(**common, enable_info_masking=False, enable_frozen_crypto=False)
    frz = CryptoPolarEncoder(**common, enable_info_masking=False)
    payload_size = enc.k_payload
    rate = payload_size/n
    keygen = ChannelBoundKeyGenerator(b'PUBLIC_SIMULATION_FIXTURE_NOT_A_DEPLOYMENT_SECRET', channel.H_Bob)
    decoders, descriptors = {}, {}
    for size in c['list_sizes']:
        for method, crypto in [('standard', False), ('crypto', True)]:
            name = f'{method}_L{size}'
            decoders[name] = make_decoder(frozen, n, c['crc'], size, crypto)
            descriptors[name] = dict(group='bob', method=method, list_size=size, scope='', fraction=0.)
    m = c['modified']
    if m['enabled']:
        decoders['modified'] = make_decoder(frozen, n, c['crc'], m['base_list'], modified=m)
        descriptors['modified'] = dict(group='modified', method='modified', list_size=m['base_list'], scope='', fraction=0.)
    eve = make_decoder(frozen, n, c['crc'], c['eve_list_size'])
    eve_standard = make_decoder(frozen, n, c['crc'], c['eve_list_size'], False)
    for scope in c['mismatch_scopes']:
        for frac in sorted(set([0.] + c['mismatch_fractions'])):
            name = f'eve_{scope}_{frac:g}'
            descriptors[name] = dict(group='eve', method='crypto', list_size=c['eve_list_size'], scope=scope, fraction=frac)
    descriptors['eve_standard'] = dict(group='eve_control', method='standard', list_size=c['eve_list_size'], scope='', fraction=0.)
    descriptors['frozen_only_inverse'] = dict(group='eve_control', method='frozen_only_inverse', list_size=0, scope='', fraction=0.)
    rows = []
    # Construction/allocation warmup excluded from timing, including retry paths.
    dummy = torch.zeros((1, n))
    key0 = torch.zeros_like(dummy)
    for decoder in list(decoders.values()) + [eve, eve_standard]:
        decoder(dummy, key0)
    for snr_idx, snr in enumerate(c['ebno_db']):
        gen = torch.Generator().manual_seed(c['seed'] + snr_idx*100003 + n*1009 + k)
        key_rng = torch.Generator().manual_seed(c['seed'] + snr_idx*99991 + n + k)
        errors = {name: [] for name in descriptors}
        timing = {name: 0. for name in descriptors}
        work = {name: dict(base_frames=0, expanded_frames=0, flip_frames=0, core_calls=0, max_list_size=0) for name in descriptors}
        def record(name, decoder, logits, key, payload):
            start = time.perf_counter()
            recovered = decoder(logits, key)
            timing[name] += time.perf_counter()-start
            errors[name].extend((recovered != payload).sum(dim=-1).tolist())
            for field, val in decoder.last_stats.items():
                work[name][field] = max(work[name][field], val) if field == 'max_list_size' else work[name][field]+val
        for offset in range(0, c['frames_per_snr'], c['batch_size']):
            batch = min(c['batch_size'], c['frames_per_snr']-offset)
            payload = torch.randint(0, 2, (batch, payload_size), generator=gen).float()
            keys = torch.cat([keygen.generate_key_stream(1, n, frame_idx=snr_idx*c['frames_per_snr']+offset+i) for i in range(batch)])
            idx = carrier_indices(batch, n, channel.n_sc, gen)
            noise_b = torch.randn((batch, n), generator=gen)
            noise_e = torch.randn((batch, n), generator=gen)
            cw = {'standard': std(payload, None), 'crypto': enc(payload, keys)}
            bob = {name: channel.transmit_bpsk(bits, snr, rate, indices=idx, noise=noise_b)[0] for name, bits in cw.items()}
            for name, dec in decoders.items():
                method = 'standard' if descriptors[name]['method'] == 'standard' else 'crypto'
                record(name, dec, bob[method], None if method == 'standard' else keys, payload)
            eve_logits = channel.transmit_bpsk(cw['crypto'], snr, rate, node='eve', eve_snr=c['eve_snr'], indices=idx, noise=noise_e)[0]
            for name, desc in descriptors.items():
                if desc['group'] == 'eve':
                    wrong, _, _ = mismatch_key(keys, k, desc['fraction'], desc['scope'], key_rng)
                    record(name, eve, eve_logits, wrong, payload)
            std_logits = channel.transmit_bpsk(cw['standard'], snr, rate, node='eve', eve_snr=c['eve_snr'], indices=idx, noise=noise_e)[0]
            record('eve_standard', eve_standard, std_logits, None, payload)
            frz_logits = channel.transmit_bpsk(frz(payload, keys), snr, rate, node='eve', eve_snr=c['eve_snr'], indices=idx, noise=noise_e)[0]
            start = time.perf_counter()
            recovered = inverse_polar_payload((frz_logits > 0).float(), info, payload_size)
            timing['frozen_only_inverse'] += time.perf_counter()-start
            errors['frozen_only_inverse'].extend((recovered != payload).sum(dim=-1).tolist())
        for name, desc in descriptors.items():
            metrics = summarize_errors(errors[name], payload_size)
            width = n-k if desc['scope'] == 'frozen' else n
            flips = int(np.floor(width*desc['fraction']+.5)) if desc['group'] == 'eve' else 0
            row = dict(n=n, k=k, payload_bits=payload_size, code_rate=k/n, payload_rate=rate,
                       ebno_db=snr, curve=name, **desc, **metrics, **work[name],
                       flipped_key_bits=flips, eligible_key_bits=width,
                       actual_mismatch_fraction=flips/width,
                       decode_seconds=timing[name], batch_amortized_ms=1000*timing[name]/metrics['frames'],
                       payload_goodput=rate*(1-metrics['bler']))
            rows.append(row)
        print(f'N={n} K={k} Eb/N0={snr:g}: {c["frames_per_snr"]} frames per curve', flush=True)
        write_rows(out/'metrics.csv', rows)
    diag = paired_diagnostics(enc, std, frz, keygen, info, c['diagnostic_samples'], c['seed'])
    (out/'diagnostics.json').write_text(json.dumps(diag, indent=2), encoding='utf-8')
    if render_plots:
        from .plotting_q1 import plot_case
        plot_case(rows, diag, out, c)
    return rows


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config, dataset, output, *, expected_tx, expected_rx, smoke=False):
    validate_config(config)
    if not smoke and len(config['ebno_db']) < 2:
        raise ValueError('Experiments need at least two SNR points; use --smoke for an integration check')
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='running', purpose='technical-smoke' if smoke else 'experiment',
                    config=config, dataset=str(dataset.resolve()), expected_tx=expected_tx,
                    expected_rx=expected_rx, python=platform.python_version(), torch=torch.__version__,
                    numpy=np.__version__, platform=platform.platform(),
                    modulation='coherent BPSK frequency-domain abstraction',
                    ebno_reference='payload bit; Bob mean effective channel normalized to one',
                    security_note='Public simulation keys; wrong derived-key bits, not master-key compromise',
                    timing_note='CPU wall time, batch-amortized; not single-frame latency',
                    hashes={})
    try:
        manifest['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        manifest['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=Path(__file__).resolve().parents[1], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        manifest['git_commit'] = None
    source_root = Path(__file__).resolve().parents[1]
    sources = [source_root/'main_phase2_crypto_q1.py']
    for folder in ('fec', 'phase2_crypto_polar'):
        sources.extend(sorted((source_root/folder).rglob('*.py')))
    manifest['source_sha256'] = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    path = output/'RUN_CONFIG.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    try:
        for name in ('H_Bob.npy', 'H_Eve.npy', 'Metadata.npz'):
            manifest['hashes'][name] = hashlib.sha256((dataset/name).read_bytes()).hexdigest()
        phase1_manifest = dataset.parent.parent/'RUN_CONFIG.json'
        if phase1_manifest.exists():
            manifest['phase1_run'] = json.loads(phase1_manifest.read_text(encoding='utf-8'))
            if manifest['phase1_run'].get('status') != 'completed':
                raise ValueError('Phase 1 manifest is not completed; do not use partial CSI')
        channel = DigitalTwinTHzChannel(str(dataset))
        if (channel.n_tx, channel.n_rx) != (expected_tx, expected_rx):
            raise ValueError('CSI antenna shape differs from requested case; generate fresh Phase 1 CSI')
        with np.load(dataset/'Metadata.npz', allow_pickle=False) as meta:
            for field, actual in [('NumTxAntennas', channel.n_tx), ('NumRxAntennas', channel.n_rx), ('NumSubcarriers', channel.n_sc)]:
                if int(meta[field]) != actual:
                    raise ValueError(f'Metadata/CSI mismatch: {field}')
            manifest['phase1_metadata'] = {name: meta[name].tolist() for name in meta.files}
        manifest['channel'] = channel.get_channel_summary()
        if not smoke:
            from .plotting_q1 import plot_channel
            plot_channel(channel, output)
        with torch.inference_mode():
            for case in config['cases']:
                target = output/f'N{case["n"]}_K{case["k"]}'
                target.mkdir()
                run_case(channel, config, case, target, render_plots=not smoke)
        if smoke:
            (output/'SMOKE_ONLY.txt').write_text(
                'TECHNICAL CHECK ONLY. No performance figures are produced.\n'
                'Run again without --smoke in a NEW output directory for BER/BLER curves.\n', encoding='utf-8')
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
