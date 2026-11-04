"""Paired fixed/DT/robust constructions; the CA-SCL decoder is held fixed."""
import hashlib
import json
import time
import numpy as np
import torch
from fec.polar.crypto_polar import CryptoPolarEncoder
from fec.polar.utils import generate_5g_ranking
from .construction import construct, mapping_bank
from .experiments import make_decoder, summarize_errors, paired_summary, write_rows
from .security_metrics import mismatch_key, inverse_polar_payload, paired_diagnostics


def curve_name(construction, suffix):
    return suffix if construction == 'fixed' else f'{construction}_{suffix}'


def build_plans(channel, c, case, profiles, out):
    start = time.perf_counter()
    construction_seconds = 0.
    settings = c['construction']
    n, k = case['n'], case['k']
    payload = k-CryptoPolarEncoder._get_crc_length(c['crc'])
    mapping = mapping_bank(n, channel.n_sc, settings['mapping_seed'])
    nominal, scenarios = profiles
    plans, arrays, entries = {}, {'carrier_indices': mapping}, {}
    for method in settings['methods']:
        plans[method], entries[method], scores = [], [], []
        for group, idx in enumerate(mapping):
            design_start = time.perf_counter()
            gains = nominal[idx][None, :]
            if method == 'robust':
                if scenarios is None:
                    raise ValueError('Robust construction needs externally supplied scenario_gains; none are fabricated')
                gains = np.concatenate((gains, scenarios[:, idx]), axis=0)
            frozen, info, score = construct(gains, k, payload/n, settings['design_ebno_db'])
            if method == 'fixed':
                frozen, info = generate_5g_ranking(k, n)
            construction_seconds += time.perf_counter()-design_start
            common = dict(frozen_pos=frozen, n=n, crc_degree=c['crc'])
            enc = CryptoPolarEncoder(**common)
            standard = CryptoPolarEncoder(**common, enable_info_masking=False, enable_frozen_crypto=False)
            frozen_only = CryptoPolarEncoder(**common, enable_info_masking=False)
            decoders = {(kind, size): make_decoder(frozen, n, c['crc'], size, kind == 'crypto')
                        for kind in ('standard', 'crypto') for size in c['list_sizes']}
            plans[method].append(dict(info=info, enc=enc, standard=standard, frozen_only=frozen_only, decoders=decoders))
            entries[method].append(dict(group=group, information_positions=info.tolist(), frozen_positions=frozen.tolist(),
                                       information_sha256=hashlib.sha256(np.asarray(info, dtype='<i8').tobytes()).hexdigest()))
            scores.append(score)
        arrays[f'{method}_log_z_bound'] = np.stack(scores)
    arrays['nominal_gain'] = nominal
    if scenarios is not None:
        arrays['scenario_gains'] = scenarios
    np.savez_compressed(out/'construction_arrays.npz', **arrays)
    record = dict(n=n, k=k, payload_bits=payload, design_ebno_db=settings['design_ebno_db'],
                  mapping='fixed groups cover all carriers; groups cycled equally across frames and methods',
                  design='smallest log Bhattacharyya upper bounds; robust uses maximum across supplied scenarios plus nominal',
                  caveat='Design bounds are not measured BER or CA-SCL predictions; improvement is not guaranteed',
                  decoder='same CA-SCL, CRC and L for all constructions; no modified decoder',
                  csi_feedback='information sets and carrier-group IDs assumed known at both ends; overhead not modeled',
                  groups=entries, design_seconds=construction_seconds, setup_seconds=time.perf_counter()-start,
                  design_timing_note='All ranking/bound computations including reference diagnostics; excludes encoder/decoder allocation')
    (out/'construction.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return plans, torch.as_tensor(mapping, dtype=torch.long), payload


def run_construction_case(channel, c, case, out, *, keygen, profiles, render_plots=True):
    n, k = case['n'], case['k']
    plans, bank, payload_size = build_plans(channel, c, case, profiles, out)
    methods = c['construction']['methods']
    rate = payload_size/n
    descriptors = {}
    for method in methods:
        for size in c['list_sizes']:
            for coding in ('standard', 'crypto'):
                descriptors[curve_name(method, f'{coding}_L{size}')] = dict(
                    construction=method, group='bob', method=coding, list_size=size, scope='', fraction=0.)
        for scope in c['mismatch_scopes']:
            for fraction in sorted(set([0.] + c['mismatch_fractions'])):
                descriptors[curve_name(method, f'eve_{scope}_{fraction:g}')] = dict(
                    construction=method, group='eve', method='crypto', list_size=c['eve_list_size'], scope=scope, fraction=fraction)
        for suffix, coding, size in [('eve_standard', 'standard', c['eve_list_size']),
                                     ('frozen_only_inverse', 'frozen_only_inverse', 0),
                                     ('eve_full_1_inverted', 'output_inversion', c['eve_list_size'])]:
            if suffix == 'eve_full_1_inverted' and ('full' not in c['mismatch_scopes'] or 1. not in c['mismatch_fractions']):
                continue
            descriptors[curve_name(method, suffix)] = dict(construction=method, group='eve_control', method=coding,
                                                          list_size=size, scope='', fraction=0.)
    # Every scheme is decoded in the same carrier-group sub-batches for fair timing.
    dummy = torch.zeros(1, n)
    for group_plans in plans.values():
        for plan in group_plans:
            for decoder in plan['decoders'].values():
                decoder(dummy, dummy)
    rows, paired_rows, records, audits = [], [], [], []
    for snr_idx, snr in enumerate(c['ebno_db']):
        generator = torch.Generator().manual_seed(c['seed'] + snr_idx*100003 + n*1009 + k)
        key_rng = torch.Generator().manual_seed(c['seed'] + snr_idx*99991 + n+k)
        errors = {name: np.zeros(c['frames_per_snr'], dtype=np.uint16) for name in descriptors}
        timing = {name: 0. for name in descriptors}
        work = {name: dict(base_frames=0, expanded_frames=0, flip_frames=0, core_calls=0, max_list_size=0) for name in descriptors}
        audit = {}

        def record(name, decoder, logits, key, payload, positions):
            start = time.perf_counter()
            recovered = decoder(logits, key)
            timing[name] += time.perf_counter()-start
            errors[name][positions] = (recovered != payload).sum(-1).numpy().astype(np.uint16)
            for field, value in decoder.last_stats.items():
                work[name][field] = max(work[name][field], value) if field == 'max_list_size' else work[name][field]+value
            return recovered

        for offset in range(0, c['frames_per_snr'], c['batch_size']):
            batch = min(c['batch_size'], c['frames_per_snr']-offset)
            payloads = torch.randint(0, 2, (batch, payload_size), generator=generator).float()
            keys = keygen.generate_key_stream(batch, n, frame_idx=snr_idx*c['frames_per_snr']+offset)
            noise_b, noise_e = torch.randn((batch, n), generator=generator), torch.randn((batch, n), generator=generator)
            # Wrong keys are identical for every construction for a paired experiment.
            wrong_keys = {}
            for scope in c['mismatch_scopes']:
                for fraction in sorted(set([0.] + c['mismatch_fractions'])):
                    wrong, count, width = mismatch_key(keys, k, fraction, scope, key_rng)
                    observed = (wrong != keys).sum(-1)
                    if not torch.all(observed == count) or (fraction == 1. and scope == 'full' and not torch.all(wrong != keys)):
                        raise RuntimeError('Actual key mismatch does not match the requested experiment')
                    wrong_keys[(scope, fraction)] = wrong
                    item = audit.setdefault((scope, fraction), dict(ebno_db=snr, scope=scope, fraction=fraction,
                        eligible_key_bits=width, expected_flips=count, observed_min=int(observed.min()),
                        observed_max=int(observed.max()), checked_frames=0, information_bit_flips=0, frozen_bit_flips=0))
                    item['checked_frames'] += batch
                    item['observed_min'] = min(item['observed_min'], int(observed.min()))
                    item['observed_max'] = max(item['observed_max'], int(observed.max()))
                    item['information_bit_flips'] += int((wrong[:, :k] != keys[:, :k]).sum())
                    item['frozen_bit_flips'] += int((wrong[:, k:] != keys[:, k:]).sum())
            group_ids = (torch.arange(batch)+offset) % len(bank)
            for group in torch.unique(group_ids).tolist():
                take = torch.nonzero(group_ids == group).flatten()
                positions = (take+offset).numpy()
                payload, key = payloads[take], keys[take]
                indices = bank[group][None, :].expand(len(take), -1)
                for method in methods:
                    plan = plans[method][group]
                    cw = {'standard': plan['standard'](payload, None), 'crypto': plan['enc'](payload, key)}
                    for coding in ('standard', 'crypto'):
                        bob = channel.transmit_bpsk(cw[coding], snr, rate, indices=indices, noise=noise_b[take])[0]
                        for size in c['list_sizes']:
                            record(curve_name(method, f'{coding}_L{size}'), plan['decoders'][(coding, size)],
                                   bob, key if coding == 'crypto' else None, payload, positions)
                    eve = plan['decoders'][('crypto', c['eve_list_size'])]
                    eve_logits = channel.transmit_bpsk(cw['crypto'], snr, rate, node='eve', eve_snr=c['eve_snr'], indices=indices, noise=noise_e[take])[0]
                    for (scope, fraction), wrong in wrong_keys.items():
                        name = curve_name(method, f'eve_{scope}_{fraction:g}')
                        recovered = record(name, eve, eve_logits, wrong[take], payload, positions)
                        if scope == 'full' and fraction == 1.:
                            inverted = curve_name(method, 'eve_full_1_inverted')
                            errors[inverted][positions] = (1-recovered != payload).sum(-1).numpy().astype(np.uint16)
                    std_logits = channel.transmit_bpsk(cw['standard'], snr, rate, node='eve', eve_snr=c['eve_snr'], indices=indices, noise=noise_e[take])[0]
                    record(curve_name(method, 'eve_standard'), plan['decoders'][('standard', c['eve_list_size'])], std_logits, None, payload, positions)
                    frz_logits = channel.transmit_bpsk(plan['frozen_only'](payload, key), snr, rate, node='eve', eve_snr=c['eve_snr'], indices=indices, noise=noise_e[take])[0]
                    start = time.perf_counter()
                    recovered = inverse_polar_payload((frz_logits > 0).float(), plan['info'], payload_size)
                    name = curve_name(method, 'frozen_only_inverse')
                    timing[name] += time.perf_counter()-start
                    errors[name][positions] = (recovered != payload).sum(-1).numpy().astype(np.uint16)
        for method in methods:
            for size in c['list_sizes']:
                pairs = [(curve_name(method, f'standard_L{size}'), curve_name(method, f'crypto_L{size}'))]
                if method != 'fixed':
                    pairs.extend((f'{coding}_L{size}', curve_name(method, f'{coding}_L{size}')) for coding in ('standard', 'crypto'))
                if method == 'robust':
                    pairs.append((curve_name('dt', f'crypto_L{size}'), curve_name(method, f'crypto_L{size}')))
                for reference, candidate in pairs:
                    paired_rows.append(paired_summary(errors[reference], errors[candidate], payload_size, snr, reference, candidate))
        for name, desc in descriptors.items():
            metrics = summarize_errors(errors[name], payload_size)
            width = n-k if desc['scope'] == 'frozen' else n
            flips = int(np.floor(width*desc['fraction']+.5)) if desc['group'] == 'eve' else 0
            rows.append(dict(n=n, k=k, payload_bits=payload_size, code_rate=k/n, payload_rate=rate,
                ebno_db=snr, curve=name, **desc, **metrics, **work[name], flipped_key_bits=flips, eligible_key_bits=width,
                actual_mismatch_fraction=flips/width, decode_seconds=timing[name],
                batch_amortized_ms=1000*timing[name]/metrics['frames'], payload_goodput=rate*(1-metrics['bler'])))
        records.append(np.stack(list(errors.values())))
        audits.extend(audit.values())
        write_rows(out/'metrics.csv', rows)
        write_rows(out/'paired_metrics.csv', paired_rows)
        write_rows(out/'key_mismatch_audit.csv', audits)
        np.savez_compressed(out/'frame_errors.npz', ebno_db=np.asarray(c['ebno_db'][:snr_idx+1]),
                            curve_names=np.asarray(list(descriptors)), frame_bit_errors=np.stack(records), payload_bits=payload_size)
        print(f'N={n} K={k} Eb/N0={snr:g}: {c["frames_per_snr"]} frames, {len(methods)} constructions; saved', flush=True)
    # The diagnostic is identified with the first DT carrier group, not all groups.
    plan = plans['dt'][0]
    diag = paired_diagnostics(plan['enc'], plan['standard'], plan['frozen_only'], keygen, plan['info'], c['diagnostic_samples'], c['seed'])
    diag.update(construction='dt', carrier_group=0)
    (out/'diagnostics.json').write_text(json.dumps(diag, indent=2), encoding='utf-8')
    if render_plots:
        from .plotting_q1 import plot_case
        plot_case(rows, diag, out, c)
    return rows
