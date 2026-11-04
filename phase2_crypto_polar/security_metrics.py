"""Key-mismatch controls and paired diagnostics; no randomness-test battery."""
import numpy as np
import torch
import copy
from fec.polar.encoding import PolarEncoder


def mismatch_key(key, k, fraction, scope, generator):
    """Flip an exact rounded count per frame in the derived keystream, not master key.

    frozen: only key[k:] is wrong; information mask is assumed known.
    full: positions drawn from all N derived key bits.
    """
    if scope not in ('frozen', 'full') or not 0 <= fraction <= 1:
        raise ValueError('Invalid key mismatch scope/fraction')
    start = k if scope == 'frozen' else 0
    width = key.shape[-1] - start
    count = int(np.floor(fraction * width + .5))
    wrong = key.clone()
    for row in range(len(key)):
        pos = torch.randperm(width, generator=generator)[:count] + start
        wrong[row, pos] = 1 - wrong[row, pos]
    return wrong, count, width


def inverse_polar_payload(codeword, info_pos, payload_size):
    """Noiseless/hard-decision attack control for frozen-only coding; no key."""
    n = codeword.shape[-1]
    transform = PolarEncoder(np.array([], dtype=int), n)
    return transform(codeword)[..., info_pos][..., :payload_size]


def paired_diagnostics(encoder, standard, frozen_only, keygen, info_pos, samples, seed):
    """One master-key-bit perturbation, holding public context, frame ID and plaintext fixed.

    Output-bit flip probabilities are descriptive, not a security certificate.
    """
    rng = torch.Generator().manual_seed(seed)
    n = encoder.n
    master = keygen.master_key
    master2 = bytes([master[0] ^ 1]) + master[1:]
    other = copy.copy(keygen)
    other.master_key = master2
    flips, weights_std, weights_cry, attack_errors = [], [], [], 0
    for frame in range(samples):
        payload = torch.randint(0, 2, (1, encoder.k_payload), generator=rng).float()
        key = keygen.generate_key_stream(1, n, frame_idx=frame)
        key2 = other.generate_key_stream(1, n, frame_idx=frame)
        cw, cw2 = encoder(payload, key), encoder(payload, key2)
        flips.append((cw != cw2).numpy()[0])
        weights_std.append(int(standard(payload, None).sum()))
        weights_cry.append(int(cw.sum()))
        recovered = inverse_polar_payload(frozen_only(payload, key), info_pos, encoder.k_payload)
        attack_errors += int((recovered != payload).sum())
    flips = np.asarray(flips)
    return dict(samples=samples, master_key_bit_flipped=0,
                mean_codeword_flip_fraction=float(flips.mean()),
                output_bit_flip_probability=flips.mean(axis=0).tolist(),
                standard_weights=weights_std, crypto_weights=weights_cry,
                frozen_only_noiseless_attack_bit_errors=attack_errors,
                frozen_only_noiseless_attack_bits=samples * encoder.k_payload,
                note='Same plaintext, public context and frame ID per pair; descriptive diagnostic, not a security proof.')
