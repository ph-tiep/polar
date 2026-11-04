"""Polar construction from parallel BPSK channels using Bhattacharyya bounds.

Z(W+) = Z1*Z2 and Z(W-) <= Z1+Z2-Z1*Z2 for independent binary-input
memoryless symmetric channels. Recursion follows this repository's F^tensor
encoder, without a bit-reversal permutation. Scores are conservative design
bounds, not measured BERs or CA-SCL error probabilities.
"""
from pathlib import Path
import hashlib
import numpy as np


def log_one_minus_exp(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x < -np.log(2), np.log1p(-np.exp(x)), np.log(-np.expm1(x)))


def polar_log_bounds(channel_log_z):
    z = np.asarray(channel_log_z, dtype=float)
    if z.ndim != 1 or len(z) < 2 or len(z) & (len(z)-1) or not np.isfinite(z).all() or np.any(z > 0):
        raise ValueError('Require a finite nonpositive log-Z vector of power-of-two length')

    def recurse(v):
        if len(v) == 1:
            return v
        a, b = np.split(v, 2)
        minus = np.minimum(0., np.logaddexp(a, b + log_one_minus_exp(a)))
        return np.concatenate((recurse(minus), recurse(a+b)))

    return recurse(z)


def construct(gains, k, payload_rate, design_ebno_db):
    """Select K smallest worst-scenario bounds; gains shape [scenario, N]."""
    gains = np.asarray(gains, dtype=float)
    if gains.ndim == 1:
        gains = gains[None, :]
    if gains.ndim != 2 or len(gains) == 0 or not np.isfinite(gains).all() or np.any(gains < 0):
        raise ValueError('Design gains must be finite nonnegative [scenario, N]')
    n = gains.shape[1]
    if not isinstance(k, int) or not 0 < k < n or not 0 < payload_rate < 1 or not np.isfinite(design_ebno_db):
        raise ValueError('Invalid construction parameters')
    # sigma^2 = 1/(2*R_payload*Eb/N0), Z = exp(-gain/(2*sigma^2)).
    log_z = -payload_rate * 10.**(design_ebno_db/10.) * gains
    bounds = np.stack([polar_log_bounds(v) for v in log_z])
    score = bounds.max(axis=0)
    info = np.sort(np.lexsort((np.arange(n), score))[:k])
    frozen = np.setdiff1d(np.arange(n), info)
    return frozen, info, score


def mapping_bank(n, carriers, seed):
    """Fixed, reproducible carrier groups; every full cycle uses every carrier."""
    if n < 1 or carriers < 1:
        raise ValueError('Positive mapping dimensions required')
    rng = np.random.default_rng(seed)
    groups = max(1, (carriers+n-1)//n)
    count = groups*n
    flat = np.concatenate([rng.permutation(carriers) for _ in range((count+carriers-1)//carriers)])[:count]
    return flat.reshape(groups, n)


def design_profiles(channel, profile_path=None):
    """External gains must use the same carriers, beam and gain reference.

    Without a file the nominal profile is the evaluation channel: this is an
    explicitly labeled perfect-profile experiment, not DT prediction validation.
    No uncertainty distribution is fabricated.
    """
    if profile_path is None:
        return np.asarray(channel.g_bob_norm, dtype=float), None, dict(
            source='evaluation CSI', assumption='perfect Bob effective-gain profile',
            independent_prediction=False, uncertainty_scenarios=0)
    path = Path(profile_path)
    with np.load(path, allow_pickle=False) as data:
        required = {'nominal_gain', 'NumTxAntennas', 'NumRxAntennas', 'NumSubcarriers', 'gain_reference'}
        if not required <= set(data.files):
            raise ValueError(f'Design profile requires {sorted(required)}')
        for field, actual in [('NumTxAntennas', channel.n_tx), ('NumRxAntennas', channel.n_rx), ('NumSubcarriers', channel.n_sc)]:
            if np.asarray(data[field]).shape != () or int(data[field]) != actual:
                raise ValueError(f'Design profile mismatch: {field}')
        if str(data['gain_reference'].item()) != 'evaluation_bob_mean':
            raise ValueError('Design gains must use gain_reference=evaluation_bob_mean')
        nominal = np.asarray(data['nominal_gain'], dtype=float)
        scenarios = np.asarray(data['scenario_gains'], dtype=float) if 'scenario_gains' in data else None
        provenance = str(data['provenance_json'].item()) if 'provenance_json' in data else None
    if nominal.shape != (channel.n_sc,) or not np.isfinite(nominal).all() or np.any(nominal < 0) or not nominal.any():
        raise ValueError('nominal_gain must be a finite nonnegative carrier vector with nonzero power')
    if scenarios is not None and (scenarios.ndim != 2 or scenarios.shape[1] != channel.n_sc or len(scenarios) < 1 or
                                  not np.isfinite(scenarios).all() or np.any(scenarios < 0) or not np.all(scenarios.sum(axis=1) > 0)):
        raise ValueError('scenario_gains must be nonempty finite nonnegative [scenario, carrier]')
    return nominal, scenarios, dict(source=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        assumption='externally supplied effective gains; provenance and uncertainty require user justification',
        independent_prediction='not independently verified', uncertainty_scenarios=0 if scenarios is None else len(scenarios),
        source_provenance_json=provenance)


def effective_design_gain(evaluation_h, design_h):
    """Gain of supplied Bob CSI on the evaluation transmitter's existing beam.

    Keeps the simulator's ideal evaluation-CSI transmit beam and coherent MRC
    assumptions. Only code construction sees the alternative channel profile.
    This is not an experiment in estimated-beamforming or receiver-CSI error.
    """
    reference, predicted = np.asarray(evaluation_h), np.asarray(design_h)
    for arr in (reference, predicted):
        if arr.ndim != 3 or min(arr.shape) < 1 or not np.iscomplexobj(arr) or not np.isfinite(arr).all():
            raise ValueError('Bob CSI must be finite complex [Rx, Tx, carrier]')
    if reference.shape != predicted.shape:
        raise ValueError('Design and evaluation CSI shapes differ; do not slice/repeat antennas')
    gain, original = [], []
    for carrier in range(reference.shape[-1]):
        _, singular, vh = np.linalg.svd(reference[:, :, carrier], full_matrices=False)
        if singular[0] <= 0:
            raise ValueError('Evaluation CSI must have nonzero power on every carrier')
        beam = vh[0].conj()
        gain.append(np.linalg.norm(predicted[:, :, carrier] @ beam)**2)
        original.append(singular[0]**2)
    result = np.asarray(gain)/np.mean(original)
    if not np.isfinite(result).all() or not result.any():
        raise ValueError('Design profile has invalid or entirely zero gain')
    return result
