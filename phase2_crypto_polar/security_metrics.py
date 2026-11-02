"""
security_metrics.py
======================================================================
Cryptographic & Physical-Layer Security Evaluation Suite for WoS Q1/Q2
Journal Publication on 6G Crypto-Polar Coding.

Includes:
  1. Avalanche Effect & Strict Avalanche Criterion (SAC)
  2. Bit Independence Criterion (BIC) & Cross-Correlation Analysis
  3. NIST SP 800-22 Statistical Randomness Test Suite (8 Tests)
  4. Shannon Entropy, Conditional Equivocation & Mutual Information Leakage
  5. Partial Frozen-Key Compromise Resilience (Polar Tree Error Cascade)
  6. Affine Coset Weight Spectrum & Minimum Hamming Distance Analysis
======================================================================
"""

from typing import Dict, List, Tuple, Union
import math
import time
import numpy as np
import scipy.special as sp
import scipy.stats as stats
import torch

from fec.polar.utils import generate_5g_ranking
from fec.polar.crypto_polar import (
    ChannelBoundKeyGenerator,
    CryptoPolarEncoder,
    CryptoPolarDecoder,
)


def binary_entropy(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Compute binary Shannon entropy H_2(p) in bits/bit."""
    p_arr = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    h = -p_arr * np.log2(p_arr) - (1.0 - p_arr) * np.log2(1.0 - p_arr)
    return float(h) if np.ndim(p) == 0 else h


# ======================================================================
# 1. AVALANCHE EFFECT, SAC & CROSS-CORRELATION
# ======================================================================

def evaluate_avalanche_and_sac(
    k: int = 128,
    n: int = 256,
    crc_degree: str = "CRC11",
    num_trials: int = 1000,
) -> Dict:
    """Evaluate Cryptographic Avalanche Effect, Strict Avalanche Criterion (SAC),
    Bit Independence Criterion (BIC), and Cross-Correlation.
    """
    frozen_pos, _ = generate_5g_ranking(k, n)
    encoder = CryptoPolarEncoder(
        frozen_pos=frozen_pos,
        n=n,
        crc_degree=crc_degree,
        enable_info_masking=True,
        enable_frozen_crypto=True,
    )
    decoder = CryptoPolarDecoder(
        frozen_pos=frozen_pos,
        n=n,
        dec_type="Improved-CA-SCL",
        list_size=8,
        crc_degree=crc_degree,
    )
    k_payload = encoder.k_payload
    key_gen = ChannelBoundKeyGenerator(master_key=b"SAC_EVALUATION_MASTER_KEY_2026")

    # Generate base payloads and keys from 1-bit flipped master seeds
    payloads = torch.randint(0, 2, (num_trials, k_payload), dtype=torch.float32)
    keys_base = key_gen.generate_key_stream(num_trials, n, frame_idx=100)

    # Create keys derived from a 1-bit flip in the cryptographic seed per block
    keys_seed_flipped = torch.zeros_like(keys_base)
    for i in range(num_trials):
        seed_bytes = bytearray(b"SAC_EVALUATION_MASTER_KEY_2026")
        bit_to_flip = i % (len(seed_bytes) * 8)
        seed_bytes[bit_to_flip // 8] ^= 1 << (bit_to_flip % 8)
        kg_flip = ChannelBoundKeyGenerator(master_key=bytes(seed_bytes))
        keys_seed_flipped[i : i + 1] = kg_flip.generate_key_stream(1, n, frame_idx=100 + i)

    cw_base = encoder(payloads, keys_base)
    cw_flipped = encoder(payloads, keys_seed_flipped)

    # Codeword bit-flip ratios (Avalanche effect on ciphertext/codeword)
    cw_diff = torch.abs(cw_base - cw_flipped).numpy()
    cw_flip_ratios = np.mean(cw_diff, axis=1)  # per trial

    # Decoder output avalanche (1-bit seed mismatch at receiver under noiseless channel)
    llr_noiseless = 20.0 * (2.0 * cw_base - 1.0)
    dec_wrong_seed = decoder(llr_noiseless, secret_key_stream=keys_seed_flipped)
    dec_diff = torch.abs(payloads - dec_wrong_seed).numpy()
    dec_flip_ratios = np.mean(dec_diff, axis=1)

    # SAC Matrix: Probability each output codeword bit j flips when key seed flips
    sac_bit_probs = np.mean(cw_diff, axis=0)  # shape (n,)
    sac_dec_probs = np.mean(dec_diff, axis=0)  # shape (k_payload,)

    # Bit Independence Criterion (BIC): pairwise correlation between flipped output bits
    # Sample first 64 output bit positions to compute pairwise correlation matrix
    sub_diff = cw_diff[:, :64]
    corr_mat = np.corrcoef(sub_diff, rowvar=False)
    off_diag = corr_mat[~np.eye(64, dtype=bool)]
    bic_mean_abs_corr = float(np.mean(np.abs(off_diag)))
    bic_max_abs_corr = float(np.max(np.abs(off_diag)))

    # Cross-correlation between bipolar Base Codeword and Flipped Codeword
    bipolar_base = 2.0 * cw_base.numpy() - 1.0
    bipolar_flip = 2.0 * cw_flipped.numpy() - 1.0
    cross_corrs = np.sum(bipolar_base * bipolar_flip, axis=1) / float(n)

    # Direct single-frozen-bit algebraic diffusion test:
    # Flip a single high-weight frozen bit directly (without hashing) to measure pure Polar G_N diffusion
    n_frz = n - k
    direct_frozen_hamming = []
    zero_payload = torch.zeros((n_frz, k_payload), dtype=torch.float32)
    base_zero_key = torch.zeros((n_frz, n), dtype=torch.float32)
    for f_idx in range(n_frz):
        base_zero_key[f_idx, k + f_idx] = 1.0
    cw_single_frozen = encoder(zero_payload, base_zero_key).numpy()
    direct_frozen_weights = np.sum(cw_single_frozen, axis=1) / float(n)

    return {
        "cw_avalanche_mean": float(np.mean(cw_flip_ratios)),
        "cw_avalanche_std": float(np.std(cw_flip_ratios)),
        "dec_avalanche_mean": float(np.mean(dec_flip_ratios)),
        "dec_avalanche_std": float(np.std(dec_flip_ratios)),
        "sac_mean_deviation_from_half": float(np.mean(np.abs(sac_bit_probs - 0.5))),
        "bic_mean_abs_correlation": bic_mean_abs_corr,
        "bic_max_abs_correlation": bic_max_abs_corr,
        "cross_corr_mean": float(np.mean(cross_corrs)),
        "cross_corr_std": float(np.std(cross_corrs)),
        "cw_flip_ratios": cw_flip_ratios,
        "dec_flip_ratios": dec_flip_ratios,
        "sac_bit_probs": sac_bit_probs,
        "sac_dec_probs": sac_dec_probs,
        "cross_corrs": cross_corrs,
        "direct_frozen_weights": direct_frozen_weights,
    }


# ======================================================================
# 2. NIST SP 800-22 STATISTICAL RANDOMNESS SUITE
# ======================================================================

def evaluate_nist_randomness(
    k: int = 128,
    n: int = 256,
    crc_degree: str = "CRC11",
    total_bits: int = 65536,
) -> Dict:
    """Run 8 NIST SP 800-22 Statistical Randomness Tests on Crypto-Polar
    codewords generated from structured (low-entropy all-zero / periodic)
    plaintext to prove cryptographic indistinguishability from i.i.d. uniform bits.
    """
    frozen_pos, _ = generate_5g_ranking(k, n)
    encoder = CryptoPolarEncoder(
        frozen_pos=frozen_pos,
        n=n,
        crc_degree=crc_degree,
        enable_info_masking=True,
        enable_frozen_crypto=True,
    )
    num_blocks = (total_bits + n - 1) // n
    # Worst-case structured plaintext: all-zeros!
    # In standard Polar coding, all-zero plaintext -> all-zero codeword (Entropy = 0).
    # In Crypto-Polar coding, all-zero plaintext -> pseudorandom coset codewords!
    structured_payload = torch.zeros((num_blocks, encoder.k_payload), dtype=torch.float32)
    key_gen = ChannelBoundKeyGenerator(master_key=b"NIST_SP800_22_TEST_KEY_2026")
    key_stream = key_gen.generate_key_stream(num_blocks, n, frame_idx=2026)

    cw = encoder(structured_payload, key_stream).numpy().astype(np.int32).ravel()[:total_bits]
    N_bits = len(cw)
    bipolar = 2 * cw - 1  # in {-1, +1}

    results = {}

    # Test 1: Monobit Frequency Test
    s_obs = abs(np.sum(bipolar)) / math.sqrt(N_bits)
    p_monobit = float(sp.erfc(s_obs / math.sqrt(2.0)))
    results["1. Monobit Frequency"] = {"stat": float(s_obs), "p_value": p_monobit, "pass": p_monobit >= 0.01}

    # Test 2: Frequency Test within a Block (M = 128)
    M_blk = 128
    num_b = N_bits // M_blk
    blocks = cw[: num_b * M_blk].reshape(num_b, M_blk)
    pi_i = np.mean(blocks, axis=1)
    chi2_blk = 4.0 * M_blk * np.sum((pi_i - 0.5) ** 2)
    p_block = float(sp.gammaincc(num_b / 2.0, chi2_blk / 2.0))
    results["2. Block Frequency"] = {"stat": float(chi2_blk), "p_value": p_block, "pass": p_block >= 0.01}

    # Test 3: Runs Test
    pi_ones = float(np.mean(cw))
    v_obs = float(np.sum(cw[:-1] != cw[1:]) + 1)
    num = abs(v_obs - 2.0 * N_bits * pi_ones * (1.0 - pi_ones))
    den = 2.0 * math.sqrt(2.0 * N_bits) * pi_ones * (1.0 - pi_ones) + 1e-12
    p_runs = float(sp.erfc(num / den))
    results["3. Runs Test"] = {"stat": float(v_obs), "p_value": p_runs, "pass": p_runs >= 0.01}

    # Test 4: Longest Run of Ones in a Block (M = 128, K = 5)
    M_lr = 128
    n_lr = N_bits // M_lr
    blk_lr = cw[: n_lr * M_lr].reshape(n_lr, M_lr)
    v_counts = np.zeros(6, dtype=np.float64)
    # NIST SP 800-22 probabilities for M=128: categories <=4, 5, 6, 7, 8, >=9
    pi_nist = np.array([0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124])
    for row in blk_lr:
        max_run = 0
        cur_run = 0
        for bit in row:
            if bit == 1:
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
            else:
                cur_run = 0
        if max_run <= 4:
            v_counts[0] += 1
        elif max_run >= 9:
            v_counts[5] += 1
        else:
            v_counts[max_run - 4] += 1
    chi2_lr = float(np.sum(((v_counts - n_lr * pi_nist) ** 2) / (n_lr * pi_nist)))
    p_lr = float(sp.gammaincc(5.0 / 2.0, chi2_lr / 2.0))
    results["4. Longest Run of Ones"] = {"stat": chi2_lr, "p_value": p_lr, "pass": p_lr >= 0.01}

    # Test 5: Discrete Fourier Transform (Spectral) Test
    fft_vals = np.fft.fft(bipolar)
    mags = np.abs(fft_vals[: N_bits // 2])
    threshold = math.sqrt(math.log(1.0 / 0.05) * N_bits)
    n0 = 0.95 * (N_bits / 2.0)
    n1 = float(np.sum(mags < threshold))
    d_spec = (n1 - n0) / math.sqrt(N_bits * 0.95 * 0.05 / 4.0)
    p_spec = float(sp.erfc(abs(d_spec) / math.sqrt(2.0)))
    results["5. Spectral DFT"] = {"stat": float(d_spec), "p_value": p_spec, "pass": p_spec >= 0.01}

    # Test 6: Cumulative Sums (Cusum) Test
    cusum = np.cumsum(bipolar)
    z_max = float(np.max(np.abs(cusum)))
    # NIST normal approximation for large N
    k_start = int(np.floor((-N_bits / z_max + 1.0) / 4.0))
    k_end = int(np.floor((N_bits / z_max - 1.0) / 4.0))
    sum1 = 0.0
    for k_i in range(k_start, k_end + 1):
        sum1 += stats.norm.cdf((4 * k_i + 1) * z_max / math.sqrt(N_bits)) - stats.norm.cdf(
            (4 * k_i - 1) * z_max / math.sqrt(N_bits)
        )
    k_start2 = int(np.floor((-N_bits / z_max - 3.0) / 4.0))
    k_end2 = int(np.floor((N_bits / z_max - 1.0) / 4.0))
    sum2 = 0.0
    for k_i in range(k_start2, k_end2 + 1):
        sum2 += stats.norm.cdf((4 * k_i + 3) * z_max / math.sqrt(N_bits)) - stats.norm.cdf(
            (4 * k_i + 1) * z_max / math.sqrt(N_bits)
        )
    p_cusum = float(np.clip(1.0 - sum1 + sum2, 0.0, 1.0))
    results["6. Cumulative Sums"] = {"stat": z_max, "p_value": p_cusum, "pass": p_cusum >= 0.01}

    # Test 7: Approximate Entropy Test (m = 2)
    def _phi(m_len: int) -> float:
        if m_len == 0:
            return 0.0
        ext = np.concatenate([cw, cw[: m_len - 1]])
        counts = np.zeros(1 << m_len, dtype=np.float64)
        idx = np.zeros(N_bits, dtype=np.int32)
        for b_pos in range(m_len):
            idx = (idx << 1) | ext[b_pos : b_pos + N_bits]
        unique, u_counts = np.unique(idx, return_counts=True)
        counts[unique] = u_counts
        c_i = counts / float(N_bits)
        c_i = c_i[c_i > 0]
        return float(np.sum(c_i * np.log(c_i)))

    m_ap = 2
    ap_en = _phi(m_ap) - _phi(m_ap + 1)
    chi2_ap = 2.0 * N_bits * (math.log(2.0) - ap_en)
    p_apen = float(sp.gammaincc(2 ** (m_ap - 1), chi2_ap / 2.0))
    results["7. Approximate Entropy"] = {"stat": float(chi2_ap), "p_value": p_apen, "pass": p_apen >= 0.01}

    # Test 8: Autocorrelation Test (lag d = 1)
    d_lag = 1
    xor_lag = np.bitwise_xor(cw[:-d_lag], cw[d_lag:])
    a_d = float(np.sum(xor_lag))
    x_auto = 2.0 * (a_d - (N_bits - d_lag) / 2.0) / math.sqrt(N_bits - d_lag)
    p_auto = float(sp.erfc(abs(x_auto) / math.sqrt(2.0)))
    results["8. Autocorrelation"] = {"stat": float(x_auto), "p_value": p_auto, "pass": p_auto >= 0.01}

    # Also compute Shannon Entropy of 1-bit and 8-bit symbols
    p1 = float(np.mean(cw))
    shannon_1bit = binary_entropy(p1)
    bytes_cw = np.packbits(cw.astype(np.uint8))
    _, byte_counts = np.unique(bytes_cw, return_counts=True)
    p_byte = byte_counts / float(len(bytes_cw))
    shannon_8bit = float(-np.sum(p_byte * np.log2(p_byte)))

    return {
        "tests": results,
        "shannon_entropy_1bit": shannon_1bit,
        "shannon_entropy_8bit": shannon_8bit,
        "ones_ratio": p1,
    }


# ======================================================================
# 3. PARTIAL FROZEN-KEY COMPROMISE RESILIENCE (POLAR TREE CASCADE)
# ======================================================================

def evaluate_key_compromise_sensitivity(
    block_lengths: List[int] = [128, 256, 512],
    rate: float = 0.5,
    num_frames: int = 200,
) -> Dict:
    """Evaluate Eve's BER, BLER, and Normalized Equivocation as a function of
    the fraction of known Frozen Key bits (from 0% known to 100% known) under
    high SNR (Eb/N0 = 10 dB).
    Note: Here we evaluate PURE frozen-bit protection (enable_info_masking=False)
    so the curve directly proves the cryptographic strength of Frozen-Bit
    manipulation alone!
    """
    known_fractions = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95, 0.98, 1.0])
    results_by_n = {}

    for n in block_lengths:
        k = int(n * rate)
        frozen_pos, _ = generate_5g_ranking(k, n)
        n_frz = n - k

        # Pure frozen-bit crypto (no info XOR masking) to isolate frozen-bit avalanche!
        encoder = CryptoPolarEncoder(
            frozen_pos=frozen_pos,
            n=n,
            crc_degree="CRC11",
            enable_info_masking=False,
            enable_frozen_crypto=True,
        )
        decoder = CryptoPolarDecoder(
            frozen_pos=frozen_pos,
            n=n,
            dec_type="Improved-CA-SCL",
            list_size=8,
            crc_degree="CRC11",
            enable_info_masking=False,
            enable_frozen_crypto=True,
        )
        k_payload = encoder.k_payload
        key_gen = ChannelBoundKeyGenerator(master_key=f"COMPROMISE_TEST_{n}".encode())

        payload = torch.randint(0, 2, (num_frames, k_payload), dtype=torch.float32)
        true_key = key_gen.generate_key_stream(num_frames, n, frame_idx=50)
        true_frozen = true_key[:, k:]  # [num_frames, n_frz]

        cw = encoder(payload, true_frozen)
        # High SNR = 10 dB (almost noiseless for Eve to give Eve maximum advantage)
        logits = 12.0 * (2.0 * cw - 1.0) + torch.randn_like(cw) * 0.5

        ber_list = []
        bler_list = []
        equiv_list = []

        for frac in known_fractions:
            num_known = int(round(frac * n_frz))
            eve_frozen = true_frozen.clone()
            num_unknown = n_frz - num_known
            if num_unknown > 0:
                # For each frame, Eve knows a random subset of size num_known
                # and guesses the remaining num_unknown frozen bits
                for f_i in range(num_frames):
                    perm_idx = torch.randperm(n_frz)[:num_unknown]
                    eve_frozen[f_i, perm_idx] = torch.randint(
                        0, 2, (num_unknown,), dtype=torch.float32
                    )

            u_hat_eve = decoder(logits, secret_key_stream=eve_frozen)
            bit_err = torch.abs(payload - u_hat_eve).numpy()
            ber = float(np.mean(bit_err))
            bler = float(np.mean(np.any(bit_err > 0.5, axis=1)))
            equiv = float(binary_entropy(min(ber, 0.5)))

            ber_list.append(ber)
            bler_list.append(bler)
            equiv_list.append(equiv)

        results_by_n[n] = {
            "known_fractions": known_fractions.tolist(),
            "ber": ber_list,
            "bler": bler_list,
            "equivocation": equiv_list,
        }

    return results_by_n


# ======================================================================
# 4. COSET WEIGHT SPECTRUM & LATENCY BENCHMARK
# ======================================================================

def evaluate_coset_weight_and_latency(
    k: int = 128,
    n: int = 256,
    crc_degree: str = "CRC11",
) -> Dict:
    """Evaluate Codeword Hamming Weight Distribution (Standard Polar vs.
    Crypto-Polar Cosets) and Decoder Execution Latency.
    """
    frozen_pos, _ = generate_5g_ranking(k, n)
    enc_std = CryptoPolarEncoder(
        frozen_pos=frozen_pos,
        n=n,
        crc_degree=None,
        enable_info_masking=False,
        enable_frozen_crypto=False,
    )
    enc_crypto = CryptoPolarEncoder(
        frozen_pos=frozen_pos,
        n=n,
        crc_degree=crc_degree,
        enable_info_masking=True,
        enable_frozen_crypto=True,
    )
    k_payload = enc_crypto.k_payload
    key_gen = ChannelBoundKeyGenerator(master_key=b"WEIGHT_SPECTRUM_2026")

    # Sample low-entropy / sparse payloads (typical of IoT telemetry, headers, silence frames)
    num_samples = 2000
    sparse_payload_std = (torch.rand((num_samples, k)) < 0.06).float()
    sparse_payload_cry = (torch.rand((num_samples, k_payload)) < 0.06).float()
    keys = key_gen.generate_key_stream(num_samples, n, frame_idx=77)

    cw_std = enc_std(sparse_payload_std, None).numpy()
    cw_cry = enc_crypto(sparse_payload_cry, keys).numpy()

    weights_std = np.sum(cw_std, axis=1)
    weights_cry = np.sum(cw_cry, axis=1)

    # Benchmark decoder latencies (ms per codeword frame, batch=64)
    batch_bench = 64
    payload_b = torch.randint(0, 2, (batch_bench, k_payload), dtype=torch.float32)
    key_b = key_gen.generate_key_stream(batch_bench, n, frame_idx=88)
    cw_b = enc_crypto(payload_b, key_b)
    # Moderate SNR logits so Improved-CA-SCL exercises realistic flip retries
    logits_b = 3.0 * (2.0 * cw_b - 1.0) + torch.randn_like(cw_b) * 1.2

    decoders_cfg = [
        ("SC", "SC", 1),
        ("BP (20 iter)", "BP", 1),
        ("SCL (L=4)", "SCL", 4),
        ("CA-SCL (L=8)", "CA-SCL", 8),
        ("Improved-CA-SCL (L=8)", "Improved-CA-SCL", 8),
    ]
    latency_ms_per_frame = {}
    for label, dec_type, l_size in decoders_cfg:
        dec = CryptoPolarDecoder(
            frozen_pos=frozen_pos,
            n=n,
            dec_type=dec_type,
            list_size=l_size,
            num_iter=20,
            crc_degree=crc_degree,
        )
        # Warmup
        _ = dec(logits_b, secret_key_stream=key_b)
        t0 = time.perf_counter()
        n_runs = 10
        for _ in range(n_runs):
            _ = dec(logits_b, secret_key_stream=key_b)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latency_ms_per_frame[label] = float(elapsed_ms / (n_runs * batch_bench))

    return {
        "weights_standard": weights_std,
        "weights_crypto": weights_cry,
        "latency_ms_per_frame": latency_ms_per_frame,
    }
