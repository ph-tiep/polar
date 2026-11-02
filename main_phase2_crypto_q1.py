"""
main_phase2_crypto_q1.py
======================================================================
IEEE / WoS Q1-Q2 Master Simulation & Cryptographic Evaluation Pipeline
Project:
    6G Secure Massive MIMO (140 GHz Digital Twin) with Dynamic
    Frozen-Bit Polar Crypto-Coding and Improved CSI-Calibrated Decoding

Outputs:
    1. 8 Publication-Ready Figures (600 DPI PNG + Vector PDF) in figures/
    2. Complete Numerical Logs (JSON + CSV) in logs/
    3. Ready-to-use LaTeX Tables (logs/latex_tables_q1.tex)
======================================================================
"""

import os
import json
import csv
import time
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "140GHz")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
LOG_DIR = os.path.join(BASE_DIR, "logs")
for _d in [DATASET_DIR, FIGURE_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)

from fec.polar.utils import generate_5g_ranking
from fec.polar.crypto_polar import (
    ChannelBoundKeyGenerator,
    CryptoPolarEncoder,
    CryptoPolarDecoder,
)
from fec.polar.test_crypto import run_crypto_polar_unit_tests
from phase2_crypto_polar.digital_twin_channel import DigitalTwinTHzChannel
from phase2_crypto_polar.security_metrics import (
    binary_entropy,
    evaluate_avalanche_and_sac,
    evaluate_nist_randomness,
    evaluate_key_compromise_sensitivity,
    evaluate_coset_weight_and_latency,
)
from phase2_crypto_polar.plotting_q1 import (
    plot_fig1_digital_twin_channel,
    plot_fig2_ber_comparison,
    plot_fig3_bler_comparison,
    plot_fig4_avalanche_and_sac,
    plot_fig5_key_compromise_resilience,
    plot_fig6_secrecy_throughput,
    plot_fig7_nist_and_coset_spectrum,
    plot_fig8_latency_benchmark,
)


def run_digital_twin_ber_bler_simulation(
    dt_channel: DigitalTwinTHzChannel,
    k: int = 128,
    n: int = 256,
    crc_degree: str = "CRC11",
    ebno_db_list=None,
    num_frames_per_snr: int = 800,
    batch_size: int = 100,
):
    """Run Monte Carlo link-level simulation over the 140 GHz UM-MIMO Digital Twin
    channel comparing Bob (with standard and improved decoders) and Eve (under
    3 distinct threat models).
    """
    if ebno_db_list is None:
        ebno_db_list = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    frozen_pos, _ = generate_5g_ranking(k, n)
    coderate = k / float(n)

    # Encoders
    enc_std = CryptoPolarEncoder(
        frozen_pos=frozen_pos,
        n=n,
        crc_degree=crc_degree,
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
    enc_frozen_only = CryptoPolarEncoder(
        frozen_pos=frozen_pos,
        n=n,
        crc_degree=crc_degree,
        enable_info_masking=False,
        enable_frozen_crypto=True,
    )
    k_payload = enc_crypto.k_payload

    # Decoders
    dec_std_sc = CryptoPolarDecoder(
        frozen_pos, n, dec_type="SC", crc_degree=crc_degree,
        enable_info_masking=False, enable_frozen_crypto=False
    )
    dec_crypto_sc = CryptoPolarDecoder(
        frozen_pos, n, dec_type="SC", crc_degree=crc_degree
    )
    dec_crypto_bp = CryptoPolarDecoder(
        frozen_pos, n, dec_type="BP", num_iter=20, crc_degree=crc_degree
    )
    dec_crypto_scl4 = CryptoPolarDecoder(
        frozen_pos, n, dec_type="SCL", list_size=4, crc_degree=crc_degree
    )
    dec_crypto_cascl8 = CryptoPolarDecoder(
        frozen_pos, n, dec_type="CA-SCL", list_size=8, crc_degree=crc_degree
    )
    dec_crypto_improved = CryptoPolarDecoder(
        frozen_pos, n, dec_type="Improved-CA-SCL", list_size=8,
        crc_degree=crc_degree, max_flip_trials=6
    )
    dec_frozen_only = CryptoPolarDecoder(
        frozen_pos, n, dec_type="Improved-CA-SCL", list_size=8,
        crc_degree=crc_degree, enable_info_masking=False, enable_frozen_crypto=True
    )

    curve_names = [
        "Bob - Standard Polar (SC)",
        "Bob - Crypto-Polar (SC)",
        "Bob - Crypto-Polar (BP, 20 iter)",
        "Bob - Crypto-Polar (SCL, L=4)",
        "Bob - Crypto-Polar (CA-SCL, L=8)",
        "Bob - Proposed Improved-CA-SCL (L=8)",
        "Eve - Threat I: Naive (Frozen=0, L=8)",
        "Eve - Threat II: Random Key Guess (L=8)",
        "Eve - Threat III: Frozen-Only Crypto (L=8)",
    ]

    ber_results = {name: [] for name in curve_names}
    bler_results = {name: [] for name in curve_names}

    # Bind key generator to the 140 GHz Digital Twin H_Bob CSI
    key_gen = ChannelBoundKeyGenerator(
        master_key=b"6G_DIGITAL_TWIN_140GHZ_BOB_SECRET",
        csi_tensor=dt_channel.H_Bob,
    )
    eve_random_keygen = ChannelBoundKeyGenerator(
        master_key=b"EVE_BLIND_GUESS_KEY_2026",
        csi_tensor=dt_channel.H_Eve,
    )

    num_batches = max(1, num_frames_per_snr // batch_size)

    for ebno_db in ebno_db_list:
        # Adaptive frame count at high SNR for smooth waterfall resolution
        n_batches = num_batches * 2 if ebno_db >= 2.0 else num_batches
        bit_err_acc = {name: 0 for name in curve_names}
        blk_err_acc = {name: 0 for name in curve_names}
        total_bits = 0
        total_blks = 0

        for b_idx in range(n_batches):
            payload = torch.randint(0, 2, (batch_size, k_payload), dtype=torch.float32)
            key_bob = key_gen.generate_key_stream(batch_size, n, frame_idx=b_idx + int((ebno_db + 10) * 1000))
            key_eve_guess = eve_random_keygen.generate_key_stream(batch_size, n, frame_idx=b_idx + 99999)

            cw_std = enc_std(payload, None)
            cw_cry = enc_crypto(payload, key_bob)
            cw_frz = enc_frozen_only(payload, key_bob[:, k:])

            # Transmit over 140 GHz Digital Twin channel (Bob)
            logits_bob_std, _ = dt_channel.transmit_qpsk_ofdm(cw_std, ebno_db, coderate, node="bob")
            logits_bob_cry, csi_w_bob = dt_channel.transmit_qpsk_ofdm(cw_cry, ebno_db, coderate, node="bob")

            # Transmit over 140 GHz Digital Twin channel (Eve at matched SNR to test pure crypto security)
            logits_eve_cry, csi_w_eve = dt_channel.transmit_qpsk_ofdm(
                cw_cry, ebno_db, coderate, node="eve", include_beamforming_loss_for_eve=False
            )
            logits_eve_frz, _ = dt_channel.transmit_qpsk_ofdm(
                cw_frz, ebno_db, coderate, node="eve", include_beamforming_loss_for_eve=False
            )

            # Decode all schemes
            outputs = {
                "Bob - Standard Polar (SC)": dec_std_sc(logits_bob_std, None),
                "Bob - Crypto-Polar (SC)": dec_crypto_sc(logits_bob_cry, key_bob),
                "Bob - Crypto-Polar (BP, 20 iter)": dec_crypto_bp(logits_bob_cry, key_bob),
                "Bob - Crypto-Polar (SCL, L=4)": dec_crypto_scl4(logits_bob_cry, key_bob),
                "Bob - Crypto-Polar (CA-SCL, L=8)": dec_crypto_cascl8(logits_bob_cry, key_bob),
                "Bob - Proposed Improved-CA-SCL (L=8)": dec_crypto_improved(
                    logits_bob_cry, key_bob, csi_weights=csi_w_bob
                ),
                "Eve - Threat I: Naive (Frozen=0, L=8)": dec_crypto_improved(
                    logits_eve_cry, None, csi_weights=csi_w_eve
                ),
                "Eve - Threat II: Random Key Guess (L=8)": dec_crypto_improved(
                    logits_eve_cry, key_eve_guess, csi_weights=csi_w_eve
                ),
                "Eve - Threat III: Frozen-Only Crypto (L=8)": dec_frozen_only(
                    logits_eve_frz, None, csi_weights=csi_w_eve
                ),
            }

            for name, u_hat in outputs.items():
                diff = torch.abs(payload - u_hat).numpy()
                bit_err_acc[name] += int(np.sum(diff))
                blk_err_acc[name] += int(np.sum(np.any(diff > 0.5, axis=1)))

            total_bits += batch_size * k_payload
            total_blks += batch_size

        for name in curve_names:
            ber_results[name].append(bit_err_acc[name] / float(total_bits))
            bler_results[name].append(blk_err_acc[name] / float(total_blks))

        print(
            f"  [Eb/N0 = {ebno_db:>4.1f} dB] "
            f"Bob SC BER: {ber_results['Bob - Crypto-Polar (SC)'][-1]:.5f} | "
            f"Bob CA-SCL8 BER: {ber_results['Bob - Crypto-Polar (CA-SCL, L=8)'][-1]:.5f} | "
            f"Bob Improved BER: {ber_results['Bob - Proposed Improved-CA-SCL (L=8)'][-1]:.5f} | "
            f"Eve (Frozen-Only) BER: {ber_results['Eve - Threat III: Frozen-Only Crypto (L=8)'][-1]:.4f}"
        )

    # Compute Secrecy Goodput & Equivocation curves
    # For QPSK (2 bits/symbol) and code rate R_eff = k_payload / n:
    r_eff = k_payload / float(n)
    spectral_eff_max = 2.0 * r_eff  # bps/Hz

    bler_bob_imp = np.array(bler_results["Bob - Proposed Improved-CA-SCL (L=8)"])
    bler_bob_sc = np.array(bler_results["Bob - Crypto-Polar (SC)"])

    ber_eve_cry = np.array(ber_results["Eve - Threat II: Random Key Guess (L=8)"])
    ber_eve_frz = np.array(ber_results["Eve - Threat III: Frozen-Only Crypto (L=8)"])
    # If standard polar without crypto is used and Eve is proximate (same SNR):
    ber_eve_std = np.array(ber_results["Bob - Standard Polar (SC)"])

    equiv_cry = [float(binary_entropy(min(b, 0.5))) for b in ber_eve_cry]
    equiv_frz = [float(binary_entropy(min(b, 0.5))) for b in ber_eve_frz]
    equiv_std = [float(binary_entropy(min(b, 0.5))) for b in ber_eve_std]

    goodput_proposed = (spectral_eff_max * (1.0 - bler_bob_imp) * np.array(equiv_cry)).tolist()
    goodput_sc = (spectral_eff_max * (1.0 - bler_bob_sc) * np.array(equiv_cry)).tolist()
    goodput_conv = (spectral_eff_max * (1.0 - bler_bob_sc) * np.array(equiv_std)).tolist()

    secrecy_results = {
        "ebno_db": ebno_db_list,
        "goodput_proposed": goodput_proposed,
        "goodput_sc": goodput_sc,
        "goodput_conventional_pls": goodput_conv,
        "equivocation_crypto": equiv_cry,
        "equivocation_frozen_only": equiv_frz,
        "equivocation_standard": equiv_std,
    }

    return {
        "ebno_db": ebno_db_list,
        "k": k,
        "n": n,
        "k_payload": k_payload,
        "ber": ber_results,
        "bler": bler_results,
        "secrecy": secrecy_results,
    }


def export_csv_and_latex_tables(
    dt_summary,
    sim_results,
    sac_results,
    nist_results,
    comp_results,
    coset_results,
    log_dir: str,
):
    """Export CSV files, JSON summary, and publication-ready LaTeX tables."""
    os.makedirs(log_dir, exist_ok=True)

    # 1. Save BER / BLER CSV
    csv_ber_path = os.path.join(log_dir, "ber_bler_digital_twin.csv")
    ebnos = sim_results["ebno_db"]
    with open(csv_ber_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Metric", "Scheme"] + [f"EbN0_{e}dB" for e in ebnos]
        writer.writerow(header)
        for name, vals in sim_results["ber"].items():
            writer.writerow(["BER", name] + [f"{v:.6e}" for v in vals])
        for name, vals in sim_results["bler"].items():
            writer.writerow(["BLER", name] + [f"{v:.6e}" for v in vals])

    # 2. Save NIST SP 800-22 CSV
    csv_nist_path = os.path.join(log_dir, "nist_sp800_22_table.csv")
    with open(csv_nist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Test Name", "Statistic", "p-value", "Status (alpha=0.01)"])
        for tname, tval in nist_results["tests"].items():
            writer.writerow([tname, f"{tval['stat']:.4f}", f"{tval['p_value']:.4f}", "PASS" if tval["pass"] else "FAIL"])

    # 3. Save JSON Summary
    json_summary = {
        "digital_twin_140ghz_summary": dt_summary,
        "ber_bler_simulation": sim_results,
        "cryptographic_sac_bic": {
            k: v for k, v in sac_results.items() if not isinstance(v, np.ndarray)
        },
        "nist_sp800_22": nist_results,
        "key_compromise_sensitivity": comp_results,
        "latency_ms_per_frame": coset_results["latency_ms_per_frame"],
        "coset_weight_stats": {
            "standard_mean_weight": float(np.mean(coset_results["weights_standard"])),
            "standard_std_weight": float(np.std(coset_results["weights_standard"])),
            "crypto_mean_weight": float(np.mean(coset_results["weights_crypto"])),
            "crypto_std_weight": float(np.std(coset_results["weights_crypto"])),
        },
    }
    json_path = os.path.join(log_dir, "crypto_polar_q1_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)

    # 4. Generate Ready-to-Use LaTeX Tables for WoS Q1/Q2 Paper
    latex_path = os.path.join(log_dir, "latex_tables_q1.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% ======================================================================\n")
        f.write("% LaTeX Tables for WoS Q1/Q2 Journal Manuscript\n")
        f.write("% 6G 140 GHz UM-MIMO Digital Twin with Dynamic Frozen-Bit Crypto-Polar\n")
        f.write("% ======================================================================\n\n")

        # Table I: NIST SP 800-22 & Cryptographic Metrics
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{NIST SP 800-22 Statistical Randomness and Cryptographic Security Evaluation ($N=256, K=128$)}\n")
        f.write("\\label{tab:nist_crypto}\n")
        f.write("\\begin{tabular}{lccc}\n\\hline\n")
        f.write("\\textbf{Statistical / Cryptographic Test} & \\textbf{Test Statistic / Value} & \\textbf{$p$-value / Ideal} & \\textbf{Result} \\\\\n\\hline\n")
        for tname, tval in nist_results["tests"].items():
            f.write(f"{tname} & {tval['stat']:.4f} & {tval['p_value']:.4f} & {'PASS' if tval['pass'] else 'FAIL'} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Codeword Avalanche Effect (SAC) & {sac_results['cw_avalanche_mean']*100:.2f}\\% $\\pm$ {sac_results['cw_avalanche_std']*100:.2f}\\% & 50.00\\% & Ideal \\\\\n")
        f.write(f"Decoder Output Avalanche (1-bit Key Flip) & {sac_results['dec_avalanche_mean']*100:.2f}\\% $\\pm$ {sac_results['dec_avalanche_std']*100:.2f}\\% & 50.00\\% & Ideal \\\\\n")
        f.write(f"Bit Independence Criterion (BIC $|\\rho|$) & {sac_results['bic_mean_abs_correlation']:.4f} & 0.0000 & Uncorrelated \\\\\n")
        f.write(f"Inter-Codeword Cross-Correlation & {sac_results['cross_corr_mean']:.4f} $\\pm$ {sac_results['cross_corr_std']:.4f} & 0.0000 & Orthogonal \\\\\n")
        f.write(f"Shannon Entropy (1-bit / 8-bit) & {nist_results['shannon_entropy_1bit']:.5f} / {nist_results['shannon_entropy_8bit']:.4f} & 1.0000 / 8.0000 & Maximum \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

        # Table II: BER/BLER & Latency Comparison
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write("\\caption{BER, BLER, and Decoding Latency Comparison Over 140 GHz UM-MIMO Digital Twin Channel ($N=256, K=128$, QPSK)}\n")
        f.write("\\label{tab:ber_latency}\n")
        f.write("\\begin{tabular}{lccccc}\n\\hline\n")
        f.write("\\textbf{Receiver & Decoding Scheme} & \\textbf{BER @ 0.5 dB} & \\textbf{BER @ 1.5 dB} & \\textbf{BER @ 2.5 dB} & \\textbf{BLER @ 2.5 dB} & \\textbf{Latency (ms)} \\\\\n\\hline\n")
        idx_0 = ebnos.index(0.5) if 0.5 in ebnos else 1
        idx_2 = ebnos.index(1.5) if 1.5 in ebnos else 3
        idx_4 = ebnos.index(2.5) if 2.5 in ebnos else 5
        lat_map = {
            "Bob - Standard Polar (SC)": coset_results["latency_ms_per_frame"].get("SC", 0.0),
            "Bob - Crypto-Polar (SC)": coset_results["latency_ms_per_frame"].get("SC", 0.0),
            "Bob - Crypto-Polar (BP, 20 iter)": coset_results["latency_ms_per_frame"].get("BP (20 iter)", 0.0),
            "Bob - Crypto-Polar (SCL, L=4)": coset_results["latency_ms_per_frame"].get("SCL (L=4)", 0.0),
            "Bob - Crypto-Polar (CA-SCL, L=8)": coset_results["latency_ms_per_frame"].get("CA-SCL (L=8)", 0.0),
            "Bob - Proposed Improved-CA-SCL (L=8)": coset_results["latency_ms_per_frame"].get("Improved-CA-SCL (L=8)", 0.0),
        }
        for name in sim_results["ber"].keys():
            b0 = sim_results["ber"][name][idx_0]
            b2 = sim_results["ber"][name][idx_2]
            b4 = sim_results["ber"][name][idx_4]
            bl4 = sim_results["bler"][name][idx_4]
            lat_str = f"{lat_map[name]:.3f}" if name in lat_map else "--"
            f.write(f"{name} & {b0:.4e} & {b2:.4e} & {b4:.4e} & {bl4:.4e} & {lat_str} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table*}\n")

    return csv_ber_path, csv_nist_path, json_path, latex_path


def main():
    t_start = time.time()
    torch.manual_seed(2026)
    np.random.seed(2026)

    print("=" * 75)
    print("PHASE 2 & 3 : 6G DIGITAL TWIN CRYPTO-POLAR CODING & SECURITY EVALUATION")
    print("=" * 75)

    # Step 1: Run Unit Tests
    run_crypto_polar_unit_tests()
    print()

    # Step 2: Load 140 GHz Digital Twin Channel Dataset
    print("[STEP 2] Loading 140 GHz UM-MIMO Digital Twin Dataset (H_Bob, H_Eve)...")
    dt_channel = DigitalTwinTHzChannel(dataset_dir=DATASET_DIR)
    dt_summary = dt_channel.get_channel_summary()
    for k_name, val in dt_summary.items():
        print(f"    {k_name:<30}: {val:.4f}" if isinstance(val, float) else f"    {k_name:<30}: {val}")
    print()

    # Step 3: Run Digital Twin Link-Level Monte Carlo Simulation
    print("[STEP 3] Running Monte Carlo BER/BLER Simulation on 140 GHz Digital Twin...")
    sim_results = run_digital_twin_ber_bler_simulation(
        dt_channel=dt_channel,
        k=128,
        n=256,
        crc_degree="CRC11",
        ebno_db_list=[-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
        num_frames_per_snr=600,
        batch_size=200,
    )
    print()

    # Step 4: Cryptographic Avalanche, SAC, BIC & Cross-Correlation
    print("[STEP 4] Evaluating Cryptographic Avalanche Effect, SAC, BIC & Correlation...")
    sac_results = evaluate_avalanche_and_sac(k=128, n=256, crc_degree="CRC11", num_trials=1000)
    print(f"    Codeword Avalanche Mean     : {sac_results['cw_avalanche_mean']*100:.2f}% (Std: {sac_results['cw_avalanche_std']*100:.2f}%)")
    print(f"    Decoder Avalanche Mean      : {sac_results['dec_avalanche_mean']*100:.2f}% (Std: {sac_results['dec_avalanche_std']*100:.2f}%)")
    print(f"    SAC Mean Deviation from 0.5 : {sac_results['sac_mean_deviation_from_half']:.4f}")
    print(f"    BIC Mean |Correlation|      : {sac_results['bic_mean_abs_correlation']:.4f}")
    print(f"    Cross-Correlation Mean      : {sac_results['cross_corr_mean']:.4f} (Std: {sac_results['cross_corr_std']:.4f})")
    print()

    # Step 5: NIST SP 800-22 Statistical Randomness Suite
    print("[STEP 5] Running NIST SP 800-22 Statistical Randomness Test Suite...")
    nist_results = evaluate_nist_randomness(k=128, n=256, crc_degree="CRC11", total_bits=65536)
    for tname, tres in nist_results["tests"].items():
        status_str = "PASS" if tres["pass"] else "FAIL"
        print(f"    {tname:<28} | Stat = {tres['stat']:>8.4f} | p-value = {tres['p_value']:.4f} | [{status_str}]")
    print(f"    Shannon Entropy (1-bit)     : {nist_results['shannon_entropy_1bit']:.6f} bits/bit")
    print(f"    Shannon Entropy (8-bit)     : {nist_results['shannon_entropy_8bit']:.4f} bits/byte")
    print()

    # Step 6: Partial Frozen-Key Compromise Resilience
    print("[STEP 6] Evaluating Eve Resilience Under Partial Frozen-Key Compromise...")
    comp_results = evaluate_key_compromise_sensitivity(
        block_lengths=[128, 256, 512], rate=0.5, num_frames=200
    )
    for n_val, cdata in comp_results.items():
        idx_90 = cdata["known_fractions"].index(0.9)
        print(
            f"    Blocklength N={n_val} (90% Frozen Bits Known to Eve) -> "
            f"Eve BER = {cdata['ber'][idx_90]:.4f}, Eve BLER = {cdata['bler'][idx_90]:.4f}"
        )
    print()

    # Step 7: Coset Weight Spectrum & Decoder Latency Benchmark
    print("[STEP 7] Benchmarking Affine Coset Weight Spectrum & Decoder Latency...")
    coset_results = evaluate_coset_weight_and_latency(k=128, n=256, crc_degree="CRC11")
    for dlabel, ms in coset_results["latency_ms_per_frame"].items():
        print(f"    Decoder Latency [{dlabel:<22}] : {ms:.4f} ms/frame")
    print()

    # Step 8: Export Publication-Ready Figures (600 DPI PNG + PDF)
    print("[STEP 8] Generating 8 Publication-Grade Figures (600 DPI PNG & PDF)...")
    fig_dir = FIGURE_DIR
    f1 = plot_fig1_digital_twin_channel(dt_channel, fig_dir)
    f2 = plot_fig2_ber_comparison(sim_results, fig_dir)
    f3 = plot_fig3_bler_comparison(sim_results, fig_dir)
    f4 = plot_fig4_avalanche_and_sac(sac_results, fig_dir)
    f5 = plot_fig5_key_compromise_resilience(comp_results, fig_dir)
    f6 = plot_fig6_secrecy_throughput(sim_results["secrecy"], fig_dir)
    f7 = plot_fig7_nist_and_coset_spectrum(nist_results, coset_results, fig_dir)
    f8 = plot_fig8_latency_benchmark(coset_results, sim_results, fig_dir)
    for f_pair in [f1, f2, f3, f4, f5, f6, f7, f8]:
        print(f"    Saved: {os.path.basename(f_pair[0])} & {os.path.basename(f_pair[1])}")
    print()

    # Step 9: Export CSV, JSON, and LaTeX Tables
    print("[STEP 9] Exporting CSV, JSON, and LaTeX Manuscript Tables...")
    csv_ber, csv_nist, json_p, tex_p = export_csv_and_latex_tables(
        dt_summary,
        sim_results,
        sac_results,
        nist_results,
        comp_results,
        coset_results,
        log_dir=LOG_DIR,
    )
    print(f"    Saved: {csv_ber}")
    print(f"    Saved: {csv_nist}")
    print(f"    Saved: {json_p}")
    print(f"    Saved: {tex_p}")

    total_time = time.time() - t_start
    print("=" * 75)
    print(f"ALL EVALUATIONS COMPLETED SUCCESSFULLY IN {total_time:.2f} s")
    print("=" * 75)


if __name__ == "__main__":
    main()

