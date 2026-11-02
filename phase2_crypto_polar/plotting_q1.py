"""
plotting_q1.py
======================================================================
Publication-Grade (WoS Q1/Q2 IEEE Transactions Style) Figure Generator
Exports 600 DPI PNG and vector PDF figures for the manuscript.
======================================================================
"""

import os
from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_ieee_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    })


def _save_both(fig, output_dir: str, basename: str):
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{basename}.png")
    pdf_path = os.path.join(output_dir, f"{basename}.pdf")
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_fig1_digital_twin_channel(dt_channel, output_dir: str):
    """Fig 1: 140 GHz UM-MIMO Digital Twin Subcarrier Channel Gain & Spatial Correlation."""
    set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)

    sc_idx = np.arange(dt_channel.n_sc)
    bob_rel_db = 10.0 * np.log10(dt_channel.g_bob_norm + 1e-12)
    eve_rel_db = 10.0 * np.log10(dt_channel.g_eve_fading_only + 1e-12)
    ax1.plot(sc_idx, bob_rel_db, color="#0055A4", label="Bob Normalized Multipath Gain Profile ($256\\times 16$ SVD-BF)", linewidth=1.5)
    ax1.plot(sc_idx, eve_rel_db, color="#D62728", linestyle="--", label="Eve Normalized Multipath Gain Profile ($256\\times 16$ MRC)", linewidth=1.4)
    ax1.set_ylabel("Relative Subcarrier Gain (dB)")
    ax1.set_title("140 GHz UM-MIMO Digital Twin Channel Response Across 512 OFDM Subcarriers")
    ax1.grid(True)
    ax1.legend(loc="lower right", frameon=True)

    ax2.plot(sc_idx, dt_channel.spatial_corr, color="#2CA02C", linewidth=1.5, label="Bob-Eve Dominant Subspace Correlation $|\\mathbf{w}_{\\mathrm{Bob}}^H \\mathbf{w}_{\\mathrm{Eve}}|$")
    ax2.axhline(np.mean(dt_channel.spatial_corr), color="#1F77B4", linestyle=":", linewidth=1.4, label=f"Mean Spatial Correlation = {np.mean(dt_channel.spatial_corr):.4f} (High-Leakage Regime)")
    ax2.set_xlabel("OFDM Subcarrier Index $m \\in [0, 511]$ (Bandwidth = 2 GHz at $f_c = 140$ GHz)")
    ax2.set_ylabel("Spatial Correlation")
    ax2.set_ylim(0.75, 0.92)
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    return _save_both(fig, output_dir, "Fig1_DigitalTwin_SubTHz_Channel_Profile")


def plot_fig2_ber_comparison(sim_results: Dict, output_dir: str):
    """Fig 2: BER vs Eb/N0 comparing Bob (across decoders) and Eve (threat models)."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    ebno = np.array(sim_results["ebno_db"])

    styles = {
        "Bob - Standard Polar (SC)": ("#7F7F7F", "v", "--"),
        "Bob - Crypto-Polar (SC)": ("#1F77B4", "o", "-"),
        "Bob - Crypto-Polar (BP, 20 iter)": ("#9467BD", "s", "-."),
        "Bob - Crypto-Polar (SCL, L=4)": ("#FF7F0E", "^", "-"),
        "Bob - Crypto-Polar (CA-SCL, L=8)": ("#2CA02C", "D", "-"),
        "Bob - Proposed Improved-CA-SCL (L=8)": ("#003366", "*", "-"),
        "Eve - Threat I: Naive (Frozen=0, L=8)": ("#D62728", "X", "--"),
        "Eve - Threat II: Random Key Guess (L=8)": ("#8C564B", "P", ":"),
        "Eve - Threat III: Frozen-Only Crypto (L=8)": ("#E377C2", "h", "-."),
    }

    for name, curve in sim_results["ber"].items():
        color, marker, ls = styles.get(name, ("#000000", "o", "-"))
        y = np.clip(np.array(curve, dtype=float), 1e-6, 1.0)
        lw = 2.3 if "Proposed" in name else 1.6
        ms = 7.5 if "Proposed" in name else 5.5
        ax.semilogy(ebno, y, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=ms, label=name)

    ax.axhline(0.5, color="black", linestyle=":", alpha=0.6, linewidth=1.0)
    ax.set_xlabel("$E_b / N_0$ (dB) at Receiver")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("BER Performance Over 140 GHz UM-MIMO Digital Twin Channel ($N=256, R=1/2$)")
    ax.set_ylim(2e-5, 1.0)
    ax.grid(True, which="both")
    ax.legend(loc="lower left", frameon=True, fontsize=7.8)

    fig.tight_layout()
    return _save_both(fig, output_dir, "Fig2_BER_Comparison_Decoders_DigitalTwin")


def plot_fig3_bler_comparison(sim_results: Dict, output_dir: str):
    """Fig 3: BLER vs Eb/N0 comparing Bob and Eve across decoders."""
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    ebno = np.array(sim_results["ebno_db"])

    styles = {
        "Bob - Standard Polar (SC)": ("#7F7F7F", "v", "--"),
        "Bob - Crypto-Polar (SC)": ("#1F77B4", "o", "-"),
        "Bob - Crypto-Polar (BP, 20 iter)": ("#9467BD", "s", "-."),
        "Bob - Crypto-Polar (SCL, L=4)": ("#FF7F0E", "^", "-"),
        "Bob - Crypto-Polar (CA-SCL, L=8)": ("#2CA02C", "D", "-"),
        "Bob - Proposed Improved-CA-SCL (L=8)": ("#003366", "*", "-"),
        "Eve - Threat I: Naive (Frozen=0, L=8)": ("#D62728", "X", "--"),
        "Eve - Threat II: Random Key Guess (L=8)": ("#8C564B", "P", ":"),
        "Eve - Threat III: Frozen-Only Crypto (L=8)": ("#E377C2", "h", "-."),
    }

    for name, curve in sim_results["bler"].items():
        color, marker, ls = styles.get(name, ("#000000", "o", "-"))
        y = np.clip(np.array(curve, dtype=float), 1e-5, 1.05)
        lw = 2.3 if "Proposed" in name else 1.6
        ms = 7.5 if "Proposed" in name else 5.5
        ax.semilogy(ebno, y, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=ms, label=name)

    ax.set_xlabel("$E_b / N_0$ (dB) at Receiver")
    ax.set_ylabel("Block Error Rate (BLER)")
    ax.set_title("Block Error Rate (BLER) Over 140 GHz UM-MIMO Digital Twin Channel ($N=256, K=128$)")
    ax.set_ylim(1e-4, 1.2)
    ax.grid(True, which="both")
    ax.legend(loc="lower left", frameon=True, fontsize=7.8)

    fig.tight_layout()
    return _save_both(fig, output_dir, "Fig3_BLER_Comparison_Decoders_DigitalTwin")


def plot_fig4_avalanche_and_sac(sac_results: Dict, output_dir: str):
    """Fig 4: Cryptographic Avalanche Histogram, SAC Bit Profile, and Cross-Correlation."""
    set_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))

    # Subplot (a): Avalanche Histogram (Codeword & Decoded Output)
    ax = axes[0]
    cw_pct = sac_results["cw_flip_ratios"] * 100.0
    dec_pct = sac_results["dec_flip_ratios"] * 100.0
    ax.hist(cw_pct, bins=24, density=True, alpha=0.65, color="#1F77B4", edgecolor="black", label=f"Codeword ($\\mu={np.mean(cw_pct):.2f}\\%$)")
    ax.hist(dec_pct, bins=24, density=True, alpha=0.55, color="#D62728", edgecolor="black", label=f"Decoded ($\\mu={np.mean(dec_pct):.2f}\\%$)")
    ax.axvline(50.0, color="#2CA02C", linestyle="--", linewidth=1.8, label="Ideal SAC (50.0%)")
    ax.set_xlabel("Bit Flip Percentage (%)")
    ax.set_ylabel("Probability Density")
    ax.set_title("(a) Avalanche Effect Distribution")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7.5)

    # Subplot (b): Strict Avalanche Criterion (SAC) Across Bit Indices
    ax = axes[1]
    bit_idx = np.arange(len(sac_results["sac_bit_probs"]))
    ax.plot(bit_idx, sac_results["sac_bit_probs"], color="#0055A4", linewidth=1.0, alpha=0.85, label="Codeword Bit Flip Prob. $P(\\Delta x_j = 1)$")
    ax.axhline(0.5, color="#D62728", linestyle="--", linewidth=1.6, label="Ideal SAC Target = 0.50")
    ax.set_ylim(0.38, 0.62)
    ax.set_xlabel("Codeword Bit Index $j \\in [0, N-1]$")
    ax.set_ylabel("Flip Probability")
    ax.set_title("(b) Strict Avalanche Criterion (SAC)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7.5)

    # Subplot (c): Cross-Correlation Distribution
    ax = axes[2]
    cc = sac_results["cross_corrs"]
    ax.hist(cc, bins=25, density=True, color="#2CA02C", alpha=0.7, edgecolor="black", label=f"$\\mu={np.mean(cc):.4f}, \\sigma={np.std(cc):.4f}$")
    ax.axvline(0.0, color="#D62728", linestyle="--", linewidth=1.6, label="Uncorrelated ($\\rho = 0$)")
    ax.set_xlabel("Normalized Cross-Correlation $\\rho(\\mathbf{x}, \\mathbf{x}')$")
    ax.set_ylabel("Probability Density")
    ax.set_title("(c) Inter-Codeword Cross-Correlation")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7.5)

    fig.tight_layout()
    return _save_both(fig, output_dir, "Fig4_Avalanche_SAC_and_Correlation")


def plot_fig5_key_compromise_resilience(comp_results: Dict, output_dir: str):
    """Fig 5: Eve's BER and BLER vs Percentage of Compromised Frozen Key Bits."""
    set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))

    colors = {128: "#FF7F0E", 256: "#1F77B4", 512: "#D62728"}
    markers = {128: "s", 256: "o", 512: "^"}

    for n, data in comp_results.items():
        pct = np.array(data["known_fractions"]) * 100.0
        ax1.plot(
            pct,
            data["ber"],
            color=colors.get(n, "#000000"),
            marker=markers.get(n, "o"),
            label=f"Pure Frozen-Crypto ($N={n}, |\\mathcal{{F}}|={n//2}$)",
        )
        ax2.plot(
            pct,
            data["bler"],
            color=colors.get(n, "#000000"),
            marker=markers.get(n, "o"),
            label=f"Pure Frozen-Crypto ($N={n}, |\\mathcal{{F}}|={n//2}$)",
        )

    ax1.axhline(0.5, color="black", linestyle=":", label="Ideal Confusion Limit (BER = 0.5)")
    ax1.set_xlabel("Percentage of Frozen Key Bits Known to Eve (%)")
    ax1.set_ylabel("Eve's Bit Error Rate (BER)")
    ax1.set_title("(a) Eve BER vs. Partial Frozen-Key Compromise")
    ax1.set_ylim(-0.02, 0.56)
    ax1.grid(True)
    ax1.legend(loc="lower left", fontsize=8)

    ax2.axhline(1.0, color="black", linestyle=":", label="Complete Block Rejection (BLER = 1.0)")
    ax2.set_xlabel("Percentage of Frozen Key Bits Known to Eve (%)")
    ax2.set_ylabel("Eve's Block Error Rate (BLER)")
    ax2.set_title("(b) Eve BLER vs. Partial Frozen-Key Compromise")
    ax2.set_ylim(-0.02, 1.08)
    ax2.grid(True)
    ax2.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    return _save_both(fig, output_dir, "Fig5_Eve_Key_Compromise_Sensitivity")


def plot_fig6_secrecy_throughput(sec_results: Dict, output_dir: str):
    """Fig 6: Effective Secure Goodput (bps/Hz) & Eve Equivocation Rate vs Eb/N0."""
    set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2))

    ebno = np.array(sec_results["ebno_db"])

    ax1.plot(ebno, sec_results["goodput_proposed"], color="#003366", marker="*", markersize=7.5, linewidth=2.2, label="Proposed Crypto-Polar + Improved-CA-SCL")
    ax1.plot(ebno, sec_results["goodput_sc"], color="#1F77B4", marker="o", linestyle="--", label="Crypto-Polar + Standard SC")
    ax1.plot(ebno, sec_results["goodput_conventional_pls"], color="#D62728", marker="X", linestyle="-.", label="Conventional Polar (No Frozen Crypto, Proximate Eve)")
    ax1.set_xlabel("$E_b / N_0$ (dB)")
    ax1.set_ylabel("Effective Secure Goodput (bps/Hz)")
    ax1.set_title("(a) Effective Secrecy Spectral Efficiency (QPSK)")
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=7.8)

    ax2.plot(ebno, sec_results["equivocation_crypto"], color="#2CA02C", marker="D", linewidth=2.0, label="Proposed Crypto-Polar (Normalized Equivocation $\\Delta_e$)")
    ax2.plot(ebno, sec_results["equivocation_frozen_only"], color="#9467BD", marker="s", linestyle="--", label="Frozen-Bit Crypto Only ($\\mathbf{u}_{\\mathcal{F}} = \\mathbf{k}_{\\mathcal{F}}$)")
    ax2.plot(ebno, sec_results["equivocation_standard"], color="#D62728", marker="v", linestyle="-.", label="Standard Polar ($\\mathbf{u}_{\\mathcal{F}} = \\mathbf{0}$, Proximate Eve)")
    ax2.axhline(1.0, color="black", linestyle=":", label="Perfect Secrecy Limit ($\\Delta_e = 1.0$)")
    ax2.set_xlabel("$E_b / N_0$ (dB)")
    ax2.set_ylabel("Normalized Eve Equivocation $H(\\mathbf{m}|\\hat{\\mathbf{m}}_{\\mathrm{Eve}})/K$")
    ax2.set_title("(b) Eavesdropper Equivocation Rate vs. SNR")
    ax2.set_ylim(-0.05, 1.1)
    ax2.grid(True)
    ax2.legend(loc="center left", fontsize=7.8)

    fig.tight_layout()
    return _save_both(fig, output_dir, "Fig6_Secrecy_Throughput_and_Equivocation")


def plot_fig7_nist_and_coset_spectrum(nist_results: Dict, coset_results: Dict, output_dir: str):
    """Fig 7: NIST SP 800-22 p-values and Affine Coset Weight Spectrum."""
    set_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))

    # Subplot (a): NIST SP 800-22 p-values
    test_names = [k.split(". ")[1] for k in nist_results["tests"].keys()]
    p_vals = [v["p_value"] for v in nist_results["tests"].values()]
    y_pos = np.arange(len(test_names))

    bars = ax1.barh(y_pos, p_vals, color="#1F77B4", edgecolor="black", height=0.58, alpha=0.85)
    ax1.axvline(0.01, color="#D62728", linestyle="--", linewidth=1.8, label="Significance $\\alpha = 0.01$")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(test_names)
    ax1.set_xlim(0, 1.15)
    ax1.set_xlabel("NIST SP 800-22 $p$-value")
    ax1.set_title("(a) NIST SP 800-22 Statistical Randomness Suite")
    ax1.grid(True, axis="x")
    ax1.legend(loc="upper right", fontsize=8)
    for bar, pv in zip(bars, p_vals):
        ax1.text(pv + 0.02, bar.get_y() + bar.get_height() / 2.0, f"{pv:.3f}", va="center", fontsize=8)

    # Subplot (b): Codeword Weight Distribution
    w_std = coset_results["weights_standard"]
    w_cry = coset_results["weights_crypto"]
    ax2.hist(w_std, bins=28, density=True, alpha=0.6, color="#FF7F0E", edgecolor="black", label=f"Standard Polar ($\\mu={np.mean(w_std):.1f}$)")
    ax2.hist(w_cry, bins=28, density=True, alpha=0.65, color="#0055A4", edgecolor="black", label=f"Crypto-Polar Cosets ($\\mu={np.mean(w_cry):.1f}$)")
    ax2.axvline(128, color="#D62728", linestyle="--", linewidth=1.6, label="Binomial Center ($N/2 = 128$)")
    ax2.set_xlabel("Codeword Hamming Weight $w_H(\\mathbf{x})$ ($N=256$)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("(b) Affine Coset Weight Spectrum Stabilization")
    ax2.grid(True)
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    return _save_both(fig, output_dir, "Fig7_NIST_Randomness_and_Coset_Spectrum")


def plot_fig8_latency_benchmark(coset_results: Dict, sim_results: Dict, output_dir: str):
    """Fig 8: Computational Latency vs Coding Gain Trade-off."""
    set_ieee_style()
    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))

    lat_dict = coset_results["latency_ms_per_frame"]
    labels = list(lat_dict.keys())
    latencies = [lat_dict[k] for k in labels]

    x = np.arange(len(labels))
    colors = ["#7F7F7F", "#9467BD", "#FF7F0E", "#2CA02C", "#003366"]
    bars = ax1.bar(x, latencies, width=0.52, color=colors, edgecolor="black", alpha=0.88)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Execution Latency per Codeword (ms)")
    ax1.set_title("Decoder Computational Latency Benchmark ($N=256, K=128$, Batch=64)")
    ax1.grid(True, axis="y")

    for bar, val in zip(bars, latencies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + max(latencies) * 0.02,
            f"{val:.3f} ms",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax1.set_ylim(0, max(latencies) * 1.18)

    fig.tight_layout()
    return _save_both(fig, output_dir, "Fig8_Complexity_and_Latency_Benchmark")

