"""
validation_metrics.py
===============================================================
IEEE Phase 1 - Channel Validation Metrics (140 GHz UM-MIMO)

Compatible with:
    TensorFlow 2.21
    Sionna RT 2.0.1

Purpose
-------
Validate the generated Digital Twin channel before exporting it
to MATLAB Phase 2.

Metrics
-------
1. Received Power
2. Path Loss
3. Power Delay Profile (PDP)
4. Mean Excess Delay
5. RMS Delay Spread
6. Number of Multipath Components
7. LOS Dominance Ratio
===============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from phase1_digital_twin import config


# ===============================================================
# BASIC METRICS
# ===============================================================

def received_power(H):
    """Average received channel power."""
    return np.mean(np.abs(H) ** 2)


def received_power_db(H):
    return 10 * np.log10(received_power(H) + 1e-30)


def estimate_path_loss(H):
    """
    Estimate path loss from CSI power.

    Returns
    -------
    Path loss (dB)
    """
    return -received_power_db(H)


# ===============================================================
# POWER DELAY PROFILE
# ===============================================================

def power_delay_profile(a, tau, rx_index=0, tx_index=0):
    """
    Compute PDP from CIR.

    Parameters
    ----------
    a : complex CIR gain tensor
    tau : CIR delay tensor

    Returns
    -------
    delays_ns
    powers_db
    """

    a_np = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
    tau_np = tau.numpy() if hasattr(tau, "numpy") else np.asarray(tau)

    # Select receiver / transmitter
    gains = a_np[rx_index, tx_index]
    delays = tau_np[rx_index, tx_index]

    # Average over antennas
    power = np.mean(np.abs(gains) ** 2, axis=(0,1))

    delays_ns = delays * 1e9
    powers_db = 10*np.log10(power + 1e-30)

    return delays_ns, powers_db


# ===============================================================
# DELAY SPREAD
# ===============================================================

def delay_statistics(a, tau, rx_index=0, tx_index=0):
    """
    Compute Mean Excess Delay and RMS Delay Spread.
    """

    delays_ns, powers_db = power_delay_profile(a, tau, rx_index, tx_index)

    power_linear = 10 ** (powers_db / 10)

    power_linear /= np.sum(power_linear)

    mean_delay = np.sum(delays_ns * power_linear)

    rms_delay = np.sqrt(
        np.sum(((delays_ns - mean_delay) ** 2) * power_linear)
    )

    return mean_delay, rms_delay


# ===============================================================
# LOS RATIO
# ===============================================================

def los_ratio(a, rx_index=0, tx_index=0):
    """
    Approximate LOS power ratio.

    First path is assumed LOS.
    """

    a_np = a.numpy() if hasattr(a, "numpy") else np.asarray(a)

    gains = a_np[rx_index, tx_index]

    power = np.mean(np.abs(gains)**2, axis=(0,1))

    los_power = power[0]
    total_power = np.sum(power)

    return los_power / (total_power + 1e-30)


# ===============================================================
# SUMMARY
# ===============================================================

def evaluate_channel(H_Bob, H_Eve, a, tau, verbose=True):
    """
    Evaluate Digital Twin channel quality.
    """

    metrics = {}

    metrics["Bob Power (dB)"] = received_power_db(H_Bob)
    metrics["Eve Power (dB)"] = received_power_db(H_Eve)

    metrics["Bob PathLoss (dB)"] = estimate_path_loss(H_Bob)
    metrics["Eve PathLoss (dB)"] = estimate_path_loss(H_Eve)

    mean_delay, rms_delay = delay_statistics(a, tau)

    metrics["Mean Delay (ns)"] = mean_delay
    metrics["RMS Delay Spread (ns)"] = rms_delay

    metrics["LOS Ratio"] = los_ratio(a)

    delays_ns, powers_db = power_delay_profile(a, tau)

    metrics["Number of Paths"] = len(delays_ns)

    if verbose:

        print("="*60)
        print("CHANNEL VALIDATION REPORT")
        print("="*60)

        for k,v in metrics.items():

            if isinstance(v,float):
                print(f"{k:<28}: {v:.4f}")
            else:
                print(f"{k:<28}: {v}")

        print("="*60)

    return metrics


# ===============================================================
# FIGURE EXPORT (IEEE 600 DPI)
# ===============================================================

def save_validation_figures(a, tau, output_dir=None):

    if output_dir is None:
        output_dir = config.DATASET_DIR

    os.makedirs(output_dir, exist_ok=True)

    delays_ns, powers_db = power_delay_profile(a, tau)

    # ----------------------------------------------------------
    # PDP Figure
    # ----------------------------------------------------------

    plt.figure(figsize=(6,4))

    plt.stem(delays_ns, powers_db, basefmt=" ")

    plt.grid(True, alpha=0.3)

    plt.xlabel("Delay (ns)")
    plt.ylabel("Power (dB)")
    plt.title("Power Delay Profile at 140 GHz")

    plt.tight_layout()

    png = os.path.join(output_dir, "PDP_140GHz.png")
    pdf = os.path.join(output_dir, "PDP_140GHz.pdf")

    plt.savefig(png, dpi=600)
    plt.savefig(pdf)

    plt.close()

    # ----------------------------------------------------------
    # CDF Delay Spread
    # ----------------------------------------------------------

    plt.figure(figsize=(5.5,4))

    sorted_delay = np.sort(delays_ns)
    cdf = np.arange(1,len(sorted_delay)+1)/len(sorted_delay)

    plt.plot(sorted_delay, cdf, linewidth=2)

    plt.grid(True, alpha=0.3)

    plt.xlabel("Delay (ns)")
    plt.ylabel("CDF")
    plt.title("CDF of Multipath Delay")

    plt.tight_layout()

    png2 = os.path.join(output_dir,"Delay_CDF.png")
    pdf2 = os.path.join(output_dir,"Delay_CDF.pdf")

    plt.savefig(png2,dpi=600)
    plt.savefig(pdf2)

    plt.close()

    print("Validation figures exported.")