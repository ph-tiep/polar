"""
save_dataset.py
======================================================================
IEEE Phase 1 : Export Digital Twin Dataset for MATLAB Phase 2

Compatible:
    TensorFlow 2.21
    Sionna RT 2.0.1

Outputs
-------
dataset/140GHz/

    H_Bob.npy
    H_Eve.npy
    CIR_Gain.npy
    CIR_Delay.npy
    CSI_Dataset_140GHz.mat
    Metadata.npz

The exported MAT file is fully compatible with MATLAB R2022b+.
======================================================================
"""

import os
import platform
import re
import numpy as np
import scipy.io as sio
import tensorflow as tf
import sionna.rt as rt

from phase1_digital_twin import config


# ==============================================================
# UTILITIES
# ==============================================================

def ensure_numpy(x):
    """Convert TensorFlow tensor into NumPy array."""
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def power_db(H):
    """Average received power in dB."""
    return float(10*np.log10(np.mean(np.abs(H)**2)+1e-30))


# ==============================================================
# EXPORT FUNCTION
# ==============================================================

def export_dataset(H_Bob,
                   H_Eve,
                   a,
                   tau,
                   validation_metrics=None,
                   verbose=True):
    """
    Export CSI/CIR dataset for MATLAB and Python.

    Parameters
    ----------
    H_Bob : ndarray (16,256,512)

    H_Eve : ndarray (16,256,512)

    a : CIR gain tensor

    tau : CIR delay tensor

    validation_metrics : dict
        Output of validation_metrics.evaluate_channel()
    """

    os.makedirs(config.DATASET_DIR, exist_ok=True)

    if verbose:
        print("="*65)
        print("STEP 5 : EXPORT DATASET")
        print("="*65)

    # ----------------------------------------------------------
    # Convert tensors
    # ----------------------------------------------------------

    H_Bob = np.asarray(H_Bob, dtype=np.complex64)
    H_Eve = np.asarray(H_Eve, dtype=np.complex64)

    a = ensure_numpy(a).astype(np.complex64)
    tau = ensure_numpy(tau).astype(np.float32)

    # ----------------------------------------------------------
    # Sanity check
    # ----------------------------------------------------------

    assert np.isfinite(H_Bob).all(), "H_Bob contains NaN."
    assert np.isfinite(H_Eve).all(), "H_Eve contains NaN."
    assert np.isfinite(a).all(), "CIR gain contains NaN."
    assert np.isfinite(tau).all(), "CIR delay contains NaN."

    # ----------------------------------------------------------
    # Frequency grid
    # ----------------------------------------------------------

    frequency_grid = (
        config.CARRIER_FREQ
        + (
            np.arange(config.NUM_SUBCARRIERS)
            - config.NUM_SUBCARRIERS/2
        ) * config.SUBCARRIER_SPACING
    ).astype(np.float64)

    # ----------------------------------------------------------
    # Save NumPy dataset
    # ----------------------------------------------------------

    np.save(os.path.join(config.DATASET_DIR, config.CSI_BOB_FILE), H_Bob)

    np.save(os.path.join(config.DATASET_DIR, config.CSI_EVE_FILE), H_Eve)

    np.save(os.path.join(config.DATASET_DIR, config.CIR_GAIN_FILE), a)

    np.save(os.path.join(config.DATASET_DIR, config.CIR_DELAY_FILE), tau)

    # ----------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------

    metadata = {

        # System
        "CarrierFrequencyGHz": config.CARRIER_FREQ/1e9,
        "BandwidthGHz": config.BANDWIDTH/1e9,
        "NumSubcarriers": config.NUM_SUBCARRIERS,
        "SubcarrierSpacingMHz": config.SUBCARRIER_SPACING/1e6,

        # Antenna
        "BSRows": config.TX_NUM_ROWS,
        "BSCols": config.TX_NUM_COLS,
        "UEColumns": config.RX_NUM_COLS,
        "UERows": config.RX_NUM_ROWS,

        "NumTxAntennas": config.TX_NUM_ANT,
        "NumRxAntennas": config.RX_NUM_ANT,

        "AntennaSpacingLambda": config.ANT_SPACING,

        # Geometry
        "BS_Position": np.array(config.BS_POS),
        "Bob_Position": np.array(config.BOB_POS),
        "Eve_Position": np.array(config.EVE_POS),

        # Ray tracing
        "MaxDepth": config.RT_MAX_DEPTH,
        "LOS": config.RT_LOS,
        "Reflection": config.RT_REFLECTION,
        "Diffraction": config.RT_DIFFRACTION,
        "Scattering": config.RT_SCATTERING,

        # Randomness
        "RandomSeed": config.RANDOM_SEED,

        # Software
        "TensorFlow": tf.__version__,
        "SionnaRT": rt.__version__,
        "CFRFrequencyConvention": "baseband offsets; carrier phase included in CIR",
        "Python": platform.python_version(),
        "Platform": platform.platform(),
    }

    np.savez(
        os.path.join(config.DATASET_DIR, "Metadata.npz"),
        **metadata,
    )

    # ----------------------------------------------------------
    # MATLAB STRUCT
    # ----------------------------------------------------------

    matlab_metadata = {}

    for k,v in metadata.items():

        if isinstance(v, str):
            matlab_metadata[k] = np.array([v], dtype=object)

        else:
            matlab_metadata[k] = np.asarray(v)

    # Validation metrics
    matlab_validation = {}

    if validation_metrics is not None:

        for k,v in validation_metrics.items():
            matlab_validation[re.sub(r"[^A-Za-z0-9_]", "_", k)] = np.asarray(v)

    # ----------------------------------------------------------
    # MAT FILE
    # ----------------------------------------------------------

    mat_dict = {

        "H_Bob": H_Bob,
        "H_Eve": H_Eve,

        "CIR_Gain": a,
        "CIR_Delay": tau,

        "FrequencyGrid": frequency_grid,

        "Power_Bob_dB": power_db(H_Bob),
        "Power_Eve_dB": power_db(H_Eve),

        "Metadata": matlab_metadata,
        "ValidationMetrics": matlab_validation,
    }

    sio.savemat(
        os.path.join(config.DATASET_DIR,
                     "CSI_Dataset_140GHz.mat"),
        mat_dict,
        do_compression=True,
        long_field_names=True,
    )

    # ----------------------------------------------------------
    # Dataset Summary
    # ----------------------------------------------------------

    if verbose:

        print("NumPy Dataset Saved")

        print(f"  H_Bob.npy     : {H_Bob.shape}")
        print(f"  H_Eve.npy     : {H_Eve.shape}")
        print(f"  CIR_Gain.npy  : {a.shape}")
        print(f"  CIR_Delay.npy : {tau.shape}")

        print()

        print("MATLAB Dataset Saved")

        print("  CSI_Dataset_140GHz.mat")

        print()

        print("Metadata Saved")

        print("  Metadata.npz")

        print("="*65)
        print("DATASET EXPORT COMPLETED SUCCESSFULLY")
        print("="*65)
