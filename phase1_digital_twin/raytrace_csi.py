"""
raytrace_csi.py
===============================================================
IEEE Phase 1 - Digital Twin CSI Extraction

Compatible with:
    TensorFlow 2.21
    Sionna RT 2.0.1
===============================================================
"""

import time
import numpy as np
import tensorflow as tf

from sionna.rt import PathSolver
from phase1_digital_twin import config

# ===============================================================
# HELPER FUNCTIONS
# ===============================================================

def build_frequency_grid():
    # Paths.cir already includes the carrier phase; CFR takes baseband offsets.
    return (np.arange(config.NUM_SUBCARRIERS, dtype=np.float64)
            - config.NUM_SUBCARRIERS / 2) * config.SUBCARRIER_SPACING


def cir_statistics(a, tau):
    a_np = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
    tau_np = tau.numpy() if hasattr(tau, "numpy") else np.asarray(tau)
    
    power = np.abs(a_np) ** 2
    valid_paths = int(np.count_nonzero((tau_np >= 0) & (np.mean(power, axis=(2, 3)) > 0)))
    
    # Loại bỏ các tia bị đệm giá trị -1.0 của Sionna RT
    valid_tau = tau_np[tau_np >= 0]
    
    if valid_tau.size == 0:
        max_delay, mean_delay = 0.0, 0.0
    else:
        max_delay = float(np.max(valid_tau) * 1e9)
        mean_delay = float(np.mean(valid_tau) * 1e9)
        
    avg_power = float(np.mean(power)) if power.size else 0.0
    
    return {
        "valid_paths": valid_paths,
        "max_delay_ns": max_delay,
        "mean_delay_ns": mean_delay,
        "average_power": avg_power,
    }

# ===============================================================
# MAIN FUNCTION
# ===============================================================

def extract_csi(scene, run_mode="debug", verbose=True):
    num_samples = config.RT_NUM_SAMPLES_FULL if run_mode.lower() == "full" else config.RT_NUM_SAMPLES_DEBUG
    if verbose:
        print("=" * 60)
        print("STEP 3 : RAY TRACING ENGINE")
        print("=" * 60)

    solver = PathSolver()
    start_time = time.time()
    
    paths = solver(
        scene=scene, max_depth=config.RT_MAX_DEPTH, samples_per_src=num_samples,
        los=config.RT_LOS, specular_reflection=config.RT_REFLECTION,
        diffraction=config.RT_DIFFRACTION, diffuse_reflection=config.RT_SCATTERING,
        synthetic_array=config.SYNTHETIC_ARRAY, seed=config.RANDOM_SEED
    )
    
    runtime = time.time() - start_time
    if verbose: 
        print(f"[1/3] Ray tracing completed in {runtime:.2f} s")

    # 1. TRÍCH XUẤT CIR
    a, tau = paths.cir(normalize_delays=False, out_type="tf")
    
    a_np = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
    tau_np = tau.numpy() if hasattr(tau, "numpy") else np.asarray(tau)

    rx_ant = config.RX_NUM_ANT
    tx_ant = config.TX_NUM_ANT
    num_rx = len(scene.receivers)
    num_tx = len(scene.transmitters)

    num_paths = tau_np.shape[-1]
    if num_paths == 0:
        raise ValueError("Ray tracing found no valid paths. Check node placement "
                         "against scene geometry and reflection settings; no CSI was exported.")
    expected_cir = (num_rx, rx_ant, num_tx, tx_ant, num_paths, 1)
    if a_np.shape != expected_cir or tau_np.shape != (num_rx, num_tx, num_paths):
        raise ValueError(f"Unexpected synthetic-array CIR shapes: {a_np.shape}, {tau_np.shape}")
    # Sionna: [rx, rx_ant, tx, tx_ant, path, time]. Preserve antenna order.
    a_np = a_np[..., 0].transpose(0, 2, 1, 3, 4)
    valid = (tau_np >= 0) & (np.mean(np.abs(a_np)**2, axis=(2, 3)) > 0)
    for idx, name in enumerate(scene.receivers):
        if not valid[idx].any():
            raise ValueError(f"No valid paths to {name}; no CSI was exported. Check node placement.")
    a = tf.convert_to_tensor(a_np, dtype=tf.complex64)
    tau = tf.convert_to_tensor(tau_np, dtype=tf.float32)

    stats = cir_statistics(a, tau)
    if verbose:
        print("[2/3] CIR extracted.")
        print(f"Valid Multipaths     : {stats['valid_paths']}")
        print(f"Maximum Delay        : {stats['max_delay_ns']:.2f} ns")
        print(f"Mean Delay           : {stats['mean_delay_ns']:.2f} ns")
    # 4. TRÍCH XUẤT CFR
    frequencies = build_frequency_grid()
    H = paths.cfr(frequencies=frequencies, normalize=False,
                  normalize_delays=False, out_type="numpy")
    expected_cfr = (num_rx, rx_ant, num_tx, tx_ant, 1, config.NUM_SUBCARRIERS)
    if H.shape != expected_cfr or num_tx != 1:
        raise ValueError(f"Unexpected CFR shape {H.shape}; expected {expected_cfr} with one BS")
    receiver_names = list(scene.receivers)
    H_Bob = H[receiver_names.index("Bob"), :, 0, :, 0, :]
    H_Eve = H[receiver_names.index("Eve"), :, 0, :, 0, :]
    for name, channel in [("Bob", H_Bob), ("Eve", H_Eve)]:
        if not np.isfinite(channel).all() or not np.any(np.abs(channel) > 0):
            raise ValueError(f"Invalid or zero CSI for {name}; no dataset exported")

    metrics = {"runtime_sec": runtime, **stats}
    return H_Bob, H_Eve, a, tau, paths, metrics