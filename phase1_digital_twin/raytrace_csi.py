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
    k = tf.range(config.NUM_SUBCARRIERS, dtype=tf.float32)
    k -= config.NUM_SUBCARRIERS / 2.0
    frequencies = config.CARRIER_FREQ + k * config.SUBCARRIER_SPACING
    return tf.cast(frequencies, tf.float32)

def cir_statistics(a, tau):
    a_np = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
    tau_np = tau.numpy() if hasattr(tau, "numpy") else np.asarray(tau)
    
    power = np.abs(a_np) ** 2
    valid_paths = int(np.count_nonzero(power))
    
    # Loại bỏ các tia bị đệm giá trị -1.0 của Sionna RT
    valid_tau = tau_np[tau_np >= 0]
    
    if valid_tau.size == 0:
        max_delay, mean_delay = 0.0, 0.0
    else:
        max_delay = float(np.max(valid_tau) * 1e9)
        mean_delay = float(np.mean(valid_tau) * 1e9)
        
    avg_power = float(np.mean(power))
    
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

    # Trích xuất số tia đa đường thực tế từ chiều cuối cùng của ma trận trễ
    num_paths = tau_np.shape[-1]

    # 2. CHUẨN HÓA KÍCH THƯỚC CIR GAIN
    # Phục hồi không gian mảng ăng-ten cho tín hiệu phức
    a_np = np.reshape(a_np, (num_rx, num_tx, rx_ant, tx_ant, num_paths))

    # 3. CHUẨN HÓA KÍCH THƯỚC DELAY
    # Trễ mảng không phụ thuộc vào từng phần tử anten (chỉ phụ thuộc TX/RX)
    tau_np = np.reshape(tau_np, (num_rx, num_tx, num_paths))

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
    H = paths.cfr(frequencies=frequencies, normalize=False, out_type="tf")
    H_np = H.numpy()
    
    if H_np.shape[0] == 1: 
        H_np = H_np[0]

    receiver_names = list(scene.receivers.keys())
    idx_bob = receiver_names.index("Bob")
    idx_eve = receiver_names.index("Eve")

    expected_shape = (rx_ant, tx_ant, config.NUM_SUBCARRIERS)
    
    def format_csi_shape(H_raw):
        H_sq = np.squeeze(H_raw)
        if H_sq.shape == (tx_ant, config.NUM_SUBCARRIERS):
            H_sq = np.tile(H_sq[np.newaxis, :, :], (rx_ant, 1, 1))
        if H_sq.size == np.prod(expected_shape):
            return np.reshape(H_sq, expected_shape)
        return H_sq

    H_Bob = format_csi_shape(H_np[idx_bob])
    H_Eve = format_csi_shape(H_np[idx_eve])

    metrics = {"runtime_sec": runtime, **stats}
    return H_Bob, H_Eve, a, tau, paths, metrics