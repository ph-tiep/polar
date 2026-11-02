"""Ray-traced CSI beamforming and coherent frequency-domain BPSK logits.

Loads [Rx, Tx, subcarrier] CSI at the dimensions supplied by Phase 1. Applies
Bob SVD beamforming and Eve MRC on Bob's transmitted beam. Does not synthesize
extra multipath or implement time-domain OFDM/CP.
"""

import os
from typing import Dict, Tuple
import numpy as np
import torch


class DigitalTwinTHzChannel:
    """140 GHz UM-MIMO Digital Twin Channel Model backed by Sionna RT CSI."""

    def __init__(self, dataset_dir: str, device: str = "cpu"):
        self.dataset_dir = dataset_dir
        self.device = device

        bob_path = os.path.join(dataset_dir, "H_Bob.npy")
        eve_path = os.path.join(dataset_dir, "H_Eve.npy")

        if not os.path.exists(bob_path) or not os.path.exists(eve_path):
            raise FileNotFoundError(
                f"Digital Twin CSI files not found in {dataset_dir}. Run main_phase1.py first."
            )

        # Shape: (N_rx=16, N_tx=256, N_sc=512)
        self.H_Bob = np.load(bob_path)
        self.H_Eve = np.load(eve_path)

        cir_gain_path = os.path.join(dataset_dir, "CIR_Gain.npy")
        cir_delay_path = os.path.join(dataset_dir, "CIR_Delay.npy")
        self.cir_gain = np.load(cir_gain_path) if os.path.exists(cir_gain_path) else None
        self.cir_delay = np.load(cir_delay_path) if os.path.exists(cir_delay_path) else None

        for name, arr in (("Bob", self.H_Bob), ("Eve", self.H_Eve)):
            if arr.ndim != 3 or min(arr.shape) < 1 or not np.isfinite(arr).all():
                raise ValueError(f"Invalid {name} CSI: expected finite [Rx, Tx, subcarrier]")
            if not np.iscomplexobj(arr) or not np.all(np.sum(abs(arr)**2, axis=(0, 1)) > 0):
                raise ValueError(f"{name} CSI must be complex and nonzero on every carrier")
        if self.H_Bob.shape != self.H_Eve.shape:
            raise ValueError("Bob/Eve CSI dimensions must match for this experiment")
        self.n_rx, self.n_tx, self.n_sc = self.H_Bob.shape
        self._compute_beamforming_and_effective_channels()

    def _compute_beamforming_and_effective_channels(self) -> None:
        """Compute per-subcarrier SVD eigen-beamforming matched to Bob and
        evaluate the resulting effective channels at Bob and Eve.
        """
        g_bob = np.zeros(self.n_sc, dtype=np.float64)
        g_eve = np.zeros(self.n_sc, dtype=np.float64)
        spatial_corr = np.zeros(self.n_sc, dtype=np.float64)
        raw_power_bob = np.zeros(self.n_sc, dtype=np.float64)
        raw_power_eve = np.zeros(self.n_sc, dtype=np.float64)

        for m in range(self.n_sc):
            Hb = self.H_Bob[:, :, m]  # (16, 256)
            He = self.H_Eve[:, :, m]  # (16, 256)

            raw_power_bob[m] = np.mean(np.abs(Hb) ** 2)
            raw_power_eve[m] = np.mean(np.abs(He) ** 2)

            # Dominant right singular vector of H_Bob (BS transmit beamformer)
            # Economy SVD on the actual rectangular CSI matrix
            u_b, s_b, vh_b = np.linalg.svd(Hb, full_matrices=False)
            w_tx = vh_b[0, :].conj()  # (256,) unit-norm transmit beamformer
            w_rx_bob = u_b[:, 0]      # (16,) unit-norm receive combiner at Bob

            # Effective scalar channel at Bob after eigen-beamforming
            h_eff_bob = np.vdot(w_rx_bob, Hb @ w_tx)
            g_bob[m] = np.abs(h_eff_bob) ** 2

            # Eve's effective vector channel when BS beams toward Bob: h_e = He @ w_tx
            h_vec_eve = He @ w_tx  # (16,)
            # Eve applies her own optimal MRC combiner: w_rx_eve = h_vec_eve / ||h_vec_eve||
            g_eve[m] = np.linalg.norm(h_vec_eve) ** 2

            # Subspace correlation between Bob's and Eve's dominant TX directions
            _, _, vh_e = np.linalg.svd(He, full_matrices=False)
            w_tx_eve_dom = vh_e[0, :].conj()
            spatial_corr[m] = np.abs(np.vdot(w_tx, w_tx_eve_dom))

        # Normalize Bob's average effective beamforming gain to unit mean
        # so Eb/N0 sweeps are directly calibrated against Bob's received SNR,
        # while preserving the exact relative frequency selectivity and Bob-vs-Eve
        # power ratio from the 140 GHz ray-traced Digital Twin!
        mean_g_bob = np.mean(g_bob) + 1e-30
        self.g_bob_norm = (g_bob / mean_g_bob).astype(np.float32)
        self.g_eve_norm = (g_eve / mean_g_bob).astype(np.float32)

        # Also compute Eve's normalized channel when Eve has comparable SNR
        # (worst-case colluding/proximate Eve directly inside main lobe)
        mean_g_eve = np.mean(g_eve) + 1e-30
        self.g_eve_fading_only = (g_eve / mean_g_eve).astype(np.float32)

        self.g_bob_db = 10.0 * np.log10(g_bob + 1e-30)
        self.g_eve_db = 10.0 * np.log10(g_eve + 1e-30)
        self.raw_power_bob_db = 10.0 * np.log10(raw_power_bob + 1e-30)
        self.raw_power_eve_db = 10.0 * np.log10(raw_power_eve + 1e-30)
        self.spatial_corr = spatial_corr.astype(np.float32)

        self.bf_gain_advantage_db = float(
            10.0 * np.log10(mean_g_bob / mean_g_eve)
        )

    def get_channel_summary(self) -> Dict[str, float]:
        """Return summary statistics of the 140 GHz UM-MIMO Digital Twin channel."""
        return {
            "num_tx_antennas": int(self.n_tx),
            "num_rx_antennas": int(self.n_rx),
            "num_subcarriers": int(self.n_sc),
            "raw_bob_power_db": float(np.mean(self.raw_power_bob_db)),
            "raw_eve_power_db": float(np.mean(self.raw_power_eve_db)),
            "bf_bob_gain_db": float(np.mean(self.g_bob_db)),
            "bf_eve_gain_db": float(np.mean(self.g_eve_db)),
            "beamforming_advantage_db": float(self.bf_gain_advantage_db),
            "mean_spatial_correlation": float(np.mean(self.spatial_corr)),
            "max_spatial_correlation": float(np.max(self.spatial_corr)),
            "bob_gain_std_db": float(np.std(self.g_bob_db)),
        }

    def transmit_bpsk(self, codewords, ebno_db, payload_rate, *, node="bob",
                      eve_snr="physical", indices=None, noise=None):
        """Frequency-domain BPSK abstraction with exact coherent channel logits.

        Eb refers to payload bits (CRC excluded). No time-domain OFDM/CP, extra
        Rician fading, or synthesized ripple. Pass shared indices/noise to pair
        schemes. Positive logit means bit 1. Noise is unit normal before scaling.
        """
        if node not in ("bob", "eve") or eve_snr not in ("physical", "equal-average"):
            raise ValueError("Invalid receiver or Eve normalization")
        if not 0 < payload_rate <= 1:
            raise ValueError("Invalid payload rate")
        if indices is None or noise is None:
            raise ValueError("Pass explicit carrier indices and unit-normal noise")
        profile = self.g_bob_norm if node == "bob" else (
            self.g_eve_norm if eve_snr == "physical" else self.g_eve_fading_only)
        gain = torch.as_tensor(profile, dtype=codewords.dtype, device=codewords.device)[indices]
        h = torch.sqrt(gain)
        sigma2 = 1 / (2 * payload_rate * 10 ** (float(ebno_db) / 10))
        y = h * (2 * codewords - 1) + noise * np.sqrt(sigma2)
        return 2 * h * y / sigma2, gain
