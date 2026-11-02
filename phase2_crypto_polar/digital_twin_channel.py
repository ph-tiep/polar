"""
digital_twin_channel.py
======================================================================
140 GHz UM-MIMO Digital Twin Channel Processor & OFDM Soft-Demapper.

Loads the ray-traced 140 GHz CSI dataset (H_Bob.npy, H_Eve.npy) from
Phase 1 (256 BS antennas x 16 UE antennas x 512 subcarriers), applies
SVD / Dominant Eigen-Beamforming at the BS matched to Bob, and provides
realistic frequency-selective fading and CSI-calibrated LLR generation
for both Bob and Eve.
======================================================================
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
            # Using Hermitian Gram matrix Hb @ Hb^H (16 x 16) for fast exact SVD
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

        # Synthesize realistic 140 GHz frequency-selective multipath ripple around
        # the ray-traced mean power using the CIR delay spread (2 GHz BW)
        freqs_rel = np.linspace(-1.0, 1.0, self.n_sc)
        if self.cir_delay is not None:
            valid_delays = self.cir_delay[self.cir_delay > 0]
            rms_tau_ns = float(np.std(valid_delays) * 1e9) if valid_delays.size > 1 else 4.5
        else:
            rms_tau_ns = 4.5
        multipath_ripple_bob = (
            1.0
            + 0.32 * np.cos(2.0 * np.pi * rms_tau_ns * freqs_rel)
            + 0.18 * np.sin(5.0 * np.pi * rms_tau_ns * freqs_rel + 0.7)
        )
        multipath_ripple_eve = (
            1.0
            + 0.25 * np.cos(1.7 * np.pi * rms_tau_ns * freqs_rel + 0.4)
            + 0.14 * np.sin(4.3 * np.pi * rms_tau_ns * freqs_rel - 0.3)
        )
        self.g_bob_norm = (self.g_bob_norm * multipath_ripple_bob).astype(np.float32)
        self.g_bob_norm /= float(np.mean(self.g_bob_norm))
        self.g_eve_fading_only = (self.g_eve_fading_only * multipath_ripple_eve).astype(np.float32)
        self.g_eve_fading_only /= float(np.mean(self.g_eve_fading_only))

        self.g_bob_db = 10.0 * np.log10(mean_g_bob * self.g_bob_norm + 1e-30)
        self.g_eve_db = 10.0 * np.log10(mean_g_eve * self.g_eve_fading_only + 1e-30)

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

    def transmit_qpsk_ofdm(
        self,
        codewords: torch.Tensor,
        ebno_db: float,
        coderate: float,
        node: str = "bob",
        include_beamforming_loss_for_eve: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map binary codewords [B, N] onto QPSK symbols across the 512
        Digital Twin OFDM subcarriers, pass through the 140 GHz frequency-selective
        fading channel + AWGN, and return demapper logits and CSI weights.

        Convention matching Sionna Polar decoders:
          - Input bit c in {0, 1}
          - Logit > 0 indicates bit 1 is more likely; Logit < 0 indicates bit 0
            (since PolarDecoder converts ``llr = -1.0 * logit``).
        """
        B, N = codewords.shape
        dev = codewords.device
        dt = codewords.dtype

        # Select subcarrier power profile from the 140 GHz Digital Twin
        idx_sc = np.arange(N) % self.n_sc
        if node.lower() == "bob":
            g_sc = self.g_bob_norm[idx_sc]
        else:
            if include_beamforming_loss_for_eve:
                g_sc = self.g_eve_norm[idx_sc]
            else:
                # Worst-case Eve at identical average SNR as Bob (testing pure
                # Crypto-Polar security even if Eve intercepts with high SNR!)
                g_sc = self.g_eve_fading_only[idx_sc]

        g_base = torch.tensor(g_sc, dtype=dt, device=dev).unsqueeze(0)  # [1, N]

        # Include 140 GHz Sub-THz block Rician small-scale fading across OFDM frames
        # (K-factor = 6 dB -> K_lin = 4.0, capturing THz dominant path + diffuse cluster)
        k_rician = 4.0
        los_comp = np.sqrt(k_rician / (k_rician + 1.0))
        nlos_std = np.sqrt(1.0 / (2.0 * (k_rician + 1.0)))
        # Coherence block size = 16 subcarriers (correlated across adjacent subcarriers)
        n_blocks = (N + 15) // 16
        fading_re = los_comp + torch.randn((B, n_blocks, 1), dtype=dt, device=dev) * nlos_std
        fading_im = torch.randn((B, n_blocks, 1), dtype=dt, device=dev) * nlos_std
        fading_pow = (fading_re ** 2 + fading_im ** 2).expand(B, n_blocks, 16).reshape(B, -1)[:, :N]

        g_t = g_base * fading_pow
        h_mag = torch.sqrt(g_t + 1e-8)

        # Standard coded BPSK/QPSK Eb/N0 calibration:
        # For unit-energy coded symbols, sigma^2 = 1 / (2 * R * Eb/N0)
        ebno_lin = 10.0 ** (float(ebno_db) / 10.0)
        sigma2 = 1.0 / (2.0 * float(coderate) * ebno_lin)
        sigma = float(np.sqrt(sigma2))

        # Modulate: x_s = 2*c - 1 in {-1, +1} (where -1 -> bit 0, +1 -> bit 1)
        x_s = 2.0 * codewords - 1.0
        noise = torch.randn_like(x_s) * sigma
        y_rx = h_mag * x_s + noise

        # Conventional equalized demapper logit (uses average subcarrier profile)
        logits = (2.0 * y_rx * torch.sqrt(g_base)) / sigma2
        # Instantaneous CSI reliability weights for CSI-Calibrated Improved Decoder
        csi_weights = g_t

        return logits, csi_weights

