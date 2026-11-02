#
# SPDX-FileCopyrightText: 2026 6G Digital Twin Crypto-Coding Project
# SPDX-License-Identifier: Apache-2.0
#
"""
crypto_polar.py
======================================================================
Joint Physical-Layer and Cryptographic Polar Coding (Crypto-Polar)
via Dynamic Frozen-Bit Coset Perturbation & Improved CSI-Calibrated
CA-SCL-Flip Decoding for 6G 140 GHz UM-MIMO Digital Twin Systems.
======================================================================
"""

from typing import Dict, Optional, Tuple, Union
import hashlib
import numpy as np
import torch

from .._compat import Block
from ..crc import CRCDecoder, CRCEncoder
from .encoding import PolarEncoder
from .decoding import (
    PolarSCDecoder,
    PolarSCLDecoder,
    PolarBPDecoder,
    _coset_sign_from_frozen_bits,
)
from .utils import generate_5g_ranking


__all__ = [
    "ChannelBoundKeyGenerator",
    "CryptoPolarEncoder",
    "CryptoPolarDecoder",
    "ImprovedCryptoPolarSCLDecoder",
]


class ChannelBoundKeyGenerator:
    """Cryptographic Key Stream Generator bound to 140 GHz Digital Twin CSI
    and a Master Secret Key using NIST FIPS 202 SHAKE-256 Sponge CSPRNG.

    Extracts physical-layer Reciprocal CSI features (dominant singular value
    profile from H_Bob) mixed with a shared secret key and frame nonce to
    produce statistically independent binary key streams for:
      - Key A (k_A): Information bit masking [batch_size, k]
      - Key B (k_F): Dynamic frozen-bit coset assignment [batch_size, n - k]
    """

    def __init__(
        self,
        master_key: Union[bytes, str] = b"6G_THZ_DIGITAL_TWIN_MASTER_KEY_2026",
        csi_tensor: Optional[np.ndarray] = None,
    ):
        if isinstance(master_key, str):
            master_key = master_key.encode("utf-8")
        self.master_key = master_key
        self.csi_digest = self._extract_csi_entropy(csi_tensor)

    @staticmethod
    def _extract_csi_entropy(csi_tensor: Optional[np.ndarray]) -> bytes:
        """Hash quantized singular value profile of the 140 GHz MIMO channel."""
        if csi_tensor is None:
            return hashlib.sha3_256(b"DEFAULT_SYNTHETIC_CSI_SEED").digest()
        arr = np.asarray(csi_tensor)
        # Sample subcarrier magnitudes deterministically
        mag_profile = np.abs(arr).ravel()[:1024].astype(np.float32)
        quantized = np.round(mag_profile / (np.mean(mag_profile) + 1e-12) * 255.0).astype(
            np.uint8
        )
        return hashlib.sha3_256(quantized.tobytes()).digest()

    def generate_key_stream(
        self,
        batch_size: int,
        n: int,
        frame_idx: int = 0,
        device: Optional[ Union[str, torch.device] ] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate binary cryptographic key stream of shape [batch_size, n]."""
        total_bits = batch_size * n
        total_bytes = (total_bits + 7) // 8

        shake = hashlib.shake_256()
        shake.update(self.master_key)
        shake.update(self.csi_digest)
        shake.update(int(frame_idx).to_bytes(8, byteorder="big", signed=False))
        raw_bytes = shake.digest(total_bytes)

        byte_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        bits_np = np.unpackbits(byte_arr)[:total_bits].reshape(batch_size, n)
        return torch.tensor(bits_np, dtype=dtype, device=device)

    @staticmethod
    def split_key_stream(
        secret_key_stream: torch.Tensor,
        k: int,
        n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split an N-bit secret key stream into:
          - key_info (Key A): [..., k] for information bit encryption
          - key_frozen (Key B): [..., n - k] for dynamic frozen bit injection
        """
        if secret_key_stream.shape[-1] != n:
            raise ValueError(
                f"Expected secret_key_stream of length n={n}, got {secret_key_stream.shape[-1]}."
            )
        key_info = secret_key_stream[..., :k]
        key_frozen = secret_key_stream[..., k:]
        return key_info, key_frozen


class CryptoPolarEncoder(PolarEncoder):
    """Joint Physical-Layer & Cryptographic Polar Encoder via Dynamic Frozen Bits.

    Given payload bits ``m`` of length ``k_payload`` (where ``k_payload = k`` if no
    CRC is used, or ``k_payload = k - k_crc`` when CRC-aided mode is active) and
    a ``secret_key_stream`` of length ``n``:
      1. Splits ``secret_key_stream`` into:
           - Key A (``key_info``, length ``k``): masks the information (or CRC-encoded)
             vector ``u_A = b_crc ^ key_info``.
           - Key B (``key_frozen``, length ``n - k``): dynamically populates the
             frozen positions ``u_F = key_frozen`` instead of static all-zeros.
      2. Performs the Polar butterfly transformation over GF(2):
           ``x = u * G_N = (u_A * G_A) ^ (key_frozen * G_F)``.

    This transforms the Polar code from a single fixed linear subspace into
    ``2^(n-k)`` cryptographically hopping affine cosets while preserving the
    minimum Hamming distance and polarization profile for the legitimate receiver.
    """

    def __init__(
        self,
        frozen_pos: np.ndarray,
        n: int,
        crc_degree: Optional[str] = None,
        enable_info_masking: bool = True,
        enable_frozen_crypto: bool = True,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        check_input: bool = True,
        **kwargs,
    ):
        super().__init__(
            frozen_pos=frozen_pos,
            n=n,
            precision=precision,
            device=device,
            check_input=check_input,
            **kwargs,
        )
        self._enable_info_masking = enable_info_masking
        self._enable_frozen_crypto = enable_frozen_crypto

        if crc_degree is not None:
            self._use_crc = True
            self._crc_degree = crc_degree
            self._k_crc_len = self._get_crc_length(crc_degree)
            self._k_payload = self._k - self._k_crc_len
            if self._k_payload <= 0:
                raise ValueError("k is too small for the specified crc_degree.")
            self._crc_encoder = CRCEncoder(
                crc_degree, k=self._k_payload, precision=precision, device=device
            )
        else:
            self._use_crc = False
            self._crc_degree = None
            self._k_crc_len = 0
            self._k_payload = self._k
            self._crc_encoder = None

    @staticmethod
    def _get_crc_length(crc_degree: str) -> int:
        deg_map = {
            "CRC24A": 24,
            "CRC24B": 24,
            "CRC24C": 24,
            "CRC16": 16,
            "CRC11": 11,
            "CRC6": 6,
        }
        if crc_degree not in deg_map:
            raise ValueError(f"Unsupported crc_degree: {crc_degree}")
        return deg_map[crc_degree]

    @property
    def k_payload(self) -> int:
        """Number of user payload bits before optional CRC."""
        return self._k_payload

    @property
    def crc_degree(self) -> Optional[str]:
        """CRC polynomial name if CRC is enabled."""
        return self._crc_degree

    def build(self, *input_shapes) -> None:
        """Accepts (bits_shape,) or (bits_shape, key_shape)."""
        bits_shape = input_shapes[0]
        if bits_shape[-1] != self._k_payload:
            raise ValueError(
                f"Expected last dimension of bits to be {self._k_payload}, got {bits_shape[-1]}."
            )

    def call(
        self,
        bits: torch.Tensor,
        secret_key_stream: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode payload bits with cryptographic frozen-bit and info-bit keys.

        :param bits: Binary tensor of shape ``[..., k_payload]``.
        :param secret_key_stream: Binary tensor of shape ``[..., n]`` (or
            ``[..., n - k]`` if only frozen key is supplied). If ``None``,
            falls back to standard all-zero frozen Polar encoding.
        :output cw: Binary codeword tensor of shape ``[..., n]``.
        """
        input_shape = bits.shape
        batch_size = int(np.prod(input_shape[:-1])) if len(input_shape) > 1 else 1
        u_payload = bits.reshape(batch_size, self._k_payload)

        # 1. Optional outer CRC encoding (before Key-A encryption so CRC also
        #    verifies key synchronization at the legitimate receiver!)
        if self._use_crc:
            u_info = self._crc_encoder(u_payload)
        else:
            u_info = u_payload

        if secret_key_stream is None:
            cw = super().call(u_info, frozen_bits=None)
            return cw.reshape(list(input_shape[:-1]) + [self._n])

        n_frz = self._n - self._k
        key_last_dim = secret_key_stream.shape[-1]

        if key_last_dim == self._n:
            key_reshaped = secret_key_stream.reshape(-1, self._n)
            if key_reshaped.shape[0] == 1 and batch_size > 1:
                key_reshaped = key_reshaped.expand(batch_size, -1)
            key_a = key_reshaped[:, : self._k]
            key_b = key_reshaped[:, self._k :]
        elif key_last_dim == n_frz:
            key_b = secret_key_stream.reshape(-1, n_frz)
            if key_b.shape[0] == 1 and batch_size > 1:
                key_b = key_b.expand(batch_size, -1)
            key_a = None
        else:
            raise ValueError(
                f"secret_key_stream last dimension must be n={self._n} or n-k={n_frz}, "
                f"got {key_last_dim}."
            )

        # 2. Apply Key A (XOR masking on information bits)
        if self._enable_info_masking and key_a is not None:
            u_info_enc = (
                torch.bitwise_xor(u_info.to(torch.uint8), key_a.to(torch.uint8))
            ).to(u_info.dtype)
        else:
            u_info_enc = u_info

        # 3. Apply Key B (Dynamic Cryptographic Frozen Bits)
        frozen_bits = key_b if self._enable_frozen_crypto else None

        # 4. Polar Butterfly Transform with Dynamic Frozen Bits
        cw = super().call(u_info_enc, frozen_bits=frozen_bits)
        return cw.reshape(list(input_shape[:-1]) + [self._n])


class ImprovedCryptoPolarSCLDecoder(Block):
    """Improved CSI-Calibrated & Critical-Set Bit-Flip Retry CA-SCL Decoder
    for Dynamic Frozen-Bit Crypto-Polar Codes in 140 GHz UM-MIMO Channels.

    Key Algorithmic Enhancements over Standard Polar Decoders:
      1. **Fast-SCL Coset Isomorphism**: Strips the dynamic frozen coset
         ``c_coset = key_frozen * G_F`` via LLR sign modulation in O(N log N)
         prior to tree traversal, preserving all Rate-0 (R0), Repetition (REP),
         and Rate-1 (R1) Fast-SCL vectorized instructions.
      2. **Subcarrier CSI-Reliability Calibration & Adaptive LLR Damping**:
         Calibrates channel LLRs using post-equalization MIMO-OFDM subcarrier
         SNRs and applies non-linear LLR damping ``alpha_scale`` to mitigate
         check-node boxplus overestimation in 140 GHz multipath fading.
      3. **Cryptographic CRC-Aided Verification & Critical-Set Flip Retry (CS-SCLF)**:
         Because Key-A masks the CRC-encoded vector ``u_info``, the CRC check
         inside the SCL candidate list must be evaluated *after* candidate-wise
         Key-A unmasking! Furthermore, if all ``L`` surviving paths fail the
         cryptographic CRC check, the decoder automatically locates the ``max_flips``
         least reliable information bit positions and performs targeted bit-flip
         list retries (SCL-Flip), yielding significant coding gain.
    """

    def __init__(
        self,
        frozen_pos: np.ndarray,
        n: int,
        list_size: int = 8,
        crc_degree: Optional[str] = "CRC11",
        enable_info_masking: bool = True,
        enable_frozen_crypto: bool = True,
        llr_damping: float = 0.88,
        max_flip_trials: int = 4,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)
        self._n = int(n)
        self._frozen_pos = np.asarray(frozen_pos, dtype=int)
        self._k = self._n - len(self._frozen_pos)
        self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
        self._list_size = int(list_size)
        self._enable_info_masking = enable_info_masking
        self._enable_frozen_crypto = enable_frozen_crypto
        self._llr_damping = float(llr_damping)
        self._max_flip_trials = int(max_flip_trials)

        # Underlying Fast-SCL engine (without internal unmasked CRC so we can
        # perform cryptographic Key-A unmasking BEFORE CRC verification!)
        self._scl_core = PolarSCLDecoder(
            frozen_pos=self._frozen_pos,
            n=self._n,
            list_size=self._list_size,
            crc_degree=None,
            precision=precision,
            device=device,
        )
        # Secondary high-diversity SCL core (2L) for Critical-Set retry on CRC failure
        self._scl_retry_core = (
            PolarSCLDecoder(
                frozen_pos=self._frozen_pos,
                n=self._n,
                list_size=min(self._list_size * 2, 32),
                crc_degree=None,
                precision=precision,
                device=device,
            )
            if self._max_flip_trials > 0
            else None
        )

        self.register_buffer(
            "_info_pos_t",
            torch.tensor(self._info_pos, dtype=torch.int64, device=self.device),
        )
        self.register_buffer(
            "_frozen_pos_t",
            torch.tensor(self._frozen_pos, dtype=torch.int64, device=self.device),
        )

        if crc_degree is not None:
            self._use_crc = True
            self._crc_encoder = CRCEncoder(
                crc_degree, precision=precision, device=device
            )
            self._crc_decoder = CRCDecoder(
                self._crc_encoder, precision=precision, device=device
            )
            self._k_crc = self._crc_encoder.crc_length
            self._k_payload = self._k - self._k_crc
        else:
            self._use_crc = False
            self._crc_encoder = None
            self._crc_decoder = None
            self._k_crc = 0
            self._k_payload = self._k

    @property
    def k_payload(self) -> int:
        return self._k_payload

    def build(self, *input_shapes) -> None:
        if input_shapes[0][-1] != self._n:
            raise ValueError("Invalid input LLR shape.")

    def _select_best_path_with_crypto_crc(
        self,
        bits_list: torch.Tensor,
        pm_list: torch.Tensor,
        key_a: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unmask all 2L candidate paths with Key-A and select the minimum-PM
        path that satisfies the CRC parity check.
        """
        B = bits_list.shape[0]
        u_hat_list = bits_list[:, :, self._info_pos_t].to(pm_list.dtype)  # [B, 2L, k]

        if self._enable_info_masking and key_a is not None:
            key_a_exp = key_a.unsqueeze(1).to(torch.uint8)  # [B, 1, k]
            u_unmasked_list = (
                torch.bitwise_xor(u_hat_list.to(torch.uint8), key_a_exp)
            ).to(pm_list.dtype)
        else:
            u_unmasked_list = u_hat_list

        if self._use_crc:
            u_payload_list, crc_valid = self._crc_decoder(u_unmasked_list)
            crc_valid_sq = crc_valid.squeeze(-1).bool()  # [B, 2L]
            pm_penalty = (1.0 - crc_valid_sq.float()) * self._scl_core.llr_max * self._k
            total_pm = pm_list + pm_penalty
            best_idx = torch.argmin(total_pm, dim=-1)
            batch_idx = torch.arange(B, device=bits_list.device)
            best_payload = u_payload_list[batch_idx, best_idx, :]
            best_crc_ok = crc_valid_sq[batch_idx, best_idx]
            return best_payload, best_crc_ok
        else:
            best_idx = torch.argmin(pm_list, dim=-1)
            batch_idx = torch.arange(B, device=bits_list.device)
            best_payload = u_unmasked_list[batch_idx, best_idx, :]
            best_crc_ok = torch.ones(B, dtype=torch.bool, device=bits_list.device)
            return best_payload, best_crc_ok

    def call(
        self,
        llr_ch: torch.Tensor,
        secret_key_stream: Optional[torch.Tensor] = None,
        csi_weights: Optional[torch.Tensor] = None,
        return_crc_status: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode channel logits/LLRs using Improved CSI-Calibrated Crypto-CA-SCL-Flip."""
        input_shape = llr_ch.shape
        B = int(np.prod(input_shape[:-1])) if len(input_shape) > 1 else 1
        logits = llr_ch.reshape(B, self._n)

        # 1. Subcarrier CSI-Reliability Calibration & Damping
        if csi_weights is not None:
            w = csi_weights.reshape(-1, self._n).to(logits.dtype)
            if w.shape[0] == 1 and B > 1:
                w = w.expand(B, -1)
            # Suppress impulse/outlier LLR overconfidence on deep-faded subcarriers
            snr_reliability = torch.sqrt(w / (torch.mean(w, dim=-1, keepdim=True) + 1e-8))
            logits = logits * torch.clamp(snr_reliability, 0.4, 1.6) * self._llr_damping
        else:
            logits = logits * self._llr_damping

        # 2. Extract Key A and Key B
        key_a, key_b = None, None
        n_frz = self._n - self._k
        if secret_key_stream is not None:
            k_dim = secret_key_stream.shape[-1]
            if k_dim == self._n:
                ks = secret_key_stream.reshape(-1, self._n)
                if ks.shape[0] == 1 and B > 1:
                    ks = ks.expand(B, -1)
                key_a = ks[:, : self._k]
                key_b = ks[:, self._k :]
            elif k_dim == n_frz:
                key_b = secret_key_stream.reshape(-1, n_frz)
                if key_b.shape[0] == 1 and B > 1:
                    key_b = key_b.expand(B, -1)
            else:
                raise ValueError(f"Invalid secret_key_stream dimension {k_dim}.")

        # 3. Apply Fast-SCL Coset Isomorphism for Dynamic Frozen Bits
        llr_true = torch.clamp(-logits, -self._scl_core.llr_max, self._scl_core.llr_max)
        if self._enable_frozen_crypto and key_b is not None:
            coset_sign = _coset_sign_from_frozen_bits(
                key_b, self._frozen_pos_t, self._n, B, llr_true.dtype
            )
            llr_coset = llr_true * coset_sign
        else:
            llr_coset = llr_true

        # 4. Primary Fast-SCL Pass
        bits_list, pm_list = self._scl_core._decode(llr_coset)
        best_payload, crc_ok = self._select_best_path_with_crypto_crc(
            bits_list, pm_list, key_a
        )

        # 5. Critical-Set Bit-Flip & Multi-Scale List Refinement (CS-SCLF) for failed CRC blocks
        if self._use_crc and self._max_flip_trials > 0 and not torch.all(crc_ok):
            failed_mask = ~crc_ok
            failed_indices = torch.nonzero(failed_mask, as_tuple=False).squeeze(-1)
            llr_failed = llr_coset[failed_indices]  # [B_fail, n]
            key_a_failed = key_a[failed_indices] if key_a is not None else None

            # Stage 5a: Expanded List (2L) with damped LLR pass on failed blocks
            if self._scl_retry_core is not None:
                bits_2l, pm_2l = self._scl_retry_core._decode(llr_failed * 0.78)
                payload_2l, crc_2l = self._select_best_path_with_crypto_crc(
                    bits_2l, pm_2l, key_a_failed
                )
                if torch.any(crc_2l):
                    rec_global = failed_indices[crc_2l]
                    best_payload[rec_global] = payload_2l[crc_2l]
                    crc_ok[rec_global] = True

            # Stage 5b: Critical-Set Channel Bit-Flip Retry on remaining failed blocks
            _, least_rel_pos = torch.topk(
                -llr_failed.abs(), k=min(self._max_flip_trials, self._n), dim=-1
            )

            for flip_idx in range(least_rel_pos.shape[1]):
                still_failed = ~crc_ok[failed_indices]
                if not torch.any(still_failed):
                    break
                active_sub_idx = torch.nonzero(still_failed, as_tuple=False).squeeze(-1)
                active_global_idx = failed_indices[active_sub_idx]

                llr_retry = llr_coset[active_global_idx].clone()
                pos_to_flip = least_rel_pos[active_sub_idx, flip_idx]
                row_idx = torch.arange(active_global_idx.shape[0], device=llr_ch.device)
                # Strongly invert the vulnerable LLR to force list exploration of opposite branch
                llr_retry[row_idx, pos_to_flip] = -torch.sign(
                    llr_retry[row_idx, pos_to_flip] + 1e-6
                ) * 12.0

                core_to_use = self._scl_retry_core if self._scl_retry_core is not None else self._scl_core
                bits_retry, pm_retry = core_to_use._decode(llr_retry)
                key_a_sub = key_a[active_global_idx] if key_a is not None else None
                payload_retry, crc_retry = self._select_best_path_with_crypto_crc(
                    bits_retry, pm_retry, key_a_sub
                )

                if torch.any(crc_retry):
                    rec_global = active_global_idx[crc_retry]
                    best_payload[rec_global] = payload_retry[crc_retry]
                    crc_ok[rec_global] = True

        out_shape = list(input_shape[:-1]) + [self._k_payload]
        best_payload = best_payload.reshape(out_shape).to(self.dtype)
        if return_crc_status:
            return best_payload, crc_ok.reshape(list(input_shape[:-1]))
        return best_payload


class CryptoPolarDecoder(Block):
    """Unified Crypto-Polar Decoder supporting SC, BP, SCL, CA-SCL, and
    Improved-CA-SCL for comparative evaluation in WoS Q1/Q2 benchmarks.
    """

    def __init__(
        self,
        frozen_pos: np.ndarray,
        n: int,
        dec_type: str = "Improved-CA-SCL",
        list_size: int = 8,
        num_iter: int = 20,
        crc_degree: Optional[str] = None,
        enable_info_masking: bool = True,
        enable_frozen_crypto: bool = True,
        max_flip_trials: int = 4,
        *,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)
        self._n = int(n)
        self._frozen_pos = np.asarray(frozen_pos, dtype=int)
        self._k = self._n - len(self._frozen_pos)
        self._dec_type = dec_type.upper()
        self._enable_info_masking = enable_info_masking
        self._enable_frozen_crypto = enable_frozen_crypto

        if self._dec_type in ("IMPROVED-CA-SCL", "CA-SCL", "IMPROVED-SCL"):
            flips = max_flip_trials if "IMPROVED" in self._dec_type else 0
            damping = 0.88 if "IMPROVED" in self._dec_type else 1.0
            self._decoder = ImprovedCryptoPolarSCLDecoder(
                frozen_pos=self._frozen_pos,
                n=self._n,
                list_size=list_size,
                crc_degree=crc_degree,
                enable_info_masking=enable_info_masking,
                enable_frozen_crypto=enable_frozen_crypto,
                llr_damping=damping,
                max_flip_trials=flips,
                precision=precision,
                device=device,
            )
            self._k_payload = self._decoder.k_payload
            self._crc_decoder = None
        else:
            if self._dec_type == "SC":
                self._decoder = PolarSCDecoder(
                    self._frozen_pos, self._n, precision=precision, device=device
                )
            elif self._dec_type == "SCL":
                self._decoder = PolarSCLDecoder(
                    self._frozen_pos,
                    self._n,
                    list_size=list_size,
                    crc_degree=None,
                    precision=precision,
                    device=device,
                )
            elif self._dec_type == "BP":
                self._decoder = PolarBPDecoder(
                    self._frozen_pos,
                    self._n,
                    num_iter=num_iter,
                    hard_out=True,
                    precision=precision,
                    device=device,
                )
            else:
                raise ValueError(f"Unsupported dec_type: {dec_type}")

            if crc_degree is not None:
                crc_enc = CRCEncoder(crc_degree, precision=precision, device=device)
                self._crc_decoder = CRCDecoder(
                    crc_enc, precision=precision, device=device
                )
                self._k_payload = self._k - crc_enc.crc_length
            else:
                self._crc_decoder = None
                self._k_payload = self._k

    @property
    def k_payload(self) -> int:
        return self._k_payload

    def build(self, *input_shapes) -> None:
        if input_shapes[0][-1] != self._n:
            raise ValueError("Invalid input shape.")

    def call(
        self,
        llr_ch: torch.Tensor,
        secret_key_stream: Optional[torch.Tensor] = None,
        csi_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(self._decoder, ImprovedCryptoPolarSCLDecoder):
            w = csi_weights if "IMPROVED" in self._dec_type else None
            return self._decoder(
                llr_ch, secret_key_stream=secret_key_stream, csi_weights=w
            )

        input_shape = llr_ch.shape
        B = int(np.prod(input_shape[:-1])) if len(input_shape) > 1 else 1
        logits = llr_ch.reshape(B, self._n)

        key_a, key_b = None, None
        n_frz = self._n - self._k
        if secret_key_stream is not None:
            k_dim = secret_key_stream.shape[-1]
            if k_dim == self._n:
                ks = secret_key_stream.reshape(-1, self._n)
                if ks.shape[0] == 1 and B > 1:
                    ks = ks.expand(B, -1)
                key_a = ks[:, : self._k]
                key_b = ks[:, self._k :]
            elif k_dim == n_frz:
                key_b = secret_key_stream.reshape(-1, n_frz)
                if key_b.shape[0] == 1 and B > 1:
                    key_b = key_b.expand(B, -1)

        frozen_bits = key_b if self._enable_frozen_crypto else None
        u_hat = self._decoder(logits, frozen_bits=frozen_bits)

        if self._enable_info_masking and key_a is not None:
            u_hat = (
                torch.bitwise_xor(u_hat.to(torch.uint8), key_a.to(torch.uint8))
            ).to(u_hat.dtype)

        if self._crc_decoder is not None:
            u_hat, _ = self._crc_decoder(u_hat)

        return u_hat.reshape(list(input_shape[:-1]) + [self._k_payload])

