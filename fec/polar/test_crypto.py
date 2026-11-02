"""
test_crypto.py
======================================================================
Verification & Unit Test Suite for Dynamic Frozen-Bit Crypto-Polar
Coding and Improved Coset-Aware Decoding (SC, BP, SCL, Improved-CA-SCL).
======================================================================
"""

import os
import sys
import torch
import numpy as np

# Ensure project root is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fec.polar.utils import generate_5g_ranking
from fec.polar.crypto_polar import (
    ChannelBoundKeyGenerator,
    CryptoPolarEncoder,
    CryptoPolarDecoder,
)


def run_crypto_polar_unit_tests():
    print("=" * 72)
    print("CRYPTO-POLAR ENCODING & IMPROVED DECODING VERIFICATION SUITE")
    print("=" * 72)

    k = 128
    n = 256
    batch_size = 32
    crc_degree = "CRC11"

    frozen_pos, info_pos = generate_5g_ranking(k, n)

    encoder = CryptoPolarEncoder(
        frozen_pos=frozen_pos,
        n=n,
        crc_degree=crc_degree,
        enable_info_masking=True,
        enable_frozen_crypto=True,
    )
    k_payload = encoder.k_payload

    key_gen = ChannelBoundKeyGenerator(master_key=b"IEEE_Q1_TEST_SECRET_2026")
    secret_key = key_gen.generate_key_stream(batch_size=batch_size, n=n, frame_idx=1)
    wrong_key = key_gen.generate_key_stream(batch_size=batch_size, n=n, frame_idx=2)

    payload_bits = torch.randint(0, 2, (batch_size, k_payload), dtype=torch.float32)
    codewords = encoder(payload_bits, secret_key)

    print(f"[1] Code Configuration : N={n}, K_polar={k}, K_payload={k_payload} ({crc_degree})")
    print(f"    Frozen Bit Count   : {len(frozen_pos)} dynamic cryptographic bits")
    print(f"    Output Codeword    : {tuple(codewords.shape)}")

    # High-SNR BPSK logits (convention: logit = 2*cw - 1 > 0 for bit 1, < 0 for bit 0)
    llr_noiseless = 20.0 * (2.0 * codewords - 1.0)

    for dec_name in ["SC", "BP", "SCL", "CA-SCL", "Improved-CA-SCL"]:
        crc_arg = crc_degree if "CA" in dec_name or dec_name in ["SC", "BP", "SCL"] else None
        decoder = CryptoPolarDecoder(
            frozen_pos=frozen_pos,
            n=n,
            dec_type=dec_name,
            list_size=8,
            num_iter=20,
            crc_degree=crc_arg,
        )
        # Bob (with correct synchronized key)
        bob_hat = decoder(llr_noiseless, secret_key_stream=secret_key)
        ber_bob = torch.mean(torch.abs(payload_bits - bob_hat)).item()

        # Eve Case 1: Naive Eve (assumes standard frozen bits = 0)
        eve_naive_hat = decoder(llr_noiseless, secret_key_stream=None)
        ber_eve_naive = torch.mean(torch.abs(payload_bits - eve_naive_hat)).item()

        # Eve Case 2: Wrong Key Eve
        eve_wrong_hat = decoder(llr_noiseless, secret_key_stream=wrong_key)
        ber_eve_wrong = torch.mean(torch.abs(payload_bits - eve_wrong_hat)).item()

        print(
            f"    Decoder [{dec_name:<15}] -> Bob BER: {ber_bob:.6f} | "
            f"Eve (Naive) BER: {ber_eve_naive:.4f} | Eve (Wrong Key) BER: {ber_eve_wrong:.4f}"
        )
        assert ber_bob == 0.0, f"Bob failed noiseless decoding for {dec_name}!"

    print("=" * 72)
    print("[SUCCESS] All Crypto-Polar Encoders & Decoders verified with 0.0 Bob BER!")
    print("=" * 72)


if __name__ == "__main__":
    run_crypto_polar_unit_tests()
