#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Sionna PHY Forward Error Correction (FEC) Package (Polar & CRC Active)."""

from . import polar
from . import crc
from .crc import CRCEncoder, CRCDecoder
from .polar import (
    PolarEncoder,
    Polar5GEncoder,
    PolarSCDecoder,
    PolarSCLDecoder,
    PolarBPDecoder,
    Polar5GDecoder,
    ChannelBoundKeyGenerator,
    CryptoPolarEncoder,
    CryptoPolarDecoder,
    ImprovedCryptoPolarSCLDecoder,
    generate_5g_ranking,
    generate_polar_transform_mat,
    generate_rm_code,
    generate_dense_polar,
)

__all__ = [
    "polar",
    "crc",
    "CRCEncoder",
    "CRCDecoder",
    "PolarEncoder",
    "Polar5GEncoder",
    "PolarSCDecoder",
    "PolarSCLDecoder",
    "PolarBPDecoder",
    "Polar5GDecoder",
    "ChannelBoundKeyGenerator",
    "CryptoPolarEncoder",
    "CryptoPolarDecoder",
    "ImprovedCryptoPolarSCLDecoder",
    "generate_5g_ranking",
    "generate_polar_transform_mat",
    "generate_rm_code",
    "generate_dense_polar",
]
