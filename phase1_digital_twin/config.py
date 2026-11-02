"""
config.py

Global configuration file for IEEE Phase 1
6G Digital Twin at 140 GHz

Compatible with:
    TensorFlow 2.21
    Sionna RT 2.0.1
"""

import os
import random
import numpy as np
import tensorflow as tf

# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PHASE1_DIR = os.path.join(BASE_DIR, "phase1_digital_twin")

SCENE_DIR = os.path.join(BASE_DIR, "munich")
SCENE_NAME = os.path.join(SCENE_DIR, "munich.xml")

DATASET_DIR = os.path.join(BASE_DIR, "dataset", "140GHz")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for folder in [DATASET_DIR, FIGURE_DIR, LOG_DIR]:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 2026

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ============================================================
# 6G SUB-THZ PARAMETERS
# ============================================================

CARRIER_FREQ = 140e9
BANDWIDTH = 2e9

NUM_SUBCARRIERS = 512
FFT_SIZE = NUM_SUBCARRIERS

SUBCARRIER_SPACING = BANDWIDTH / NUM_SUBCARRIERS

FREQ_START = CARRIER_FREQ - BANDWIDTH / 2
FREQ_END = CARRIER_FREQ + BANDWIDTH / 2

# ============================================================
# OFDM PARAMETERS
# ============================================================

CP_LENGTH = 64
SAMPLING_RATE = BANDWIDTH

OFDM_SYMBOL_DURATION = 1 / SUBCARRIER_SPACING

# ============================================================
# RADIO PARAMETERS
# ============================================================

NOISE_FIGURE = 7.0
TEMPERATURE = 290

BOLTZMANN = 1.380649e-23

THERMAL_NOISE_POWER = (
    BOLTZMANN *
    TEMPERATURE *
    BANDWIDTH
)

THERMAL_NOISE_DBM = 10 * np.log10(THERMAL_NOISE_POWER / 1e-3)

TX_POWER_DBM = 30
TX_POWER_WATT = 10 ** ((TX_POWER_DBM - 30) / 10)

# ============================================================
# DIGITAL TWIN OPTIONS
# ============================================================

SYNTHETIC_ARRAY = True

ENABLE_MOLECULAR_ABSORPTION = True
ENABLE_LOGNORMAL_SHADOWING = True

SHADOW_STD_LOS = 4
SHADOW_STD_NLOS = 7

# ============================================================
# ANTENNA CONFIGURATION
# ============================================================

TX_NUM_ROWS = 16
TX_NUM_COLS = 16

RX_NUM_ROWS = 4
RX_NUM_COLS = 4

TX_NUM_ANT = TX_NUM_ROWS * TX_NUM_COLS
RX_NUM_ANT = RX_NUM_ROWS * RX_NUM_COLS

ANT_SPACING = 0.5

TX_PATTERN = "tr38901"
RX_PATTERN = "tr38901"

TX_POLARIZATION = "V"
RX_POLARIZATION = "V"

# ============================================================
# NODE POSITIONS
# ============================================================

BS_POS = [0.0, 0.0, 25.0]
BS_ORI = [0.0, 0.0, 0.0]

BOB_POS = [45.0, 10.0, 1.5]
BOB_ORI = [0.0, 0.0, 0.0]

EVE_POS = [42.0, 12.0, 1.5]
EVE_ORI = [0.0, 0.0, 0.0]

# ============================================================
# RAY TRACING PARAMETERS
# ============================================================

RT_MAX_DEPTH = 3

RT_NUM_SAMPLES_DEBUG = int(2e5)
RT_NUM_SAMPLES_FULL = int(1e6)

RT_LOS = True
RT_REFLECTION = True
RT_DIFFRACTION = False
RT_SCATTERING = False

# ============================================================
# OUTPUT FILE NAMES
# ============================================================

CSI_BOB_FILE = "H_Bob.npy"
CSI_EVE_FILE = "H_Eve.npy"

CIR_GAIN_FILE = "CIR_Gain.npy"
CIR_DELAY_FILE = "CIR_Delay.npy"

PATHLOSS_FILE = "PathLoss.npy"
PDP_FILE = "PowerDelayProfile.npy"

MAT_DATASET_FILE = "CSI_Dataset_140GHz.mat"

# ============================================================
# IEEE FIGURE OUTPUT
# ============================================================

PDP_FIG = os.path.join(FIGURE_DIR, "PDP_140GHz.png")
CSI_MAG_FIG = os.path.join(FIGURE_DIR, "CSI_Magnitude.png")
PATHLOSS_FIG = os.path.join(FIGURE_DIR, "PathLoss_CDF.png")
TOPOLOGY_FIG = os.path.join(FIGURE_DIR, "Network_Topology.png")

FIGURE_DPI = 600

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wavelength():
    """Return wavelength in meter."""
    c = 299792458.0
    return c / CARRIER_FREQ


def antenna_spacing_meter():
    """Physical antenna spacing."""
    return ANT_SPACING * wavelength()


def print_configuration():
    """Pretty print IEEE experiment configuration."""

    print("=" * 60)
    print("140 GHz DIGITAL TWIN CONFIGURATION")
    print("=" * 60)

    print(f"Carrier Frequency   : {CARRIER_FREQ/1e9:.1f} GHz")
    print(f"Bandwidth           : {BANDWIDTH/1e9:.1f} GHz")
    print(f"Subcarriers         : {NUM_SUBCARRIERS}")
    print(f"Subcarrier Spacing  : {SUBCARRIER_SPACING/1e6:.4f} MHz")
    print()

    print(f"BS Array            : {TX_NUM_ROWS} x {TX_NUM_COLS}")
    print(f"UE Array            : {RX_NUM_ROWS} x {RX_NUM_COLS}")
    print(f"Antenna Spacing     : {ANT_SPACING:.1f} λ")
    print()

    print(f"Scene File          : {SCENE_NAME}")
    print(f"Dataset Folder      : {DATASET_DIR}")

    print("=" * 60)