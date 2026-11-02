"""
deploy_nodes.py
===============================================================
IEEE Phase 1 – UM-MIMO Node Deployment

Compatible with:
    Sionna RT 2.0.1
    TensorFlow 2.21

Deployment:
    • BS : 16 × 16 planar array (256 antennas)
    • Bob: 4 × 4 planar array (16 antennas)
    • Eve: 4 × 4 planar array (16 antennas)
===============================================================
"""

import numpy as np

from sionna.rt import (
    PlanarArray,
    Transmitter,
    Receiver,
)

from phase1_digital_twin import config


# ===============================================================
# HELPER FUNCTIONS
# ===============================================================

def euclidean_distance(p1, p2):
    """3D Euclidean distance."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return float(np.linalg.norm(p1 - p2))


def azimuth_elevation(tx_pos, rx_pos):
    """
    Compute azimuth and elevation angles (degree)
    from transmitter to receiver.
    """

    tx = np.asarray(tx_pos, dtype=float)
    rx = np.asarray(rx_pos, dtype=float)

    diff = rx - tx

    dx, dy, dz = diff

    horizontal = np.sqrt(dx**2 + dy**2)

    azimuth = np.degrees(np.arctan2(dy, dx))
    elevation = np.degrees(np.arctan2(dz, horizontal))

    return azimuth, elevation


# ===============================================================
# ARRAY CONFIGURATION
# ===============================================================

def build_bs_array():
    """16×16 Base Station Array"""

    return PlanarArray(
        num_rows=config.TX_NUM_ROWS,
        num_cols=config.TX_NUM_COLS,
        vertical_spacing=config.ANT_SPACING,
        horizontal_spacing=config.ANT_SPACING,
        pattern=config.TX_PATTERN,
        polarization=config.TX_POLARIZATION,
    )


def build_rx_array():
    """4×4 UE Array"""

    return PlanarArray(
        num_rows=config.RX_NUM_ROWS,
        num_cols=config.RX_NUM_COLS,
        vertical_spacing=config.ANT_SPACING,
        horizontal_spacing=config.ANT_SPACING,
        pattern=config.RX_PATTERN,
        polarization=config.RX_POLARIZATION,
    )


# ===============================================================
# REMOVE OLD NODES
# ===============================================================

def remove_existing_nodes(scene):
    """
    Make notebook / VSCode execution idempotent.
    """

    if "BS" in scene.transmitters:
        scene.remove(scene.transmitters["BS"])

    if "Bob" in scene.receivers:
        scene.remove(scene.receivers["Bob"])

    if "Eve" in scene.receivers:
        scene.remove(scene.receivers["Eve"])


# ===============================================================
# DEPLOYMENT REPORT
# ===============================================================

def print_topology():
    """
    IEEE topology report.
    """

    d_bb = euclidean_distance(config.BS_POS, config.BOB_POS)
    d_be = euclidean_distance(config.BS_POS, config.EVE_POS)
    d_bobe = euclidean_distance(config.BOB_POS, config.EVE_POS)

    az_bob, el_bob = azimuth_elevation(config.BS_POS, config.BOB_POS)
    az_eve, el_eve = azimuth_elevation(config.BS_POS, config.EVE_POS)

    print("=" * 60)
    print("EXPERIMENTAL SETUP : NODE TOPOLOGY")
    print("=" * 60)

    print(f"Base Station Position : {config.BS_POS}")
    print(f"Bob Position          : {config.BOB_POS}")
    print(f"Eve Position          : {config.EVE_POS}")

    print("-" * 60)

    print(f"BS Array Size         : {config.TX_NUM_ROWS} × {config.TX_NUM_COLS}")
    print(f"UE Array Size         : {config.RX_NUM_ROWS} × {config.RX_NUM_COLS}")

    print(f"BS Antennas           : {config.TX_NUM_ANT}")
    print(f"UE Antennas           : {config.RX_NUM_ANT}")

    print("-" * 60)

    print(f"BS → Bob Distance     : {d_bb:.2f} m")
    print(f"BS → Eve Distance     : {d_be:.2f} m")
    print(f"Bob ↔ Eve Separation  : {d_bobe:.2f} m")

    print("-" * 60)

    print(f"BS → Bob Azimuth      : {az_bob:.2f}°")
    print(f"BS → Bob Elevation    : {el_bob:.2f}°")

    print(f"BS → Eve Azimuth      : {az_eve:.2f}°")
    print(f"BS → Eve Elevation    : {el_eve:.2f}°")

    print("=" * 60)


# ===============================================================
# MAIN DEPLOYMENT FUNCTION
# ===============================================================

def setup_transceivers(scene, verbose=True):
    """
    Deploy BS, Bob and Eve.

    Returns
    -------
    scene, bs, bob, eve
    """

    if verbose:
        print("[1/3] Configuring UM-MIMO Planar Arrays...")

    # ----------------------------------------------------------
    # Remove existing nodes
    # ----------------------------------------------------------
    remove_existing_nodes(scene)

    # ----------------------------------------------------------
    # Create arrays
    # ----------------------------------------------------------
    tx_array = build_bs_array()
    rx_array = build_rx_array()

    # Sionna RT 2.0.1 API
    scene.tx_array = tx_array
    scene.rx_array = rx_array

    if verbose:
        print("      -> BS Array  : 16×16 (256 elements)")
        print("      -> UE Arrays : 4×4 (16 elements)")

    # ----------------------------------------------------------
    # Deploy Base Station
    # ----------------------------------------------------------
    bs = Transmitter(
        name="BS",
        position=config.BS_POS,
        orientation=config.BS_ORI,
    )

    scene.add(bs)

    # ----------------------------------------------------------
    # Deploy Bob
    # ----------------------------------------------------------
    bob = Receiver(
        name="Bob",
        position=config.BOB_POS,
        orientation=config.BOB_ORI,
    )

    scene.add(bob)

    # ----------------------------------------------------------
    # Deploy Eve
    # ----------------------------------------------------------
    eve = Receiver(
        name="Eve",
        position=config.EVE_POS,
        orientation=config.EVE_ORI,
    )

    scene.add(eve)

    # ----------------------------------------------------------
    # Verify deployment
    # ----------------------------------------------------------
    assert "BS" in scene.transmitters
    assert "Bob" in scene.receivers
    assert "Eve" in scene.receivers

    if verbose:

        print("[2/3] Transceivers successfully deployed.")

        print_topology()

        print("[3/3] Deployment Completed.")

    return scene, bs, bob, eve