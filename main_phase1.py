"""
main_phase1.py
======================================================================
IEEE Phase 1 : 140 GHz Digital Twin Dataset Generation Pipeline

Project:
    6G Secure Massive MIMO with Polar Crypto-Coding

Environment:
    Python 3.11
    TensorFlow 2.21
    Sionna RT 2.0.1

Pipeline
--------
1. Load Digital Twin Scene.
2. Replace unsupported ITU materials.
3. Deploy UM-MIMO BS / Bob / Eve.
4. Perform Ray Tracing.
5. Convert CIR -> CSI.
6. Validate channel statistics.
7. Export MATLAB/Python dataset.
======================================================================
"""

import time
import traceback
import tensorflow as tf
import sionna

from phase1_digital_twin import config

from phase1_digital_twin.build_scene import init_and_map_scene
from phase1_digital_twin.deploy_nodes import setup_transceivers
from phase1_digital_twin.raytrace_csi import extract_csi
from phase1_digital_twin.validation_metrics import (
    evaluate_channel,
    save_validation_figures,
)
from phase1_digital_twin.save_dataset import export_dataset


# ===============================================================
# GPU CONFIGURATION
# ===============================================================

def configure_hardware():
    """Configure GPU memory growth if CUDA is available."""

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("[System] CPU execution.")
        return

    print(f"[System] GPU detected ({len(gpus)} device(s)).")

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("[System] TensorFlow GPU memory growth enabled.")

    except RuntimeError as e:
        print(e)


# ===============================================================
# PRINT CONFIGURATION
# ===============================================================

def print_configuration():
    print("=" * 60)
    print("140 GHz DIGITAL TWIN CONFIGURATION")
    print("=" * 60)

    print(f"Carrier Frequency   : {config.CARRIER_FREQ/1e9:.1f} GHz")
    print(f"Bandwidth           : {config.BANDWIDTH/1e9:.1f} GHz")

    print(f"Subcarriers         : {config.NUM_SUBCARRIERS}")
    print(f"Subcarrier Spacing  : {config.SUBCARRIER_SPACING/1e6:.4f} MHz")

    print()

    print(f"BS Array            : {config.TX_NUM_ROWS} × {config.TX_NUM_COLS}")
    print(f"UE Array            : {config.RX_NUM_ROWS} × {config.RX_NUM_COLS}")

    print(f"Antenna Spacing     : {config.ANT_SPACING} λ")

    print()

    print(f"Scene File          : {config.SCENE_NAME}")
    print(f"Dataset Folder      : {config.DATASET_DIR}")

    print("=" * 60)
    print()


# ===============================================================
# MAIN PIPELINE
# ===============================================================

def main(run_mode="debug"):

    pipeline_start = time.time()

    print("=" * 70)
    print("PHASE 1 : DIGITAL TWIN + RAY TRACING DATASET GENERATION")
    print("=" * 70)

    print(f"TensorFlow Version : {tf.__version__}")
    print(f"Sionna Version     : {sionna.__version__}")
    print()

    print_configuration()

    configure_hardware()

    timing = {}

    try:

        # ------------------------------------------------------
        # STEP 1
        # ------------------------------------------------------

        t0 = time.time()

        scene = init_and_map_scene(verbose=True)

        timing["Build Scene"] = time.time() - t0

        print()

        # ------------------------------------------------------
        # STEP 2
        # ------------------------------------------------------

        t0 = time.time()

        scene, bs, bob, eve = setup_transceivers(
            scene,
            verbose=True,
        )

        timing["Deploy Nodes"] = time.time() - t0

        print()

        # ------------------------------------------------------
        # STEP 3
        # ------------------------------------------------------

        t0 = time.time()

        (
            H_Bob,
            H_Eve,
            a,
            tau,
            paths,
            ray_metrics,
        ) = extract_csi(
            scene,
            run_mode=run_mode,
            verbose=True,
        )

        timing["Ray Tracing"] = time.time() - t0

        print()

        # ------------------------------------------------------
        # STEP 4
        # ------------------------------------------------------

        t0 = time.time()

        validation = evaluate_channel(
            H_Bob,
            H_Eve,
            a,
            tau,
            verbose=True,
        )

        timing["Validation"] = time.time() - t0

        print()

        # ------------------------------------------------------
        # STEP 5
        # ------------------------------------------------------

        t0 = time.time()

        save_validation_figures(
            a,
            tau,
            output_dir=config.DATASET_DIR,
        )

        timing["Figures"] = time.time() - t0

        print()

        # ------------------------------------------------------
        # STEP 6
        # ------------------------------------------------------

        t0 = time.time()

        validation.update(ray_metrics)

        export_dataset(
            H_Bob,
            H_Eve,
            a,
            tau,
            validation_metrics=validation,
            verbose=True,
        )

        timing["Export Dataset"] = time.time() - t0

        # ------------------------------------------------------
        # SUMMARY
        # ------------------------------------------------------

        total_runtime = time.time() - pipeline_start

        print()

        print("=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)

        for k, v in timing.items():
            print(f"{k:<25}: {v:>8.2f} s")

        print("-" * 70)
        print(f"{'TOTAL':<25}: {total_runtime:>8.2f} s")
        print("=" * 70)

        print()

        print("OUTPUT DATASET")

        print(f"Directory : {config.DATASET_DIR}")

        print("""
Files Generated
---------------
H_Bob.npy
H_Eve.npy
CIR_Gain.npy
CIR_Delay.npy
CSI_Dataset_140GHz.mat
Metadata.npz
PDP_140GHz.png
PDP_140GHz.pdf
Delay_CDF.png
Delay_CDF.pdf
""")

        print("=" * 70)
        print("PHASE 1 COMPLETED SUCCESSFULLY")
        print("READY FOR PHASE 2 (MATLAB MASSIVE MIMO + PLS)")
        print("=" * 70)

    except Exception:

        print()
        print("=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)

        traceback.print_exc()

        raise


# ===============================================================
# ENTRY POINT
# ===============================================================

if __name__ == "__main__":

    # debug  -> nhanh (~2e5 rays)
    # full   -> dataset bài báo (1e6 rays)

    main(run_mode="debug")