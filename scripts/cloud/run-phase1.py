"""Run Phase 1 in an isolated output folder with an explicit scene layout."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# RT must initialize its native libraries before TensorFlow.
import sionna.rt
import main_phase1


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['debug', 'full'], default='debug')
    parser.add_argument('--layout', choices=['original', 'outdoor-review'], default='original')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--tx-array', type=int, nargs=2, metavar=('ROWS', 'COLS'), default=[16, 16])
    parser.add_argument('--rx-array', type=int, nargs=2, metavar=('ROWS', 'COLS'), default=[4, 4])
    args = parser.parse_args()
    if min(args.tx_array + args.rx_array) < 1:
        parser.error('Array dimensions must be positive')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (args.output_dir or ROOT / 'results' / f'phase1-{args.layout}-{stamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = main_phase1.config
    config.TX_NUM_ROWS, config.TX_NUM_COLS = args.tx_array
    config.RX_NUM_ROWS, config.RX_NUM_COLS = args.rx_array
    config.TX_NUM_ANT = config.TX_NUM_ROWS * config.TX_NUM_COLS
    config.RX_NUM_ANT = config.RX_NUM_ROWS * config.RX_NUM_COLS
    if args.layout == 'outdoor-review':
        # Original BS sits below a roof at z=26 m. This explicit review
        # scenario moves only the BS to outdoor space with clear Bob/Eve LOS.
        config.BS_POS = [20.0, 20.0, 25.0]
    manifest = dict(tx_array=args.tx_array, rx_array=args.rx_array, layout=args.layout, mode=args.mode, status='running',
                    BS=config.BS_POS, Bob=config.BOB_POS, Eve=config.EVE_POS,
                    output_dir=str(output), seed=config.RANDOM_SEED,
                    samples_per_src=(config.RT_NUM_SAMPLES_FULL if args.mode == 'full' else config.RT_NUM_SAMPLES_DEBUG),
                    scene=config.SCENE_NAME,
                    note=('BS moved to outdoor space; Bob/Eve unchanged' if args.layout == 'outdoor-review' else 'Original node coordinates unchanged'))
    path = output / 'RUN_CONFIG.json'
    path.write_text(json.dumps(manifest, indent=2))
    try:
        main_phase1.main(run_mode=args.mode, output_dir=output)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    else:
        manifest['status'] = 'completed'
    finally:
        path.write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    run()
