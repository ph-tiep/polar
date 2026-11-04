"""Run the advisor-review CA-SCL experiments. See RUN_VSCODE.md."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from phase2_crypto_polar.experiments import run, validate_config

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/phase2_dt_review.json')
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--expected-tx', type=int, required=True)
    parser.add_argument('--expected-rx', type=int, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plan', action='store_true', help='Print resolved config; no CSI loading or experiment')
    parser.add_argument('--smoke', action='store_true', help='Tiny integration run, not scientific results')
    parser.add_argument('--key-file', type=Path, help='Existing 32-byte local shared key; otherwise create a private simulation key')
    parser.add_argument('--key-nonce', help='32 hex characters; use the old manifest nonce only to replay a simulation')
    parser.add_argument('--design-profiles', type=Path, help='NPZ containing calibrated nominal_gain and optional scenario_gains')
    args = parser.parse_args()
    c = json.loads(args.config.read_text(encoding='utf-8'))
    if args.smoke:
        c.update(frames_per_snr=2, batch_size=1, ebno_db=[0.], diagnostic_samples=2)
        c['cases'] = c['cases'][:1]
        print('SMOKE CHECK: one SNR, two frames; no performance figures. Run without --smoke for curves.')
    validate_config(c)
    if args.plan:
        print(json.dumps(dict(config=c, dataset=str(args.dataset_dir.resolve()), expected_tx=args.expected_tx, expected_rx=args.expected_rx,
                              design_profiles=str(args.design_profiles) if args.design_profiles else 'evaluation CSI (perfect-profile assumption)',
                              key_assumption='shared secret; new private local key unless --key-file is provided'), indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    kind = 'smoke' if args.smoke else 'review'
    output = args.output_dir or ROOT/'results'/f'phase2-{kind}-{stamp}'
    run(c, args.dataset_dir, output, expected_tx=args.expected_tx, expected_rx=args.expected_rx, smoke=args.smoke,
        key_file=args.key_file, key_nonce=args.key_nonce, design_profile_file=args.design_profiles)
    print(f'Completed: {output.resolve()}')
    if args.smoke:
        print('Next: run without --smoke and use a new output directory.')


if __name__ == '__main__':
    main()
