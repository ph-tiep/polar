"""Regenerate readable figures from a completed Phase 2 run, without simulation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from phase2_crypto_polar.plotting_q1 import plot_case, validate_plot_data

TEXT_FIELDS = {'curve', 'group', 'method', 'scope', 'construction'}
INT_FIELDS = {'n', 'k', 'payload_bits', 'list_size', 'frames', 'bits', 'bit_errors',
              'block_errors', 'base_frames', 'expanded_frames', 'flip_frames',
              'core_calls', 'max_list_size', 'flipped_key_bits', 'eligible_key_bits'}


def read_metrics(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return [{key: value if key in TEXT_FIELDS else None if value == '' else
             int(value) if key in INT_FIELDS else float(value)
             for key, value in row.items()} for row in rows]


def replot(run_dir, output_dir):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    manifest_path = run_dir/'RUN_CONFIG.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('purpose') != 'experiment':
        raise ValueError('This is a smoke/unknown run. Select phase2-review or run without --smoke.')
    if manifest.get('status') != 'completed':
        raise ValueError('Run is not completed; no final research curves can be generated.')
    if len(set(manifest['config']['ebno_db'])) < 2:
        raise ValueError('Only one SNR point: no curve can be recovered from this run.')
    inputs = []
    for case in manifest['config']['cases']:
        folder = f'N{int(case["n"])}_K{int(case["k"])}'
        metrics = run_dir/folder/'metrics.csv'
        diag_path = run_dir/folder/'diagnostics.json'
        rows = read_metrics(metrics)
        validate_plot_data(rows)
        if {r['ebno_db'] for r in rows} != set(manifest['config']['ebno_db']):
            raise ValueError('CSV SNR grid does not match the run manifest')
        for r in rows:
            if (r['n'], r['k']) != (case['n'], case['k']):
                raise ValueError('CSV N/K do not match the run manifest')
        for curve in {r['curve'] for r in rows}:
            points = [r['ebno_db'] for r in rows if r['curve'] == curve]
            if len(points) != len(set(points)) or set(points) != set(manifest['config']['ebno_db']):
                raise ValueError(f'Duplicate or missing SNR values for {curve}')
        inputs.append((folder, metrics, diag_path, rows, json.loads(diag_path.read_text(encoding='utf-8'))))
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance = dict(status='running', source_run=str(run_dir),
                      action='replot measured data only; no simulation, no smoothing',
                      source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), cases={})
    shutil.copyfile(manifest_path, output_dir/'SOURCE_RUN_CONFIG.json')
    try:
        for folder, metrics, diag_path, rows, diag in inputs:
            target = output_dir/folder
            target.mkdir()
            shutil.copyfile(metrics, target/'metrics.csv')
            shutil.copyfile(diag_path, target/'diagnostics.json')
            for name in ('construction.json', 'construction_arrays.npz', 'key_mismatch_audit.csv', 'paired_metrics.csv', 'frame_errors.npz'):
                source = metrics.parent/name
                if source.exists():
                    shutil.copyfile(source, target/name)
            provenance['cases'][folder] = dict(source_metrics_sha256=hashlib.sha256(metrics.read_bytes()).hexdigest())
            plot_case(rows, diag, target, manifest['config'])
            assert (target/'metrics.csv').read_bytes() == metrics.read_bytes()
        provenance['status'] = 'completed'
    except BaseException as exc:
        provenance.update(status='failed', error=str(exc))
        raise
    finally:
        (output_dir/'REPLOT_INFO.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(f'Figures: {replot(args.run_dir, args.output_dir)}')


if __name__ == '__main__':
    main()
