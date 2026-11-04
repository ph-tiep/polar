"""Export construction profiles from explicit Bob CSI files; no synthetic errors."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from phase2_crypto_polar.construction import effective_design_gain


def export(evaluation_bob, nominal_bob, scenarios, output):
    paths = [Path(evaluation_bob), Path(nominal_bob), *map(Path, scenarios)]
    reference = np.load(paths[0], allow_pickle=False)
    nominal = effective_design_gain(reference, np.load(paths[1], allow_pickle=False))
    alternatives = [effective_design_gain(reference, np.load(p, allow_pickle=False)) for p in paths[2:]]
    provenance = dict(inputs=[dict(path=str(p.resolve()), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        order='evaluation, nominal, scenarios',
        assumptions='same frequency grid and antenna ordering; fixed evaluation-CSI transmit beam and coherent MRC',
        scope='construction-profile mismatch only; no simulated estimation-error distribution')
    arrays = dict(nominal_gain=nominal, NumTxAntennas=reference.shape[1], NumRxAntennas=reference.shape[0],
                  NumSubcarriers=reference.shape[2], gain_reference='evaluation_bob_mean', provenance_json=json.dumps(provenance))
    if alternatives:
        arrays['scenario_gains'] = np.stack(alternatives)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-bob', type=Path, required=True)
    parser.add_argument('--nominal-bob', type=Path, required=True)
    parser.add_argument('--scenario-bob', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(export(args.evaluation_bob, args.nominal_bob, args.scenario_bob, args.output))


if __name__ == '__main__':
    main()
