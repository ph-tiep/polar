"""Independent design checks and tiny integration fixtures, not research runs."""
import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from phase2_crypto_polar.construction import construct, polar_log_bounds, mapping_bank, design_profiles
from phase2_crypto_polar.construction_experiments import build_plans
from phase2_crypto_polar.digital_twin_channel import DigitalTwinTHzChannel
from phase2_crypto_polar.experiments import run, validate_config
from phase2_crypto_polar.keys import SharedKeyGenerator, load_run_key
from replot_phase2 import replot

ROOT = Path(__file__).resolve().parents[1]


def fixture_data(path):
    path.mkdir()
    h = np.sqrt(np.linspace(.1, 1.9, 64))[None, None, :].astype(np.complex64)
    np.save(path/'H_Bob.npy', h)
    np.save(path/'H_Eve.npy', .8*h)
    np.savez(path/'Metadata.npz', NumTxAntennas=1, NumRxAntennas=1, NumSubcarriers=64)


def fixture_config():
    c = json.loads((ROOT/'configs/phase2_dt_review.json').read_text())
    c.update(frames_per_snr=3, batch_size=2, cases=[dict(n=32, k=16)], crc='CRC6',
             ebno_db=[0., 1.], list_sizes=[2], eve_list_size=2, diagnostic_samples=2)
    c['modified']['base_list'] = 2
    return c


class ConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_recursion_order_against_exhaustive_bec_linear_constraints(self):
        # For a BEC the bound is exact. Enumerate erasures and all unknown
        # future source bits with a separately built Kronecker generator.
        probs = np.array([.13, .27, .52, .81])
        kernel = np.array([[1, 0], [1, 1]], dtype=np.uint8)
        matrix = np.kron(kernel, kernel)
        exact = np.zeros(4)
        for erasures in itertools.product([False, True], repeat=4):
            erased = np.array(erasures)
            probability = np.prod(np.where(erased, probs, 1-probs))
            for bit in range(4):
                ambiguous = False
                for future in itertools.product([0, 1], repeat=3-bit):
                    delta = np.zeros(4, dtype=np.uint8)
                    delta[bit] = 1
                    delta[bit+1:] = future
                    if not np.any(((delta@matrix) % 2)[~erased]):
                        ambiguous = True
                        break
                exact[bit] += probability*ambiguous
        np.testing.assert_allclose(np.exp(polar_log_bounds(np.log(probs))), exact, atol=1e-13)

    def test_awgn_initial_bound_matches_density_overlap(self):
        from scipy.integrate import quad
        from scipy.stats import norm
        gain, rate, snr = .7, .4, 1.5
        variance = 1/(2*rate*10**(snr/10))
        overlap = quad(lambda y: np.sqrt(norm.pdf(y, np.sqrt(gain), np.sqrt(variance))*
                                         norm.pdf(y, -np.sqrt(gain), np.sqrt(variance))), -12, 12)[0]
        _, _, scores = construct([gain, gain], 1, rate, snr)
        self.assertAlmostEqual(np.exp(scores[1]), overlap**2, places=12)

    def test_bounds_extremes_and_robust_selection(self):
        gains = np.array([[0., .01, .1, 2., 3., .4, .2, 8.], [1., 2., .1, .1, .2, 8., .3, .01]])
        frozen, info, score = construct(gains, 4, .4, 2.)
        individual = np.stack([construct(g, 4, .4, 2.)[2] for g in gains])
        np.testing.assert_allclose(score, individual.max(0))
        self.assertEqual(len(info), 4)
        self.assertEqual(len(set(info) & set(frozen)), 0)
        self.assertLessEqual(score[info].max(), score[frozen].min())
        self.assertTrue(np.isfinite(polar_log_bounds(np.array([0., -1e-300, -1e4, -1e4]))).all())
        for bad in ([1., -2.], [np.nan, 0.], [-1., -2., -3.]):
            with self.assertRaises(ValueError):
                polar_log_bounds(bad)

    def test_mapping_full_cycles_and_reproducibility(self):
        for n, carriers in [(32, 64), (128, 64), (32, 70)]:
            bank = mapping_bank(n, carriers, 2026)
            flat = bank.ravel()
            for offset in range(0, len(flat)-carriers+1, carriers):
                self.assertEqual(set(flat[offset:offset+carriers]), set(range(carriers)))
            np.testing.assert_array_equal(bank, mapping_bank(n, carriers, 2026))

    def test_key_replay_batching_and_domain_separation(self):
        key, nonce = bytes(range(32)), bytes(range(16))
        generator = SharedKeyGenerator(key, nonce, n=128, k=64, crc='CRC11')
        batched = generator.generate_key_stream(3, 128, 5)
        single = torch.cat([generator.generate_key_stream(1, 128, frame) for frame in range(5, 8)])
        torch.testing.assert_close(batched, single)
        for changed in (SharedKeyGenerator(key, bytes(16), n=128, k=64, crc='CRC11'),
                        SharedKeyGenerator(key, nonce, n=128, k=32, crc='CRC11')):
            self.assertFalse(torch.equal(batched, changed.generate_key_stream(3, 128, 5)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret, run_nonce, meta = load_run_key(root)
            self.assertEqual(len(secret), 32)
            self.assertNotIn(secret.hex(), json.dumps(meta))
            self.assertEqual(hashlib.sha256(secret).hexdigest(), meta['fingerprint_sha256'])
            secret2, nonce2, _ = load_run_key(root, meta['local_key_file'], meta['nonce'])
            self.assertEqual(secret, secret2)
            self.assertEqual(run_nonce, nonce2)

    def test_profile_requirements_and_roundtrip_every_group(self):
        c = fixture_config()
        c['construction']['methods'].append('robust')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_data(root/'data')
            ch = DigitalTwinTHzChannel(str(root/'data'))
            nominal, scenarios, meta = design_profiles(ch)
            self.assertIsNone(scenarios)
            self.assertFalse(meta['independent_prediction'])
            out = root/'plans'; out.mkdir()
            with self.assertRaisesRegex(ValueError, 'externally supplied'):
                build_plans(ch, c, c['cases'][0], (nominal, None), out)
            path = root/'profiles.npz'
            np.savez(path, nominal_gain=nominal, scenario_gains=nominal[::-1][None, :],
                     NumTxAntennas=1, NumRxAntennas=1, NumSubcarriers=64, gain_reference='evaluation_bob_mean')
            predicted, scenarios, meta = design_profiles(ch, path)
            self.assertEqual(meta['uncertainty_scenarios'], 1)
            plans, bank, payload_size = build_plans(ch, c, c['cases'][0], (predicted, scenarios), out)
            self.assertEqual(bank.shape, (2, 32))
            generator = torch.Generator().manual_seed(782)
            payload = torch.randint(0, 2, (2, payload_size), generator=generator).float()
            keys = torch.randint(0, 2, (2, 32), generator=generator).float()
            for method in plans:
                for plan in plans[method]:
                    for coding in ('standard', 'crypto'):
                        enc = plan['enc'] if coding == 'crypto' else plan['standard']
                        key = keys if coding == 'crypto' else None
                        torch.testing.assert_close(plan['decoders'][(coding, 2)](20*(2*enc(payload, key)-1), key), payload)
            np.savez(path, nominal_gain=nominal, NumTxAntennas=2, NumRxAntennas=1, NumSubcarriers=64, gain_reference='evaluation_bob_mean')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                design_profiles(ch, path)

    def test_all_configs_and_no_modified_decoder_confound(self):
        for path in (ROOT/'configs').glob('phase2*.json'):
            validate_config(json.loads(path.read_text()))
        c = fixture_config()
        c['modified']['enabled'] = True
        with self.assertRaisesRegex(ValueError, 'hold CA-SCL fixed'):
            validate_config(c)

    def test_export_profiles_projects_onto_same_transmit_beam(self):
        from prepare_design_profiles import export
        from phase2_crypto_polar.construction import effective_design_gain
        # Reference beam is [1,0]; the other antenna must not contribute.
        reference = np.array([[[2, 2], [0, 0]]], dtype=np.complex64)
        predicted = np.array([[[1, 3], [100, 100]]], dtype=np.complex64)
        np.testing.assert_allclose(effective_design_gain(reference, predicted), [.25, 2.25])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root/'evaluation.npy', reference)
            np.save(root/'nominal.npy', predicted)
            export(root/'evaluation.npy', root/'nominal.npy', [root/'evaluation.npy'], root/'profiles.npz')
            with np.load(root/'profiles.npz', allow_pickle=False) as data:
                np.testing.assert_allclose(data['nominal_gain'], [.25, 2.25])
                np.testing.assert_allclose(data['scenario_gains'], [[1., 1.]])
                self.assertEqual(len(json.loads(data['provenance_json'].item())['inputs']), 3)
            with self.assertRaises(FileExistsError):
                export(root/'evaluation.npy', root/'nominal.npy', [], root/'profiles.npz')

    def test_pipeline_paired_records_replay_and_replot(self):
        c = fixture_config()
        c['construction']['methods'].append('robust')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_data(root/'data')
            profile = root/'profiles.npz'
            nominal = np.linspace(.1, 1.9, 64)
            np.savez(profile, nominal_gain=nominal, scenario_gains=nominal[::-1][None, :],
                     NumTxAntennas=1, NumRxAntennas=1, NumSubcarriers=64, gain_reference='evaluation_bob_mean')
            run(c, root/'data', root/'out', expected_tx=1, expected_rx=1, design_profile_file=profile)
            manifest = json.loads((root/'out/RUN_CONFIG.json').read_text())
            self.assertEqual(manifest['status'], 'completed')
            folder = root/'out/N32_K16'
            with (folder/'metrics.csv').open() as f:
                rows = list(csv.DictReader(f))
            with np.load(folder/'frame_errors.npz') as data:
                self.assertEqual(data['frame_bit_errors'].shape[::2], (2, 3))
                names = data['curve_names'].tolist()
                for row in rows:
                    counts = data['frame_bit_errors'][int(float(row['ebno_db'])), names.index(row['curve'])]
                    self.assertEqual(counts.sum(), int(row['bit_errors']))
                for prefix in ('', 'dt_', 'robust_'):
                    direct = data['frame_bit_errors'][:, names.index(prefix+'eve_full_1'), :]
                    inverted = data['frame_bit_errors'][:, names.index(prefix+'eve_full_1_inverted'), :]
                    np.testing.assert_array_equal(direct+inverted, np.full_like(direct, 10))
            with (folder/'key_mismatch_audit.csv').open() as f:
                audits = list(csv.DictReader(f))
            for item in audits:
                self.assertEqual(item['observed_min'], item['expected_flips'])
                self.assertEqual(item['observed_max'], item['expected_flips'])
                self.assertEqual(int(item['checked_frames']), 3)
            source_bytes = (folder/'metrics.csv').read_bytes()
            replot(root/'out', root/'redrawn')
            target = root/'redrawn/N32_K16'
            self.assertEqual(source_bytes, (target/'metrics.csv').read_bytes())
            self.assertTrue((target/'construction.json').exists())
            self.assertTrue((target/'Fig2_Bob_L2_BER.png').exists())
            self.assertFalse((root/'redrawn/private').exists())
            # Same config, simulation key and nonce reproduce all frame errors.
            run(c, root/'data', root/'replay', expected_tx=1, expected_rx=1, smoke=True,
                key_file=manifest['key_material']['local_key_file'], key_nonce=manifest['key_material']['nonce'], design_profile_file=profile)
            replay = json.loads((root/'replay/RUN_CONFIG.json').read_text())
            self.assertEqual(replay['key_material']['fingerprint_sha256'], manifest['key_material']['fingerprint_sha256'])
            self.assertEqual(replay['key_material']['nonce'], manifest['key_material']['nonce'])
            with np.load(folder/'frame_errors.npz') as first, np.load(root/'replay/N32_K16/frame_errors.npz') as second:
                np.testing.assert_array_equal(first['frame_bit_errors'], second['frame_bit_errors'])
            self.assertFalse((root/'replay/N32_K16/Fig2_Bob_BER.png').exists())


if __name__ == '__main__':
    unittest.main()
