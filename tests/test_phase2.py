"""Technical regression checks; fixtures are not research channel results."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from fec.polar.crypto_polar import CryptoPolarEncoder, ChannelBoundKeyGenerator
from fec.polar.decoding import PolarSCLDecoder
from fec.polar.utils import generate_5g_ranking
from phase2_crypto_polar.experiments import make_decoder, summarize_errors, carrier_indices, validate_config, run, paired_summary
from phase2_crypto_polar.digital_twin_channel import DigitalTwinTHzChannel
from phase2_crypto_polar.security_metrics import mismatch_key, inverse_polar_payload

ROOT = Path(__file__).resolve().parents[1]


class Phase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(47)

    def test_roundtrip_and_baseline_no_retry(self):
        frozen, _ = generate_5g_ranking(16, 32)
        payload = torch.randint(0, 2, (8, 10)).float()
        keys = torch.randint(0, 2, (8, 32)).float()
        for size in (2, 4):
            for crypto in (False, True):
                enc = CryptoPolarEncoder(frozen, 32, 'CRC6', enable_info_masking=crypto, enable_frozen_crypto=crypto)
                dec = make_decoder(frozen, 32, 'CRC6', size, crypto)
                out = dec(20*(2*enc(payload, keys)-1), keys)
                torch.testing.assert_close(out, payload)
                self.assertEqual(dec.last_stats['expanded_frames'], 0)
                self.assertEqual(dec.last_stats['flip_frames'], 0)

    def test_standard_matches_existing_crc_scl(self):
        frozen, _ = generate_5g_ranking(16, 32)
        enc = CryptoPolarEncoder(frozen, 32, 'CRC6', enable_info_masking=False, enable_frozen_crypto=False)
        ours = make_decoder(frozen, 32, 'CRC6', 4, False)
        reference = PolarSCLDecoder(frozen, 32, list_size=4, crc_degree='CRC6')
        payload = torch.randint(0, 2, (12, 10)).float()
        logits = 2*(2*enc(payload, None)-1)+torch.randn(12, 32)*2
        torch.testing.assert_close(ours(logits, None), reference(logits)[:, :10])

    def test_expanded_cases_noiseless_payload_crc_roundtrip(self):
        c = json.loads((ROOT/'configs/phase2_expanded.json').read_text())
        for case in c['cases']:
            with self.subTest(**case):
                n, k = case['n'], case['k']
                frozen, _ = generate_5g_ranking(k, n)
                enc = CryptoPolarEncoder(frozen, n, 'CRC11')
                dec = make_decoder(frozen, n, 'CRC11', 2)
                payload = torch.randint(0, 2, (1, k-11)).float()
                key = torch.randint(0, 2, (1, n)).float()
                torch.testing.assert_close(dec(20*(2*enc(payload, key)-1), key), payload)

    def test_retry_telemetry_and_disabled_variant(self):
        frozen, _ = generate_5g_ranking(16, 32)
        m = dict(damping=1., retry_damping=1., expand_list=True, max_flip_trials=2)
        dec = make_decoder(frozen, 32, 'CRC6', 2, modified=m)
        dec(torch.randn(10, 32), torch.randint(0, 2, (10, 32)).float())
        self.assertEqual(dec.last_stats['base_frames'], 10)
        self.assertGreater(dec.last_stats['expanded_frames'], 0)
        self.assertEqual(dec.last_stats['max_list_size'], 4)
        self.assertLessEqual(dec.last_stats['flip_frames'], 20)

    def test_exact_wrong_key_counts_and_frozen_scope(self):
        key = torch.zeros(3, 256)
        for scope, width in [('frozen', 128), ('full', 256)]:
            for p in (0., .01, .25, .5, 1.):
                wrong, count, actual_width = mismatch_key(key, 128, p, scope, torch.Generator().manual_seed(1))
                self.assertEqual(actual_width, width)
                self.assertTrue(torch.all(wrong.sum(dim=1) == count))
                if scope == 'frozen':
                    self.assertEqual(int(wrong[:, :128].sum()), 0)

    def test_fig5_full_key_protocol_and_complement_control(self):
        for n, k in [(128, 64), (256, 64)]:
            with self.subTest(n=n, k=k):
                frozen, _ = generate_5g_ranking(k, n)
                encoder = CryptoPolarEncoder(frozen, n, 'CRC11')
                decoder = make_decoder(frozen, n, 'CRC11', 8)
                generator = torch.Generator().manual_seed(n+37)
                payload = torch.randint(0, 2, (2, k-11), generator=generator).float()
                alice_key = torch.randint(0, 2, (2, n), generator=generator).float()
                bob_key = alice_key.clone()
                torch.testing.assert_close(alice_key, bob_key)
                for p in (.01, .25, .5, 1.):
                    eve_key, count, width = mismatch_key(alice_key, k, p, 'full', generator)
                    self.assertEqual(width, n)
                    self.assertTrue(torch.all((eve_key != alice_key).sum(-1) == count))
                    if p == .01:
                        self.assertEqual(count, 1 if n == 128 else 3)
                    if p == 1.:
                        torch.testing.assert_close(eve_key, 1-alice_key)
                cw = encoder(payload, alice_key)
                complement_cw = encoder(payload, 1-alice_key)
                self.assertTrue(torch.all((cw != complement_cw).sum(-1) == 1))
                torch.testing.assert_close(decoder(20*(2*cw-1), 1-alice_key), payload)

    def test_frozen_only_inverse_control(self):
        frozen, info = generate_5g_ranking(16, 32)
        enc = CryptoPolarEncoder(frozen, 32, 'CRC6', enable_info_masking=False)
        payload = torch.randint(0, 2, (5, 10)).float()
        cw = enc(payload, torch.randint(0, 2, (5, 32)).float())
        torch.testing.assert_close(inverse_polar_payload(cw, info, 10), payload)

    def test_zero_errors_not_fabricated_floor(self):
        stats = summarize_errors([0]*10, 117)
        self.assertEqual(stats['ber'], 0)
        self.assertEqual(stats['bler'], 0)
        self.assertAlmostEqual(stats['zero_error_upper95'], 1-.05**.1)
        self.assertGreater(stats['bler_high95'], 0)

    def test_carrier_mapping_uses_full_band(self):
        idx = carrier_indices(3, 70, 32, torch.Generator().manual_seed(2))
        for row in idx:
            self.assertEqual(len(set(row[:32].tolist())), 32)
            self.assertEqual(len(set(row[32:64].tolist())), 32)
        self.assertTrue((idx < 32).all())

    def test_channel_exact_gain_and_llr_and_invalid_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            h = np.ones((1, 1, 32), dtype=np.complex64)
            np.save(p/'H_Bob.npy', h)
            np.save(p/'H_Eve.npy', 2*h)
            ch = DigitalTwinTHzChannel(tmp)
            np.testing.assert_allclose(ch.g_bob_norm, 1)
            np.testing.assert_allclose(ch.g_eve_norm, 4)
            bits = torch.tensor([[0., 1.]])
            logits, gain = ch.transmit_bpsk(bits, 0, .5, indices=torch.tensor([[0, 1]]), noise=torch.zeros_like(bits))
            torch.testing.assert_close(logits, torch.tensor([[-2., 2.]]))
            logits_e, _ = ch.transmit_bpsk(bits, 0, .5, node='eve', indices=torch.tensor([[0, 1]]), noise=torch.zeros_like(bits))
            torch.testing.assert_close(logits_e, 4*logits)
            np.save(p/'H_Eve.npy', h*0)
            with self.assertRaisesRegex(ValueError, 'nonzero'):
                DigitalTwinTHzChannel(tmp)

    def test_config_requires_matching_expanded_baseline(self):
        c = json.loads((ROOT/'configs/phase2_review.json').read_text())
        validate_config(c)
        c['list_sizes'] = [4, 8]
        with self.assertRaisesRegex(ValueError, 'expanded-list'):
            validate_config(c)

    def test_paired_metrics_preserve_discordant_frames(self):
        result = paired_summary([0, 2, 1, 0], [1, 0, 2, 0], 10, 1., 'reference', 'candidate')
        self.assertEqual(result['reference_only_block_errors'], 1)
        self.assertEqual(result['candidate_only_block_errors'], 1)
        self.assertEqual(result['both_block_errors'], 1)
        self.assertEqual(result['both_correct'], 1)
        self.assertAlmostEqual(result['ber_delta_candidate_minus_reference'], 0.)
        self.assertEqual(result['bler_delta_candidate_minus_reference'], 0.)

    def test_list_only_ablation_has_no_flip_retries(self):
        frozen, _ = generate_5g_ranking(16, 32)
        settings = dict(damping=1., retry_damping=1., expand_list=True, max_flip_trials=0)
        decoder = make_decoder(frozen, 32, 'CRC6', 2, modified=settings)
        generator = torch.Generator().manual_seed(72)
        logits = torch.randn(10, 32, generator=generator)
        keys = torch.randint(0, 2, (10, 32), generator=generator).float()
        decoder(logits, keys)
        self.assertGreater(decoder.last_stats['expanded_frames'], 0)
        self.assertEqual(decoder.last_stats['flip_frames'], 0)
        self.assertEqual(decoder.last_stats['max_list_size'], 4)

    def test_plot_zeros_are_not_replaced_with_bounds(self):
        from phase2_crypto_polar.plotting_q1 import line, validate_plot_data
        import matplotlib.pyplot as plt
        rows = [dict(ebno_db=0, ber=.1, frames=100), dict(ebno_db=1, ber=0., frames=100)]
        fig, ax = plt.subplots()
        line(ax, rows, 'ber', 'test')
        values = ax.lines[0].get_ydata()
        self.assertEqual(values[0], .1)
        self.assertTrue(np.isnan(values[1]))
        self.assertEqual(len(ax.collections), 0)
        plt.close(fig)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            validate_plot_data([dict(ebno_db=0)])

    def test_pipeline_fixture_partial_batch_and_outputs(self):
        c = json.loads((ROOT/'configs/phase2_review.json').read_text())
        c.update(frames_per_snr=3, batch_size=2, cases=[dict(n=32, k=16)], crc='CRC6',
                 ebno_db=[0.], list_sizes=[2, 4], eve_list_size=2, diagnostic_samples=2)
        c['modified']['base_list'] = 2
        c['modified']['max_flip_trials'] = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'; data.mkdir()
            h = np.ones((1, 1, 32), dtype=np.complex64)
            np.save(data/'H_Bob.npy', h)
            np.save(data/'H_Eve.npy', .8*h)
            np.savez(data/'Metadata.npz', NumTxAntennas=1, NumRxAntennas=1, NumSubcarriers=32)
            run(c, data, root/'out', expected_tx=1, expected_rx=1, smoke=True)
            manifest = json.loads((root/'out/RUN_CONFIG.json').read_text())
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['purpose'], 'technical-smoke')
            import csv
            with (root/'out/N32_K16/metrics.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(all(int(r['frames']) == 3 for r in rows))
            self.assertTrue(all(int(r['bits']) == 30 for r in rows))
            self.assertTrue({'adaptive_list_only','damping_only','flip_only'} <= {r['curve'] for r in rows})
            with np.load(root/'out/N32_K16/frame_errors.npz', allow_pickle=False) as frames:
                self.assertEqual(frames['frame_bit_errors'].shape, (1, len(rows), 3))
                for index, row in enumerate(rows):
                    counts = frames['frame_bit_errors'][0, index]
                    self.assertEqual(int(counts.sum()), int(row['bit_errors']))
                    self.assertEqual(int(np.count_nonzero(counts)), int(row['block_errors']))
            diag = json.loads((root/'out/N32_K16/diagnostics.json').read_text())
            self.assertEqual(diag['frozen_only_noiseless_attack_bit_errors'], 0)
            self.assertFalse((root/'out/N32_K16/Fig2_Bob_BER.png').exists())
            self.assertTrue((root/'out/SMOKE_ONLY.txt').exists())
            from replot_phase2 import replot
            with self.assertRaisesRegex(ValueError, 'smoke'):
                replot(root/'out', root/'replot-smoke')
            self.assertFalse((root/'replot-smoke').exists())
            # Separate tiny two-SNR fixture checks the real plotting/replot path.
            c['ebno_db'] = [0., 1.]
            run(c, data, root/'experiment', expected_tx=1, expected_rx=1)
            original = (root/'experiment/N32_K16/metrics.csv').read_bytes()
            replot(root/'experiment', root/'replotted')
            self.assertEqual(original, (root/'experiment/N32_K16/metrics.csv').read_bytes())
            self.assertEqual(original, (root/'replotted/N32_K16/metrics.csv').read_bytes())
            self.assertTrue((root/'replotted/N32_K16/Fig2_Bob_BER.png').exists())


if __name__ == '__main__':
    unittest.main()
