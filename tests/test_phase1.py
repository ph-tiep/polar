"""Regression checks for empty ray tracing and valid-path validation."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import sionna.rt  # Native initialization must precede TensorFlow.
import numpy as np
from phase1_digital_twin.raytrace_csi import build_frequency_grid, extract_csi
from phase1_digital_twin.validation_metrics import power_delay_profile, delay_statistics, los_ratio


class Phase1Tests(unittest.TestCase):
    def test_baseband_frequency_grid(self):
        f = build_frequency_grid()
        self.assertEqual(f[256], 0)
        self.assertEqual(f[0], -1e9)
        np.testing.assert_array_equal(np.diff(f), 3906250)

    def test_padding_excluded_and_first_arrival_not_first_slot(self):
        a = np.array([[[[[0, 1, 2]]]]], dtype=np.complex64)
        tau = np.array([[[-1, 20e-9, 10e-9]]])
        delays, power = power_delay_profile(a, tau)
        np.testing.assert_allclose(delays, [20, 10])
        mean, rms = delay_statistics(a, tau)
        self.assertAlmostEqual(mean, 12, places=5)
        self.assertAlmostEqual(rms, 4, places=5)
        self.assertAlmostEqual(los_ratio(a, tau=tau), .8, places=5)

    def test_empty_pdp_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, 'No valid'):
            power_delay_profile(np.zeros((1,1,1,1,0)), np.zeros((1,1,0)))

    def test_empty_trace_stops_before_export_or_cfr(self):
        paths = SimpleNamespace(cir=lambda **kw: (np.zeros((2,16,1,256,0,1)), np.zeros((2,1,0))))
        scene = SimpleNamespace(receivers={'Bob': None, 'Eve': None}, transmitters={'BS': None})
        with patch('phase1_digital_twin.raytrace_csi.PathSolver') as solver:
            solver.return_value.return_value = paths
            with self.assertRaisesRegex(ValueError, 'no valid paths'):
                extract_csi(scene, verbose=False)


if __name__ == '__main__':
    unittest.main()
