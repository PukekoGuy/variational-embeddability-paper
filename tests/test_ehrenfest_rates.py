import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


class EhrenfestRatesTests(unittest.TestCase):
    def test_panels_use_saved_times_and_all_eight_directed_rates(self):
        from plotting.ehrenfest_rates import plot_ehrenfest_rates
        times = np.array([0., .25, 1.])
        true = np.zeros((3, 5, 5))
        estimate = np.zeros_like(true)
        for i in range(4):
            true[:, i, i + 1] = [i + 1, i + 2, i + 3]
            true[:, i + 1, i] = [i + 11, i + 12, i + 13]
        estimate[:] = true + .5
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(directory) / 'generators.npz'
            np.savez(saved, times=times, true_generator=true,
                     mean_estimated_generators=[estimate], individuals=[1000])
            fig, axes = plot_ehrenfest_rates(saved, show=False)
            self.addCleanup(plt.close, fig)
            self.assertEqual(axes.shape, (4, 2))
            for row in range(4):
                for col in range(2):
                    estimated_line, true_line = axes[row, col].lines
                    expected = np.array([row + 1, row + 2, row + 3]) + 10 * col
                    np.testing.assert_allclose(true_line.get_ydata(), expected)
                    np.testing.assert_allclose(estimated_line.get_ydata(), expected + .5)
                    np.testing.assert_allclose(true_line.get_xdata(), times)
                    np.testing.assert_allclose(estimated_line.get_xdata(), times)

    def test_new_example_fits_one_sample_and_replots_saved_data(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'rates'
            command = [sys.executable, str(ROOT / 'run_example.py'), 'ehrenfest_rates',
                       '--output-prefix', str(prefix), '--no-show']
            fitted = subprocess.run(command + ['--individuals', '20', '--num-observations', '3',
                                    '--num-intervals', '4', '--max-iter', '2'],
                                    capture_output=True, text=True, timeout=30,
                                    env={**os.environ, 'MPLBACKEND': 'Agg'})
            self.assertEqual(fitted.returncode, 0, fitted.stderr)
            with np.load(f'{prefix}_generators.npz') as saved:
                self.assertEqual(saved['mean_estimated_generators'].shape, (1, 5, 5, 5))
                np.testing.assert_array_equal(saved['individuals'], [20])
            self.assertTrue(Path(f'{prefix}_rates.png').exists())
            before = Path(f'{prefix}_generators.npz').read_bytes()
            replotted = subprocess.run(command + ['--plot-only'], capture_output=True,
                                       text=True, timeout=30,
                                       env={**os.environ, 'MPLBACKEND': 'Agg'})
            self.assertEqual(replotted.returncode, 0, replotted.stderr)
            self.assertEqual(Path(f'{prefix}_generators.npz').read_bytes(), before)
