"""Exercise the public imports and module launch commands after reorganisation."""
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Reorganisation(unittest.TestCase):
    def test_paper_modules_can_be_launched_directly(self):
        self.assertIsNotNone(importlib.util.find_spec('paper'))
        for name in ('comparison', 'fixed_vs_jump',
                     'adjoint_vs_gradient', 'ehrenfest',
                     'financial_birth_death'):
            with self.subTest(application=name):
                result = subprocess.run(
                    [sys.executable, '-m', 'paper.' + name, '--help'], cwd=ROOT,
                    capture_output=True, text=True, timeout=30,
                    env={**os.environ, 'MPLBACKEND': 'Agg'},
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_ehrenfest_can_run_without_paper_or_plotting(self):
        from experiments import ehrenfest
        self.assertTrue(callable(getattr(ehrenfest, 'generate_data', None)))
        with tempfile.TemporaryDirectory() as folder:
            prefix = Path(folder) / 'nested' / 'ehrenfest'
            rows = ehrenfest.generate_data(
                num_particles=1, jump_frequency=1., individuals=(20,),
                num_observations=3, num_intervals=10, num_trials=2,
                max_iter=2, output_prefix=prefix,
            )
            self.assertTrue(np.isfinite(rows[0]['mise']))
            trials = np.genfromtxt(f'{prefix}_trials.csv', delimiter=',', names=True)
            self.assertAlmostEqual(rows[0]['mise'], np.mean(trials['ise']))
            self.assertTrue(Path(f'{prefix}_generators.npz').exists())

    def test_saved_transition_matrix_is_loaded_through_observations(self):
        from experiments import observations
        self.assertTrue(callable(getattr(observations, 'load_transition_matrix', None)))
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / 'transition.csv'
            filename.write_text('0.7,0.3\n0,1\n')
            np.testing.assert_allclose(observations.load_transition_matrix(filename),
                                       [[.7, .3], [0., 1.]])
            filename.write_text('0.7,0.4\n0,1\n')
            with self.assertRaises(ValueError):
                observations.load_transition_matrix(filename)


if __name__ == '__main__':
    unittest.main()
