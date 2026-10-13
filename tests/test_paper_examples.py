import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

class PaperExamples(unittest.TestCase):
    def test_launcher_exists(self):
        self.assertTrue((ROOT / 'run_example.py').exists())

    def test_all_help_commands(self):
        for name in ('comparison', 'fixed_vs_jump', 'adjoint_vs_gradient',
                     'ehrenfest', 'financial_birth_death'):
            result = subprocess.run([sys.executable, str(ROOT/'run_example.py'), name, '--help'],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)

    def test_comparison_saved_data_plot_does_not_fit(self):
        from experiments.comparisons import generate_comparison
        from plotting.comparison_panels import plot_comparison_results
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'comparison.npz'
            generate_comparison(filename, [10], num_observations=(3, 4), constant=4,
                                  num_intervals=20, max_iter=2, seed=8)
            from solvers import FixedTimeAdjoint, JumpTimeGradient
            with patch.object(FixedTimeAdjoint, 'optimise_A', side_effect=AssertionError('fit called')), \
                 patch.object(JumpTimeGradient, 'optimise_A', side_effect=AssertionError('fit called')):
                fig, axes = plot_comparison_results(filename, show=False)
            self.assertEqual(axes.shape, (2, 2))
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_financial_bundled_input_from_other_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'financial'
            result = subprocess.run([sys.executable, str(ROOT/'run_example.py'),
                'financial_birth_death', '--max-iter', '2', '--output-prefix', str(prefix)],
                cwd=directory, capture_output=True, text=True, timeout=30,
                env={**os.environ, 'MPLBACKEND': 'Agg'})
            self.assertEqual(result.returncode, 0, result.stderr)
            generator = np.loadtxt(str(prefix)+'_generator.csv', delimiter=',')
            self.assertEqual(generator.shape, (8, 8))
            self.assertTrue(np.all(np.isfinite(generator)))
            np.testing.assert_allclose(generator[-1], 0)
            self.assertTrue(Path(str(prefix)+'_rates.csv').exists())
            self.assertTrue(Path(str(prefix)+'_matrices_latex.txt').exists())
