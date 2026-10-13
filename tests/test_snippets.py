"""Each learning example must run without fitting a large experiment."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class Snippets(unittest.TestCase):
    def test_minimal_examples_run(self):
        names = ('fixed_time_adjoint', 'fixed_time_gradient', 'jump_time_gradient',
                 'weak_birth_death', 'strong_birth_death', 'plot_generator_path',
                 'plot_level_rates', 'plot_mise')
        with tempfile.TemporaryDirectory() as folder:
            for name in names:
                with self.subTest(name=name):
                    result = subprocess.run([sys.executable, '-m', 'snippets.' + name],
                        cwd=ROOT, capture_output=True, text=True, timeout=30,
                        env={**os.environ, 'MPLBACKEND':'Agg', 'MPLCONFIGDIR':folder})
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('finite=True', result.stdout)

    def test_level_plot_returns_figure_without_display(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from plotting import plot_level_rates
        a = np.zeros((4, 3, 3))
        fig, axes = plot_level_rates([a], a, show=False)
        self.assertEqual(axes.shape, (2, 2))
        self.assertEqual(len(axes[0, 0].lines), 2)
        plt.close(fig)

    def test_mise_plot_can_be_saved_without_display(self):
        import matplotlib.pyplot as plt
        from plotting import plot_mise
        fig, ax = plot_mise([10, 100], [0.1, 0.01], show=False)
        self.assertEqual(len(ax.lines), 1)
        self.assertAlmostEqual(ax.lines[0].get_ydata()[1], 0.01)
        plt.close(fig)
