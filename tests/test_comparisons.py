"""Check trial aggregation without running expensive optimisations."""
import csv
from experiments import comparisons as module
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


class TrialAveragesTest(unittest.TestCase):
    def test_average_errors_and_times_for_each_size(self):
        for name, methods, run, fields in (
            ('fixed_vs_jump', ('FixedTimeAdjoint', 'JumpTimeGradient'),
             module.run_jump_comparison_experiments, module.JUMP_FIELDS),
            ('adjoint_vs_gradient', ('FixedTimeAdjoint', 'FixedTimeGradient'),
             module.run_adjoint_experiments, module.ADJOINT_FIELDS),
        ):
            with self.subTest(module=name), tempfile.TemporaryDirectory() as folder:
                clock = [0.0]

                def make_fit():
                    estimates = iter((1., 3., 2., 4.))

                    def fit(*args):
                        value = next(estimates)
                        clock[0] += value
                        return np.full((5, 2, 2), value)
                    return fit

                filename = Path(folder) / 'results.csv'
                with patch.object(module, 'perf_counter', side_effect=lambda: clock[0]), \
                     patch.object(getattr(module, methods[0]), 'optimise_A', side_effect=make_fit()), \
                     patch.object(getattr(module, methods[1]), 'optimise_A', side_effect=make_fit()):
                    result = run(
                        lambda t: np.zeros((2, 2)), 2, [0., 1.], 4, [3, 6],
                        results_file=filename, seed=7, num_trials=2,
                    )
                # Mean squared error is 5 and 10; squaring the mean would give 4 and 9.
                for field in fields[1:3]:
                    np.testing.assert_allclose(result.metrics[field], [5., 10.])
                for field in fields[3:5]:
                    np.testing.assert_allclose(result.metrics[field], [2., 3.])
                for paths in result.estimated_paths.values():
                    np.testing.assert_allclose(paths[0], 2.)
                    np.testing.assert_allclose(paths[1], 3.)
                with filename.open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 2)
                self.assertEqual([int(row['num_trials']) for row in rows], [2, 2])
                np.testing.assert_allclose([float(row[fields[1]]) for row in rows], [5., 10.])

    def test_reject_invalid_trial_counts(self):
        for run in (module.run_jump_comparison_experiments, module.run_adjoint_experiments):
            for trials in (0, -1, 1.5, True):
                with self.subTest(function=run.__name__, trials=trials), self.assertRaises(ValueError):
                    run(
                        lambda t: np.zeros((2, 2)), 2, [0., 1.], 4, [3], num_trials=trials,
                    )


if __name__ == '__main__':
    unittest.main()
