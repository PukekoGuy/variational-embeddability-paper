"""Experiment regularisation settings reach the fitted objective."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import comparisons, ehrenfest
from solvers import (FixedTimeAdjoint, FixedTimeGradient, JumpTimeGradient,
                     BirthDeathEmbedder, StrongBirthDeathEmbedder, BirthDeathJumpTimeGradient)


class ExperimentWeightsTests(unittest.TestCase):
    def test_comparison_entry_points_apply_both_weights(self):
        # At a constant all-one two-state path, only magnitude contributes:
        # ||A||_F^2 = 4, so R = 2*mu0 = .6. This checks the real objectives
        # reached by experiment wrappers, not just constructor arguments.
        seen = []

        def inspect_fit(solver, *args):
            theta = np.ones((solver.num_intervals + 1, solver.num_rates))
            if isinstance(solver, JumpTimeGradient):
                objective = solver.objective_grad(theta.ravel(), [], [])[0]
            else:
                objective = solver.objective_grad(theta.ravel(), [0, solver.num_intervals],
                                                   np.empty((2, 0), dtype=int))[0]
            self.assertAlmostEqual(objective, .6)
            self.assertEqual(solver.mu1, .7)
            seen.append(type(solver))
            return np.zeros((solver.num_intervals + 1, 2, 2))

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(FixedTimeAdjoint, 'optimise_A', autospec=True, side_effect=inspect_fit), \
             patch.object(FixedTimeGradient, 'optimise_A', autospec=True, side_effect=inspect_fit), \
             patch.object(JumpTimeGradient, 'optimise_A', autospec=True, side_effect=inspect_fit):
            for generate in (comparisons.generate_adjoint_comparison,
                             comparisons.generate_jump_comparison):
                generate(results_file=Path(directory)/'comparison.csv', target_jumps=2,
                         num_intervals=20, num_individuals_list=[3], num_trials=1,
                         seed=4, mu0=.3, mu1=.7)
            comparisons.generate_comparison(results_file=None, num_individuals_range=[3],
                                            num_observations=(2, 3), num_intervals=20,
                                            seed=4, mu0=.3, mu1=.7)
        self.assertIn(FixedTimeAdjoint, seen)
        self.assertIn(FixedTimeGradient, seen)
        self.assertIn(JumpTimeGradient, seen)

    def test_ehrenfest_generator_exposes_independent_weights(self):
        original = BirthDeathEmbedder.optimise_A

        def inspect_fit(solver, *args, **kwargs):
            self.assertEqual((solver.mu0, solver.mu1), (.3, .7))
            return original(solver, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(BirthDeathEmbedder, 'optimise_A', autospec=True, side_effect=inspect_fit):
            rows = ehrenfest.generate_data(num_particles=1, individuals=[3],
                                           num_observations=3, num_intervals=6,
                                           num_trials=1, max_iter=2, mu0=.3, mu1=.7,
                                           output_prefix=Path(directory)/'ehrenfest')
            self.assertTrue(np.isfinite(rows[0]['fixed_mise']))
