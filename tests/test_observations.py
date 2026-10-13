"""The canonical simulator and generator tuning work without legacy samplers."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments import generators
from experiments.observations import Observation
from experiments.comparisons import generate_comparison


class Observations(unittest.TestCase):
    def test_seeded_panel_respects_explicit_initial_states(self):
        generator = lambda t: np.zeros((2, 2))
        sampler = Observation(generator, 2, [0., .5, 1.], num_intervals=4, seed=7)
        panel = sampler.simulate_fixed_observations(3, initial_states=np.array([1, 0, 1]))
        np.testing.assert_array_equal(panel, [[1, 0, 1], [1, 0, 1], [1, 0, 1]])
        jumps, states = sampler.simulate_individual_jump_times_observations(
            3, initial_states=np.array([1, 0, 1]))
        self.assertEqual(jumps, [[], [], []])
        self.assertEqual(states, [[1], [0], [1]])

    def test_tuning_constant_exit_rate_matches_requested_jumps(self):
        tune = getattr(generators, 'tune_generator_for_target_jumps', None)
        self.assertTrue(callable(tune))
        factory = lambda rate: lambda t: np.array([[-rate, rate], [rate, -rate]])
        generator = tune(factory, rank=2, target_num_jumps=1.5, num_intervals=20)
        np.testing.assert_allclose(generator(.5), [[-1.5, 1.5], [1.5, -1.5]], atol=1e-10)

    def test_four_panel_comparison_repeats_with_explicit_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            options = dict(num_individuals_range=[10], num_observations=(3, 4),
                           constant=4, num_intervals=20, max_iter=2, seed=7)
            first = generate_comparison(Path(folder) / 'first.npz', **options)
            second = generate_comparison(Path(folder) / 'second.npz', **options)
            for key in ('fixed_estimates', 'jump_estimates', 'fixed_true', 'jump_true'):
                np.testing.assert_array_equal(first[key], second[key])


if __name__ == '__main__':
    unittest.main()
