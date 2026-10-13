"""Transition counting and optimizer reuse for fixed and jump-time data."""
import unittest
from unittest.mock import patch

import numpy as np

from solvers.fixed_time_adjoint import FixedTimeAdjoint as Adjoint
from solvers.fixed_time_gradient import FixedTimeGradient as Direct
from solvers.jump_time_gradient import JumpTimeGradient as Jumps


class PrecomputedCountsTest(unittest.TestCase):
    fixed_data = np.array([[0, 0, 1], [0, 1, 0], [1, 1, 0]])
    jump_times = [[0, 2], [], [3]]
    jump_states = [[0, 1, 0], [1], [0, 1]]

    def test_fixed_counts_include_self_transitions_and_empty_rows(self):
        for cls in (Adjoint, Direct):
            with self.subTest(method=cls.__module__):
                emb = cls(3, num_intervals=4)
                counts = emb.transition_counts(self.fixed_data)
                np.testing.assert_array_equal(counts, [
                    [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
                    [[1, 1, 0], [0, 1, 0], [0, 0, 0]],
                ])

    def test_jump_counts_include_no_jump_individuals_and_boundary_jumps(self):
        counts = Jumps(2, num_intervals=4).transition_counts(
            self.jump_times, self.jump_states)
        np.testing.assert_array_equal(counts, [
            [[1, 1], [0, 1]], [[1, 0], [0, 2]],
            [[1, 0], [1, 1]], [[1, 1], [0, 1]],
        ])

    def test_each_fit_counts_once_despite_multiple_gradient_evaluations(self):
        for cls in (Adjoint, Direct, Jumps):
            with self.subTest(method=cls.__module__):
                emb = cls(2, num_intervals=4, mu=.03, kappa=1, max_iter=10)
                args = ((self.jump_times, self.jump_states) if cls is Jumps
                        else ([0., .5, 1.], self.fixed_data))
                # Spies retain real counting, gradient evaluation, and optimization.
                with patch.object(emb, 'transition_counts', wraps=emb.transition_counts) as count, \
                     patch.object(emb, 'objective_grad_from_counts',
                                  wraps=emb.objective_grad_from_counts) as evaluate:
                    fit = emb.optimise_A(*args)
                    self.assertEqual(count.call_count, 1)
                    self.assertGreater(evaluate.call_count, 1)
                    self.assertTrue(np.all(np.isfinite(fit)))
                    np.testing.assert_allclose(fit.sum(axis=2), 0., atol=1e-12)
                    # A subsequent fit must prepare its own counts, not reuse a cache.
                    emb.optimise_A(*args)
                    self.assertEqual(count.call_count, 2)

    def test_count_objectives_preserve_counts_and_have_correct_gradients(self):
        x = np.linspace(.3, 1.4, 10)
        for cls in (Adjoint, Direct, Jumps):
            with self.subTest(method=cls.__module__):
                emb = cls(2, num_intervals=4, mu=.03, kappa=1)
                if cls is Jumps:
                    counts = emb.transition_counts(self.jump_times, self.jump_states)
                    args = (counts,)
                else:
                    counts = emb.transition_counts(self.fixed_data)
                    args = ([0, 2, 4], counts)
                original_counts = counts.copy()
                _, gradient = emb.objective_grad_from_counts(x, *args)
                numerical = np.zeros_like(x)
                for i in range(len(x)):
                    delta = np.zeros_like(x)
                    delta[i] = 1e-5
                    numerical[i] = (
                        emb.objective_grad_from_counts(x + delta, *args)[0]
                        - emb.objective_grad_from_counts(x - delta, *args)[0]
                    ) / (2e-5)
                np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-8)
                np.testing.assert_array_equal(counts, original_counts)


if __name__ == '__main__':
    unittest.main()
