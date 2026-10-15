import unittest
import numpy as np



class BirthDeathJumpTests(unittest.TestCase):
    def solver(self, **options):
        from solvers import BirthDeathJumpTimeGradient
        return BirthDeathJumpTimeGradient(3, num_intervals=4, **options)

    def test_counts_include_stays_and_jumps_at_both_boundary_intervals(self):
        solver = self.solver(absorbing_final=False)
        counts = solver.transition_counts([[0, 3], [], [2]], [[0, 1, 2], [2], [1, 0]])
        np.testing.assert_array_equal(counts, [
            [[0, 1, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 2, 0], [0, 0, 1]],
            [[0, 0, 0], [1, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 0, 1]],
        ])

    def test_likelihood_includes_no_jump_probability_and_only_neighbour_rates(self):
        solver = self.solver(absorbing_final=False, mu0=0., mu1=0.)
        # One chain: 0 -> 1, stay at 1, 1 -> 2, stay at 2.
        x = np.full((5, 4), .8)
        objective, _ = solver.objective_grad(x.ravel(), [[0, 2]], [[0, 1, 2]])
        self.assertEqual(solver.num_rates, 4)
        self.assertAlmostEqual(objective, -np.log(.2 * .6 * .2 * .8), places=9)

    def test_gradient_with_both_penalties_and_absorbing_constraint(self):
        for absorbing in (False, True):
            solver = self.solver(absorbing_final=absorbing, mu0=.3, mu1=.7)
            x = np.linspace(.2, .8, 20)
            inputs = ([[0, 2], []], [[0, 1, 2], [2]])
            _, analytic = solver.objective_grad(x, *inputs)
            numerical = np.zeros_like(x)
            for i in range(x.size):
                step = np.zeros_like(x)
                step[i] = 1e-6
                numerical[i] = (solver.objective_grad(x + step, *inputs)[0]
                                - solver.objective_grad(x - step, *inputs)[0]) / 2e-6
            np.testing.assert_allclose(analytic, numerical, atol=1e-8)
            if absorbing:
                np.testing.assert_array_equal(analytic.reshape(5, 4)[:, -1], 0.)

    def test_fit_preserves_birth_death_structure_and_returns_status(self):
        solver = self.solver(absorbing_final=False, max_iter=10)
        fitted, result = solver.optimise_A([[0, 2], [3]], [[0, 1, 2], [2, 1]],
                                           return_result=True)
        self.assertEqual(fitted.shape, (5, 3, 3))
        np.testing.assert_array_equal(fitted[:, 0, 2], 0.)
        np.testing.assert_array_equal(fitted[:, 2, 0], 0.)
        np.testing.assert_allclose(fitted.sum(axis=2), 0., atol=1e-12)
        self.assertTrue(np.isfinite(result.fun))
        self.assertTrue(np.all(np.eye(3) + fitted / 4 >= 0))

    def test_dense_fixed_observations_match_jump_objective_and_gradient(self):
        from solvers import BirthDeathEmbedder
        jump = self.solver(absorbing_final=False, mu0=.3, mu1=.7)
        fixed = BirthDeathEmbedder(3, num_intervals=4, absorbing_final=False,
                                   mu0=.3, mu1=.7)
        histories = ([[0, 2], [3]], [[0, 1, 2], [2, 1]])
        # Snapshots of those two jump histories at all five grid points.
        panel = np.array([[0, 2], [1, 2], [1, 2], [2, 2], [2, 1]])
        x = np.linspace(.2, .8, 20)
        ju, jg = jump.objective_grad(x, *histories)
        fu, fg = fixed.objective_grad(x, list(range(5)), panel)
        self.assertAlmostEqual(ju, fu)
        np.testing.assert_allclose(jg, fg, atol=1e-12)

    def test_rejects_invalid_or_non_birth_death_histories(self):
        solver = self.solver(absorbing_final=False)
        for times, states in (([[0]], [[0, 2]]), ([[1, 1]], [[0, 1, 0]]),
                              ([[4]], [[0, 1]]), ([[.5]], [[0, 1]]),
                              ([[0]], [[0]]), ([[0]], []),
                              ([[]], [[3]]), ([[0]], [[0., .5]])):
            with self.subTest(times=times, states=states), self.assertRaises(ValueError):
                solver.transition_counts(times, states)
        with self.assertRaises(ValueError):
            self.solver().transition_counts([[0]], [[2, 1]])
