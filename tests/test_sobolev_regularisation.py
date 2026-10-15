"""Full-matrix Sobolev penalties, Riemann weights, and their rate gradients."""
import unittest
import numpy as np
from solvers import FixedTimeAdjoint, FixedTimeGradient, JumpTimeGradient, BirthDeathEmbedder


class SobolevRegularisationTests(unittest.TestCase):
    solvers = (FixedTimeAdjoint, FixedTimeGradient, JumpTimeGradient, BirthDeathEmbedder)

    def evaluate(self, solver, theta):
        x = theta.ravel()
        if isinstance(solver, JumpTimeGradient):
            return solver.objective_grad(x, [], [])
        return solver.objective_grad(x, [0, solver.num_intervals],
                                     np.empty((2, 0), dtype=int))

    def make_solver(self, cls, **weights):
        options = {'absorbing_final': False} if cls is BirthDeathEmbedder else {}
        return cls(3, num_intervals=2, **options, **weights)

    def test_constant_paths_have_full_matrix_magnitude_penalty(self):
        # Constant all-one rates: general generator norm squared = 18;
        # birth/death generator norm squared = 10, including diagonals.
        for cls in self.solvers:
            with self.subTest(solver=cls.__name__):
                solver = self.make_solver(cls, mu0=.2, mu1=.2)
                theta = np.ones((3, solver.num_rates))
                objective, _ = self.evaluate(solver, theta)
                self.assertAlmostEqual(objective, 1. if cls is BirthDeathEmbedder else 1.8)

    def test_independent_weights_use_left_riemann_sum_and_full_matrix_differences(self):
        for cls in self.solvers:
            for mu0, mu1 in ((.3, 0.), (0., .7), (.3, .7)):
                with self.subTest(solver=cls.__name__, mu0=mu0, mu1=mu1):
                    solver = self.make_solver(cls, mu0=mu0, mu1=mu1)
                    theta = np.zeros((3, solver.num_rates))
                    theta[:, 0] = [1., 2., 4.]
                    # A has just (-r, r) in row zero, so ||A||^2=2*r^2.
                    # dt=1/2; left sum uses r=1,2, excludes final r=4.
                    objective, gradient = self.evaluate(solver, theta)
                    self.assertAlmostEqual(objective, 2.5 * mu0 + 10. * mu1)
                    expected = np.array([mu0 - 4*mu1, 2*mu0 - 4*mu1, 8*mu1])
                    np.testing.assert_allclose(gradient.reshape(theta.shape)[:, 0], expected)

    def test_analytic_gradients_include_diagonal_coupling_and_fixed_boundary(self):
        for cls in self.solvers:
            modes = (False, True) if cls is BirthDeathEmbedder else (False,)
            for absorbing in modes:
                with self.subTest(solver=cls.__name__, absorbing=absorbing):
                    solver = self.make_solver(cls, mu0=.3, mu1=.7)
                    if cls is BirthDeathEmbedder:
                        solver.absorbing_final = absorbing
                    theta = np.linspace(.1, .8, 3 * solver.num_rates).reshape(3, -1)
                    _, analytic = self.evaluate(solver, theta)
                    numerical = np.zeros(theta.size)
                    h = 1e-6
                    for i in range(theta.size):
                        step = np.zeros(theta.size)
                        step[i] = h
                        plus = self.evaluate(solver, theta + step.reshape(theta.shape))[0]
                        minus = self.evaluate(solver, theta - step.reshape(theta.shape))[0]
                        numerical[i] = (plus - minus) / (2*h)
                    np.testing.assert_allclose(analytic, numerical, atol=1e-8)

    def test_weights_must_be_finite_and_nonnegative(self):
        for cls in self.solvers:
            for name in ('mu0', 'mu1'):
                for bad in (-1., float('nan'), float('inf')):
                    with self.subTest(solver=cls.__name__, weight=name, value=bad):
                        with self.assertRaises(ValueError):
                            self.make_solver(cls, **{name: bad})
