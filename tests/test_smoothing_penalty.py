"""The adjoint rate formula must match smoothing every generator entry."""
import unittest

import numpy as np

from solvers.fixed_time_adjoint import FixedTimeAdjoint as Adjoint
from solvers.fixed_time_gradient import FixedTimeGradient as Direct
from solvers.jump_time_gradient import JumpTimeGradient as Jumps


class SmoothingPenaltyTest(unittest.TestCase):
    def test_adjoint_matches_full_matrix_penalty_and_gradient(self):
        rng = np.random.default_rng(47)
        intervals = 4
        for rank in (2, 3, 5):
            for mu in (0., .07):
                with self.subTest(rank=rank, mu0=mu, mu1=mu):
                    x = rng.uniform(.2, .7, (intervals + 1) * rank * (rank - 1))
                    adjoint = Adjoint(rank, intervals, mu0=mu, mu1=mu)
                    # Zero counts isolate smoothing from the likelihood.
                    counts = np.zeros((1, rank, rank))
                    actual_u, actual_g = adjoint.objective_grad_from_counts(
                        x, [0, intervals], counts)
                    direct_u, direct_g = Direct(rank, intervals, mu0=mu, mu1=mu).objective_grad_from_counts(
                        x, [0, intervals], counts)
                    jump_u, jump_g = Jumps(rank, intervals, mu0=mu, mu1=mu).objective_grad_from_counts(
                        x, np.zeros((intervals, rank, rank)))
                    for expected_u, expected_g in ((direct_u, direct_g), (jump_u, jump_g)):
                        np.testing.assert_allclose(actual_u, expected_u, rtol=1e-12, atol=1e-12)
                        np.testing.assert_allclose(actual_g, expected_g, rtol=1e-12, atol=1e-12)
                    direction = rng.normal(size=x.size)
                    direction /= np.linalg.norm(direction)
                    h = 1e-5
                    numerical = (
                        adjoint.objective_grad_from_counts(x + h * direction, [0, intervals], counts)[0]
                        - adjoint.objective_grad_from_counts(x - h * direction, [0, intervals], counts)[0]
                    ) / (2 * h)
                    np.testing.assert_allclose(actual_g @ direction, numerical, rtol=1e-7, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
