"""Time-varying generator estimation from fixed-time panels, using direct gradients."""

import numpy as np
from scipy.optimize import minimize
from .matrix_helpers import A_mat_from_rates, matrix_to_rates, fixed_transition_counts
from .regularisation import regularisation_weights, sobolev_objective_grad

class FixedTimeGradient:
    def __init__(self, rank, num_intervals=100, max_iter=500,
                 *, mu0=1e-2, mu1=1e-2):
        self.num_intervals = num_intervals
        self.dt = 1.0 / num_intervals
        self.mu0, self.mu1 = regularisation_weights(mu0, mu1)
        self.rank = rank
        self.num_rates = rank * (rank - 1)
        self.epsilon = 1e-12
        self.maxiter = max_iter

    def optimise_A(self, obs_times, obs_data):
        N = self.num_intervals + 1
        x0 = np.full((N, self.num_rates), 1.0).ravel()
        obs_times = self.times_to_intervals(obs_times)
        counts = self.transition_counts(obs_data)

        max_rate = 0.99 / (self.dt * (self.rank - 1))
        res = minimize(
            self.objective_grad_from_counts,
            x0,
            args=(obs_times, counts),
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=self.maxiter, disp=False),
            bounds=[(0.0, max_rate)] * (N * self.num_rates),
        )

        theta = res.x.reshape(N, self.num_rates)
        A = A_mat_from_rates(theta, self.rank)

        return A 

    def transition_counts(self, obs_data):
        """Return counts[r, i, j] for each observation interval, including stays."""
        return fixed_transition_counts(obs_data, self.rank)

    def objective_grad(self, x, obs_times, obs_data):
        """Raw-data wrapper for gradient checks; obs_times are grid indices."""
        counts = self.transition_counts(obs_data)
        return self.objective_grad_from_counts(x, obs_times, counts)

    def objective_grad_from_counts(self, x, obs_times, counts):
        """Evaluate using precomputed counts and observation grid indices."""
        theta = x.reshape(self.num_intervals + 1, self.num_rates)
        A = A_mat_from_rates(theta, self.rank)

        U = 0.0
        matrix_gradient = np.zeros_like(A)

        for k in range(1, len(obs_times)):
            U += self.objective_grad_on_interval(
                A, obs_times, counts[k - 1], k, matrix_gradient
            )

        penalty, penalty_gradient = sobolev_objective_grad(A, self.dt, self.mu0, self.mu1)
        U += penalty
        matrix_gradient += penalty_gradient

        G = np.stack([
            matrix_to_rates(grad, self.rank, self.num_rates)
            for grad in matrix_gradient
        ])

        return U, G.ravel()

    def objective_grad_on_interval(self, A, obs_times, counts, k, G):
        t0 = obs_times[k - 1]
        t1 = obs_times[k]
        length = t1 - t0
        I = np.eye(self.rank)

        factors = [I + self.dt * A[tau] for tau in range(t0, t1)]

        forward = [I]
        for factor in factors:
            forward.append(forward[-1] @ factor)

        backward = [I for _ in range(length + 1)]
        for i in range(length - 1, -1, -1):
            backward[i] = factors[i] @ backward[i + 1]

        transition = forward[-1]
        U = -np.sum(counts * np.log(transition + self.epsilon))
        M = counts / (transition + self.epsilon)

        for i, tau in enumerate(range(t0, t1)):
            G[tau] -= (
                self.dt
                * forward[i].T
                @ M
                @ backward[i + 1].T
            )

        return U

    def times_to_intervals(self, times):
        return [int(round(t * self.num_intervals)) for t in times]
