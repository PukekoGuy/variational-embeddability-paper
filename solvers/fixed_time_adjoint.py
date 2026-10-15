"""Time-varying generator estimation from fixed-time panels, using adjoints."""

import numpy as np
from scipy.optimize import minimize
from .matrix_helpers import A_mat_from_rates, adjoint_to_rates, fixed_transition_counts
from .regularisation import regularisation_weights, sobolev_objective_grad

class FixedTimeAdjoint:
    def __init__(self, rank, num_intervals=100, max_iter=500,
                 *, mu0=1e-2, mu1=1e-2):
        self.num_intervals = num_intervals
        self.dt = 1.0 / num_intervals
        self.mu0, self.mu1 = regularisation_weights(mu0, mu1)
        self.rank = rank
        self.num_rates = rank * (rank - 1)
        self.epsilon = 1e-12
        self.maxiter = max_iter

    def exponentials(self, A):
        return [
            np.eye(self.rank) + self.dt*A[k] #expm(self.dt*A)
            for k in range(self.num_intervals)
        ]

    def transition_counts(self, obs_data):
        """Return counts[r, i, j] for each observation interval, including stays."""
        return fixed_transition_counts(obs_data, self.rank)

    def objective_grad(self, x, obs_times, obs_data):
        """Raw-data wrapper for gradient checks; obs_times are grid indices."""
        counts = self.transition_counts(obs_data)
        return self.objective_grad_from_counts(x, obs_times, counts)

    def objective_grad_from_counts(self, x, obs_times, transition_counts):
        """Evaluate using precomputed counts and observation grid indices."""
        theta = x.reshape(self.num_intervals + 1, self.num_rates)

        A = A_mat_from_rates(theta, self.rank)
        E = self.exponentials(A)

        U = 0.0
        rate_gradient = np.zeros_like(theta)

        for r in range(len(obs_times) - 1):

            a = obs_times[r]
            b = obs_times[r + 1]

            counts = transition_counts[r]

            # Local forward states:
            #
            # prefix[q] = E_a ... E_{a+q-1}
            #
            # so prefix[0] = I
            prefix = [np.eye(self.rank)]

            for k in range(a, b):
                prefix.append(prefix[-1] @ E[k])

            transition = prefix[-1]

            # Negative log likelihood
            U -= np.sum(
                counts * np.log(transition + self.epsilon)
            )

            # Terminal adjoint:
            #
            # dJ/dT = -counts / T
            Lam = -counts / (
                transition + self.epsilon
            )

            # Backward adjoint
            for k in range(b - 1, a - 1, -1):

                Fk = prefix[k - a]

                # dJ/dA_k = dt F_k^T Lambda_{k+1}
                M = self.dt * Fk.T @ Lam

                rate_gradient[k] += adjoint_to_rates(M, self.rank, self.num_rates)

                # Lambda_k = Lambda_{k+1} E_k^T
                Lam = Lam @ E[k].T

        penalty, penalty_gradient = sobolev_objective_grad(A, self.dt, self.mu0, self.mu1)
        U += penalty
        # Each free rate changes its off-diagonal entry and its row diagonal.
        off_diagonal = ~np.eye(self.rank, dtype=bool)
        diagonal = np.diagonal(penalty_gradient, axis1=1, axis2=2)
        rate_gradient += (penalty_gradient[:, off_diagonal]
                          - np.repeat(diagonal, self.rank - 1, axis=1))

        return U, rate_gradient.ravel()
    
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
        #P = self.forward(A)

        return A #, P, res.fun

    def times_to_intervals(self, times):
        return [int(round(t * self.num_intervals)) for t in times]
