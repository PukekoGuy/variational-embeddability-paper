"""Time-varying birth-death generator estimation from fixed-time panels."""

import numpy as np
from scipy.optimize import minimize
from .birth_death_helpers import A_mat_from_birth_death_rates, adjoint_to_birth_death_rates, num_birth_death_rates

class BirthDeathEmbedder:
    """Estimate a time-varying, level-dependent birth/death generator path.

    At every grid point the free variables are ordered as
    [lambda_0, ..., lambda_{n-2}, mu_1, ..., mu_{n-1}].
    By default mu_{n-1} is fixed to zero (absorbing final state).
    Set absorbing_final=False to fit a nonabsorbing final state.
    """

    def __init__(self, rank, num_intervals=100, mu=1e-2, kappa=1.0, max_iter=500, absorbing_final=True, epsilon=1e-12,
                 euler_rate_cap=0.49, initial_rate=1.0, ftol=1e-9, gtol=1e-5):
        if rank < 2:
            raise ValueError("rank must be at least 2")
        self.num_intervals = num_intervals
        self.dt = 1.0 / num_intervals
        self.mu = mu
        self.kappa = kappa
        self.rank = rank
        self.num_rates = num_birth_death_rates(rank)
        self.absorbing_final = absorbing_final
        self.epsilon = epsilon
        self.euler_rate_cap = euler_rate_cap
        self.initial_rate = initial_rate
        self.ftol = ftol
        self.gtol = gtol
        if not 0 < euler_rate_cap < 0.5:
            raise ValueError("euler_rate_cap must be between 0 and 0.5")
        self.maxiter = max_iter

    def exponentials(self, A):
        """Euler transition factors for the first num_intervals grid points."""
        return [
            np.eye(self.rank) + self.dt * A[k]
            for k in range(self.num_intervals)
        ]

    def objective_grad(self, x, obs_intervals, obs_data):
        theta = np.asarray(x).reshape(self.num_intervals + 1, self.num_rates).copy()
        if self.absorbing_final:
            theta[:, -1] = 0.0
        A = A_mat_from_birth_death_rates(theta, self.rank)
        E = self.exponentials(A)
        levels = np.arange(self.rank - 1)

        objective = 0.0
        rate_gradient = np.zeros_like(theta)

        for r in range(len(obs_intervals) - 1):
            start = obs_intervals[r]
            stop = obs_intervals[r + 1]
            starts = obs_data[r]
            ends = obs_data[r + 1]

            counts = np.zeros((self.rank, self.rank))
            np.add.at(counts, (starts, ends), 1.0)

            prefix = [np.eye(self.rank)]
            for k in range(start, stop):
                prefix.append(prefix[-1] @ E[k])
            transition = prefix[-1]

            objective -= self.kappa * np.sum(
                counts * np.log(transition + self.epsilon)
            )
            adjoint = -self.kappa * counts / (transition + self.epsilon)

            for k in range(stop - 1, start - 1, -1):
                forward_to_k = prefix[k - start]
                generator_gradient = self.dt * forward_to_k.T @ adjoint
                # Each rate changes one off-diagonal entry and its row diagonal.
                rate_gradient[k, :self.rank - 1] += (
                    generator_gradient[levels, levels + 1] - generator_gradient[levels, levels])
                rate_gradient[k, self.rank - 1:] += (
                    generator_gradient[levels + 1, levels] - generator_gradient[levels + 1, levels + 1])
                adjoint = adjoint @ E[k].T

        differences = theta[1:] - theta[:-1]
        objective += 0.5 * self.mu / self.dt * np.sum(differences**2)
        rate_gradient[1:] += self.mu / self.dt * differences
        rate_gradient[:-1] -= self.mu / self.dt * differences
        if self.absorbing_final:
            rate_gradient[:, -1] = 0.0

        return objective, rate_gradient.ravel()

    def optimise_A(self, obs_times, obs_data, return_result=False):
        """Estimate A(t) from an array shaped (observation times, individuals)."""
        obs_data = np.asarray(obs_data, dtype=int)
        if obs_data.ndim != 2 or obs_data.shape[0] != len(obs_times):
            raise ValueError(
                "obs_data must have shape (len(obs_times), num_individuals)"
            )
        if np.any(obs_data < 0) or np.any(obs_data >= self.rank):
            raise ValueError("obs_data contains a state outside 0,...,rank-1")

        obs_intervals = self.times_to_intervals(obs_times)
        grid_points = self.num_intervals + 1
        # An interior birth/death row has two outgoing Euler probabilities.
        # This box bound guarantees their sum remains below one.
        max_rate = self.euler_rate_cap / self.dt
        initial_rate = min(self.initial_rate, max_rate)
        theta0 = np.full((grid_points, self.num_rates), initial_rate)
        if self.absorbing_final:
            theta0[:, -1] = 0.0
        x0 = theta0.ravel()
        rate_bounds = [(0.0, max_rate)] * self.num_rates
        if self.absorbing_final:
            rate_bounds[-1] = (0.0, 0.0)
        result = minimize(
            lambda z: self.objective_grad(z, obs_intervals, obs_data),
            x0,
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=self.maxiter, ftol=self.ftol, gtol=self.gtol),
            bounds=rate_bounds * grid_points,
        )

        theta = result.x.reshape(grid_points, self.num_rates)
        A = A_mat_from_birth_death_rates(theta, self.rank)
        if return_result:
            return A, result
        return A

    def times_to_intervals(self, times):
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError("obs_times must be a one-dimensional sequence")
        if np.any(times < 0.0) or np.any(times > 1.0) or np.any(np.diff(times) <= 0):
            raise ValueError("obs_times must be strictly increasing in [0, 1]")
        intervals = np.rint(times * self.num_intervals).astype(int)
        if np.any(np.diff(intervals) <= 0):
            raise ValueError("two observation times map to the same grid interval")
        return intervals.tolist()
