"""Birth/death generator paths fitted to grid-indexed jump histories."""

import numpy as np
from scipy.optimize import minimize

from .weak_birth_death_adjoint import BirthDeathEmbedder
from .birth_death_helpers import A_mat_from_birth_death_rates
from .regularisation import sobolev_objective_grad


class BirthDeathJumpTimeGradient(BirthDeathEmbedder):
    """Use jump/no-jump likelihood with only 2*(rank-1) rates per grid point.

    Configuration matches BirthDeathEmbedder, including mu0, mu1,
    and the optional absorbing final state. Jump times identify Euler intervals:
    a jump at k changes the state from grid point k to grid point k+1.
    """

    def transition_counts(self, jump_times, states):
        """Count jumps and stays without expanding every individual's full path."""
        if len(jump_times) != len(states):
            raise ValueError('Provide one state history per jump-time history')
        counts = np.zeros((self.num_intervals, self.rank, self.rank))
        occupancy_changes = np.zeros((self.num_intervals + 1, self.rank))
        for times, history in zip(jump_times, states):
            times, history = np.asarray(times), np.asarray(history)
            if (times.ndim != 1 or history.ndim != 1
                    or history.size != times.size + 1
                    or not np.issubdtype(history.dtype, np.integer)
                    or np.any(history < 0) or np.any(history >= self.rank)):
                raise ValueError('Histories must contain an initial state and one valid state per jump')
            if times.size and (not np.issubdtype(times.dtype, np.integer)
                               or np.any(times < 0) or np.any(times >= self.num_intervals)
                               or np.any(np.diff(times) <= 0)):
                raise ValueError('Jump times must be strictly increasing integer interval indices')
            if np.any(np.abs(np.diff(history)) != 1):
                raise ValueError('Birth/death jumps must connect neighbouring states')
            if self.absorbing_final and np.any(history[:-1] == self.rank - 1):
                raise ValueError('The absorbing final state cannot have outgoing jumps')
            occupancy_changes[0, history[0]] += 1
            for t, before, after in zip(times, history[:-1], history[1:]):
                counts[t, before, after] += 1
                occupancy_changes[t + 1, before] -= 1
                occupancy_changes[t + 1, after] += 1
        occupancy = np.cumsum(occupancy_changes, axis=0)[:-1]
        levels = np.arange(self.rank)
        counts[:, levels, levels] = occupancy - counts.sum(axis=2)
        return counts

    def objective_grad(self, x, jump_times, states):
        """Raw-history wrapper; fitting prepares transition counts only once."""
        return self.objective_grad_from_counts(x, self.transition_counts(jump_times, states))

    def objective_grad_from_counts(self, x, counts):
        theta = np.asarray(x).reshape(self.num_intervals + 1, self.num_rates).copy()
        if self.absorbing_final:
            theta[:, -1] = 0.
        A = A_mat_from_birth_death_rates(theta, self.rank)
        probabilities = np.eye(self.rank) + self.dt * A[:-1]
        safe_probabilities = probabilities + self.epsilon
        objective = -np.sum(counts * np.log(safe_probabilities))
        penalty, matrix_gradient = sobolev_objective_grad(A, self.dt, self.mu0, self.mu1)
        objective += penalty
        matrix_gradient[:-1] -= self.dt * counts / safe_probabilities
        levels = np.arange(self.rank - 1)
        rate_gradient = np.concatenate((
            matrix_gradient[:, levels, levels + 1] - matrix_gradient[:, levels, levels],
            matrix_gradient[:, levels + 1, levels] - matrix_gradient[:, levels + 1, levels + 1],
        ), axis=1)
        if self.absorbing_final:
            rate_gradient[:, -1] = 0.
        return objective, rate_gradient.ravel()

    def optimise_A(self, jump_times, states, return_result=False):
        """Fit a birth/death path, optionally returning the optimizer status."""
        counts = self.transition_counts(jump_times, states)
        grid_points = self.num_intervals + 1
        max_rate = self.euler_rate_cap / self.dt
        theta = np.full((grid_points, self.num_rates), min(self.initial_rate, max_rate))
        bounds = [(0., max_rate)] * self.num_rates
        if self.absorbing_final:
            theta[:, -1] = 0.
            bounds[-1] = (0., 0.)
        result = minimize(
            self.objective_grad_from_counts, theta.ravel(), args=(counts,), jac=True,
            method='L-BFGS-B', bounds=bounds * grid_points,
            options=dict(maxiter=self.maxiter, ftol=self.ftol, gtol=self.gtol),
        )
        A = A_mat_from_birth_death_rates(result.x.reshape(theta.shape), self.rank)
        return (A, result) if return_result else A
