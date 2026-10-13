"""Time-varying generator estimation from discrete jump observations."""

import numpy as np
from scipy.optimize import minimize
from .matrix_helpers import A_mat_from_rates, matrix_to_rates

class JumpTimeGradient:
    def __init__(self, rank, num_intervals=100, mu=1e-2, kappa=1e2, max_iter=500):
        self.num_intervals = num_intervals
        self.dt = 1.0 / num_intervals
        self.mu = mu
        self.kappa = kappa
        self.rank = rank
        self.num_rates = rank * (rank - 1)
        self.epsilon = 1e-12
        self.maxiter = max_iter

    def optimise_A(self, jump_times, states):
        """Estimate A(t) from compressed full-path data.

        jump_times are *interval indices*, exactly as returned by
        Observation.simulate_individual_jump_times_observations().
        Transition counts are computed once and reused by every optimizer step.
        """
        N = self.num_intervals + 1
        x0 = np.full((N, self.num_rates), 1.0).ravel()

        counts = self.transition_counts(jump_times, states)

        max_rate = 0.99 / (self.dt * (self.rank - 1))
        res = minimize(
            self.objective_grad_from_counts,
            x0,
            args=(counts,),
            jac=True,
            method="L-BFGS-B",
            options=dict(maxiter=self.maxiter),
            bounds=[(0.0, max_rate)] * (N * self.num_rates),
        )

        theta = res.x.reshape(N, self.num_rates)
        A = A_mat_from_rates(theta, self.rank)
        return A

    def transition_counts(self, jump_times, states):
        """Reconstruct C[t,i,j] = number of observed i->j transitions in interval t.

        A diagonal entry C[t,i,i] is a no-jump observation.  Since jump times and
        post-jump states are known, the compressed data contain this information.
        """
        counts = np.zeros((self.num_intervals, self.rank, self.rank), dtype=float)

        for individual_jump_times, individual_states in zip(jump_times, states):
            state = individual_states[0]
            jump_number = 0

            for t in range(self.num_intervals):
                if (
                    jump_number < len(individual_jump_times)
                    and individual_jump_times[jump_number] == t
                ):
                    new_state = individual_states[jump_number + 1]
                    counts[t, state, new_state] += 1.0
                    state = new_state
                    jump_number += 1
                else:
                    counts[t, state, state] += 1.0

            if jump_number != len(individual_jump_times):
                raise ValueError("jump_times must be sorted interval indices in [0, num_intervals-1]")

        return counts

    def objective_grad(self, x, jump_times, states):
        """Compatibility wrapper used for gradient checking/debugging."""
        counts = self.transition_counts(jump_times, states)
        return self.objective_grad_from_counts(x, counts)

    def objective_grad_from_counts(self, x, counts):
        theta = x.reshape(self.num_intervals + 1, self.num_rates)
        A = A_mat_from_rates(theta, self.rank)

        U = 0.0
        matrix_gradient = np.zeros_like(A)

        # Exact likelihood for the same Euler-discretised Markov chain used by
        # the simulator: P_t = I + dt A_t.
        P = np.eye(self.rank)[None, :, :] + self.dt * A[: self.num_intervals]
        safe_P = P + self.epsilon

        U -= self.kappa * np.sum(counts * np.log(safe_P))
        matrix_gradient[: self.num_intervals] -= (
            self.kappa * self.dt * counts / safe_P
        )

        differences = A[1:] - A[:-1]
        U += 0.5 * self.mu / self.dt * np.sum(differences**2)
        matrix_gradient[1:] += self.mu / self.dt * differences
        matrix_gradient[:-1] -= self.mu / self.dt * differences

        rate_gradient = np.stack([
            matrix_to_rates(grad, self.rank, self.num_rates)
            for grad in matrix_gradient
        ])

        return U, rate_gradient.ravel()
