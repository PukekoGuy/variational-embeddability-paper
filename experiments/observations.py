"""Seeded Euler observations and transition-matrix CSV input."""

from pathlib import Path

import numpy as np

class Observation:
    """Cache transitions once, then sample independent individuals in batches.

    Jump times are interval indices, and fixed observation
    times are rounded to the nearest grid point. The generator must be a
    deterministic function of time. Initial states default to uniform draws.
    """

    def __init__(self, generator, rank, times, num_intervals=1000, seed=None):
        if rank < 1 or num_intervals < 1:
            raise ValueError("rank and num_intervals must be positive")
        times = np.asarray(times, dtype=float)
        if (times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times))
                or np.any(np.diff(times) < 0) or times[0] < 0 or times[-1] > 1):
            raise ValueError("times must be sorted within [0, 1], with at least two entries")
        self.rank = rank
        self.num_intervals = num_intervals
        self.rng = np.random.default_rng(seed)
        self.true_A = np.stack([generator(k / num_intervals)
                                for k in range(num_intervals + 1)])
        if self.true_A.shape != (num_intervals + 1, rank, rank):
            raise ValueError("generator matrix shape must match rank")
        steps = np.eye(rank)[None, :, :] + self.true_A[:-1] / num_intervals
        indices = np.rint(times * num_intervals).astype(int)
        self.fixed_transitions = []
        for a, b in zip(indices[:-1], indices[1:]):
            product = np.eye(rank)
            for step in steps[a:b]:
                product = product @ step
            self.fixed_transitions.append(product)
        self.jump_cdf = self._cdf(steps)
        self.fixed_cdf = self._cdf(np.asarray(self.fixed_transitions))

    @staticmethod
    def _cdf(transitions):
        if (not np.all(np.isfinite(transitions)) or np.any(transitions < 0)
                or not np.allclose(transitions.sum(axis=-1), 1, atol=1e-10, rtol=0)):
            raise ValueError("Invalid Euler probabilities; increase num_intervals or check generator")
        # Normalisation removes accumulated roundoff in matrix products.
        probabilities = transitions / transitions.sum(axis=-1, keepdims=True)
        return np.cumsum(probabilities, axis=-1)[..., :-1]

    def _initial_states(self, n, initial_states):
        if n < 1:
            raise ValueError("num_individuals must be positive")
        if initial_states is None:
            return self.rng.integers(self.rank, size=n)
        states = np.asarray(initial_states)
        if (states.shape != (n,) or not np.issubdtype(states.dtype, np.integer)
                or np.any(states < 0) or np.any(states >= self.rank)):
            raise ValueError("initial_states must contain one valid integer state per individual")
        return states.copy()

    def _draw(self, cdf, states):
        # One uniform per individual; inverse categorical CDF, for any rank.
        return np.sum(self.rng.random(states.size)[:, None] >= cdf[states], axis=1)

    def simulate_fixed_observations(self, num_individuals, initial_states=None):
        states = self._initial_states(num_individuals, initial_states)
        paths = np.empty((len(self.fixed_cdf) + 1, num_individuals), dtype=int)
        paths[0] = states
        for k, cdf in enumerate(self.fixed_cdf, start=1):
            states = self._draw(cdf, states)
            paths[k] = states
        return paths

    def simulate_individual_jump_times_observations(
        self, num_individuals, initial_states=None, batch_size=10000,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        initial = self._initial_states(num_individuals, initial_states)
        jump_times = [[] for _ in range(num_individuals)]
        states = [[int(state)] for state in initial]
        for start in range(0, num_individuals, batch_size):
            current = initial[start:start + batch_size].copy()
            for t, cdf in enumerate(self.jump_cdf):
                new = self._draw(cdf, current)
                # Python work is needed only for actual changes, not every step.
                for local in np.flatnonzero(new != current):
                    individual = start + int(local)
                    jump_times[individual].append(t)
                    states[individual].append(int(new[local]))
                current = new
        return jump_times, states


def load_transition_matrix(csv_path: str | Path) -> np.ndarray:
    """Load and validate a numeric transition matrix Pi from CSV."""
    csv_path = Path(csv_path)
    Pi = np.loadtxt(csv_path, delimiter=",", dtype=float)

    if Pi.ndim != 2 or Pi.shape[0] != Pi.shape[1]:
        raise ValueError(f"Pi must be square; got shape {Pi.shape}.")
    if not np.all(np.isfinite(Pi)):
        raise ValueError("Pi contains NaN or infinite values.")
    if np.any(Pi < -1e-12):
        raise ValueError("Pi contains negative entries.")

    row_sums = Pi.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(
            "Rows of Pi must sum to 1. Row sums are:\n"
            + np.array2string(row_sums, precision=12)
        )

    # Remove harmless tiny negatives caused by floating-point I/O.
    Pi = np.maximum(Pi, 0.0)
    return Pi
