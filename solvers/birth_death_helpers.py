"""Conversions between level-dependent birth/death rates and generators.

Rate vectors use the ordering

    [lambda_0, ..., lambda_{n-2}, mu_1, ..., mu_{n-1}],

where lambda_i is the rate i -> i+1 and mu_i is the rate i -> i-1.
"""

import numpy as np


def num_birth_death_rates(rank):
    if rank < 2:
        raise ValueError("rank must be at least 2")
    return 2 * (rank - 1)


def rates_to_birth_death_gen(rates, rank):
    """Return the rank-by-rank birth/death generator for one time point."""
    rates = np.asarray(rates, dtype=float)
    expected = num_birth_death_rates(rank)
    if rates.shape != (expected,):
        raise ValueError(f"expected {expected} rates, got shape {rates.shape}")

    births = rates[: rank - 1]
    deaths = rates[rank - 1 :]
    A = np.zeros((rank, rank), dtype=float)

    for level, rate in enumerate(births):
        A[level, level + 1] = rate
        A[level, level] -= rate

    for offset, rate in enumerate(deaths):
        level = offset + 1
        A[level, level - 1] = rate
        A[level, level] -= rate

    return A


def A_mat_from_birth_death_rates(theta, rank):
    """Convert a time-by-rate array into a time-by-rank-by-rank array."""
    theta = np.asarray(theta, dtype=float)
    expected = num_birth_death_rates(rank)
    if theta.ndim != 2 or theta.shape[1] != expected:
        raise ValueError(f"theta must have shape (num_times, {expected})")
    return np.stack([rates_to_birth_death_gen(row, rank) for row in theta])


def birth_death_basis_matrix(rate_index, rank):
    """Derivative of the generator with respect to one ordered rate."""
    total = num_birth_death_rates(rank)
    if not 0 <= rate_index < total:
        raise IndexError(f"rate_index must lie in [0, {total})")

    B = np.zeros((rank, rank), dtype=float)
    if rate_index < rank - 1:
        row = rate_index
        col = row + 1
    else:
        row = rate_index - (rank - 1) + 1
        col = row - 1

    B[row, col] = 1.0
    B[row, row] = -1.0
    return B


def adjoint_to_birth_death_rates(M, rank):
    """Apply the dual of ``rates_to_birth_death_gen`` to a matrix M."""
    M = np.asarray(M, dtype=float)
    if M.shape != (rank, rank):
        raise ValueError(f"M must have shape ({rank}, {rank})")

    gradient = np.empty(num_birth_death_rates(rank), dtype=float)
    for m in range(len(gradient)):
        B = birth_death_basis_matrix(m, rank)
        gradient[m] = np.sum(M * B)
    return gradient


def birth_death_gen_to_rates(generator):
    """Extract the ordered nearest-neighbour rates from a generator matrix."""
    generator = np.asarray(generator, dtype=float)
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise ValueError("generator must be square")

    rank = generator.shape[0]
    births = np.array([generator[i, i + 1] for i in range(rank - 1)])
    deaths = np.array([generator[i, i - 1] for i in range(1, rank)])
    return np.concatenate((births, deaths))
