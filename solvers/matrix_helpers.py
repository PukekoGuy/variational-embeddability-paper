import numpy as np

def fixed_transition_counts(obs_data, rank):
    """Count state pairs once for each consecutive pair of observation times."""
    counts = np.zeros((len(obs_data) - 1, rank, rank), dtype=float)
    for r in range(len(counts)):
        np.add.at(counts[r], (obs_data[r], obs_data[r + 1]), 1.0)
    return counts

def rates_to_gen(rates, rank):
    A = np.zeros((rank, rank))
    m = 0

    for i in range(rank):
        for j in range(rank):
            if i != j:
                A[i, j] = rates[m]
                A[i, i] -= rates[m]
                m += 1

    return A

def A_mat_from_rates(theta, rank):
    return np.stack([rates_to_gen(row, rank) for row in theta])

def basis_matrix(m, rank):
    n = rank
    B=np.zeros((n,n))
    row=m//(n-1)
    off=m%(n-1)
    col=off if off<row else off+1
    B[row,col]=1.
    B[row,row]=-1.
    return B

def adjoint_to_rates(M, rank, num_rates):
    # correct dual of rates_to_gen
    g = np.zeros(num_rates)
    n = rank
    k = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            B = np.zeros((n, n))
            B[i, j] = 1.0
            B[i, i] = -1.0
            g[k] = np.sum(M * B)
            k += 1
    return g

def matrix_to_rates(mat, rank, num_rates):
    rate = np.empty(num_rates)
    m = 0

    for i in range(rank):
        for j in range(rank):
            if i != j:
                rate[m] = mat[i, j] - mat[i, i]
                m += 1

    return rate
