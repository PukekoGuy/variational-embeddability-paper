"""Full-matrix Sobolev regularisation on a uniform time grid."""

import numpy as np


def regularisation_weights(mu0, mu1):
    """Validate the two independent Sobolev weights."""
    if not (np.isfinite(mu0) and mu0 >= 0 and np.isfinite(mu1) and mu1 >= 0):
        raise ValueError('mu0 and mu1 must be finite and nonnegative')
    return mu0, mu1


def sobolev_objective_grad(A, dt, mu0, mu1):
    """Return the penalty and its gradient in generator-matrix coordinates.

    For N intervals, use the left Riemann sum
    (mu0*dt/2) sum_{k=0}^{N-1} ||A_k||_F^2 and the finite-difference sum
    (mu1/(2*dt)) sum_{k=0}^{N-1} ||A_{k+1}-A_k||_F^2.
    Every matrix entry, including each diagonal, contributes.
    """
    differences = A[1:] - A[:-1]
    objective = (0.5 * mu0 * dt * np.sum(A[:-1]**2)
                 + 0.5 * mu1 / dt * np.sum(differences**2))
    gradient = np.zeros_like(A)
    gradient[:-1] += mu0 * dt * A[:-1]
    derivative_gradient = mu1 / dt * differences
    gradient[1:] += derivative_gradient
    gradient[:-1] -= derivative_gradient
    return objective, gradient
