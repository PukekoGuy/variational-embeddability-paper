"""Constant birth-death embedding with an absorbing final state.

Rates are births followed by deaths; the last death rate is fixed to zero."""

from __future__ import annotations
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

def birth_death_generator(rates: np.ndarray, rank: int) -> np.ndarray:
    """Construct a rank x rank birth-death generator from 2(rank-1) rates.

    rates = [lambda_0, ..., lambda_{rank-2}, mu_1, ..., mu_{rank-1}].
    The final death-rate entry is ignored: the final state is absorbing.
    """
    rates = np.asarray(rates, dtype=float)
    expected = 2 * (rank - 1)
    if rates.shape != (expected,):
        raise ValueError(f"Expected {expected} rates, got shape {rates.shape}.")

    births = rates[: rank - 1]
    deaths = rates[rank - 1 :]

    A = np.zeros((rank, rank), dtype=float)

    # i -> i+1
    for i, lam in enumerate(births):
        A[i, i + 1] = lam
        A[i, i] -= lam

    # i -> i-1, where deaths[j] is mu_{j+1}
    for j, mu in enumerate(deaths[:-1]):
        i = j + 1
        A[i, i - 1] = mu
        A[i, i] -= mu

    return A

def generator_gradient_to_rates(generator_gradient: np.ndarray) -> np.ndarray:
    """Pull a Frobenius gradient in A back to birth/death rate variables."""
    G = np.asarray(generator_gradient, dtype=float)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("generator_gradient must be square.")

    rank = G.shape[0]
    grad = np.zeros(2 * (rank - 1), dtype=float)

    # dA/dlambda_i = E_{i,i+1} - E_{i,i}
    for i in range(rank - 1):
        grad[i] = G[i, i + 1] - G[i, i]

    # dA/dmu_i = E_{i,i-1} - E_{i,i}, i = 1,...,rank-2.
    # mu_{rank-1} is fixed to zero for the absorbing final state.
    for i in range(1, rank - 1):
        grad[(rank - 1) + (i - 1)] = G[i, i - 1] - G[i, i]

    return grad

class StrongBirthDeathEmbedder:
    """Fit one constant birth-death generator A to a target matrix Pi."""

    def __init__(
        self,
        mu0: float = 1.0e-4,
        max_iter: int = 10000,
        rate_upper_bound: float | None = None,
    ) -> None:
        if mu0 < 0:
            raise ValueError("mu0 must be nonnegative.")
        if rate_upper_bound is not None and rate_upper_bound <= 0:
            raise ValueError("rate_upper_bound must be positive when supplied.")

        self.mu0 = float(mu0)
        self.max_iter = int(max_iter)
        self.rate_upper_bound = rate_upper_bound

    def objective_grad(self, rates: np.ndarray, Pi: np.ndarray) -> tuple[float, np.ndarray]:
        """Return objective and exact gradient with respect to birth/death rates.

        For P = exp(A), the loss is

            U(A) = 1/2 ||P-Pi||_F^2 + mu0/2 ||A||_F^2.

        The adjoint of the Frechet derivative of exp at A is evaluated by
        ``expm_frechet(A.T, P-Pi)``.  This avoids a time discretisation of the
        forward or adjoint equations.
        """
        rank = Pi.shape[0]
        A = birth_death_generator(rates, rank)
        P = expm(A)
        residual = P - Pi

        objective = (
            0.5 * np.sum(residual * residual)
            + 0.5 * self.mu0 * np.sum(A * A)
        )

        # If L_exp(A,E) is the Frechet derivative of exp(A), then its
        # Frobenius adjoint applied to residual is L_exp(A^T, residual).
        grad_A_fit = expm_frechet(
            A.T, residual, compute_expm=False
        )
        grad_A = grad_A_fit + self.mu0 * A
        grad_rates = generator_gradient_to_rates(grad_A)

        return float(objective), grad_rates

    @staticmethod
    def initial_rates_from_target(Pi: np.ndarray) -> np.ndarray:
        """Use adjacent one-year transition probabilities as a simple start."""
        rank = Pi.shape[0]
        births = np.array([max(Pi[i, i + 1], 1e-6) for i in range(rank - 1)])
        deaths = np.array([max(Pi[i, i - 1], 1e-6) for i in range(1, rank)])
        deaths[-1] = 0.0
        return np.concatenate([births, deaths])

    def optimise(self, Pi: np.ndarray, x0: np.ndarray | None = None):
        """Optimise nonnegative birth/death rates for the target Pi.

        Returns
        -------
        A_hat : ndarray
            Estimated constant birth-death generator.
        P_hat : ndarray
            exp(A_hat), the fitted one-period transition matrix.
        rates_hat : ndarray
            Estimated rates in the order
            [lambda_0,...,lambda_{n-2},mu_1,...,mu_{n-1}].
            The final entry mu_{n-1} is fixed to zero.
        result : scipy.optimize.OptimizeResult
            Full L-BFGS-B result.
        """
        Pi = np.asarray(Pi, dtype=float)
        if Pi.ndim != 2 or Pi.shape[0] != Pi.shape[1]:
            raise ValueError("Pi must be square.")

        rank = Pi.shape[0]
        num_rates = 2 * (rank - 1)

        if x0 is None:
            x0 = self.initial_rates_from_target(Pi)
        else:
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (num_rates,):
                raise ValueError(f"x0 must have shape ({num_rates},).")
            if np.any(x0 < 0):
                raise ValueError("x0 must contain nonnegative rates.")

        upper = self.rate_upper_bound
        bounds = [(0.0, upper)] * num_rates
        bounds[-1] = (0.0, 0.0)

        result = minimize(
            fun=lambda z: self.objective_grad(z, Pi),
            x0=x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": 1e-14, "gtol": 1e-10},
        )

        rates_hat = result.x
        A_hat = birth_death_generator(rates_hat, rank)
        P_hat = expm(A_hat)
        return A_hat, P_hat, rates_hat, result
