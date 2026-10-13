"""Fit one constant generator to a transition matrix with an absorbing final state."""
import numpy as np
from scipy.linalg import expm
from solvers import StrongBirthDeathEmbedder
from solvers.strong_birth_death_adjoint import birth_death_generator


def main():
    # Births first, then deaths; the last death rate is fixed to zero.
    true_generator = birth_death_generator(np.array([0.8, 0.4, 0.3, 0.0]), rank=3)
    transition = expm(true_generator)  # one unit of elapsed time
    solver = StrongBirthDeathEmbedder(max_iter=20)
    estimate, fitted_transition, rates, result = solver.optimise(transition)
    print(f'generator shape={estimate.shape}, finite={np.isfinite(estimate).all()}, converged={result.success}')
    return estimate, fitted_transition


if __name__ == '__main__':
    main()
