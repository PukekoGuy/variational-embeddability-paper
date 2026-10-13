"""Fit a time-varying birth-death generator from fixed observations."""
import numpy as np
from experiments import Observation
from experiments import birth_death_generator
from solvers import BirthDeathEmbedder


def main():
    generator = birth_death_generator(3, [0.8, 0.4], [0.3, 0.5])
    times = np.linspace(0., 1., 5)
    observations = Observation(generator, 3, times, num_intervals=20, seed=7)
    panel = observations.simulate_fixed_observations(100)
    # Both endpoints can leave their states; absorbing_final defaults to True.
    solver = BirthDeathEmbedder(3, num_intervals=20, absorbing_final=False, max_iter=20)
    estimate, result = solver.optimise_A(times, panel, return_result=True)
    print(f'generator shape={estimate.shape}, finite={np.isfinite(estimate).all()}, converged={result.success}')
    return estimate


if __name__ == '__main__':
    main()
