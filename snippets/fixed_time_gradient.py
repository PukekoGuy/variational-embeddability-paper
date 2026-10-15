"""Fit a two-state generator from a panel of fixed-time observations."""
import numpy as np
from experiments import Observation, trig_generator
from solvers import FixedTimeGradient


def main():
    times = np.linspace(0., 1., 5)
    observations = Observation(trig_generator(), 2, times, num_intervals=20, seed=7)
    panel = observations.simulate_fixed_observations(100)  # shape: (5 times, 100 people)
    solver = FixedTimeGradient(rank=2, num_intervals=20, mu0=.01, mu1=.01, max_iter=20)
    estimate = solver.optimise_A(times, panel)  # shape: (21 grid points, 2, 2)
    print(f'generator shape={estimate.shape}, finite={np.isfinite(estimate).all()}')
    return estimate


if __name__ == '__main__':
    main()
