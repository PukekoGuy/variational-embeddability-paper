"""Fit a generator from compressed jump observations on an Euler grid."""
import numpy as np
from experiments import Observation, trig_generator
from solvers import JumpTimeGradient


def main():
    observations = Observation(trig_generator(), 2, [0., 1.], num_intervals=20, seed=7)
    jump_times, states = observations.simulate_individual_jump_times_observations(100)
    # Each person's times are integer interval indices; states include the initial state.
    solver = JumpTimeGradient(rank=2, num_intervals=20, mu=0.01, kappa=1., max_iter=20)
    estimate = solver.optimise_A(jump_times, states)
    print(f'generator shape={estimate.shape}, finite={np.isfinite(estimate).all()}')
    return estimate


if __name__ == '__main__':
    main()
