"""Trigonometric and level-dependent birth-death generator specifications."""

import numpy as np
from scipy.optimize import fsolve
from solvers.birth_death_helpers import rates_to_birth_death_gen

def trig_generator(f=1, A=1):
    return lambda t: np.array([
        [A*(-1 - np.cos(2 * np.pi * f * t)), A*(1 + np.cos(2 * np.pi * f * t))],
        [A*(1 + np.sin(2 * np.pi * f * t)), A*(-1 - np.sin(2 * np.pi * f * t))],
    ])


def tune_generator_for_target_jumps(
    generator_with_parameter, rank, target_num_jumps, *, num_intervals=100,
    initial_guess=1., initial_distribution=None, num_increments=100,
):
    """Return a generator whose expected jump count matches the target.

    ``generator_with_parameter(parameter)`` creates a generator callable. The
    expectation uses Euler transition products and a left Riemann sum over
    [0, 1]. Tuning is independent of observation sampling and random seeds.
    """
    if initial_distribution is None:
        initial_distribution = np.ones(rank) / rank
    initial_distribution = np.asarray(initial_distribution, dtype=float)

    def error(parameter):
        generator = generator_with_parameter(float(np.asarray(parameter).item()))
        expected = 0.
        for k in range(num_increments):
            t = k / num_increments
            transition = np.eye(rank)
            for interval in range(round(t * num_intervals)):
                transition = transition @ (np.eye(rank) + generator(interval / num_intervals) / num_intervals)
            expected -= initial_distribution @ transition @ np.diag(generator(t)) / num_increments
        return expected - target_num_jumps

    parameter, _, status, message = fsolve(error, initial_guess, full_output=True)
    if status != 1:
        raise RuntimeError(f'Generator tuning failed: {message}')
    return generator_with_parameter(float(parameter.item()))


def _evaluate_level_rates(specification, t, count, name):
    """Evaluate either a vector-valued callable or a sequence of rate specs."""
    if callable(specification):
        values = np.asarray(specification(t), dtype=float)
        if values.ndim == 0 and count == 1:
            values = values.reshape(1)
    else:
        values = np.asarray(
            [value(t) if callable(value) else value for value in specification],
            dtype=float,
        )

    if values.shape != (count,):
        raise ValueError(f"{name} must provide exactly {count} level-specific rates")
    if np.any(values < 0):
        raise ValueError(f"{name} must be non-negative")
    return values


def birth_death_generator(rank, birth_rates, death_rates):
    """Build a time-dependent generator with a distinct rate at every level.

    ``birth_rates(t)`` must return [lambda_0(t), ..., lambda_{n-2}(t)] and
    ``death_rates(t)`` must return [mu_1(t), ..., mu_{n-1}(t)]. Each argument
    may instead be a sequence of scalar values and/or scalar-valued callables.
    """
    count = rank - 1

    def generator(t):
        births = _evaluate_level_rates(birth_rates, t, count, "birth_rates")
        deaths = _evaluate_level_rates(death_rates, t, count, "death_rates")
        return rates_to_birth_death_gen(np.concatenate((births, deaths)), rank)

    return generator
