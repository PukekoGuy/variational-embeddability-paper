"""Ehrenfest model, continuous-time simulation and repeated fitting."""

import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from solvers.birth_death_helpers import rates_to_birth_death_gen
from solvers import BirthDeathEmbedder

DEFAULT_OUTPUT_PREFIX = Path(__file__).resolve().parents[1] / "results" / "ehrenfest"

def ehrenfest_generator(num_particles=6, jump_frequency=6.,
                         volume_a_fn=lambda t: 1., volume_b_fn=lambda t: 1.):
    """Return Q(t) for the two chambers, with nonabsorbing endpoints."""
    levels = np.arange(num_particles + 1)

    def generator(t):
        va, vb = volume_a_fn(t), volume_b_fn(t)
        if not (np.isfinite(va) and np.isfinite(vb) and va > 0 and vb > 0):
            raise ValueError('Chamber volumes must be finite and positive.')
        down = levels / va
        up = (num_particles - levels) / vb
        total = down + up
        rates = jump_frequency * np.concatenate((up[:-1] / total[:-1],
                                                 down[1:] / total[1:]))
        return rates_to_birth_death_gen(rates, num_particles + 1)

    return generator

def transition_matrices(generator, rank, observation_times, rtol=1e-9, atol=1e-11):
    """Solve P'=P Q(t) between successive observations."""
    transitions = []
    for start, stop in zip(observation_times[:-1], observation_times[1:]):
        solution = solve_ivp(
            lambda t, p: (p.reshape(rank, rank) @ generator(t)).ravel(),
            (start, stop), np.eye(rank).ravel(), rtol=rtol, atol=atol,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        transitions.append(solution.y[:, -1].reshape(rank, rank))
    return np.asarray(transitions)

def simulate_panel(transitions, num_individuals, rng, initial_distribution):
    """Draw each chain's state at each observation time."""
    rank = transitions.shape[1]
    paths = np.zeros((len(transitions) + 1, num_individuals), dtype=int)
    paths[0] = rng.choice(rank, size=num_individuals, p=initial_distribution)
    for individual in range(num_individuals):
        for k, transition in enumerate(transitions):
            state = paths[k, individual]
            paths[k + 1, individual] = rng.choice(rank, p=transition[state])
    return paths


def save_results(results_file, rows):
    """Save a table of experiment results."""
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(generator, rank, fixed_times, num_intervals,
                    num_individuals_range, output_prefix=DEFAULT_OUTPUT_PREFIX,
                    mu=1., kappa=1., max_iter=5000, num_trials=5, seed=1008,
                    initial_distribution=None, rtol=1e-9, atol=1e-11,
                    ftol=1e-8, gtol=1e-5):
    """Simulate, fit and average trial errors for each panel size."""
    fixed_times = np.asarray(fixed_times)
    num_individuals_range = list(num_individuals_range)
    grid = np.linspace(0., 1., num_intervals + 1)
    # Avoid silently rounding observation times in the Euler-based fitter.
    if not np.allclose(fixed_times * num_intervals, np.rint(fixed_times * num_intervals)):
        raise ValueError('Observation times must lie on the fitting grid.')
    true_A = np.stack([generator(t) for t in grid])
    transitions = transition_matrices(generator, rank, fixed_times, rtol, atol)
    if initial_distribution is None:
        initial_distribution = np.ones(rank) / rank
    rng = np.random.default_rng(seed)
    embedder = BirthDeathEmbedder(
        rank, num_intervals, mu, kappa, max_iter,
        absorbing_final=False, ftol=ftol, gtol=gtol,
    )
    rows, trials, estimated_paths = [], [], []
    for n in num_individuals_range:
        estimates, errors, fit_times, successes = [], [], [], []
        for trial in range(num_trials):
            paths = simulate_panel(transitions, n, rng, initial_distribution)
            start = perf_counter()
            estimate, result = embedder.optimise_A(fixed_times, paths, return_result=True)
            fit_seconds = perf_counter() - start
            error = np.mean((estimate[:-1] - true_A[:-1])**2) / num_intervals
            estimates.append(estimate)
            errors.append(error)
            fit_times.append(fit_seconds)
            successes.append(result.success)
            trials.append(dict(num_individuals=n, trial=trial + 1, ise=error,
                               success=result.success, iterations=result.nit,
                               fit_seconds=fit_seconds, message=str(result.message)))
            print(f'n={n}, trial={trial + 1}/{num_trials}: ISE={error:.5g}, '
                  f'converged={result.success}', flush=True)
        estimated_paths.append(np.mean(estimates, axis=0))
        rows.append(dict(num_individuals=n, mise=float(np.mean(errors)),
                         fit_seconds=float(np.mean(fit_times)), num_trials=num_trials,
                         converged_trials=int(sum(successes))))

    save_results(f'{output_prefix}_results.csv', rows)
    save_results(f'{output_prefix}_trials.csv', trials)
    np.savez_compressed(f'{output_prefix}_generators.npz', times=grid,
                        observation_times=fixed_times, true_generator=true_A,
                        mean_estimated_generators=estimated_paths,
                        individuals=list(num_individuals_range))
    return true_A, estimated_paths, rows


def generate_data(num_particles=6, jump_frequency=6.,
                  volume_a_fn=lambda t: 1., volume_b_fn=lambda t: 1.,
                  individuals=(100, 300, 1000, 3000, 10000), num_observations=20,
                  num_intervals=190, num_trials=5, seed=1008,
                  mu=1., kappa=1., max_iter=5000, initial_distribution=None,
                  rtol=1e-9, atol=1e-11, ftol=1e-8, gtol=1e-5,
                  output_prefix=DEFAULT_OUTPUT_PREFIX):
    """Build the Ehrenfest model and run the panel-size experiment.

    Edit volume_a_fn and volume_b_fn to supply any positive volume functions.
    For evenly spaced observations, use num_intervals divisible by
    num_observations - 1. The likelihood is summed and mu is fixed across sizes.
    """
    generator = ehrenfest_generator(num_particles, jump_frequency,
                                    volume_a_fn, volume_b_fn)
    times = np.linspace(0., 1., num_observations)
    _, _, rows = run_experiments(
        generator, num_particles + 1, times, num_intervals, individuals,
        output_prefix=output_prefix, mu=mu, kappa=kappa, max_iter=max_iter,
        num_trials=num_trials, seed=seed, initial_distribution=initial_distribution,
        rtol=rtol, atol=atol, ftol=ftol, gtol=gtol,
    )
    return rows
