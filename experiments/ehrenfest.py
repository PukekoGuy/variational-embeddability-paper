"""Ehrenfest model, continuous-time simulation and repeated fitting."""

import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp
from solvers.birth_death_helpers import rates_to_birth_death_gen
from solvers import BirthDeathEmbedder, BirthDeathJumpTimeGradient
from .observations import Observation

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


def simulate_paired_observations(sampler, fixed_times, num_individuals, initial_distribution):
    """Return periodic snapshots and jump histories from the same Euler chains."""
    initial = sampler.rng.choice(sampler.rank, size=num_individuals, p=initial_distribution)
    jump_times, states = sampler.simulate_individual_jump_times_observations(
        num_individuals, initial_states=initial)
    indices = np.rint(np.asarray(fixed_times) * sampler.num_intervals).astype(int)
    panel = np.empty((len(indices), num_individuals), dtype=int)
    for individual, (events, history) in enumerate(zip(jump_times, states)):
        # A jump in interval k is visible at grid point k+1, not point k.
        panel[:, individual] = np.asarray(history)[np.searchsorted(events, indices, side='left')]
    return panel, jump_times, states


def run_experiments(generator, rank, fixed_times, num_intervals,
                    num_individuals_range, output_prefix=DEFAULT_OUTPUT_PREFIX,
                    mu0=1., mu1=1., max_iter=5000, num_trials=5, seed=1008,
                    initial_distribution=None, ftol=1e-8, gtol=1e-5):
    """Compare fixed/jump fits on paired Euler trajectories, averaging trial ISEs.

    Return the true path, a dictionary of mean paths by method, and metric rows.
    """
    fixed_times = np.asarray(fixed_times)
    num_individuals_range = list(num_individuals_range)
    if num_trials < 1 or not isinstance(num_trials, (int, np.integer)):
        raise ValueError('num_trials must be a positive integer')
    if not num_individuals_range or any(n < 1 for n in num_individuals_range):
        raise ValueError('Sample sizes must be positive')
    grid = np.linspace(0., 1., num_intervals + 1)
    if not np.allclose(fixed_times * num_intervals, np.rint(fixed_times * num_intervals)):
        raise ValueError('Observation times must lie on the fitting grid.')
    sampler = Observation(generator, rank, fixed_times, num_intervals, seed)
    true_A = sampler.true_A
    if initial_distribution is None:
        initial_distribution = np.ones(rank) / rank
    initial_distribution = np.asarray(initial_distribution, dtype=float)
    if (initial_distribution.shape != (rank,) or np.any(initial_distribution < 0)
            or not np.all(np.isfinite(initial_distribution))
            or not np.isclose(initial_distribution.sum(), 1.)):
        raise ValueError('initial_distribution must be a probability vector of length rank')
    options = dict(num_intervals=num_intervals, max_iter=max_iter, mu0=mu0, mu1=mu1,
                   absorbing_final=False, ftol=ftol, gtol=gtol)
    solvers = dict(fixed=BirthDeathEmbedder(rank, **options),
                   jump=BirthDeathJumpTimeGradient(rank, **options))
    solvers['fixed'].times_to_intervals(fixed_times)
    rows, trials = [], []
    estimated_paths = {name: [] for name in solvers}
    for n in num_individuals_range:
        estimates = {name: [] for name in solvers}
        trial_rows = []
        for trial in range(num_trials):
            panel, jump_times, states = simulate_paired_observations(
                sampler, fixed_times, n, initial_distribution)
            record = dict(num_individuals=n, trial=trial + 1)
            for name, solver in solvers.items():
                inputs = (fixed_times, panel) if name == 'fixed' else (jump_times, states)
                start = perf_counter()
                estimate, result = solver.optimise_A(*inputs, return_result=True)
                elapsed = perf_counter() - start
                error = float(np.mean((estimate - true_A)**2))
                estimates[name].append(estimate)
                record.update({f'{name}_ise': error, f'{name}_success': bool(result.success),
                               f'{name}_iterations': int(result.nit),
                               f'{name}_fit_seconds': elapsed, f'{name}_message': str(result.message)})
                print(f'n={n}, trial={trial + 1}/{num_trials}, {name}: '
                      f'ISE={error:.5g}, converged={result.success}', flush=True)
            trials.append(record)
            trial_rows.append(record)
        row = dict(num_individuals=n, num_trials=num_trials)
        for name in solvers:
            estimated_paths[name].append(np.mean(estimates[name], axis=0))
            row.update({f'{name}_mise': float(np.mean([r[f'{name}_ise'] for r in trial_rows])),
                        f'{name}_fit_seconds': float(np.mean([r[f'{name}_fit_seconds'] for r in trial_rows])),
                        f'{name}_converged_trials': sum(r[f'{name}_success'] for r in trial_rows)})
        rows.append(row)

    save_results(f'{output_prefix}_results.csv', rows)
    save_results(f'{output_prefix}_trials.csv', trials)
    np.savez_compressed(f'{output_prefix}_generators.npz', times=grid,
                        observation_times=fixed_times, true_generator=true_A,
                        mean_fixed_estimated_generators=estimated_paths['fixed'],
                        mean_jump_estimated_generators=estimated_paths['jump'],
                        individuals=num_individuals_range, mu0=mu0, mu1=mu1,
                        simulation_method='paired_euler')
    return true_A, estimated_paths, rows


def generate_data(num_particles=6, jump_frequency=6.,
                  volume_a_fn=lambda t: 1., volume_b_fn=lambda t: 1.,
                  individuals=(100, 300, 1000, 3000, 10000), num_observations=20,
                  num_intervals=190, num_trials=5, seed=1008,
                  mu0=1., mu1=1., max_iter=5000, initial_distribution=None,
                  ftol=1e-8, gtol=1e-5,
                  output_prefix=DEFAULT_OUTPUT_PREFIX):
    """Build the Ehrenfest model and compare fixed/jump fits over sample sizes.

    Edit volume_a_fn and volume_b_fn to supply any positive volume functions.
    For evenly spaced observations, use num_intervals divisible by
    num_observations - 1. The likelihood is summed and both regularisation weights are fixed across sizes.
    """
    generator = ehrenfest_generator(num_particles, jump_frequency,
                                    volume_a_fn, volume_b_fn)
    times = np.linspace(0., 1., num_observations)
    _, _, rows = run_experiments(
        generator, num_particles + 1, times, num_intervals, individuals,
        output_prefix=output_prefix, mu0=mu0, mu1=mu1, max_iter=max_iter,
        num_trials=num_trials, seed=seed, initial_distribution=initial_distribution,
        ftol=ftol, gtol=gtol,
    )
    return rows
