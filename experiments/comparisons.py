"""Repeated fitting for two-state comparisons with shared trial aggregation."""
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from .observations import Observation
from .generators import trig_generator, tune_generator_for_target_jumps
from solvers import FixedTimeAdjoint, FixedTimeGradient, JumpTimeGradient

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
ADJOINT_RESULTS_FILE = RESULTS_DIR / 'adjoint_vs_gradient_results.csv'
JUMP_RESULTS_FILE = RESULTS_DIR / 'fixed_vs_jump_results.csv'
COMPARISON_RESULTS_FILE = RESULTS_DIR / 'comparison.npz'
ADJOINT_FIELDS = (
    'num_individuals', 'adjoint_mise', 'gradient_mise',
    'adjoint_execution_seconds', 'gradient_execution_seconds',
    'simulation_seconds', 'iteration_seconds', 'num_trials',
)
JUMP_FIELDS = (
    'num_individuals', 'fixed_mise', 'jump_mise',
    'fixed_execution_seconds', 'jump_execution_seconds',
    'fixed_simulation_seconds', 'jump_simulation_seconds', 'iteration_seconds', 'num_trials',
)


@dataclass
class ComparisonResult:
    """True path, mean estimated paths by method, and saved metric columns."""

    true_generator: np.ndarray
    estimated_paths: dict[str, list[np.ndarray]]
    metrics: dict[str, np.ndarray]


def _run_batched_comparison(
    generator, rank, fixed_times, num_intervals, sizes, results_file,
    mu0, mu1, max_iter, seed, num_trials, *, compare_jumps, batch_size=10000,
):
    """Simulate and fit two methods, sharing validation, timing and trial means."""
    if (isinstance(num_trials, (bool, np.bool_))
            or not isinstance(num_trials, (int, np.integer)) or num_trials < 1):
        raise ValueError('num_trials must be a positive integer')
    sizes = np.asarray(list(sizes), dtype=int)
    if sizes.ndim != 1 or sizes.size == 0 or np.any(sizes < 1):
        raise ValueError('individual counts must contain positive panel sizes')
    if compare_jumps and batch_size < 1:
        raise ValueError('batch_size must be positive')

    fields = JUMP_FIELDS if compare_jumps else ADJOINT_FIELDS
    names = ('fixed', 'jump') if compare_jumps else ('adjoint', 'gradient')
    start = perf_counter()
    observations = Observation(generator, rank, fixed_times, num_intervals, seed)
    solvers = (
        FixedTimeAdjoint(rank, num_intervals, max_iter=max_iter, mu0=mu0, mu1=mu1),
        (JumpTimeGradient if compare_jumps else FixedTimeGradient)(
            rank, num_intervals, max_iter=max_iter, mu0=mu0, mu1=mu1),
    )
    print(f'Shared transition setup: {perf_counter() - start:.3f}s', flush=True)
    true_A = observations.true_A
    paths = {name: [] for name in names}
    rows = []
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for n in sizes:
            trial_rows = []
            sums = [np.zeros_like(true_A, dtype=float) for _ in names]
            for trial in range(num_trials):
                print(f'n={n}: trial {trial + 1}/{num_trials}', flush=True)
                iteration_start = perf_counter()
                start = perf_counter()
                fixed_data = observations.simulate_fixed_observations(n)
                fixed_simulation = perf_counter() - start
                row = {}
                if compare_jumps:
                    start = perf_counter()
                    jump_times, jump_states = observations.simulate_individual_jump_times_observations(
                        n, batch_size=batch_size)
                    row['jump_simulation_seconds'] = perf_counter() - start
                    row['fixed_simulation_seconds'] = fixed_simulation
                    second_input = (jump_times, jump_states)
                else:
                    row['simulation_seconds'] = fixed_simulation
                    second_input = (fixed_times, fixed_data)
                for i, (name, solver, inputs) in enumerate(zip(
                    names, solvers, ((fixed_times, fixed_data), second_input),
                )):
                    start = perf_counter()
                    estimate = solver.optimise_A(*inputs)
                    row[f'{name}_execution_seconds'] = perf_counter() - start
                    sums[i] += estimate
                    row[f'{name}_mise'] = np.mean((true_A - estimate)**2)
                # Release all references before timing cleanup or allocating the next panel.
                del inputs, second_input, fixed_data
                if compare_jumps:
                    del jump_times, jump_states
                row['iteration_seconds'] = perf_counter() - iteration_start
                trial_rows.append(row)
            for name, total in zip(names, sums):
                paths[name].append(total / num_trials)
            row = {key: float(np.mean([trial[key] for trial in trial_rows]))
                   for key in fields[1:-1]}
            row.update(num_individuals=int(n), num_trials=int(num_trials))
            rows.append(row)
            writer.writerow(row)
            output.flush()
            print(f'n={n}: mean {names[0]}-MISE={row[fields[1]]:.6g}, '
                  f'{names[1]}-MISE={row[fields[2]]:.6g}, '
                  f'total/trial={row["iteration_seconds"]:.3f}s', flush=True)
    return ComparisonResult(
        true_generator=true_A, estimated_paths=paths,
        metrics={field: np.asarray([row[field] for row in rows]) for field in fields},
    )


def run_adjoint_experiments(
    generator, rank, fixed_times, num_intervals, num_individuals_range,
    results_file=ADJOINT_RESULTS_FILE, mu0=1e-2, mu1=1e-2, max_iter=500,
    seed=None, num_trials=5,
):
    """Fit adjoint/direct methods to the same panel; return paths and trial means.

    Returns a ComparisonResult with paths keyed by 'adjoint' and 'gradient'.
    Paths are mean estimates; errors
    and all CSV timings are per-trial means, computed before averaging paths.
    Fit timings include solver data preparation. Iteration timings include
    simulation, both fits, errors and cleanup, but exclude printing/CSV writing.
    """
    return _run_batched_comparison(
        generator, rank, fixed_times, num_intervals, num_individuals_range,
        results_file, mu0, mu1, max_iter, seed, num_trials, compare_jumps=False,
    )


def run_jump_comparison_experiments(
    generator, rank, fixed_times, num_intervals, num_individuals_list,
    results_file=JUMP_RESULTS_FILE, mu0=1e-4, mu1=1e-4, max_iter=500,
    seed=None, batch_size=10000, num_trials=5,
):
    """Fit fixed/jump methods to separate samples from the same generator.

    Returns a ComparisonResult with paths keyed by 'fixed' and 'jump'.
    CSV timings include separate fixed/jump simulation columns.
    """
    return _run_batched_comparison(
        generator, rank, fixed_times, num_intervals, num_individuals_list,
        results_file, mu0, mu1, max_iter, seed, num_trials,
        compare_jumps=True, batch_size=batch_size,
    )

def _run_panel_experiment(
    generator, sizes, observation_times, num_intervals,
    mu0, mu1, max_iter, seed, *, observe_jumps,
):
    """Fit one panel with the same simulator used by the pairwise comparisons."""
    observations = Observation(generator, 2, observation_times, num_intervals, seed)
    solver_type = JumpTimeGradient if observe_jumps else FixedTimeAdjoint
    solver = solver_type(2, num_intervals=num_intervals,
                         mu0=mu0, mu1=mu1, max_iter=max_iter)
    estimates, errors = [], []
    for n in sizes:
        if observe_jumps:
            inputs = observations.simulate_individual_jump_times_observations(int(n))
        else:
            inputs = (observation_times, observations.simulate_fixed_observations(int(n)))
        estimate = solver.optimise_A(*inputs)
        estimates.append(estimate)
        errors.append(np.mean((observations.true_A - estimate)**2))
    return estimates, observations.true_A, np.asarray(errors)


def generate_fixed_time_performance(
    results_file, *, mu0=1e-2, mu1=1e-2, target_jumps=1,
    num_observations=20, num_intervals=100, num_individuals_list=None,
    max_iter=500, num_trials=5, seed=1008,
):
    """Run fresh fixed-time performance trials; save mean errors and fit times."""
    if num_individuals_list is None:
        num_individuals_list = np.rint(np.geomspace(1, 1e6, 12)).astype(int)
    sizes = np.asarray(list(num_individuals_list), dtype=int)
    if sizes.size == 0 or np.any(sizes < 1) or num_trials < 1:
        raise ValueError('Sample sizes and num_trials must be positive')
    generator = tune_generator_for_target_jumps(lambda A: trig_generator(A=A), 2, target_jumps)
    times = np.linspace(0., 1., num_observations)
    observations = Observation(generator, 2, times, num_intervals, seed)
    solver = FixedTimeAdjoint(2, num_intervals, mu0=mu0, mu1=mu1, max_iter=max_iter)
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with results_file.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=(
            'num_individuals', 'fixed_mise', 'fixed_execution_seconds', 'num_trials'))
        writer.writeheader()
        for n in sizes:
            errors, seconds = [], []
            for trial in range(num_trials):
                panel = observations.simulate_fixed_observations(int(n))
                start = perf_counter()
                estimate = solver.optimise_A(times, panel)
                seconds.append(perf_counter() - start)
                errors.append(float(np.mean((estimate - observations.true_A)**2)))
                print(f'n={n}, trial={trial + 1}/{num_trials}: ISE={errors[-1]:.6g}', flush=True)
            row = dict(num_individuals=int(n), fixed_mise=float(np.mean(errors)),
                       fixed_execution_seconds=float(np.mean(seconds)), num_trials=num_trials)
            rows.append(row)
            writer.writerow(row)
            output.flush()
    return rows


def generate_adjoint_comparison(
    results_file=ADJOINT_RESULTS_FILE, target_jumps=1, num_observations=20,
    rank=2, num_intervals=1000, num_individuals_list=None,
    max_iter=500, seed=None, num_trials=5, mu0=1e-2, mu1=1e-2,
):
    """Tune a two-state model and compare adjoint/direct fits over panel sizes."""
    if rank != 2:
        raise ValueError("trig_generator is rank 2; use a run_*_experiments function for other generators")
    if num_individuals_list is None:
        num_individuals_list = np.rint(np.geomspace(1, 1e6, 12)).astype(int)
    start = perf_counter()
    generator = tune_generator_for_target_jumps(
        lambda A: trig_generator(A=A), rank, target_jumps)
    print(f"Generator tuning: {perf_counter() - start:.3f}s", flush=True)
    return run_adjoint_experiments(
        generator, rank, np.linspace(0.0, 1.0, num_observations),
        num_intervals, num_individuals_list, results_file=results_file,
        max_iter=max_iter, seed=seed, num_trials=num_trials, mu0=mu0, mu1=mu1,
    )


def generate_jump_comparison(
    results_file=JUMP_RESULTS_FILE, target_jumps=8, rank=2,
    num_intervals=1000, num_individuals_list=None, max_iter=500,
    seed=None, batch_size=10000, num_trials=5, mu0=1., mu1=1.,
):
    if rank != 2:
        raise ValueError("trig_generator is rank 2; use a run_*_experiments function for other generators")
    if num_individuals_list is None:
        num_individuals_list = np.rint(np.geomspace(1, 1e6, 6)).astype(int)
    start = perf_counter()
    generator = tune_generator_for_target_jumps(
        lambda A: trig_generator(A=A), rank, target_jumps)
    print(f"Generator tuning: {perf_counter() - start:.3f}s", flush=True)
    return run_jump_comparison_experiments(
        generator, rank, np.linspace(0, 1, target_jumps),
        num_intervals, num_individuals_list, results_file=results_file,
        max_iter=max_iter, seed=seed, batch_size=batch_size, mu0=mu0, mu1=mu1, num_trials=num_trials,
    )


def generate_comparison(results_file=COMPARISON_RESULTS_FILE, num_individuals_range=(1000,) * 10,
                  num_observations=(20, 10), constant=10, *, num_intervals=1000,
                  mu0=1e-2, mu1=1e-2, max_iter=500, seed=None):
    """Simulate and fit all four panels, saving arrays to an NPZ file.

    Fixed-time targets count / constant expected jumps; jump-time targets count.
    """
    sizes = np.asarray(list(num_individuals_range), dtype=int)
    counts = tuple(num_observations)
    if sizes.ndim != 1 or sizes.size == 0 or np.any(sizes < 1):
        raise ValueError('num_individuals_range must contain positive panel sizes')
    if len(counts) != 2 or any(not isinstance(n, (int, np.integer)) or n < 2 for n in counts):
        raise ValueError('num_observations must contain two integers of at least 2')
    if not np.isfinite(constant) or constant <= 0:
        raise ValueError('constant must be finite and positive')
    if not isinstance(num_intervals, (int, np.integer)) or num_intervals < 1:
        raise ValueError('num_intervals must be a positive integer')
    if max(counts) > num_intervals + 1:
        raise ValueError('num_intervals is too small for the observation counts')
    rng = np.random.default_rng(seed)
    fixed, jump = [], []
    for count in counts:
        times = np.linspace(0., 1., count)
        for results, target, observe_jumps in (
            (fixed, count / constant, False), (jump, count, True),
        ):
            generator = tune_generator_for_target_jumps(
                lambda A: trig_generator(A=A), 2, target, num_intervals=num_intervals)
            results.append(_run_panel_experiment(
                generator, sizes, times, num_intervals, mu0, mu1, max_iter,
                rng, observe_jumps=observe_jumps,
            ))
    data = dict(individuals=sizes, num_observations=np.asarray(counts), constant=np.asarray(constant))
    for name, results in (('fixed', fixed), ('jump', jump)):
        data[name + '_estimates'] = np.asarray([r[0] for r in results])
        data[name + '_true'] = np.asarray([r[1] for r in results])
        data[name + '_mise'] = np.asarray([r[2] for r in results])
    if results_file is not None:
        filename = Path(results_file)
        filename.parent.mkdir(parents=True, exist_ok=True)
        # Opening explicitly prevents numpy from silently appending a suffix.
        with filename.open('wb') as file:
            np.savez_compressed(file, **data)
    return data
