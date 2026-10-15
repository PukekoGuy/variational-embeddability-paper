# Paper examples

Run from the project root.

## Adjoint vs gradient

```sh
python -m paper.adjoint_vs_gradient --target-jumps 1 --num-observations 20 --num-intervals 1000 --max-iter 500 --num-trials 5 --figure-file results/adjoint_vs_gradient.png --no-show
```

## Fixed-time vs jump-time

```sh
python -m paper.fixed_vs_jump --num-intervals 1000 --max-iter 500 --num-trials 5 --batch-size 10000 --figure-file results/fixed_vs_jump.png --no-show
```

## Four-panel comparison

```sh
python -m paper.comparison --individuals 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 --num-observations 20 10 --constant 10 --num-intervals 1000 --max-iter 500 --no-show
```

## Ehrenfest

```sh
python -c 'import numpy as np; from experiments.ehrenfest import generate_data; from plotting.ehrenfest_results import plot_ehrenfest_results; generate_data(num_particles=6, jump_frequency=6., volume_a_fn=lambda t: np.exp(-t), volume_b_fn=lambda t: 1., individuals=(100, 300, 1000, 3000, 10000), num_observations=20, num_intervals=190, num_trials=5, mu=1., kappa=1., max_iter=5000, seed=1008, ftol=1e-8, gtol=1e-5, rtol=1e-9, atol=1e-11, output_prefix="results/ehrenfest_paper"); plot_ehrenfest_results("results/ehrenfest_paper_results.csv", "results/ehrenfest_paper_mise.png", show=False)'
```

## Ehrenfest rates over time

This separate example fits one sample of 5,000 independent chains, with four
particles (five states). It plots four rows of birth rates on the left and death rates on the right, each
against the true rate over time, using chamber volumes `exp(-t)` and `1`.
The existing Ehrenfest MISE example remains available.

```sh
python run_example.py ehrenfest_rates --individuals 5000 --no-show
python run_example.py ehrenfest_rates --plot-only --no-show
```

Outputs use the separate `results/ehrenfest_rates` prefix: `_generators.npz`,
`_results.csv`, `_trials.csv`, and `_rates.png`. The single trial's convergence
status is recorded in `_trials.csv`. Replotting uses only the saved generators.

## Financial birth-death

```sh
python -m paper.financial_birth_death --kappa 1e4 --mu0 1e-4 --max-iter 2000 --decimal-places 6 --output-prefix results/financial_birth_death
```

## Adjoint vs gradient — dense sample sizes

```sh
python -c 'from experiments.comparisons import generate_adjoint_comparison; from plotting.comparison_results import plot_adjoint_vs_gradient_results; generate_adjoint_comparison(results_file="results/adjoint_vs_gradient_dense.csv", target_jumps=1, num_observations=20, num_intervals=100, num_individuals_list=range(500, 10001, 500), max_iter=500, num_trials=1); plot_adjoint_vs_gradient_results("results/adjoint_vs_gradient_dense.csv", "results/adjoint_vs_gradient_dense.png", show=False)'
```

## Fixed-time vs jump-time — dense sample sizes

```sh
python -c 'import numpy as np; from experiments import trig_generator, tune_generator_for_target_jumps; from experiments.comparisons import run_jump_comparison_experiments; from plotting.comparison_results import plot_fixed_vs_jump_results; generator=tune_generator_for_target_jumps(lambda A: trig_generator(A=A), 2, 8); run_jump_comparison_experiments(generator, 2, np.linspace(0., 1., 8), 1000, range(100, 10000, 100), results_file="results/fixed_vs_jump_dense.csv", mu=.01, kappa=1., max_iter=1000, num_trials=1); plot_fixed_vs_jump_results("results/fixed_vs_jump_dense.csv", "results/fixed_vs_jump_dense.png", show=False)'
```
