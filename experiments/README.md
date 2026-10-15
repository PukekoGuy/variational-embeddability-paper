# Experiments

[Paper experiment commands](../paper/README.md)

## Ehrenfest

Fits fixed-time and jump-time birth-death solvers to the same Euler-simulated
chains in each trial. Saved metrics and generator paths include both methods;
the MISE plot overlays their errors. Use a fitting grid fine enough for valid
Euler probabilities. The `transition_matrices` and `simulate_panel` utilities
remain available separately for continuous-time transition sampling.

```sh
python -c 'import numpy as np; from experiments.ehrenfest import generate_data; from plotting.ehrenfest_results import plot_ehrenfest_results; generate_data(num_particles=6, jump_frequency=6., volume_a_fn=lambda t: np.exp(-t), volume_b_fn=lambda t: 1., individuals=(100, 300, 1000, 3000, 10000), num_observations=20, num_intervals=190, num_trials=5, mu0=1., mu1=1., max_iter=5000, seed=1008, ftol=1e-8, gtol=1e-5, output_prefix="results/ehrenfest_paper"); plot_ehrenfest_results("results/ehrenfest_paper_results.csv", "results/ehrenfest_paper_mise.png", show=False)'
```
