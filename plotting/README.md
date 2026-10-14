# Plot saved paper results

Run after the [paper experiments](../paper/README.md).

```sh
python -m paper.adjoint_vs_gradient --plot-only --figure-file results/adjoint_vs_gradient.png --no-show
python -m paper.fixed_vs_jump --plot-only --figure-file results/fixed_vs_jump.png --no-show
python -m paper.comparison --plot-only --no-show
python -m paper.ehrenfest_rates --plot-only --no-show
python -c 'from plotting.ehrenfest_results import plot_ehrenfest_results; plot_ehrenfest_results("results/ehrenfest_paper_results.csv", "results/ehrenfest_paper_mise.png", show=False)'
```
