"""Plot all ten rates from a saved, single-sample six-state Ehrenfest fit."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .generator_paths import plot_level_rates


def plot_ehrenfest_rates(generators_file, figure_file=None, *, show=True):
    """Return a 5-by-2 rate grid: births left, deaths right."""
    with np.load(generators_file, allow_pickle=False) as data:
        times = data['times']
        true = data['true_generator']
        estimates = data['mean_estimated_generators']
        individuals = data['individuals']
    if (true.shape != (len(times), 6, 6)
            or estimates.shape != (1, len(times), 6, 6)
            or individuals.shape != (1,)):
        raise ValueError('Expected one saved sample size and six-state generators')
    fig, axes = plot_level_rates(estimates, true, times=times, show=False)
    axes = axes.T
    fig.set_size_inches(10, 14)
    grid = fig.add_gridspec(5, 2)
    for (row, col), ax in np.ndenumerate(axes):
        ax.set_subplotspec(grid[row, col])
        ax.set_box_aspect(.5)
        ax.set_xlabel('Time' if row == 4 else '')
    for ax in axes.flat:
        ax.grid(True, alpha=.3)
        ax.legend()
    fig.suptitle(f'Ehrenfest rates: {int(individuals[0])} simulations')
    fig.tight_layout()
    if figure_file is not None:
        Path(figure_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_file, dpi=180)
    if show:
        plt.show()
    return fig, axes
