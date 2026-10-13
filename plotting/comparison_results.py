"""Plot CSV metrics from fixed/jump and adjoint/direct comparisons."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from .cosmetics import LABEL_FONT_SIZE, LEGEND_FONT_SIZE, TICK_LABEL_FONT_SIZE, TITLE_FONT_SIZE


def _plot_comparison_results(results_file, methods, figure_file, show, *, report_slopes=False):
    data = np.atleast_1d(np.genfromtxt(results_file, delimiter=",", names=True))
    n = data['num_individuals']
    fig, (mise_ax, time_ax) = plt.subplots(1, 2, figsize=(13, 5))
    for name, color in methods:
        mise = data[f'{name}_mise']
        mise_ax.loglog(n, mise, 'o-', color=color, label=name)
        time_ax.loglog(n, data[f'{name}_execution_seconds'], 'o-', color=color, label=name)
        if (report_slopes and len(np.unique(n)) > 1 and np.all(n > 0)
                and np.all(np.isfinite(mise) & (mise > 0))):
            slope = linregress(np.log(n), np.log(mise)).slope
            print(f'{name} MISE slope = {slope:.3f}')
    for ax in (mise_ax, time_ax):
        ax.set_xlabel('Number of individuals sampled', fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_LABEL_FONT_SIZE)
        ax.grid(True, which='both', alpha=.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE)
    mise_ax.set_ylabel('Mean integrated square error', fontsize=LABEL_FONT_SIZE)
    mise_ax.set_title('MISE vs. Number of Individuals', fontsize=TITLE_FONT_SIZE)
    time_ax.set_ylabel('Elapsed time (seconds)', fontsize=LABEL_FONT_SIZE)
    time_ax.set_title('Simulation vs . Number of Individuals', fontsize=TITLE_FONT_SIZE)
    fig.tight_layout()
    if figure_file is not None:
        figure_file = Path(figure_file)
        figure_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_file, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig, (mise_ax, time_ax)


def plot_fixed_vs_jump_results(results_file, figure_file=None, show=True):
    return _plot_comparison_results(
        results_file, (('fixed', 'tab:blue'), ('jump', 'tab:orange')), figure_file, show,
    )


def plot_adjoint_vs_gradient_results(results_file, figure_file=None, show=True, method='both'):
    """Plot saved errors and fitting times for adjoint, gradient, or both methods."""
    if method not in ('both', 'adjoint', 'gradient'):
        raise ValueError("method must be 'both', 'adjoint', or 'gradient'")
    methods = [(name, color) for name, color in (('adjoint', 'tab:blue'), ('gradient', 'tab:orange'))
               if method == 'both' or name == method]
    return _plot_comparison_results(results_file, methods, figure_file, show, report_slopes=True)


def plot_mise(x_range, MISE, *, show=True):
    """Plot positive errors against sample size; return figure and axes."""
    log_x = np.log(x_range)
    log_y = np.log(MISE)
    line = linregress(log_x,log_y)
    slope = line.slope

    fig, ax = plt.subplots()
    ax.loglog(x_range, MISE)
    ax.set_ylabel("Mean integrated square error", fontsize=LABEL_FONT_SIZE)
    ax.set_xlabel("Number of individuals sampled", fontsize=LABEL_FONT_SIZE)
    ax.set_title(f'Slope = {slope}', fontsize=TITLE_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(True)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
