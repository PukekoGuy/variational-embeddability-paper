"""Plot fixed-time generator estimation errors and fitting times."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .cosmetics import (LABEL_FONT_SIZE, LEGEND_FONT_SIZE,
                        TICK_LABEL_FONT_SIZE, TITLE_FONT_SIZE)


def plot_fixed_time_performance(results_file, figure_file=None,
                 *, show=True):
    """Plot freshly saved fixed-time errors and mean fitting times."""
    data = np.atleast_1d(np.genfromtxt(results_file, delimiter=',', names=True))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, field, title, ylabel, location in zip(
        axes, ('fixed_mise', 'fixed_execution_seconds'),
        ('MISE vs. Number of Individuals', 'Time vs Number of Individuals'),
        ('Mean integrated square error', 'Elapsed time (seconds)'),
        ('upper right', 'lower left'),
    ):
        ax.loglog(data['num_individuals'], data[field], 'o-', color='tab:blue', label='fixed-time')
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel('Number of individuals sampled', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_LABEL_FONT_SIZE)
        ax.grid(True, which='both', alpha=.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE, loc=location)
    fig.tight_layout()
    if figure_file is not None:
        figure_file = Path(figure_file)
        figure_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_file, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes

