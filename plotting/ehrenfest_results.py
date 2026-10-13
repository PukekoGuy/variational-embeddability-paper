"""Render saved Ehrenfest error metrics."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_ehrenfest_results(results_file,
                           figure_file=None, show=True,
                           figsize=(7, 5), dpi=180, grid_alpha=.3):
    """Plot saved MISE, without rerunning the experiment."""
    data = np.atleast_1d(np.genfromtxt(results_file, delimiter=',', names=True))
    fig, ax = plt.subplots(figsize=figsize)
    ax.loglog(data['num_individuals'], data['mise'], 'o-', label='Mean squared error / intervals')
    ax.set_xlabel('Number of independent observed chains')
    ax.set_ylabel('Mean squared entry error / number of intervals')
    ax.set_title('Ehrenfest generator estimation')
    ax.grid(True, which='both', alpha=grid_alpha)
    ax.legend()
    fig.tight_layout()
    if figure_file is not None:
        Path(figure_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_file, dpi=dpi)
    if show:
        plt.show()
    return fig
