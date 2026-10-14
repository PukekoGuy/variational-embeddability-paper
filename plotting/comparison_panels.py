"""Compose the four-panel comparison from precomputed generator estimates."""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from .cosmetics import PLOT_FONT_SIZE
from .generator_paths import plot_generator_path

TRUE_PATH_WIDTH_FACTOR = 2.0


def plot_comparison_panels(data, *, plot_frequency=1, fontsize=PLOT_FONT_SIZE,
                           figsize=(12, 12), show=True, different_colors=True):
    """Consume saved arrays; return fig, axes.

    Set ``different_colors=False`` to draw all estimated paths in blue.
    The dashed true path uses ``TRUE_PATH_WIDTH_FACTOR`` times the default width.
    """
    if not isinstance(plot_frequency, (int, np.integer)) or plot_frequency < 1:
        raise ValueError('plot_frequency must be a positive integer')
    fig, axes = plt.subplots(2, 2, figsize=figsize, layout='constrained')
    sizes = data['individuals']
    constant = float(data['constant'])
    for row, count in enumerate(data['num_observations']):
        for col, name in enumerate(('fixed', 'jump')):
            title = (f'Fixed-time adjoint: {count} observations\nExpected jumps = {count / constant:g}'
                     if name == 'fixed' else
                     f'Jump-time gradient: {count} expected observations\nExpected jumps = {count:g}')
            plot_generator_path(data[name + '_estimates'][row], data[name + '_true'][row],
                                   sizes, times=np.linspace(0., 1., count) if col == 0 else None,
                                   plot_frequency=plot_frequency, ax=axes[row, col],
                                   title=title, fontsize=fontsize,
                                   different_colors=different_colors,
                                   true_linewidth=TRUE_PATH_WIDTH_FACTOR * plt.rcParams['lines.linewidth'])
    if show:
        plt.show()
    return fig, axes


def plot_comparison_results(results_file, figure_file=None, show=True, **options):
    """Render saved arrays only; no simulation or optimisation occurs."""
    with np.load(results_file, allow_pickle=False) as data:
        fig, axes = plot_comparison_panels(data, show=False, **options)
    if figure_file is not None:
        Path(figure_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_file, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes
