"""Compare fixed/jump estimates of all ten six-state Ehrenfest rates."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_ehrenfest_rates(generators_file, figure_file=None, *, show=True):
    """Return a 5-by-2 rate grid: births left, deaths right."""
    with np.load(generators_file, allow_pickle=False) as data:
        times = data['times']
        true = data['true_generator']
        fixed_estimates = data['mean_fixed_estimated_generators']
        jump_estimates = data['mean_jump_estimated_generators']
        individuals = data['individuals']
    if (true.shape != (len(times), 6, 6)
            or fixed_estimates.shape != (1, len(times), 6, 6)
            or individuals.shape != (1,)):
        raise ValueError('Expected one saved sample size and six-state generators')
    if jump_estimates.shape != fixed_estimates.shape:
        raise ValueError('Jump-time and fixed-time estimates must have matching shapes')
    fig, axes = plt.subplots(5, 2, figsize=(10, 14))
    for level, (birth_ax, death_ax) in enumerate(axes):
        for ax, source, target in ((birth_ax, level, level + 1),
                                   (death_ax, level + 1, level)):
            ax.plot(times, fixed_estimates[0, :, source, target],
                    label='Fixed-time', alpha=.85)
            ax.plot(times, jump_estimates[0, :, source, target],
                    label='Jump-time', alpha=.85)
            ax.plot(times, true[:, source, target], 'k--', label='true')
            ax.grid(True, alpha=.3)
            ax.legend()
        birth_ax.set_title(rf'birth $\lambda_{{{level}}}$')
        death_ax.set_title(rf'death $\mu_{{{level + 1}}}$')
    for ax in axes[-1]:
        ax.set_xlabel('Time')
    fig.suptitle(f'Ehrenfest rates: {int(individuals[0])} independent boxes')
    fig.tight_layout()
    if figure_file is not None:
        Path(figure_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_file, dpi=180)
    if show:
        plt.show()
    return fig, axes
