"""Render precomputed two-state paths and birth/death level rates."""

import numpy as np
import matplotlib.pyplot as plt


def plot_generator_path(
    gen_paths, true_A, num_individuals_range, times=None, plot_frequency=1,
    *, ax=None, show=False, title=None, fontsize=14, observation_markers=False
):
    """Plot a two-state rate-space curve and always return its Axes.

    Supply ``ax`` to compose subplots. Set ``show=True`` to display the figure.
    ``title=None`` uses the default title; an empty string omits it.
    """
    assert gen_paths[0][0].shape == (2, 2)

    if ax is None:
        _, ax = plt.subplots()

    num_individuals_to_plot = list(num_individuals_range)[::plot_frequency]
    gen_paths_to_plot = gen_paths[::plot_frequency]

    true_x = [true_A[t][0, 1] for t in range(len(true_A))]
    true_y = [true_A[t][1, 0] for t in range(len(true_A))]

    ax.plot(true_x, true_y, label="true", color="C1", linestyle="--")

    cmap = plt.colormaps["viridis"]
    denom = max(1, len(num_individuals_to_plot) - 1)
    for k, (num_individuals, gen_path) in enumerate(zip(num_individuals_to_plot, gen_paths_to_plot)):
        x = [gen_path[t][0, 1] for t in range(len(gen_path))]
        y = [gen_path[t][1, 0] for t in range(len(gen_path))]
        color = cmap(k / denom)
        ax.plot(x, y, label=f"{num_individuals}", color=color, alpha=0.5)

    if (times is not None) and observation_markers:
        for t in times:
            T = int(round(t * (len(gen_paths[0]) - 1)))
            ax.plot(true_x[T], true_y[T], marker="x", linestyle="None", color="black", markersize=8, mew=2)

    ax.set_xlabel(r"$A(t)_{12}$", fontsize=fontsize)
    ax.set_ylabel(r"$A(t)_{21}$", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.axis("scaled")
    ax.set_title(
        "Estimated vs true curve from matrix entries" if title is None else title,
        fontsize=fontsize + 2,
    )
    ax.grid(True)
    if len(num_individuals_to_plot) < 5:
        ax.legend(fontsize=fontsize)
    if show:
        plt.show()
    return ax


def plot_level_rates(generator_paths, true_A=None, decimal_places=None, *, show=True):
    """Plot every estimated birth and death rate as a function of time."""
    if decimal_places is not None and decimal_places < 0:
        raise ValueError("decimal_places must be nonnegative")
    paths = [np.asarray(path) for path in generator_paths]
    if not paths:
        raise ValueError("generator_paths must not be empty")

    rank = paths[0].shape[1]
    fig, axes = plt.subplots(2, rank - 1, squeeze=False, figsize=(4 * (rank - 1), 6))

    for level in range(rank - 1):
        for path in paths:
            axes[0, level].plot(path[:, level, level + 1], alpha=0.45)
            axes[1, level].plot(path[:, level + 1, level], alpha=0.45)
        if true_A is not None:
            axes[0, level].plot(true_A[:, level, level + 1], "k--", label="true")
            axes[1, level].plot(true_A[:, level + 1, level], "k--", label="true")
        axes[0, level].set_title(rf"birth $\lambda_{{{level}}}$")
        axes[1, level].set_title(rf"death $\mu_{{{level + 1}}}$")
        axes[1, level].set_xlabel("time-grid index")

    if decimal_places is not None:
        from matplotlib.ticker import FormatStrFormatter

        for ax in axes.flat:
            ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimal_places}f"))

    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes
