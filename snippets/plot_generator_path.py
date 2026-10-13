"""Draw precomputed two-state generator paths; no solver is needed."""
import numpy as np
from experiments import trig_generator
from plotting import plot_generator_path


def main():
    times = np.linspace(0., 1., 21)
    true_path = np.stack([trig_generator()(t) for t in times])
    example_estimate = 0.9 * true_path  # illustrative estimate, not a fitted model
    ax = plot_generator_path([example_estimate], true_path, [100], show=False)
    # ax.figure.savefig('generator_path.png')
    print(f'plot axes=1, finite={np.isfinite(example_estimate).all()}')
    return ax.figure


if __name__ == '__main__':
    main()
